# -*- coding: utf-8 -*-
"""
Source function assembly (plan section 5).

Local source function, LTE, monochromatic:

    S(Omega_i) = (1 - omega) * B + omega * int Phi(Omega' -> Omega_i) I dOmega'
                 + [beam first-scattering term, fixed, never iterated]

The scattering integral over *all* cells is one BLAS matmul:
I (n_cells x N_dir) @ Phi_w.T, where Phi_w[i, j] = Phi_HG(Omega_j -> Omega_i) * w_j
comes from direction_grid.hg_matrix(g).

If g varies spatially, bucket cells by g and do one matmul per bucket
(plan section 2); phase 2 uses a scalar g.

Delta-M (Wiscombe) scaling (delta_M >= 1, plan/delta-m-plan.md) splits the
normalized HG matrix exactly via direction_grid.hg_delta_matrices and
rescales the source by 1/(1 - omega * g**M) with an explicit
-omega * g**M * I term; the caller transports with the scattering-only
reduced extinction delta_extinction(kappa_e, kappa_s, g, M)
= kappa_e - kappa_s * g**M.
"""

import numpy as np

from rt3d.directions import hg


def single_scattering_albedo(kappa_e, kappa_s):
    """
    omega = kappa_s / kappa_e with the vacuum guard kappa_e -> 0 (omega = 0).
    """
    kappa_e = np.asarray(kappa_e, dtype=float)
    kappa_s = np.asarray(kappa_s, dtype=float)
    omega = np.zeros_like(kappa_e)
    np.divide(kappa_s, kappa_e, out=omega, where=kappa_e > 0.0)
    return omega


def mean_intensity(I, dg):
    """
    Angle-averaged mean intensity J = (1/4pi) int I dOmega per cell.

    Parameters
    ----------
    I : (n_cells, N_dir) ndarray
    dg : direction_grid

    Returns
    -------
    J : (n_cells,) ndarray
    """
    return (I @ dg.weights) / (4.0 * np.pi)


def flux_vector(I, dg):
    """
    Radiation flux vector F = int I Omega dOmega per cell.

    Returns
    -------
    F : (n_cells, 3) ndarray
    """
    return (I * dg.weights[None, :]) @ dg.directions


def delta_extinction(kappa_e, kappa_s, g, M):
    """
    Delta-M transport extinction kappa_e_delta = kappa_e - kappa_s * g**M
    = kappa_e * (1 - omega * g**M) (plan/delta-m-plan.md section 1).

    Only the *scattering* part of the extinction is delta-scaled: the
    phase function applies to scattering alone, so keeping absorption
    unscaled (a) leaves pure absorption (kappa_s = 0) exactly
    Beer-Lambert (validation anchor 3 holds with delta on) and (b) keeps
    the explicit -omega * g**M * I source term's iteration gain
    omega * g**M (1 - g) / (1 - omega * g**M) < 1 for all omega < 1,
    whereas scaling the full extinction by (1 - g**M) diverges for
    omega < g**M.

    Parameters
    ----------
    kappa_e : float or ndarray
        Extinction coefficient (physical, not scaled).
    kappa_s : float or ndarray
        Scattering coefficient (physical).
    g : float or (n_cells,) ndarray
        HG asymmetry parameter, scalar or per-cell; must satisfy
        -1 < g < 1.
    M : int
        Delta-M order. M == 0 leaves kappa_e unchanged (delta off).

    Returns
    -------
    kappa_e_delta : float or ndarray
        kappa_e - kappa_s * g**M; vacuum cells (kappa_e == 0) stay 0.
    """
    if M == 0:
        return np.asarray(kappa_e, dtype=float)
    if M < 1 or int(M) != M:
        raise ValueError(f"M must be a non-negative integer, got {M}")
    g = np.asarray(g, dtype=float)
    if np.any(g <= -1.0) or np.any(g >= 1.0):
        raise ValueError("g must satisfy -1 < g < 1")
    return np.asarray(kappa_e, dtype=float) - np.asarray(kappa_s, dtype=float) * g**M


def _bucket_structure(cache, g_cells, g_decimals):
    """
    Bucket cells by g rounded to g_decimals into bucket-contiguous blocks
    (plan/optimization-plan.md P6).  Cached in Phi_cache: the per-iteration
    np.unique + boolean-mask scans and the per-bucket fancy-index gathers
    dominated the source cost; after the first call only the slice matmuls
    remain.  Stable sort keeps the original cell order inside a bucket, so
    results are bit-comparable to the naive bucket loop.

    Returns (gv_unique, blocks, order, inverse): blocks are (start, stop)
    into the sorted arrays; sorted = cells[order].
    """
    key = ("_buckets", int(g_decimals))
    got = cache.get(key)
    if got is not None:
        return got
    gv = np.round(np.asarray(g_cells, dtype=float), g_decimals)
    order = np.argsort(gv, kind="stable")
    gvs = gv[order]
    change = np.flatnonzero(gvs[1:] != gvs[:-1]) + 1
    bounds = np.concatenate(([0], change, [gvs.size]))
    gv_u = [float(gvs[bounds[i]]) for i in range(bounds.size - 1)]
    blocks = [(int(bounds[i]), int(bounds[i + 1]))
              for i in range(bounds.size - 1)]
    inv = np.empty(order.size, dtype=np.intp)
    inv[order] = np.arange(order.size)
    got = (gv_u, blocks, order, inv)
    cache[key] = got
    return got


def scattering_source(I, dg, omega, B, Phi_w=None, g=0.0, beam_scat=None,
                      g_cells=None, g_decimals=2, Phi_cache=None, delta_M=0):
    """
    Direction-dependent source function

        S[cell, i] = (1 - omega) B + omega * sum_j w_j Phi(Omega_j -> Omega_i) I_j
                     + beam_scat[cell, i]

    With delta_M >= 1 the delta-M (Wiscombe) form is used instead
    (plan/delta-m-plan.md):

        S = [ (1-omega) B
              + omega * ( I @ Phi_w_delta.T + sum_n I @ moments[n].T )
              - omega * g**M * I ] / (1 - omega * g**M)

    (per-cell factors when omega varies, which it always does).  Here
    (Phi_w_delta, moments) = dg.hg_delta_matrices(g, delta_M) is the
    *exact* split of the row-normalized weighted HG matrix,
    Phi_w == Phi_w_delta + sum(moments).  The scattering integral is
    therefore unchanged; only the explicit -omega * g**M * I term and the
    1/(1 - omega * g**M) scaling are added, and the caller must transport
    with the reduced extinction kappa_e - kappa_s * g**M
    (delta_extinction).  The converged solution is identical to
    delta_M = 0 up to the short-characteristics discretization of the
    explicit term; only the *scattering* part of the extinction is
    scaled, so pure absorption (omega = 0) is left exactly unchanged.

    Parameters
    ----------
    I : (n_cells, N_dir) ndarray
        Intensities at cell centers from the previous sweep.
    dg : direction_grid
    omega : (n_cells,) ndarray
        Single scattering albedo.
    B : (n_cells,) ndarray
        Planck function (thermal emission), may be zeros.
    Phi_w : (N_dir, N_dir) ndarray, optional
        Weighted HG matrix; built from dg.hg_matrix(g) if not given.
        Ignored when g_cells is given, and ignored when delta_M >= 1
        (the split matrices are built from dg.hg_delta_matrices instead).
    g : float, optional
        Asymmetry parameter used only when Phi_w is None and g_cells is
        None, by default 0.0.
    beam_scat : (n_cells, N_dir) ndarray, optional
        Fixed first-scattering source of the solar beam
        (see beam_scattering_source); never iterated.  Added unchanged —
        with delta_M >= 1 the caller must pre-scale it via
        beam_scattering_source(delta_M=...).
    g_cells : (n_cells,) ndarray, optional
        Spatially varying asymmetry parameter. Cells are bucketed by
        g rounded to g_decimals and one BLAS matmul is done per bucket
        (plan section 2).
    g_decimals : int, optional
        Bucket resolution for g_cells, by default 2 (0.01 steps).
    Phi_cache : dict, optional
        Optional cache of the HG matrices, reused across iterations.
        Plain HG matrices are stored under the float key g_value
        ({g_value: Phi_w}); delta-M splits are stored under the tuple
        key (g_value, delta_M) ({(g_value, M): (Phi_w_delta, moments)}),
        so both kinds coexist and old plain-matrix entries stay valid
        for delta_M = 0.
    delta_M : int, optional
        Delta-M order, by default 0 (off, existing behavior unchanged).

    Returns
    -------
    S : (n_cells, N_dir) ndarray
    """
    if delta_M == 0:
        if g_cells is None:
            if Phi_w is None:
                Phi_w = dg.hg_matrix(g)
            S = (1.0 - omega)[:, None] * B[:, None] + omega[:, None] * (I @ Phi_w.T)
        else:
            cache = {} if Phi_cache is None else Phi_cache
            gv_u_list, rows_list, order, inv = _bucket_structure(
                cache, g_cells, g_decimals
            )
            S = np.broadcast_to(
                (1.0 - omega)[:, None] * B[:, None], (omega.size, dg.n_dir)
            ).copy()
            # bucket-contiguous blocks: one gather, per-bucket contiguous
            # slice matmuls, one scatter (the per-iteration boolean masks
            # and fancy-index gathers of the naive bucket loop dominated)
            S_b = S[order]
            om_b = omega[order]
            I_b = I[order]
            for gv_u, rows in zip(gv_u_list, rows_list):
                Phi_u = cache.get(gv_u)
                if Phi_u is None:
                    Phi_u = dg.hg_matrix(gv_u)
                    cache[gv_u] = Phi_u
                a, b = rows
                S_b[a:b] += om_b[a:b, None] * (I_b[a:b] @ Phi_u.T)
            S = S_b[inv]
    elif delta_M < 1 or int(delta_M) != delta_M:
        raise ValueError(f"delta_M must be a non-negative integer, got {delta_M}")
    elif g_cells is None:
        cache = {} if Phi_cache is None else Phi_cache
        key = (float(g), int(delta_M))
        split = cache.get(key)
        if split is None:
            split = dg.hg_delta_matrices(g, delta_M)
            cache[key] = split
        Phi_w_delta, moments = split
        gM = float(g) ** delta_M
        scat = I @ Phi_w_delta.T
        for Pn_w in moments:
            scat += I @ Pn_w.T
        denom = 1.0 - omega * gM
        S = ((1.0 - omega)[:, None] * B[:, None]
             + omega[:, None] * scat - omega[:, None] * gM * I) / denom[:, None]
    else:
        cache = {} if Phi_cache is None else Phi_cache
        gv_u_list, rows_list, order, inv = _bucket_structure(
            cache, g_cells, g_decimals
        )
        S_b = np.empty((omega.size, dg.n_dir))
        om_b = omega[order]
        B_b = B[order]
        I_b = I[order]
        for gv_u, (a, b) in zip(gv_u_list, rows_list):
            key = (float(gv_u), int(delta_M))
            split = cache.get(key)
            if split is None:
                split = dg.hg_delta_matrices(gv_u, delta_M)
                cache[key] = split
            Phi_w_delta, moments = split
            gM = gv_u ** delta_M
            scat = I_b[a:b] @ Phi_w_delta.T
            for Pn_w in moments:
                scat += I_b[a:b] @ Pn_w.T
            denom = 1.0 - om_b[a:b] * gM
            S_b[a:b] = ((1.0 - om_b[a:b])[:, None] * B_b[a:b, None]
                        + om_b[a:b, None] * scat
                        - om_b[a:b, None] * gM * I_b[a:b]) / denom[:, None]
        S = S_b[inv]
    if beam_scat is not None:
        S = S + beam_scat
    return S


def beam_scattering_source(dg, omega_sun, F0, tau_b, omega, g, g_cells=None,
                           delta_M=0):
    """
    First-scattering source of the direct solar beam (plan section 1):

        omega * Phi_HG(Omega_sun -> Omega_i) * F0 * exp(-tau_b)

    This term is fixed (the beam is not on the quadrature and is never
    iterated on it).  The phase-function vector is rescaled so that
    sum_i w_i Phi_i == 1 (same quadrature normalization as
    direction_grid.hg_matrix(normalize=True)): the beam first-scatter
    power then passes through the ordinate set exactly, instead of
    carrying the raw HG quadrature error (significant at large g).

    With delta_M >= 1 the per-cell albedo factor omega is replaced by
    omega / (1 - omega * g**M) (per-cell g for g_cells): the beam enters
    the delta-M RTE as kappa_s Phi F0 exp(-tau_b) / kappa_e_delta with
    kappa_e_delta = kappa_e - kappa_s * g**M.  tau_b itself stays the
    PHYSICAL beam optical depth (the direct beam attenuates with total
    extinction and is not on the quadrature).  The phase-function vector
    itself stays the FULL normalized HG — the beam is not on the
    quadrature, the Legendre-moment subtraction applies only to the
    diffuse field.

    Parameters
    ----------
    dg : direction_grid
    omega_sun : (3,) ndarray
        Unit beam propagation direction.
    F0 : float
        Beam irradiance at the domain top.
    tau_b : (n_cells,) ndarray
        Beam optical depth at cell centers (transport.beam_optical_depth).
    omega : (n_cells,) ndarray
        Single scattering albedo.
    g : float
        HG asymmetry parameter (scalar case).
    g_cells : (n_cells,) ndarray, optional
        Spatially varying HG asymmetry parameter; overrides g.
    delta_M : int, optional
        Delta-M order, by default 0 (off, no scaling).

    Returns
    -------
    S_beam : (n_cells, N_dir) ndarray
    """
    omega_sun = np.asarray(omega_sun, dtype=float)
    cos_sc = np.clip(dg.directions @ omega_sun, -1.0, 1.0)
    if g_cells is None:
        Phi = hg(cos_sc, g)  # (N_dir,)
        Phi = Phi / (dg.weights @ Phi)
        if delta_M >= 1:
            om = omega / (1.0 - omega * float(g) ** delta_M)
        else:
            om = omega
        S_beam = om[:, None] * (F0 * np.exp(-tau_b))[:, None] * Phi[None, :]
        return S_beam
    Phi = hg(cos_sc[None, :], np.asarray(g_cells, dtype=float)[:, None])
    Phi = Phi / (Phi @ dg.weights)[:, None]
    if delta_M >= 1:
        om = omega / (1.0 - omega * np.asarray(g_cells, dtype=float) ** delta_M)
    else:
        om = omega
    S_beam = om[:, None] * (F0 * np.exp(-tau_b))[:, None] * Phi
    return S_beam
