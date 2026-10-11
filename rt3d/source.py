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


def scattering_source(I, dg, omega, B, Phi_w=None, g=0.0, beam_scat=None,
                      g_cells=None, g_decimals=2, Phi_cache=None):
    """
    Direction-dependent source function

        S[cell, i] = (1 - omega) B + omega * sum_j w_j Phi(Omega_j -> Omega_i) I_j
                     + beam_scat[cell, i]

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
        Ignored when g_cells is given.
    g : float, optional
        Asymmetry parameter used only when Phi_w is None and g_cells is
        None, by default 0.0.
    beam_scat : (n_cells, N_dir) ndarray, optional
        Fixed first-scattering source of the solar beam
        (see beam_scattering_source); never iterated.
    g_cells : (n_cells,) ndarray, optional
        Spatially varying asymmetry parameter. Cells are bucketed by
        g rounded to g_decimals and one BLAS matmul is done per bucket
        (plan section 2).
    g_decimals : int, optional
        Bucket resolution for g_cells, by default 2 (0.01 steps).
    Phi_cache : dict, optional
        Optional {g_value: Phi_w} cache for the HG matrices, reused
        across iterations.

    Returns
    -------
    S : (n_cells, N_dir) ndarray
    """
    if g_cells is None:
        if Phi_w is None:
            Phi_w = dg.hg_matrix(g)
        S = (1.0 - omega)[:, None] * B[:, None] + omega[:, None] * (I @ Phi_w.T)
    else:
        cache = {} if Phi_cache is None else Phi_cache
        gv = np.round(np.asarray(g_cells, dtype=float), g_decimals)
        S = np.broadcast_to(
            (1.0 - omega)[:, None] * B[:, None], (omega.size, dg.n_dir)
        ).copy()
        for gv_u in np.unique(gv):
            Phi_u = cache.get(gv_u)
            if Phi_u is None:
                Phi_u = dg.hg_matrix(float(gv_u))
                cache[gv_u] = Phi_u
            rows = gv == gv_u
            S[rows] += omega[rows, None] * (I[rows] @ Phi_u.T)
    if beam_scat is not None:
        S = S + beam_scat
    return S


def beam_scattering_source(dg, omega_sun, F0, tau_b, omega, g, g_cells=None):
    """
    First-scattering source of the direct solar beam (plan section 1):

        omega * Phi_HG(Omega_sun -> Omega_i) * F0 * exp(-tau_b)

    This term is fixed (the beam is not on the quadrature and is never
    iterated on it).  The phase-function vector is rescaled so that
    sum_i w_i Phi_i == 1 (same quadrature normalization as
    direction_grid.hg_matrix(normalize=True)): the beam first-scatter
    power then passes through the ordinate set exactly, instead of
    carrying the raw HG quadrature error (significant at large g).

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

    Returns
    -------
    S_beam : (n_cells, N_dir) ndarray
    """
    omega_sun = np.asarray(omega_sun, dtype=float)
    cos_sc = np.clip(dg.directions @ omega_sun, -1.0, 1.0)
    if g_cells is None:
        Phi = hg(cos_sc, g)  # (N_dir,)
        Phi = Phi / (dg.weights @ Phi)
        return omega[:, None] * (F0 * np.exp(-tau_b))[:, None] * Phi[None, :]
    Phi = hg(cos_sc[None, :], np.asarray(g_cells, dtype=float)[:, None])
    Phi = Phi / (Phi @ dg.weights)[:, None]
    return omega[:, None] * (F0 * np.exp(-tau_b))[:, None] * Phi
