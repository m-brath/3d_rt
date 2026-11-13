# -*- coding: utf-8 -*-
"""
Plane-parallel 1D slab case (plan/plan.md phase 2 / plan/mc-plan.md M1).

The same configuration is used by `scripts/phase2_slab.py` (short
characteristics) and `scripts/mc_forward_slab.py` (Monte Carlo), so the two
solver families see exactly the same problem:

    slab          z in [0, tau/kappa_e], kappa_e uniform, periodic x/y,
                  cubic cells (hx = hy = hz)
    scattering    isotropic (g = 0) or HG, single scattering albedo omega
    emission      B = 0 (pure scattering / pure absorption)
    boundaries    I = top_I for ordinates entering through the top face,
                  vacuum at the bottom, periodic sides

Reference solutions (identical to the ones used in phase 2):

    `d1o_J`        1D discrete ordinates on the *same* z cells with 64
                   Gauss-Legendre mu levels — the angular error is negligible,
                   so this is the strong reference for a 42-ordinate SC run
                   and for the angle-exact MC runs (whose only error is then
                   statistical plus the reference's own cell discretization)
    `two_stream_J` analytic Eddington two-stream solution with Marshak
                   boundary conditions — context curve only
"""

import numpy as np

from rt3d.geometry import cuboid, unigrid


def slab_grid(tau=5.0, kappa_e=1.0, nz=50, nxy=4):
    """
    Slab lattice: nz z-cells of the optical thickness ``tau / nz`` each,
    cubic cells, periodic x/y over an ``nxy x nxy`` footprint.

    Returns
    -------
    grid : rt3d.geometry.unigrid
    box : rt3d.geometry.cuboid
        Periodic box matching the grid.
    hz : float
    """
    hz = tau / kappa_e / nz
    hx = hz
    x = (np.arange(nxy) + 0.5) * hx
    y = (np.arange(nxy) + 0.5) * hx
    z = (np.arange(nz) + 0.5) * hz
    grid = unigrid(x, y, z)
    box = cuboid(0.0, nxy * hx, 0.0, nxy * hx, grid.box[2][0], grid.box[2][1])
    return grid, box, hz


def slab_profile(grid, nz):
    """
    Vertical profile selector: the x,y-averaged slab is represented by one
    column of the 3D field (flat-index order k*ny*nx + j*nx + i, column
    (i, j) = (1, 1)).
    """
    def profile(field):
        return np.asarray(field).reshape(grid.nz, grid.nx, grid.ny)[:, 1, 1]

    return profile


def d1o_J(kappa_e, omega, hz, nz, top_I=1.0, n_iter=4000, tol=1e-12,
          mu_levels=64):
    """
    1D discrete-ordinates reference ``J(k)`` of the slab, cells k = 0
    (bottom) .. nz-1 (top).  Plane-parallel, isotropic scattering, B = 0,
    isotropic incident intensity `top_I` at the top face, vacuum at the
    bottom, plain lambda iteration.  Same implementation as the reference in
    `scripts/phase2_slab.py`.
    """
    from scipy.special import roots_legendre

    mu, w = roots_legendre(mu_levels)
    mu, w = mu[mu > 0], w[mu > 0]        # J = 1/2 sum w (I_up + I_dn), sum w = 1

    S = np.zeros(nz)
    h_half = hz / 2.0
    for _ in range(n_iter):
        J = np.zeros(nz)
        for mu_m, w_m in zip(mu, w):
            tau_h = kappa_e * h_half / mu_m        # half-cell optical depth
            # downward: enters at the top face with I = top_I
            I = np.zeros(nz)
            t_prev = np.exp(-tau_h)                # top face -> center of cell nz-1
            I[nz - 1] = top_I * t_prev + S[nz - 1] * (1 - t_prev)
            for k in range(nz - 2, -1, -1):
                # center k+1 -> center k: half in cell k+1, half in cell k
                e = np.exp(-2 * tau_h)
                I[k] = (
                    I[k + 1] * e
                    + S[k + 1] * (1 - np.exp(-tau_h)) * np.exp(-tau_h)
                    + S[k] * (1 - np.exp(-tau_h))
                )
            J += 0.5 * w_m * I
            # upward: enters at the bottom face with I = 0
            I = np.zeros(nz)
            t_prev = np.exp(-tau_h)
            I[0] = S[0] * (1 - t_prev)
            for k in range(1, nz):
                e = np.exp(-2 * tau_h)
                I[k] = (
                    I[k - 1] * e
                    + S[k - 1] * (1 - np.exp(-tau_h)) * np.exp(-tau_h)
                    + S[k] * (1 - np.exp(-tau_h))
                )
            J += 0.5 * w_m * I
        S_new = omega * J
        if np.max(np.abs(S_new - S)) < tol:
            S = S_new
            break
        S = S_new
    return J                                        # from the last converged sweep


def two_stream_J(kappa_e, omega, z0, z1, zc, top_I=1.0):
    """
    Analytic Eddington two-stream solution (Marshak BCs: I_down(top) = top_I,
    I_up(bottom) = 0 with I+- = J +- H), evaluated at the cell centers `zc`.
    Same as the reference in `scripts/phase2_slab.py`.
    """
    lam = kappa_e * np.sqrt(3.0 * (1.0 - omega))
    gam = np.sqrt((1.0 - omega) / 3.0)
    M = np.array(
        [
            [(1 - gam) * np.exp(lam * z0), (1 + gam) * np.exp(-lam * z0)],
            [(1 + gam) * np.exp(lam * z1), -(1 - gam) * np.exp(-lam * z1)],
        ]
    )
    b = np.array([0.0, top_I])
    A, C = np.linalg.solve(M, b)
    return A * np.exp(lam * np.asarray(zc)) + C * np.exp(-lam * np.asarray(zc))
