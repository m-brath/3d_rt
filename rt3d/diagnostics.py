# -*- coding: utf-8 -*-
"""
Diagnostics (plan section 6): heating rate, flux divergence, global energy
balance, all-sky output images.

Energy conservation (validation anchor 4): from the RTE,

    div F = int (Omega . grad I) dOmega = 4 pi S - kappa_e 4 pi J
          = 4 pi kappa_a (B - J)      (S = (1-omega) B + omega J)

so the flux divergence of the converged intensity field must equal the
net heating rate 4 pi kappa_a (B - J) per cell.
"""

import numpy as np

from rt3d.source import mean_intensity


def heating_rate(kappa_a, J, B):
    """
    Net radiative heating rate per volume: 4 pi kappa_a (B - J).

    Positive = radiation gains energy from the matter (cooling gas);
    with B = 0 this is the (negative) absorption of the radiation field.
    """
    return 4.0 * np.pi * np.asarray(kappa_a) * (np.asarray(B) - np.asarray(J))


def flux_divergence(grid, F):
    """
    Flux divergence per cell from cell-center fluxes (uniform unigrid):
    central differences in x, y with periodic wrap, central in z interior,
    one-sided at top/bottom faces.

    F : (n_cells, 3) ndarray, k-major cell ordering.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    F4 = np.asarray(F, dtype=float).reshape(nz, ny, nx, 3)

    def spacing(arr):
        return float(arr[1] - arr[0]) if arr.size > 1 else 1.0

    dx, dy, dz = spacing(grid.x), spacing(grid.y), spacing(grid.z)

    dFx = (np.roll(F4, -1, axis=2) - np.roll(F4, 1, axis=2))[..., 0] / (2 * dx)
    dFy = (np.roll(F4, -1, axis=1) - np.roll(F4, 1, axis=1))[..., 1] / (2 * dy)
    dFz = np.zeros((nz, ny, nx))
    if nz > 2:
        dFz[1:-1] = (F4[2:, :, :, 2] - F4[:-2, :, :, 2]) / (2 * dz)
    dFz[0] = (F4[1, :, :, 2] - F4[0, :, :, 2]) / dz
    dFz[-1] = (F4[-1, :, :, 2] - F4[-2, :, :, 2]) / dz
    return (dFx + dFy + dFz).reshape(grid.n_cells)


def _face_mu_flux(I, dg, idx):
    """Signed F_z = sum w omega_z I per cell for ordinates idx."""
    mu = dg.directions[idx, 2]
    return I[:, idx] @ (dg.weights[idx] * mu)


def energy_balance(grid, I, dg, kappa_a, B):
    """
    Global energy conservation check on a uniform unigrid.

    Net outward flux through the top and bottom faces (net F_z linearly
    extrapolated from the boundary-layer cell centers to the face,
    second-order; periodic sides contribute nothing) versus the
    volume-integrated heating rate int 4 pi kappa_a (B - J) dV =
    int div F dV.  The net F_z includes both hemisphere ordinates: with
    a ground, the upward (reflected + emitted) part leaves through the
    bottom face as well.

    Returns
    -------
    dict with 'out_top', 'out_bottom' (outward power), 'heating'
    (volume-integrated 4 pi kappa_a (B - J), i.e. radiation energy gain)
    and 'residual' = out_top + out_bottom - heating.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    A = (grid.x[1] - grid.x[0]) * (grid.y[1] - grid.y[0]) if nx > 1 else 1.0
    V = A * (grid.z[1] - grid.z[0]) if nz > 1 else 1.0

    I = np.asarray(I)
    all_idx = np.arange(dg.n_dir)
    Fz = _face_mu_flux(I, dg, all_idx).reshape(nz, ny, nx)
    if nz > 2:
        Fz_top = 1.5 * Fz[-1] - 0.5 * Fz[-2]
        Fz_bot = 1.5 * Fz[0] - 0.5 * Fz[1]
    else:
        Fz_top = Fz[-1]
        Fz_bot = Fz[0]
    out_top = Fz_top.sum() * A
    out_bottom = (-Fz_bot).sum() * A

    J = mean_intensity(I, dg)
    heating = heating_rate(kappa_a, J, B).sum() * V
    return {
        "out_top": out_top,
        "out_bottom": out_bottom,
        "heating": heating,
        "residual": out_top + out_bottom - heating,
    }


def allsky(grid, I, dg, face="top"):
    """
    All-sky output images: outgoing intensity at a horizontal boundary
    face for every ordinate leaving that face (uniform unigrid).

    face='top'    : upward ordinates, top cell layer (outgoing radiation
                    at the top of the domain)
    face='bottom' : downward ordinates, bottom cell layer (downwelling
                    view at the ground)

    Returns
    -------
    dict with 'omega' (n_out, 3), 'weights' (n_out,), 'images'
    (n_out, ny, nx) intensity images.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    I = np.asarray(I)
    if face == "top":
        cells = slice((nz - 1) * ny * nx, nz * ny * nx)
        idx = dg.up_idx
    elif face == "bottom":
        cells = slice(0, ny * nx)
        idx = dg.down_idx
    else:
        raise ValueError("face must be 'top' or 'bottom'")
    images = I[cells][:, idx].reshape(ny, nx, idx.size).transpose(2, 0, 1)
    return {
        "omega": dg.directions[idx],
        "weights": dg.weights[idx],
        "images": images,
    }
