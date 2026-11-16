# -*- coding: utf-8 -*-
"""
Diagnostics (plan section 6): heating rate, flux divergence, global energy
balance, all-sky output images, angular variance of the intensity.

Energy conservation (verification invariant 4): from the RTE,

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


def _is_octree(grid):
    """Duck-typed octree test (avoids a circular import at module level)."""
    return hasattr(grid, "leaves_all") and hasattr(grid, "leaf_of_node")


def cell_volumes(grid):
    """
    Cell volume per grid cell.

    Unigrid: constant ``hx hy hz``.  Octree: ``(2 half)^3`` per leaf, so
    coarse leaves carry their full volume and no volume is counted twice
    (the leaves tile the domain).
    """
    if _is_octree(grid):
        half = np.asarray(grid.half, dtype=float)
        return (2.0 * half) ** 3
    hx = grid.x[1] - grid.x[0] if grid.nx > 1 else 1.0
    hy = grid.y[1] - grid.y[0] if grid.ny > 1 else 1.0
    hz = grid.z[1] - grid.z[0] if grid.nz > 1 else 1.0
    return np.full(grid.n_cells, hx * hy * hz)


def _face_mu_flux(I, dg, idx):
    """Signed F_z = sum w omega_z I per cell for ordinates idx."""
    mu = dg.directions[idx, 2]
    return I[:, idx] @ (dg.weights[idx] * mu)


def _octree_face_flux(grid, Fz, top):
    """
    F_z on the top/bottom face, linearly extrapolated from the boundary
    leaves and their interior neighbours.

    Each boundary leaf is extrapolated from its own center value F_i and the
    value of the leaf one face inward, F_j, using the actual center spacing
    ``dz = |c_i,z - c_j,z|``:  ``F_face = F_i + h_i (F_i - F_j) / dz``
    (exact for a linear F, second order for smooth F; ``h_i`` is the leaf
    half size, i.e. the center-to-face distance).  The neighbour is found by
    locating a point just inside the opposite face, so a 2:1 graded mesh
    still pairs every boundary leaf with a real neighbor.

    Returns ``(flux, area)`` arrays over the boundary leaves.
    """
    z0, z1 = grid.box[2]
    hz = np.asarray(grid.half, dtype=float)     # cube leaves: one half size
    centers = grid.centers
    if top:
        sel = np.flatnonzero(np.abs(centers[:, 2] + hz - z1) <= 1e-9)
        probe_z = centers[sel, 2] - hz[sel] - 1e-9
    else:
        sel = np.flatnonzero(np.abs(centers[:, 2] - hz - z0) <= 1e-9)
        probe_z = centers[sel, 2] + hz[sel] + 1e-9
    if sel.size == 0:
        raise ValueError("grid has no leaves touching the requested face")

    probe = np.column_stack([centers[sel, 0], centers[sel, 1], probe_z])
    j = grid.locate(probe)
    dz = np.abs(centers[sel, 2] - centers[j, 2])
    if np.any(dz <= 0.0):
        raise ValueError("degenerate face extrapolation: neighbour is the leaf")
    face = Fz[sel] + hz[sel] * (Fz[sel] - Fz[j]) / dz
    area = (2.0 * hz[sel]) ** 2
    return face, area


def face_powers(grid, I, dg):
    """
    Outward radiative power through the horizontal domain faces.

    ``out_top`` is ``int F_z dA`` over the top face and ``out_bottom`` is
    ``int (-F_z) dA`` over the bottom face (positive outward); the periodic
    side faces contribute nothing.  Both hemispheres are included in F_z,
    so a reflecting/emitting ground leaves through the bottom face as well.

    Unigrid: net F_z linearly extrapolated from the two boundary layers
    (second order in ``hz``).  Octree: per-boundary-leaf extrapolation from
    the leaf and its inward neighbour (see `_octree_face_flux`).

    Parameters
    ----------
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
    I : (n_cells, N_dir) ndarray
    dg : rt3d.directions.direction_grid

    Returns
    -------
    dict with 'out_top', 'out_bottom', 'flux_top', 'flux_bottom', 'area'
    (flux/area per boundary cell, for diagnostics).
    """
    I = np.asarray(I, dtype=float)
    Fz = _face_mu_flux(I, dg, np.arange(dg.n_dir))

    if _is_octree(grid):
        top_flux, top_area = _octree_face_flux(grid, Fz, top=True)
        bot_flux, bot_area = _octree_face_flux(grid, Fz, top=False)
        return {
            "out_top": float(np.sum(top_flux * top_area)),
            "out_bottom": float(-np.sum(bot_flux * bot_area)),
            "flux_top": top_flux,
            "flux_bottom": bot_flux,
            "area": (top_area, bot_area),
        }

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    A = (grid.x[1] - grid.x[0]) * (grid.y[1] - grid.y[0]) if nx > 1 else 1.0
    Fz4 = Fz.reshape(nz, ny, nx)
    if nz > 2:
        Fz_top = 1.5 * Fz4[-1] - 0.5 * Fz4[-2]
        Fz_bot = 1.5 * Fz4[0] - 0.5 * Fz4[1]
    else:
        Fz_top = Fz4[-1]
        Fz_bot = Fz4[0]
    return {
        "out_top": float(Fz_top.sum() * A),
        "out_bottom": float(-Fz_bot.sum() * A),
        "flux_top": Fz_top.ravel(),
        "flux_bottom": Fz_bot.ravel(),
        "area": (np.full(Fz_top.size, A), np.full(Fz_bot.size, A)),
    }


def flux_divergence(grid, F):
    """
    Flux divergence from cell-center fluxes.

    Unigrid: central differences in x, y with periodic wrap, central in z
    interior, one-sided at top/bottom faces (the original implementation).

    Octree: per leaf, the same central difference evaluated between the
    centers of the leaves found on either side along each axis (a point just
    inside each face is located, so 2:1 graded meshes — and coarse/fine
    neighbours — give a real finite-difference pair); one-sided at the
    domain faces.  On a uniform octree this is the unigrid stencil; on a
    graded mesh the difference quotient is centred halfway between the two
    neighbour centers instead of at the leaf center, so the truncation error
    is O(h) rather than O(h^2).  The global closure (`energy_balance`) does
    not use this route — it integrates face powers directly — so the graded
    error only affects local div-F maps.

    F : (n_cells, 3) ndarray, cell-major.
    """
    F = np.asarray(F, dtype=float)
    if _is_octree(grid):
        return _flux_divergence_octree(grid, F)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    F4 = F.reshape(nz, ny, nx, 3)

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


def _flux_divergence_octree(grid, F):
    """Per-leaf divergence with face-probe finite differences (see above)."""
    centers = grid.centers
    n = centers.shape[0]
    grad = np.zeros((n, 3))
    eps = 1e-9
    for axis in range(3):
        h = np.asarray(grid.half, dtype=float)
        plus = centers.copy()
        minus = centers.copy()
        plus[:, axis] += h + eps
        minus[:, axis] -= h + eps
        jp = grid.locate(plus)
        jm = grid.locate(minus)
        # a probe clamped back into the leaf means that face is a domain
        # boundary: use the one-sided difference against the interior leaf
        outside_plus = jp == np.arange(n)
        outside_minus = jm == np.arange(n)
        denom = centers[jp, axis] - centers[jm, axis]
        ok = (~outside_plus) & (~outside_minus) & (denom != 0.0)
        g = np.zeros(n)
        g[ok] = (F[jp[ok], axis] - F[jm[ok], axis]) / denom[ok]
        hi = ~ok & ~outside_minus & (jm != np.arange(n))
        if np.any(hi):
            d = centers[:, axis] - centers[jm, axis]
            good = d != 0.0
            idx = np.flatnonzero(hi & good)
            g[idx] = (F[idx, axis] - F[jm[idx], axis]) / d[idx]
        lo = ~ok & ~outside_plus & (jp != np.arange(n))
        if np.any(lo):
            d = centers[jp, axis] - centers[:, axis]
            good = d != 0.0
            idx = np.flatnonzero(lo & good)
            g[idx] = (F[jp[idx], axis] - F[idx, axis]) / d[idx]
        grad[:, axis] = g
    return grad.sum(axis=1)



def angular_variance(I, dg):
    """
    Angular variance of the intensity around the mean intensity, per cell:

        sigma^2(r) = (1/4pi) * int (I(r, Omega) - J(r))^2 dOmega
                   ~ (1/4pi) * sum_i w_i (I_i - J)^2

    Zero for an isotropic field; measures the angular anisotropy of the
    radiation field on the ordinate quadrature (units of I^2).  For a field
    I = J + a * Omega_z on an isotropic direction grid this is ~ a^2 / 3.

    Parameters
    ----------
    I : (n_cells, N_dir) ndarray
        Cell-center intensities per ordinate.
    dg : rt3d.directions.direction_grid

    Returns
    -------
    (n_cells,) ndarray
    """
    I = np.asarray(I, dtype=float)
    J = mean_intensity(I, dg)
    return ((I - J[:, None]) ** 2 * dg.weights).sum(axis=1) / (4.0 * np.pi)


def energy_balance(grid, I, dg, kappa_a, B):
    """
    Global energy conservation check (verification invariant 4), any grid.

    Net outward flux through the top and bottom faces vs the volume-
    integrated heating rate ``int 4 pi kappa_a (B - J) dV = int div F dV``.
    The net F_z includes both hemisphere ordinates: with a ground, the
    upward (reflected + emitted) part leaves through the bottom face as
    well.  Periodic sides contribute nothing.

    The face powers come from `face_powers` -- on the unigrid a linear
    extrapolation of the two boundary layers (second order in hz), on the
    octree a per-boundary-leaf extrapolation from the leaf and its inward
    neighbour (see `_octree_face_flux`; the leaves tile every face, so no
    area is missed or double-counted).  The volume integral uses
    `cell_volumes`, i.e. ``(2 half)^3`` on the octree.  The residual
    therefore measures *only* the face-extrapolation error, not a volume
    bookkeeping artifact.

    Returns
    -------
    dict with 'out_top', 'out_bottom' (outward power), 'heating'
    (volume-integrated 4 pi kappa_a (B - J), i.e. radiation energy gain),
    'residual' = out_top + out_bottom - heating, and 'relative' =
    residual / max(|out_top| + |out_bottom|, |heating|, tiny).
    """
    I = np.asarray(I, dtype=float)
    fp = face_powers(grid, I, dg)
    J = mean_intensity(I, dg)
    heating = float(np.sum(heating_rate(kappa_a, J, B) * cell_volumes(grid)))

    out_top = fp["out_top"]
    out_bottom = fp["out_bottom"]
    residual = out_top + out_bottom - heating
    scale = max(abs(out_top) + abs(out_bottom), abs(heating), 1e-30)
    return {
        "out_top": out_top,
        "out_bottom": out_bottom,
        "heating": heating,
        "residual": residual,
        "relative": residual / scale,
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
