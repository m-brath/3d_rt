# -*- coding: utf-8 -*-
"""
Lambertian + emissive ground support for the MC family (phase M6,
plan/solver-test-matrix-plan.md section 7).

The ground *constitutive law* is not re-implemented here — `ground_radiance`
calls `rt3d.transport.ground_inflow`, so the SC and MC boundary conditions are
the same code by construction.  What this module adds is the MC-side sampling
machinery the ground needs:

* `bottom_columns` — the per-cell data of the cells touching the bottom face
  (the ground-level columns): footprint center/half-extent in x/y, footprint
  area, cell volume, in the `rt3d.transport.bottom_mask` order;
* `ground_radiance` — ``eps_g B_g + A_g (F_down + F_beam)/pi`` per column
  (via `rt3d.transport.ground_inflow`);
* `ground_beam_irradiance` — direct-beam irradiance at the ground
  (`rt3d.transport.beam_irradiance_ground`, physical kappa_e);
* `sample_ground_face` — positions on the bottom face sampled from a
  piecewise-constant per-column radiance (the 2-D analogue of
  `rt3d.mc.sources.sample_births`: categorical over columns with weights
  ``I_col * area_col``, uniform in the column footprint), paired with
  `rt3d.mc.sources.face_inflow_directions` cosine-weighted upward directions;
* `ground_face_weight` — the per-history power of such a population
  (``pi * sum_col I_col area_col / n``, the face-crossing convention of
  section 5 of plan/mc-plan.md);
* `down_irradiance_from_segments` — the per-column ground-level downward
  irradiance ``F_down`` measured from a set of flight segments (the MC twin of
  `rt3d.transport.downward_irradiance`: ``F_down_col = sum w |mu| len /
  V_col`` over segments in the bottom cell with ``mu < 0``).
"""

import numpy as np

from rt3d.mc.estimators import cell_volumes
from rt3d.mc.sources import face_inflow_directions
from rt3d.transport import beam_irradiance_ground, bottom_mask, ground_inflow


def ground_or_raise(bc):
    """
    Return ``bc.ground`` if set, rejecting the variable surface
    (flat ground only, same guard as the SC solver, plan section 1).
    """
    ground = getattr(bc, "ground", None)
    if ground is not None and ground.is_variable():
        raise NotImplementedError(
            "variable ground surface: the MC solvers support only a flat "
            "ground at the domain bottom (same guard as rt3d.solver)"
        )
    return ground


def bottom_columns(grid):
    """
    Ground-level column data, ordered like `rt3d.transport.bottom_mask`.

    Returns a dict with
        'mask' : (n_cells,) bool   the bottom mask
        'hx'   : (n_b,) float      x half-extent of the column footprint
        'hy'   : (n_b,) float      y half-extent
        'area' : (n_b,) float      footprint area
        'vol'  : (n_b,) float      cell volume (the F_down normalisation)
    """
    from rt3d.octree import octree_grid

    mb = bottom_mask(grid)
    vol_all = cell_volumes(grid)
    if isinstance(grid, octree_grid):
        h = grid.half[mb]
        hx = hy = h
    else:
        (x0, x1), (y0, y1), (_z0, _z1) = grid.box
        hx = np.full(mb.sum(), (x1 - x0) / grid.nx / 2.0)
        hy = np.full(mb.sum(), (y1 - y0) / grid.ny / 2.0)
    return {
        "mask": mb,
        "idx": np.flatnonzero(mb),
        "hx": hx,
        "hy": hy,
        "area": hx * hy * 4.0,
        "vol": vol_all[mb],
        "vol_all": vol_all,
    }


def ground_radiance(grid, ground, F_down=None, beam_irr=0.0):
    """
    Lambertian ground radiance per bottom column — **the same
    `rt3d.transport.ground_inflow` the SC solver uses**.

    ``F_down``/``beam_irr`` are per-bottom-column arrays (`bottom_mask` order)
    or scalars/0 for absent contributions; ``None`` means "no downward
    diffuse irradiance" (only the fixed emissivity/beam terms apply).
    """
    mb = bottom_mask(grid)
    if F_down is None or (np.ndim(F_down) == 0 and float(F_down) == 0.0):
        F_down = np.zeros(int(mb.sum()))
    return ground_inflow(F_down, ground, beam_irr=beam_irr)


def ground_beam_irradiance(grid, beam, kappa_e, tau_b, periodic_box=None):
    """
    Direct-beam irradiance at the ground below each bottom column
    (`rt3d.transport.beam_irradiance_ground`; ``tau_b`` is the beam optical
    depth at cell centers on the **physical** kappa_e, as in the SC solver).
    """
    return beam_irradiance_ground(grid, beam["omega_sun"], kappa_e, tau_b,
                                  beam["F0"], periodic_box=periodic_box)


def sample_ground_face(n, grid, I_col, rng, cols=None):
    """
    ``n`` seeding positions on the bottom face (offset just inside the domain)
    from the piecewise-constant per-column radiance ``I_col`` (>= 0).

    Columns are drawn categorically with weights ``I_col * area_col`` — the
    face-crossing measure of a Lambertian face source — and the position is
    uniform inside the column footprint.  Directions are NOT sampled here:
    pair with `rt3d.mc.sources.face_inflow_directions(n, 2, -1, rng)`.

    Returns ``(positions, col_idx)``; ``col_idx`` indexes the
    `bottom_mask`-ordered column array.  ``n == 0`` returns empty arrays.
    """
    if cols is None:
        cols = bottom_columns(grid)
    I_col = np.asarray(I_col, dtype=float)
    w = I_col * cols["area"]
    if not np.all(I_col >= 0.0):
        raise ValueError("ground radiance must be non-negative")
    Q = float(w.sum())
    z0 = grid.box[2][0]
    if n <= 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    if Q <= 0.0:
        raise ValueError("sample_ground_face called with zero total radiance")
    idx = rng.choice(I_col.size, size=int(n), p=w / Q)
    cen = grid.centers[cols["mask"]][idx]
    eps = 1e-9 * min(b[1] - b[0] for b in grid.box)
    pos = np.empty((int(n), 3))
    pos[:, 0] = cen[:, 0] + rng.uniform(-cols["hx"][idx], cols["hx"][idx])
    pos[:, 1] = cen[:, 1] + rng.uniform(-cols["hy"][idx], cols["hy"][idx])
    pos[:, 2] = z0 + eps
    return pos, idx


def ground_face_weight(I_col, cols, n):
    """
    Per-history power of a ground population of ``n`` histories seeded with
    radiance ``I_col``: ``pi * sum_col I_col area_col / n`` (plan/mc-plan.md
    section 5 face convention, inhomogeneous generalisation).
    """
    n = int(n)
    return float(np.pi * np.sum(np.asarray(I_col) * cols["area"]) / n)


def upward_directions(n, rng):
    """Cosine-weighted upward directions (alias for the face convention)."""
    return face_inflow_directions(int(n), 2, -1, rng)


def down_irradiance_from_segments(rec, weights, cols):
    """
    Per-column ground-level downward irradiance of a transported population
    (MC twin of `rt3d.transport.downward_irradiance`).

    ``rec`` is a segment record dict with keys ``'cell'`` (segment cell idx),
    ``'hist'`` (history idx of the segment), ``'len'`` and ``'dirs_hist'``
    (per-history flight direction), as produced by
    `rt3d.mc.forward._single_flight` / the segment gathers of
    `rt3d.mc.tracking.track_histories`; ``weights`` is the per-history power.

    ``F_down_col = sum_seg w |mu| len / V_col`` over segments with ``mu < 0``
    in the bottom cell — unbiased for the volume-averaged irradiance of the
    bottom cell.  Returns ``(n_b,)`` in `bottom_mask` order.
    """
    cell = np.asarray(rec["cell"])
    hist = np.asarray(rec["hist"])
    seglen = np.asarray(rec["len"], dtype=float)
    d = np.asarray(rec["dirs_hist"], dtype=float)
    if d.shape[0] == 1 and hist.size != 1:
        d = np.repeat(d, hist.size, axis=0)
    mu = d[hist, 2]
    m = mu < 0.0
    if not np.any(m):
        return np.zeros(cols["vol"].size)
    acc = np.zeros(cols["vol"].size)
    inv = -np.ones(cols["vol_all"].size, dtype=np.int64)
    inv[cols["idx"]] = np.arange(cols["idx"].size)
    slot = inv[cell[m]]
    ok = slot >= 0
    np.add.at(acc, slot[ok],
              np.asarray(weights)[hist[m]][ok] * np.abs(mu[m])[ok] * seglen[m][ok])
    return acc / cols["vol"]


def ground_column_lookup(grid, points, cols):
    """
    Column index (in `bottom_mask` order) of the bottom-face footprint
    containing the (x, y) of each row of ``points`` — the ground hit point of
    a backward march exiting through the bottom face (phase M6, stage-2
    ground inflow).  Unigrid: integer floor into the k = 0 layer; octree:
    `octree_grid.locate` of a point just above ``z0`` (clamped strictly
    inside, so the bottom-touching leaf of that column is returned).
    """
    from rt3d.octree import octree_grid

    p = np.atleast_2d(np.asarray(points, dtype=float))
    if isinstance(grid, octree_grid):
        z0 = grid.box[2][0]
        q = np.column_stack([p[:, 0], p[:, 1],
                             np.full(p.shape[0], z0 + 1e-6 * (grid.box[2][1] - z0))])
        leaf = grid.locate(q)
        inv = -np.ones(int(grid.n_leaf), dtype=np.int64)
        inv[cols["idx"]] = np.arange(cols["idx"].size)
        col = inv[leaf]
        if np.any(col < 0):
            raise RuntimeError("ground_column_lookup: locate() returned a leaf "
                               "not touching the bottom face")
        return col
    (x0, _x1), (y0, _y1), (_z0, _z1) = grid.box
    dx = (grid.box[0][1] - x0) / grid.nx
    dy = (grid.box[1][1] - y0) / grid.ny
    ix = np.clip(((p[:, 0] - x0) / dx).astype(int), 0, grid.nx - 1)
    iy = np.clip(((p[:, 1] - y0) / dy).astype(int), 0, grid.ny - 1)
    return iy * grid.nx + ix
