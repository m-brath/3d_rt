# -*- coding: utf-8 -*-
"""
Woodcock null-collision tracking core of the MC solver family
(plan/mc-plan.md section 2).

One flight-sampler / event-decision core shared by every MC solver:

- **mode="cell"** (default): on entry to a cell (unigrid cell or octree
  leaf) set ``kappa_max = kappa_e(cell)``.  On a piecewise-constant grid the
  null-collision rate ``kappa_max - kappa_e`` is exactly zero — every
  collision is real — while the code remains a Woodcock tracker.
- **mode="global"**: ``kappa_max = max_domain(kappa_e)`` (or a supplied
  cap), so low-opacity / vacuum regions produce **real null collisions**
  (direction and weight unchanged, flight continues).  This is the
  reference implementation of the null-collision concept and enables a
  fully vectorized photon-batch loop: every live photon is advanced one
  flight at a time and nulls in vacuum are cheap.

Both modes share the same event-decision logic (`event_decision`) and the
same DDA traversal kernels; the tests (tests/test_mc_tracking.py) assert
that cell and global mode agree statistically on the transmittance anchor.

The two DDA walkers (`walk_lattice`, `walk_octree`) reproduce the SC
marches `rt3d.geometry.march_lattice` and `rt3d.octree.march_octree`
segment-for-segment (same lattice formulas, midpoint cell attribution,
periodic-wrap snap / eps re-locate), so MC and SC see exactly the same
geometry (plan/mc-plan.md M0: "periodic wrap + ground entry identical to
march_lattice / march_octree segment counts").

RNG: NumPy `default_rng` (PCG64), reproducible seeds.  The numba fast path
follows the rt3d/jit.py pattern and arrives with M4; every entry point
here is the reference NumPy path.
"""

import numpy as np

from rt3d.geometry import ray_box_exit
from rt3d.mc.sources import hg_scatter_directions
from rt3d.octree import octree_grid


# =============================================================================
# collision event codes and scalar Woodcock decision
# =============================================================================

EV_EXIT = -1     # ray left the domain / its path budget without a real event
EV_SCATTER = 0   # real scattering event (HG direction resampled by caller)
EV_ABSORB = 1    # true absorption
EV_NULL = 2      # null collision: direction and weight unchanged, continue

_EVENT_NAMES = {EV_EXIT: "exit", EV_SCATTER: "scatter",
                EV_ABSORB: "absorb", EV_NULL: "null"}


def event_decision(u, kappa_max, kappa_s, kappa_e):
    """
    Woodcock event decision at the end of a sampled flight.

    A flight was sampled with the majorant rate ``kappa_max >= kappa_e``;
    the event type follows from the cumulative rates scatter (``kappa_s``),
    absorb (``kappa_e``), null (``kappa_max - kappa_e``):

        u * kappa_max <= kappa_s            -> scatter
        u * kappa_max <= kappa_e            -> absorb
        otherwise                           -> null

    In cell mode ``kappa_max = kappa_e(cell)`` the null branch is reached
    with probability zero — the same code path serves both modes.

    Parameters
    ----------
    u : (N,) or scalar ndarray
        Uniform samples in [0, 1).
    kappa_max : float or (N,) ndarray
        Majorant rate used to sample the flight (must be > 0).
    kappa_s, kappa_e : (N,) ndarray or float
        Local scattering / extinction coefficients at the collision point.

    Returns
    -------
    event : (N,) ndarray of int
        EV_SCATTER / EV_ABSORB / EV_NULL.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    kappa_s = np.atleast_1d(np.asarray(kappa_s, dtype=float))
    kappa_e = np.atleast_1d(np.asarray(kappa_e, dtype=float))
    rate = u * kappa_max
    return np.where(rate <= kappa_s, EV_SCATTER,
                    np.where(rate <= kappa_e, EV_ABSORB, EV_NULL))


def flight_distance(u, kappa_max):
    """
    Woodcock flight length: the majorant ``exp(-int kappa_max ds)`` is
    inverted to an exponential for constant ``kappa_max``,

        s = -ln(1 - u) / kappa_max,

    which for piecewise-constant majorants is sampled per majorant piece
    (memoryless property at cell entries and null collisions).  Returns
    ``inf`` where ``kappa_max == 0`` (the flight never terminates — the
    photon crosses such a cell without any event).
    """
    u = np.asarray(u, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -np.log1p(-u) / np.asarray(kappa_max, dtype=float)
    return np.where(np.asarray(kappa_max, dtype=float) > 0.0, s, np.inf)


def path_integral_from_segments(seg_ray, seg_cell, seg_len, cell_field, n_rays):
    """
    Per-ray path integral ``sum_segments cell_field[cell] * length`` from a
    segment record (as produced by `walk_lattice` / `walk_octree`).  Used
    for the analytic ``tau = int kappa_e ds`` of the transmittance anchor
    and later for deterministic tallies.
    """
    out = np.zeros(n_rays, dtype=float)
    np.add.at(out, seg_ray, np.asarray(cell_field)[seg_cell] * seg_len)
    return out


# =============================================================================
# DDA face-distance formulas (identical to the SC marches)
# =============================================================================


def _lattice_face_times(p, d, grid):
    """
    Distance to the next lattice plane along per-ray directions — same
    snap formulas as rt3d.geometry.march_lattice.

    Returns (t_face, axis, side) with side = sign(d[axis]).
    """
    n = p.shape[0]
    t = np.full(n, np.inf)
    axis = np.zeros(n, dtype=int)
    side = np.zeros(n, dtype=int)
    for ax, (h, s0) in enumerate(
        ((grid.hx, grid.box[0][0]), (grid.hy, grid.box[1][0]),
         (grid.hz, grid.box[2][0]))
    ):
        if not np.isfinite(h):
            continue
        eps = 1e-9 * h
        m = np.floor((p[:, ax] - s0 + eps) / h)
        base = s0 + m * h
        tn = np.full(n, np.inf)
        pos = d[:, ax] > 0.0
        neg = d[:, ax] < 0.0
        if pos.any():
            tn[pos] = (base[pos] + h - p[pos, ax]) / d[pos, ax]
        if neg.any():
            # if the point sits exactly on the lower face of its cell, the
            # next face in -axis direction is one cell further down
            nxt = np.where(p[:, ax] > base + eps, base, base - h)
            tn[neg] = (nxt[neg] - p[neg, ax]) / d[neg, ax]
        upd = tn < t
        t = np.where(upd, tn, t)
        axis = np.where(upd, ax, axis)
        side = np.where(upd, np.where(d[:, ax] > 0.0, 1, -1), side)
    return t, axis, side


def _octree_face_times(p, d, cell, oct, periodic_box):
    """
    Distance to the current leaf face along per-ray directions — same
    formulas as rt3d.octree.march_octree (leaf box uses the periodic image
    of the leaf center closest to the ray position).

    Returns (t_face, axis, side) with side = sign(d[axis]).
    """
    c = oct.centers[cell].copy()
    h = oct.half[cell]
    if periodic_box is not None:
        c[:, 0] += periodic_box.Lx * np.round((p[:, 0] - c[:, 0]) / periodic_box.Lx)
        c[:, 1] += periodic_box.Ly * np.round((p[:, 1] - c[:, 1]) / periodic_box.Ly)

    n = p.shape[0]
    t = np.full(n, np.inf)
    axis = np.zeros(n, dtype=int)
    side = np.zeros(n, dtype=int)
    for ax in range(3):
        da = d[:, ax]
        pos = da > 0.0
        neg = da < 0.0
        if not (pos.any() or neg.any()):
            continue
        tf = np.full(n, np.inf)
        tf[pos] = (c[pos, ax] + h[pos] - p[pos, ax]) / da[pos]
        tf[neg] = (c[neg, ax] - h[neg] - p[neg, ax]) / da[neg]
        upd = tf < t
        t = np.where(upd, tf, t)
        axis = np.where(upd, ax, axis)
        side = np.where(upd, np.where(da > 0.0, 1, -1), side)
    return t, axis, side


# =============================================================================
# DDA segment walkers (SC-geometry twins)
# =============================================================================


def walk_lattice(starts, directions, grid, budgets=None, periodic_box=None,
                 max_steps=None):
    """
    Exact DDA segment walk through the unigrid lattice, vectorized over
    rays, with per-ray unit directions and path-length budgets.

    The traversal is the *same* algorithm as rt3d.geometry.march_lattice
    (lattice face formulas, midpoint cell attribution, ray-box exit with
    the phase-5 periodic-wrap snap, zero-length-step filter), generalized
    to per-ray directions and budgets.  With a shared direction and budgets
    beyond the box exit the segment records are identical to
    `march_lattice` ray-for-ray (asserted in tests/test_mc_tracking.py).

    Parameters
    ----------
    starts : (K, 3) ndarray
        Ray start points inside the domain box.
    directions : (3,) or (K, 3) ndarray
        Unit ray directions.
    grid : rt3d.geometry.unigrid
    budgets : (K,) or scalar, optional
        Path-length budget per ray; None -> march to the box exit.
    periodic_box : rt3d.geometry.cuboid, optional
        Rays exiting through a periodic x/y face wrap and continue.

    Returns
    -------
    seg_ray, seg_cell, seg_len : ndarray
        Step-major segment records (ray index, traversed cell, length).
    exit_t : (K,) ndarray
        Total path length travelled (box exit or budget end).
    exit_axis, exit_side : (K,) ndarray of int
        Face the ray exited through (exit_axis = -1 for budget end).
    """
    o = np.atleast_2d(np.asarray(starts, dtype=float))
    K = o.shape[0]
    d = np.asarray(directions, dtype=float)
    if d.ndim == 1:
        d = np.tile(d, (K, 1))
    p = o.copy()
    remaining = (
        np.full(K, np.inf) if budgets is None
        else np.broadcast_to(np.asarray(budgets, dtype=float), (K,)).astype(float).copy()
    )

    if max_steps is None:
        diag = np.sqrt(sum((b[1] - b[0]) ** 2 for b in grid.box))
        max_path = diag
        if periodic_box is not None:
            zmax = np.abs(grid.box[2][1] - grid.box[2][0]) / np.maximum(np.abs(d[:, 2]), 1e-300)
            max_path = max(max_path, np.max(np.minimum(remaining, zmax)))
        planes_per_length = (np.abs(d) / np.array([grid.hx, grid.hy, grid.hz])).sum(axis=1)
        max_steps = int(np.ceil(max_path * planes_per_length.max())) + 4

    box = grid.box
    active = np.arange(K)
    exit_t = np.zeros(K)
    exit_axis = np.zeros(K, dtype=int)
    exit_side = np.zeros(K, dtype=int)
    budget_end = np.zeros(K, dtype=bool)
    steps = []

    for _ in range(max_steps):
        if active.size == 0:
            break
        pa = p[active]
        da = d[active]

        t_face, fax, fside = _lattice_face_times(pa, da, grid)
        t_exit, eax, esd = ray_box_exit(pa, da, box)
        t = np.minimum(np.minimum(t_face, t_exit), remaining[active])
        t = np.maximum(t, 0.0)

        # cell attributed at the segment midpoint (march_lattice convention)
        cell = grid.locate(pa + (0.5 * t)[:, None] * da)

        keep = t > 1e-12
        if np.any(keep):
            steps.append((active[keep].copy(), cell[keep], t[keep]))
        p[active] = pa + t[:, None] * da
        exit_t[active] += t
        remaining[active] -= t

        exited = t >= t_exit - 1e-12 * np.maximum(1.0, np.linalg.norm(pa, axis=1))

        # periodic x/y exits wrap and keep marching; all other exits end the ray
        wrapped = np.zeros(active.size, dtype=bool)
        if periodic_box is not None:
            wrapped = exited & (
                ((eax == 0) & periodic_box.periodic_x) | ((eax == 1) & periodic_box.periodic_y)
            )
            if np.any(wrapped):
                pw = periodic_box.wrap_xy(p[active[wrapped]])
                # phase-5 snap: a ray exiting within float noise of a
                # periodic face must re-enter at the *opposite* face
                dw = da[wrapped]
                for axis, (lo, hi) in enumerate(
                    ((periodic_box.x0, periodic_box.x1), (periodic_box.y0, periodic_box.y1))
                ):
                    if not (periodic_box.periodic_x if axis == 0 else periodic_box.periodic_y):
                        continue
                    tol = 1e-9 * (hi - lo)
                    at_lo = (dw[:, axis] < 0.0) & (pw[:, axis] <= lo + tol)
                    pw[at_lo, axis] = hi
                    at_hi = (dw[:, axis] > 0.0) & (pw[:, axis] >= hi - tol)
                    pw[at_hi, axis] = lo
                p[active[wrapped]] = pw

        terminated = exited & ~wrapped
        newly_done = active[terminated]
        exit_axis[newly_done] = eax[terminated]
        exit_side[newly_done] = esd[terminated]

        # budget exhaustion (path shorter than the box exit) also ends a ray
        active = active[~terminated]
        if budgets is not None:
            rm = remaining[active] <= 1e-12
            if np.any(rm):
                budget_end[active[rm]] = True
                active = active[~rm]
        if active.size == 0:
            break

    exit_axis[budget_end] = -1
    if len(steps) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty.copy(), np.zeros(0), exit_t, exit_axis, exit_side
    seg_ray = np.concatenate([s[0] for s in steps])
    seg_cell = np.concatenate([s[1] for s in steps])
    seg_len = np.concatenate([s[2] for s in steps])
    return seg_ray, seg_cell, seg_len, exit_t, exit_axis, exit_side


def walk_octree(starts, directions, oct, budgets, periodic_box=None):
    """
    Exact DDA segment walk through the octree: structurally identical to
    rt3d.octree.march_octree (same face-distance formulas, same eps
    re-locate offset, same periodic wrap, same max-step estimate), so the
    segment records match that march element-for-element (asserted in
    tests/test_mc_tracking.py).

    Parameters
    ----------
    starts : (K, 3) ndarray
    directions : (3,) or (K, 3) ndarray
    oct : rt3d.octree.octree_grid
    budgets : (K,) or scalar
        Path-length budget per ray (choose it to end at the domain
        entry/exit face, as precompute_octree_march does).
    periodic_box : rt3d.geometry.cuboid, optional

    Returns
    -------
    seg_ray, seg_leaf, seg_len : ndarray
        Step-major segment records.
    """
    o = np.atleast_2d(np.asarray(starts, dtype=float))
    K = o.shape[0]
    if K == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty.copy(), np.zeros(0)
    d = np.asarray(directions, dtype=float)
    if d.ndim == 1:
        d = np.tile(d, (K, 1))
    budget = np.broadcast_to(np.asarray(budgets, dtype=float), (K,)).astype(float)

    p = o.copy()
    remaining = budget.copy()
    node = oct.locate(p)

    h_min = oct.half.min() * 2.0
    planes_per_length = (np.abs(d) / h_min).sum(axis=1)
    max_steps = int(np.ceil(budget.max() * planes_per_length.max())) + 8

    seg_ray = []
    seg_leaf = []
    seg_len = []
    eps = 1e-9 * h_min

    for _ in range(max_steps):
        act = np.flatnonzero(remaining > 1e-12)
        if act.size == 0:
            break
        pa = p[act]
        na = node[act]
        h = oct.half[na]
        c = oct.centers[na].copy()
        if periodic_box is not None:
            c[:, 0] += periodic_box.Lx * np.round((pa[:, 0] - c[:, 0]) / periodic_box.Lx)
            c[:, 1] += periodic_box.Ly * np.round((pa[:, 1] - c[:, 1]) / periodic_box.Ly)

        t_face = np.full(act.size, np.inf)
        for axis in range(3):
            da = d[act, axis]
            pos = da > 0.0
            neg = da < 0.0
            if not (pos.any() or neg.any()):
                continue
            tf = np.empty(act.size)
            tf[pos] = (c[pos, axis] + h[pos] - pa[pos, axis]) / da[pos]
            tf[neg] = (c[neg, axis] - h[neg] - pa[neg, axis]) / da[neg]
            t_face = np.minimum(t_face, tf)

        step = np.minimum(t_face, remaining[act])
        step = np.maximum(step, 0.0)

        keep = step > 1e-12
        if np.any(keep):
            seg_ray.append(act[keep])
            seg_leaf.append(na[keep])
            seg_len.append(step[keep])

        pa = pa + step[:, None] * d[act]
        if periodic_box is not None:
            pa = periodic_box.wrap_xy(pa)
        p[act] = pa
        remaining[act] -= step

        move = remaining[act] > 1e-12
        if np.any(move):
            mact = act[move]
            pp = p[mact] + eps * d[mact]
            if periodic_box is not None:
                pp = periodic_box.wrap_xy(pp)
            node[mact] = oct.locate(pp)

    if len(seg_ray) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0)
    return (
        np.concatenate(seg_ray),
        np.concatenate(seg_leaf),
        np.concatenate(seg_len),
    )


# =============================================================================
# batched Woodcock flight sampler — the shared core of every MC solver
# =============================================================================


def sample_collisions(starts, directions, budgets, grid, kappa_e, kappa_s,
                      mode="cell", kappa_max=None, periodic_box=None, seed=0,
                      jit=False):
    """
    Sample the first **real** collision (scatter or true absorption) of
    `K` photons flying Woodcock null-collision flights through the grid.

    Both majorant modes run through the same loop and the same event
    decision; the only difference is the majorant used to sample a flight:

    - ``mode="cell"``: on entry to a cell, ``kappa_max = kappa_e(cell)`` —
      nulls have exactly zero probability on the piecewise-constant cells,
      a fresh exponential is sampled in every cell.
    - ``mode="global"``: ``kappa_max = kappa_max`` argument or
      ``max(kappa_e)`` — the flight exponential spans cell boundaries; a
      flight that ends in a lower-opacity cell produces a **null
      collision** (photon keeps flying, fresh exponential, direction and
      weight unchanged).

    Scattering chains (regenerating diffuse photons) are *not* part of
    this primitive — it returns the first real event per photon, which is
    everything the M0 anchors (transmittance, cell-vs-global equivalence)
    need; the stage-1 solvers (M1) loop over it.

    ``jit=True`` routes to the numba per-photon kernel
    (`rt3d.mc.jitmc.sample_collisions_jit`) when usable — same physics and
    return dict, numba's own random stream (statistically equivalent,
    plan M4); silently falls back to the NumPy path when numba is
    unavailable.

    Parameters
    ----------
    starts : (K, 3) ndarray
        Photon start positions inside the domain.
    directions : (3,) or (K, 3) ndarray
        Photon directions.  (Scatter direction resampling is the
        caller's job; see rt3d.mc.sources.hg_directions.)
    budgets : (K,) or scalar
        Path-length budget per photon (e.g. Lz/|Omega_z| for a top-seeded
        beam photon in a periodic box).
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
    kappa_e, kappa_s : (n_cells / n_leaf,) ndarray
        Cell/leaf fields (kappa_s may be all-zero for the pure
        absorption / transmittance anchor).
    mode : {"cell", "global"}
    kappa_max : float, optional
        Global-mode majorant cap (default ``max(kappa_e)``); ignored in
        cell mode.
    periodic_box : rt3d.geometry.cuboid, optional
        Periodic x/y wrap (same convention as the SC marches).
    seed : int
        PCG64 seed; recorded in every MC output (plan section 5).

    Returns
    -------
    dict with keys
        'coll_distance' : (K,) float
            Path length to the first real collision; ``inf`` if the
            photon exited on budget first.
        'coll_event' : (K,) int
            EV_SCATTER / EV_ABSORB at the collision, EV_EXIT otherwise.
        'coll_cell' : (K,) int
            Cell/leaf index of the collision (-1 if exited).
        'exit_t' : (K,) float
            Path length travelled when the photon left the domain or its
            budget (equal to coll_distance for collided photons).
        'n_null' : (K,) int
            Number of null collisions consumed per photon — zero in cell
            mode by construction, the load-bearing diagnostic of the
            global majorant mode.
        'survived' : (K,) bool
            True where the photon exited without a real collision — the
            transmittance tally of anchor 3.
    """
    if jit:
        from rt3d.mc import jitmc

        if jitmc.jit_mc_usable(grid, periodic_box):
            return jitmc.sample_collisions_jit(
                starts, directions, budgets, grid, kappa_e, kappa_s,
                mode=mode, kappa_max=kappa_max, periodic_box=periodic_box,
                seed=seed)

    if mode not in ("cell", "global"):
        raise ValueError(f"mode must be 'cell' or 'global', got {mode!r}")
    octree = isinstance(grid, octree_grid)

    kappa_e = np.asarray(kappa_e, dtype=float)
    kappa_s = np.asarray(kappa_s, dtype=float)
    n_cells = grid.n_leaf if octree else grid.n_cells
    if kappa_e.shape != (n_cells,) or kappa_s.shape != (n_cells,):
        raise ValueError(f"kappa fields must have shape ({n_cells},)")
    if np.any(kappa_e < 0.0) or np.any(kappa_s < 0.0) or np.any(kappa_s > kappa_e):
        raise ValueError("need 0 <= kappa_s <= kappa_e in every cell")

    o = np.atleast_2d(np.asarray(starts, dtype=float))
    K = o.shape[0]
    d = np.asarray(directions, dtype=float)
    if d.ndim == 1:
        d = np.tile(d, (K, 1))
    budget = np.broadcast_to(np.asarray(budgets, dtype=float), (K,)).astype(float).copy()

    kmax_global = float(np.max(kappa_e)) if kappa_max is None else float(kappa_max)
    if kmax_global <= 0.0:
        raise ValueError("kappa_max must be positive")

    box = grid.box
    h_min = (grid.half.min() * 2.0) if octree else min(grid.hx, grid.hy, grid.hz)
    # re-locate offset: must dominate the +1e-9 floor epsilon of
    # unigrid.locate (otherwise the probe snaps back to the cell just
    # left and the cell state lags one cell behind the position), but stay
    # far below one cell so it can never skip a cell — the face-normal
    # advance is >= 1e-6 * h_min * min|Omega_axis| ~ 1e-8 * h_min
    eps = 1e-6 * h_min

    rng = np.random.default_rng(seed)

    p = o.copy()
    if periodic_box is not None:
        p = periodic_box.wrap_xy(p)
    cell = grid.locate(p)
    remaining = budget
    traveled = np.zeros(K)
    pending = np.full(K, np.inf)  # remaining length of the current flight

    coll_s = np.full(K, np.inf)
    coll_event = np.full(K, EV_EXIT, dtype=int)
    coll_cell = np.full(K, -1, dtype=int)
    n_null = np.zeros(K, dtype=int)  # null collisions consumed per photon
    exit_t = np.zeros(K)
    exit_axis = np.full(K, -1, dtype=int)
    exit_side = np.zeros(K, dtype=int)

    active = remaining > 1e-12
    max_steps = int(
        np.ceil(np.nanmax(budget) * (np.abs(d) / h_min).sum(axis=1).max())
    ) + 8 * K + 128  # flights + cell crossings + null events, generous

    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        # sample a fresh flight wherever none is pending (start of photon,
        # after a null, and — cell mode only — at every cell entry)
        need = ~np.isfinite(pending[idx])
        if np.any(need):
            i2 = idx[need]
            kmax = kappa_e[cell[i2]] if mode == "cell" else kmax_global
            pending[i2] = flight_distance(rng.random(i2.size), kmax)

        pa = p[idx]
        da = d[idx]
        if octree:
            t_face, fax, fside = _octree_face_times(pa, da, cell[idx], grid, periodic_box)
        else:
            t_face, fax, fside = _lattice_face_times(pa, da, grid)
        t_box, bax, bsd = ray_box_exit(pa, da, box)

        pend = pending[idx]
        step = np.minimum(np.minimum(np.minimum(pend, t_face), t_box), remaining[idx])
        step = np.maximum(step, 0.0)

        hit = (step == pend) & np.isfinite(step)
        p[idx] = pa + step[:, None] * da
        traveled[idx] += step
        remaining[idx] -= step
        pending[idx] -= step  # inf-safe: null / vacuum flights stay inf

        # --- flight terminated: event decision (identical in both modes)
        if np.any(hit):
            ih = idx[hit]
            c = cell[ih]
            kmax = kappa_e[c] if mode == "cell" else kmax_global
            ev = event_decision(rng.random(ih.size), kmax, kappa_s[c], kappa_e[c])
            null = ev == EV_NULL
            if np.any(null):
                # null collision: same position, same direction, the flight
                # simply continues with a fresh exponential
                pending[ih[null]] = np.inf
                n_null[ih[null]] += 1
            real = ~null
            if np.any(real):
                ir = ih[real]
                coll_s[ir] = traveled[ir]
                coll_event[ir] = ev[real]
                coll_cell[ir] = c[real]
                exit_t[ir] = traveled[ir]
                exit_axis[ir] = -1
                active[ir] = False

        # --- crossed a cell or domain boundary (and the flight survived)
        moved = ~hit
        crossed = moved & ((step == t_face) | (step == t_box))
        if np.any(crossed):
            ic = idx[crossed]
            if periodic_box is not None:
                # wrap x/y with the phase-5 snap: a ray landing within
                # float noise on a periodic face must re-enter at the
                # *opposite* face, otherwise ray_box_exit keeps returning
                # t = 0 and the photon stalls at the face
                pw = periodic_box.wrap_xy(p[ic])
                dw = d[ic]
                for axis, (lo, hi) in enumerate(
                    ((periodic_box.x0, periodic_box.x1), (periodic_box.y0, periodic_box.y1))
                ):
                    if not (periodic_box.periodic_x if axis == 0 else periodic_box.periodic_y):
                        continue
                    tol = 1e-9 * (hi - lo)
                    at_lo = (dw[:, axis] < 0.0) & (pw[:, axis] <= lo + tol)
                    pw[at_lo, axis] = hi
                    at_hi = (dw[:, axis] > 0.0) & (pw[:, axis] >= hi - tol)
                    pw[at_hi, axis] = lo
                p[ic] = pw
            at_box = step[crossed] == t_box[crossed]
            periodic_ax = np.zeros(ic.size, dtype=bool)
            if periodic_box is not None:
                periodic_ax = ((bax[crossed] == 0) & periodic_box.periodic_x) | (
                    (bax[crossed] == 1) & periodic_box.periodic_y
                )
            term = at_box & ~periodic_ax
            fin_ax = np.where(term, bax[crossed], -1)
            fin_sd = np.where(term, bsd[crossed], 0)
            # safety net: the leaf/lattice plane can win the min by ulps
            # at a non-periodic domain face -> detect via position beyond
            # the box (periodic axes are already wrapped and never violate)
            tol = 1e-9 * h_min
            for axis, (lo, hi) in enumerate(box):
                b = p[ic, axis] < lo - tol
                a = p[ic, axis] > hi + tol
                new = (b | a) & ~term
                term |= b | a
                fin_ax[new] = axis
                fin_sd[new] = np.where(b[new], -1, 1)

            if np.any(~term):  # interior crossing: continue into the next cell
                cont = ic[~term]
                pp = p[cont] + eps * d[cont]
                if periodic_box is not None:
                    pp = periodic_box.wrap_xy(pp)
                cell[cont] = grid.locate(pp)
                if mode == "cell":
                    pending[cont] = np.inf
            if np.any(term):
                tq = ic[term]
                exit_axis[tq] = fin_ax[term]
                exit_side[tq] = fin_sd[term]
                active[tq] = False

        # --- any ray whose budget is exhausted (incl. exactly at a face) stops
        active[active & (remaining <= 1e-12)] = False

    # photons that never hit a real event: exit path length = full travelled
    # path (budget end or domain exit)
    exit_t[coll_event == EV_EXIT] = traveled[coll_event == EV_EXIT]

    return {
        "coll_distance": coll_s,
        "coll_event": coll_event,
        "coll_cell": coll_cell,
        "n_null": n_null,
        "exit_t": exit_t,
        "exit_axis": exit_axis,
        "exit_side": exit_side,
        "survived": coll_event == EV_EXIT,
        "mode": mode,
        "kappa_max": kmax_global if mode == "global" else None,
        "seed": int(seed),
        "n_histories": K,
    }


# =============================================================================
# flight geometry helpers
# =============================================================================


def _path_cap(path_cap, grid):
    """Default flight-length cap: 100 domain diagonals."""
    if path_cap is not None:
        return float(path_cap)
    return 100.0 * float(np.sqrt(sum((b[1] - b[0]) ** 2 for b in grid.box)))


def _exit_distance(p, d, grid, periodic_box=None):
    """
    Uncapped distance to the domain exit: with a periodic x/y box the exit is
    a z face (``(z_face - p_z)/Omega_z``, ``inf`` for ``Omega_z = 0``),
    otherwise the ordinary box exit.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    d = np.atleast_2d(np.asarray(d, dtype=float))
    if periodic_box is None:
        t, _, _ = ray_box_exit(p, d, grid.box)
        return t
    z0, z1 = grid.box[2]
    dz = d[:, 2]
    t = np.full(p.shape[0], np.inf)
    up = dz > 0.0
    dn = dz < 0.0
    t[up] = (z1 - p[up, 2]) / dz[up]
    t[dn] = (z0 - p[dn, 2]) / dz[dn]
    return t


def flight_budget(p, d, grid, periodic_box=None, path_cap=None):
    """
    Path-length budget of a flight from `p` along `d`: the distance to the
    domain exit, capped at `path_cap` (default 100 domain diagonals).

    The cap truncates the near-grazing outliers (``|Omega_z|`` tiny) that
    would otherwise make the marching loop spend an enormous number of empty
    steps in vacuum; `track_histories` reports how often it fires.  The
    result is always finite, as `sample_collisions` requires.
    """
    t = _exit_distance(p, d, grid, periodic_box=periodic_box)
    return np.minimum(t, _path_cap(path_cap, grid))


def flight_segments(starts, directions, ends, grid, periodic_box=None):
    """
    Segment records ``(history, cell, length)`` of straight flights truncated
    at the path lengths `ends` — the geometric path the flight sampler walked.
    """
    if isinstance(grid, octree_grid):
        return walk_octree(starts, directions, grid, ends,
                           periodic_box=periodic_box)
    sr, sc, sl, *_ = walk_lattice(starts, directions, grid, budgets=ends,
                                  periodic_box=periodic_box)
    return sr, sc, sl


# =============================================================================
# analog history chains — stage-1 transport of the explicit MC variant (M1)
# =============================================================================


def track_histories(starts, directions, grid, kappa_e, kappa_s, tally,
                    weights=None, g=0.0, mode="cell", kappa_max=None,
                    periodic_box=None, seed=0, scatter=True,
                    max_collisions=20000, path_cap=None, tally_direct=None,
                    is_direct=None, implicit_capture=False,
                    roulette_floor=None, jit=False):
    """
    Fly analog MC histories generation by generation with the M0 flight
    sampler, tallying every flight's track lengths as they are produced and
    resampling HG directions at real scattering events
    (plan/mc-plan.md section 3, stage1="explicit").

    The transport itself is `sample_collisions`, called once per generation on
    the still-live sub-population: it already handles both majorant modes
    (cell / global with **real null collisions**), the event decision
    (scatter / true absorption / null) and the periodic-wrap snap, so the
    chain adds only the generation loop, the HG resampling and the tally
    deposition.  With ``scatter=False`` the scattering coefficient is replaced
    by zero — every real collision removes the particle, which is the
    fixed-source transport operator of the lambda-iteration variant.

    Variance reduction (M4, plan section 2):

    ``implicit_capture=True``
        No history terminates at a true absorption.  Instead **every** real
        collision multiplies the history weight by the collision albedo
        ``omega = kappa_s / kappa_e`` and the chain continues like at a
        scattering (HG direction resampling).  The Woodcock density of real
        collisions times the albedo product reproduces the removal series
        exactly: the expected weight reaching path length s is
        ``omega^{N(s)}``-averaged, ``exp(-tau_a(s))`` — the same absorption
        attenuation the analog chain achieves by removal, with the score
        spread over many small weights instead of few 0/1 terminations.
        The `absorbed` output then marks only roulette/zero-weight kills.
    ``roulette_floor=f`` (implies implicit_capture chains terminating early)
        Russian roulette at every real collision: a history with ``w < f``
        survives with probability 1/2 (weight doubled) or is killed; zero
        weights are always killed.  Keeps the expected weight unchanged.
    ``jit=True``
        Route every generation's flight sampling through the numba kernel
        (`rt3d.mc.jitmc`) when usable — statistically equivalent, numba's
        own random stream (not the NumPy PCG64 stream).

    Parameters
    ----------
    starts : (K, 3) ndarray
        History start positions.
    directions : (3,) or (K, 3) ndarray
        History start directions.
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
    kappa_e, kappa_s : (n_cells,) ndarray
    tally : rt3d.mc.estimators.track_tally or compatible
        Receives ``add(history, cell, length, weight, direction)`` for every
        flight segment (diffuse part of the radiation field).
    weights : (K,) ndarray, optional
        Power carried per history (plan section 5 weight conventions);
        default 1.  Analog chains keep their weight — Russian roulette and
        implicit capture are the M4 variance-reduction step.
    g : float or (n_cells,) ndarray
        HG asymmetry parameter for the scattering resampling.
    mode, kappa_max, periodic_box, seed
        As in `sample_collisions`.
    scatter : bool
        False -> removal-only transport (no particle scattering).
    max_collisions : int
        Cap on the number of generations (chains of a purely scattering
        medium in a closed box can be long); histories still live at the cap
        are reported in 'capped'.
    path_cap : float, optional
        Flight-length cap passed to `flight_budget`.
    tally_direct : track_tally, optional
        Second tally for the **unscattered direct-beam** part of the
        histories.  Only meaningful together with `is_direct`.
    is_direct : (K,) ndarray of bool, optional
        Marks beam particles that have not scattered yet; their flights go to
        `tally_direct` instead of `tally`, which reproduces the
        direct/diffuse split of the SC solver (the direct beam is not part of
        the diffuse ``J`` there).  A particle leaves the direct population at
        its first scattering event.

    Returns
    -------
    dict with keys
        'n_collisions' : (K,) int   real collisions (scatter + absorption)
        'n_null'       : (K,) int   null collisions consumed
        'n_scatter'    : (K,) int   scattering events
        'absorbed'     : (K,) bool  terminated by true absorption
        'exited'       : (K,) bool  left the domain
        'capped'       : (K,) bool  still live at max_collisions
        'path'         : (K,) float total path length flown
        'n_generations', 'mode', 'seed', 'scatter', 'n_truncated'
    """
    o = np.atleast_2d(np.asarray(starts, dtype=float))
    K = o.shape[0]
    d = np.asarray(directions, dtype=float)
    if d.ndim == 1:
        d = np.tile(d, (K, 1))
    d = d / np.linalg.norm(d, axis=1)[:, None]

    kappa_e = np.asarray(kappa_e, dtype=float)
    kappa_s = np.asarray(kappa_s, dtype=float)
    ks = kappa_s if scatter else np.zeros_like(kappa_s)
    omega = np.where(kappa_e > 0.0, ks / np.maximum(kappa_e, 1e-300), 0.0)

    pos = o.copy()
    if periodic_box is not None:
        pos = periodic_box.wrap_xy(pos)

    w = np.ones(K) if weights is None else np.array(weights, dtype=float)
    direct = None if is_direct is None else np.asarray(is_direct, dtype=bool).copy()

    live = np.arange(K)
    n_coll = np.zeros(K, dtype=np.int64)
    n_null = np.zeros(K, dtype=np.int64)
    n_scat = np.zeros(K, dtype=np.int64)
    path = np.zeros(K)
    absorbed = np.zeros(K, dtype=bool)
    exited = np.zeros(K, dtype=bool)
    n_trunc = 0

    rng = np.random.default_rng(seed)
    gen = 0
    while live.size and gen < int(max_collisions):
        pl = pos[live]
        dl = d[live]
        raw = _exit_distance(pl, dl, grid, periodic_box=periodic_box)
        budget = np.minimum(raw, _path_cap(path_cap, grid))
        n_trunc += int(np.count_nonzero(raw > budget))

        res = sample_collisions(pl, dl, budget, grid, kappa_e, ks, mode=mode,
                                kappa_max=kappa_max, periodic_box=periodic_box,
                                seed=seed + 1_000_003 * gen + 1, jit=jit)
        seg_end = res["exit_t"]
        path[live] += seg_end

        # --- track lengths of this flight, deposited into the tallies
        sh, sc, sl = flight_segments(pl, dl, seg_end, grid,
                                     periodic_box=periodic_box)
        if sh.size:
            wseg = w[live[sh]]
            dseg = dl[sh]
            if direct is None or tally_direct is None:
                tally.add(live[sh], sc, sl, wseg, dseg)
            else:
                dm = direct[live[sh]]
                if np.any(~dm):
                    tally.add(live[sh][~dm], sc[~dm], sl[~dm], wseg[~dm], dseg[~dm])
                if np.any(dm):
                    tally_direct.add(live[sh][dm], sc[dm], sl[dm], wseg[dm], dseg[dm])

        # --- event bookkeeping
        ev = res["coll_event"]
        n_coll[live] += (ev != EV_EXIT).astype(np.int64)
        n_null[live] += res["n_null"]

        scat = ev == EV_SCATTER
        exit_ev = ev == EV_EXIT
        exited[live[exit_ev]] = True
        absorbed[live[~scat & ~exit_ev]] = True

        if implicit_capture:
            # the albedo product carries the removal: multiply at EVERY real
            # collision and keep all real-event histories alive; zero
            # weights (pure absorption cells) are killed immediately
            ir = ~exit_ev
            if np.any(ir):
                cr = res["coll_cell"][ir]
                w[live[ir]] *= omega[cr]
            cont = ir & (w[live] > 0.0)
            absorbed[live[~scat & ~exit_ev]] = False
            absorbed[live[ir & ~cont]] = True
        else:
            cont = scat

        if not np.any(cont):
            break   # every history in this generation was absorbed or exited

        isl = live[cont]
        coll_cell = res["coll_cell"][cont]
        pos[isl] = pl[cont] + res["coll_distance"][cont, None] * dl[cont]
        if periodic_box is not None:
            pos[isl] = periodic_box.wrap_xy(pos[isl])
        d[isl] = hg_scatter_directions(dl[cont], coll_cell, g, rng)
        n_scat[live[scat]] += 1
        if direct is not None:
            direct[isl] = False

        # --- Russian roulette on the surviving population
        if roulette_floor is not None and isl.size:
            low = np.flatnonzero(w[isl] < float(roulette_floor))
            if low.size:
                li = isl[low]
                surv = (w[li] > 0.0) & (rng.random(li.size) < 0.5)
                w[li[surv]] *= 2.0
                absorbed[li[~surv]] = True
                keep = np.ones(isl.size, dtype=bool)
                keep[low[~surv]] = False
                isl = isl[keep]

        live = isl
        gen += 1

    capped = np.zeros(K, dtype=bool)
    capped[live] = True

    return {
        "n_collisions": n_coll,
        "n_null": n_null,
        "n_scatter": n_scat,
        "absorbed": absorbed,
        "exited": exited,
        "capped": capped,
        "path": path,
        "n_generations": gen,
        "n_truncated": n_trunc,
        "mode": mode,
        "seed": int(seed),
        "scatter": bool(scatter),
        "implicit_capture": bool(implicit_capture),
        "roulette_floor": roulette_floor,
        "jit": bool(jit),
    }
