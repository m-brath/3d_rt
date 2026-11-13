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
                      mode="cell", kappa_max=None, periodic_box=None, seed=0):
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
