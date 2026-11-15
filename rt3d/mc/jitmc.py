# -*- coding: utf-8 -*-
"""
Numba fast path for the MC flight sampler (plan/mc-plan.md M4).

Scalar clone of `rt3d.mc.tracking.sample_collisions`: the Woodcock flight
sampling, the event decision (scatter / true absorption / null), the DDA
cell stepping on BOTH grids (unigrid lattice and octree leaf lattice with
periodic image centers), the periodic x/y wrap with the phase-5 face snap
and the path-budget bookkeeping — one compiled per-photon loop instead of
the vectorized `active`-mask loops of the NumPy reference.

Same `rt3d.jit` degradation pattern: without numba installed this module
still imports (`HAVE_NUMBA` False) and `jit_mc_usable` returns False, so
callers stay on the NumPy reference path (locked by tests).  The kernels
use numba's own seeded generator (`np.random.seed` inside the kernel):
streams differ from the NumPy PCG64 reference by construction, statistical
equivalence — same collision/event/exit distributions within 2 sigma — is
what the callers and tests rely on, not per-ray identity.
"""

import numpy as np

from rt3d.jit import HAVE_NUMBA, _njit

if HAVE_NUMBA:
    from rt3d.jit import _locate_scalar, _oct_arrays

__all__ = ["sample_collisions_jit", "jit_mc_usable"]

EV_EXIT = -1
EV_SCATTER = 0
EV_ABSORB = 1


if HAVE_NUMBA:

    @_njit
    def _unigrid_walk(starts, dirs, budget, ke, ks, cell_mode, kmax_global,
                      prm,
                      coll_s, coll_ev, coll_cell, exit_t, n_null,
                      exit_ax, exit_sd):
        """prm = (x0, x1, y0, y1, z0, z1, hx, hy, hz, nx, ny, nz,
        per_x, per_y, seed); flat cell order (k*ny + j)*nx + i."""
        x0 = prm[0]
        x1 = prm[1]
        y0 = prm[2]
        y1 = prm[3]
        z0 = prm[4]
        z1 = prm[5]
        hx = prm[6]
        hy = prm[7]
        hz = prm[8]
        nx = int(prm[9])
        ny = int(prm[10])
        nz = int(prm[11])
        per_x = prm[12] > 0.5
        per_y = prm[13] > 0.5
        np.random.seed(int(prm[14]))

        K = starts.shape[0]
        h_min = min(hx, hy, hz)
        eps_off = 1e-6 * h_min
        Lx = x1 - x0
        Ly = y1 - y0

        for r in range(K):
            px = starts[r, 0]
            py = starts[r, 1]
            pz = starts[r, 2]
            dx = dirs[r, 0]
            dy = dirs[r, 1]
            dz = dirs[r, 2]
            rem = budget[r]
            if per_x:
                px = x0 + np.mod(px - x0, Lx)
            if per_y:
                py = y0 + np.mod(py - y0, Ly)

            i = int(np.floor((px - x0) / hx + 1e-9))
            j = int(np.floor((py - y0) / hy + 1e-9))
            k = int(np.floor((pz - z0) / hz + 1e-9))
            if i < 0:
                i = 0
            elif i > nx - 1:
                i = nx - 1
            if j < 0:
                j = 0
            elif j > ny - 1:
                j = ny - 1
            if k < 0:
                k = 0
            elif k > nz - 1:
                k = nz - 1
            cell = (k * ny + j) * nx + i

            traveled = 0.0
            pend = np.inf
            rem0 = rem if np.isfinite(rem) else 1e12
            max_steps = int(np.ceil(
                rem0 * ((abs(dx) + abs(dy) + abs(dz)) / h_min
                        + (kmax_global if not cell_mode else 0.0)))) + 64

            for _ in range(max_steps):
                if rem <= 1e-12:
                    break
                if not np.isfinite(pend):
                    km = ke[cell] if cell_mode else kmax_global
                    if km > 0.0:
                        pend = -np.log1p(-np.random.random()) / km
                    else:
                        pend = np.inf

                # --- next lattice face (snap formulas of _lattice_face_times)
                t_face = np.inf
                if dx > 0.0:
                    e = 1e-9 * hx
                    base = x0 + np.floor((px - x0 + e) / hx) * hx
                    tf = (base + hx - px) / dx
                    if tf < t_face:
                        t_face = tf
                elif dx < 0.0:
                    e = 1e-9 * hx
                    base = x0 + np.floor((px - x0 + e) / hx) * hx
                    nxt = base if px > base + e else base - hx
                    tf = (nxt - px) / dx
                    if tf < t_face:
                        t_face = tf
                if dy > 0.0:
                    e = 1e-9 * hy
                    base = y0 + np.floor((py - y0 + e) / hy) * hy
                    tf = (base + hy - py) / dy
                    if tf < t_face:
                        t_face = tf
                elif dy < 0.0:
                    e = 1e-9 * hy
                    base = y0 + np.floor((py - y0 + e) / hy) * hy
                    nxt = base if py > base + e else base - hy
                    tf = (nxt - py) / dy
                    if tf < t_face:
                        t_face = tf
                if dz > 0.0:
                    e = 1e-9 * hz
                    base = z0 + np.floor((pz - z0 + e) / hz) * hz
                    tf = (base + hz - pz) / dz
                    if tf < t_face:
                        t_face = tf
                elif dz < 0.0:
                    e = 1e-9 * hz
                    base = z0 + np.floor((pz - z0 + e) / hz) * hz
                    nxt = base if pz > base + e else base - hz
                    tf = (nxt - pz) / dz
                    if tf < t_face:
                        t_face = tf

                # --- box exit (slab method)
                t_box = np.inf
                ax = -1
                sd = 0
                if dx > 0.0:
                    tb = (x1 - px) / dx
                    if tb < t_box:
                        t_box = tb
                        ax = 0
                        sd = 1
                elif dx < 0.0:
                    tb = (x0 - px) / dx
                    if tb < t_box:
                        t_box = tb
                        ax = 0
                        sd = -1
                if dy > 0.0:
                    tb = (y1 - py) / dy
                    if tb < t_box:
                        t_box = tb
                        ax = 1
                        sd = 1
                elif dy < 0.0:
                    tb = (y0 - py) / dy
                    if tb < t_box:
                        t_box = tb
                        ax = 1
                        sd = -1
                if dz > 0.0:
                    tb = (z1 - pz) / dz
                    if tb < t_box:
                        t_box = tb
                        ax = 2
                        sd = 1
                elif dz < 0.0:
                    tb = (z0 - pz) / dz
                    if tb < t_box:
                        t_box = tb
                        ax = 2
                        sd = -1

                # --- flight collision inside the current cell
                if np.isfinite(pend) and pend <= t_face and pend <= t_box \
                        and pend <= rem:
                    px = px + pend * dx
                    py = py + pend * dy
                    pz = pz + pend * dz
                    traveled += pend
                    rem -= pend
                    km = ke[cell] if cell_mode else kmax_global
                    rate = np.random.random() * km
                    if rate <= ks[cell]:
                        coll_ev[r] = EV_SCATTER
                    elif rate <= ke[cell]:
                        coll_ev[r] = EV_ABSORB
                    else:
                        n_null[r] += 1
                        pend = np.inf
                        continue
                    coll_s[r] = traveled
                    coll_cell[r] = cell
                    exit_t[r] = traveled
                    break

                # --- domain exit or periodic wrap at the box face
                if t_box <= t_face and t_box <= rem:
                    px = px + t_box * dx
                    py = py + t_box * dy
                    pz = pz + t_box * dz
                    traveled += t_box
                    rem -= t_box
                    if (ax == 0 and per_x) or (ax == 1 and per_y):
                        # wrap + snap, then continue as an interior crossing.
                        # The position stays on the (wrapped) face — the
                        # eps*d offset is a probe for the cell locate only,
                        # matching the NumPy sampler, so `traveled`
                        # accumulates the exact geometric exit distance.
                        if per_x:
                            px = x0 + np.mod(px - x0, Lx)
                            if px <= x0 + 1e-9 * Lx and dx < 0.0:
                                px = x1 - 1e-12 * Lx
                            elif px >= x1 - 1e-9 * Lx and dx > 0.0:
                                px = x0 + 1e-12 * Lx
                        if per_y:
                            py = y0 + np.mod(py - y0, Ly)
                            if py <= y0 + 1e-9 * Ly and dy < 0.0:
                                py = y1 - 1e-12 * Ly
                            elif py >= y1 - 1e-9 * Ly and dy > 0.0:
                                py = y0 + 1e-12 * Ly
                        qx = px + eps_off * dx
                        qy = py + eps_off * dy
                        qz = pz + eps_off * dz
                        if per_x:
                            qx = x0 + np.mod(qx - x0, Lx)
                        if per_y:
                            qy = y0 + np.mod(qy - y0, Ly)
                        i = int(np.floor((qx - x0) / hx + 1e-9))
                        j = int(np.floor((qy - y0) / hy + 1e-9))
                        k = int(np.floor((qz - z0) / hz + 1e-9))
                        if i < 0:
                            i = 0
                        elif i > nx - 1:
                            i = nx - 1
                        if j < 0:
                            j = 0
                        elif j > ny - 1:
                            j = ny - 1
                        if k < 0:
                            k = 0
                        elif k > nz - 1:
                            k = nz - 1
                        cell = (k * ny + j) * nx + i
                        if cell_mode:
                            pend = np.inf
                        else:
                            pend -= t_box
                        continue
                    exit_ax[r] = ax
                    exit_sd[r] = sd
                    break

                # --- interior lattice crossing
                if t_face <= rem:
                    px = px + t_face * dx
                    py = py + t_face * dy
                    pz = pz + t_face * dz
                    traveled += t_face
                    rem -= t_face
                    if per_x:
                        px = x0 + np.mod(px - x0, Lx)
                    if per_y:
                        py = y0 + np.mod(py - y0, Ly)
                    # position stays on the face plane; the eps probe only
                    # identifies the entered cell (NumPy sampler parity) so
                    # traveled accumulates the exact exit distance
                    qx = px + eps_off * dx
                    qy = py + eps_off * dy
                    qz = pz + eps_off * dz
                    if per_x:
                        qx = x0 + np.mod(qx - x0, Lx)
                    if per_y:
                        qy = y0 + np.mod(qy - y0, Ly)
                    i = int(np.floor((qx - x0) / hx + 1e-9))
                    j = int(np.floor((qy - y0) / hy + 1e-9))
                    k = int(np.floor((qz - z0) / hz + 1e-9))
                    if i < 0:
                        i = 0
                    elif i > nx - 1:
                        i = nx - 1
                    if j < 0:
                        j = 0
                    elif j > ny - 1:
                        j = ny - 1
                    if k < 0:
                        k = 0
                    elif k > nz - 1:
                        k = nz - 1
                    cell = (k * ny + j) * nx + i
                    if cell_mode:
                        pend = np.inf
                    else:
                        pend -= t_face
                    continue

                # --- path budget exhausted exactly at a face
                px = px + rem * dx
                py = py + rem * dy
                pz = pz + rem * dz
                traveled += rem
                rem = 0.0

            if coll_ev[r] == EV_EXIT and exit_t[r] == 0.0:
                exit_t[r] = traveled

    @_njit
    def _octree_walk(starts, dirs, budget, ke, ks, cell_mode, kmax_global,
                     prm, prm_i, centers_all, children_all, leaf_of_node,
                     leaf_centers, leaf_half,
                     coll_s, coll_ev, coll_cell, exit_t, n_null,
                     exit_ax, exit_sd, seed):
        np.random.seed(seed)
        x0 = prm[0]
        x1 = prm[1]
        y0 = prm[2]
        y1 = prm[3]
        z0 = prm[4]
        z1 = prm[5]
        Lx = prm[6]
        Ly = prm[7]
        h_min = prm[11]
        periodic = prm_i[4] > 0
        eps_off = 1e-6 * h_min

        K = starts.shape[0]
        for r in range(K):
            px = starts[r, 0]
            py = starts[r, 1]
            pz = starts[r, 2]
            dx = dirs[r, 0]
            dy = dirs[r, 1]
            dz = dirs[r, 2]
            rem = budget[r]
            leaf = _locate_scalar(px, py, pz, prm, prm_i,
                                  centers_all, children_all, leaf_of_node)

            traveled = 0.0
            pend = np.inf
            max_steps = int(np.ceil(
                (rem if np.isfinite(rem) else 1e30)
                * ((abs(dx) + abs(dy) + abs(dz)) / h_min
                   + (kmax_global if not cell_mode else 0.0)))) + 64

            for _ in range(max_steps):
                if rem <= 1e-12:
                    break
                if not np.isfinite(pend):
                    km = ke[leaf] if cell_mode else kmax_global
                    if km > 0.0:
                        pend = -np.log1p(-np.random.random()) / km
                    else:
                        pend = np.inf

                c0 = leaf_centers[leaf, 0]
                c1 = leaf_centers[leaf, 1]
                c2 = leaf_centers[leaf, 2]
                h = leaf_half[leaf]
                if periodic:
                    c0 += Lx * np.round((px - c0) / Lx)
                    c1 += Ly * np.round((py - c1) / Ly)

                t_face = np.inf
                if dx > 0.0:
                    t_face = (c0 + h - px) / dx
                elif dx < 0.0:
                    t_face = (c0 - h - px) / dx
                if dy > 0.0:
                    tf = (c1 + h - py) / dy
                    if tf < t_face:
                        t_face = tf
                elif dy < 0.0:
                    tf = (c1 - h - py) / dy
                    if tf < t_face:
                        t_face = tf
                if dz > 0.0:
                    tf = (c2 + h - pz) / dz
                    if tf < t_face:
                        t_face = tf
                elif dz < 0.0:
                    tf = (c2 - h - pz) / dz
                    if tf < t_face:
                        t_face = tf

                t_box = np.inf
                ax = -1
                sd = 0
                if dx > 0.0:
                    tb = (x1 - px) / dx
                    if tb < t_box:
                        t_box = tb
                        ax = 0
                        sd = 1
                elif dx < 0.0:
                    tb = (x0 - px) / dx
                    if tb < t_box:
                        t_box = tb
                        ax = 0
                        sd = -1
                if dy > 0.0:
                    tb = (y1 - py) / dy
                    if tb < t_box:
                        t_box = tb
                        ax = 1
                        sd = 1
                elif dy < 0.0:
                    tb = (y0 - py) / dy
                    if tb < t_box:
                        t_box = tb
                        ax = 1
                        sd = -1
                if dz > 0.0:
                    tb = (z1 - pz) / dz
                    if tb < t_box:
                        t_box = tb
                        ax = 2
                        sd = 1
                elif dz < 0.0:
                    tb = (z0 - pz) / dz
                    if tb < t_box:
                        t_box = tb
                        ax = 2
                        sd = -1

                if np.isfinite(pend) and pend <= t_face and pend <= t_box \
                        and pend <= rem:
                    px = px + pend * dx
                    py = py + pend * dy
                    pz = pz + pend * dz
                    traveled += pend
                    rem -= pend
                    km = ke[leaf] if cell_mode else kmax_global
                    rate = np.random.random() * km
                    if rate <= ks[leaf]:
                        coll_ev[r] = EV_SCATTER
                    elif rate <= ke[leaf]:
                        coll_ev[r] = EV_ABSORB
                    else:
                        n_null[r] += 1
                        pend = np.inf
                        continue
                    coll_s[r] = traveled
                    coll_cell[r] = leaf
                    exit_t[r] = traveled
                    break

                if t_box <= t_face and t_box <= rem:
                    px = px + t_box * dx
                    py = py + t_box * dy
                    pz = pz + t_box * dz
                    traveled += t_box
                    rem -= t_box
                    if periodic and (ax == 0 or ax == 1):
                        if ax == 0:
                            px = x0 + np.mod(px - x0, Lx)
                            if px <= x0 + 1e-9 * Lx and dx < 0.0:
                                px = x1 - 1e-12 * Lx
                            elif px >= x1 - 1e-9 * Lx and dx > 0.0:
                                px = x0 + 1e-12 * Lx
                        else:
                            py = y0 + np.mod(py - y0, Ly)
                            if py <= y0 + 1e-9 * Ly and dy < 0.0:
                                py = y1 - 1e-12 * Ly
                            elif py >= y1 - 1e-9 * Ly and dy > 0.0:
                                py = y0 + 1e-12 * Ly
                        # position stays on the face; eps probe -> leaf
                        qx = px + eps_off * dx
                        qy = py + eps_off * dy
                        qz = pz + eps_off * dz
                        if periodic:
                            qx = x0 + np.mod(qx - x0, Lx)
                            qy = y0 + np.mod(qy - y0, Ly)
                        leaf = _locate_scalar(qx, qy, qz, prm, prm_i,
                                              centers_all, children_all,
                                              leaf_of_node)
                        if cell_mode:
                            pend = np.inf
                        else:
                            pend -= t_box
                        continue
                    exit_ax[r] = ax
                    exit_sd[r] = sd
                    break

                if t_face <= rem:
                    px = px + t_face * dx
                    py = py + t_face * dy
                    pz = pz + t_face * dz
                    traveled += t_face
                    rem -= t_face
                    if periodic:
                        px = x0 + np.mod(px - x0, Lx)
                        py = y0 + np.mod(py - y0, Ly)
                    # eps probe for the leaf locate only (position stays put)
                    qx = px + eps_off * dx
                    qy = py + eps_off * dy
                    qz = pz + eps_off * dz
                    if periodic:
                        qx = x0 + np.mod(qx - x0, Lx)
                        qy = y0 + np.mod(qy - y0, Ly)
                    leaf = _locate_scalar(qx, qy, qz, prm, prm_i,
                                          centers_all, children_all,
                                          leaf_of_node)
                    if cell_mode:
                        pend = np.inf
                    else:
                        pend -= t_face
                    continue

                px = px + rem * dx
                py = py + rem * dy
                pz = pz + rem * dz
                traveled += rem
                rem = 0.0

            if coll_ev[r] == EV_EXIT and exit_t[r] == 0.0:
                exit_t[r] = traveled


def jit_mc_usable(grid, periodic_box=None):
    """
    True when the numba fast path can replace the NumPy sampler for this
    grid / periodic box (numba installed; octrees on their
    `rt3d.jit.jit_usable` conditions — both x and y wrapped; unigrids need
    finite spacings).
    """
    if not HAVE_NUMBA:
        return False
    if periodic_box is not None and not (
        periodic_box.periodic_x and periodic_box.periodic_y
    ):
        return False
    from rt3d.octree import octree_grid

    if isinstance(grid, octree_grid):
        return grid.n_leaf > 0
    return np.isfinite(grid.hx) and np.isfinite(grid.hy) \
        and np.isfinite(grid.hz)


def sample_collisions_jit(starts, directions, budgets, grid, kappa_e,
                          kappa_s, mode="cell", kappa_max=None,
                          periodic_box=None, seed=0):
    """
    Numba clone of `rt3d.mc.tracking.sample_collisions` — identical
    parameter set and return dict; statistically equivalent (NOT the same
    random stream; numba uses its own seeded generator, plan section 2).
    Raises when numba is unavailable — callers gate on `jit_mc_usable`.
    """
    if not HAVE_NUMBA:
        raise RuntimeError("numba not installed — use the NumPy sampler")
    if mode not in ("cell", "global"):
        raise ValueError(f"mode must be 'cell' or 'global', got {mode!r}")

    from rt3d.geometry import ray_box_exit  # noqa: F401 (parity docs)
    from rt3d.octree import octree_grid

    kappa_e = np.ascontiguousarray(kappa_e, dtype=float)
    kappa_s = np.ascontiguousarray(kappa_s, dtype=float)
    octree = isinstance(grid, octree_grid)
    n_cells = grid.n_leaf if octree else grid.n_cells
    if kappa_e.shape != (n_cells,) or kappa_s.shape != (n_cells,):
        raise ValueError(f"kappa fields must have shape ({n_cells},)")
    if np.any(kappa_e < 0.0) or np.any(kappa_s < 0.0) or np.any(kappa_s > kappa_e):
        raise ValueError("need 0 <= kappa_s <= kappa_e in every cell")

    o = np.ascontiguousarray(np.atleast_2d(np.asarray(starts, dtype=float)))
    K = o.shape[0]
    d = np.asarray(directions, dtype=float)
    if d.ndim == 1:
        d = np.tile(d, (K, 1))
    d = np.ascontiguousarray(d)
    bud = np.ascontiguousarray(
        np.broadcast_to(np.asarray(budgets, dtype=float), (K,)).astype(float))

    kmax_global = float(np.max(kappa_e)) if kappa_max is None else float(kappa_max)
    if kmax_global <= 0.0:
        raise ValueError("kappa_max must be positive")

    coll_s = np.full(K, np.inf)
    coll_ev = np.full(K, EV_EXIT, dtype=np.int64)
    coll_cell = np.full(K, -1, dtype=np.int64)
    n_null = np.zeros(K, dtype=np.int64)
    exit_t = np.zeros(K)
    exit_ax = np.full(K, -1, dtype=np.int64)
    exit_sd = np.zeros(K, dtype=np.int64)

    if octree:
        prm, prm_i, ca, _ha, ch, ln, lc, lh = _oct_arrays(grid, periodic_box)
        _octree_walk(
            o, d, bud, kappa_e, kappa_s, mode == "cell", kmax_global,
            np.ascontiguousarray(prm, dtype=np.float64),
            np.ascontiguousarray(prm_i, dtype=np.int64),
            np.ascontiguousarray(ca), np.ascontiguousarray(ch),
            np.ascontiguousarray(ln), np.ascontiguousarray(lc),
            np.ascontiguousarray(lh),
            coll_s, coll_ev, coll_cell, exit_t, n_null,
            exit_ax, exit_sd, int(seed))
    else:
        bx = grid.box
        prm = np.array([bx[0][0], bx[0][1], bx[1][0], bx[1][1],
                        bx[2][0], bx[2][1], grid.hx, grid.hy, grid.hz,
                        grid.nx, grid.ny, grid.nz,
                        1.0 if periodic_box is not None else 0.0,
                        1.0 if periodic_box is not None else 0.0,
                        float(seed)], dtype=float)
        # NOTE: periodic_box here means BOTH x and y wrap (jit_mc_usable
        # gate); a box with only one periodic axis never reaches this path.
        _unigrid_walk(
            o, d, bud, kappa_e, kappa_s, mode == "cell", kmax_global, prm,
            coll_s, coll_ev, coll_cell, exit_t, n_null,
            exit_ax, exit_sd)

    return {
        "coll_distance": coll_s,
        "coll_event": coll_ev,
        "coll_cell": coll_cell,
        "exit_t": exit_t,
        "n_null": n_null,
        "survived": coll_ev == EV_EXIT,
        "exit_axis": exit_ax,
        "exit_side": exit_sd,
        "mode": mode,
        "kappa_max": kmax_global if mode == "global" else None,
        "seed": int(seed),
        "n_histories": K,
    }
