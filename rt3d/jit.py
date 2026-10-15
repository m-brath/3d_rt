# -*- coding: utf-8 -*-
"""
numba-accelerated octree marching and formal-solution sweeps
(plan/optimization-plan.md, P1).

The NumPy DDA march (rt3d.octree.march_octree) re-gathers all active rays
every step and re-descends the full tree from the base lattice after every
face crossing; on deep octrees this costs seconds per ordinate and
materialises 26-37M segments per ordinate (~26 GB for 42 directions).

The kernels here run the same DDA per ray in compiled code and, for the
transport sweep, *fuse* the march with the piecewise-constant formal
solution: tau and source contributions are accumulated directly into the
per-leaf outputs, so the segment arrays are never materialised at all.
Rays are independent, so the loops run in parallel over rays (numba
prange).

All kernels reproduce the reference algorithm operation-for-operation
(same face-distance formulas, same eps offsets, same locate descent), so
segment lists are bit-comparable and fused sweeps agree with the NumPy
sweep to the last ulp of the transcendental functions.

Every entry point degrades gracefully: rt3d.jit.HAVE_NUMBA is False when
numba is not installed, and the callers in rt3d.transport fall back to the
reference NumPy implementations.
"""

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba missing -> reference paths
    HAVE_NUMBA = False

    def njit(*a, **k):
        def _wrap(f):
            return f

        return _wrap

    prange = range


# =============================================================================
# scalar kernels (plain compiled helpers)
# =============================================================================

if HAVE_NUMBA:
    _njit = njit(cache=True)
    _njit_par = njit(cache=True, parallel=True)
else:  # pragma: no cover
    def _njit(f):
        return f

    def _njit_par(f):
        return f


@_njit
def _locate_scalar(px, py, pz, prm, prm_i, centers_all, children_all,
                   leaf_of_node):
    """
    Leaf index containing (px, py, pz) — scalar clone of
    octree_grid.locate (clamp to the box, base-lattice index, descent).
    prm = (x0, x1, y0, y1, z0, z1, Lx, Ly, Lz, h0, eps, h_min),
    prm_i = (nx0, ny0, nz0, max_level, periodic).
    """
    x0, x1, y0, y1, z0, z1 = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5]
    h0 = prm[9]
    nx0, ny0, nz0, max_level = prm_i[0], prm_i[1], prm_i[2], prm_i[3]

    lo = 1e-12
    if px < x0 + lo:
        px = x0 + lo
    elif px > x1 - lo:
        px = x1 - lo
    if py < y0 + lo:
        py = y0 + lo
    elif py > y1 - lo:
        py = y1 - lo
    if pz < z0 + lo:
        pz = z0 + lo
    elif pz > z1 - lo:
        pz = z1 - lo

    i = int(np.floor((px - x0) / h0))
    j = int(np.floor((py - y0) / h0))
    k = int(np.floor((pz - z0) / h0))
    if i < 0:
        i = 0
    elif i > nx0 - 1:
        i = nx0 - 1
    if j < 0:
        j = 0
    elif j > ny0 - 1:
        j = ny0 - 1
    if k < 0:
        k = 0
    elif k > nz0 - 1:
        k = nz0 - 1

    node = k * nx0 * ny0 + j * nx0 + i
    for _ in range(max_level):
        if children_all[node, 0] < 0:
            break
        octant = (
            (1 if px > centers_all[node, 0] else 0)
            + 2 * (1 if py > centers_all[node, 1] else 0)
            + 4 * (1 if pz > centers_all[node, 2] else 0)
        )
        node = children_all[node, octant]
    return leaf_of_node[node]


@_njit
def _march_ray(r, origin, d0, d1, d2, budget, prm, prm_i,
               centers_all, half_all, children_all, leaf_of_node,
               leaf_centers, leaf_half,
               seg_leaf, seg_len, pos, fill, merge):
    """
    Per-ray DDA march, scalar clone of the rt3d.octree.march_octree loop
    (same face-distance formulas, same eps re-locate offset, same
    periodic wrap).  Segments are recorded in traversal order.

    fill=False: only count segments (returns the count, writes nothing).
    fill=True:  write segments at seg_leaf[pos:]/seg_len[pos:] and return
                the number written.
    merge=True: merge collinear consecutive segments that stay in the same
                leaf into one segment (no-op on a leaf-face DDA unless the
                re-locate returns the same leaf).
    """
    p0 = origin[r, 0]
    p1 = origin[r, 1]
    p2 = origin[r, 2]
    x0, x1, y0, y1, z0, z1 = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5]
    Lx, Ly = prm[6], prm[7]
    eps = prm[10]
    h_min = prm[11]
    periodic = prm_i[4]

    remaining = budget[r]
    leaf = _locate_scalar(p0, p1, p2, prm, prm_i,
                          centers_all, children_all, leaf_of_node)

    planes_per_length = (abs(d0) + abs(d1) + abs(d2)) / h_min
    max_steps = int(np.ceil(remaining * planes_per_length)) + 8

    n_seg = 0
    pend_leaf = -1
    pend_len = 0.0

    for _ in range(max_steps):
        if remaining <= 1e-12:
            break

        c0 = leaf_centers[leaf, 0]
        c1 = leaf_centers[leaf, 1]
        c2 = leaf_centers[leaf, 2]
        h = leaf_half[leaf]
        if periodic:
            c0 += Lx * np.round((p0 - c0) / Lx)
            c1 += Ly * np.round((p1 - c1) / Ly)

        t_face = np.inf
        if d0 > 0.0:
            t_face = (c0 + h - p0) / d0
        elif d0 < 0.0:
            t_face = (c0 - h - p0) / d0
        if d1 > 0.0:
            tf = (c1 + h - p1) / d1
            if tf < t_face:
                t_face = tf
        elif d1 < 0.0:
            tf = (c1 - h - p1) / d1
            if tf < t_face:
                t_face = tf
        if d2 > 0.0:
            tf = (c2 + h - p2) / d2
            if tf < t_face:
                t_face = tf
        elif d2 < 0.0:
            tf = (c2 - h - p2) / d2
            if tf < t_face:
                t_face = tf

        step = t_face if t_face < remaining else remaining
        if step < 0.0:
            step = 0.0

        if step > 1e-12:
            if merge:
                if pend_leaf == leaf:
                    pend_len += step
                else:
                    if pend_leaf >= 0:
                        if fill:
                            seg_leaf[pos] = pend_leaf
                            seg_len[pos] = pend_len
                            pos += 1
                        n_seg += 1
                    pend_leaf = leaf
                    pend_len = step
            else:
                if fill:
                    seg_leaf[pos] = leaf
                    seg_len[pos] = step
                    pos += 1
                n_seg += 1

        p0 += step * d0
        p1 += step * d1
        p2 += step * d2
        if periodic:
            p0 = x0 + np.mod(p0 - x0, Lx)
            p1 = y0 + np.mod(p1 - y0, Ly)
        remaining -= step

        if remaining > 1e-12:
            q0 = p0 + eps * d0
            q1 = p1 + eps * d1
            q2 = p2 + eps * d2
            if periodic:
                q0 = x0 + np.mod(q0 - x0, Lx)
                q1 = y0 + np.mod(q1 - y0, Ly)
            leaf = _locate_scalar(q0, q1, q2, prm, prm_i,
                                  centers_all, children_all, leaf_of_node)

    if merge and pend_leaf >= 0:
        if fill:
            seg_leaf[pos] = pend_leaf
            seg_len[pos] = pend_len
        n_seg += 1
    return n_seg


# =============================================================================
# public kernels
# =============================================================================


if HAVE_NUMBA:

    @_njit_par
    def _march_counts(origin, d0, d1, d2, budget, prm, prm_i,
                      centers_all, half_all, children_all, leaf_of_node,
                      leaf_centers, leaf_half, merge):
        n = origin.shape[0]
        counts = np.empty(n, dtype=np.int64)
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        for r in prange(n):
            counts[r] = _march_ray(r, origin, d0, d1, d2, budget, prm, prm_i,
                                   centers_all, half_all, children_all,
                                   leaf_of_node, leaf_centers, leaf_half,
                                   empty_i, empty_f, 0, False, merge)
        return counts

    @_njit_par
    def _march_fill(origin, d0, d1, d2, budget, prm, prm_i,
                    centers_all, half_all, children_all, leaf_of_node,
                    leaf_centers, leaf_half,
                    seg_leaf, seg_len, starts, merge):
        n = origin.shape[0]
        for r in prange(n):
            _march_ray(r, origin, d0, d1, d2, budget, prm, prm_i,
                       centers_all, half_all, children_all, leaf_of_node,
                       leaf_centers, leaf_half,
                       seg_leaf, seg_len, starts[r], True, merge)

    @_njit_par
    def _octree_exact_sweep(origin, d0, d1, d2, budget, prm, prm_i,
                            centers_all, half_all, children_all,
                            leaf_of_node, leaf_centers, leaf_half,
                            kappa_e, S, I_in):
        """
        Fused exact march + piecewise-constant formal solution, per ray.

        Segments are traversed center -> entry; the attenuation of a
        segment source to the ray's leaf center is exp(-prefix_before),
        exactly the recursion of
        transport.sweep_direction_octree_exact (same accumulation order).
        """
        n = origin.shape[0]
        I = np.empty(n, dtype=np.float64)
        for r in prange(n):
            p0 = origin[r, 0]
            p1 = origin[r, 1]
            p2 = origin[r, 2]
            x0, x1, y0, y1 = prm[0], prm[1], prm[2], prm[3]
            Lx, Ly = prm[6], prm[7]
            eps = prm[10]
            h_min = prm[11]
            periodic = prm_i[4]

            remaining = budget[r]
            leaf = _locate_scalar(p0, p1, p2, prm, prm_i,
                                  centers_all, children_all, leaf_of_node)
            planes_per_length = (abs(d0) + abs(d1) + abs(d2)) / h_min
            max_steps = int(np.ceil(remaining * planes_per_length)) + 8

            tau_tot = 0.0
            contrib = 0.0
            for _ in range(max_steps):
                if remaining <= 1e-12:
                    break
                c0 = leaf_centers[leaf, 0]
                c1 = leaf_centers[leaf, 1]
                c2 = leaf_centers[leaf, 2]
                h = leaf_half[leaf]
                if periodic:
                    c0 += Lx * np.round((p0 - c0) / Lx)
                    c1 += Ly * np.round((p1 - c1) / Ly)

                t_face = np.inf
                if d0 > 0.0:
                    t_face = (c0 + h - p0) / d0
                elif d0 < 0.0:
                    t_face = (c0 - h - p0) / d0
                if d1 > 0.0:
                    tf = (c1 + h - p1) / d1
                    if tf < t_face:
                        t_face = tf
                elif d1 < 0.0:
                    tf = (c1 - h - p1) / d1
                    if tf < t_face:
                        t_face = tf
                if d2 > 0.0:
                    tf = (c2 + h - p2) / d2
                    if tf < t_face:
                        t_face = tf
                elif d2 < 0.0:
                    tf = (c2 - h - p2) / d2
                    if tf < t_face:
                        t_face = tf

                step = t_face if t_face < remaining else remaining
                if step < 0.0:
                    step = 0.0

                if step > 1e-12:
                    tau = kappa_e[leaf] * step
                    contrib += S[leaf] * (-np.expm1(-tau)) * np.exp(-tau_tot)
                    tau_tot += tau

                p0 += step * d0
                p1 += step * d1
                p2 += step * d2
                if periodic:
                    p0 = x0 + np.mod(p0 - x0, Lx)
                    p1 = y0 + np.mod(p1 - y0, Ly)
                remaining -= step

                if remaining > 1e-12:
                    q0 = p0 + eps * d0
                    q1 = p1 + eps * d1
                    q2 = p2 + eps * d2
                    if periodic:
                        q0 = x0 + np.mod(q0 - x0, Lx)
                        q1 = y0 + np.mod(q1 - y0, Ly)
                    leaf = _locate_scalar(q0, q1, q2, prm, prm_i,
                                          centers_all, children_all,
                                          leaf_of_node)

            I[r] = contrib + I_in[r] * np.exp(-tau_tot)
        return I

    @_njit_par
    def _octree_beam_tau(origin, d0, d1, d2, budget, prm, prm_i,
                         centers_all, half_all, children_all,
                         leaf_of_node, leaf_centers, leaf_half,
                         kappa_e):
        """Fused exact march + beam optical depth accumulation per ray."""
        n = origin.shape[0]
        tau = np.empty(n, dtype=np.float64)
        for r in prange(n):
            p0 = origin[r, 0]
            p1 = origin[r, 1]
            p2 = origin[r, 2]
            x0, x1, y0, y1 = prm[0], prm[1], prm[2], prm[3]
            Lx, Ly = prm[6], prm[7]
            eps = prm[10]
            h_min = prm[11]
            periodic = prm_i[4]

            remaining = budget[r]
            leaf = _locate_scalar(p0, p1, p2, prm, prm_i,
                                  centers_all, children_all, leaf_of_node)
            planes_per_length = (abs(d0) + abs(d1) + abs(d2)) / h_min
            max_steps = int(np.ceil(remaining * planes_per_length)) + 8

            acc = 0.0
            for _ in range(max_steps):
                if remaining <= 1e-12:
                    break
                c0 = leaf_centers[leaf, 0]
                c1 = leaf_centers[leaf, 1]
                c2 = leaf_centers[leaf, 2]
                h = leaf_half[leaf]
                if periodic:
                    c0 += Lx * np.round((p0 - c0) / Lx)
                    c1 += Ly * np.round((p1 - c1) / Ly)

                t_face = np.inf
                if d0 > 0.0:
                    t_face = (c0 + h - p0) / d0
                elif d0 < 0.0:
                    t_face = (c0 - h - p0) / d0
                if d1 > 0.0:
                    tf = (c1 + h - p1) / d1
                    if tf < t_face:
                        t_face = tf
                elif d1 < 0.0:
                    tf = (c1 - h - p1) / d1
                    if tf < t_face:
                        t_face = tf
                if d2 > 0.0:
                    tf = (c2 + h - p2) / d2
                    if tf < t_face:
                        t_face = tf
                elif d2 < 0.0:
                    tf = (c2 - h - p2) / d2
                    if tf < t_face:
                        t_face = tf

                step = t_face if t_face < remaining else remaining
                if step < 0.0:
                    step = 0.0

                if step > 1e-12:
                    acc += kappa_e[leaf] * step

                p0 += step * d0
                p1 += step * d1
                p2 += step * d2
                if periodic:
                    p0 = x0 + np.mod(p0 - x0, Lx)
                    p1 = y0 + np.mod(p1 - y0, Ly)
                remaining -= step

                if remaining > 1e-12:
                    q0 = p0 + eps * d0
                    q1 = p1 + eps * d1
                    q2 = p2 + eps * d2
                    if periodic:
                        q0 = x0 + np.mod(q0 - x0, Lx)
                        q1 = y0 + np.mod(q1 - y0, Ly)
                    leaf = _locate_scalar(q0, q1, q2, prm, prm_i,
                                          centers_all, children_all,
                                          leaf_of_node)

            tau[r] = acc
        return tau

    @_njit_par
    def _locate_points(points, prm, prm_i, centers_all, children_all,
                       leaf_of_node):
        n = points.shape[0]
        out = np.empty(n, dtype=np.int64)
        for r in prange(n):
            out[r] = _locate_scalar(points[r, 0], points[r, 1], points[r, 2],
                                    prm, prm_i, centers_all, children_all,
                                    leaf_of_node)
        return out


# =============================================================================
# python-side wrappers
# =============================================================================


def _oct_arrays(oct, periodic_box=None):
    """(prm, prm_i, centers_all, half_all, children_all, leaf_of_node,
    leaf_centers, leaf_half).

    The periodic flag in prm_i is set per call (the cached octree arrays
    default to non-periodic); jit_usable guarantees both x and y wrap.
    """
    a = oct.march_arrays()
    prm_i = a["prm_i"]
    if periodic_box is not None:
        prm_i = prm_i.copy()
        prm_i[4] = 1
    return (a["prm"], prm_i, a["centers_all"], a["half_all"],
            a["children_all"], a["leaf_of_node"],
            a["centers"], a["half"])


def march_octree_jit(origin, direction, oct, path_len, periodic_box=None,
                     merge=False):
    """
    numba clone of rt3d.octree.march_octree (shared unit direction).

    Returns
    -------
    seg_ray, seg_leaf, seg_len : ndarray
        Ray-major (all segments of ray 0, then ray 1, ...); traversal
        order within each ray preserved.  Bit-comparable to the reference
        march with merge=False.
    """
    o = np.atleast_2d(np.asarray(origin, dtype=float))
    d = np.asarray(direction, dtype=float)
    if d.ndim != 1:
        raise ValueError("march_octree_jit needs a single shared direction")
    budget = np.ascontiguousarray(
        np.broadcast_to(np.asarray(path_len, dtype=float), (o.shape[0],))
    )
    o = np.ascontiguousarray(o)
    prm, prm_i, ca, ha, ch, ln, lc, lh = _oct_arrays(oct, periodic_box)

    counts = _march_counts(o, d[0], d[1], d[2], budget, prm, prm_i,
                           ca, ha, ch, ln, lc, lh, merge)
    offsets = np.zeros(o.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    m = int(offsets[-1])
    seg_leaf = np.empty(m, dtype=np.int64)
    seg_len = np.empty(m, dtype=np.float64)
    if m:
        _march_fill(o, d[0], d[1], d[2], budget, prm, prm_i,
                    ca, ha, ch, ln, lc, lh, seg_leaf, seg_len,
                    offsets[:-1], merge)
    seg_ray = np.repeat(np.arange(o.shape[0], dtype=np.int64), counts)
    return seg_ray, seg_leaf, seg_len


def sweep_direction_octree_exact_jit(oct, omega, budget, kappa_e, S, I_in,
                                     periodic_box=None):
    """
    Fused exact march + formal solution for one ordinate
    (rt3d.transport.sweep_direction_octree_exact equivalent, no segment
    arrays).  I_in : (n_leaf,) inflow intensity of each ray's entry face.
    """
    omega = np.asarray(omega, dtype=float)
    d = -omega  # backward march direction
    prm, prm_i, ca, ha, ch, ln, lc, lh = _oct_arrays(oct, periodic_box)
    return _octree_exact_sweep(
        np.ascontiguousarray(oct.centers), d[0], d[1], d[2],
        np.ascontiguousarray(budget), prm, prm_i, ca, ha, ch, ln, lc, lh,
        np.ascontiguousarray(kappa_e, dtype=np.float64),
        np.ascontiguousarray(S, dtype=np.float64),
        np.ascontiguousarray(I_in, dtype=np.float64),
    )


def beam_optical_depth_octree_jit(oct, omega_sun, kappa_e, budget,
                                  periodic_box=None):
    """Fused beam march (rt3d.transport.beam_optical_depth_octree)."""
    omega_sun = np.asarray(omega_sun, dtype=float)
    d = -omega_sun
    prm, prm_i, ca, ha, ch, ln, lc, lh = _oct_arrays(oct, periodic_box)
    return _octree_beam_tau(
        np.ascontiguousarray(oct.centers), d[0], d[1], d[2],
        np.ascontiguousarray(budget), prm, prm_i, ca, ha, ch, ln, lc, lh,
        np.ascontiguousarray(kappa_e, dtype=np.float64),
    )


def locate_jit(points, oct):
    """numba clone of octree_grid.locate (parallel over points)."""
    p = np.ascontiguousarray(np.atleast_2d(np.asarray(points, dtype=float)))
    prm, prm_i, ca, _ha, ch, ln, _lc, _lh = _oct_arrays(oct)
    return _locate_points(p, prm, prm_i, ca, ch, ln)


def jit_usable(oct, periodic_box=None, direction=None):
    """
    True when the numba path can replace the reference implementation for
    this octree / periodic box / direction (both x and y periodic, shared
    direction, omega_z != 0 not required here).
    """
    if not HAVE_NUMBA:
        return False
    if periodic_box is not None and not (
        periodic_box.periodic_x and periodic_box.periodic_y
    ):
        return False
    if direction is not None and np.asarray(direction).ndim != 1:
        return False
    return oct.n_leaf > 0


# =============================================================================
# unigrid chain / SC sweeps (P3 + P5): single-ordinate compiled kernels.
#
# The NumPy sweeps are gather-dominated and numpy's fancy indexing holds
# the GIL, so thread pools over the NumPy sweeps do not scale (measured:
# slower).  These kernels reproduce the NumPy recursion operation-for-
# operation; njit calls release the GIL, so sweep_all*(n_jobs > 1) runs
# them on a thread pool for true per-ordinate parallelism.
# =============================================================================


if HAVE_NUMBA:

    @_njit
    def _chain_sweep(kappa_e, S_d, I, parent, roots, rax, rsd,
                     rs_r, rs_c, rs_l,
                     waves_flat, wave_offs,
                     rel, sub_delta, sub_len, periodic,
                     nx, ny, nxny,
                     top_I, bottom_I_scalar, side_I,
                     bottom_I_arr, use_bottom):
        n = kappa_e.shape[0]
        for q in range(n):
            I[q] = 0.0

        # --- roots: entry -> center accumulation (step-major segments)
        R = roots.size
        tau_acc = np.zeros(R)
        src_acc = np.zeros(R)
        for s in range(rs_r.size):
            r = rs_r[s]
            cell = rs_c[s]
            tau = kappa_e[cell] * rs_l[s]
            src_acc[r] += S_d[cell] * (-np.expm1(-tau)) * np.exp(-tau_acc[r])
            tau_acc[r] += tau
        for r in range(R):
            ax = rax[r]
            sd = rsd[r]
            if ax == 2:
                if sd > 0:
                    I_in = top_I
                elif use_bottom:
                    I_in = bottom_I_arr[roots[r]]
                else:
                    I_in = bottom_I_scalar
            else:
                I_in = side_I
            I[roots[r]] = I_in * np.exp(-tau_acc[r]) + src_acc[r]

        # --- interior cells, BFS waves
        M = sub_len.size
        for w in range(wave_offs.size - 1):
            a = wave_offs[w]
            b = wave_offs[w + 1]
            nw = b - a
            if nw == 0:
                continue
            tau_tot = np.zeros(nw)
            contrib = np.zeros(nw)
            prefix = np.zeros(nw)
            if periodic:
                # decompose the wave cells once (div/mod are the costly
                # integer ops; the per-row offsets are pure adds)
                wi_a = np.empty(nw, dtype=np.int64)
                wj_a = np.empty(nw, dtype=np.int64)
                wk_a = np.empty(nw, dtype=np.int64)
                for q in range(nw):
                    wv = waves_flat[a + q]
                    wi_a[q] = wv % nx
                    wj_a[q] = (wv // nx) % ny
                    wk_a[q] = wv // nxny
            for m in range(M):
                r0 = rel[m, 0]
                r1 = rel[m, 1]
                r2 = rel[m, 2]
                sl = sub_len[m]
                if periodic:
                    for q in range(nw):
                        cell = ((wk_a[q] + r2) * nxny
                                + ((wj_a[q] + r1) % ny) * nx
                                + (wi_a[q] + r0) % nx)
                        tau_tot[q] += kappa_e[cell] * sl
                else:
                    dm = sub_delta[m]
                    for q in range(nw):
                        tau_tot[q] += kappa_e[waves_flat[a + q] + dm] * sl
            for m in range(M):
                r0 = rel[m, 0]
                r1 = rel[m, 1]
                r2 = rel[m, 2]
                sl = sub_len[m]
                if periodic:
                    for q in range(nw):
                        cell = ((wk_a[q] + r2) * nxny
                                + ((wj_a[q] + r1) % ny) * nx
                                + (wi_a[q] + r0) % nx)
                        tau = kappa_e[cell] * sl
                        prefix[q] += tau
                        contrib[q] += (S_d[cell] * (-np.expm1(-tau))
                                       * np.exp(-(tau_tot[q] - prefix[q])))
                else:
                    dm = sub_delta[m]
                    for q in range(nw):
                        cell = waves_flat[a + q] + dm
                        tau = kappa_e[cell] * sl
                        prefix[q] += tau
                        contrib[q] += (S_d[cell] * (-np.expm1(-tau))
                                       * np.exp(-(tau_tot[q] - prefix[q])))
            for q in range(nw):
                wv = waves_flat[a + q]
                I[wv] = I[parent[wv]] * np.exp(-tau_tot[q]) + contrib[q]
        return I

    @_njit
    def _sc_sweep(kappa_e, S_d, I, t_f, weights, order, slice_offs,
                  node_idx, node_face, wrap,
                  top_I, bottom_I_scalar, side_I,
                  bottom_I_arr, use_bottom, wrap_tol, max_passes):
        n = kappa_e.shape[0]
        for q in range(n):
            I[q] = 0.0
        E = np.empty(n)
        src = np.empty(n)
        for q in range(n):
            tau = kappa_e[q] * t_f
            E[q] = np.exp(-tau)
            src[q] = S_d[q] * (-np.expm1(-tau))

        n_sl = slice_offs.size - 1
        ls = slice_offs[n_sl - 1]
        le = slice_offs[n_sl]
        feed_prev = np.zeros(le - ls)
        first = True
        for _ in range(max_passes):
            for L in range(n_sl):
                s0 = slice_offs[L]
                s1 = slice_offs[L + 1]
                for q in range(s0, s1):
                    acc = 0.0
                    for m in range(4):
                        idx = node_idx[m, q]
                        face = node_face[m, q]
                        if face == 0:
                            vals = I[idx] * E[idx] + src[idx]
                        elif face == 1:
                            vals = top_I
                        elif face == 2:
                            if use_bottom:
                                vals = bottom_I_arr[idx]
                            else:
                                vals = bottom_I_scalar
                        else:
                            vals = side_I
                        acc += weights[m] * vals
                    cells = order[q]
                    I[cells] = acc * E[cells] + src[cells]
            if not wrap:
                break
            if first:
                first = False
            else:
                # ring convergence check on the wrap feed (last computed
                # layer), same criterion as sweep_direction_sc
                mx = 0.0
                sc_ = 0.0
                for q in range(le - ls):
                    cur = I[order[ls + q]]
                    dv = cur - feed_prev[q]
                    if dv < 0.0:
                        dv = -dv
                    if dv > mx:
                        mx = dv
                    fv = cur if cur >= 0.0 else -cur
                    if fv > sc_:
                        sc_ = fv
                if sc_ < 1e-30:
                    sc_ = 1e-30
                if mx <= wrap_tol * sc_:
                    break
            for q in range(le - ls):
                feed_prev[q] = I[order[ls + q]]
        return I


def _chain_wave_cache(chain):
    """Concatenated wave arrays for the kernel (cached on the chain)."""
    cached = chain.get("_wave_cache")
    if cached is not None:
        return cached
    waves = chain["waves"]
    if waves:
        flat = np.concatenate(waves).astype(np.int64)
    else:
        flat = np.zeros(0, dtype=np.int64)
    offs = np.zeros(len(waves) + 1, dtype=np.int64)
    np.cumsum([w.size for w in waves], out=offs[1:])
    chain["_wave_cache"] = (flat, offs)
    return chain["_wave_cache"]


def chain_sweep_jit(grid, chain, kappa_e, S_d, bc, bottom_I=None):
    """
    numba clone of rt3d.transport.sweep_direction (one ordinate).
    Same recursion and accumulation order; njit releases the GIL.
    """
    waves_flat, wave_offs = _chain_wave_cache(chain)
    rs_r, rs_c, rs_l = chain["root_steps"]
    st = chain.get("_jit_static")
    if st is None:
        st = (
            np.ascontiguousarray(chain["parent"], dtype=np.int64),
            np.ascontiguousarray(chain["roots"], dtype=np.int64),
            np.ascontiguousarray(chain["root_exit_axis"], dtype=np.int64),
            np.ascontiguousarray(chain["root_exit_side"], dtype=np.int64),
            np.ascontiguousarray(chain["rel"]),
            np.ascontiguousarray(chain["sub_delta"]),
            np.ascontiguousarray(chain["sub_len"], dtype=np.float64),
        )
        chain["_jit_static"] = st
    parent, roots, rax, rsd, rel, sub_delta, sub_len = st
    bI = np.ascontiguousarray(bottom_I) if bottom_I is not None \
        else np.zeros(0, dtype=np.float64)
    I = np.empty(grid.n_cells, dtype=np.float64)
    _chain_sweep(
        np.ascontiguousarray(kappa_e, dtype=np.float64),
        np.ascontiguousarray(S_d, dtype=np.float64),
        I,
        parent, roots, rax, rsd,
        rs_r, rs_c, rs_l,
        waves_flat, wave_offs,
        rel, sub_delta, sub_len,
        chain["periodic"],
        grid.nx, grid.ny, grid.nx * grid.ny,
        float(bc.top_I), float(bc.bottom_I), float(bc.side_I),
        bI, bottom_I is not None,
    )
    return I


def _sc_slice_cache(sc):
    cached = sc.get("_slice_cache")
    if cached is not None:
        return cached
    slices = sc["slices"]
    offs = np.empty(len(slices) + 1, dtype=np.int64)
    offs[0] = slices[0][0]
    for i, (_s0, s1) in enumerate(slices):
        offs[i + 1] = s1
    sc["_slice_cache"] = offs
    return offs


def sc_sweep_jit(grid, sc, kappa_e, S_d, bc, bottom_I=None,
                 wrap_tol=1e-12, max_passes=None):
    """
    numba clone of rt3d.short_characteristics.sweep_direction_sc
    (one ordinate).  Boundary faces are applied inline per stencil node
    (the NumPy per-layer np.any(face) / np.where passes disappear — P5).
    """
    if max_passes is None:
        max_passes = max(4 * sc["n_layers"], 64)
    st = sc.get("_jit_static")
    if st is None:
        st = (
            np.ascontiguousarray(sc["weights"], dtype=np.float64),
            np.ascontiguousarray(sc["order"], dtype=np.int64),
            np.ascontiguousarray(sc["node_idx"], dtype=np.int64),
            np.ascontiguousarray(sc["node_face"]),
        )
        sc["_jit_static"] = st
    weights, order, node_idx, node_face = st
    bI = np.ascontiguousarray(bottom_I) if bottom_I is not None \
        else np.zeros(0, dtype=np.float64)
    I = np.empty(grid.n_cells, dtype=np.float64)
    _sc_sweep(
        np.ascontiguousarray(kappa_e, dtype=np.float64),
        np.ascontiguousarray(S_d, dtype=np.float64),
        I,
        float(sc["t_f"]),
        weights,
        order,
        _sc_slice_cache(sc),
        node_idx,
        node_face,
        sc["wrap"],
        float(bc.top_I), float(bc.bottom_I), float(bc.side_I),
        bI, bottom_I is not None,
        float(wrap_tol), int(max_passes),
    )
    return I
