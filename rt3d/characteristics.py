# -*- coding: utf-8 -*-
"""
Predecessor chains for the short-characteristics sweep (plan section 4).

Per ordinate Omega and target cell c, the upwind cell u is the cell whose
center-to-center direction u_center -> c_center is closest to Omega
(bipolar / Rybicki-Hummer connection). On the uniform Cartesian grid this
means one integer lattice stencil s per ordinate: u = c - s, and the
sub-segments of the straight path u_center -> c_center are the *same*
(relative offsets and lengths) for every cell, which makes the transport
recursion a handful of vectorized gathers per BFS wave.

The stencil always has s_z != 0: no ordinate has Omega_z = 0 (guaranteed by
the direction-grid rotation), and z is the only non-periodic dimension, so
every chain terminates at the top or bottom boundary and the dependency
graph is a forest rooted at boundary-entry cells.
"""

import numpy as np

from rt3d.geometry import march_lattice, ray_box_exit


def stencil_for_direction(omega, h, max_offset=None):
    """
    Integer lattice stencil for one ordinate: the offset s with |s_i| <=
    max_offset, s != 0, s_z != 0, whose lattice direction s*h is closest to
    omega.

    max_offset=None picks the radius automatically: resolving a direction
    Omega on an anisotropic lattice requires offsets up to ~3 * max(h)/min(h)
    per axis (e.g. hz = hx/4 needs s_z ~ 12 for a ray with dz/dx ~ 3), so
    the radius scales with the grid anisotropy. A too-small radius leaves
    large, n_refine-independent stencil errors in the characteristic
    directions.

    Parameters
    ----------
    omega : (3,) ndarray
        Unit ordinate direction.
    h : (3,) array_like
        Cell sizes (hx, hy, hz).
    max_offset : int, optional
        Search radius in cells per axis; None -> automatic (see above).

    Returns
    -------
    s : (3,) ndarray of int
    vhat : (3,) ndarray
        Unit vector along the actual characteristic direction s*h.
    """
    omega = np.asarray(omega, dtype=float)
    h = np.asarray(h, dtype=float)

    if max_offset is None:
        max_offset = int(min(24, max(3, np.ceil(3.0 * h.max() / h.min()))))

    # vectorized search over the (2r+1)^3 candidate box (the pure-Python
    # loop cost minutes at r ~ 60 on strongly anisotropic lattices)
    vals = np.arange(-max_offset, max_offset + 1, dtype=np.int64)
    A, B, C = np.meshgrid(vals, vals, vals, indexing="ij")
    s_all = np.column_stack([A.ravel(), B.ravel(), C.ravel()])
    s_all = s_all[s_all[:, 2] != 0]  # s_z != 0 (and hence s != 0)
    if s_all.shape[0] == 0:
        raise ValueError(f"no stencil found for omega={omega} within max_offset={max_offset}")

    v = s_all.astype(float) * h
    norm = np.sqrt(np.einsum("ij,ij->i", v, v))
    cosv = (v @ omega) / norm
    n2 = np.einsum("ij,ij->i", s_all, s_all)

    # same preference as the reference loop: best cosine (within the 1e-12
    # tolerance band), ties toward the shorter characteristic; exact ties
    # keep the itertools.product order (stable)
    band = np.flatnonzero(cosv >= cosv.max() - 1e-12)
    pick = band[np.argmin(n2[band])]
    best_s = s_all[pick]

    v = best_s * h
    return best_s, v / np.linalg.norm(v)


def subsegments_uniform(s, h):
    """
    Exact sub-segments of the straight path from the center of cell -s to the
    center of cell 0 on a lattice with cell sizes h.

    The traversal runs in lattice-normalized coordinates q = p / h, where the
    path is -s -> 0 along s/|s|; physical segment lengths are scaled by
    |s*h| / |s|.

    Returns
    -------
    rel : (M, 3) ndarray of int
        Relative cell offsets traversed, in parent -> target order.
    lengths : (M,) ndarray
        Segment lengths (physical units).
    """
    s = np.asarray(s, dtype=int)
    h = np.asarray(h, dtype=float)

    u = s / np.linalg.norm(s)  # direction in q-space
    scale = np.linalg.norm(s * h) / np.linalg.norm(s)  # physical length per q-unit

    p = -s.astype(float)
    t_end = float(np.linalg.norm(s))

    rel = []
    lengths = []
    t_acc = 0.0
    while t_end - t_acc > 1e-12:
        t_next = np.inf
        for axis in range(3):
            ua = u[axis]
            if ua == 0.0:
                continue
            # cell faces in q-space are at half-integers: the next face
            # strictly above p (ua > 0) or strictly below p (ua < 0)
            if ua > 0:
                face = np.floor(p[axis] + 0.5) + 0.5
            else:
                face = np.ceil(p[axis] - 0.5) - 0.5
            t_face = (face - p[axis]) / ua
            if t_face > 1e-12:
                t_next = min(t_next, t_face)
        t = min(t_next, t_end - t_acc)
        mid = p + 0.5 * t * u
        rel.append(np.round(mid).astype(int))
        lengths.append(t)
        p = p + t * u
        t_acc += t

    return np.array(rel, dtype=int), np.array(lengths, dtype=float) * scale


def build_chain(grid, omega, periodic_box=None, max_offset=None):
    """
    Predecessor chain for one ordinate on the unigrid.

    Parameters
    ----------
    grid : rt3d.geometry.unigrid
    omega : (3,) ndarray
        Unit ordinate direction (propagation direction of the intensity).
    periodic_box : rt3d.geometry.cuboid, optional
        If given, x/y parent indices wrap and chains terminate only at the
        z-boundaries.
    max_offset : int, optional
        Stencil search radius; None -> automatic from the grid anisotropy
        (see stencil_for_direction).

    Returns
    -------
    chain : dict with keys
        's'          (3,) int stencil
        'vhat'       (3,) characteristic unit direction
        'parent'     (n_cells,) int, flat index of the upwind cell, -1 for roots
        'roots'      (R,) int flat indices of boundary-entry cells
        'waves'      list of (n_w,) int arrays, topological order; every
                     non-root cell appears in a wave strictly after its parent
        'sub_cells'  removed in P4: see 'rel' / 'sub_delta' below
        'rel'        (M, 3) int16 relative sub-segment cell offsets
                     (parent -> target order); absolute cell indices are
                     formed in the sweep from the wave cell indices
                     (plan/optimization-plan.md P4: the old (M, n_cells)
                     int64 table cost 228 MB @50^3, ~1.5 GB @100^3)
        'sub_delta'  (M,) int64 flat-index delta rel_i + nx*rel_j +
                     nx*ny*rel_k (non-periodic: cell = wave + delta)
        'periodic'   bool, x/y-wrapped chain family
        'sub_len'    (M,) segment lengths
        'root_steps' (ridx, cell, length) arrays, step-major march segments
                     from each root center backwards to the boundary entry
        'root_exit'  (R,) pair arrays (axis, side) of the boundary entry face
    """
    h = np.array([grid.hx, grid.hy, grid.hz])
    s, vhat = stencil_for_direction(omega, h, max_offset=max_offset)
    rel, lengths = subsegments_uniform(s, h)

    n = grid.n_cells
    idx = np.arange(n)
    i, j, k = grid.unravel(idx)

    pi, pj, pk = i - s[0], j - s[1], k - s[2]
    if periodic_box is not None:
        pi = np.mod(pi, grid.nx)
        pj = np.mod(pj, grid.ny)
        inside = (pk >= 0) & (pk < grid.nz)
    else:
        inside = (pi >= 0) & (pi < grid.nx) & (pj >= 0) & (pj < grid.ny) & (pk >= 0) & (pk < grid.nz)

    parent = np.where(inside, grid.ravel(pi, pj, pk), -1)
    roots = np.flatnonzero(parent < 0)

    # topological waves: non-periodic -> min over stencil axes of the
    # distance to the upwind boundary; periodic -> z-layer index (x,y wrap,
    # so only z strictly decreases along a chain)
    if periodic_box is not None:
        # chains advance monotonically in z (x,y wrap); depth counts
        # boundary-to-interior layers along the propagation direction
        depth = k if s[2] > 0 else (grid.nz - 1 - k)
    else:
        depth = np.full(n, np.inf)
        for axis, (sz, nn) in enumerate(zip(s, (grid.nx, grid.ny, grid.nz))):
            if sz == 0:
                continue
            coord = (i, j, k)[axis]
            d = coord if sz > 0 else (nn - 1 - coord)
            depth = np.minimum(depth, d)
        depth = depth.astype(int)

    max_depth = int(depth.max())
    waves = []
    for d in range(1, max_depth + 1):
        w = np.flatnonzero((depth == d) & (parent >= 0))
        if w.size:
            waves.append(w)

    # P4: the sub-segment cells of a non-root cell c are c + rel[m] (the
    # straight path stays inside the parent..target bounding box, so for
    # non-periodic chains the flat index is simply wave + sub_delta[m]);
    # periodic chains wrap x/y, resolved in the sweep from the wave coords
    rel = rel.astype(np.int16)
    nxny = grid.nx * grid.ny
    sub_delta = (
        rel[:, 0].astype(np.int64)
        + grid.nx * rel[:, 1].astype(np.int64)
        + nxny * rel[:, 2].astype(np.int64)
    )

    # boundary-entry chains: march from each root center backwards along
    # -vhat until the ray enters the domain (i.e. exits the box going back)
    root_steps, root_exit_t, rax, rsd = march_lattice(
        grid.centers[roots], -vhat, grid, periodic_box=periodic_box
    )
    rs_r = np.concatenate([s_[0] for s_ in root_steps]).astype(np.int64)
    rs_c = np.concatenate([s_[1] for s_ in root_steps]).astype(np.int64)
    rs_l = np.concatenate([s_[2] for s_ in root_steps]).astype(np.float64)

    return {
        "s": s,
        "vhat": vhat,
        "parent": parent,
        "roots": roots,
        "waves": waves,
        "rel": rel,
        "sub_delta": sub_delta,
        "periodic": periodic_box is not None,
        "sub_len": lengths,
        "root_steps": (rs_r, rs_c, rs_l),
        "root_exit_t": root_exit_t,
        "root_exit_axis": rax,
        "root_exit_side": rsd,
    }


# =============================================================================
# general predecessor chains on the octree (phase 3)
# =============================================================================


def build_chain_octree(oct, omega, periodic_box=None, cos_min=0.9,
                       max_hops=8):
    """
    Predecessor chain for one ordinate on the octree (plan section 4).

    For each leaf c the upwind leaf u is chosen like the unigrid lattice
    stencil, generalized to the graded grid: march backwards from c's
    center along -Omega over a limited hop budget and pick the traversed
    leaf whose center-to-center vector is best aligned with Omega.  If no
    candidate reaches cos_min alignment the leaf is rooted and carries an
    exact march to the domain entry point (the characteristic is then the
    true ray).  The characteristic of a non-root leaf is the straight line
    u_center -> c_center, split into sub-segments by an exact DDA march
    (bipolar / Rybicki-Hummer).  Dependency cycles, which the
    center-to-center connection can in principle form on a graded grid,
    are detected by topological peeling and resolved by rooting.

    Parameters
    ----------
    oct : rt3d.octree.octree_grid
    omega : (3,) ndarray
        Unit ordinate direction (Omega_z != 0 expected).
    periodic_box : rt3d.geometry.cuboid, optional
    cos_min : float, optional
        Minimum alignment cosine for a predecessor candidate, by default 0.9.
    max_hops : float, optional
        Candidate search budget in own-cell widths along -Omega, by default 8.

    Returns
    -------
    chain : dict with keys
        'parent'      (n_leaf,) int, upwind leaf or -1 for roots
        'roots'       (R,) int leaf indices of boundary-entry chains
        'waves'       list of arrays, topological order (parent before child)
        'root_seg'    (seg_ray, seg_leaf, seg_len, offsets) node-major,
                      ray index local to 'roots', segments center -> entry
        'root_exit_axis', 'root_exit_side'  (R,) entry face of each root
        'sub_seg'     (seg_ray, seg_leaf, seg_len, offsets) node-major,
                      ray index = leaf, segments parent_center -> leaf center
    """
    from rt3d.octree import march_octree, node_major_segments

    omega = np.asarray(omega, dtype=float)
    n = oct.n_leaf
    c = oct.centers
    h = oct.half

    # --- distance from every leaf center to the domain entry along -Omega
    z = c[:, 2]
    z0, z1 = oct.box[2]
    if periodic_box is not None:
        # x/y wrap: entry is always a z-face (Omega_z != 0)
        t_entry = np.where(
            omega[2] < 0.0, (z1 - z) / (-omega[2]), (z - z0) / omega[2]
        )
    else:
        t_entry, _, _ = ray_box_exit(c, -omega, oct.box)

    # --- candidate search: short backward march, best-aligned traversed leaf
    budget = np.minimum(t_entry, max_hops * 2.0 * h)
    sr, sl, sLen = march_octree(c, -omega, oct, budget, periodic_box)
    v = c[sr] - c[sl]
    if periodic_box is not None:  # minimum image across the periodic faces
        v[:, 0] -= periodic_box.Lx * np.round(v[:, 0] / periodic_box.Lx)
        v[:, 1] -= periodic_box.Ly * np.round(v[:, 1] / periodic_box.Ly)
    dist = np.linalg.norm(v, axis=1)
    cosv = np.where(dist > 1e-12, (v @ omega) / np.maximum(dist, 1e-12), -1.0)
    # last segment per ray after lexsort by (ray, cos, dist) = best candidate
    order = np.lexsort((dist, cosv, sr))
    sr_s, cos_s, leaf_s = sr[order], cosv[order], sl[order]
    last = np.r_[np.flatnonzero(sr_s[1:] != sr_s[:-1]), sr_s.size - 1]
    best_cos = np.full(n, -1.0)
    best_leaf = np.full(n, -1)
    best_cos[sr_s[last]] = cos_s[last]
    best_leaf[sr_s[last]] = leaf_s[last]
    parent = np.where(best_cos >= cos_min, best_leaf, -1)
    parent[parent == np.arange(n)] = -1

    # --- topological waves by peeling; stalled nodes (cycles) become roots
    processed = parent < 0
    remaining = ~processed
    waves = []
    while remaining.any():
        p_safe = np.where(parent >= 0, parent, 0)
        ready = remaining & (parent >= 0) & processed[p_safe]
        w = np.flatnonzero(ready)
        if w.size == 0:
            cyc = np.flatnonzero(remaining)
            parent[cyc] = -1
            processed[cyc] = True
            remaining[cyc] = False
            continue
        waves.append(w)
        processed[w] = True
        remaining[w] = False
    roots = np.flatnonzero(parent < 0)

    # --- root marches: center backwards to the domain entry point
    if periodic_box is not None:
        rax = np.full(roots.size, 2)
        rsd = np.where(omega[2] < 0.0, 1, -1)
    else:
        _, rax, rsd = ray_box_exit(c[roots], -omega, oct.box)
    sr2, sl2, sLen2 = march_octree(c[roots], -omega, oct, t_entry[roots],
                                   periodic_box)
    root_seg = node_major_segments(sr2, sl2, sLen2, roots.size)

    # --- sub-segments: exact march parent_center -> leaf center
    nonroot = np.flatnonzero(parent >= 0)
    if nonroot.size:
        start = c[parent[nonroot]]
        delta = c[nonroot] - start
        if periodic_box is not None:  # minimum image characteristic
            delta[:, 0] -= periodic_box.Lx * np.round(delta[:, 0] / periodic_box.Lx)
            delta[:, 1] -= periodic_box.Ly * np.round(delta[:, 1] / periodic_box.Ly)
        dist = np.linalg.norm(delta, axis=1)
        dirn = delta / dist[:, None]
        sr3, sl3, sLen3 = march_octree(start, dirn, oct, dist, periodic_box)
        sr3 = nonroot[sr3]
        sub_seg = node_major_segments(sr3, sl3, sLen3, n)
    else:
        e = np.zeros(0, dtype=int)
        sub_seg = (e, e.copy(), np.zeros(0), np.zeros(n + 1, dtype=int))

    return {
        "parent": parent,
        "roots": roots,
        "waves": waves,
        "root_seg": root_seg,
        "root_exit_axis": rax,
        "root_exit_side": rsd,
        "sub_seg": sub_seg,
    }
