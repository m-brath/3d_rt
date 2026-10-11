# -*- coding: utf-8 -*-
"""
Flat struct-of-arrays octree (plan section 3).

Cubic cells, 2:1 face balance, max-depth cap. Refinement is driven by the
*input* extinction field:

    kappa_e * h > tau_max                          (optical depth per cell)
    (max(kappa_e) - min(kappa_e)) / mean(kappa_e) > eps   (cloud edges)

Point location is a vectorized descent from the base lattice (O(depth),
no per-level dense tables). The DDA march is vectorized over rays with a
per-ray path-length budget, mirroring geometry.march_lattice on the
unigrid.
"""

import numpy as np

from rt3d.geometry import ray_box_exit


class octree_grid(object):
    """
    Octree over an axis-aligned cuboid, built from cubic base cells of size
    h0 (h0 must divide all three side lengths).

    Attributes
    ----------
    cuboid : rt3d.geometry.cuboid
    box : tuple
        ((x0, x1), (y0, y1), (z0, z1)) — same form as unigrid.box.
    h0 : float
        Base cell size.
    n_leaf : int
        Number of leaves.
    centers : (n_leaf, 3) ndarray
        Leaf centers (leaf-index order).
    half : (n_leaf,) ndarray
        Leaf half-size.
    level : (n_leaf,) ndarray of int
        Octree level (base cells are level 0, cell size h0 / 2**level).
    kappa_e, kappa_s, g, T : (n_leaf,) ndarray
        Cell-centered field samples (filled by sample_fields).
    """

    def __init__(self, cuboid, h0):
        self.cuboid = cuboid
        self.box = (
            (cuboid.x0, cuboid.x1),
            (cuboid.y0, cuboid.y1),
            (cuboid.z0, cuboid.z1),
        )
        self.h0 = float(h0)
        self.nx0 = int(round(cuboid.Lx / h0))
        self.ny0 = int(round(cuboid.Ly / h0))
        self.nz0 = int(round(cuboid.Lz / h0))
        for name, n, L in (
            ("x", self.nx0, cuboid.Lx),
            ("y", self.ny0, cuboid.Ly),
            ("z", self.nz0, cuboid.Lz),
        ):
            if n < 1 or not np.isclose(n * h0, L):
                raise ValueError(f"h0 = {h0} must divide the {name} side length {L}")

        # node arrays (all nodes, leaves and internal)
        n_base = self.nx0 * self.ny0 * self.nz0
        bi, bj, bk = self._base_unravel(np.arange(n_base))
        self.centers_all = np.column_stack(
            [
                cuboid.x0 + (bi + 0.5) * h0,
                cuboid.y0 + (bj + 0.5) * h0,
                cuboid.z0 + (bk + 0.5) * h0,
            ]
        )
        self.half_all = np.full(n_base, h0 / 2.0)
        self.level_all = np.zeros(n_base, dtype=int)
        self.children_all = np.full((n_base, 8), -1, dtype=int)
        self.max_level = 0

        self._compact()

    def _base_unravel(self, idx):
        i = idx % self.nx0
        j = (idx // self.nx0) % self.ny0
        k = idx // (self.nx0 * self.ny0)
        return i, j, k

    def _base_id(self, i, j, k):
        return k * self.nx0 * self.ny0 + j * self.nx0 + i

    # ------------------------------------------------------------------ build

    def split(self, nodes):
        """Split the given leaf nodes into their 8 children."""
        nodes = np.asarray(nodes, dtype=int)
        nodes = nodes[self.children_all[nodes, 0] < 0]  # leaves only
        if nodes.size == 0:
            return
        n_new = nodes.size * 8
        idx = np.arange(n_new).reshape(-1, 8) + self.centers_all.shape[0]
        self.children_all[nodes] = idx
        c = self.centers_all[nodes]
        h = self.half_all[nodes]
        lvl = self.level_all[nodes] + 1
        off = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]],
                       dtype=float) * 0.5
        self.centers_all = np.concatenate(
            [self.centers_all,
             (c[:, None, :] + off[None, :, :] * h[:, None, None]).reshape(-1, 3)]
        )
        self.half_all = np.concatenate([self.half_all, np.repeat(h * 0.5, 8)])
        self.level_all = np.concatenate([self.level_all, np.repeat(lvl, 8)])
        self.children_all = np.concatenate(
            [self.children_all, np.full((n_new, 8), -1, dtype=int)]
        )
        self.max_level = int(max(self.max_level, lvl.max()))

    def leaves_all(self):
        """Indices of all current leaf nodes (node numbering)."""
        return np.flatnonzero(self.children_all[:, 0] < 0)

    def refine(self, kappa_e_fn, tau_max=0.5, eps_grad=0.3, max_depth=6):
        """
        Top-down refinement: split a leaf when its corner-sampled kappa_e
        violates tau_max or the gradient criterion (plan section 3).
        """
        while True:
            leaves = self.leaves_all()
            if leaves.size == 0:
                break
            c = self.centers_all[leaves]
            h = self.half_all[leaves]
            # sample kappa_e at the 8 corners of each leaf
            off = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]],
                           dtype=float)
            pts = (c[:, None, :] + off[None, :, :] * h[:, None, None]).reshape(-1, 3)
            k = np.asarray(kappa_e_fn(pts[:, 0], pts[:, 1], pts[:, 2])).reshape(-1)
            k = k.reshape(-1, 8)
            kmax = k.max(axis=1)
            kmin = k.min(axis=1)
            kmean = k.mean(axis=1)
            cell_size = 2.0 * h
            tau_crit = kmax * cell_size > tau_max
            # gradient criterion only where the cell is not effectively
            # transparent (relative gradients in the vacuum tails of the
            # cloud are large but physically irrelevant)
            grad_crit = (
                (kmax > 0.0)
                & (kmean > 0.0)
                & (kmax * cell_size > 1e-2 * tau_max)
                & ((kmax - kmin) / kmean > eps_grad)
            )
            split = (tau_crit | grad_crit) & (self.level_all[leaves] < max_depth)
            if not split.any():
                break
            self.split(leaves[split])

    def balance_21(self, max_depth=6):
        """
        Enforce 2:1 face balance: a leaf may differ from any face neighbor
        by at most one level.  Each face is probed at a 3x3 grid (transverse
        offsets -h/2, 0, +h/2) so that a deeper neighbor cell of any allowed
        size is always hit; the loop repeats until no splits remain.
        """
        while True:
            self._compact()  # refresh leaf tables for locate()
            leaves = self.leaves_all()
            c = self.centers_all[leaves]
            h = self.half_all[leaves]
            split = np.zeros(leaves.size, dtype=bool)
            for axis in range(3):
                trans = [a for a in range(3) if a != axis]
                for side in (-1.0, 1.0):
                    for o1 in (-0.5, 0.0, 0.5):
                        for o2 in (-0.5, 0.0, 0.5):
                            probe = c.copy()
                            probe[:, axis] += side * (h + 1e-9 * h + 1e-12)
                            probe[:, trans[0]] += o1 * h
                            probe[:, trans[1]] += o2 * h
                            inside = self.cuboid.contains(probe)
                            nb = np.where(inside, self.locate(probe), -1)
                            nb_lvl = np.where(nb >= 0, self.level[nb], -1)
                            split |= nb_lvl > self.level_all[leaves] + 1
            if not split.any():
                break
            self.split(leaves[split])

    def _compact(self):
        """Freeze the current leaves into the public leaf-indexed arrays."""
        leaves = self.leaves_all()
        self.leaf_of_node = np.full(self.centers_all.shape[0], -1, dtype=int)
        self.leaf_of_node[leaves] = np.arange(leaves.size)
        self.n_leaf = leaves.size
        self.centers = self.centers_all[leaves]
        self.half = self.half_all[leaves]
        self.level = self.level_all[leaves]
        self.max_level = int(self.level.max()) if leaves.size else 0
        # children in leaf numbering for locate(): internal nodes are only
        # needed for the descent, which works on node numbering; store the
        # node-level arrays as-is.

    def finalize(self):
        """Call after build: compact leaf arrays and allocate field slots."""
        self._compact()
        self.kappa_e = np.zeros(self.n_leaf)
        self.kappa_s = np.zeros(self.n_leaf)
        self.g = np.zeros(self.n_leaf)
        self.T = np.zeros(self.n_leaf)

    # ------------------------------------------------------------- locate

    def locate(self, points):
        """
        Leaf index containing each (N, 3) point (vectorized descent from the
        base lattice; points are clamped to the domain).
        """
        p = np.atleast_2d(np.asarray(points, dtype=float)).copy()
        # clamp just inside the box
        for axis, (lo, hi) in enumerate(self.box):
            p[:, axis] = np.clip(p[:, axis], lo + 1e-12, hi - 1e-12)
        i = np.floor((p[:, 0] - self.box[0][0]) / self.h0).astype(int)
        j = np.floor((p[:, 1] - self.box[1][0]) / self.h0).astype(int)
        k = np.floor((p[:, 2] - self.box[2][0]) / self.h0).astype(int)
        i = np.clip(i, 0, self.nx0 - 1)
        j = np.clip(j, 0, self.ny0 - 1)
        k = np.clip(k, 0, self.nz0 - 1)
        node = self._base_id(i, j, k)
        for _ in range(self.max_level):
            ch = self.children_all[node]
            sel = np.flatnonzero(ch[:, 0] >= 0)
            if sel.size == 0:
                break
            octant = (
                (p[sel, 0] > self.centers_all[node[sel], 0]).astype(int)
                + 2 * (p[sel, 1] > self.centers_all[node[sel], 1]).astype(int)
                + 4 * (p[sel, 2] > self.centers_all[node[sel], 2]).astype(int)
            )
            node[sel] = self.children_all[node[sel], octant]
        return self.leaf_of_node[node]

    # ------------------------------------------------------------- fields

    def sample_fields(self, kappa_e=None, kappa_s=None, g=None, T=None):
        """Sample callables (x, y, z) -> array at leaf centers."""
        c = self.centers
        if kappa_e is not None:
            self.kappa_e = np.asarray(kappa_e(c[:, 0], c[:, 1], c[:, 2]), dtype=float)
        if kappa_s is not None:
            self.kappa_s = np.asarray(kappa_s(c[:, 0], c[:, 1], c[:, 2]), dtype=float)
        if g is not None:
            self.g = np.asarray(g(c[:, 0], c[:, 1], c[:, 2]), dtype=float)
        if T is not None:
            self.T = np.asarray(T(c[:, 0], c[:, 1], c[:, 2]), dtype=float)


# =============================================================================
# vectorized DDA march through the octree
# =============================================================================


def march_octree(origin, direction, oct, path_len, periodic_box=None):
    """
    March K rays from `origin` through the octree, each along its own unit
    direction, each with its own path-length budget. Segments are recorded
    in traversal order per ray.

    Parameters
    ----------
    origin : (K, 3) ndarray
        Ray start points inside the domain.
    direction : (3,) or (K, 3) ndarray
        Unit ray direction(s).
    oct : octree_grid
    path_len : (K,) or scalar
        Path-length budget per ray.
    periodic_box : rt3d.geometry.cuboid, optional
        If given, rays crossing a periodic x/y face wrap and continue.

    Returns
    -------
    seg_ray : (M,) ndarray of int
        Ray index of each segment (traversal order within each ray).
    seg_leaf : (M,) ndarray of int
        Octree leaf traversed by the segment.
    seg_len : (M,) ndarray
        Segment length.
    """
    o = np.atleast_2d(np.asarray(origin, dtype=float))
    K = o.shape[0]
    if K == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty.copy(), np.zeros(0)
    d = np.asarray(direction, dtype=float)
    if d.ndim == 1:
        d = np.tile(d, (K, 1))
    budget = np.broadcast_to(np.asarray(path_len, dtype=float), (K,)).astype(float)

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
            # face distances use the periodic image of the current leaf
            # closest to the ray position (p is kept wrapped to the box)
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


def node_major_segments(seg_ray, seg_leaf, seg_len, n_nodes):
    """
    Regroup step-major march segments into node-major contiguous blocks.

    Returns
    -------
    seg_ray, seg_leaf, seg_len : ndarray
        Sorted by node index (traversal order preserved within a node).
    offsets : (n_nodes + 1,) ndarray of int
        Block boundaries: segments of node c are [offsets[c], offsets[c+1]).
    """
    order = np.argsort(seg_ray, kind="stable")
    sr = seg_ray[order]
    counts = np.bincount(sr, minlength=n_nodes)
    offsets = np.zeros(n_nodes + 1, dtype=int)
    np.cumsum(counts, out=offsets[1:])
    return sr, seg_leaf[order], seg_len[order], offsets
