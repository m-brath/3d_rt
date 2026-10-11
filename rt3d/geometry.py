# -*- coding: utf-8 -*-
"""
Domain geometry: cuboid with periodic x/y sides and flat ground at the bottom,
uniform Cartesian cell lattice (unigrid), ray-box intersection and a
vectorized grid-traversal (DDA) ray march.

The unigrid is the phase-1 spatial grid; the octree (rt3d/octree.py, phase 3)
replaces it later. Both expose the same interface used by the transport sweep:
cell centers, cell boxes, and O(1) point location.
"""

import numpy as np


# =============================================================================
# cuboid domain
# =============================================================================


class cuboid(object):
    """
    Axis-aligned cuboid domain. Top (z = z1) is space, bottom (z = z0) is
    flat ground, x and y are periodic.

    Attributes
    ----------
    x0, x1, y0, y1, z0, z1 : float
        Face coordinates.
    periodic_x, periodic_y : bool
        Whether the corresponding sides wrap.
    """

    def __init__(self, x0, x1, y0, y1, z0, z1, periodic_x=True, periodic_y=True):
        if not (x0 < x1 and y0 < y1 and z0 < z1):
            raise ValueError("cuboid bounds must be ordered")
        self.x0, self.x1 = float(x0), float(x1)
        self.y0, self.y1 = float(y0), float(y1)
        self.z0, self.z1 = float(z0), float(z1)
        self.periodic_x = bool(periodic_x)
        self.periodic_y = bool(periodic_y)

    @property
    def Lx(self):
        return self.x1 - self.x0

    @property
    def Ly(self):
        return self.y1 - self.y0

    @property
    def Lz(self):
        return self.z1 - self.z0

    def contains(self, points):
        """Boolean mask whether each (N, 3) point lies inside the cuboid."""
        p = np.atleast_2d(points)
        return (
            (p[:, 0] >= self.x0)
            & (p[:, 0] <= self.x1)
            & (p[:, 1] >= self.y0)
            & (p[:, 1] <= self.y1)
            & (p[:, 2] >= self.z0)
            & (p[:, 2] <= self.z1)
        )

    def wrap_xy(self, points):
        """Wrap (N, 3) points periodically in x and y into the cuboid."""
        p = np.atleast_2d(points).copy()
        if self.periodic_x:
            p[:, 0] = self.x0 + np.mod(p[:, 0] - self.x0, self.Lx)
        if self.periodic_y:
            p[:, 1] = self.y0 + np.mod(p[:, 1] - self.y0, self.Ly)
        return p


# =============================================================================
# ray-box intersection (slab method)
# =============================================================================


def ray_box_exit(origin, direction, box):
    """
    Distance from an interior point to the box exit along `direction`.

    Parameters
    ----------
    origin : (N, 3) or (3,) ndarray
        Ray start points, assumed inside the box.
    direction : (3,) or (N, 3) ndarray
        Unit ray directions.
    box : tuple
        ((x0, x1), (y0, y1), (z0, z1)) face coordinates.

    Returns
    -------
    t_exit : (N,) ndarray
        Distance to the exit point.
    exit_axis : (N,) ndarray of int
        Axis (0/1/2) of the face the ray exits through.
    exit_side : (N,) ndarray of int
        -1 / +1: lower / upper face of that axis.
    """
    o = np.atleast_2d(origin).astype(float)
    d = np.asarray(direction, dtype=float)
    if d.ndim == 1:
        d = np.tile(d, (o.shape[0], 1))

    n = o.shape[0]
    t_exit = np.full(n, np.inf)
    exit_axis = np.zeros(n, dtype=int)
    exit_side = np.zeros(n, dtype=int)

    for axis, (lo, hi) in enumerate(box):
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = (lo - o[:, axis]) / d[:, axis]
            t_hi = (hi - o[:, axis]) / d[:, axis]
        # forward intersections only
        t_lo = np.where(d[:, axis] < 0.0, t_lo, np.inf)
        t_hi = np.where(d[:, axis] > 0.0, t_hi, np.inf)
        t_axis = np.minimum(t_lo, t_hi)
        side = np.where(d[:, axis] > 0.0, 1, -1)

        upd = t_axis < t_exit
        t_exit = np.where(upd, t_axis, t_exit)
        exit_axis = np.where(upd, axis, exit_axis)
        exit_side = np.where(upd, side, exit_side)

    return t_exit, exit_axis, exit_side


# =============================================================================
# uniform Cartesian cell lattice
# =============================================================================


class unigrid(object):
    """
    Uniform Cartesian cell lattice defined by cell-center coordinates.

    Cell (i, j, k) has center (x[i], y[j], z[k]) and flat index
    k*ny*nx + j*nx + i. The domain box extends half a cell beyond the
    outermost centers.

    Attributes
    ----------
    x, y, z : (nx/ny/nz,) ndarray
        Cell-center coordinates (uniformly spaced).
    hx, hy, hz : float
        Cell sizes.
    nx, ny, nz, n_cells : int
    centers : (n_cells, 3) ndarray
        Cell centers, flat-index order.
    box : tuple
        ((x0, x1), (y0, y1), (z0, z1)) domain faces.
    """

    def __init__(self, x, y, z):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        for name, arr in (("x", self.x), ("y", self.y), ("z", self.z)):
            if arr.ndim != 1 or arr.size < 1:
                raise ValueError(f"{name} must be a 1D array of cell centers")
        # a single-cell dimension gets a nominal unit width so the domain box
        # stays finite (plane-parallel slab = 1 cell + periodic sides)
        self.hx = self._spacing(self.x, "x")
        self.hy = self._spacing(self.y, "y")
        self.hz = self._spacing(self.z, "z")

        self.nx, self.ny, self.nz = self.x.size, self.y.size, self.z.size
        self.n_cells = self.nx * self.ny * self.nz

        # centers in flat-index order (i fastest), consistent with ravel/unravel
        i, j, k = self.unravel(np.arange(self.n_cells))
        self.centers = np.column_stack([self.x[i], self.y[j], self.z[k]])

        self.box = (
            (self.x[0] - self.hx / 2, self.x[-1] + self.hx / 2),
            (self.y[0] - self.hy / 2, self.y[-1] + self.hy / 2),
            (self.z[0] - self.hz / 2, self.z[-1] + self.hz / 2),
        )

    @staticmethod
    def _spacing(arr, name):
        if arr.size == 1:
            return 1.0
        h = arr[1] - arr[0]
        if h <= 0 or not np.allclose(np.diff(arr), h):
            raise ValueError(f"{name} cell centers must be uniformly spaced, increasing")
        return h

    def cell_box(self, idx):
        """((N, 3), (N, 3)) lower/upper corners of cells with flat index idx."""
        i, j, k = self.unravel(idx)
        lo = np.column_stack([self.x[i] - self.hx / 2, self.y[j] - self.hy / 2, self.z[k] - self.hz / 2])
        hi = lo + np.array([self.hx, self.hy, self.hz])
        return lo, hi

    def unravel(self, idx):
        """Flat index -> (i, j, k) index arrays."""
        idx = np.asarray(idx)
        i = idx % self.nx
        j = (idx // self.nx) % self.ny
        k = idx // (self.nx * self.ny)
        return i, j, k

    def ravel(self, i, j, k):
        """(i, j, k) index arrays -> flat index."""
        return np.asarray(k) * self.nx * self.ny + np.asarray(j) * self.nx + np.asarray(i)

    def locate(self, points):
        """
        O(1) point location: flat index of the cell containing each (N, 3)
        point, clamped to the domain.
        """
        p = np.atleast_2d(points)
        i = np.floor((p[:, 0] - (self.x[0] - self.hx / 2)) / self.hx + 1e-9).astype(int)
        j = np.floor((p[:, 1] - (self.y[0] - self.hy / 2)) / self.hy + 1e-9).astype(int)
        k = np.floor((p[:, 2] - (self.z[0] - self.hz / 2)) / self.hz + 1e-9).astype(int)
        i = np.clip(i, 0, self.nx - 1)
        j = np.clip(j, 0, self.ny - 1)
        k = np.clip(k, 0, self.nz - 1)
        return self.ravel(i, j, k)





# =============================================================================
# vectorized grid-traversal (DDA) march on the unigrid
# =============================================================================


def march_lattice(origin, direction, grid, max_steps=None, periodic_box=None):
    """
    March K rays from `origin` along a shared unit `direction` through the
    unigrid, recording the exact sequence of traversed cells and segment
    lengths (Amanatides & Woo grid traversal).

    Parameters
    ----------
    origin : (K, 3) ndarray
        Ray start points inside the domain box.
    direction : (3,) ndarray
        Shared unit direction.
    grid : unigrid
    max_steps : int, optional
        Safety cap on traversal steps, by default enough to cross the box.
    periodic_box : cuboid, optional
        If given, rays exiting through a periodic x/y face are wrapped and the
        march continues; only top/bottom exits terminate the ray.

    Returns
    -------
    steps : list of (ray_idx, cell_idx, length) ndarray tuples
        Step-major: at step s, ray_idx are the still-active rays and each
        contributes the segment (cell_idx, length). Rays appear in traversal
        order; a ray's last segment is the one that ends at its exit.
    exit_t : (K,) ndarray
        Total path length from origin to exit for each ray.
    exit_axis, exit_side : (K,) ndarray of int
        Face the ray finally exits through.
    """
    o = np.atleast_2d(origin).astype(float)
    d = np.asarray(direction, dtype=float)
    K = o.shape[0]
    p = o.copy()

    if max_steps is None:
        # non-periodic path <= box diagonal; with periodic wrap the path is
        # bounded by the full z traversal Lz/|Omega_z|. Each traversal step
        # crosses one lattice plane, so the step count is bounded by
        # path * sum_i |d_i| / h_i.
        diag = np.sqrt(sum((b[1] - b[0]) ** 2 for b in grid.box))
        max_path = diag
        if periodic_box is not None and d[2] != 0.0:
            max_path = max(max_path, periodic_box.Lz / abs(d[2]))
        planes_per_length = sum(
            abs(d[i]) / h for i, h in enumerate((grid.hx, grid.hy, grid.hz)) if np.isfinite(h)
        )
        max_steps = int(np.ceil(max_path * planes_per_length)) + 4

    box = grid.box
    starts = np.array([b[0] for b in box])
    sizes = np.array([grid.hx, grid.hy, grid.hz])

    active = np.arange(K)
    exit_t = np.zeros(K)
    exit_axis = np.zeros(K, dtype=int)
    exit_side = np.zeros(K, dtype=int)
    steps = []

    for _ in range(max_steps):
        if active.size == 0:
            break
        pa = p[active]

        # distance to next internal lattice face per axis
        t_face = np.full(active.size, np.inf)
        for axis in range(3):
            da = d[axis]
            if da == 0.0 or not np.isfinite(sizes[axis]):
                continue
            h = sizes[axis]
            s0 = starts[axis]
            eps = 1e-9 * h
            # snap points that sit (numerically) on a lattice plane up to the
            # upper cell, otherwise floor() can pick the lower cell by float
            # noise and the ray stalls on zero-length steps
            m = np.floor((pa[:, axis] - s0 + eps) / h)
            base = s0 + m * h
            if da > 0:
                nxt = base + h
            else:
                # if the point sits exactly on the lower face of its cell,
                # the next face in -axis direction is one cell further down
                nxt = np.where(pa[:, axis] > base + eps, base, base - h)
            t_face = np.minimum(t_face, (nxt - pa[:, axis]) / da)

        t_exit, ax, sd = ray_box_exit(pa, d, box)
        t = np.minimum(t_face, t_exit)
        # guard against zero-length steps from float noise at faces
        t = np.maximum(t, 0.0)

        # identify the traversed cell at the segment midpoint: a ray starting
        # exactly on a cell face would otherwise be attributed to the cell it
        # is leaving rather than the one it is entering
        cell = grid.locate(pa + (0.5 * t)[:, None] * d)

        # drop zero-length segments (float noise at faces) from the record
        keep = t > 1e-12
        if np.any(keep):
            steps.append((active[keep].copy(), cell[keep], t[keep]))
        p[active] = pa + t[:, None] * d
        exit_t[active] += t

        exited = t >= t_exit - 1e-12 * np.maximum(1.0, np.linalg.norm(pa, axis=1))

        # exits through periodic x/y faces wrap around and keep marching;
        # all other exits (top, bottom, non-periodic sides) terminate the ray
        wrapped = np.zeros(active.size, dtype=bool)
        if periodic_box is not None:
            wrapped = exited & (
                ((ax == 0) & periodic_box.periodic_x) | ((ax == 1) & periodic_box.periodic_y)
            )
            if np.any(wrapped):
                pw = periodic_box.wrap_xy(p[active[wrapped]])
                # A ray that exited exactly (or within float noise) on a
                # periodic face must re-enter at the *opposite* face:
                # mod() maps an exact lower-face point back onto the lower
                # face, and a point within noise of it would re-exit
                # immediately with a zero-length step, stalling the march
                # until max_steps without ever recording an exit.
                for axis, (lo, hi) in enumerate(
                    ((periodic_box.x0, periodic_box.x1), (periodic_box.y0, periodic_box.y1))
                ):
                    if not (periodic_box.periodic_x if axis == 0 else periodic_box.periodic_y):
                        continue
                    tol = 1e-9 * (hi - lo)
                    if d[axis] < 0:
                        at_lo = pw[:, axis] <= lo + tol
                        pw[at_lo, axis] = hi
                    elif d[axis] > 0:
                        at_hi = pw[:, axis] >= hi - tol
                        pw[at_hi, axis] = lo
                p[active[wrapped]] = pw

        terminated = exited & ~wrapped
        newly_done = active[terminated]
        exit_axis[newly_done] = ax[terminated]
        exit_side[newly_done] = sd[terminated]

        active = active[~terminated]
        if active.size == 0:
            break

    return steps, exit_t, exit_axis, exit_side


# =============================================================================
# ground surface hook
# =============================================================================


class ground_surface(object):
    """
    Flat Lambertian + emissive ground at z = z0 (plan section 1).

    I_up = A_g/pi * (F_down + F_beam) + eps_g * B_g, with the solar beam
    reflected Lambertian as well (F_beam = F0 |mu_sun| exp(-tau_b) at the
    ground).  Monochromatic: the ground emission is the Planck value B_g
    at the working frequency, given as an input like every other B.

    Variable-surface hook (plan section 1 / phase 6): `height` is either a
    scalar (flat ground, the default) or a callable f(x, y) -> z.  The
    transport currently supports only a flat ground at the domain bottom;
    the solver raises NotImplementedError when `is_variable()` is True.
    """

    def __init__(self, albedo=0.0, emissivity=1.0, B_g=0.0, height=0.0):
        self.albedo = float(albedo)
        self.emissivity = float(emissivity)
        self.B_g = float(B_g)
        self._height = height

    def height(self, x=None, y=None):
        """Surface height z_g(x, y); scalar for a flat ground."""
        if callable(self._height):
            return self._height(x, y)
        return self._height

    def is_variable(self):
        """True if a variable surface was supplied through the hook."""
        return callable(self._height)
