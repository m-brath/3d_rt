# -*- coding: utf-8 -*-
"""Phase 0 anchors: cuboid, ray-box intersection, unigrid, DDA march."""

import numpy as np
import pytest

from rt3d.geometry import cuboid, march_lattice, ray_box_exit, unigrid


@pytest.fixture
def grid():
    x = np.linspace(0.5, 9.5, 10)
    return unigrid(x, x.copy(), x.copy())


def test_unigrid_index_roundtrip(grid):
    idx = np.arange(grid.n_cells)
    i, j, k = grid.unravel(idx)
    assert np.array_equal(grid.ravel(i, j, k), idx)
    assert np.array_equal(grid.locate(grid.centers), idx)


def test_unigrid_rejects_nonuniform():
    with pytest.raises(ValueError):
        unigrid(np.array([0.0, 1.0, 3.0]), np.array([0.0]), np.array([0.0]))


@pytest.mark.parametrize(
    "d",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.6, 0.5, -0.4]),
        np.array([-0.1, 0.3, 0.95]),
    ],
)
def test_ray_box_exit(grid, d):
    rng = np.random.default_rng(7)
    o = rng.uniform(0.2, 9.8, (50, 3))
    t, ax, sd = ray_box_exit(o, d, grid.box)
    # analytic: first face hit along the ray
    for k in range(len(o)):
        best = np.inf
        for axis, (lo, hi) in enumerate(grid.box):
            if d[axis] > 0:
                tt = (hi - o[k, axis]) / d[axis]
                if tt < best:
                    best, axk, sdk = tt, axis, 1
            elif d[axis] < 0:
                tt = (lo - o[k, axis]) / d[axis]
                if tt < best:
                    best, axk, sdk = tt, axis, -1
        assert np.isclose(t[k], best)
        assert ax[k] == axk and sd[k] == sdk


@pytest.mark.parametrize(
    "d",
    [np.array([0.6, 0.5, -0.4]), np.array([-0.1, 0.3, 0.95])],
)
def test_march_nonperiodic_path(grid, d):
    rng = np.random.default_rng(11)
    o = rng.uniform(0.2, 9.8, (100, 3))
    steps, exit_t, eax, esd = march_lattice(o, d, grid)
    tot = np.zeros(len(o))
    for ridx, cell, ln in steps:
        tot[ridx] += ln
    assert np.allclose(tot, exit_t)
    for axis in range(3):
        m = (eax == axis) & (esd == np.sign(d[axis]))
        if np.any(m):
            face = grid.box[axis][1 if d[axis] > 0 else 0]
            assert np.allclose(exit_t[m], (face - o[m, axis]) / d[axis])


@pytest.mark.parametrize(
    "d",
    [np.array([0.6, 0.5, -0.4]), np.array([-0.1, 0.3, 0.95])],
)
def test_march_periodic(grid, d):
    box = cuboid(0, 10, 0, 10, 0, 10)
    rng = np.random.default_rng(13)
    o = rng.uniform(0.2, 9.8, (100, 3))
    steps, exit_t, eax, esd = march_lattice(o, d, grid, periodic_box=box)
    # every ray must exit through the z-face opposite to d_z
    assert np.all((eax == 2) & (esd == np.sign(d[2])))
    z_exit = grid.box[2][0] if d[2] < 0 else grid.box[2][1]
    assert np.allclose(exit_t, np.abs(z_exit - o[:, 2]) / abs(d[2]))
    # traversed cells must be face-contiguous (with wrap)
    last = {}
    for ridx, cell, ln in steps:
        for r, c in zip(ridx, cell):
            if r in last:
                i0, j0, k0 = grid.unravel(last[r])
                i1, j1, k1 = grid.unravel(c)
                di = min(abs(i0 - i1), grid.nx - abs(i0 - i1))
                dj = min(abs(j0 - j1), grid.ny - abs(j0 - j1))
                assert di + dj + abs(k0 - k1) == 1
            last[r] = c


def test_march_vertical_ray(grid):
    o = np.array([[5.5, 5.5, 9.5]])
    steps, exit_t, eax, esd = march_lattice(o, np.array([0.0, 0.0, -1.0]), grid)
    assert np.isclose(exit_t[0], 9.5)
    cells = np.concatenate([c for _, c, _ in steps])
    assert len(cells) == 10
