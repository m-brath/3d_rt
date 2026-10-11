# -*- coding: utf-8 -*-
"""Phase 1 anchors: SC sweep and beam sweep with kappa_s = 0 (Beer-Lambert)."""

import numpy as np
import pytest

from rt3d.characteristics import build_chain, stencil_for_direction, subsegments_uniform
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, ray_box_exit, unigrid
from rt3d.transport import beam_optical_depth, boundary_conditions, sweep_direction


@pytest.fixture
def slab():
    z = np.linspace(0.5, 9.5, 40)
    grid = unigrid(np.array([0.0]), np.array([0.0]), z)
    box = cuboid(-0.5, 0.5, -0.5, 0.5, 0.0, 10.0)
    return grid, box


@pytest.fixture
def box3d():
    x = np.linspace(0.5, 5.5, 6)
    y = np.linspace(0.5, 3.5, 4)
    z = np.linspace(0.5, 7.5, 8)
    return unigrid(x, y, z)


def test_stencil_closest_and_nonzero_z():
    dg = icosahedric_grid(n_refine=1)
    h = np.array([1.0, 1.0, 0.5])
    for omega in dg.directions:
        s, vhat = stencil_for_direction(omega, h)
        assert s[2] != 0
        # stencil must be the angle-optimal offset within the search radius
        best = max(
            ((np.array(t) * h) @ omega) / np.linalg.norm(np.array(t) * h)
            for t in _offsets(2)
        )
        cos_s = (s * h) @ omega / np.linalg.norm(s * h)
        assert cos_s > best - 1e-12


def _offsets(R):
    import itertools

    return [t for t in itertools.product(range(-R, R + 1), repeat=3)
            if t != (0, 0, 0) and t[2] != 0]


def test_subsegments_cover_path():
    for s in ((1, 1, -1), (2, -1, 3), (0, 0, -1), (3, 2, -1)):
        h = np.array([1.0, 0.7, 0.3])
        rel, lengths = subsegments_uniform(np.array(s), h)
        assert np.isclose(lengths.sum(), np.linalg.norm(np.array(s, float) * h))
        # first segment lies in the parent cell, last in the target cell
        assert np.array_equal(rel[0], -np.array(s))
        assert np.array_equal(rel[-1], np.zeros(3, int))
        # consecutive segments advance at most one cell per axis, always
        # monotonically from parent toward target (diagonal jumps happen
        # when the ray crosses a lattice edge/corner exactly)
        for a, b in zip(rel[:-1], rel[1:]):
            d = b - a
            assert np.all(np.abs(d) <= 1)
            assert np.all(d * np.sign(s) >= 0)


def test_beer_lambert_slab_periodic(slab):
    """kappa_s = 0, B = 0: I = I_entry * exp(-kappa * path) for every
    ordinate and cell (validation anchor 3)."""
    grid, box = slab
    kappa = 0.7
    kappa_e = np.full(grid.n_cells, kappa)
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.3)
    S = np.zeros(grid.n_cells)
    z_top, z_bot = grid.box[2][1], grid.box[2][0]
    for d in range(dg.n_dir):
        chain = build_chain(grid, dg.directions[d], periodic_box=box)
        I = sweep_direction(grid, chain, kappa_e, S, bc)
        vz = chain["vhat"][2]
        entry_I = bc.top_I if vz < 0 else bc.bottom_I
        z_entry = z_top if vz < 0 else z_bot
        path = np.abs(z_entry - grid.centers[:, 2]) / abs(vz)
        assert np.allclose(I, entry_I * np.exp(-kappa * path), atol=1e-12)


def test_isothermal_equilibrium(slab):
    """kappa_s = 0, S = B, I_in = B on both z-faces -> I = B exactly."""
    grid, box = slab
    kappa_e = np.full(grid.n_cells, 0.7)
    B = 2.5
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=B, bottom_I=B)
    S = np.full(grid.n_cells, B)
    for d in range(dg.n_dir):
        chain = build_chain(grid, dg.directions[d], periodic_box=box)
        I = sweep_direction(grid, chain, kappa_e, S, bc)
        assert np.allclose(I, B, atol=1e-12)


def test_beer_lambert_box3d_nonperiodic(box3d):
    """Non-periodic box: entry face varies per cell; analytic path from the
    backward ray-box exit."""
    grid = box3d
    rng = np.random.default_rng(3)
    kappa_e = rng.uniform(0.1, 2.0, grid.n_cells)
    dg = icosahedric_grid(n_refine=0)  # 12 ordinates: cheap here
    bc = boundary_conditions(top_I=1.0, bottom_I=0.5, side_I=0.25)
    S = np.zeros(grid.n_cells)
    for d in range(dg.n_dir):
        chain = build_chain(grid, dg.directions[d], periodic_box=None)
        I = sweep_direction(grid, chain, kappa_e, S, bc)
        # exact path: integrate kappa_e along the straight backward ray from
        # each center to the box entry point, using the chain itself
        # (roots carry the full path; here verify against a DDA march)
        from rt3d.geometry import march_lattice

        steps, exit_t, eax, esd = march_lattice(
            grid.centers, -chain["vhat"], grid, periodic_box=None
        )
        tau = np.zeros(grid.n_cells)
        for ridx, cell, ln in steps:
            tau[ridx] += kappa_e[cell] * ln
        I_in = bc.inflow(eax, esd)
        assert np.allclose(I, I_in * np.exp(-tau), atol=1e-12)


def test_beam_optical_depth_box3d(box3d):
    grid = box3d
    rng = np.random.default_rng(5)
    kappa_e = rng.uniform(0.0, 1.5, grid.n_cells)
    sun = np.array([0.3, -0.4, -0.86])
    sun = sun / np.linalg.norm(sun)
    tau = beam_optical_depth(grid, sun, kappa_e)
    assert np.all(tau >= 0)
    # top-layer cells: tau = kappa * half cell path
    _, _, k = grid.unravel(np.arange(grid.n_cells))
    top = k == grid.nz - 1
    expected = kappa_e[top] * grid.hz / 2 / abs(sun[2])
    assert np.allclose(tau[top], expected)


def test_two_cell_chain_exact():
    """Hand-computed chain: one cell in x,y, two cells in z, vertical
    ordinate, different kappa and S per cell."""
    grid = unigrid(np.array([0.0]), np.array([0.0]), np.array([0.5, 1.5]))
    omega = np.array([0.0, 0.0, -1.0])
    chain = build_chain(grid, omega)
    assert np.array_equal(chain["s"], np.array([0, 0, -1]))
    kappa_e = np.array([1.0, 3.0])  # bottom cell kappa=1, top cell kappa=3
    S = np.array([2.0, 5.0])
    bc = boundary_conditions(top_I=7.0, bottom_I=11.0)
    I = sweep_direction(grid, chain, kappa_e, S, bc)
    # top cell (index 1) is the root: entry at z=2, center z=1.5, length 0.5
    tau_top = 3.0 * 0.5
    I1 = 7.0 * np.exp(-tau_top) + 5.0 * (1 - np.exp(-tau_top))
    assert np.isclose(I[1], I1)
    # bottom cell (index 0): chain top center -> bottom center splits into
    #   top-cell half (len 0.5, kappa 3, S 5) + bottom-cell half (0.5, 1, 2)
    t1, t2 = 3.0 * 0.5, 1.0 * 0.5
    I0 = (
        I1 * np.exp(-(t1 + t2))
        + 5.0 * (1 - np.exp(-t1)) * np.exp(-t2)
        + 2.0 * (1 - np.exp(-t2))
    )
    assert np.isclose(I[0], I0)
