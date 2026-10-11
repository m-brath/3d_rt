# -*- coding: utf-8 -*-
"""Phase 2 anchors: scattering source, plain lambda iteration."""

import numpy as np
import pytest
from scipy.special import expn

from rt3d.characteristics import build_chain
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid
from rt3d.solver import lambda_iteration
from rt3d.source import mean_intensity, single_scattering_albedo
from rt3d.transport import boundary_conditions, sweep_all


@pytest.fixture
def slab():
    z = np.linspace(0.5, 9.5, 40)
    grid = unigrid(np.array([0.0]), np.array([0.0]), z)
    box = cuboid(-0.5, 0.5, -0.5, 0.5, 0.0, 10.0)
    return grid, box


def test_albedo_guard():
    kappa_e = np.array([0.0, 1.0, 2.0])
    kappa_s = np.array([0.0, 0.5, 2.0])
    omega = single_scattering_albedo(kappa_e, kappa_s)
    assert np.allclose(omega, [0.0, 0.5, 1.0])


def test_pure_absorption_matches_phase1(slab):
    """kappa_s = 0: the iteration must reproduce the single sweep exactly."""
    grid, box = slab
    rng = np.random.default_rng(11)
    kappa_e = rng.uniform(0.1, 2.0, grid.n_cells)
    B = rng.uniform(0.5, 3.0, grid.n_cells)
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.4)
    res = lambda_iteration(grid, dg, kappa_e, np.zeros_like(kappa_e), B, bc,
                           periodic_box=box, n_iter=2)
    I_ref, _ = sweep_all(grid, dg, kappa_e, B, bc, periodic_box=box)
    assert np.allclose(res["I"], I_ref, atol=1e-12)
    assert res["history"][0] < 1e-12  # fixed point after one sweep


def test_isothermal_fixed_point(slab):
    """B = I_in on all faces: S = B is the exact fixed point for any omega."""
    grid, box = slab
    kappa_e = np.full(grid.n_cells, 0.7)
    kappa_s = 0.6 * kappa_e
    B = np.full(grid.n_cells, 2.5)
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=2.5, bottom_I=2.5)
    res = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc,
                           periodic_box=box, n_iter=200, tol=1e-12)
    assert res["converged"]
    assert np.allclose(res["I"], 2.5, atol=1e-10)
    assert np.allclose(res["J"], 2.5, atol=1e-10)


def test_uniform_source_E2(slab):
    """Uniform isotropic source S in a slab with vacuum boundaries:
    J(z) = S * [1 - 1/2 E2(tau_down) - 1/2 E2(tau_up)] (exact angular
    integral; half-space surface limit J = S/2). Agreement is limited only
    by the 42-ordinate quadrature + lattice stencil."""
    grid, box = slab
    S0 = 1.0
    kappa = 0.8
    kappa_e = np.full(grid.n_cells, kappa)
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=0.0, bottom_I=0.0)
    S = np.full(grid.n_cells, S0)
    I, _ = sweep_all(grid, dg, kappa_e, S, bc, periodic_box=box)
    J = mean_intensity(I, dg)
    z = grid.centers[:, 2]
    z0, z1 = grid.box[2][0], grid.box[2][1]
    J_exact = S0 * (
        1.0 - 0.5 * expn(2, kappa * (z - z0)) - 0.5 * expn(2, kappa * (z1 - z))
    )
    # 42 ordinates on an anisotropic lattice (hz = hx/4): interior cells
    # agree to ~2e-3; the boundary cells carry the intrinsic stencil error
    # (intensity transported along vhat but weighted as Omega). The
    # per-ordinate check below is the exact anchor.
    assert np.max(np.abs(J - J_exact)) < 2e-2
    # exact per-ordinate check: I = S (1 - exp(-tau_path)) along vhat,
    # tau_path measured from the entry face (bottom for vz > 0, top for vz < 0)
    for d in range(dg.n_dir):
        chain = build_chain(grid, dg.directions[d], periodic_box=box)
        vz = chain["vhat"][2]
        path = np.where(vz > 0, z - z0, z1 - z) / abs(vz)
        assert np.allclose(I[:, d], S0 * (1.0 - np.exp(-kappa * path)), atol=1e-12)


def test_symmetric_slab(slab):
    """Symmetric illumination -> J symmetric about the midplane."""
    grid, box = slab
    kappa_e = np.full(grid.n_cells, 0.5)
    kappa_s = 0.25 * kappa_e
    B = np.zeros(grid.n_cells)
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=1.0, bottom_I=1.0)
    res = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc,
                           periodic_box=box, n_iter=200, tol=1e-10)
    assert res["converged"]
    J = res["J"]
    assert np.allclose(J, J[::-1], atol=1e-12)


def test_conservative_flux(slab):
    """omega = 1, B = 0, g = 0: no absorption -> net flux F_z constant in z
    (within the cell-center sampling error of the discrete ordinates)."""
    grid, box = slab
    kappa_e = np.full(grid.n_cells, 0.5)
    kappa_s = kappa_e
    B = np.zeros(grid.n_cells)
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    res = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, g=0.0,
                           periodic_box=box, n_iter=500, tol=1e-10)
    assert res["converged"]
    Fz = res["F"][:, 2]
    variation = (Fz.max() - Fz.min()) / Fz.mean()
    assert variation < 5e-2  # discretization-level, not physics-level
    # net flux is downward (negative), smaller in magnitude than the
    # incident diffuse flux pi (the conservative tau=5 slab diffusely
    # reflects most of it), and nonzero: 0 < |Fz| < pi
    assert np.all(Fz < 0) and np.all(np.abs(Fz) < np.pi)


def test_convergence_history(slab):
    """Plain LI converges geometrically for omega = 0.5."""
    grid, box = slab
    kappa_e = np.full(grid.n_cells, 0.5)
    kappa_s = 0.5 * kappa_e
    B = np.zeros(grid.n_cells)
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    res = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc,
                           periodic_box=box, n_iter=100, tol=1e-10)
    assert res["converged"]
    h = np.array(res["history"])
    assert np.all(np.diff(h) < 0)  # monotone decrease
