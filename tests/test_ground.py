# -*- coding: utf-8 -*-
"""Phase 5: Lambertian ground BC, energy-conservation diagnostics, all-sky."""

import numpy as np
import pytest

from rt3d.characteristics import build_chain
from rt3d.diagnostics import (
    allsky,
    energy_balance,
    flux_divergence,
    heating_rate,
)
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, ground_surface, unigrid
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions


# ------------------------------------------------------- Beer-Lambert + ground


def _slab(omega=0.0, nz=8):
    """Uniform kappa=1 slab, periodic x/y, z in [0, 2]."""
    x = (np.arange(4) + 0.5) * 0.5
    y = (np.arange(4) + 0.5) * 0.5
    z = (np.arange(nz) + 0.5) * 2.0 / nz
    grid = unigrid(x, y, z)
    box = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    kappa_e = np.full(grid.n_cells, 1.0)
    kappa_s = np.full(grid.n_cells, omega)
    return grid, box, kappa_e, kappa_s


def _analytic_ground(dg, z1, kappa, A, vhat=None, z_bot=None):
    """Per-ordinate analytic I for kappa_s = 0, B = 0, top_I = 1,
    Lambertian ground albedo A, uniform kappa, ground at z = 0.

    vhat : per-ordinate characteristic direction (default: the ordinate
        itself).  The unigrid chains transport along the lattice-stencil
        direction vhat, so the discrete reference must use it for the
        path; the ground irradiance always uses the physical omega_z
        (same convention as the solver).
    z_bot : z of the bottom-layer cell centers used for F_down
        (default: 0, exact for the octree).
    """
    if vhat is None:
        vz = dg.directions[:, 2]
    else:
        vz = np.asarray(vhat)
    if z_bot is None:
        z_bot = 0.0
    mu = dg.directions[:, 2]
    F_down = np.sum(
        dg.weights[dg.down_idx] * np.abs(mu[dg.down_idx])
        * np.exp(-kappa * (z1 - z_bot) / np.abs(vz[dg.down_idx]))
    )
    I_g = A * F_down / np.pi
    return I_g, F_down


@pytest.mark.parametrize("A", [0.0, 0.3, 0.9])
def test_ground_beer_lambert_unigrid(A):
    dg = icosahedric_grid(n_refine=1)
    grid, box, kappa_e, kappa_s = _slab()
    bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=A))
    res = lambda_iteration(grid, dg, kappa_e, kappa_s,
                           np.zeros(grid.n_cells), bc, periodic_box=box,
                           n_iter=5, tol=1e-12)
    assert res["converged"]

    z1 = grid.box[2][1]
    # discrete reference: lattice-stencil characteristic direction per
    # ordinate (the chains transport along vhat, not omega)
    vz = np.array([
        build_chain(grid, dg.directions[d], periodic_box=box)["vhat"][2]
        for d in range(dg.n_dir)
    ])
    I_g, _ = _analytic_ground(dg, z1, 1.0, A, vhat=vz, z_bot=grid.z[0])
    zc = grid.centers[:, 2]
    I_ref = np.where(
        vz[None, :] < 0.0,
        np.exp(-(z1 - zc[:, None]) / np.abs(vz[None, :])),
        I_g * np.exp(-zc[:, None] / vz[None, :]),
    )
    assert np.max(np.abs(res["I"] - I_ref)) < 1e-12


@pytest.mark.parametrize("A", [0.0, 0.6])
def test_ground_beer_lambert_octree(A):
    dg = icosahedric_grid(n_refine=1)
    box = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    oct = octree_grid(box, h0=0.5)
    oct.refine(lambda x, y, z: np.full_like(z, 1.0), tau_max=1e6,
               eps_grad=1e6, max_depth=0)
    oct.balance_21(max_depth=0)
    oct.finalize()
    kappa_e = np.full(oct.n_leaf, 1.0)
    kappa_s = np.zeros(oct.n_leaf)

    bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=A))
    res = lambda_iteration(oct, dg, kappa_e, kappa_s,
                           np.zeros(oct.n_leaf), bc, periodic_box=box,
                           n_iter=5, tol=1e-12)
    assert res["converged"]

    z1 = box.z1
    # F_down is sampled at bottom-face leaf centers (z = h0/2), like the solver
    I_g, _ = _analytic_ground(dg, z1, 1.0, A, z_bot=oct.h0 / 2)
    mu = dg.directions[:, 2]
    zc = oct.centers[:, 2]
    I_ref = np.where(
        mu[None, :] < 0.0,
        np.exp(-(z1 - zc[:, None]) / np.abs(mu[None, :])),
        I_g * np.exp(-zc[:, None] / mu[None, :]),
    )
    assert np.max(np.abs(res["I"] - I_ref)) < 1e-12


def test_ground_emission_analytic():
    """eps=1, A=0, B_g=1, no top illumination, kappa_s=0: upward ordinates
    carry the Lambertian ground emission, attenuated along the lattice
    characteristic to each center."""
    dg = icosahedric_grid(n_refine=1)
    grid, box, kappa_e, kappa_s = _slab()
    bc = boundary_conditions(ground=ground_surface(albedo=0.0, B_g=1.0))
    res = lambda_iteration(grid, dg, kappa_e, kappa_s,
                           np.zeros(grid.n_cells), bc, periodic_box=box,
                           n_iter=5, tol=1e-12)
    vz = np.array([
        build_chain(grid, dg.directions[d], periodic_box=box)["vhat"][2]
        for d in range(dg.n_dir)
    ])
    zc = grid.centers[:, 2]
    I_ref = np.where(
        vz[None, :] > 0.0, np.exp(-zc[:, None] / vz[None, :]), 0.0
    )
    assert np.max(np.abs(res["I"] - I_ref)) < 1e-12


# ---------------------------------------------------------------- diagnostics


def _converged_small_run():
    dg = icosahedric_grid(n_refine=1)
    grid, box, kappa_e, kappa_s = _slab(omega=0.5)
    bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=0.3))
    res = lambda_iteration(grid, dg, kappa_e, kappa_s,
                           np.zeros(grid.n_cells), bc, periodic_box=box,
                           n_iter=500, tol=1e-9, ng=True)
    assert res["converged"]
    return dg, grid, res


def test_div_F_equals_heating_rate():
    """Validation anchor 4: div F = 4 pi kappa_a (B - J) pointwise
    (finite-difference accuracy on this coarse slab)."""
    dg, grid, res = _converged_small_run()
    kappa_a = 1.0 - res["omega"]  # kappa_e = 1, kappa_s = omega
    divF = flux_divergence(grid, res["F"])
    heat = heating_rate(kappa_a, res["J"], 0.0)
    scale = np.max(np.abs(heat))
    # interior only: one-sided differences at top/bottom carry O(h) error
    k = np.arange(grid.n_cells) // (grid.ny * grid.nx)
    interior = (k > 0) & (k < grid.nz - 1)
    assert np.max(np.abs(divF[interior] - heat[interior])) / scale < 0.05


def test_energy_balance_closure():
    dg, grid, res = _converged_small_run()
    kappa_a = 1.0 - res["omega"]
    bal = energy_balance(grid, res["I"], dg, kappa_a, 0.0)
    scale = max(abs(bal["out_top"]), abs(bal["heating"]))
    assert abs(bal["residual"]) / scale < 0.05
    # with B = 0 the radiation field must lose energy (net absorption)
    assert bal["heating"] < 0.0


def test_allsky_images():
    """kappa_s = 0, ground albedo A: top all-sky images of the upward
    ordinates are the analytic attenuated ground radiance."""
    dg = icosahedric_grid(n_refine=1)
    grid, box, kappa_e, kappa_s = _slab()
    A = 0.4
    bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=A))
    res = lambda_iteration(grid, dg, kappa_e, kappa_s,
                           np.zeros(grid.n_cells), bc, periodic_box=box,
                           n_iter=5, tol=1e-12)
    sky = allsky(grid, res["I"], dg, face="top")
    n_up = dg.up_idx.size
    assert sky["images"].shape == (n_up, grid.ny, grid.nx)
    vz = np.array([
        build_chain(grid, dg.directions[d], periodic_box=box)["vhat"][2]
        for d in range(dg.n_dir)
    ])
    I_g, _ = _analytic_ground(dg, grid.box[2][1], 1.0, A, vhat=vz,
                              z_bot=grid.z[0])
    z_top = grid.z[-1]
    ref = I_g * np.exp(-z_top / vz[dg.up_idx])
    assert np.max(np.abs(sky["images"] - ref[:, None, None])) < 1e-12


# ------------------------------------------------- variable-surface hook (ph 6)


def test_ground_height_hook():
    gs = ground_surface(albedo=0.3, height=1.5)
    assert not gs.is_variable()
    assert gs.height() == 1.5
    assert gs.height(3.0, -2.0) == 1.5

    gs = ground_surface(height=lambda x, y: 0.1 * np.sin(x) * np.cos(y))
    assert gs.is_variable()
    assert np.isclose(gs.height(0.3, 0.4), 0.1 * np.sin(0.3) * np.cos(0.4))


def test_variable_ground_raises():
    """Transport only implements the flat ground; the hook must fail loudly."""
    dg = icosahedric_grid(n_refine=1)
    grid, box, kappa_e, kappa_s = _slab()
    gs = ground_surface(albedo=0.3, height=lambda x, y: 0.1 * np.sin(x))
    bc = boundary_conditions(top_I=1.0, ground=gs)
    with pytest.raises(NotImplementedError):
        lambda_iteration(grid, dg, kappa_e, kappa_s,
                         np.zeros(grid.n_cells), bc, periodic_box=box,
                         n_iter=2)
