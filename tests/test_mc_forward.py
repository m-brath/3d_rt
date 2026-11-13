# -*- coding: utf-8 -*-
"""
MC phase M1 anchors (plan/mc-plan.md section 7): stage-1 mean intensity
``J`` by both MC variants against the phase-2 references.

  1. 1D slab (tau=5, g=0, omega in {0.5, 0.9}): MC ``J(z)`` matches the 1D
     discrete-ordinates reference within the combined 2 sigma band plus a
     discretization allowance (the reference is a cell-center solution, MC
     tallies cell averages; MC transport itself is angle-exact)
  2. the lambda-iteration and explicit-scattering variants agree with each
     other within combined 2 sigma on the slab and on the gaussian cloud
  3. Ng acceleration (crn=True) needs far fewer iterations than plain LI
     at omega = 0.9 and lands on the same source
  4. global energy closure (validation anchor 4): net downward face flux ==
     total absorption 4 pi kappa_a int J dV
  5. direct beam, pure absorber: the explicit variant's unscattered-beam
     tally J_direct matches F0 exp(-tau_b) / 4 pi analytically
  6. emission-rate convention: slab results depend on tau = kappa_e * z only
     (kappa_e = 2 and kappa_e = 1 at equal tau agree against the 1D-DO
     reference)
  7. octree + global majorant mode run on the gaussian cloud and agree
     with each other statistically
  8. traceability fields + unimplemented-feature guards
"""

import numpy as np
import pytest

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.cases.slab_1d import d1o_J, slab_grid
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid
from rt3d.mc.forward import j_mc
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.transport import beam_optical_depth, boundary_conditions


# --------------------------------------------------------------------------
# shared case builders
# --------------------------------------------------------------------------


def _slab(tau=5.0, kappa_e=1.0, nz=50, nxy=4):
    grid, box, hz = slab_grid(tau, kappa_e, nz, nxy)

    def col(field):
        """vertical profile: column (i,j)=(1,1) of a (n_cells,) or (n_cells,3) field"""
        a = np.asarray(field)
        if a.ndim == 1:
            return a.reshape(grid.nz, grid.nx, grid.ny)[:, 1, 1]
        return a.reshape(grid.nz, grid.nx, grid.ny, 3)[:, 1, 1, :]

    return grid, box, hz, col


@pytest.fixture(scope="module")
def d1o_half():
    return d1o_J(1.0, 0.5, 0.1, 50)


@pytest.fixture(scope="module")
def d1o_om9():
    return d1o_J(1.0, 0.9, 0.1, 50)


# --------------------------------------------------------------------------
# 1 + 2 + 6: slab J vs 1D-DO reference, variant agreement, kappa scaling
# --------------------------------------------------------------------------


def test_slab_j_vs_1d_do_lambda(d1o_half):
    """lambda variant: J(z) within 2 sigma + 3% of J_top of the 1D-DO reference."""
    grid, box, hz, col = _slab()
    ke = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=1.0)
    r = j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, g=0.0,
             periodic_box=box, stage1="lambda", n_histories=100000,
             n_boundary_histories=100000, n_batches=200, seed=11,
             n_iter=200, tol=1e-8)
    J, sJ = col(r["J"]), col(r["sigma_J"])
    dev = np.abs(J - d1o_half)
    assert dev.max() <= 2.0 * sJ.max() + 0.03 * d1o_half.max()
    # 90% of the cells must sit inside the pure statistical 2 sigma band
    assert np.count_nonzero(dev <= 2.0 * sJ + 1e-4) >= 0.9 * grid.nz


def test_slab_j_vs_1d_do_explicit(d1o_half):
    """explicit variant: same acceptance band as the lambda variant."""
    grid, box, hz, col = _slab()
    ke = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=1.0)
    r = j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, g=0.0,
             periodic_box=box, stage1="explicit", n_histories=100000,
             n_boundary_histories=100000, n_batches=200, seed=12)
    J, sJ = col(r["J"]), col(r["sigma_J"])
    dev = np.abs(J - d1o_half)
    assert dev.max() <= 2.0 * sJ.max() + 0.03 * d1o_half.max()


def test_slab_variants_agree(d1o_half):
    """lambda and explicit variants agree with each other within 2 sigma."""
    grid, box, hz, col = _slab()
    ke = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=1.0)
    common = dict(g=0.0, periodic_box=box, n_histories=100000,
                  n_boundary_histories=100000, n_batches=200)
    rl = j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, seed=13,
              stage1="lambda", n_iter=200, tol=1e-8, **common)
    rx = j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, seed=14,
              stage1="explicit", **common)
    dev = np.abs(col(rl["J"]) - col(rx["J"]))
    sig = np.hypot(col(rl["sigma_J"]), col(rx["sigma_J"]))
    assert dev.max() <= 2.0 * sig.max()


def test_slab_kappa_scaling_emission_convention(d1o_half):
    """
    Birth weights must carry the emission rate kappa_e * S: at fixed optical
    depth tau = kappa_e * L the slab solution is identical, so a kappa_e = 2
    run must match the kappa_e = 1 reference within the same statistical band
    (a missing kappa_e factor would bias J strongly and diverge the LI).
    """
    grid, box, hz, col = _slab(kappa_e=2.0)
    ke = np.full(grid.n_cells, 2.0)
    bc = boundary_conditions(top_I=1.0)
    r = j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, g=0.0,
             periodic_box=box, stage1="lambda", n_histories=100000,
             n_boundary_histories=100000, n_batches=200, seed=15,
             n_iter=200, tol=1e-8, crn=True, birth="uniform")
    J, sJ = col(r["J"]), col(r["sigma_J"])
    dev = np.abs(J - d1o_half)
    assert dev.max() <= 2.0 * sJ.max() + 0.03 * d1o_half.max()
    assert r["converged"]
    assert r["n_iter"] < 100


# --------------------------------------------------------------------------
# 3: Ng acceleration of the MC lambda iteration
# --------------------------------------------------------------------------


def test_ng_reduces_iterations(d1o_om9):
    grid, box, hz, col = _slab()
    ke = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=1.0)
    common = dict(stage1="lambda", g=0.0, periodic_box=box, n_histories=100000,
                  n_boundary_histories=100000, n_batches=200, seed=16,
                  n_iter=300, tol=1e-8, crn=True, birth="uniform")
    rli = j_mc(grid, ke, 0.9 * ke, np.zeros(grid.n_cells), bc, **common)
    rng_ = j_mc(grid, ke, 0.9 * ke, np.zeros(grid.n_cells), bc, ng=True, **common)
    assert rli["converged"] and rng_["converged"]
    assert rng_["n_iter"] < 0.6 * rli["n_iter"]
    Jng, Jli = col(rng_["J"]), col(rli["J"])
    sig = np.hypot(col(rng_["sigma_J"]), col(rli["sigma_J"]))
    assert np.abs(Jng - Jli).max() <= 2.0 * sig.max() + 1e-3
    assert np.abs(Jng - d1o_om9).max() <= 2.0 * col(rng_["sigma_J"]).max() + 0.04


def test_ng_requires_crn():
    grid, box, hz, col = _slab(nz=8, nxy=2)
    ke = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=1.0)
    with pytest.raises(ValueError, match="crn"):
        j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, periodic_box=box,
             stage1="lambda", ng=True, n_histories=1000, n_iter=5)


def test_crn_requires_uniform_birth():
    grid, box, hz, col = _slab(nz=8, nxy=2)
    ke = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=1.0)
    with pytest.raises(ValueError, match="birth"):
        j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, periodic_box=box,
             stage1="lambda", crn=True, birth="source", n_histories=1000,
             n_iter=5)


# --------------------------------------------------------------------------
# 4: global energy closure (anchor 4 companion)
# --------------------------------------------------------------------------


def test_global_energy_closure_slab():
    """
    Total absorption 4 pi kappa_a int J dA dz must equal the net downward
    flux Fz(top face) - Fz(bottom face) (per unit area) — convergence of the
    field enters through J, consistency of the track-length flux tally
    through F, so this closes anchor 4 for MC stage 1.
    """
    grid, box, hz, col = _slab()
    omega = 0.5
    ke = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=1.0)
    r = j_mc(grid, ke, omega * ke, np.zeros(grid.n_cells), bc, g=0.0,
             periodic_box=box, stage1="lambda", n_histories=200000,
             n_boundary_histories=200000, n_batches=200, seed=17,
             n_iter=200, tol=1e-8)
    J = col(r["J"])
    Fz = col(r["F"])[:, 2]
    absorbed = 4.0 * np.pi * (1.0 - omega) * J.sum() * hz   # per unit area
    # cell-center fluxes extrapolated to the faces (2nd order)
    Ftop = 1.5 * Fz[-1] - 0.5 * Fz[-2]
    Fbot = 1.5 * Fz[0] - 0.5 * Fz[1]
    net = Fbot - Ftop                                       # positive = deposited
    assert abs(net - absorbed) < 0.05 * absorbed


# --------------------------------------------------------------------------
# 5: direct beam as particle stream
# --------------------------------------------------------------------------


def _beam_case():
    x = np.linspace(0.5, 4.5, 5)
    z = np.linspace(0.5, 9.5, 19)
    grid = unigrid(x, x.copy(), z)
    box = cuboid(0.0, 5.0, 0.0, 5.0, grid.box[2][0], grid.box[2][1])
    sun = np.array([0.3, -0.2, -np.sqrt(1.0 - 0.09 - 0.04)])
    sun = sun / np.linalg.norm(sun)
    return grid, box, sun


def test_beam_pure_absorption_direct_tally():
    """
    Pure absorber, beam only: the explicit variant's separate unscattered-
    beam tally must reproduce F0 exp(-tau_b) / 4 pi, and the diffuse J must
    be exactly zero (no children without scattering).
    """
    grid, box, sun = _beam_case()
    ke = np.full(grid.n_cells, 0.5)
    beam = {"omega_sun": sun, "F0": 2.0}
    r = j_mc(grid, ke, np.zeros(grid.n_cells), np.zeros(grid.n_cells),
             boundary_conditions(), g=0.0, periodic_box=box, stage1="explicit",
             beam=beam, n_beam_histories=200000, n_histories=0,
             n_boundary_histories=0, n_batches=100, seed=18)
    tau = beam_optical_depth(grid, sun, ke, periodic_box=box)
    Jan = 2.0 * np.exp(-tau) / (4.0 * np.pi)
    dev = np.abs(r["J_total"] - Jan)
    assert dev.max() <= 3.0 * r["sigma_J_total"].max()
    assert r["J"].max() == 0.0


def test_beam_lambda_matches_explicit():
    """
    Scattering beam case: the lambda variant's diffuse J (bfs first-scatter
    particle stream + omega*J iterations) must agree with the explicit
    variant's diffuse J within combined statistics, and with the SC solver
    within the 42-ordinate angular resolution allowance.
    """
    grid, box, sun = _beam_case()
    ke = np.full(grid.n_cells, 0.5)
    omega = 0.8
    beam = {"omega_sun": sun, "F0": 2.0}
    common = dict(g=0.0, periodic_box=box, n_batches=100)
    rx = j_mc(grid, ke, omega * ke, np.zeros(grid.n_cells),
              boundary_conditions(), stage1="explicit", beam=beam,
              n_beam_histories=200000, n_histories=0, n_boundary_histories=0,
              seed=19, **common)
    rl = j_mc(grid, ke, omega * ke, np.zeros(grid.n_cells),
              boundary_conditions(), stage1="lambda", beam=beam,
              n_beam_histories=100000, n_histories=50000,
              n_boundary_histories=0, seed=20, n_iter=120, tol=1e-7, **common)
    dev = np.abs(rx["J"] - rl["J"])
    sig = np.hypot(rx["sigma_J"], rl["sigma_J"])
    # batch sigma underestimates the per-iteration fluctuation of a lambda
    # iterate (accumulated source noise, amplified ~ 1/(1 - omega)), so the
    # pointwise band gets a 4 sigma allowance; outliers must stay bounded
    assert np.count_nonzero(dev <= 4.0 * sig) >= 0.9 * grid.n_cells
    assert dev.max() <= 8.0 * sig.max()

    dg = icosahedric_grid(n_refine=1)
    sc = lambda_iteration(grid, dg, ke, omega * ke, np.zeros(grid.n_cells),
                          boundary_conditions(), g=0.0, periodic_box=box,
                          beam=beam, n_iter=500, tol=1e-9)
    # MC-vs-SC difference = statistical + 42-ordinate angular error
    dev_sc = np.abs(rl["J"] - sc["J"])
    assert dev_sc.max() <= 8.0 * sig.max() + 0.05 * sc["J"].max()


# --------------------------------------------------------------------------
# 2 + 7: gaussian cloud — variant agreement, octree grids and modes
# --------------------------------------------------------------------------


def _cloud_case(nz_side=25):
    case = gaussian_cloud(omega_cap=0.9)
    box = case["box"]
    x = np.linspace(-5.0, 5.0, nz_side)
    grid = unigrid(x, x.copy(), np.linspace(0.5, 9.5, 19))
    ke = case["kappa_e"](grid.centers[:, 0], grid.centers[:, 1], grid.centers[:, 2])
    ks = case["kappa_s"](grid.centers[:, 0], grid.centers[:, 1], grid.centers[:, 2])
    g = case["g"](grid.centers[:, 0], grid.centers[:, 1], grid.centers[:, 2])
    return case, box, grid, ke, ks, g


def test_cloud_lambda_matches_explicit():
    """lambda (fresh-sampling default) vs explicit MC on the cloud: 90% of
    cells inside combined 2 sigma (noise floor of plain LI at omega ~ 0.9)."""
    case, box, grid, ke, ks, g = _cloud_case()
    bc = boundary_conditions(top_I=1.0)
    common = dict(g=g, periodic_box=box, n_batches=100,
                  n_histories=60000, n_boundary_histories=60000)
    rl = j_mc(grid, ke, ks, np.zeros(grid.n_cells), bc, stage1="lambda",
              seed=21, n_iter=150, tol=1e-7, **common)
    rx = j_mc(grid, ke, ks, np.zeros(grid.n_cells), bc, stage1="explicit",
              seed=22, **common)
    dev = np.abs(rl["J"] - rx["J"])
    sig = np.hypot(rl["sigma_J"], rx["sigma_J"])
    assert np.count_nonzero(dev <= 2.0 * sig) >= 0.9 * grid.n_cells
    assert dev.max() <= 6.0 * sig.max()     # no structural outliers


def test_cloud_octree_modes_agree():
    """octree cloud run in cell and global majorant mode agree statistically,
    and the global mode produces the expected null collisions."""
    case, box, grid, ke, ks, g = _cloud_case()
    oct_ = octree_grid(box, h0=2.5)
    oct_.refine(case["kappa_e"], tau_max=1.0, eps_grad=0.5, max_depth=3)
    oct_.balance_21(max_depth=3)
    oct_.finalize()
    oct_.sample_fields(kappa_e=case["kappa_e"])
    ks_o = case["kappa_s"](oct_.centers[:, 0], oct_.centers[:, 1], oct_.centers[:, 2])
    g_o = case["g"](oct_.centers[:, 0], oct_.centers[:, 1], oct_.centers[:, 2])
    bc = boundary_conditions(top_I=1.0)
    common = dict(g=g_o, periodic_box=box, n_batches=100,
                  n_histories=40000, n_boundary_histories=40000)
    rc = j_mc(oct_, oct_.kappa_e, ks_o, np.zeros(oct_.n_leaf), bc,
              stage1="lambda", mode="cell", seed=23, n_iter=120, tol=1e-7,
              **common)
    rg = j_mc(oct_, oct_.kappa_e, ks_o, np.zeros(oct_.n_leaf), bc,
              stage1="lambda", mode="global", seed=23, n_iter=120, tol=1e-7,
              **common)
    assert rg["n_null"] > 10 * max(rc["n_null"], 1)
    dev = np.abs(rc["J"] - rg["J"])
    sig = np.hypot(rc["sigma_J"], rg["sigma_J"])
    assert np.count_nonzero(dev <= 2.0 * sig) >= 0.95 * oct_.n_leaf


# --------------------------------------------------------------------------
# 8: traceability + guards
# --------------------------------------------------------------------------


def test_traceability_and_guards():
    grid, box, hz, col = _slab(nz=8, nxy=2)
    ke = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=1.0)
    r = j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, g=0.0,
             periodic_box=box, stage1="lambda", n_histories=5000,
             n_boundary_histories=5000, n_batches=25, seed=25, n_iter=60,
             tol=1e-6)
    for key, want in (("seed", 25), ("stage1", "lambda"), ("mode", "cell"),
                      ("n_batches", 25), ("n_histories", 5000)):
        assert r[key] == want
    assert r["n_boundary_histories"] == 5000
    assert len(r["history"]) == r["n_iter"]
    assert len(r["s_max_history"]) == r["n_iter"]

    class ground_bc(object):
        top_I = 1.0
        bottom_I = 0.0
        side_I = 0.0
        ground = object()          # non-None -> must raise

    with pytest.raises(NotImplementedError):
        j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), ground_bc(), g=0.0,
             periodic_box=box, stage1="explicit", n_histories=1000)

    with pytest.raises(ValueError, match="stage1"):
        j_mc(grid, ke, 0.5 * ke, np.zeros(grid.n_cells), bc, periodic_box=box,
             stage1="adjoint", n_histories=100)
