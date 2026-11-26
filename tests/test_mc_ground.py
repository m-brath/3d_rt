# -*- coding: utf-8 -*-
"""
Phase M6 (plan/solver-test-matrix-plan.md section 7): the Lambertian,
emissive ground in the **MC family**, closed against the analytic laws the
SC solver already implements (`rt3d/ground.py` -> wrapped by
`rt3d/mc/ground.py`).

The MC ground support has three, mutually consistent realisations, each
tested here against a closed-form answer:

* **fixed populations** (`j_mc`, both stage-1 variants): thermal emission
  ``eps_g B_g`` and the ground-reflected beam ``A_g F_beam / pi`` are seeded
  on the bottom face with cosine-weighted directions; the albedo part
  ``A_g F_down / pi`` of the *iterated* field is re-seeded per lambda
  iteration (the exact twin of the SC downward-sweep ordering in
  `rt3d.solver.lambda_iteration`).
* **analog reflection** (explicit stage 1 in `track_histories`): a diffuse
  history hitting the ground reflects with weight ``A_g`` and a cosine
  direction; the reflected *beam* is deliberately not analog (the beam is a
  fixed source population, mirroring the SC split).
* **adjoint scores** (`rt3d/mc/backward.py`): a backward ray reaching the
  ground scores ``eps_g B_g`` plus ``A_g F_beam / pi`` and then continues
  cosine-weighted with ``wt *= A_g``.

Conventions recorded here (measured, see plan section 7):

* MC ``J`` is the **angle-averaged** mean intensity ``int I dOmega / 4pi``:
  a Lambertian ground of radiance ``B_g`` in vacuum gives ``J = B_g / 2``
  and ``F_z = pi B_g`` (net upward flux ``pi B_g`` out of the ground);
* a Lambertian radiance ``L`` over the upward hemisphere gives ``J = L / 2``
  (so the reflected-beam field ``L = A F0 mu_sun / pi`` has ``J = A F0
  mu_sun / (2 pi)``);
* the direct-beam tally obeys ``4 pi int J_direct dV / V = F0 mu_sun``
  exactly (unscattered-beam bookkeeping, verified independently).
"""

import numpy as np
import pytest

from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, ground_surface, unigrid
from rt3d.mc.backward import backward_all_sky
from rt3d.mc.forward import j_mc, radiance_from_source
from rt3d.mc.ground import ground_radiance
from rt3d.solver import lambda_iteration
from rt3d.transport import bottom_mask, boundary_conditions, downward_irradiance

pytestmark = pytest.mark.mc

N, L = 16, 2.0


def _grid_box():
    v = np.linspace(0.0, L, N, endpoint=False) + L / N / 2
    return unigrid(v, v, v), cuboid(0.0, L, 0.0, L, 0.0, L)


# near-vacuum extinction: the majorant tracker needs kappa_e > 0, but 1e-9 in
# a 2 m box is optically thin to 2e-9 -- analytic vacuum comparisons apply
VAC = 1e-9


# ---------------------------------------------------------------- 1. emission
# vacuum + emissive ground: every upward ray carries B_g; the angle-averaged
# J is the half-space angular average B_g/2 and the net flux is pi B_g.


def test_ground_emission_vacuum_analytic():
    grid, box = _grid_box()
    nc = grid.n_cells
    bc = boundary_conditions(
        top_I=0.0, bottom_I=0.0,
        ground=ground_surface(albedo=0.0, emissivity=1.0, B_g=1.0))
    r = j_mc(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc), bc,
             periodic_box=box, stage1="explicit", n_histories=0,
             n_boundary_histories=0, n_ground_histories=40000,
             n_batches=50, seed=7, jit=True)
    J, sJ = r["J"], r["sigma_J"]
    assert abs(J.mean() - 0.5) < 3.0 * sJ.mean() / np.sqrt(nc) + 0.01, \
        f"vacuum ground emission J mean {J.mean():.4f} (exact 0.5)"
    assert abs(r["F"][:, 2].mean() - np.pi) < 0.01, \
        f"net upward flux {r['F'][:, 2].mean():.4f} (exact pi B_g)"


def test_ground_emission_lambda_equals_explicit():
    """The two stage-1 variants see the same fixed ground source: with the
    identical seeded ground population they must agree exactly (both re-weight
    / re-seed the *same* constant inflow, no scattering in vacuum)."""
    grid, box = _grid_box()
    nc = grid.n_cells
    bc = boundary_conditions(
        top_I=0.0, bottom_I=0.0,
        ground=ground_surface(albedo=0.0, emissivity=1.0, B_g=1.0))
    kw = dict(periodic_box=box, n_histories=0, n_boundary_histories=0,
              n_ground_histories=40000, n_batches=50, seed=7)
    rE = j_mc(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc), bc,
              stage1="explicit", jit=False, **kw)
    rL = j_mc(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc), bc,
              stage1="lambda", **kw)
    assert np.allclose(rE["J"], rL["J"], atol=1e-12), \
        "lambda variant changed the fixed ground population"


def test_ground_emission_jit_parity():
    grid, box = _grid_box()
    nc = grid.n_cells
    bc = boundary_conditions(
        top_I=0.0, bottom_I=0.0,
        ground=ground_surface(albedo=0.0, emissivity=1.0, B_g=1.0))
    kw = dict(periodic_box=box, n_histories=0, n_boundary_histories=0,
              n_ground_histories=40000, n_batches=50, seed=7)
    a = j_mc(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc), bc,
             stage1="explicit", jit=True, **kw)
    b = j_mc(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc), bc,
             stage1="explicit", jit=False, **kw)
    assert np.allclose(a["J"], b["J"], atol=1e-12), \
        "ground tracking jit/numpy paths differ beyond round-off"


def test_adjoint_ground_emission_cones_exact():
    """Adjoint all-sky at an interior point: in vacuum every upward cone
    connects straight to the emitting ground, score eps_g B_g = 1 exactly."""
    grid, box = _grid_box()
    nc = grid.n_cells
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(
        top_I=0.0, bottom_I=0.0,
        ground=ground_surface(albedo=0.0, emissivity=1.0, B_g=1.0))
    ab = backward_all_sky(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc),
                          bc, dg, np.array([1.0, 1.0, 1.0]), periodic_box=box,
                          n_seeds=4000, n_batches=50, seed=3)
    assert np.allclose(ab["I"], 1.0, atol=5e-3), \
        f"adjoint ground-emission cone scores {ab['I'].min():.4f}..{ab['I'].max():.4f}"


# --------------------------------------------- 2. Beer-Lambert + albedo ground
# pure absorber (kappa_e = 1, no scattering in the *volume*), top inflow 1,
# Lambertian ground of albedo A: the exact angle-averaged field at height z is
#   J(z) = 1/2 int_0^1 exp(-kappa (L-z)/mu) dmu          (direct, top I=1)
#        + 1/2 I_g   int_0^1 exp(-kappa  z   /mu) dmu    (ground radiance I_g)
# with I_g = A F_down^direct(ground)/pi and F_down^direct =
# 2 pi int_0^1 mu exp(-kappa L/mu) dmu (the ground sees only the direct beam,
# there is no volume source).  The MC explicit and lambda variants must land
# within their own statistical noise.


def _analytic_j(zc, kappa, I_g):
    nodes, wq = np.polynomial.legendre.leggauss(400)
    mu = 0.5 * (nodes + 1.0)
    wmu = 0.5 * wq
    Fd0 = 2 * np.pi * np.sum(wmu * mu * np.exp(-kappa * L / mu))
    direct = 0.5 * np.sum(wmu[None, :] *
                          np.exp(-kappa * (L - zc[:, None]) / mu[None, :]),
                          axis=1)
    grnd = 0.5 * I_g * np.sum(wmu[None, :] *
                              np.exp(-kappa * zc[:, None] / mu[None, :]),
                              axis=1)
    return direct + grnd, Fd0


@pytest.mark.parametrize("albedo", [0.0, 0.3, 0.9])
def test_pure_absorber_albedo_ground_vs_analytic(albedo):
    grid, box = _grid_box()
    nc = grid.n_cells
    kappa = 1.0
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0,
                             ground=ground_surface(albedo=albedo))
    kw = dict(periodic_box=box, n_histories=0, n_boundary_histories=40000,
              n_ground_histories=40000, n_batches=50, seed=11)
    rE = j_mc(grid, np.full(nc, kappa), np.zeros(nc), np.zeros(nc), bc,
              stage1="explicit", jit=True, **kw)
    rL = j_mc(grid, np.full(nc, kappa), np.zeros(nc), np.zeros(nc), bc,
              stage1="lambda", **kw)

    _Jan_direct, Fd0 = _analytic_j(grid.centers[:, 2], kappa, 0.0)
    I_g = albedo * Fd0 / np.pi
    Jan, _ = _analytic_j(grid.centers[:, 2], kappa, I_g)

    noise = np.sqrt(np.mean(rE["sigma_J"] ** 2))
    devE = np.sqrt(np.mean((rE["J"] - Jan) ** 2))
    devL = np.sqrt(np.mean((rL["J"] - Jan) ** 2))
    assert devE < 2.5 * noise, (
        f"explicit MC vs analytic albedo-{albedo} field dev {devE:.3e} "
        f"(noise {noise:.3e})")
    assert devL < 2.5 * noise, (
        f"lambda MC vs analytic albedo-{albedo} field dev {devL:.3e} "
        f"(noise {noise:.3e})")


@pytest.mark.parametrize("albedo", [0.3, 0.9])
def test_adjoint_cone_attenuation_over_albedo_ground(albedo):
    """Adjoint cones over the reflecting ground in the *pure-absorber*
    limit: upward radiance at height z is I_g exp(-kappa z / |mu|)."""
    grid, box = _grid_box()
    nc = grid.n_cells
    kappa = 1.0
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0,
                             ground=ground_surface(albedo=albedo))
    pt = np.array([1.0, 1.0, 1.0])
    _J, Fd0 = _analytic_j(grid.centers[:, 2], kappa, 0.0)
    I_g = albedo * Fd0 / np.pi
    ab = backward_all_sky(grid, np.full(nc, kappa), np.zeros(nc), np.zeros(nc),
                          bc, dg, pt, periodic_box=box, n_seeds=20000,
                          n_batches=50, seed=5)
    muc = np.abs(dg.directions[ab["up_idx"], 2])
    Iana = I_g * np.exp(-kappa * pt[2] / muc)
    dev = np.sqrt(np.mean((ab["I"] - Iana) ** 2))
    noise = np.sqrt(np.mean(ab["sigma_I"] ** 2))
    assert dev < 2.5 * noise, (
        f"adjoint cones over albedo-{albedo} ground dev {dev:.3e} "
        f"(noise {noise:.3e})")


# ----------------------------------------------------- 4. reflected beam law
# vacuum, vertical beam F0, albedo ground: the reflected-beam population is
# Lambertian with radiance A F0 mu_sun / pi, so J_diffuse = A F0 mu_sun/2pi
# and the direct tally stays exactly F0 mu_sun / 4pi on average; the analog
# chain must NOT double-count the beam reflection (n_ground_reflections = 0
# here: in vacuum nothing scatters into downward diffuse histories).


def test_reflected_beam_populations():
    grid, box = _grid_box()
    nc = grid.n_cells
    A = 0.3
    bc = boundary_conditions(top_I=0.0, bottom_I=0.0,
                             ground=ground_surface(albedo=A))
    beam = {"omega_sun": np.array([0.0, 0.0, -1.0]), "F0": 1.0}
    kw = dict(periodic_box=box, beam=beam, n_histories=0,
              n_boundary_histories=0, n_beam_histories=40000,
              n_ground_histories=40000, n_batches=50, seed=13)
    r = j_mc(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc), bc, g=0.0,
             stage1="explicit", jit=True, **kw)
    Jdiff, sJ = r["J"], r["sigma_J"]
    exact = A / (2 * np.pi)
    assert abs(Jdiff.mean() - exact) < 3.0 * sJ.mean() + 1e-3, (
        f"reflected-beam J {Jdiff.mean():.4f} (exact {exact:.4f})")
    exact_dir = 1.0 / (4 * np.pi)
    assert abs(r["J_direct"].mean() - exact_dir) < 3e-3, (
        f"direct tally {r['J_direct'].mean():.5f} (exact {exact_dir:.5f})")
    assert r["n_ground_reflections"] == 0, (
        "beam histories reflected analogously at the ground: the fixed "
        "reflected population and the analog chain would double-count")

    rL = j_mc(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc), bc, g=0.0,
              stage1="lambda", **kw)
    assert abs(rL["J_fixed"].mean() - exact) < 3e-3, (
        f"lambda-variant fixed ground source {rL['J_fixed'].mean():.4f} "
        f"(exact {exact:.4f})")


def test_adjoint_beam_score_over_ground():
    """Adjoint ray reaching the ground scores A F0 mu_sun / pi (the beam
    irradiance reflected diffusely); in vacuum the score is exact."""
    grid, box = _grid_box()
    nc = grid.n_cells
    A = 0.3
    bc = boundary_conditions(top_I=0.0, bottom_I=0.0,
                             ground=ground_surface(albedo=A))
    beam = {"omega_sun": np.array([0.0, 0.0, -1.0]), "F0": 1.0}
    dg = icosahedric_grid(n_refine=1)
    ab = backward_all_sky(grid, np.full(nc, VAC), np.zeros(nc), np.zeros(nc),
                          bc, dg, np.array([1.0, 1.0, 1.0]), beam=beam,
                          periodic_box=box, n_seeds=4000, n_batches=50, seed=9)
    Iana = np.full(ab["I"].size, A * 1.0 * 1.0 / np.pi)
    assert np.max(np.abs(ab["I"] - Iana)) < 1e-6, (
        f"adjoint beam score dev {np.max(np.abs(ab['I'] - Iana)):.3e}")


# ------------------------------------ 5. stage-2 ground_I vs the SC solver
# feeding the MC deterministic stage-2 march the SC-consistent ground radiance
# must reproduce the SC all-sky solve: the z-entry ordinate is a pure z path
# (no staircase, F11-free) and must match to ~1e-2 (discretisation of the
# ground column); oblique ordinates carry the recorded F11 march-vs-chain gap.


def test_stage2_ground_I_matches_sc_solver():
    grid, box = _grid_box()
    nc = grid.n_cells
    A = 0.5
    kappa = 1.0
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0,
                             ground=ground_surface(albedo=A))
    dg = icosahedric_grid(n_refine=1)
    ke, ks, B = np.full(nc, kappa), np.full(nc, 0.5), np.zeros(nc)
    res = lambda_iteration(grid, dg, ke, ks, B, bc, periodic_box=box,
                           n_iter=400, tol=1e-10, ng=True, delta_M=0)
    assert res["converged"]
    mb = bottom_mask(grid)
    F_down = downward_irradiance(res["I"][:, dg.down_idx], dg)[mb]
    I_g = ground_radiance(grid, bc.ground, F_down)
    mc = radiance_from_source(grid, dg, res["kappa_e_delta"], res["S"], bc,
                              periodic_box=box, estimator="deterministic",
                              ground_I=I_g)

    tmp = dg.directions[:, 0] ** 2 + dg.directions[:, 1] ** 2
    tmp[dg.directions[:, 2] >= 0] = np.inf
    iz = int(np.argmin(tmp))                      # downward z-entry ordinate
    dev_z = np.max(np.abs(mc["I"][:, iz] - res["I"][:, iz]))
    assert dev_z < 2e-2, f"z-entry ordinate dev {dev_z:.3e} (measured 7.5e-3)"
    dev_all = np.max(np.abs(mc["I"] - res["I"]))
    assert dev_all < 0.35, (
        f"march-vs-chain oblique gap moved: {dev_all:.3e} (F11 reference range, "
        "measured 2.8e-1)")


# ---------------------------------------------------------- variable height
# the only supported ground geometry stays the flat bottom face (plan M6):
# a variable-height ground must fail loudly, never silently ignore heights.


def test_variable_height_ground_still_rejected():
    grid, box = _grid_box()
    nc = grid.n_cells
    bc = boundary_conditions(top_I=1.0,
                             ground=ground_surface(albedo=0.3,
                                                   height=lambda x, y: 0.0 * x))
    with pytest.raises(NotImplementedError, match="variable ground surface"):
        j_mc(grid, np.full(nc, 1.0), np.zeros(nc), np.zeros(nc), bc,
             periodic_box=box, stage1="explicit", n_histories=0,
             n_boundary_histories=1000, n_ground_histories=1000,
             n_batches=10, seed=1)
