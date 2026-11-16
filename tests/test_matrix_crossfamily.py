# -*- coding: utf-8 -*-
"""
T2 of the solver-matrix plan (plan/solver-test-matrix-plan.md section 3):
**zero-noise cross-family identity** -- the sharpest tripwire in the suite.

``rt3d.mc.forward.radiance_from_source(estimator="deterministic")`` evaluates
the *same* line integral that the SC octree sweep runs: for every centre and
ordinate, march backwards along the characteristic and integrate
``kappa_e S exp(-tau)`` plus the boundary inflow.  On an octree the two are
therefore the same algorithm, so feeding the MC estimator the deterministic
solver's own converged source must reproduce the solver's intensity to
round-off.  This exercises, in one deterministic assertion costing seconds:

* the boundary-inflow convention (which face, which inflow, beam vs diffuse),
* the direct-beam optical depth bookkeeping on both grids,
* the δ-M extinction split (the stage-2 march must use
  ``result["kappa_e_delta"]``, not the physical ``kappa_e``),
* the octree exact-march ordering and segment bookkeeping.

The **negative control** (finding F2) feeds the *physical* ``kappa_e`` to a
δ-M run and requires the identity to *fail* by a large margin, proving that
the test can fail.

On the **unigrid** the identity is necessarily a band, not an equality: the
chain sweeps integrate centre-to-centre through the lattice stencil, the SC
sweeps interpolate a four-point bilinear stencil, and the MC march is a DDA
segment integral -- three different discretizations of the same RTE.  The
measured spread is pinned two-sided so a change of the stage-2 march geometry
is visible instead of hiding under a one-sided "< 1" bound.
"""

import numpy as np
import pytest

from matrix_cases import (
    cloud_octree,
    cloud_unigrid,
    direction_grid,
    solve_cloud,
)
from matrix_stats import assert_banded, rel_max_dev

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.geometry import ground_surface
from rt3d.mc.backward import backward_detector_mc
from rt3d.mc.forward import j_mc, radiance_from_source
from rt3d.short_characteristics import sweep_all_sc
from rt3d.transport import boundary_conditions, sweep_all, sweep_all_octree_exact

# solver convergence for the identity runs: the stage-2 march uses the
# solver's *returned* source, so the residual shows up as a (pinned) gap
# against the solver's own I, while the same-source comparison stays exact
SOLVE_TOL = 1e-8

T2_CONFIGS = [
    # (grid, delta_M, beam) -- the full δ-M sweep on d2, spot checks on d3
    ("oct2", 0, True),
    ("oct2", 1, True),
    ("oct2", 2, True),
    ("oct2", 1, False),
    ("oct3", 1, True),
    ("oct3", 0, False),
]


@pytest.fixture(scope="module")
def oct2():
    """Depth-2 cloud octree (6 952 leaves)."""
    return cloud_octree(2.0, 2)


@pytest.fixture(scope="module")
def oct3():
    """Depth-3 cloud octree (25 264 leaves): the F2 probe grid."""
    return cloud_octree(2.0, 3)


# =============================================================================
# octree: the identity
# =============================================================================

@pytest.mark.parametrize("grid_name,delta_M,beam", T2_CONFIGS,
                         ids=[f"{g}-dM{d}-beam{int(b)}"
                              for g, d, b in T2_CONFIGS])
def test_octree_deterministic_identity(grid_name, delta_M, beam, request):
    """
    MC deterministic stage 2 == SC exact octree sweep for identical inputs.

    ``result["S"]`` is the solver's converged source (including the beam
    first-scatter term and the δ-M scaling); ``result["kappa_e_delta"]`` is
    the reduced extinction that source was converged with.  Feeding both to
    the MC estimator must reproduce a fresh SC sweep at round-off (measured
    <= 1.2e-12, i.e. the two integrations differ only in summation order).
    """
    grid = request.getfixturevalue(grid_name)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.5)
    box = gaussian_cloud(omega_cap=0.9)["box"]
    res = solve_cloud(grid, ng=True, delta_M=delta_M, beam=beam,
                      n_iter=100, tol=SOLVE_TOL)
    kappa = res["kappa_e_delta"]
    S = res["S"]

    mc = radiance_from_source(grid, direction_grid(1), kappa, S, bc,
                              periodic_box=box, estimator="deterministic")
    assert mc["estimator"] == "deterministic"
    assert not np.any(mc["sigma_I"]), "the deterministic estimator is exact"

    sc_I, _ = sweep_all_octree_exact(grid, direction_grid(1), kappa, S, bc,
                                     periodic_box=box)
    assert_banded(rel_max_dev(mc["I"], sc_I), 0.0, 1e-10,
                  f"MC deterministic vs SC sweep, same source ({grid_name}, "
                  f"delta_M={delta_M}, beam={beam})")

    # the solver's own I differs only by the convergence residual (tol)
    assert_banded(rel_max_dev(mc["I"], res["I"]), 0.0, 1e-6,
                  f"MC deterministic vs solver I ({grid_name}, "
                  f"delta_M={delta_M}, beam={beam})")


def test_identity_needs_the_delta_m_extinction(oct2):
    """
    Negative control (finding F2): under δ-M the stage-2 march must use the
    *reduced* extinction.  Physical ``kappa_e`` breaks the identity by 5-10 %
    (measured 9.5e-2 at M=1, 5.8e-2 at M=2) -- a drift that a 1e-8 identity
    would otherwise silently tolerate.
    """
    bc = boundary_conditions(top_I=1.0, bottom_I=0.5)
    box = gaussian_cloud(omega_cap=0.9)["box"]
    kappa_phys = cloud_octree(2.0, 2).kappa_e

    for delta_M in (1, 2):
        res = solve_cloud(oct2, ng=True, delta_M=delta_M, n_iter=100,
                          tol=SOLVE_TOL)
        right = radiance_from_source(
            oct2, direction_grid(1), res["kappa_e_delta"], res["S"], bc,
            periodic_box=box, estimator="deterministic")["I"]
        wrong = radiance_from_source(
            oct2, direction_grid(1), kappa_phys, res["S"], bc,
            periodic_box=box, estimator="deterministic")["I"]
        dev = rel_max_dev(wrong, right)
        assert_banded(dev, 1e-2, np.inf,
                      f"negative control: physical kappa_e under delta-M"
                      f"{delta_M} must break the identity")


def test_deterministic_estimator_ignores_sampling_controls(oct2):
    """
    The deterministic estimator is a line integral: ``mode`` (Woodcock
    majorant), ``jit``, ``seed`` and ``n_histories`` must not change one bit.
    """
    bc = boundary_conditions(top_I=1.0, bottom_I=0.5)
    box = gaussian_cloud(omega_cap=0.9)["box"]
    res = solve_cloud(oct2, ng=True, delta_M=1, n_iter=100, tol=SOLVE_TOL)
    kw = dict(periodic_box=box, estimator="deterministic")
    base = radiance_from_source(oct2, direction_grid(1),
                                res["kappa_e_delta"], res["S"], bc, **kw)
    for extra in (dict(mode="global"), dict(jit=True), dict(seed=12345),
                  dict(n_histories=3, n_batches=2)):
        other = radiance_from_source(oct2, direction_grid(1),
                                     res["kappa_e_delta"], res["S"], bc,
                                     **kw, **extra)
        assert np.array_equal(other["I"], base["I"]), (
            f"deterministic estimator changed with {extra}")


# =============================================================================
# unigrid: a pinned band, not an identity
# =============================================================================

def test_unigrid_deterministic_band():
    """
    The three unigrid discretizations of the same RTE, measured spread.

    Same converged ``S`` fed to (a) the chain sweep, (b) the SC sweep, (c) the
    MC DDA line integral.  Measured deviations of (c) from (a) 0.929 and from
    (b) 0.904 -- the same order as the chain-vs-SC field spread pinned in T1,
    so this is a discretization property of the unigrid, not a stage-2 error.
    The two-sided bands make a change of the march geometry visible.
    """
    u24 = cloud_unigrid(24)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.5)
    box = gaussian_cloud(omega_cap=0.9)["box"]
    res = solve_cloud(u24, ng=True, delta_M=1, n_iter=100, tol=SOLVE_TOL)
    kappa, S = res["kappa_e_delta"], res["S"]
    dg = direction_grid(1)

    mc = radiance_from_source(u24, dg, kappa, S, bc, periodic_box=box,
                              estimator="deterministic")["I"]
    I_chain, _ = sweep_all(u24, dg, kappa, S, bc, periodic_box=box)
    I_sc, _ = sweep_all_sc(u24, dg, kappa, S, bc, periodic_box=box)

    assert_banded(rel_max_dev(mc, I_chain), 0.90, 0.96,
                  "unigrid MC deterministic vs chain sweep")
    assert_banded(rel_max_dev(mc, I_sc), 0.87, 0.94,
                  "unigrid MC deterministic vs SC sweep")
    # the MC field must still be a plausible radiance field, not a shifted one
    assert np.all(np.isfinite(mc)) and mc.min() >= 0.0
    assert mc.max() <= 1.0 + 1e-12


# =============================================================================
# M6 (plan section 7): the flat Lambertian ground is supported by every MC
# entry point -- the old "raises NotImplementedError" guard is replaced by a
# positive smoke test (the physics itself is pinned analytically in
# tests/test_mc_ground.py; here only the plumbing across the octree path).
# =============================================================================


def test_ground_stage2_march_identity(oct2):
    """
    Finding F13 guard: on the **octree all-sky ground case** the stage-2
    march fed the solver's own (delta-M) source and the SC-consistent
    ground radiance must reproduce the solver's ``I`` to round-off -- the
    F2 identity extended over the ground inflow.  Before the periodic-wrap
    fix in the march's ground-column lookup, wrapped rays read clamped
    boundary columns and this deviated by 1.2e-1 (max) -- an inhomogeneous
    ground field (albedo + beam over the cloud) is what exposes it; a
    constant ground radiance hides the bug completely.
    """
    from rt3d.mc.ground import ground_beam_irradiance, ground_radiance
    from rt3d.transport import (beam_optical_depth_octree, bottom_mask,
                                downward_irradiance)

    from matrix_cases import cloud_beam

    box = gaussian_cloud(omega_cap=0.9)["box"]
    dg = direction_grid(1)
    res = solve_cloud(oct2, bc="ground", albedo=0.3, emissivity=1.0, B_g=1.0,
                      beam=True, ng=True, delta_M=1, n_iter=400, tol=1e-8)
    assert res["converged"]
    mb = bottom_mask(oct2)
    beam = cloud_beam()
    F_down = downward_irradiance(res["I"][:, dg.down_idx], dg)[mb]
    tau_b = beam_optical_depth_octree(oct2, beam["omega_sun"],
                                      oct2.kappa_e, periodic_box=box)
    beam_irr = ground_beam_irradiance(oct2, beam, oct2.kappa_e, tau_b,
                                      periodic_box=box)
    I_g = ground_radiance(oct2, bc_ground_for(albedo=0.3, B_g=1.0).ground,
                          F_down, beam_irr=beam_irr)
    mc = radiance_from_source(oct2, dg, res["kappa_e_delta"], res["S"],
                              bc_ground_for(albedo=0.3, B_g=1.0),
                              periodic_box=box, estimator="deterministic",
                              ground_I=I_g)
    assert_banded(rel_max_dev(mc["I"], res["I"]), 0.0, 1e-6,
                  "ground stage-2 march vs solver I (octree, F13 guard)")


def bc_ground_for(albedo=0.3, emissivity=1.0, B_g=1.0):
    return boundary_conditions(
        top_I=1.0,
        ground=ground_surface(albedo=albedo, emissivity=emissivity, B_g=B_g))


def test_mc_ground_accepted_j_mc(oct2):
    bc = boundary_conditions(
        top_I=1.0, ground=ground_surface(albedo=0.3, emissivity=1.0, B_g=1.0))
    box = gaussian_cloud(omega_cap=0.9)["box"]
    r = j_mc(oct2, oct2.kappa_e, oct2.kappa_s, np.full(oct2.n_leaf, 0.5), bc,
             g=oct2.g, periodic_box=box, stage1="explicit", n_histories=2000,
             n_boundary_histories=2000, n_ground_histories=2000,
             n_batches=20, seed=3, jit=True)
    assert np.isfinite(r["J"]).all() and r["J"].min() >= 0.0
    assert r["n_ground_histories"] == 2000


def test_mc_ground_accepted_stage2(oct2):
    from rt3d.mc.ground import ground_radiance

    # S = 0, no face inflows, only the emitting ground: a stage-2 march
    # that actually reads ``ground_I`` must light up the upward ordinates
    # and leave the downward ones dark.  (With S = top_I = 1 every path
    # would read exactly 1 *with or without* the ground -- that choice is
    # blind to this wiring, hence the dark-medium setup here.)
    bc = boundary_conditions(
        top_I=0.0, ground=ground_surface(albedo=0.0, emissivity=1.0, B_g=1.0))
    box = gaussian_cloud(omega_cap=0.9)["box"]
    I_g = ground_radiance(oct2, bc.ground)          # fixed part only, F_down=0
    mc = radiance_from_source(oct2, direction_grid(1), oct2.kappa_e,
                              np.zeros(oct2.n_leaf), bc, periodic_box=box,
                              estimator="deterministic", ground_I=I_g)
    assert np.isfinite(mc["I"]).all()
    dg = direction_grid(1)
    horiz2 = dg.directions[:, 0] ** 2 + dg.directions[:, 1] ** 2
    up = int(np.argmin(np.where(dg.directions[:, 2] <= 0, np.inf, horiz2)))
    down = int(np.argmin(np.where(dg.directions[:, 2] >= 0, np.inf, horiz2)))
    assert mc["I"][:, up].max() > 0.9, (
        "stage-2 ground inflow missing: no optically-thin column carries the "
        "ground emission to the top")
    assert mc["I"][:, up].min() > 0.0, (
        "stage-2 ground inflow produced a negative upward radiance")
    assert mc["I"][:, down].max() < 1e-9, (
        "stage-2 ground inflow leaked into downward ordinates")


def test_mc_ground_accepted_backward_detector(oct2):
    bc = boundary_conditions(
        top_I=1.0, ground=ground_surface(albedo=0.3, emissivity=1.0, B_g=1.0))
    box = gaussian_cloud(omega_cap=0.9)["box"]
    d = backward_detector_mc(
        oct2, oct2.kappa_e, oct2.kappa_s, np.zeros(oct2.n_leaf), bc,
        {"point": (0.0, 0.0, 1.0),
         "omega0": np.array([0.0, 0.0, -1.0]), "cone": 0.3,
         "n_seeds": 400},
        periodic_box=box, jit=False,
    )
    assert np.isfinite(d["I"]).all() and d["I"] >= 0.0
    # the ground sits directly below the detector cone: with B = 0 and no
    # beam, every score is eps_g B_g attenuated on the path -> I in (0, 1]
    assert d["I"] <= 1.0 + 1e-12
