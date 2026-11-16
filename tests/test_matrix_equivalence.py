# -*- coding: utf-8 -*-
"""
T1 of the solver-matrix plan (plan/solver-test-matrix-plan.md section 3):
**deterministic variant equivalence**.

One converged solve per solver variant on the *same* case, grid and boundary
conditions, then differenced pairwise at the solver level (not at the sweep
level, which ``tests/test_transport.py``, ``test_short_characteristics.py``
and ``test_jit.py`` already cover):

* ``jit`` (octree fused march vs NumPy reference march) and ``n_jobs``
  (threaded per-ordinate sweeps): agreement at round-off.
* δ-M 0/1/2: the explicit ``-omega g**M I`` term is discretized with the
  same short-characteristics stencil, so the converged fields differ by an
  O(tau_cell) band -- pinned from measurement, not assumed zero.
* Ng acceleration vs plain lambda iteration: same fixed point, fewer
  iterations.
* ``chain`` vs ``sc`` transport: two independent discretizations of the same
  RTE; the band is the documented angular stencil error of oblique ordinates
  and is *not* a solver defect (it shrinks with the cell size, T5 ladder).
* ground on/off, beam on/off, periodic vs non-periodic: structure checks,
  linearity in the boundary drive, and the exact "vacuum ground" identity
  (a zero-albedo, zero-emissivity ground is bit-identical to
  ``bottom_I = 0``).
* guards: ``transport="sc"`` on an octree raises ``ValueError``, a variable
  ground height raises ``NotImplementedError``, vacuum cells stay finite.

All bands were measured on 2026-11-16 (Apple M4 Pro, 42 ordinates,
``omega_cap = 0.9``, ``tol = 1e-6``) and are recorded in
plan/solver-test-matrix-plan.md section 2.  A failing band means either a
real regression or a deliberate accuracy change that must re-pin the band.
"""

import numpy as np
import pytest

from matrix_cases import (
    cloud_beam,
    cloud_case,
    cloud_fields,
    cloud_octree,
    cloud_unigrid,
    common_bc,
    direction_grid,
    solve_cloud,
)
from matrix_stats import assert_banded, rel_max_dev

from rt3d.geometry import ground_surface
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions

TOL = 1e-6
ZERO_BC = {"bottom_I": 0.0}

@pytest.fixture(scope="module")
def u24():
    """The 24^3 unigrid of the shared cloud case."""
    return cloud_unigrid(24)


@pytest.fixture(scope="module")
def oct2():
    """Depth-2 cloud octree (6 952 leaves): enough modes, cheap to solve."""
    return cloud_octree(2.0, 2)


# =============================================================================
# numerics: jit and threading must not change the answer
# =============================================================================

def test_n_jobs_is_bit_identical(u24):
    """Threaded per-ordinate sweeps reproduce the sequential field exactly."""
    ref = solve_cloud(u24, ng=True, delta_M=1)
    par = solve_cloud(u24, ng=True, delta_M=1, n_jobs=4)
    assert par["n_iter"] == ref["n_iter"]
    assert np.array_equal(par["I"], ref["I"]), "n_jobs=4 changed the intensity"
    assert np.array_equal(par["J"], ref["J"])


def test_octree_jit_matches_numpy_reference(oct2):
    """
    Fused numba march+sweep vs the NumPy reference march on the octree.

    The two paths accumulate the same segments in a different order, so the
    agreement is round-off (measured 1.0e-12 relative), not bit equality.
    """
    fused = solve_cloud(oct2, jit=None, ng=True, delta_M=1)
    numpy = solve_cloud(oct2, jit=False, ng=True, delta_M=1)
    assert fused["n_iter"] == numpy["n_iter"]
    assert_banded(rel_max_dev(fused["I"], numpy["I"]), 0.0, 1e-9,
                  "octree jit I deviation (fused vs NumPy march)")
    assert_banded(rel_max_dev(fused["J"], numpy["J"]), 0.0, 1e-9,
                  "octree jit J deviation (fused vs NumPy march)")


# =============================================================================
# delta-M: same fixed point up to the explicit-term discretization band
# =============================================================================

def test_delta_m_unigrid_bands(u24):
    """
    δ-M 1 and 2 vs δ-M 0 on u24^3.

    Measured: ΔJ/J = 8.4e-3 (M=1) and 8.5e-3 (M=2), ΔI/I = 4.1e-2 and
    3.1e-2 -- the O(tau_cell) explicit ``-omega g**M I`` term.  δ-M must also
    *save* iterations on this forward-peaked cloud.
    """
    ref = solve_cloud(u24, ng=True, delta_M=0)
    dm1 = solve_cloud(u24, ng=True, delta_M=1)
    dm2 = solve_cloud(u24, ng=True, delta_M=2)

    assert_banded(rel_max_dev(dm1["J"], ref["J"]), 0.0, 1.5e-2,
                  "delta-M1 vs M0 J (u24)")
    assert_banded(rel_max_dev(dm1["I"], ref["I"]), 0.0, 6e-2,
                  "delta-M1 vs M0 I (u24)")
    assert_banded(rel_max_dev(dm2["J"], ref["J"]), 0.0, 1.5e-2,
                  "delta-M2 vs M0 J (u24)")
    assert_banded(rel_max_dev(dm2["I"], ref["I"]), 0.0, 6e-2,
                  "delta-M2 vs M0 I (u24)")
    # M2 is a refinement of M1, so it sits between M1 and M0
    assert_banded(rel_max_dev(dm2["J"], dm1["J"]), 0.0, 1e-2,
                  "delta-M2 vs M1 J (u24)")
    # delta-M is what actually cuts the iteration count on this cloud:
    # measured 12 (M=1) and 16 (M=2) vs 34 sweeps (M=0)
    assert_banded(ref["n_iter"] / dm1["n_iter"], 2.0, np.inf,
                  "delta-M0 / delta-M1 iteration ratio (u24)")


def test_delta_m_octree_bands(oct2):
    """
    The same δ-M band on the octree, where the explicit term is integrated
    along exact characteristics (measured ΔJ/J = 1.3e-2, ΔI/I = 2.6e-2).
    """
    ref = solve_cloud(oct2, ng=True, delta_M=0)
    dm1 = solve_cloud(oct2, ng=True, delta_M=1)
    assert_banded(rel_max_dev(dm1["J"], ref["J"]), 0.0, 2.5e-2,
                  "delta-M1 vs M0 J (octree d2)")
    assert_banded(rel_max_dev(dm1["I"], ref["I"]), 0.0, 5e-2,
                  "delta-M1 vs M0 I (octree d2)")
    assert dm1["n_iter"] < ref["n_iter"]


# =============================================================================
# Ng acceleration
# =============================================================================

def test_ng_same_fixed_point(u24, oct2):
    """
    Ng 1D and plain lambda iteration converge to the same source field.

    The *fixed point* is the physics; the iteration count is an efficiency
    property and is asserted separately, because on this omega-capped cloud
    Ng is not a clear win (finding F6).
    """
    for grid, name in ((u24, "u24"), (oct2, "octree d2")):
        ng = solve_cloud(grid, ng=True, delta_M=0)
        li = solve_cloud(grid, ng=None, delta_M=0)
        assert_banded(rel_max_dev(ng["J"], li["J"]), 0.0, 1e-5,
                      f"Ng vs plain LI J ({name})")
        assert_banded(rel_max_dev(ng["I"], li["I"]), 0.0, 1e-5,
                      f"Ng vs plain LI I ({name})")
        assert ng["converged"] and li["converged"]


def test_ng_iteration_cost_is_bounded(u24, oct2):
    """
    Ng must not make the iteration *pathologically* slower (finding F6).

    Measured on the omega <= 0.9 cloud at tol = 1e-6: u24^3 charges
    57 (Ng) vs 34 (LI) sweeps -- a 1.7x penalty, because the spectral
    radius is already small there -- while the octree d2 saves a little
    (27 vs 30).  The bands below are "no regression" guards: they fail if
    the acceleration starts oscillating or stops being applied at all.
    """
    ng_u = solve_cloud(u24, ng=True, delta_M=0)
    li_u = solve_cloud(u24, ng=None, delta_M=0)
    assert_banded(ng_u["n_iter"] / li_u["n_iter"], 0.9, 2.0,
                  "Ng / plain-LI iteration ratio (u24)")

    ng_o = solve_cloud(oct2, ng=True, delta_M=0)
    li_o = solve_cloud(oct2, ng=None, delta_M=0)
    assert_banded(ng_o["n_iter"] / li_o["n_iter"], 0.5, 1.0,
                  "Ng / plain-LI iteration ratio (octree d2)")
    assert ng_o["n_iter"] < li_o["n_iter"]


# =============================================================================
# chain vs short characteristics
# =============================================================================

def test_chain_vs_sc_same_physics(u24):
    """
    Two independent unigrid transports of the *same* RTE.

    Measured ΔJ/J = 0.26 and ΔI/I = 0.93 on u24^3: the documented oblique-
    ordinate stencil/path error (rt3d/short_characteristics.py docstring),
    which is why the T5 ladder checks that both converge to the fine-octree
    reference instead of being forced to agree here.
    """
    chain = solve_cloud(u24, ng=True, delta_M=1)
    sc = solve_cloud(u24, ng=True, delta_M=1, transport="sc")
    assert_banded(rel_max_dev(sc["J"], chain["J"]), 0.0, 0.35,
                  "sc vs chain J (u24, common bc)")
    assert_banded(rel_max_dev(sc["I"], chain["I"]), 0.0, 1.2,
                  "sc vs chain I (u24, common bc)")
    # both conserve the same fixed-point residual and the same sign structure
    assert np.all(sc["J"] > 0.0) and np.all(chain["J"] > 0.0)


def test_chain_vs_sc_with_ground(u24):
    """The same band with the Lambertian ground active (measured 0.31/0.94)."""
    chain = solve_cloud(u24, bc="ground", albedo=0.3, ng=True, delta_M=1)
    sc = solve_cloud(u24, bc="ground", albedo=0.3, ng=True, delta_M=1,
                     transport="sc")
    assert_banded(rel_max_dev(sc["J"], chain["J"]), 0.0, 0.40,
                  "sc vs chain J (u24, Lambertian ground)")
    assert_banded(rel_max_dev(sc["I"], chain["I"]), 0.0, 1.2,
                  "sc vs chain I (u24, Lambertian ground)")
    # the ground must add radiation for every transport (compare against the
    # same ground switched to a black, non-emissive surface -- *not* against
    # common_bc, whose bottom_I = 0.5 inflow is a far stronger drive)
    black = solve_cloud(u24, bc="ground", albedo=0.0, emissivity=0.0,
                        B_g=0.0, ng=True, delta_M=1)
    assert chain["J"].max() > black["J"].max()
    assert sc["J"].max() > black["J"].max()


# =============================================================================
# ground, beam and periodic structure
# =============================================================================

def test_zero_ground_equals_face_inflow(u24):
    """albedo=0, emissivity=0 ground is bit-identical to bottom_I = 0."""
    with_ground = solve_cloud(u24, bc="ground", albedo=0.0, emissivity=0.0,
                              B_g=0.0, **ZERO_BC, ng=True, delta_M=1)
    with_face = solve_cloud(u24, top_I=1.0, bottom_I=0.0, ng=True, delta_M=1)
    assert np.array_equal(with_ground["I"], with_face["I"])
    assert np.array_equal(with_ground["J"], with_face["J"])


@pytest.mark.parametrize("transport", ["chain", "sc"])
def test_ground_reflection_is_linear_in_incident_field(u24, transport):
    """
    A Lambertian ground is a linear boundary operator: doubling the top inflow
    doubles the whole field when the ground is the only scatterer besides the
    medium (beam off, no thermal emission).
    """
    kw = dict(bc="ground", top_I=1.0, albedo=0.4, emissivity=0.0, beam=False,
              ng=True, delta_M=1, transport=transport)
    one = solve_cloud(u24, **kw)
    two = solve_cloud(u24, top_I=2.0, albedo=0.4, emissivity=0.0, beam=False,
                      ng=True, delta_M=1, transport=transport, bc="ground")
    assert np.allclose(two["J"], 2.0 * one["J"], rtol=0.0, atol=1e-10)
    assert np.allclose(two["I"], 2.0 * one["I"], rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("transport", ["chain", "sc"])
def test_ground_albedo_monotone(u24, transport):
    """More reflective ground -> more radiation everywhere (measured)."""
    kw = dict(bc="ground", ng=True, delta_M=1, transport=transport)
    a0 = solve_cloud(u24, albedo=0.0, **kw)
    a3 = solve_cloud(u24, albedo=0.3, **kw)
    a6 = solve_cloud(u24, albedo=0.6, **kw)
    assert a0["J"].max() < a3["J"].max() < a6["J"].max()
    # measured: 0.509 -> 0.609 -> 0.724 on u24 (chain)
    assert_banded(a3["J"].max() / a0["J"].max(), 1.1, 1.3,
                  f"Jmax(alb=0.3)/Jmax(alb=0) ({transport})")


def test_ground_emission_is_linear_in_B_g(u24):
    """Ground emission enters linearly (eps = 1, no albedo, beam off)."""
    kw = dict(bc="ground", albedo=0.0, emissivity=1.0, beam=False,
              ng=True, delta_M=1)
    b25 = solve_cloud(u24, B_g=0.25, **kw)
    b50 = solve_cloud(u24, B_g=0.50, **kw)
    b0 = solve_cloud(u24, B_g=0.0, **kw)
    # the emission increment must scale exactly linearly with B_g; the runs
    # stop at tol = 1e-6, so allow that residual in the difference
    assert np.allclose(b50["J"] - b0["J"], 2.0 * (b25["J"] - b0["J"]),
                       rtol=1e-4, atol=1e-12)


def test_beam_and_periodic_sides_change_the_field(u24):
    """
    Structure: the solar beam matters, and so do the periodic sides.

    Not an accuracy band -- this only guards against a silently dropped
    ``beam`` or ``periodic_box`` in the solver plumbing.
    """
    ref = solve_cloud(u24, ng=True, delta_M=1)
    no_beam = solve_cloud(u24, beam=False, ng=True, delta_M=1)
    no_box = solve_cloud(u24, periodic=False, ng=True, delta_M=1)
    assert_banded(rel_max_dev(no_beam["J"], ref["J"]), 1e-3, np.inf,
                  "beam on/off J (u24)")
    assert_banded(rel_max_dev(no_box["J"], ref["J"]), 0.1, np.inf,
                  "periodic vs non-periodic J (u24)")


# =============================================================================
# guards
# =============================================================================

def test_sc_transport_rejects_octree(oct2):
    """``transport="sc"`` is unigrid-only; the octree follows exact marches."""
    with pytest.raises(ValueError, match="only available on the unigrid"):
        solve_cloud(oct2, transport="sc")


def test_invalid_transport_rejected(u24):
    """An unknown transport name is rejected before any sweep."""
    with pytest.raises(ValueError, match="transport must be"):
        solve_cloud(u24, transport="upwind")


@pytest.mark.parametrize("grid_fixture", ["u24", "oct2"])
def test_variable_ground_not_implemented(grid_fixture, request):
    """The variable-height ground hook raises instead of silently ignoring it."""
    grid = request.getfixturevalue(grid_fixture)
    kappa_e, kappa_s, g, B = cloud_fields(grid)
    bc = boundary_conditions(
        top_I=1.0,
        ground=ground_surface(albedo=0.3,
                              height=lambda x, y: 0.1 * np.sin(x)),
    )
    with pytest.raises(NotImplementedError, match="variable ground"):
        lambda_iteration(grid, direction_grid(1), kappa_e, kappa_s, B, bc,
                         g=g, periodic_box=cloud_case()["box"], n_iter=2)


def test_vacuum_cells_stay_finite(u24):
    """
    ``kappa_e = 0`` cells: omega = 0, no 0/0, no NaN/Inf, and the beam is
    unattenuated through the vacuum (intensity may reach the top inflow).
    """
    kappa_e, kappa_s, g, B = cloud_fields(u24)
    vacuum = kappa_e < 1e-8
    assert np.count_nonzero(vacuum) > 100        # the cloud case has a vacuum
    kappa_e = np.where(vacuum, 0.0, kappa_e)
    kappa_s = np.where(vacuum, 0.0, kappa_s)

    r = lambda_iteration(
        u24, direction_grid(1), kappa_e, kappa_s, B, common_bc(),
        g=g, periodic_box=cloud_case()["box"], beam=cloud_beam(),
        n_iter=40, tol=TOL, ng=True, delta_M=1,
    )
    assert np.all(np.isfinite(r["I"]))
    assert np.all(np.isfinite(r["J"]))
    assert np.all(r["I"] >= 0.0)                 # no negative radiance
    assert r["I"].max() <= 1.0 + 1e-12           # never above the top inflow
