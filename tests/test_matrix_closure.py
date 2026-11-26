# -*- coding: utf-8 -*-
"""
T4 of the solver-matrix plan (plan/solver-test-matrix-plan.md section 3):
**global physics closures per family** -- AGENTS.md invariant 4, made
octree-capable by plan section 8 (closes finding F4: `energy_balance` used
to raise ``AttributeError`` on an octree).

Closure statement for a converged field:

    out_top + out_bottom  ==  int 4 pi kappa_a (B - J) dV

(periodic sides carry no net power).  What the residual actually measures
(plan finding F7, measured in this pass):

* the *geometry* is exact -- leaves tile every face once and the domain
  volume exactly, and a flux that is linear in z is closed to round-off;
* the **face extrapolation** is the dominant error for absorbing media: on
  an analytically exact pure-absorption field (no solver involved) the
  residual of `energy_balance` is +10.2 % (12^3), +3.97 % (24^3), +1.32 %
  (48^3), +0.39 % (96^3) -- a fitted slope of ~1.6 in h, limited by the
  boundary-layer kink of the intensity at the inflow faces.  Replacing the
  extrapolated face flux by the analytic one closes the same field to
  3e-10;
* for a **conservative medium** (``kappa_a = 0``, ``B = 0``) the volume side
  vanishes identically and ``F_z`` is constant in z, so the residual is a
  clean extrapolation number that *does* converge: u24 3.3e-5 -> u48
  6.6e-7, and on the octree it tracks the **base cell** ``h0``
  (1.4e-3 at h0 = 1.25 -> 1.3e-4 at h0 = 0.625) rather than ``max_depth``,
  because refinement adds leaves inside the cloud while the face leaves
  stay at base size;
* for an **absorbing** cloud the residual (~4 % on u24^3, ~1 % on the
  octree, and strongly transport-dependent: chain +4.2 %, sc -22 %) is
  dominated by the 42-ordinate angular/quadrature error of the solution and
  does not shrink u24 -> u48.  It is therefore recorded as a reference range and used as
  a *discriminating diagnostic*, never as an accuracy criterion.

Cross-family, the same closure is applied to Monte Carlo tallies as far as
their estimators allow: A2 (`radiance_from_source`) has per-ordinate ``I``
so its closure is directly comparable to SC; A1 (`j_mc`) has only ``J``/``F``
per cell, so it enters through the **absorbed power**
``int 4 pi kappa_a (B - J) dV`` with its batch sigma.
"""

import numpy as np
import pytest

from matrix_cases import (
    OCTREE_H0,
    cloud_beam,
    cloud_case,
    cloud_fields,
    cloud_octree,
    cloud_unigrid,
    common_bc,
    direction_grid,
    solve_cloud,
)
from matrix_stats import assert_banded, loglog_slope

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.diagnostics import (
    cell_volumes,
    energy_balance,
    face_powers,
    flux_divergence,
    heating_rate,
)
from rt3d.geometry import unigrid
from rt3d.mc.forward import j_mc, radiance_from_source
from rt3d.solver import lambda_iteration
from rt3d.source import mean_intensity
from rt3d.transport import sweep_all_octree_exact

# =============================================================================
# helpers / fixtures
# =============================================================================

# recorded closure residuals (relative, signed): label -> (measured, lo, hi)
CLOSURE_PINS = {
    "conservative-u24": (3.255e-05, 1e-5, 1e-4),
    "conservative-u48": (6.551e-07, 1e-8, 1e-5),
    "conservative-oct2-h0=1.25": (-1.403e-03, -3e-3, -5e-4),
    "conservative-oct3-h0=1.25": (-1.356e-03, -3e-3, -5e-4),
    "conservative-oct2-h0=0.625": (-1.336e-04, -5e-4, -3e-5),
    "absorbing-u24-chain": (4.227e-02, 0.030, 0.055),
    "absorbing-u24-sc": (-2.233e-01, -0.260, -0.190),
    "absorbing-oct2": (-1.044e-02, -0.015, -0.006),
    "absorbing-oct3": (-9.724e-03, -0.014, -0.006),
}


def absorbed_power(grid, J, kappa_a, B=0.0):
    """
    Volume-integrated heating rate ``int 4 pi kappa_a (B - J) dV``.

    Note the argument order of `rt3d.diagnostics.heating_rate`
    ``(kappa_a, J, B)`` -- it differs from the Monte Carlo estimators'
    ``rt3d.mc.estimators.heating_rate(kappa_a, B, J)``; both return
    ``4 pi kappa_a (B - J)``, but swapping them silently flips the sign.
    Negative = the radiation field is absorbed (the usual case for B = 0).
    """
    return float(np.sum(heating_rate(kappa_a, J, B) * cell_volumes(grid)))


@pytest.fixture(scope="module")
def u24():
    return cloud_unigrid(24)


@pytest.fixture(scope="module")
def oct2():
    """Octree, base cell 1.25 m, depth 2 (6 952 leaves)."""
    return cloud_octree(2.0, 2)


def _unigrid_n(n):
    box = cloud_case()["box"]
    xs = np.linspace(box.x0, box.x1, n + 1)[:-1] + (box.Lx / n) / 2
    zs = np.linspace(box.z0, box.z1, n + 1)[:-1] + (box.Lz / n) / 2
    return unigrid(xs, xs, zs)


def _conservative_solve(grid):
    """
    omega = 1 (kappa_s = kappa_e), B = 0, no beam, no ground -- the closure
    side that must vanish identically (memoized: these runs need ~100
    lambda iterations).
    """
    if not hasattr(_conservative_solve, "cache"):
        _conservative_solve.cache = {}
    if id(grid) not in _conservative_solve.cache:
        case = gaussian_cloud(omega_cap=1.0)
        box = case["box"]
        kappa_e = np.maximum(case["kappa_e"](*grid.centers.T), 0.0)
        _conservative_solve.cache[id(grid)] = lambda_iteration(
            grid, direction_grid(1), kappa_e, kappa_e.copy(),
            np.zeros(kappa_e.shape), common_bc(bottom_I=0.5),
            g=case["g"](*grid.centers.T), periodic_box=box,
            n_iter=400, tol=1e-8, ng=True, delta_M=0,
        )
    return _conservative_solve.cache[id(grid)]


def _closure_of_conservative(grid):
    return energy_balance(grid, _conservative_solve(grid)["I"],
                          direction_grid(1), np.zeros(grid.centers.shape[0]),
                          0.0)


# =============================================================================
# 1. geometry of the new closure (exact, zero noise)
# =============================================================================

@pytest.mark.parametrize("grid_fixture", ["u24", "oct2"])
def test_face_area_and_volume_bookkeeping(grid_fixture, request):
    """
    Boundary leaves tile each horizontal face exactly once and the leaves
    tile the domain volume -- on both grids.  A missed or double-counted
    face/leaf would corrupt every closure number, and it is checked here
    before any physics is involved.
    """
    grid = request.getfixturevalue(grid_fixture)
    box = cloud_case()["box"]
    dg = direction_grid(1)
    n_cells = cell_volumes(grid).size
    assert n_cells == (grid.n_leaf if hasattr(grid, "n_leaf")
                       else grid.n_cells)
    fp = face_powers(grid, np.ones((n_cells, dg.n_dir)), dg)
    area_top, area_bottom = fp["area"]
    A = box.Lx * box.Ly
    assert_banded(area_top.sum() / A, 1 - 1e-12, 1 + 1e-12,
                  f"{grid_fixture}: top face area / Lx*Ly")
    assert_banded(area_bottom.sum() / A, 1 - 1e-12, 1 + 1e-12,
                  f"{grid_fixture}: bottom face area / Lx*Ly")
    assert_banded(cell_volumes(grid).sum() / (A * box.Lz), 1 - 1e-12,
                  1 + 1e-12, f"{grid_fixture}: cell volume / box volume")


@pytest.mark.parametrize("grid_fixture", ["u24", "oct2"])
def test_face_powers_analytic_linear_flux(grid_fixture, request):
    """
    ``I = (1 + z)(1 + mu)`` has ``F_z(z) = (1 + z) sum_i w_i mu_i^2``, which
    is linear in z, so the two-point face extrapolation must be exact.
    """
    grid = request.getfixturevalue(grid_fixture)
    box = cloud_case()["box"]
    dg = direction_grid(1)
    mu = dg.directions[:, 2]
    k = float(np.sum(dg.weights * mu**2))
    z = grid.centers[:, 2]
    fp = face_powers(grid, (1.0 + z)[:, None] * (1.0 + mu)[None, :], dg)
    A = box.Lx * box.Ly
    assert_banded(fp["out_top"] / (A * k * (1.0 + box.z1)), 1 - 1e-12,
                  1 + 1e-12, f"{grid_fixture}: top face power (linear F_z)")
    assert_banded(fp["out_bottom"] / (-A * k * (1.0 + box.z0)), 1 - 1e-12,
                  1 + 1e-12, f"{grid_fixture}: bottom face power (linear F_z)")


# =============================================================================
# 2. conservative medium: the converging invariant (F7)
# =============================================================================

@pytest.mark.parametrize("label,grid_factory", [
    ("conservative-u24", lambda: cloud_unigrid(24)),
    ("conservative-oct2-h0=1.25", lambda: cloud_octree(2.0, 2)),
    ("conservative-oct3-h0=1.25", lambda: cloud_octree(2.0, 3)),
])
def test_closure_conservative_medium(label, grid_factory):
    """
    ``kappa_a = 0`` -> the heating side is exactly zero and ``F_z`` is
    constant in z, so the two face powers must cancel: the residual is a
    pure face-extrapolation number.
    """
    grid = grid_factory()
    res = _conservative_solve(grid)
    assert res["converged"]
    bal = _closure_of_conservative(grid)
    assert abs(bal["heating"]) < 1e-12            # kappa_a = 0 -> exact zero
    measured, lo, hi = CLOSURE_PINS[label]
    assert_banded(bal["relative"], lo, hi, f"closure residual, {label}")


@pytest.mark.slow
def test_closure_conservative_improves_with_base_cell():
    """
    Halving the cell size must reduce the conservative closure residual
    (plan section 8: "the closure residual must -> 0 as the grid refines,
    on both grids"), on the unigrid and in the octree's **base cell**.

    Measured: u24 3.26e-5 -> u48 6.55e-7 (fitted slope 5.9 in h);
    octree h0 1.25 -> 0.625 gives 1.40e-3 -> 1.34e-4 (slope 3.0, i.e. a
    factor 10.5).  ``max_depth`` at fixed ``h0`` barely helps (1.40e-3 ->
    1.36e-3): refinement adds leaves inside the cloud, and the face leaves
    -- which set the closure -- stay at base size.
    """
    coarse, fine = cloud_unigrid(24), cloud_unigrid(48)
    e_c = abs(_closure_of_conservative(coarse)["relative"])
    e_f = abs(_closure_of_conservative(fine)["relative"])
    assert e_f < 0.1 * e_c
    assert_banded(loglog_slope([1.0 / 48, 1.0 / 24], [e_f, e_c]), 1.5, np.inf,
                  "unigrid conservative closure slope in h")

    o_coarse = cloud_octree(2.0, 2, h0=OCTREE_H0)
    o_fine = cloud_octree(2.0, 2, h0=OCTREE_H0 / 2)
    e_oc = abs(_closure_of_conservative(o_coarse)["relative"])
    e_of = abs(_closure_of_conservative(o_fine)["relative"])
    assert_banded(e_of / e_oc, 0.0, 0.3,
                  "octree conservative closure reduction for h0/2")
    assert_banded(loglog_slope([OCTREE_H0 / 2, OCTREE_H0], [e_of, e_oc]),
                  1.0, np.inf, "octree conservative closure slope in h0")
    # depth alone must not be reported as convergence
    o_deep = cloud_octree(2.0, 3, h0=OCTREE_H0)
    e_od = abs(_closure_of_conservative(o_deep)["relative"])
    assert_banded(e_od / e_oc, 0.5, 1.5,
                  "octree closure change from max_depth alone (h0 fixed)")


# =============================================================================
# 3. the face extrapolation's own error law (no solver involved)
# =============================================================================

def _analytic_absorber(grid, kappa, dg, top_I=1.0, bottom_I=0.5):
    """
    Exact pure-absorption field of the face inflows: ``I = I_in exp(-kappa
    s)`` along each straight characteristic, with ``div F = -4 pi kappa J``
    satisfied point by point.  `energy_balance` applied to it measures only
    its own face extrapolation.
    """
    mu = dg.directions[:, 2]
    z0, z1 = grid.box[2]
    path = np.where(mu[None, :] < 0.0,
                    (z1 - grid.centers[:, 2, None]) / np.abs(mu)[None, :],
                    (grid.centers[:, 2, None] - z0) / np.abs(mu)[None, :])
    infl = np.where(mu[None, :] < 0.0, top_I, bottom_I)
    return infl * np.exp(-kappa * np.clip(path, 0.0, None))


def _analytic_layer(z, kappa, dg, top_I=1.0, bottom_I=0.5):
    """Analytic pure-absorption field on an arbitrary z array, (len(z), N_dir)."""
    mu = dg.directions[:, 2]
    box = cloud_case()["box"]
    z0, z1 = box.z0, box.z1
    z = np.asarray(z, dtype=float)[:, None]
    path = np.where(mu[None, :] < 0.0,
                    (z1 - z) / np.abs(mu)[None, :], (z - z0) / np.abs(mu)[None, :])
    infl = np.where(mu[None, :] < 0.0, top_I, bottom_I)
    return infl * np.exp(-kappa * np.clip(path, 0.0, None))


def test_face_extrapolation_error_law():
    """
    Plan section 8 asks the closure to "state its own error"; this records that
    error on a field whose physics is known exactly.

    Measured relative closure residual of `energy_balance` on the analytic
    absorber: +10.2 % (12^3), +3.97 % (24^3), +1.32 % (48^3), +0.39 % (96^3)
    -> fitted slope ~1.4 in h: monotone, but *not* second order.  Two
    channels add up to it, and both are recorded separately:

    * the face extrapolation (face value vs first interior cell differ by
      O(kappa h / |mu|), and the 42-ordinate set has |mu| ~ 0.1), and
    * the cell-midpoint volume rule for ``4 pi kappa_a J`` (measured 8.95e-3
      at 24^3, slope ~1.4): grazing ordinates vary appreciably across a cell.
    """
    dg = direction_grid(1)
    box = cloud_case()["box"]
    kappa = 0.5
    mu = dg.directions[:, 2]
    zf = np.linspace(box.z0, box.z1, 20001)
    heating_exact = float(np.trapezoid(
        -kappa * (_analytic_layer(zf, kappa, dg) * dg.weights).sum(axis=1),
        zf) * box.Lx * box.Ly)

    errs, hs, vol_errs = [], [], []
    for n in (12, 24, 48, 96):
        grid = _unigrid_n(n)
        I = _analytic_absorber(grid, kappa, dg)
        bal = energy_balance(grid, I, dg, np.full(grid.n_cells, kappa), 0.0)
        errs.append(abs(bal["relative"]))
        hs.append(1.0 / n)
        mid = absorbed_power(grid, mean_intensity(I, dg),
                             np.full(grid.n_cells, kappa), 0.0)
        vol_errs.append(abs(mid - heating_exact) / abs(heating_exact))
    for n, e in zip((12, 24, 48, 96), errs):
        assert_banded(e, 0.0, 0.15, f"analytic absorber closure, n={n}")
    assert errs[0] > errs[1] > errs[2] > errs[3], f"not monotone: {errs}"
    assert_banded(loglog_slope(hs, errs), 1.2, np.inf,
                  "closure residual slope in h (analytic absorber)")
    assert_banded(errs[1] / errs[0], 0.15, 0.6,
                  "closure residual reduction u24/u12 (analytic field)")
    assert_banded(vol_errs[1], 1e-3, 2e-2,
                  "cell-midpoint volume quadrature error, 24^3")
    assert_banded(loglog_slope(hs, vol_errs), 0.8, np.inf,
                  "volume quadrature error slope in h")


def test_closure_identity_with_analytic_integrals():
    """
    With **analytic** face fluxes and a **fine** volume quadrature the same
    field closes exactly (measured 2.7e-10 relative).  This validates the
    convention chain -- sign of ``out_bottom``, face area,
    ``F_z = int I mu dOmega`` -- so the few-percent residuals above are
    attributable to the two discretization channels and not to a
    bookkeeping error.
    """
    box = cloud_case()["box"]
    dg = direction_grid(1)
    kappa = 0.5
    mu = dg.directions[:, 2]
    slant = box.Lz / np.abs(mu)
    flux_top = (np.where(mu < 0.0, 1.0, 0.5 * np.exp(-kappa * slant))
                @ (dg.weights * mu))
    flux_bottom = (np.where(mu < 0.0, 1.0 * np.exp(-kappa * slant), 0.5)
                   @ (dg.weights * mu))
    A = box.Lx * box.Ly
    zf = np.linspace(box.z0, box.z1, 200001)
    heating = float(np.trapezoid(
        -kappa * (_analytic_layer(zf, kappa, dg) * dg.weights).sum(axis=1),
        zf) * A)
    closed = flux_top * A - flux_bottom * A - heating
    assert_banded(abs(closed / heating), 0.0, 1e-6,
                  "analytic absorber closed with analytic face/volume sides")


# =============================================================================
# 4. absorbing case: recorded reference ranges (discriminating, not an acceptance check)
# =============================================================================

@pytest.mark.parametrize("variant", ["chain", "sc"])
def test_closure_absorbing_unigrid_band(u24, variant):
    """
    Absorbing matrix case (omega capped at 0.9), no beam, on u24^3, recorded
    per transport: chain +4.23 %, sc -22.3 %.  The size is the
    42-ordinate/transport error of the *solution* (it does not shrink
    u24 -> u48), so it is a diagnostic that separates the two transports,
    not a physics acceptance check.
    """
    kappa_e, kappa_s, _g, B = cloud_fields(u24)
    res = solve_cloud(u24, ng=True, delta_M=1, beam=False, n_iter=120,
                      tol=1e-7, transport=variant)
    bal = energy_balance(u24, res["I"], direction_grid(1), kappa_e - kappa_s,
                         B)
    label = f"absorbing-u24-{variant}"
    measured, lo, hi = CLOSURE_PINS[label]
    assert_banded(bal["relative"], lo, hi, f"closure residual, {label}")


@pytest.mark.parametrize("grid_factory,label", [
    (lambda: cloud_octree(2.0, 2), "absorbing-oct2"),
    (lambda: cloud_octree(2.0, 3), "absorbing-oct3"),
])
def test_closure_absorbing_octree_band(grid_factory, label):
    """
    Same case on the octree: -1.04 % (d2) and -1.28 % (d3) vs +4.2 % on
    u24^3 -- the refined leaves resolve the absorbing core, so the residual
    drops by 4x.  Still bounded by a recorded range rather than asserted to vanish.
    """
    grid = grid_factory()
    kappa_e, kappa_s, _g, B = cloud_fields(grid)
    res = solve_cloud(grid, ng=True, delta_M=1, beam=False, n_iter=120,
                      tol=1e-7)
    bal = energy_balance(grid, res["I"], direction_grid(1), kappa_e - kappa_s,
                         B)
    _measured, lo, hi = CLOSURE_PINS[label]
    assert_banded(bal["relative"], lo, hi, f"closure residual, {label}")


# =============================================================================
# 4b. invariant-4 closure on the all-sky GROUND case (plan section 11 step 7)
# =============================================================================
# With bc.ground the bottom face carries the emitted + reflected part of the
# field (`rt3d/diagnostics.energy_balance` counts the full net F_z through the
# bottom face, so the ground law closes the bookkeeping exactly: the residual
# stays at the face-extrapolation level).  The direct beam and the ground
# exchange enter the diffuse-field balance only through the source `S` and the
# ground inflow already contained in `I`, so a march or tally that dropped the
# ground term would blow this residual up by orders of magnitude.
#
# Measured (ng, delta-M 1, beam, albedo 0.3, eps B_g = 1, converged 400 it):
#   oct2 -3.5e-3, oct3 -2.2e-3  (vs -1.0..-1.3e-2 on the no-ground absorbing
#   case: the ground emission dominates the bottom face and the *relative*
#   residual improves);  u24^3 +8.7e-2 (F11/F7 unigrid plateau, recorded wide).

@pytest.mark.parametrize("grid_factory,label,pin", [
    (lambda: cloud_octree(2.0, 2), "ground-oct2", (-6e-3, -1e-4)),
    (lambda: cloud_octree(2.0, 3), "ground-oct3", (-4e-3, -1e-4)),
    (lambda: cloud_unigrid(24), "ground-u24", (0.05, 0.13)),
])
def test_closure_ground_case_band(grid_factory, label, pin):
    grid = grid_factory()
    res = solve_cloud(grid, bc="ground", albedo=0.3, emissivity=1.0, B_g=1.0,
                      beam=True, ng=True, delta_M=1, n_iter=400, tol=1e-8)
    assert res["converged"]
    kappa_e, kappa_s, _g, B = cloud_fields(grid)
    bal = energy_balance(grid, res["I"], direction_grid(1),
                         kappa_e - kappa_s, B)
    assert_banded(bal["relative"], pin[0], pin[1], f"ground closure residual, {label}")
    # the bottom face must be carrying the ground emission (strongly negative
    # net outward power = net UP into the domain); a dropped ground term
    # would leave the bottom face nearly dark.
    assert bal["out_bottom"] < -100.0, (
        f"ground emission missing from the bottom-face power: "
        f"out_bottom = {bal['out_bottom']:.1f}")
    # octree closure improves with depth (F7 face-extrapolation law)
    if label == "ground-oct3":
        oct2 = cloud_octree(2.0, 2)
        r2 = solve_cloud(oct2, bc="ground", albedo=0.3, emissivity=1.0,
                         B_g=1.0, beam=True, ng=True, delta_M=1,
                         n_iter=400, tol=1e-8)
        k2 = cloud_fields(oct2)
        b2 = energy_balance(oct2, r2["I"], direction_grid(1),
                            k2[0] - k2[1], k2[3])
        assert abs(bal["relative"]) < abs(b2["relative"])


# =============================================================================
# 5. cross-family absorbed power (SC vs MC A1 vs MC A2)
# =============================================================================

A1_SEEDS = (31, 47)
A1_HISTORIES = 200_000


def _a1_absorbed(grid, kappa_e, kappa_s, B, g, seed, mode="cell",
                 histories=A1_HISTORIES):
    """A1 explicit-variant absorbed power and its batch sigma."""
    m = j_mc(grid, kappa_e, kappa_s, B, common_bc(bottom_I=0.5), g=g,
             periodic_box=cloud_case()["box"], stage1="explicit",
             n_histories=0, n_boundary_histories=histories, n_batches=100,
             seed=seed, mode=mode, jit=True)
    power = abs(absorbed_power(grid, m["J_total"], kappa_e - kappa_s, B))
    weight = 4.0 * np.pi * (kappa_e - kappa_s) * cell_volumes(grid)
    sigma = float(np.sqrt(np.sum((weight * m["sigma_J_total"]) ** 2)))
    return power, sigma


def test_cross_family_absorbed_power(oct2):
    """
    Total absorbed power on the same octree case, per family.

    Absorbed power is reported as a positive magnitude (the heating-rate
    integral itself is negative for B = 0).  Measured: SC-42 144.94, SC-162
    141.09, MC A1 (2e5 boundary histories) 142.07 +- 0.26 (seed 31) /
    141.7 +- 0.26 (seed 47), A1 global-majorant mode 142.20 +- 0.26.
    A1's statistical error is 0.18 %, so the 2 %
    SC-42 gap is the **angular quadrature error** of 42 ordinates: with 162
    ordinates SC lands within 0.7 % of A1.  Asserted: both seeds inside the
    range, the gap shrinks with more ordinates, and the majorant mode does
    not move the answer beyond statistics.
    """
    kappa_e, kappa_s, g, B = cloud_fields(oct2)
    box = cloud_case()["box"]
    bc = common_bc(bottom_I=0.5)

    sc = {}
    for n_refine in (1, 2):
        dg = direction_grid(n_refine)
        res = lambda_iteration(oct2, dg, kappa_e, kappa_s, B, bc, g=g,
                               periodic_box=box, n_iter=120, tol=1e-7,
                               ng=True, delta_M=1)
        sc[dg.n_dir] = abs(absorbed_power(oct2, res["J"], kappa_e - kappa_s, B))
    assert_banded(abs(sc[162] - sc[42]) / sc[42], 0.0, 0.05,
                  "SC absorbed power shift 42 -> 162 ordinates")

    powers = []
    for seed in A1_SEEDS:
        p, s = _a1_absorbed(oct2, kappa_e, kappa_s, B, g, seed)
        powers.append((p, s))
        assert_banded(abs(p - sc[42]) / sc[42], 0.0, 0.03,
                      f"A1 vs SC-42 absorbed power (seed {seed})")
        assert_banded(s / p, 0.0, 0.005, f"A1 relative sigma (seed {seed})")
    (p1, s1), (p2, s2) = powers
    assert_banded(abs(p1 - p2) / np.hypot(s1, s2), 0.0, 6.0,
                  "A1 seed-to-seed spread in combined sigma")

    # more ordinates close the gap towards the noise-free-in-angle A1 value
    assert abs(sc[162] - p1) < abs(sc[42] - p1) - 2.0 * s1
    assert_banded(abs(sc[162] - p1) / p1, 0.0, 0.015,
                  "A1 vs SC-162 absorbed power")

    pg, sg = _a1_absorbed(oct2, kappa_e, kappa_s, B, g, A1_SEEDS[0],
                          mode="global")
    assert_banded(abs(pg - p1) / np.hypot(s1, sg), 0.0, 6.0,
                  "A1 cell vs global majorant absorbed power")


def test_mc_stage2_closure_reproduces_sc(oct2):
    """
    MC stage 2 (`radiance_from_source`, deterministic line integral) closed
    on its own tallies gives the same absorbed power and the same closure
    residual as the SC sweep of the same source (T2 identity), to round-off
    -- the closure machinery is family-independent, so T4 numbers are
    directly comparable across families.
    """
    dg = direction_grid(1)
    bc = common_bc(bottom_I=0.5)
    box = cloud_case()["box"]
    kappa_e, kappa_s, _g, B = cloud_fields(oct2)
    kappa_a = kappa_e - kappa_s

    res = solve_cloud(oct2, ng=True, delta_M=1, beam=False, n_iter=120,
                      tol=1e-7)
    mc = radiance_from_source(oct2, dg, res["kappa_e_delta"], res["S"], bc,
                              periodic_box=box, estimator="deterministic")
    b_mc = energy_balance(oct2, mc["I"], dg, kappa_a, B)
    b_solver = energy_balance(oct2, res["I"], dg, kappa_a, B)

    p_mc = absorbed_power(oct2, mean_intensity(mc["I"], dg), kappa_a, B)
    p_solver = absorbed_power(oct2, res["J"], kappa_a, B)
    assert p_mc < 0.0 and p_solver < 0.0        # B = 0 -> net absorption
    assert_banded(abs(p_mc - p_solver) / abs(p_solver), 0.0, 1e-7,
                  "MC stage-2 vs converged-solver absorbed power")
    assert_banded(abs(b_mc["relative"] - b_solver["relative"]), 0.0, 1e-7,
                  "MC stage-2 vs converged-solver closure residual")

    # next-flight MC stage 2 closes consistently too, within its statistics
    nf = radiance_from_source(oct2, dg, res["kappa_e_delta"], res["S"], bc,
                              periodic_box=box, estimator="next_flight",
                              n_histories=25, n_batches=25, seed=77,
                              jit=True)
    p_nf = absorbed_power(oct2, nf["J"], kappa_a, B)
    assert p_nf < 0.0
    assert_banded(abs(p_nf - p_mc) / abs(p_mc), 0.0, 0.01,
                  "MC next-flight vs deterministic absorbed power")


# =============================================================================
# 6. local invariant: div F vs heating rate, per cell, on both grids
# =============================================================================

def test_flux_divergence_octree_local(u24, oct2):
    """
    Per-cell ``div F`` of the conservative-medium solution, where the exact
    answer is ``4 pi kappa_a (B - J) = 0`` (so the local residual is the
    scheme + stencil error alone).

    Measured mean |div F| / (|F_z|/Lz): 0.052 (u24, max 0.42) vs 0.115
    (octree d2, max 1.03) -- the new octree stencil (face-probe differences
    between neighbour leaf centers) is the same order as the unigrid
    central-difference stencil, a factor ~2 worse at h0 = 1.25 m.  Recorded as
    a reference range; the global closures above are the sharp version of this invariant.
    """
    dg = direction_grid(1)
    res_u = _conservative_solve(u24)
    div_u = flux_divergence(u24, res_u["F"])
    scale_u = np.abs(res_u["F"][:, 2]).mean() / cloud_case()["box"].Lz
    mean_u, max_u = (np.mean(np.abs(div_u)) / scale_u,
                     np.max(np.abs(div_u)) / scale_u)

    res_o = _conservative_solve(oct2)
    div_o = flux_divergence(oct2, res_o["F"])
    scale_o = np.abs(res_o["F"][:, 2]).mean() / cloud_case()["box"].Lz
    mean_o, max_o = (np.mean(np.abs(div_o)) / scale_o,
                     np.max(np.abs(div_o)) / scale_o)

    assert_banded(mean_u, 0.0, 0.10, "u24 mean |div F| / scale")
    assert_banded(mean_o, 0.0, 0.20, "octree d2 mean |div F| / scale")
    assert_banded(max_o, 0.0, 2.0, "octree d2 max |div F| / scale")
    assert_banded(mean_o / mean_u, 1.0, 3.0,
                  "octree / unigrid local div-F residual ratio")


def test_guard_energy_balance_accepts_octree(oct2):
    """Finding F4 is closed: every diagnostics entry point takes an octree."""
    dg = direction_grid(1)
    res = solve_cloud(oct2, ng=True, delta_M=1, n_iter=40, tol=1e-5)
    kappa_e, kappa_s, _g, B = cloud_fields(oct2)
    kappa_a = kappa_e - kappa_s

    bal = energy_balance(oct2, res["I"], dg, kappa_a, B)
    assert np.isfinite(bal["residual"]) and np.isfinite(bal["relative"])
    fp = face_powers(oct2, res["I"], dg)
    assert len(fp["area"]) == 2 and fp["area"][0].size > 0
    assert np.all(np.isfinite(flux_divergence(oct2, res["F"])))
    assert cell_volumes(oct2).shape == (oct2.n_leaf,)
