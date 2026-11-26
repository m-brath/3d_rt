# -*- coding: utf-8 -*-
"""
T5 of the solver-matrix plan (plan/solver-test-matrix-plan.md section 3):
the **refinement ladder**, the reference-based **ranking** of the transports,
and **convergence-order fits**.

This is the tier that turns "variant A and variant B disagree by 31 %" into
"both converge, at measured rates, to measured limits".  Errors are measured
against a *finer grid of the same family* rather than against another variant,
so a shared bias cannot hide as agreement — and, per plan findings F10/F11,
gaps **between** families are recorded as reference ranges with a stated cause rather than
asserted to vanish.

Design decisions that the ladder probes forced (plan findings F10 and F11, all
asserted below):

* **Each grid family gets its own ladder.**  On the octree, resolution is set
  by three knobs — base cell ``h0``, the optical-depth criterion ``tau_max``
  and ``max_depth`` — and they are *not* interchangeable.  The ladder is taken
  in ``tau_max`` at fixed depth 7: the depth ladder **saturates** (at fixed
  ``h0``/``tau_max`` the τ criterion is already binding at depth ~5, so depth
  3/4/5 give the same solution), and ``h0`` barely moves the cell-centered
  field at all (measured ≤ 3.3e-4 rms J over ``h0`` 5 → 1.25).
* **The unigrid ladder converges in its own family but not toward the
  octree.**  Internally the unigrid is textbook first order (Cauchy errors
  2.8e-2 → 1.4e-2 → 6.7e-3, slope 1.00 in h), yet the whole family plateaus
  at ~4 % rms J from the octree reference at every resolution u12…u96.  The
  cause is finding F11: neither unigrid short-characteristics stencil follows
  the true characteristic, so both carry a bounded, **resolution-independent**
  grazing-ordinate error — 11–14 % rms against the analytic exponential in
  the pure-absorber limit, where the octree sweep is exact to round-off.
  Verification invariant 3 is therefore stated **per family** below, and the
  unigrid-vs-march gap is recorded as an h-independent range.
* **Convergence of the iteration is a precondition**, not an afterthought: a
  rung solved to a loose tolerance measures its iteration error, and the
  slope fit then reports nonsense.  Every rung here is solved with
  ``n_iter=400, tol=1e-8`` and asserts ``converged`` (an earlier, uncorrected
  probe ranked a 5.9 % octree rung that was simply unconverged).
* **Monte Carlo order is measured by seed splitting**, not against the
  deterministic reference: the unigrid-vs-octree transport bias of this case
  (~4 % rms J, F11) is larger than the statistical error at every history
  count the acceptance check can afford, so a deviation-vs-reference fit cannot see
  ``N^-0.5`` at all.

Yardstick: ``octree tau_max 0.25, depth 7`` (the τ criterion needs level 6, so
depth 7 is not binding — identical to depth 6, 107 108 leaves).  Reference
rung of the plan/report: ``octree tau_max 0.5, depth 5`` — 50 380 leaves,
exactly one criterion step coarser than the yardstick.  Sufficiency rung:
``tau_max 0.125, depth 7``.
"""

import time

import numpy as np
import pytest

from matrix_cases import (
    cloud_case,
    cloud_fields,
    cloud_octree,
    cloud_unigrid,
    common_bc,
    direction_grid,
)
from matrix_stats import (
    assert_banded,
    cell_scale,
    field_errors,
    h_equivalent,
    loglog_slope,
    rel_max_dev,
    rel_rms_dev,
)
from rt3d.diagnostics import cell_volumes, face_powers, heating_rate
from rt3d.mc.forward import j_mc, radiance_from_source
from rt3d.solver import lambda_iteration
from rt3d.source import mean_intensity

pytestmark = [pytest.mark.slow, pytest.mark.reference]

BOX = cloud_case()["box"]
DG = direction_grid(1)

YARD_TAU, YARD_DEPTH = 0.25, 7
SUFF_TAU = 0.125
REF_TAU, REF_DEPTH = 0.5, 5

# recorded acceptance intervals (measured first, then frozen — plan section 3)
SLOPE_BAND = (0.6, 1.6)
TAU_SLOPE_BAND = (0.3, 1.0)
MC_SLOPE_BAND = (-0.65, -0.35)

# measured rms-J self-error of the reference rung against the yardstick;
# everything in this module scales relative to this number
REF_SELF_ERROR = 0.00549


def solve(grid, transport="chain", delta_M=1, ng=True, tol=1e-8, n_iter=400):
    """
    Deterministic solve of the matrix case; returns ``(result, wall_s)``.

    The iteration budget is deliberately generous: a ladder rung that is not
    converged measures the iteration error, not the discretization error, and
    the slope fit then fits noise.  Every rung asserts ``converged``.
    """
    kappa_e, kappa_s, g, B = cloud_fields(grid)
    t0 = time.perf_counter()
    res = lambda_iteration(
        grid, DG, kappa_e, kappa_s, B, common_bc(bottom_I=0.5),
        g=g, periodic_box=BOX, beam=None, n_iter=n_iter, tol=tol, ng=ng,
        delta_M=delta_M, transport=transport,
    )
    dt = time.perf_counter() - t0
    assert res["converged"], (
        f"ladder rung not converged: {res['n_iter']} iterations, residual "
        f"{res['history'][-1]:.3e} (budget {n_iter}, tol {tol})")
    return res, dt


def observables(grid, res):
    """
    Scalar observables of a solution: out-top / out-bottom power through the
    horizontal faces, volume-integrated absorbed power, and the
    **volume-weighted** domain-mean J.  All four are grid-independent numbers
    (no cross-grid sampling involved), so they are directly comparable across
    grid families.

    The volume weighting is not cosmetic: an octree's leaves come in several
    sizes, so a plain ``np.mean`` over leaves weights a 0.156 m leaf in the
    cloud core the same as a 1.25 m leaf in the vacuum and gives a number
    that is not comparable to the unigrid's mean (measured 0.450 vs 0.601 --
    an artefact of the statistic, not of the physics).
    """
    kappa_e, kappa_s, _g, B = cloud_fields(grid)
    fp = face_powers(grid, res["I"], DG)
    vol = cell_volumes(grid)
    return {
        "out_top": float(fp["out_top"]),
        "out_bottom": float(fp["out_bottom"]),
        "power": float(np.sum(heating_rate(kappa_e - kappa_s, res["J"], B)
                              * vol)),
        "mean_J": float(np.average(res["J"], weights=vol)),
    }


def rung_errors(grid, res, yard_grid, yard_res):
    """Field + observable errors of a rung against the yardstick solution."""
    emax, erms = field_errors(grid, res["J"], yard_grid, yard_res["J"])
    s, sref = observables(yard_grid, yard_res), observables(grid, res)
    out = {"J_max": emax, "J_rms": erms, "h": cell_scale(grid),
           "h_eq": h_equivalent(grid)}
    out.update({k: abs(s[k] - sref[k]) / abs(sref[k]) for k in s})
    return out


def ladder(rungs, yard_grid, yard_res, **solve_kw):
    """Solves every rung; returns ``[(label, errors, wall_s, (grid, res)), ...]``."""
    rows = []
    for label, make in rungs:
        grid = make()
        res, dt = solve(grid, **solve_kw)
        rows.append((label, rung_errors(grid, res, yard_grid, yard_res), dt,
                     (grid, res)))
    return rows


def assert_monotone(rows, key, label):
    errs = [e[key] for _label, e, _dt, _sol in rows]
    assert all(a > b for a, b in zip(errs, errs[1:])), \
        f"{label}: errors not monotone along the ladder: {errs}"
    return errs


@pytest.fixture(scope="module")
def yard():
    """The yardstick solution: octree tau_max 0.25, depth 7 (~45 s solve)."""
    grid = cloud_octree(YARD_TAU, YARD_DEPTH)
    res, dt = solve(grid)
    return grid, res, dt


@pytest.fixture(scope="module")
def sufficiency():
    """tau_max 0.125 at depth 7: strictly finer than the yardstick."""
    grid = cloud_octree(SUFF_TAU, YARD_DEPTH)
    res, dt = solve(grid)
    return grid, res, dt


@pytest.fixture(scope="module")
def ref():
    """The plan's reference rung: octree tau_max 0.5, depth 5 (~20 s solve)."""
    grid = cloud_octree(REF_TAU, REF_DEPTH)
    res, dt = solve(grid)
    return grid, res, dt


@pytest.fixture(scope="module")
def u96():
    """Unigrid-family yardstick: 96^3 cells (884 736 cells)."""
    grid = cloud_unigrid(96)
    res, dt = solve(grid)
    return grid, res, dt


def test_octree_yardstick_is_sufficient(ref, yard, sufficiency):
    """
    The yardstick's own error, measured against a rung one criterion step
    finer (``tau_max`` 0.25 → 0.125, i.e. level 6 → 7): measured ~2e-3 rms J.
    The reference rung's error against the yardstick is measured at **0.55 %
    rms J, 5.3 % max J, 0.20 % absorbed power, 0.4 % mean J** — the numbers
    that set the floor for every error reported by the tiers below.

    Also asserted (plan finding F10, part 1): at fixed ``h0`` and ``tau_max``
    the **depth ladder saturates** — depth 4 gives a grid identical to the
    depth-5 reference (the τ criterion is already binding at level ≤ 5), so
    "refine the octree deeper" is not a convergence path.

    Measured (T5 run): the yardstick-vs-0.125 self-error is **6.8e-3 rms J**,
    i.e. *larger* than the reference rung's 5.5e-3 — the ``tau_max`` knob
    **saturates below τ ≈ 1**: further τ refinement at fixed ``h0`` stops
    paying (the octree's internal convergence floor at this base lattice is
    ≈ 7e-3 rms J, same magnitude as the REF-vs-yardstick error itself).  The
    reference range below fixes that floor; a yardstick whose self-error grew past it
    would mean the octree family stopped being self-consistent.
    """
    yard_grid, yard_res, _t = yard
    suff_grid, suff_res, _ts = sufficiency
    assert suff_grid.n_leaf > yard_grid.n_leaf, (
        f"tau_max 0.125 at depth 7 did not produce a finer grid than the "
        f"yardstick ({suff_grid.n_leaf} vs {yard_grid.n_leaf} leaves) — the "
        "sufficiency rung is meaningless")
    _emax_s, erms_s = field_errors(yard_grid, yard_res["J"], suff_grid,
                                   suff_res["J"])
    assert_banded(erms_s, 1e-3, 1.2e-2,
                  "yardstick self-error vs 0.125 rung (τ-saturation floor)")

    ref_grid, ref_res, _t_ref = ref
    emax, erms = field_errors(ref_grid, ref_res["J"], yard_grid, yard_res["J"])
    assert_banded(erms, 3e-4, 1e-2, "reference rung self-error (rms J)")
    assert_banded(emax, 1e-2, 1e-1, "reference rung self-error (max J)")
    assert abs(erms - REF_SELF_ERROR) < 4e-3, (
        f"reference self-error moved to {erms:.3e}; REF_SELF_ERROR = "
        f"{REF_SELF_ERROR:.3e} — re-record the ladder reference ranges")

    sref, s = observables(ref_grid, ref_res), observables(yard_grid, yard_res)
    for key in ("out_top", "out_bottom", "power", "mean_J"):
        assert_banded(abs(s[key] - sref[key]) / abs(sref[key]), 0.0, 0.02,
                      f"reference rung self-error in {key}")

    deep = cloud_octree(REF_TAU, 4)
    assert deep.n_leaf == ref_grid.n_leaf, (
        "octree depth 4 at tau_max 0.5 no longer yields the same grid as "
        f"depth 5 ({deep.n_leaf} vs {ref_grid.n_leaf} leaves) — the τ "
        "criterion is no longer binding before the cap, so the "
        "depth-saturation statement (F10) needs revising")


TAU_RUNGS = [(f"oct-tau{tau}", lambda tau=tau: cloud_octree(tau, YARD_DEPTH))
             for tau in (8.0, 4.0, 2.0, 1.0, REF_TAU)]


def test_octree_tau_criterion_ladder_converges(yard):
    """
    Octree ladder in the refinement criterion, ``tau_max = 8, 4, 2, 1, 0.5``
    at fixed ``h0 = 1.25`` and depth cap 7, against the ``tau_max = 0.25``
    yardstick.  Measured rms-J errors 1.33e-2 → 8.6e-3 → 5.7e-3 → 3.9e-3
    (depth-5 probe values; depth 7 is not binding on these rungs) → the
    reference self-error at the last rung.

    Fitted in ``tau_max`` — the octree's error falls sub-linearly in its
    criterion (~tau_max**0.6, plan finding F10).  A fit in the cost-matched
    resolution ``h_eq = (V/n_leaf)**(1/3)`` would give slope ~3 and is
    reported for information only: the τ criterion buys leaf count, not
    uniform resolution, so the scheme's order does not show up in ``h_eq``.

    Monotonicity is asserted on rms J and absorbed power.  The **max-J error
    is not monotone** (measured 5.29e-2 → 5.13e-2 → 5.26e-2), and neither is
    **mean_J** at the finest rungs (3.0e-4 → 2.7e-4 → 3.2e-4): the worst
    cells and the volume average alike sit at the τ-saturation floor of the
    base lattice, not at the cloud-resolution rung — both are recorded as
    ranges instead (F10, saturation floor).
    """
    yard_grid, yard_res, _t = yard
    rows = ladder(TAU_RUNGS, yard_grid, yard_res)
    for key in ("J_rms", "power"):
        assert_monotone(rows, key, f"octree tau ladder ({key})")
    for key in ("out_top", "out_bottom"):
        assert_banded(rows[-1][1][key], 0.0, 0.02,
                      f"octree tau ladder: finest rung error in {key}")
    for label, e, dt, _sol in rows:
        assert_banded(e["J_max"], 0.03, 0.08,
                      f"octree tau ladder {label}: max-J floor range")
    assert_monotone(rows[:-1], "mean_J", "octree tau ladder (mean_J rungs)")
    assert_banded(rows[-1][1]["mean_J"], 0.0, 5e-4,
                  "octree tau ladder: REF-rung mean_J floor range")

    taus = [float(t) for t in (8.0, 4.0, 2.0, 1.0, REF_TAU)]
    slope_tau = loglog_slope(taus, [e["J_rms"] for _l, e, _d, _s in rows])
    assert_banded(slope_tau, *TAU_SLOPE_BAND,
                  "octree tau ladder: fitted slope in tau_max")
    for label, e, dt, _sol in rows:
        print(f"ladder {label:12s} leaves~{BOX.Lx * BOX.Ly * BOX.Lz / e['h_eq'] ** 3:8.0f}"
              f" h_eq={e['h_eq']:.4f} {dt:6.1f}s  J_rms err {e['J_rms']:.3e}"
              f"  power err {e['power']:.3e}")
    heq = [e["h_eq"] for _l, e, _d, _s in rows]
    print(f"octree tau ladder: slope in h_eq = "
          f"{loglog_slope(heq, [e['J_rms'] for _l, e, _d, _s in rows]):.2f} "
          "(information only, see docstring)")


def test_octree_depth_ladder_saturates(ref):
    """
    Plan finding F10, part 1, quantified: at fixed ``h0``/``tau_max = 0.5``
    the depth rungs 1 and 2 still carry real error (measured 2.2e-2 and
    1.6e-2 rms J vs the depth-5 reference), but depth 3 is already within
    2.4e-4 — the τ criterion binds before the cap.  A "deeper octree" is
    therefore *not* a convergence path on this case, and the ladder in
    section 2 must (and does) refine ``tau_max`` at a non-binding cap.
    """
    ref_grid, ref_res, _tr = ref
    errs = []
    for depth in (1, 2, 3):
        grid = cloud_octree(REF_TAU, depth)
        res, _dt = solve(grid)
        errs.append(field_errors(grid, res["J"], ref_grid, ref_res["J"])[1])
    assert errs[0] > errs[1] > errs[2], \
        f"depth ladder errors not decreasing: {errs}"
    assert errs[2] < 5e-4, (
        f"depth 3 no longer saturates (err {errs[2]:.3e} vs depth 5) — "
        "F10's saturation statement needs revising")
    assert errs[2] < 0.05 * errs[1], \
        f"depth 3 is not deep enough to be saturated: {errs}"


def test_octree_base_cell_barely_moves_the_field(ref):
    """
    Plan finding F10, part 2 — and the measurement that **killed** the
    "octree base-lattice halo" hypothesis for the cross-family gap: at fixed
    ``tau_max = 0.5`` and depth 5, ``h0`` = 5.0 and 2.5 (base cell edges 10 m
    and 5 m, i.e. whole-domain and half-domain base cells) move the
    cell-centered J field by only 3.3e-4 and ~1e-4 rms relative to the
    h0 = 1.25 reference, and the volume-mean J does not move at all to five
    decimals.  ``h0`` matters for *face-evaluated* quantities (the energy
    closure, finding F7), not for the cell-centered radiation field.

    So the 4 % unigrid-vs-octree gap of section 6 is not the base lattice:
    its cause is documented in section 6 itself (finding F11).
    """
    ref_grid, ref_res, _tr = ref
    ref_obs = observables(ref_grid, ref_res)
    errs = []
    for h0 in (5.0, 2.5):
        grid = cloud_octree(REF_TAU, REF_DEPTH, h0=h0)
        res, _dt = solve(grid)
        emax, erms = field_errors(grid, res["J"], ref_grid, ref_res["J"])
        dmean = abs(observables(grid, res)["mean_J"] - ref_obs["mean_J"]) \
            / abs(ref_obs["mean_J"])
        assert_banded(erms, 0.0, 0.01, f"h0 = {h0}: rms J movement vs h0 = 1.25")
        assert_banded(dmean, 0.0, 1e-3, f"h0 = {h0}: mean-J relative movement")
        errs.append((h0, erms, emax))
    print("octree h0 insensitivity:", errs)


UNI_RUNGS = [(f"u{n}", lambda n=n: cloud_unigrid(n)) for n in (12, 24, 48)]


def test_unigrid_ladder_converges_in_its_own_family(u96):
    """
    Unigrid 12^3 -> 48^3 against **96^3** (same family, so the yardstick
    carries no inter-family transport offset) — plus the honest convergence
    measure for a family with an h-independent scheme error (F11).

    Two facts, both asserted (measured T5 run):

    * the **Cauchy ladder** — rung differences u12-u24, u24-u48, u48-u96 —
      is textbook first order: 2.79e-2, 1.36e-2, 6.72e-3 rms J, fitted
      slope **1.03** in h.  This is what "the unigrid family converges"
      means;
    * the distance **toward u96** does *not* halve along the ladder
      (1.09e-2, 7.18e-3, 6.72e-3, fitted slope 0.35): the F11 transport
      error is a structured grazing-ordinate pattern that changes shape
      with h instead of nesting.  Fitting *that* to first order would be
      wrong, so it is only asserted monotone and bounded by a recorded range.
    """
    yard_grid, yard_res, _t = u96
    rows = ladder(UNI_RUNGS, yard_grid, yard_res)
    for key in ("J_rms", "J_max", "out_top", "out_bottom", "power", "mean_J"):
        assert_monotone(rows, key, f"unigrid ladder ({key})")
    for _label, e, _dt, _s in rows:
        assert_banded(e["J_rms"], 5e-3, 2e-2,
                      "unigrid-vs-u96 rms J distance (non-nesting F11 pattern)")

    cauchy, hs = [], []
    pairs = [(0, 1), (1, 2), (2, 3)]
    sols = [(g, r) for _l, _e, _dt, (g, r) in rows] + [(yard_grid, yard_res)]
    for ia, ib in pairs:
        ga, ra = sols[ia]
        gb, rb = sols[ib]
        _emax, erms = field_errors(ga, ra["J"], gb, rb["J"])
        cauchy.append(erms)
        hs.append(cell_scale(ga) - cell_scale(gb))
    assert all(a > b for a, b in zip(cauchy, cauchy[1:])), \
        f"unigrid Cauchy ladder not decreasing: {cauchy}"
    slope = loglog_slope(hs, cauchy)
    assert_banded(slope, *SLOPE_BAND, "unigrid Cauchy ladder: fitted slope in h")
    for (ia, ib), e in zip(pairs, cauchy):
        print(f"cauchy u{12 * 2 ** ia} vs u{12 * 2 ** ib}: J_rms {e:.3e}")
    print(f"unigrid Cauchy slope: {slope:.2f}")
    for label, e, dt, _sol in rows:
        print(f"ladder {label:6s} h={e['h']:.4f} {dt:7.1f}s  "
              f"J_rms err vs u96 {e['J_rms']:.3e}")


def lambda_solved(grid, kappa):
    """Pure-absorber solve of ``grid`` (uniform kappa, zero source)."""
    n = grid.n_leaf if hasattr(grid, "n_leaf") else grid.n_cells
    res = lambda_iteration(
        grid, DG, np.full(n, kappa), np.zeros(n), np.zeros(n),
        common_bc(bottom_I=0.5),
        g=np.zeros(n), periodic_box=BOX, beam=None,
        n_iter=400, tol=1e-8, ng=True, delta_M=0, transport="chain",
    )
    assert res["converged"], "pure-absorber solve did not converge"
    return res


def test_invariant3_pure_absorber_holds_per_family():
    """
    AGENTS.md verification invariant 3 (`kappa_s = 0`, `B = 0` -> transmittance
    = `exp(-tau)` along *every* ray), stated **per transport family** — the
    ladder probes showed it is family-dependent, and the plan records it that
    way (finding F11).

    Pure absorber (uniform ``kappa_e``, ``kappa_s = 0``, ``B = 0``, no beam,
    delta-M off) on the periodic box: the exact solution is
    ``I_in exp(-kappa_e s)`` with ``s`` the distance from the cell center to
    the top/bottom face along the backward ray (x/y are periodic, so
    oblique rays always enter through z; ``mu = 0`` ordinates are optically
    infinite and go to zero inflow).  Measured rms/max relative deviation
    of ``I`` against that analytic field:

    | family | kappa 0.25 | kappa 1.0 |
    |---|---|---|
    | octree sweep            | 6.5e-16 / max 2.5e-15 | same, round-off |
    | unigrid DDA march (deterministic line integral) | 2.3e-15 | 6.3e-15 |
    | unigrid `chain` sweep   | **rms 1.16e-1, max 2.8e-1** | **rms 1.40e-1, max 3.2e-1** |

    The unigrid stencil's effective path is the cell-to-cell staircase
    (documented in `docs/sc-transport.md` §12.2/§12.5, recorded for `sc` by
    `test_short_characteristics.py::test_transmittance_staircase_error`);
    the error is bounded and resolution-independent, not a bug.
    """
    bc = common_bc(bottom_I=0.5)
    z0, z1 = cloud_unigrid(12).box[2]
    mu = DG.directions[:, 2]

    def analytic_I(centers, kappa):
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(
                mu[None, :] < 0.0,
                (z1 - centers[:, 2, None]) / np.abs(mu)[None, :],
                (centers[:, 2, None] - z0) / np.abs(mu)[None, :],
            )
        inflow = np.where(mu < 0.0, 1.0, 0.5)
        return inflow[None, :] * np.exp(-kappa * np.clip(s, 0.0, None))

    for kappa in (0.25, 1.0):
        o = cloud_octree(REF_TAU, REF_DEPTH)
        res_o = lambda_solved(o, kappa)
        Iana_o = analytic_I(o.centers, kappa)
        assert rel_max_dev(res_o["I"], Iana_o) < 1e-12, (
            "octree sweep no longer reproduces exp(-tau) to round-off: "
            f"max dev {rel_max_dev(res_o['I'], Iana_o):.3e}")

        u = cloud_unigrid(24)
        res_u = lambda_solved(u, kappa)
        march = radiance_from_source(
            u, DG, np.full(u.n_cells, kappa), np.zeros(u.n_cells), bc,
            periodic_box=BOX, estimator="deterministic",
        )
        Iana_u = analytic_I(u.centers, kappa)
        assert rel_max_dev(march["I"], Iana_u) < 1e-12, (
            "unigrid deterministic march no longer exact: "
            f"{rel_max_dev(march['I'], Iana_u):.3e}")

        dev_rms = rel_rms_dev(res_u["I"], Iana_u)
        dev_max = rel_max_dev(res_u["I"], Iana_u)
        assert_banded(dev_rms, 0.05, 0.25,
                      f"unigrid chain vs analytic exp(-tau), kappa={kappa} (rms)")
        assert_banded(dev_max, 0.1, 0.55,
                      f"unigrid chain vs analytic exp(-tau), kappa={kappa} (max)")
        print(f"invariant 3, kappa={kappa}: octree "
              f"{rel_max_dev(res_o['I'], Iana_o):.1e}, march "
              f"{rel_max_dev(march['I'], Iana_u):.1e}, chain rms {dev_rms:.3e}"
              f" max {dev_max:.3e}")


def test_unigrid_chain_transport_error_is_resolution_independent(ref, u96):
    """
    The quantitative statement of plan finding F11, and the explanation of
    the F10 plateau: solving the matrix case on the unigrid and then
    evaluating the **exact line integral of that converged source** along
    the same rays (`radiance_from_source(..., estimator='deterministic')`,
    identical to the octree sweep to 1.2e-12 — T2) leaves an rms-J deviation
    of the *transport scheme itself* that does not shrink with h:

    | grid | chain vs exact march (rms J) | volume-mean J: chain / march |
    |---|---|---|
    | u12^3 | 4.05e-2 | 0.5979 / 0.5795 |
    | u24^3 | 3.98e-2 | 0.6006 / 0.5827 |
    | u48^3 | 3.95e-2 | 0.6012 / 0.5836 |

    Recorded: every rung's error lies in the same range and u48 is not better
    than u12 by more than a few percent.  The chain-vs-march offset in the
    volume-mean J is recorded at 2–5 %.

    This is why the unigrid ladder of section 5 is fitted **within** the
    family, and why the unigrid-vs-octree-reference gap is recorded here
    rather than asserted to converge: the octree reference (volume-mean J
    0.58321) agrees with the unigrid *exact march* (0.5827 at u24) to 0.1 %,
    while the unigrid chain solution sits 3 % higher.
    """
    bc = common_bc(bottom_I=0.5)
    ref_grid, ref_res, _tr = ref
    ref_mean = observables(ref_grid, ref_res)["mean_J"]

    means, errs = [], []
    for n in (12, 24, 48):
        grid = cloud_unigrid(n)
        res, _dt = solve(grid)
        march = radiance_from_source(
            grid, DG, res["kappa_e_delta"], res["S"], bc, periodic_box=BOX,
            estimator="deterministic",
        )
        Jm = mean_intensity(march["I"], DG)
        e = rel_rms_dev(res["J"], Jm)
        errs.append(e)
        mean_m = float(np.average(Jm, weights=cell_volumes(grid)))
        means.append((float(np.average(res["J"], weights=cell_volumes(grid))),
                      mean_m))
        print(f"u{n}: chain vs exact march rms J {e:.3e}, mean J "
              f"{means[-1][0]:.5f} vs {mean_m:.5f}")

    for e in errs:
        assert_banded(e, 0.03, 0.05, "unigrid chain-vs-march transport error (rms J)")
    assert errs[2] > 0.8 * errs[0], (
        f"the unigrid transport error shrank by refinement ({errs}) — the "
        "resolution-independence of F11 no longer holds; re-check the "
        "staircase analysis and re-record the reference ranges")

    for mean_c, mean_m in means:
        assert_banded(abs(mean_c - mean_m) / abs(mean_m), 0.02, 0.05,
                      "unigrid chain-vs-march mean-J offset")

    assert_banded(abs(ref_mean - means[1][1]) / ref_mean, 0.0, 5e-3,
                  "octree reference vs unigrid exact-march mean J")

    u_grid, u_res, _tu = u96
    _emax, erms = field_errors(u_grid, u_res["J"], ref_grid, ref_res["J"])
    assert_banded(erms, 0.03, 0.06, "unigrid(u96)-vs-octree-reference rms J gap")
    assert erms > 5 * REF_SELF_ERROR, (
        f"the cross-family gap ({erms:.3e}) collapsed toward the reference "
        f"self-error ({REF_SELF_ERROR:.3e}) — either the unigrid transport "
        "was fixed (re-record everything) or a drift equalised them by luck")


COST_RUNGS = (
    ("u24/sc", lambda: cloud_unigrid(24), "sc"),
    ("u24/chain", lambda: cloud_unigrid(24), "chain"),
    ("u48/chain", lambda: cloud_unigrid(48), "chain"),
    ("oct-d3(tau2)/chain", lambda: cloud_octree(2.0, 3), "chain"),
    ("oct-d3(tau0.5)/chain", lambda: cloud_octree(REF_TAU, 3), "chain"),
)


def test_transport_ranking_at_equal_cost(ref):
    """
    The hypothesis of ``plan/transport-ranking-plan.md``, enforced as an
    assertion: at comparable cost the accuracy order is
    **octree >= chain >= sc**.

    Measured against the octree reference (rms J error, converged, wall
    time): u24/sc 9.41e-2 (1.7 s), u24/chain 4.10e-2 (0.9 s), u48/chain
    4.03e-2 (6.8 s), oct d3(tau 2) 5.7e-3 (7.6 s), oct d3(tau 0.5) 2.4e-4.
    The unigrid numbers sit on the F11 transport plateau — refinement does
    not help them — so the octree's margin over the unigrid transports is
    structural, not a resolution accident.
    """
    ref_grid, ref_res, _t = ref
    ref_obs = observables(ref_grid, ref_res)
    by = {}
    for label, make, transport in COST_RUNGS:
        grid = make()
        res, dt = solve(grid, transport=transport)
        err_J = field_errors(grid, res["J"], ref_grid, ref_res["J"])[1]
        err_P = abs(observables(grid, res)["power"] - ref_obs["power"]) \
            / abs(ref_obs["power"])
        by[label] = (dt, err_J, err_P)
        print(f"cost {label:22s} cells/leaves={len(res['J']):7d}  {dt:7.2f}s"
              f"  J_rms err {err_J:.3e}  power err {err_P:.3e}")

    assert by["u24/chain"][1] < 0.6 * by["u24/sc"][1], (
        f"chain must clearly beat sc on u24: {by['u24/chain'][1]:.3e} vs "
        f"{by['u24/sc'][1]:.3e}")
    assert by["u24/chain"][2] < by["u24/sc"][2], \
        "chain must beat sc on u24 in absorbed power"
    assert by["u24/chain"][0] < 1.5 * by["u24/sc"][0], \
        "chain and sc must be compared at comparable cost"

    assert by["oct-d3(tau2)/chain"][0] < 3.0 * by["u48/chain"][0], \
        "octree d3 must not be much more expensive than u48"
    assert by["oct-d3(tau2)/chain"][1] < 0.5 * by["u24/chain"][1], \
        "octree d3(tau 2) must clearly beat u24/chain in J"
    assert by["oct-d3(tau2)/chain"][2] < by["u24/sc"][2], \
        "octree d3(tau 2) must beat u24/sc in absorbed power"

    best_oct = min(v[1] for k, v in by.items() if k.startswith("oct"))
    chain = by["u24/chain"][1]
    worst_sc = by["u24/sc"][1]
    assert best_oct <= chain < worst_sc, (
        f"ranking violation: octree {best_oct:.3e} >= chain {chain:.3e} "
        f">= sc {worst_sc:.3e}")


def test_delta_m_does_not_cancel_refinement(ref):
    """
    Plan section 12: the ladder reports delta-M on *and* off, so a delta-M
    bias cannot silently cancel a refinement error.

    Measured at the reference rung (oct tau 0.5 d5, converged both ways):
    delta-M off moves J by only **4.5e-4 rms** (delta-M 2 by 1.6e-4), i.e.
    ~10x *smaller* than the rung's refinement self-error (5.5e-3) — delta-M
    is not the reference's dominant error term, the recorded transport/grid
    errors are.  Its real effect is on the **iteration count**: 90 vs 17 on
    u24 (44 vs 18 on u48).  And it does not interact with the plateau: the
    u24 rung sits at 4.094e-2 (off) vs 4.103e-2 (on) rms J against the
    reference — the same F11 error either way.
    """
    ref_grid, ref_res, _t = ref
    res_off, _t_off = solve(ref_grid, delta_M=0)
    res_d2, _t_d2 = solve(ref_grid, delta_M=2)

    shift_off = rel_rms_dev(res_off["J"], ref_res["J"])
    shift_d2 = rel_rms_dev(res_d2["J"], ref_res["J"])
    assert_banded(shift_off, 1e-4, 2e-3, "delta-M off shift of the reference field")
    assert_banded(shift_d2, 0.0, 5e-4, "delta-M 2 shift of the reference field")
    assert shift_off < 0.5 * REF_SELF_ERROR, (
        f"delta-M ({shift_off:.3e}) is becoming comparable to the reference "
        f"refinement error ({REF_SELF_ERROR:.3e}) — delta-M behaviour changed")

    iters = {}
    for dm in (0, 1):
        grid = cloud_unigrid(24)
        res, _dt = solve(grid, delta_M=dm)
        iters[dm] = res["n_iter"]
    assert iters[0] > 3 * iters[1], (
        f"delta-M no longer accelerates convergence on u24: {iters[0]} vs "
        f"{iters[1]} iterations")

    errs = {}
    for dm in (0, 1):
        grid = cloud_unigrid(24)
        res, _dt = solve(grid, delta_M=dm)
        errs[dm] = field_errors(grid, res["J"], ref_grid, ref_res["J"])[1]
    assert abs(errs[0] - errs[1]) < 2e-3, (
        f"delta-M interacts with the unigrid transport error: {errs[0]:.3e}"
        f" (off) vs {errs[1]:.3e} (on)")


MC_HISTORY_RUNGS = (5000, 20000, 100000, 500000)


def test_mc_error_falls_as_1_over_sqrt_N():
    """
    The Monte Carlo order statement of plan section 3 T5, measured the way
    statistics allow: **two independent seeds per history count**, differenced
    against each other.  Their difference is free of the (N-independent)
    grid/estimator bias, so its rms falls exactly like ``N^-0.5``; a
    deviation-vs-reference fit would be swamped by the ~4 % unigrid-vs-octree
    transport bias of this case (plan finding F11, section 6 above).

    Assertions: the fitted log-log slope of the seed-split **field** error
    against the history count lies in [-0.65, -0.35] (measured -0.49).  The
    **absorbed-power** column deliberately does *not* get an N^-0.5 or a
    sigma-consistency assertion, because the measurement says it must not:
    the seed-split power error (2.5e-2 → 6.3e-3 → 1.0e-4 → 1.5e-4) is ~2.5x
    *larger* than the per-cell sigma quadrature at low N (whole-photon
    tracks correlate neighbouring cells) and ~10-20x *smaller* at high N
    (the domain-integrated power is anti-correlated across cells — nearly
    conserved by construction).  The per-cell sigma is a per-cell statement;
    only the seed split measures the aggregate (finding F9, aggregate
    side).  Recorded instead: the power error drops by >= 2.5x from N to 4N
    for the first two doublings and stays below 5e-4 at high statistics.
    """
    grid = cloud_octree(2.0, 2)
    kappa_e, kappa_s, g, B = cloud_fields(grid)
    weight = 4.0 * np.pi * (kappa_e - kappa_s) * cell_volumes(grid)

    hs, err_field, err_power, sigma_ratio = [], [], [], []
    for hist in MC_HISTORY_RUNGS:
        runs = [
            j_mc(grid, kappa_e, kappa_s, B, common_bc(bottom_I=0.5), g=g,
                 periodic_box=BOX, stage1="explicit", n_histories=0,
                 n_boundary_histories=hist, n_batches=100, seed=seed,
                 mode="cell", jit=True)
            for seed in (101, 202)
        ]
        a, b = runs
        e_field = rel_rms_dev(a["J_total"], b["J_total"]) / np.sqrt(2.0)

        pa = abs(float(np.sum(weight * a["J_total"])))
        pb = abs(float(np.sum(weight * b["J_total"])))
        e_power = abs(pa - pb) / np.sqrt(2.0) / pa
        sa = np.sqrt(np.sum((weight * a["sigma_J_total"]) ** 2)) / pa
        sb = np.sqrt(np.sum((weight * b["sigma_J_total"]) ** 2)) / pa

        hs.append(float(hist))
        err_field.append(e_field)
        err_power.append(e_power)
        sigma_ratio.append(e_power / np.hypot(sa, sb) / np.sqrt(2.0))
        print(f"MC N={hist:7d}: seed-split field err {e_field:.3e}"
              f", power err {e_power:.3e}, reported sigma {sa:.3e}/{sb:.3e}")

    assert all(x > y for x, y in zip(err_field, err_field[1:])), \
        f"MC field error not monotone in N: {err_field}"
    assert_banded(loglog_slope(hs, err_field), *MC_SLOPE_BAND,
                  "MC field error slope vs history count (expect -0.5)")
    for a, b in zip(err_power, err_power[1:3]):
        assert a > 2.5 * b, (
            "MC absorbed-power seed-split error not falling ~N^-0.5 at "
            f"low N: {err_power}")
    for hist, e in zip(MC_HISTORY_RUNGS[2:], err_power[2:]):
        assert e < 5e-4, (
            f"MC absorbed-power seed-split error {e:.3e} above the measured "
            f"high-statistics floor at N={hist}")
    for hist, ratio in zip(MC_HISTORY_RUNGS, sigma_ratio):
        if hist <= 5_000:
            assert ratio > 1.0, (
                "aggregate power error should exceed the per-cell sigma "
                f"quadrature at low N (track correlation): {ratio:.2f}")
        else:
            assert ratio < 0.8, (
                "per-cell sigma quadrature should overestimate the "
                f"aggregate power error at higher N: {ratio:.2f}")
