# -*- coding: utf-8 -*-
"""
T3 of the solver-matrix plan (plan/solver-test-matrix-plan.md section 3):
**statistical Monte Carlo <-> deterministic consistency** with the noise
model of section 6.

Unlike the existing ``test_mc_*.py`` files (which validate MC *accuracy*
against 1D references and against SC with 2-sigma bands), here the
**statistics themselves** are under test:

* **Finding F3** — the batch standard error of a sparse-collision tally is
  meaningless in cells whose rays rarely collide.  The **noise-validity
  mask** of section 6.2 (signal-relative sigma floor + expected number of
  scoring flights from the *ray optical depth*, never from the observed
  deviation) turns the heavy-tailed z distributions into a testable
  statistic.  Both sides are asserted: the masked distribution is Gaussian
  (rms z ~ 1, |z| > 3 rare), **and** the unmasked one is still pathological
  (rms z ~ 20-120, max |z| ~ 1e25 for the adjoint all-sky at tau = 0 rays) --
  otherwise the mask would be vacuous.
* **Finding F9** — pooling per-cell z-scores into a *mean* is not an
  independent-sample mean: one history contributes to many cells, so the
  naive ``std/sqrt(n)`` underestimates the standard error of the mean by a
  factor ~2.2 (measured over independent seed pairs).  Mean-z assertions are
  therefore pinned bands, not 3-sigma-of-the-mean tests.  Also measured
  here: differencing two paths at the **same seed** gives rms z = 0.44
  instead of 0.85 -- the numba and NumPy streams at one seed are partly
  correlated, so path comparisons must use different seeds.
* **Section 6.1** — fixed seeds, and every statistical statement backed by
  two independent seeds where it is cheap enough.
* **Section 6.3** — the aggregate error norm must match the aggregate sigma
  norm (measured ratio 0.998-1.000), and fall like ``N^-0.5``.

Measured reference values (2026-11-16, octree d2 = 6 952 leaves, 42
ordinates, matrix case, no beam):

===========================================  ==========================
quantity                                     measured
===========================================  ==========================
A1 per-cell sigma / J                        7.6 %
A1 seed-to-seed pooled z                     rms 0.83-0.87, mean +-0.05
A1 cell- vs global-majorant pooled z         rms 0.85-0.86
A1 jit vs NumPy path (different seeds)       rms 0.84-0.85
A1 jit vs NumPy path (same seed)             rms 0.44
A2 next-flight masked z (100 flights/ordinate) mean +0.02, rms 1.09, |z|>3 1.1 %
A2 next-flight unmasked z                    rms 17-121, |z|>3 2.0-2.9 %
A2 agg |I - I_det| / agg sigma norm          0.998 (100 flights), 1.000 (25)
adjoint all-sky masked z (6 points x 21)     mean -0.12 +- 0.09, rms 0.97, max 3.0
adjoint all-sky unmasked max |z|             2e25  (tau = 0 rays, sigma = 0)
adjoint all-sky max dev vs SC column         0.45-0.80 %
===========================================  ==========================
"""

import numpy as np
import pytest

from matrix_cases import (
    cloud_beam,
    cloud_case,
    cloud_fields,
    cloud_octree,
    common_bc,
    direction_grid,
    solve_cloud,
)
from matrix_stats import (
    assert_banded,
    combined_sigma,
    gaussian_tail,
    hypothetically_clean,
    next_flight_mask,
    ray_optical_depth_octree,
    rel_rms_dev,
    z_scores,
    z_stats,
)

from rt3d.mc.backward import backward_all_sky
from rt3d.mc.forward import j_mc, radiance_from_source

pytestmark = pytest.mark.mc

BOX = cloud_case()["box"]

A1_HISTORIES = 200_000
A1_SEEDS = (21, 22)
A1_MODE_SEEDS = (5, 6)
A1_PATH_SEEDS = (31, 131)          # must differ (finding F9)

NF_FLIGHTS = 100                   # backward flights per ray/ordinate
NF_BATCHES = 25
NF_SEEDS = (11, 12)

ADJ_SEEDS = (14, 15)
ADJ_SEEDS_COUNT = 20_000

# pooled-z band for two *independent* runs of the same estimator/mode/path
# (measured 0.83-0.86: the 100-batch sigma is ~15 % conservative)
INDEP_RMS_BAND = (0.65, 1.05)
INDEP_MEAN_BAND = 0.10


@pytest.fixture(scope="module")
def oct2():
    return cloud_octree(2.0, 2)


@pytest.fixture(scope="module")
def sc_solved(oct2):
    """Converged SC solve of the matrix case on octree d2 (no beam)."""
    return solve_cloud(oct2, ng=True, delta_M=1, beam=False, n_iter=120,
                       tol=1e-7)


@pytest.fixture(scope="module")
def tau_ray(oct2, sc_solved):
    """Ray optical depth of exactly the stage-2/adjoint ray set (F3 mask input)."""
    return ray_optical_depth_octree(oct2, direction_grid(1),
                                    sc_solved["kappa_e_delta"], BOX)


def _a1(oct2, seed, histories=A1_HISTORIES, mode="cell", jit=True):
    kappa_e, kappa_s, g, B = cloud_fields(oct2)
    return j_mc(oct2, kappa_e, kappa_s, B, common_bc(bottom_I=0.5), g=g,
                periodic_box=BOX, stage1="explicit", n_histories=0,
                n_boundary_histories=histories, n_batches=100, seed=seed,
                mode=mode, jit=jit)


def _pooled_z(a, b, key="J_total", sigma_key="sigma_J_total"):
    """Pooled z statistics of two independent tallies of the same quantity."""
    sig = combined_sigma(a[sigma_key], b[sigma_key])
    z = z_scores(a[key] - b[key], sig, np.zeros_like(sig))
    ok = sig > 1e-6 * float(np.max(np.abs(a[key])))
    return z_stats(z[ok]), z


def _assert_independent_pair(stats, label):
    """Pooled z of two independent MC runs: rms and mean bands (F9)."""
    assert stats["n"] > 1000, f"{label}: only {stats['n']} usable points"
    assert_banded(stats["rms"], *INDEP_RMS_BAND, f"{label}: rms z")
    assert_banded(abs(stats["mean"]), 0.0, INDEP_MEAN_BAND, f"{label}: |mean z|")
    assert_banded(stats["tail3"], 0.0, 0.03, f"{label}: |z|>3 fraction")
    assert_banded(stats["tail2"], 0.0, 0.12, f"{label}: |z|>2 fraction")
    return stats


# =============================================================================
# 0. the statistics machinery itself (synthetic data, zero solver cost)
# =============================================================================

def test_noise_model_helpers_selftest():
    """
    The z / tail helpers must recognise clean Gaussian data and reject heavy
    tails -- otherwise a green masked-z test below would mean nothing.
    """
    rng = np.random.default_rng(2026)
    clean = rng.standard_normal(200_000)
    s = z_stats(clean)
    assert hypothetically_clean(s)
    assert abs(s["mean"]) < 0.01 and 0.95 < s["rms"] < 1.05
    assert abs(s["tail3"] - gaussian_tail(3.0)) < 2e-3

    heavy = np.concatenate([clean[:190_000], 60.0 * rng.standard_normal(10_000)])
    assert not hypothetically_clean(z_stats(heavy))
    with pytest.raises(AssertionError):
        _assert_independent_pair(z_stats(heavy), "synthetic heavy tail")


def test_next_flight_mask_is_informative():
    """
    The mask must drop points whose rays essentially never collide -- where
    the batch sigma is 0-or-nothing noise -- and keep points with a
    trustworthy tally, **without ever looking at the deviation**.
    """
    sigma = np.array([0.0, 1e-9, 1e-3, 1e-3, 5e-2])
    signal = np.full(5, 1.0)
    tau = np.array([0.0, 10.0, 0.01, 1.0, 5.0])
    keep = next_flight_mask(sigma, signal, tau, n_histories=25,
                            eps_rel=1e-6, min_scoring=8.0)
    assert keep.tolist() == [False, False, False, True, True]
    # with 25 flights a point needs tau >~ 0.35 to be expected to score at all
    assert not next_flight_mask(np.ones(1), np.ones(1), np.array([0.05]),
                                n_histories=25)[0]
    assert next_flight_mask(np.ones(1), np.ones(1), np.array([1.0]),
                            n_histories=25)[0]
    # a zero-sigma point is always dropped, whatever the signal
    assert not next_flight_mask(np.array([0.0]), np.ones(1), np.array([9.0]),
                                n_histories=1000)[0]


# =============================================================================
# 1. A1 (j_mc, explicit variant): reproducibility, mode and path parity
# =============================================================================

def test_a1_same_seed_is_bit_identical(oct2):
    """Section 6.1: the same seed must reproduce the tally bit-for-bit."""
    a = _a1(oct2, 5)
    b = _a1(oct2, 5)
    assert np.array_equal(a["J_total"], b["J_total"])
    assert np.array_equal(a["sigma_J_total"], b["sigma_J_total"])


def test_a1_seed_to_seed_pooled_z(oct2):
    """
    Two independent seeds of one estimator must agree within their combined
    batch statistics.  Measured rms z 0.83-0.87, i.e. the 100-batch sigma is
    ~15 % conservative -- pinned, because a *rising* rms would mean the
    error bars are lying.

    The mean z is asserted as a band (|mean z| <= 0.10), **not** as
    "within 3 standard errors of 0": the per-cell errors are spatially
    correlated (one history crosses many cells), and the measured spread of
    mean z over independent pairs is 0.023 while ``std/sqrt(n)`` says 0.010
    (finding F9).
    """
    stats, z = _pooled_z(_a1(oct2, A1_SEEDS[0]), _a1(oct2, A1_SEEDS[1]))
    _assert_independent_pair(stats, "A1 seed-to-seed")


def test_a1_majorant_cell_vs_global_mode(oct2):
    """
    ``mode='cell'`` (per-cell majorant + null collisions) and
    ``mode='global'`` (single majorant, Woodcock tracking on one bound) are
    different samplers of the same tally; differenced at *different* seeds
    they must land in the same pooled-z band as an independent pair
    (measured 0.85 / 0.86).
    """
    ref = _a1(oct2, A1_MODE_SEEDS[0], mode="cell")
    for mode in ("cell", "global"):
        other = _a1(oct2, A1_MODE_SEEDS[1], mode=mode)
        stats, _ = _pooled_z(ref, other)
        _assert_independent_pair(stats, f"A1 cell-vs-{mode} majorant")


def test_a1_jit_path_matches_numpy_path(oct2):
    """
    The numba fast path and the NumPy reference path are two independent
    samplers (M4/M5: deliberately different streams), so parity is
    *statistical*: pooled z in the independent-pair band, no systematic
    offset, and the fields equal to within the per-cell noise of the tally
    itself -- measured rms relative deviation **8.1 %**, which is what two
    independent A1 runs at 2e5 boundary histories give (their per-cell
    relative sigma is 7.6 %).  Anyone who wants a sharper path comparison
    needs many more histories, not a different reference.
    """
    jit_run = _a1(oct2, A1_PATH_SEEDS[0], jit=True)
    numpy_run = _a1(oct2, A1_PATH_SEEDS[1], jit=False)
    assert not np.array_equal(jit_run["J_total"], numpy_run["J_total"])
    stats, _ = _pooled_z(jit_run, numpy_run)
    _assert_independent_pair(stats, "A1 jit-vs-NumPy (different seeds)")
    assert_banded(rel_rms_dev(jit_run["J_total"], numpy_run["J_total"]),
                  0.02, 0.15, "A1 jit vs NumPy rms relative deviation of J")


def test_a1_same_seed_paths_are_correlated(oct2):
    """
    Finding F9, the counterpart of the test above: at the **same** seed the
    numba and NumPy streams partly coincide, so their difference is *not*
    an independent pair (measured rms z = 0.44 vs 0.85 for different seeds).

    Pinned so that a change which (say) decorrelates the streams, or which
    makes the two paths share draws exactly, fails loudly -- and so that
    nobody validates path parity with one seed and then trusts the
    sigma-normalised deviation.
    """
    a = _a1(oct2, 7, jit=True)
    b = _a1(oct2, 7, jit=False)
    stats, _ = _pooled_z(a, b)
    assert_banded(stats["rms"], 0.30, 0.60,
                  "A1 jit-vs-NumPy rms z at identical seed (stream overlap)")
    assert stats["rms"] < 0.75 * INDEP_RMS_BAND[0], (
        "same-seed paths no longer correlated -- the stream-mapping changed")


# =============================================================================
# 2. A2 next-flight vs the zero-noise reference, with the F3 mask
# =============================================================================

def _stage2_pair(oct2, sc_solved, seed, flights=NF_FLIGHTS, batches=NF_BATCHES):
    """Deterministic line integral + next-flight tally of the same source."""
    dg = direction_grid(1)
    bc = common_bc(bottom_I=0.5)
    common = dict(kappa_e=sc_solved["kappa_e_delta"], S_total=sc_solved["S"],
                  bc=bc, periodic_box=BOX)
    ref = radiance_from_source(oct2, dg, estimator="deterministic", **common)
    nf = radiance_from_source(oct2, dg, estimator="next_flight",
                              n_histories=flights, n_batches=batches,
                              seed=seed, jit=True, **common)
    return ref, nf


@pytest.mark.parametrize("seed", NF_SEEDS)
def test_a2_next_flight_masked_z_vs_reference(oct2, sc_solved, tau_ray, seed):
    """
    Next-flight stage-2 tally against the deterministic line integral of the
    same source (zero noise), evaluated with the section 6.2 mask.

    Masked (measured at 100 flights/ordinate): mean z +0.02 (a small
    *positive* offset, consistent with the Jensen bias of 1/sigma_hat at 25
    batches), rms z 1.09, |z| > 3 in 1.1 % of points, max |z| ~ 15-32.
    Unmasked: rms z 17-121 and |z| > 3 in ~2.4 % -- the F3 pathology is
    still present, which is what makes the mask non-vacuous.
    """
    ref, nf = _stage2_pair(oct2, sc_solved, seed)
    keep = next_flight_mask(nf["sigma_I"], nf["I"], tau_ray,
                            n_histories=NF_FLIGHTS,
                            min_scoring=0.16 * NF_FLIGHTS)
    z = z_scores(nf["I"], nf["sigma_I"], ref["I"])

    assert keep.mean() > 0.3, f"mask keeps only {100 * keep.mean():.1f}%"
    masked = z_stats(z[keep])
    assert masked["n"] > 0.3 * z.size
    assert_banded(abs(masked["mean"]), 0.0, 0.08,
                  f"A2 next-flight masked mean z (seed {seed})")
    assert_banded(masked["rms"], 0.85, 1.35,
                  f"A2 next-flight masked rms z (seed {seed})")
    assert_banded(masked["tail3"], 0.0, 0.025,
                  f"A2 next-flight masked |z|>3 (seed {seed})")

    unmasked = z_stats(z)
    assert unmasked["rms"] > 5.0 * masked["rms"], (
        f"unmasked rms {unmasked['rms']:.1f} no longer much worse than masked "
        f"{masked['rms']:.2f} -- did the estimator or the mask change?")
    assert unmasked["max_abs"] > 100.0 * masked["max_abs"]
    assert_banded(unmasked["tail3"], 0.015, 0.035,
                  f"A2 next-flight unmasked |z|>3 fraction (seed {seed})")


def test_a2_aggregate_error_matches_sigma_norm(oct2, sc_solved):
    """
    Section 6.3's headline, insensitive to the tail problem: the aggregate
    norm of (MC - reference) must equal the aggregate norm of the reported
    sigma.  Measured ratio **0.998** (100 flights) and **1.000** (25
    flights) -- the tally error is exactly where the error bars say it is,
    in aggregate, even though the per-point z distribution is heavy-tailed.

    The same pair also gives the Monte Carlo order at fixed grid: the error
    ratio 100/25 flights is 0.500 = 1/sqrt(4).
    """
    coarse = _stage2_pair(oct2, sc_solved, NF_SEEDS[0], flights=25, batches=25)
    fine = _stage2_pair(oct2, sc_solved, NF_SEEDS[0], flights=NF_FLIGHTS,
                        batches=NF_BATCHES)
    devs, ratios = {}, {}
    for name, (ref, nf) in (("25", coarse), (str(NF_FLIGHTS), fine)):
        dev = rel_rms_dev(nf["I"], ref["I"])
        pred = np.sqrt(np.mean(nf["sigma_I"] ** 2)) / np.sqrt(np.mean(ref["I"] ** 2))
        devs[name], ratios[name] = dev, dev / pred
        assert_banded(ratios[name], 0.90, 1.10,
                      f"next-flight agg deviation / agg sigma ({name} flights)")
    assert_banded(devs[str(NF_FLIGHTS)] / devs["25"], 0.40, 0.62,
                  "next-flight error ratio 25 -> 100 flights (expect 0.50)")


# =============================================================================
# 3. B (exact-HG adjoint all-sky) vs the SC columns, masked
# =============================================================================

def _adjoint_points(oct2, fracs=(0.98, 0.90, 0.80, 0.70, 0.60, 0.50)):
    """Leaf centers at the given box-height fractions (top of cloud -> below)."""
    z0, z1 = BOX.z0, BOX.z1
    return [int(np.argmin(np.abs(oct2.centers[:, 2] - (z0 + f * (z1 - z0)))))
            for f in fracs]


@pytest.mark.parametrize("seed", ADJ_SEEDS)
def test_backward_all_sky_masked_z_vs_sc_columns(oct2, sc_solved, tau_ray, seed):
    """
    Adjoint all-sky radiance at six leaf centers, per upward icosahedral
    ordinate (``cone_scale = 0``, exact node directions -- the mode that is
    directly comparable with an SC ``I[leaf, ordinate]`` column), differenced
    against the converged SC columns with per-ordinate sigma and the section
    6.2 mask (a point on a top-layer leaf looks up through tau = 0 rays whose
    sigma is exactly 0, and the unmasked z there is ~1e25).

    Measured over 6 points x 21 ordinates (114 kept points): mean z -0.12
    +- 0.09, rms z 0.97, |z| > 3 in 0 %, max |z| 3.0.  The deviation itself
    is far smaller than sigma -- 0.45-0.80 % max against the 42-ordinate SC
    column -- so this test says "the adjoint tally is consistent with SC to
    well within its own noise", which is the strongest statement available
    while the SC angular error (T5) is the larger term.
    """
    dg = direction_grid(1)
    kappa_e, kappa_s, g, _B = cloud_fields(oct2)
    bc = common_bc(bottom_I=0.5)
    up = dg.up_idx

    zs, kept, max_unmasked, devs = [], 0, 0.0, []
    for leaf in _adjoint_points(oct2):
        rs = backward_all_sky(oct2, kappa_e, kappa_s, _B, bc, dg,
                              tuple(oct2.centers[leaf]), g=g, periodic_box=BOX,
                              n_seeds=ADJ_SEEDS_COUNT, n_batches=100,
                              seed=seed, jit=True)
        assert rs["up_idx"].tolist() == up.tolist()
        Iref = sc_solved["I"][leaf, up]
        z = (rs["I"] - Iref) / np.maximum(rs["sigma_I"], 1e-30)
        keep = next_flight_mask(rs["sigma_I"], rs["I"], tau_ray[leaf, up],
                                n_histories=ADJ_SEEDS_COUNT, min_scoring=10.0)
        zs.append(z[keep])
        kept += int(keep.sum())
        max_unmasked = max(max_unmasked, float(np.max(np.abs(z))))
        devs.append(np.max(np.abs(rs["I"] - Iref)) / np.max(np.abs(Iref)))

    stats = z_stats(np.concatenate(zs))
    assert kept >= 100 and stats["n"] == kept
    assert_banded(stats["rms"], 0.60, 1.40,
                  f"adjoint all-sky masked rms z (seed {seed})")
    assert_banded(abs(stats["mean"]), 0.0, 0.35,
                  f"adjoint all-sky masked mean z (seed {seed})")
    assert_banded(stats["tail3"], 0.0, 0.05,
                  f"adjoint all-sky masked |z|>3 (seed {seed})")
    assert max_unmasked > 1e6, (
        f"unmasked max |z| = {max_unmasked:.3g}: the tau -> 0 pathology has "
        f"vanished -- mask or estimator changed")
    for leaf, d in zip(_adjoint_points(oct2), devs):
        assert_banded(d, 0.0, 0.02,
                      f"adjoint all-sky max deviation vs SC column (leaf {leaf})")


def test_backward_all_sky_sigma_shrinks_like_1_over_sqrt_n(oct2):
    """
    The adjoint all-sky sigma must fall like ``1/sqrt(n_seeds)`` (the MC
    order statement of section 3 T5, done here cheaply at fixed grid).
    Measured ratio at 2e4 -> 2e5 seeds: 0.32 (1/sqrt(10) = 0.316).
    """
    dg = direction_grid(1)
    kappa_e, kappa_s, g, _B = cloud_fields(oct2)
    bc = common_bc(bottom_I=0.5)
    leaf = _adjoint_points(oct2)[2]
    point = tuple(oct2.centers[leaf])
    kw = dict(g=g, periodic_box=BOX, n_batches=100, seed=3, jit=True)
    s_small = backward_all_sky(oct2, kappa_e, kappa_s, _B, bc, dg, point,
                               n_seeds=20_000, **kw)
    s_big = backward_all_sky(oct2, kappa_e, kappa_s, _B, bc, dg, point,
                             n_seeds=200_000, **kw)
    sel = (s_small["sigma_I"] > 0) & (s_big["sigma_I"] > 0)
    assert np.count_nonzero(sel) >= 5
    assert_banded(float(np.mean(s_big["sigma_I"][sel]
                                / s_small["sigma_I"][sel])), 0.20, 0.45,
                  "adjoint sigma ratio for 10x more seeds (expect 0.316)")


# =============================================================================
# 4. M6 all-sky ground: MC vs SC on the full case (plan section 7)
# =============================================================================
# The all-sky configuration -- top inflow, 30-deg beam, absorbing/scattering
# cloud, and a Lambertian/emissive ground (albedo 0.3, eps B_g = 1) -- was
# MC-forbidden until phase M6.  The MC ground law itself is pinned
# analytically in tests/test_mc_ground.py; what this section pins is the
# *cross-family* behaviour on the hard case, and the statistical honesty of
# the merged ground/beam tallies.
#
# Measured (oct2, seeds 101/202, 120k histories per population, **after
# the F12 uninitialised-buffer fix in the segment marcher** — before it,
# axis-parallel directions (beam dy = 0 exactly) mixed heap garbage into
# face crossing times and every aggregate here wobbled +-5 % run to run):
#   pooled seed-to-seed per-cell z: mean -0.016, rms 0.635, |z|>3 0
#       -- the merged-tally sigma is valid (mildly conservative), like the
#       A1 band of section 1; seed-split aggregate power noise is 1.8x the
#       per-cell quadrature -- a real but modest ground-track correlation
#       (finding F9 aggregate side, same order as the no-ground ladder);
#   convention: MC 'J_total' includes the uncollided beam column, SC 'J'
#       does not -- that term alone is +15.0 (6.8 %) of the power, so SC
#       gets the analytic F0 exp(-tau_b)/4pi column before differencing;
#   with conventions matched: MC 231.1 vs SC(dM1) 236.2, i.e. -2.15 %;
#       per-cell z vs SC: mean -0.30, rms 0.80, |z|>3 0.1 % -- a coherent
#       offset at the quarter-sigma level, not noise. SC is grid-stable
#       (221.3 -> 222.9 power over tau 2 d2 -> tau 0.5 d3, plus direct).

GROUND_HISTORIES = 120_000
GROUND_SEEDS = (101, 202)
GROUND_B = 1.0


def _bc_ground():
    from rt3d.geometry import ground_surface
    from rt3d.transport import boundary_conditions

    return boundary_conditions(
        top_I=1.0,
        ground=ground_surface(albedo=0.3, emissivity=1.0, B_g=GROUND_B))


def _beam():
    return cloud_beam()


@pytest.fixture(scope="module")
def sc_ground(oct2):
    """Converged SC solve of the all-sky ground case on oct2."""
    return solve_cloud(oct2, bc="ground", albedo=0.3, emissivity=1.0,
                       B_g=GROUND_B, beam=True, ng=True, delta_M=1,
                       n_iter=150, tol=1e-8)


@pytest.fixture(scope="module")
def ground_runs(oct2):
    """Two independent explicit-MC runs of the all-sky case (cached)."""
    kappa_e, kappa_s, g, B = cloud_fields(oct2)
    return [
        j_mc(oct2, kappa_e, kappa_s, B, _bc_ground(), g=g, beam=_beam(),
             periodic_box=BOX, stage1="explicit",
             n_histories=GROUND_HISTORIES,
             n_boundary_histories=GROUND_HISTORIES,
             n_beam_histories=GROUND_HISTORIES,
             n_ground_histories=GROUND_HISTORIES,
             n_batches=300, seed=seed, jit=True)
        for seed in GROUND_SEEDS
    ]


def test_ground_merged_tallies_have_valid_sigma(ground_runs):
    """
    The M6 ground/beam populations are merged into one history index space
    before batching -- this test fails loudly if a population drops out of
    the batch statistics (the pooled z of two independent runs would then
    have rms >> 1).  Measured rms 0.64, in the same mildly-conservative
    band as the A1 pair of section 1.
    """
    a, b = ground_runs
    stats, _ = _pooled_z(a, b)
    assert stats["n"] == len(a["J_total"]), "usable-point loss in the ground case"
    assert_banded(stats["rms"], 0.45, 1.10, "all-sky ground seed-pair pooled z rms")
    assert_banded(abs(stats["mean"]), 0.0, 0.10, "all-sky ground seed-pair |mean z|")
    assert_banded(stats["tail3"], 0.0, 0.01, "all-sky ground seed-pair |z|>3")


def test_ground_crossfamily_observables(oct2, sc_ground, ground_runs):
    """
    Absorbed power and mean J of the MC all-sky solution vs the SC
    all-sky solution on the *same* oct2 leaves, conventions matched (see
    header).  Bands pinned from measurement: MC power sits ~2 % **below**
    SC with ~1.5 % seed uncertainty; the agreement is dominated by a
    coherent (non-noise) offset, so the honest band is the measured one,
    not "they agree".
    """
    from rt3d.diagnostics import cell_volumes
    from rt3d.mc.forward import _beam_j_analytic

    kappa_e, kappa_s, _g, _B = cloud_fields(oct2)
    vol = cell_volumes(oct2)
    weight = 4.0 * np.pi * (kappa_e - kappa_s) * vol

    # convention match: MC 'J_total' carries the uncollided beam column
    # ('J_direct' tally), the SC 'J' does not (the SC solver accounts for the
    # beam only through its first scattering, 'beam_scat', in the source).
    # Give SC the analytic uncollided column F0 exp(-tau_b)/4pi -- the same
    # field the MC 'J_direct_analytic' key reports.
    sun = _beam()["omega_sun"]
    J_sc = sc_ground["J"] + _beam_j_analytic(oct2, sun, kappa_e,
                                             _beam()["F0"], periodic_box=BOX)

    Jmc = np.mean([r["J_total"] for r in ground_runs], axis=0)
    P_mc = float(np.sum(weight * Jmc))
    P_sc = float(np.sum(weight * J_sc))
    dP = (P_mc - P_sc) / P_sc
    assert_banded(dP, -0.05, 0.0, "all-sky ground: MC vs SC absorbed power")

    seed_P = np.array([float(np.sum(weight * r["J_total"]))
                       for r in ground_runs])
    seed_split = abs(seed_P[0] - seed_P[1]) / np.sqrt(2.0)
    quad = float(np.sqrt(np.sum((weight * ground_runs[0]["sigma_J_total"]) ** 2)))
    assert_banded(seed_split / quad, 0.5, 3.0,
                  "all-sky ground: seed-split / per-cell-quadrature power noise")

    meanJ_mc = float(np.mean(Jmc))
    meanJ_sc = float(np.mean(J_sc))
    assert_banded((meanJ_mc - meanJ_sc) / meanJ_sc, -0.04, 0.0,
                  "all-sky ground: MC vs SC leaf-mean J")

    z = z_scores(ground_runs[0]["J_total"] - J_sc,
                 ground_runs[0]["sigma_J_total"], np.zeros_like(J_sc))
    ok = ground_runs[0]["sigma_J_total"] > 1e-6 * float(np.max(np.abs(J_sc)))
    stats = z_stats(z[ok])
    assert stats["n"] > 0.9 * J_sc.size
    assert_banded(abs(stats["mean"]), 0.0, 0.6,
                  "all-sky ground: per-cell mean z vs SC column")
    assert_banded(stats["rms"], 0.5, 1.2,
                  "all-sky ground: per-cell rms z vs SC column")
    assert_banded(stats["tail3"], 0.0, 0.01,
                  "all-sky ground: per-cell |z|>3 vs SC column")
    print(f"all-sky ground: P_mc {P_mc:.1f} (seed split {seed_split:.1f}), "
          f"P_sc {P_sc:.1f}, dP {100 * dP:+.2f} %; mean J "
          f"{meanJ_mc:.4f} vs {meanJ_sc:.4f}; z vs SC rms {stats['rms']:.2f}")
