# -*- coding: utf-8 -*-
"""
Statistical helpers for the solver-matrix tests (plan/solver-test-matrix-plan.md
section 6): z-scores against a deterministic reference, the **noise-validity
mask** that makes per-cell z statistics meaningful, pooled chi-square, tail
fractions and convergence-order fits.

Why the mask exists (plan finding F3): the stage-2 next-flight estimator gives
every ray `n_histories` backward flights that score the source at their first
*real* collision and 0 when they exit first.  The batch standard error is then
grossly optimistic in optically thin cells (0-or-1 scoring flights per ray), so
the per-cell z-scores there have rms ~40 and max ~1e4 even though the solver is
correct.  A z test is only meaningful where
``n_histories * (1 - exp(-tau_ray)) >= min_scoring``
(i.e. the expected number of scoring flights per ray is large enough for the
batch estimator), and where the reported sigma is above a signal-relative
floor.
"""

import numpy as np

TINY = 1e-30


# =============================================================================
# error measures
# =============================================================================


def rel_max_dev(est, ref):
    """max |est - ref| / max |ref| (the convention of scripts/mc_compare.py)."""
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if est.shape != ref.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {ref.shape}")
    return float(np.max(np.abs(est - ref)) / max(np.max(np.abs(ref)), TINY))


def rel_rms_dev(est, ref):
    """rms(est - ref) / rms(ref) — insensitive to single-cell outliers."""
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(
        np.sqrt(np.mean((est - ref) ** 2)) / max(np.sqrt(np.mean(ref**2)), TINY)
    )


def z_scores(est, sigma, ref):
    """z = (est - ref)/sigma; NaN where sigma == 0 (no information)."""
    est = np.asarray(est, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ref = np.asarray(ref, dtype=float)
    z = np.full(est.shape, np.nan)
    ok = sigma > 0.0
    z[ok] = (est[ok] - ref[ok]) / sigma[ok]
    return z


def combined_sigma(s1, s2):
    """Uncertainty of a difference of two independent estimates."""
    return np.sqrt(np.asarray(s1, dtype=float) ** 2 + np.asarray(s2, dtype=float) ** 2)


# =============================================================================
# cross-grid sampling and characteristic cell size (refinement ladder, T5)
# =============================================================================


def cell_scale(grid):
    """
    Characteristic cell size of a grid: the unigrid spacing, or the **finest**
    leaf edge of an octree (``2*h0 / 2**max_level``).

    Used as the ``h`` of a convergence-order fit.  Note that the octree's
    *boundary* faces keep base-size leaves, so quantities evaluated on a face
    refine with ``h0`` instead -- see plan finding F7 and
    `test_matrix_closure.py`.
    """
    from rt3d.octree import octree_grid

    if isinstance(grid, octree_grid):
        return float(2.0 * grid.h0 / 2 ** int(grid.max_level))
    return float(min(grid.hx, grid.hy, grid.hz))


def h_equivalent(grid):
    """
    Spacing of a uniform grid with the same number of cells: ``(V / n)**(1/3)``.

    The cost-matched resolution measure of the refinement ladder: it makes an
    octree rung and a unigrid rung comparable (same work per solve), and it is
    the right variable for a fit over an octree ladder refined by the
    optical-depth criterion, where the finest leaf edge saturates while the
    cell count keeps growing (plan finding F10).
    """
    from rt3d.octree import octree_grid

    if isinstance(grid, octree_grid):
        box = grid.cuboid
        volume = box.Lx * box.Ly * box.Lz
        n = grid.n_leaf
    else:
        (x0, x1), (y0, y1), (z0, z1) = grid.box
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        n = grid.n_cells
    return float((volume / n) ** (1.0 / 3.0))


def sample_at(grid, ref_grid, ref_field):
    """
    Piecewise-constant sampling of a per-cell field of `ref_grid` at the cell
    centers of `grid` (octree: `octree_grid.locate`; unigrid: its own
    ``locate``).  Both grids must cover the same box, which is the case for
    every rung of the ladder.
    """
    ref_field = np.asarray(ref_field, dtype=float)
    idx = ref_grid.locate(grid.centers)
    if ref_field.ndim == 1:
        return ref_field[idx]
    return ref_field[idx]


def field_errors(grid, field, ref_grid, ref_field):
    """
    ``(max, rms)`` relative deviation of a per-cell field against the same
    field of a finer reference grid, sampled at this grid's cell centers.
    """
    ref = sample_at(grid, ref_grid, ref_field)
    return rel_max_dev(field, ref), rel_rms_dev(field, ref)


# =============================================================================
# ray optical depth (the noise-validity criterion of the mask)
# =============================================================================


def _block_sum(values, offsets):
    """
    Sum `values` over blocks given by `offsets` (``n_blocks + 1`` entries),
    ignoring empty blocks.  `np.add.reduceat` would misbehave on repeated
    offsets, so the non-empty blocks are the only ones passed through.
    """
    counts = np.diff(offsets)
    out = np.zeros(counts.size)
    nz = counts > 0
    if np.any(nz):
        starts = np.asarray(offsets[:-1])[nz]
        out[nz] = np.add.reduceat(values, starts)
    return out


def ray_optical_depth_octree(oct, dg, kappa_e, periodic_box=None):
    """
    Optical depth from every leaf center to the entry face, per ordinate.

    Uses the same exact characteristic march as the SC sweep and the MC
    stage-2 walk (`precompute_octree_march`, reference NumPy march), so
    ``tau[i, d]`` is the ``tau_ray`` of the next-flight estimator.

    Returns
    -------
    (n_leaf, n_dir) ndarray
    """
    from rt3d.transport import precompute_octree_march

    kappa_e = np.asarray(kappa_e, dtype=float)
    out = np.zeros((oct.n_leaf, dg.n_dir))
    for d in range(dg.n_dir):
        march = precompute_octree_march(
            oct, dg.directions[d], periodic_box=periodic_box, jit=False
        )
        tau_seg = kappa_e[march["leaf"]] * march["length"]
        out[:, d] = _block_sum(tau_seg, march["offsets"])
    return out


def ray_optical_depth_unigrid(grid, dg, kappa_e, periodic_box=None):
    """
    Optical depth from every cell center to the entry face, per ordinate, on
    the unigrid (lattice DDA march along ``-omega``, the stage-2 convention).

    Returns
    -------
    (n_cells, n_dir) ndarray
    """
    from rt3d.geometry import march_lattice

    kappa_e = np.asarray(kappa_e, dtype=float)
    out = np.zeros((grid.n_cells, dg.n_dir))
    for d in range(dg.n_dir):
        steps, _exit_t, _ax, _sd = march_lattice(
            grid.centers, -np.asarray(dg.directions[d], dtype=float), grid,
            periodic_box=periodic_box,
        )
        acc = np.zeros(grid.n_cells)
        for ray_idx, cell_idx, length in steps:
            np.add.at(acc, ray_idx, kappa_e[cell_idx] * length)
        out[:, d] = acc
    return out


def ray_optical_depth(grid, dg, kappa_e, periodic_box=None):
    """Dispatch to the octree/unigrid ray optical depth."""
    from rt3d.octree import octree_grid

    if isinstance(grid, octree_grid):
        return ray_optical_depth_octree(grid, dg, kappa_e, periodic_box)
    return ray_optical_depth_unigrid(grid, dg, kappa_e, periodic_box)


# =============================================================================
# noise-validity mask
# =============================================================================


def next_flight_mask(sigma_I, signal, tau_ray, n_histories,
                     eps_rel=1e-6, min_scoring=8.0):
    """
    Boolean mask of cells/ordinates where the next-flight z statistic is
    trustworthy.

    A point is kept when

    * the reported sigma exceeds ``eps_rel * max|I|`` (signal-relative floor,
      so a numerically-degenerate batch variance cannot produce a huge z), and
    * the expected number of *scoring* flights per ray,
      ``n_histories * (1 - exp(-tau_ray))``, is at least ``min_scoring`` —
      otherwise the batch standard error underestimates the sparse-hit
      variance (plan finding F3).

    Parameters
    ----------
    sigma_I : ndarray, shape of the tally
    signal : ndarray, same shape
        The estimate itself (defines the sigma floor).
    tau_ray : ndarray, same shape
        Optical depth from the tally point to the entry face.
    n_histories : int
        Backward flights per ray.
    """
    sigma_I = np.asarray(sigma_I, dtype=float)
    signal = np.asarray(signal, dtype=float)
    tau = np.asarray(tau_ray, dtype=float)
    floor = eps_rel * max(float(np.max(np.abs(signal))), TINY)
    p_score = -np.expm1(-np.maximum(tau, 0.0))
    return (sigma_I > floor) & (n_histories * p_score >= min_scoring)


# =============================================================================
# pooled statistics
# =============================================================================


def z_stats(z):
    """
    Pooled summary of a z-score array (NaNs/infs dropped).

    Returns
    -------
    dict with 'n', 'mean', 'se_mean', 'rms', 'chi2', 'tail2', 'tail3',
    'tail5', 'max_abs'.
    """
    z = np.asarray(z, dtype=float).ravel()
    z = z[np.isfinite(z)]
    n = int(z.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "se_mean": np.nan, "rms": np.nan,
                "chi2": np.nan, "tail2": np.nan, "tail3": np.nan,
                "tail5": np.nan, "max_abs": np.nan}
    mean = float(z.mean())
    rms = float(np.sqrt(np.mean(z**2)))
    se = float(z.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {
        "n": n,
        "mean": mean,
        "se_mean": se,
        "rms": rms,
        "chi2": float(np.mean(z**2)),
        "tail2": float(np.mean(np.abs(z) > 2.0)),
        "tail3": float(np.mean(np.abs(z) > 3.0)),
        "tail5": float(np.mean(np.abs(z) > 5.0)),
        "max_abs": float(np.max(np.abs(z))),
    }


def gaussian_tail(k):
    """Two-sided tail probability of the standard normal beyond |z| > k."""
    import math

    return math.erfc(k / math.sqrt(2.0))


def hypothetically_clean(stats, n_min=None):
    """
    Structural sanity of a z distribution: mean consistent with 0 and rms in
    the Gaussian band.  Used as a self-test of the statistics helpers
    themselves (synthetic Gaussian data must pass, heavy-tailed data must
    not).
    """
    n = stats["n"]
    if n == 0 or not np.isfinite(stats["rms"]):
        return False
    mean_ok = abs(stats["mean"]) <= 3.0 * stats["se_mean"] if n > 1 else True
    return bool(mean_ok and 0.7 <= stats["rms"] <= 1.6)


# =============================================================================
# convergence-order fits
# =============================================================================


def loglog_slope(x, y):
    """
    Least-squares slope of ``log y`` against ``log x`` (x, y > 0 entries
    only).  First-order-in-h transport schemes give ~1.0; Monte Carlo
    convergence gives ~-0.5 against the history count.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        raise ValueError("need at least two positive (x, y) pairs")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


# =============================================================================
# assertion helpers (report the measured value on failure)
# =============================================================================


def assert_banded(value, lo, hi, label):
    """Assert ``lo <= value <= hi`` with a message that shows the value."""
    value = float(value)
    if not (lo <= value <= hi):
        raise AssertionError(
            f"{label}: measured {value:.6g}, outside the pinned band "
            f"[{lo:.6g}, {hi:.6g}] (plan/solver-test-matrix-plan.md section 2)"
        )


def assert_rel_close(est, ref, tol, label):
    """Assert the max relative deviation is <= tol."""
    d = rel_max_dev(est, ref)
    if not (d <= tol):
        raise AssertionError(f"{label}: max|est-ref|/max|ref| = {d:.3e} > {tol:.3e}")
    return d


def assert_rms_close(est, ref, tol, label):
    """Assert the rms relative deviation is <= tol."""
    d = rel_rms_dev(est, ref)
    if not (d <= tol):
        raise AssertionError(f"{label}: rms(est-ref)/rms(ref) = {d:.3e} > {tol:.3e}")
    return d


def assert_z_consistent(z, label, rms_hi=1.6, rms_lo=0.7, mean_sigmas=3.0,
                        tail3_hi=None):
    """
    Assert a pooled z distribution is statistically consistent: mean within
    ``mean_sigmas`` standard errors of 0, rms inside [rms_lo, rms_hi], and
    optionally the |z| > 3 tail fraction below ``tail3_hi``.
    """
    s = z_stats(z)
    if s["n"] < 10:
        raise AssertionError(f"{label}: only {s['n']} usable z values")
    if s["n"] > 1 and abs(s["mean"]) > mean_sigmas * s["se_mean"]:
        raise AssertionError(
            f"{label}: mean z = {s['mean']:+.3f} +- {s['se_mean']:.3f} "
            f"({s['mean'] / s['se_mean']:+.1f} sigma from 0)"
        )
    if not (rms_lo <= s["rms"] <= rms_hi):
        raise AssertionError(
            f"{label}: rms z = {s['rms']:.3f}, outside [{rms_lo}, {rms_hi}]"
        )
    if tail3_hi is not None and s["tail3"] > tail3_hi:
        raise AssertionError(
            f"{label}: |z|>3 fraction {100 * s['tail3']:.2f}% > "
            f"{100 * tail3_hi:.2f}% (Gaussian {100 * gaussian_tail(3):.2f}%)"
        )
    return s
