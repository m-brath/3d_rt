# Plan — full cross-solver test matrix (all SC and MC solvers, one gate + one report)

Goal: a single, *sophisticated* validation of **every solver variant in this repo** —
the deterministic short-characteristics family (`rt3d.solver.lambda_iteration` and its
transport / grid / iteration / δ-M / backend options) and the Monte Carlo family
(`rt3d.mc`: stage-1 `j_mc`, stage-2 `radiance_from_source`, adjoint `backward_*`) — as a
pass/fail pytest gate plus a CSV/figure report script.

Decisions taken with the user (2026-11-16): pytest gate **and** report script; both solver
families cross-checked; runtime is not a constraint (thorough); correctness is anchored on
a **fine-octree reference + refinement ladder** (extends the idea of
`plan/transport-ranking-plan.md`); today's measured imperfection bands are **pinned as
regression thresholds**; the **MC Lambertian-ground gap is fixed as part of this work**
(new phase M6 below).

---

## 1. What "all solvers" means here (the variant space being tested)

Deterministic family — `rt3d.solver.lambda_iteration` (rt3d/solver.py):

| knob | values | implementation |
|---|---|---|
| grid | unigrid, octree | `rt3d/geometry.py`, `rt3d/octree.py` |
| transport | `chain` (lattice-stencil), `sc` (single-cell SC), octree exact marches | `rt3d/characteristics.py`, `rt3d/short_characteristics.py`, `rt3d/transport.py:precompute_octree_march` |
| backend | fused numba (`jit=None/True`), NumPy reference (`jit=False`) | `rt3d/jit.py` |
| threading | `n_jobs` 1 / N | `concurrent.futures` over ordinates |
| iteration | plain lambda, `ng1d` Ng acceleration | `rt3d/accelerator.py` |
| phase function | `delta_M` 0 / 1 / 2, scalar or per-cell `g` | `rt3d/source.py:delta_extinction, scattering_source` |
| physics | `B` thermal, solar beam, Lambertian+emissive ground, periodic x/y | `rt3d/source.py`, `rt3d/transport.py`, `rt3d/geometry.py:ground_surface` |

Monte Carlo family — `rt3d/mc/`:

| solver | variants |
|---|---|
| A1 `j_mc` | `stage1="lambda"` / `"explicit"`, `mode="cell"` / `"global"` (+`kappa_max`), `birth="source"`/`"uniform"`, `crn`, `ng`, M4 VR (`implicit_capture`, `roulette_floor`), `jit` on/off |
| A2 `radiance_from_source` | `estimator="next_flight"` / `"deterministic"`, `mode`, `jit`, tally binning on the icosahedral grid |
| B `backward_detector_mc` / `backward_all_sky` | cone modes (`cone_scale` 0 / small / 1), beam coupling, VR, `mode`, `jit` |

Capability matrix as measured today (the test must encode it, not rediscover it):

| capability | SC chain | SC sc | SC octree | MC A1 | MC A2 | MC B |
|---|---|---|---|---|---|---|
| thermal `B` | ok | ok | ok | ok | ok | ok |
| solar beam | ok | ok | ok | ok (particle stream) | ok (fixed source) | ok (exact-HG score) |
| Lambertian ground | ok | ok | ok | **raises** | **raises** | **raises** |
| variable ground | raises | raises | raises | raises | raises | raises |
| angular quadrature | 42/162/642 ordinates | same | same | **none** (stage 1) | tally binning only | **exact HG** |
| energy closure helper | `diagnostics.energy_balance` | same | **unigrid-only** | `mc.estimators.heating_rate` | same | — |

---

## 2. Measured findings that shape the test (probe, 2026-11-16, Apple M4 Pro, `pyarts`)

From throwaway probes of every entry point on the MANUAL/compare case (gaussian cloud,
ω capped 0.9, g peak ≈ 0.95, 42 ordinates, top inflow, 30° beam). These are the numbers
the regression bands below are pinned to.

**Wall times (42 ordinates, tol 1e-6):**

| run | wall | iterations |
|---|---|---|
| u24³ chain, δ-M 1, Ng | 0.44 s | 12 |
| u24³ chain, δ-M 0, Ng | 1.86 s | 57 (see F6) |
| u24³ sc, δ-M 1, Ng | 0.58 s | 12 |
| u24³ chain, δ-M 1, Ng, `n_jobs=4` | 0.43 s | 12 |
| u24³ chain, δ-M 0, plain LI | 1.84 s | 34 |
| u48³ chain, δ-M 1, Ng | 3.82 s | 13 |
| octree d3 (25 264 leaves), δ-M 1, Ng, fused numba | 4.78 s | 12 |
| octree d3, δ-M 1, Ng, `jit=False` (NumPy reference) | 20.44 s | 12 |
| octree d3, δ-M 0, Ng, fused numba | 11.83 s | 30 |
| octree d4 (34 476 leaves), δ-M 1, Ng, fused numba | 7.65 s | 12 |
| MC A1 explicit (20k+20k+20k histories, octree d3, jit) | 0.57 s | — |
| MC A1 explicit, `jit=False` | 0.99 s | — |
| MC A1 lambda (10k/iter × 25 iters, octree d3, jit / numpy) | 4.69 / 4.68 s | stalls, see F5 |
| MC A2 `next_flight`, 20 flights/ordinate, octree d2 (6 952 leaves) | 8.2 s | — |
| MC A2 `next_flight`, 50 flights/ordinate, same grid | 16.9 s | — |
| MC A2 `next_flight`, 25 flights/ordinate, octree d3 (25 264 leaves) | 50.3 s | — |
| MC A2 `deterministic`, octree d2 / d3 | 2.3 s / 11.7 s | — |
| MC B `backward_all_sky`, 20k seeds/ordinate, octree d2 | 1.01 s | — |
| MC B `backward_all_sky`, 200k seeds/ordinate, octree d2 | 8.93 s | — |

**Finding F1 — MC has no ground BC.** `j_mc`, `radiance_from_source` and
`backward_detector_mc` each raise `NotImplementedError` when `bc.ground is not None`
(rt3d/mc/forward.py:373, :834; rt3d/mc/backward.py:197). Today the *all-sky with ground*
case — the most physically interesting one, and the one in MANUAL §11 /
`scripts/compare_transport.py` — is therefore **SC-only**. Fixed by phase M6 (§7).

**Finding F2 — a zero-noise cross-family identity exists, and it is δ-M-sensitive.**
`radiance_from_source(estimator="deterministic")` fed the SC's own converged direction-
resolved `S` reproduces the SC `I` field to **1.4e-9** (relative to max |I|) on the octree.
Feeding the *physical* `kappa_e` instead of `result["kappa_e_delta"]` on a δ-M run breaks
it to **9.5e-2**. That is the sharpest tripwire in the whole matrix: deterministic, costs
seconds, and it fails on any drift in BC inflow conventions, beam τ bookkeeping, octree
march ordering, or the δ-M extinction split. (The exact identity was already noted in
`plot/mc_compare.txt`: 9.6e-11.)

**Finding F3 — per-cell MC z-scores are heavy-tailed, so a naive 2σ test is meaningless.**
Next-flight stage-2 tallies on the octree (20 flights/ordinate) give, over all cells with
`sigma_I > 0` (75 % of cells): rms z ≈ **44**, |z| > 3 in **3.7 %** of cells (Gaussian
expectation 0.27 %), max |z| ≈ **1.8e4**, and the numbers get *worse*, not better, with
more flights (50 flights → rms z ≈ 63). Cause: in optically thin cells the batch standard
error grossly underestimates the tally uncertainty (0-or-1 collision statistics), so
σ → small while the deviation stays. `scripts/mc_compare.py` sidesteps this by comparing
only "sigma-resolved cells" (117 759 of 360 k), where rms z = 1.05. Any serious test needs
an explicit **noise-validity mask** (§6.2) and must *report* the tail fraction rather than
assume Gaussianity.

**Finding F4 — anchor 4 (div F = heating) is unigrid-only.**
`rt3d/diagnostics.py:flux_divergence` and `energy_balance` index `grid.nx/ny/nz` and raise
`AttributeError` on an octree. So the energy-conservation anchor currently cannot be
evaluated on the octree grids where the accuracy ranking lives. Fixed by §8.

**Finding F5 — the MC λ-variant residual floor.** With fresh sampling per iteration
(`crn=False`, the default) the λ-variant residual stalls at the tally noise floor: after
25 iterations on the cloud octree `history[-1] = 1.9e-1`, i.e. it does **not** reach
`tol=1e-8`; with `crn=True` on an inhomogeneous cloud the frozen noisy operator can
diverge (documented in the `j_mc` docstring). Consequence for the matrix: the λ-variant is
validated on *agreement of its J field*, never on "it converged".

**Finding F6 — Ng acceleration is not an iteration win on the cloud case (F6, T1 pass).**
The plan's §3 expectation ("Ng vs plain LI: iteration-count ratio ≥ 2") could not be
reproduced.  Measured at tol = 1e-6, δ-M 0 (the regime where Ng is supposed to matter):

| grid | plain LI | Ng |
|---|---|---|
| u24³ | 34 sweeps / 1.84 s | 57 sweeps / 1.86 s |
| octree d2 (6 952 leaves) | 30 sweeps | 27 sweeps |

At δ-M 1 both take 12 sweeps on u24³ -- Ng changes nothing.  Raising the albedo cap to
0.99/0.999 leaves the picture unchanged (LI 38, Ng 46 sweeps), so this is not the ω cap.
The fixed point is identical (ΔJ/J = 3.8e-7), which is what the T1 test asserts; the
iteration-count assertion is downgraded to a *no-pathological-slowdown* band (Ng/LI ∈
[0.9, 2.0] on u24³, and Ng < LI on the octree).  The likely cause is that the plane-
averaged (z) Λ̄ of `rt3d.accelerator.ng1d` is a poor model of this localized 3-D Gaussian
blob, so the extrapolated source overshoots.  **Not a test-suite defect and not fixed
here** -- the acceleration is still exercised for fixed-point agreement, and a better
Λ̄ (or 3-D) remains a solver-side improvement.

**Finding F7 — anchor 4 is octree-capable now, and its residual is the face
extrapolation, not the physics (T4 pass).**
`rt3d/diagnostics.py` grew `cell_volumes`, `face_powers` and a per-leaf
`flux_divergence`, and `energy_balance` dispatches on grid type (unigrid path
bit-unchanged).  Octree faces are closed leaf-by-leaf with a two-point linear
extrapolation from the leaf and its located inward neighbour (exact for a flux
that is linear in z — verified to round-off), and `cell_volumes` gives
`(2·half)³`, so leaf areas sum to exactly `Lx·Ly` and leaf volumes to the box
volume.

Measured closure residuals (`relative` = residual / max(face power, |heating|)):

| case | u24³ | u48³ | oct d2 (h0 1.25) | oct d3 (h0 1.25) | oct d2 (h0 0.625) |
|---|---|---|---|---|---|
| conservative (κ_a = 0, B = 0, no beam) | +3.26e-5 | +6.55e-7 | −1.40e-3 | −1.36e-3 | −1.34e-4 |
| absorbing matrix case (ω≤0.9, no beam) | +4.23e-2 (chain) / −2.23e-1 (sc) | +4.60e-2 | −1.04e-2 | −1.28e-2 | −6.84e-3 |

Three conclusions that shape the T4 assertions:

1. **The face extrapolation is the error, and it is ~1.6-order, not 2.** On an
   analytically exact pure-absorption field (no solver involved) `energy_balance`
   leaves +10.2 % (12³), +3.97 % (24³), +1.32 % (48³), +0.39 % (96³); replacing the
   extrapolated face flux by the analytic one closes the same field to 3e-10.
   The intensity has a kink at the inflow faces (face value vs first interior cell
   differ by O(κ h/|μ|), and the 42-ordinate set has |μ| ≈ 0.1), so the two-point
   extrapolation degrades there.
2. **On the octree the closure tracks `h0`, not `max_depth`** (1.40e-3 → 1.36e-3
   from d2→d3, but 1.40e-3 → 1.34e-4 when `h0` is halved): refinement adds leaves
   inside the cloud while the leaves on the faces stay at base size.  The
   convergence fit must therefore be made in `h0` (the T4 test asserts exactly
   that, and that depth alone does *not* help).
3. **Absorbing residuals are dominated by the solution's angular/stencil error**
   (they do not shrink u24 → u48, and chain/sc differ by a factor 5), so they are
   pinned as bands and used as a discriminating diagnostic only.

Local anchor (`div F` vs `4πκ_a(B−J)`, conservative field where the exact answer
is 0): mean |div F|/(|F_z|/Lz) = 0.052 max 0.42 (u24) vs 0.115 max 1.03 (oct d2).
Cross-family absorbed power on oct d2: SC-42 144.94, SC-162 141.09, MC A1 (2·10⁵
boundary histories) 142.07 ± 0.26 — i.e. the 2 % SC-vs-MC gap is the **42-ordinate
quadrature error**, while the MC statistical error is 0.18 %; 162 ordinates close it
to 0.7 %.

**Finding F8 — two same-named `heating_rate` helpers with swapped arguments.**
`rt3d.diagnostics.heating_rate(kappa_a, J, B)` and
`rt3d.mc.estimators.heating_rate(kappa_a, B, J)` both return
`4π κ_a (B − J)`, but their argument order is *reversed*.  Nothing is wrong today
(each call site matches its own import), but swapping them flips the sign of every
heating/closure number — this cost a sign bug in the first draft of the T4 helper.
Follow-up (not done here, it touches existing call sites): make
`rt3d.mc.estimators.heating_rate` an alias of the diagnostics one, or rename it
`heating_rate_from_B_J`.  The matrix tests now carry a comment at the helper and
assert the sign (`absorbed power < 0` for `B = 0`).

**Finding F9 — MC error bars are only meaningful per point, and pooled means
need a correlation allowance (T3 pass).**
Measured on the matrix case, octree d2 (6 952 leaves), A1 explicit variant,
2·10⁵ boundary histories, 100 batches:

* per-cell relative sigma of `J_total` = **7.6 %**; two independent seeds then
  differ by **8.1 % rms**, and the pooled z distribution has **rms 0.83–0.87**
  (so the 100-batch sigma is ~15 % *conservative*, not optimistic), |z| > 2 in
  2–4 %, |z| > 3 in 0.1 %;
* **the mean of per-cell z-scores is not an independent-sample mean.**  One
  history contributes to many cells, so the measured spread of `mean z` over
  independent seed pairs is 0.023 while `std/√n` claims 0.010 — a factor 2.2.
  A "mean z within 3 standard errors of 0" test would reject correct code; T3
  therefore pins `|mean z|` as a band (≤ 0.10) instead;
* **the same seed does not decorrelate the numba and NumPy paths.**  Differenced
  at one seed their pooled z rms is **0.44**, versus **0.85** for different
  seeds — the streams overlap partly despite M4/M5.  Path parity must be tested
  at *different* seeds (T3 pins both numbers);
* the F3 mask is needed by the **adjoint** family as well: at a top-layer leaf,
  2 of 21 upward ordinates have τ_ray = 0 and hence σ = 0, and the unmasked
  max |z| there is **2·10²⁵**; masked over 6 points × 21 ordinates (114 points)
  the adjoint all-sky tally has mean z −0.12 ± 0.09, rms 0.97, max |z| 3.0;
* the stage-2 next-flight tally, masked at 100 flights/ordinate, has mean z
  **+0.02** (a small positive offset consistent with the Jensen bias of 1/σ̂ at
  25 batches), rms 1.09, |z| > 3 in 1.1 %, while unmasked it is rms 17–121 and
  |z| > 3 in ~2.4 % — the mask is doing real work.  Its **aggregate** deviation
  against the zero-noise deterministic line integral matches the aggregate
  sigma norm to **0.998**, and halves exactly when the flight count is
  quadrupled (0.500).

**Finding F10 — the octree's three resolution knobs are not interchangeable, and
the two grid families converge to different limits (T5 probes).**
All numbers below are the matrix case, 42 ordinates, Ng, δ-M 1, every rung
converged (`n_iter=400`, `tol=1e-8`, 17–32 iterations, residual ~1e-9), rms
relative deviation of `J` measured after piecewise-constant sampling of the
yardstick onto the coarser rung's centers (`matrix_stats.field_errors`):

*unigrid family (self-Cauchy, h = spacing):* u12→u24 2.79e-2, u24→u48 1.36e-2,
u48→u96 6.72e-3 → fitted slope **1.00**, clean first order.

*octree family at h0 = 1.25:*
| ladder | rungs | rms-J error vs `oct(τ=0.5, depth 5)` |
|---|---|---|
| optical-depth criterion `tau_max` (depth 5) | 8 / 4 / 2 / 1 | 1.33e-2 / 8.61e-3 / 5.74e-3 / 3.94e-3 |
| `max_depth` (τ = 0.5) | 1 / 2 / 3 / 4 | 2.21e-2 / 1.57e-2 / 2.39e-4 / **0 (identical grid)** |
| base cell `h0` (τ = 0.5, depth 5) | 5.0 | **3.32e-4** |

Three consequences for the ladder design:

1. **Depth refinement saturates**: at fixed `h0` and `tau_max` the τ criterion
   is already binding at depth 3, so depth 3 ≡ depth 5.  "Refine the octree
   deeper" is not a convergence path; the ladder must be taken in `tau_max`.
2. **The base cell size barely matters at fixed depth** (h0 5 → 1.25 changes J
   by 3e-4), which is consistent with finding F7 — base-cell sensitivity shows
   up in *face-evaluated* quantities (the closure), not in the cell-centered
   field.
3. **The cross-family gap does not shrink**: unigrid-vs-octree-reference rms-J
   is 3.38e-2 (u12), 4.10e-2 (u24), 4.03e-2 (u48), 4.20e-2 (u96) — a plateau,
   while the unigrid family converges internally and the octree family's whole
   internal spread across all three knobs is ≤ 1.3e-2.  The gap lives in the
   **vacuum** (mean |ΔJ| 1.95e-2 over vacuum cells, 6.4e-3 in the cloud core),
   and the input fields are identical (∫κ_e dV agrees to 0.1 %, τ_z through the
   cloud center 9.9997 on every grid).  So the plateau is **not** an
   input-representation or halo effect: the two transport families sit at
   different solutions — explained by finding F11 below.

Method note that this probe also produced: a plain `np.mean(J)` over an octree's
leaves is **not** comparable to a unigrid's mean (leaves of 8 different sizes are
weighted equally — measured 0.450 vs 0.601 for the same physical case).  All
cross-grid observables in T5 use volume-weighted means.

**Finding F11 — the unigrid `chain` transport carries an h-independent error;
the octree's exact-characteristic transport does not (T5 probes).**
Zero-noise test: solve the matrix case on a unigrid with `transport='chain'`, then
evaluate the **exact line integral of that converged source** along the same rays
(`radiance_from_source(..., estimator='deterministic')`, which is identical to the
octree sweep to 1.2e-12 — finding F2 / T2).  The deviation is the transport
scheme's own error:

| grid | chain vs exact march (rms J) | chain vs exact march (max J) | volume-mean J chain / exact |
|---|---|---|---|
| u12³ | 4.05e-2 | 9.4e-2 | 0.5979 / 0.5795 |
| u24³ | 3.98e-2 | 1.02e-1 | 0.6006 / 0.5827 |

It **does not shrink when the grid is refined**, which is exactly the plateau of
F10 item 3 — and it explains the cross-family numbers: the octree reference's
volume-mean J (0.5832) agrees with the unigrid *exact-march* value (0.5827 at u24)
to 0.1 %, while the unigrid `chain` solution sits at 0.6006 (+3.1 %).  So the
unigrid family converges cleanly (F10, slope 1.0) to a solution that is ~4 % rms
(≈3 % in the mean, 5.7 % in absorbed power) away from the transport-exact one,
while the octree family is self-consistent with the exact march by construction.

Consequences for the test matrix:
* the octree reference is the right yardstick, and the unigrid rungs must be
  pinned as *bands* (their error is not removable by refinement), not fitted to
  zero;
* the "ladder" of the unigrid family measures its **self-consistency** (Cauchy
  convergence), and a separate assertion pins its distance from the exact march;
* the plan's ranking statement (octree ≥ chain ≥ sc) is now measured to be
  stronger than assumed: on this case the octree's characteristic march is exact,
  so the unigrid transports are the only ones carrying this term.

Root cause (**documented behaviour, not a regression**): neither unigrid short-
characteristics stencil follows the true characteristic — the `chain` sweep's path
is the cell-to-cell chain and the `sc` sweep's effective direction is set by its
bilinear weights — so both carry a **bounded, resolution-independent grazing-
ordinate error** (`docs/sc-transport.md` §12.2 and §12.5 say exactly this, and
`tests/test_short_characteristics.py::test_transmittance_staircase_error` pins it
for `transport='sc'`).  The decisive zero-physics measurement, pure absorber
(`kappa_s = 0`, `B = 0`, no beam) with uniform `kappa_e` on u24, 42 ordinates,
against the analytic `I_in exp(-kappa s)`:

| scheme | rms dev vs analytic | max dev |
|---|---|---|
| unigrid `chain` (kappa 0.25 / 1.0) | 1.16e-1 / 1.40e-1 | 2.8e-1 / 3.2e-1 |
| unigrid march + exact segment integral (`radiance_from_source`, deterministic) | 2.3e-15 / 6.3e-15 | — |
| octree `chain` | 2.5e-15 / 2.8e-15 | 6.5e-16 / 1.0e-15 |

Structure of the unigrid error (u24, matrix case): worst per-ordinate deviations
at the grazing ordinates of the icosahedral set — 6.0e-1, 4.9e-1, 4.6e-1 at
|mu| = 0.112 / 0.105 / 0.114 — best 8e-3, corr(dev, |mu|) = +0.29; switching the
periodic x/y wrap off makes it *larger* (9.5e-2 vs 4.0e-2), so it is not a wrap
bug; on a 1D slab it is only 1.0e-2 (and 1.7e-3 with a single transverse column),
which is why every 1D-based validation in the suite passes.

Consequences for AGENTS.md validation anchor 3 (`kappa_s = 0`, `B = 0` →
transmittance = `exp(-tau)` along *every* ray): it holds for the octree sweep and
for the march-based estimators to round-off, and on the unigrid only for z-entry
ordinates.  The matrix must therefore state the anchor **per family** and pin the
unigrid violation as a band, and any statement that "the unigrid ladder converges
to the same solution as the octree" is false at the 4 % level.

Documentation defect found alongside it: `plan/plan-sc-unigrid.md` §3 item 3 said
the SC staircase deviation "is O(h) and shrinks under refinement", which both
`docs/sc-transport.md` §12.2 and the measurement above contradict (it does not
shrink with h).  **Corrected 2026-11-16.**

**Scope decision (user, 2026-11-16):** the unigrid transport error is **pinned
as a finding — the solver is left unchanged**.  No M7 fix phase, no new
`transport='march'` option in this pass; the matrix tests all treat the
unigrid numbers as bands with the F11 cause stated.

**Finding F12 — the MC segment marcher read uninitialised memory for
axis-parallel directions; MC results with an exactly axis-parallel direction
were run-to-run nondeterministic (found while pinning the M6 all-sky case).**
The numpy segment march in `rt3d/mc/tracking.py` computed face-crossing times
per axis into a buffer `tf = np.empty(act.size)` and filled only the `da > 0`
and `da < 0` entries; entries with `da == 0` stayed uninitialised heap garbage
and entered `t_face = np.minimum(t_face, tf)`.  Direction components are
*exactly* zero for the matrix beam (0.5, 0, −0.866) and for the z-poles, so
every beam history (and pole-containing ray) took garbage steps: repeated
identical `j_mc` calls at the same seed varied the absorbed power of the
all-sky ground case by ±5 %, while isotropic-direction cases (A1) were
unaffected — which is why every pre-M6 determinism test passed.  Fixed to
`np.full(act.size, np.inf)` (2026-11-16); the fix is in the M6 scope because
the ground/beam populations are what exposed it.  All pre-fix all-sky probe
numbers (the apparent "MC +5 % above SC" and "aggregate noise 9x quadrature")
were artefacts of this bug; the post-fix measurement is in the T3 module
header: MC power −2.15 % vs SC, seed-split/quadrature ratio 1.8.
**Regression guard:** `test_mc_ground.py` pairs (explicit vs lambda, jit vs
numpy) at equal seeds and requires agreement at 1e-12 — with this bug the
paired runs differed at the percent level (the ground-seeded and vertical
beam cases both contain exactly axis-parallel rays).

**Finding F13 — the stage-2/adjoint ground-column lookup read unwrapped
periodic exit points (fixed 2026-11-16, found by the report run).**
`radiance_from_source(ground_I=...)` and `backward_all_sky` identify the
ground column of a bottom exit from the exit point of the *backward* march.
With periodic x/y the march budget extends into unwrapped frames (exit
points up to ~3 box widths out), and `ground_column_lookup` (octree
`locate`/unigrid index) silently **clamps** to the boundary column: every
wrapped ray scored the wrong column's ground radiance (measured on oct2:
**56 %** of all up-ordinate ground-column lookups return a different
column with and without the fold).  Consequences, all
measured on the oct2 all-sky case: the F2 march-vs-solver identity, exact
to 1e-12 without ground, breaks at **1.2e-1 max** on the ground case
(only with a *per-column varying* I_g — a constant ground radiance hides
it, which is why every M6 unit test passed); the adjoint all-sky at
near-vacuum leaves carries a coherent +5-7 % bias (masked mean z +6.5)
against the solver columns.  Fixed by folding exit points through
`periodic_box.wrap_xy` before the lookup (forward march octree +
periodic-unigrid branches, adjoint ground-hit point).  Post-fix: march-vs-
solver identity 5.6e-17 (pure absorber+beam) / 8.6e-10 (full case, tol-
limited); adjoint mean z back to ~0.  Regression guard:
`test_matrix_crossfamily.py::test_ground_stage2_march_identity`.  Note the
forward MC (`track_histories`) is unaffected — its analog ground reflection
needs no column lookup, and its wrap bookkeeping already went through
`wrap_xy`.

**Finding F14 — the beam first-scatter source: SC's quadrature-normalized
HG vs MC's analytic HG differ by 4-8x at g → 0.9 on the 42-ordinate grid
(documented design difference, not a bug; report run, 2026-11-16).**
`rt3d.source.beam_scattering_source` rescales the per-cell beam phase
function so `sum_i w_i Phi_i == 1` on the direction grid (deliberate: the
beam first-scatter power then passes through the ordinate set exactly —
same convention as `hg_matrix(normalize=True)`, required for the exact
delta-M split).  The MC solvers instead sample the *analytic* HG.  On the
42-ordinate icosahedral grid the raw quadrature defect `s(g) =
sum_i w_i Phi_analytic` is ~1.000 to g = 0.5, but 1.06-1.11 at g = 0.7 and
**4.0-8.9 at g = 0.9-0.925** — i.e. SC's beam source sits a factor
1/s ≈ 0.11-0.25 below the MC one inside the dense cloud core.  Observed
as: adjoint all-sky (analytic HG) vs SC columns **with beam** shows
coherent per-ordinate deviations of 6-15 % and masked rms z up to ~30 at
near-vacuum probe points — orders of magnitude above noise, and the same
mechanism contributes to the -2.15 % MC-vs-SC power offset pinned in T3 §4
(the forward MC carries the analytic-HG beam source into a field the SC
solve normalized).  It is an *angular-discretization* statement of exactly
the T5 kind: tighter direction grids drive s(g) → 1 and the two families
together; on 42 ordinates every beam-case cross-family per-ordinate
comparison is F14-limited, not noise-limited.

---

## 3. Test architecture: six tiers

Each tier is a separate pytest module so that selection by marker/file is meaningful.

### T0 — invariant & anchor sweep across every variant (fast, no full solve)
File: `tests/test_matrix_anchors.py`

AGENTS.md anchors 1–3 re-run **per variant**, in one parametrized sweep, so a new backend
cannot quietly drop an anchor:
1. `sum(w) == 4π` for icosahedral refinement 0–3 (already in `test_directions.py`; here
   also asserted through the MC sampling path: `isotropic_directions`,
   `face_inflow_directions` measure identity, HG inverse-CDF Legendre moments vs
   `hg_matrix` quadrature).
2. HG normalization `∫Φ dΩ = 1` analytic + quadrature + MC sampler, for g = 0 / ±0.3 /
   0.9 / 0.99.
3. Beer-Lambert `κ_s = 0, B = 0`: transmittance = `exp(-τ)` on **identical rays** for
   chain, sc, octree-exact (jit and NumPy), MC `mode="cell"` and `mode="global"`, the
   adjoint walk, and the beam τ path `beam_optical_depth(_octree)` — all against one
   analytic ray-integral reference, plus mutual agreement between variants.

### T1 — deterministic variant equivalence (tight tolerances, no statistics)
File: `tests/test_matrix_equivalence.py`

Same case, same grid, one solve per variant, then differenced pairwise:
- `jit=None` vs `jit=False`, and `n_jobs=1` vs `n_jobs=4`: **bit-identical** `I`
  (sweeps already are per `test_jit.py`; asserted at the *solver* level, including the
  ground/beam/periodic path and per-cell `g` bucketing).
- δ-M 1 and 2 vs δ-M 0: converged fields agree within the known O(τ_cell) explicit-term
  gap. **Measured in the T1 pass (tol 1e-6, u24³, common bc):** max|ΔJ|/max J =
  8.4e-3 (M1) / 8.5e-3 (M2), max|ΔI|/max I = 4.1e-2 / 3.1e-2; on octree d2 1.3e-2 /
  2.6e-2. The pinned bands are 1.5e-2 (J) and 6e-2 (I) on u24³, 2.5e-2 / 5e-2 on the
  octree. δ-M *does* cut the iteration count on this cloud: 12 (M1) and 16 (M2) vs 34
  sweeps (M0).
- Ng vs plain LI: same fixed point within tol (ΔJ/J = 3.8e-7), but see **F6** — Ng is
  *not* faster on this case, so the test pins "no pathological slowdown" instead of an
  iteration ratio.
- `chain` vs `sc` on identical cells: the field-wide difference stays in the known band
  (**measured 0.26 J / 0.93 I** on u24³ with face inflows, 0.31 / 0.94 with the
  Lambertian ground) **and shrinks with the cell size** (ladder, T5). Note that the
  *peak* values agree much better than the field-wide measure: ground-case Jmax 0.6087
  (chain) vs 0.5924 (sc) = 2.7 %, while oblique ordinates in the optically thin parts
  of the field carry the large deviations.
- ground on/off, beam on/off, periodic vs non-periodic sides: structure checks + the
  exact phase-5 ground anchors re-run for `chain`, `sc` and octree (already in
  `test_ground.py` for chain/octree; `sc` gets its own).
- guards: `transport="sc"` + octree → `ValueError`; variable `ground.height` →
  `NotImplementedError`; `kappa_e = 0` vacuum cells → `omega = 0`, no NaN/Inf.

### T2 — zero-noise cross-family identity (SC ≡ MC deterministic sweep)
File: `tests/test_matrix_crossfamily.py`

Feed the MC stage-2 deterministic line integral the SC's own converged source and require
it to reproduce SC `I`. Parametrized over: octree d2/d3 × δ-M 0/1/2 (fed as
`result["kappa_e_delta"]`) × beam on/off × `mode="cell"`/`"global"` × `jit` on/off.
Assert **max |Δ| / max|I| ≤ 1e-8** (probe 1.4e-9; `mc_compare.txt` 9.6e-11).
Negative control: feeding physical `kappa_e` under δ-M ≥ 1 must *fail* the identity by a
≥1e-2 margin (proves the test can fail). Also asserts `radiance_from_source` with
`bc.ground` raises until M6 lands, and afterwards reproduces the SC ground inflow.
On the unigrid the same estimator is compared with a **band**, not exactly: the `chain`
sweep is a lattice stencil, not a line integral (M2 decision); `transport="sc"` gives a
tighter comparison than `chain`.

### T3 — statistical MC ↔ SC and MC ↔ MC consistency (noise model of §6.2)
File: `tests/test_matrix_statistics.py`

Common (ground-free) case until M6, all-sky case afterwards:
- A1 explicit vs A1 lambda: `J` agrees within combined σ (probe σ/J ≈ 0.36 at
  20k+20k+20k histories → the matrix uses ≥ 2·10⁵ histories per population to get a ~4 %
  band; `crn` off for inhomogeneous cases per F5).
- A1 vs SC `J`: agreement band pinned from the refinement ladder, with the *statistical*
  part removed (compare A1 ± σ against the SC-vs-reference error of T5, so MC noise and SC
  discretization error are budgeted separately, not conflated).
- A2 `next_flight` vs SC `I`: masked z statistics (§6.2) — pinned band: masked mean z ≈ 0
  within ±3/√N, masked rms z ∈ [0.7, 1.6], tail fraction |z| > 3 ≤ 1.5 %; the unmasked
  tail fraction is **reported** and asserted only to stay inside the measured band
  (3.7 % at 20 flights/ordinate) so a *worsening* tail fails loudly.
- A2 chain (isotropic `ω J` source) vs A2 fed by SC's direction-resolved source: the
  difference is the documented O(g) source-model term, asserted as a *deterministic* band,
  never folded into noise.
- B (exact-HG adjoint) vs SC at pixels: pinned from probe — at 20k seeds/ordinate,
  rms z = 1.19, all 21 upward ordinates within 3σ, σ/I ≤ 3.3 %; at 200k seeds rms z = 1.03.
  Band: rms z ≤ 1.6, max |z| ≤ 3.5, and B-vs-SC pixel deviation ≤ the SC angular-
  quadrature error measured against SC-162/642 (mc_compare: −2.8 % vs SC-642).
- MC ↔ MC: `mode="cell"` vs `"global"` (null collisions), `jit` vs NumPy, VR on vs off,
  seed reproducibility (same seed bit-identical, different seeds within combined σ), and a
  χ² seed-independence check over ≥ 8 seeds.

### T4 — global physics closures per family (anchor 4)
File: `tests/test_matrix_closure.py`

For each family on the same case: net face power vs `∫4πκ_a(B−J)dV`, using the new
octree-capable closure of §8. Acceptance bands pinned from measurement (today: SC unigrid
−7.0 % with 2-point face extrapolation on the slab, −0.6 % on the cloud octree by the
ad-hoc cell-center estimate in `mc_compare.txt`) — the new octree closure replaces that
ad-hoc number with a proper face-area closure and the band is re-pinned once measured.
Cross-family: SC vs A1 vs A2 absorbed power agree within a few %.

### T5 — refinement ladder, reference ranking, convergence order
File: `tests/test_matrix_ladder.py` (marked `slow`, `reference`) + the report script (§9)

**Redesigned after the ladder probes** (findings F10/F11; the original spec is
kept at the end of this subsection).  As built:

- *Yardstick / reference rungs*: octree `(τ_max 0.25, depth 7)` as yardstick (τ
  criterion needs level 6 → depth 7 never binding), `(τ_max 0.125, depth 7)` as
  sufficiency rung, `(τ_max 0.5, depth 5)` as the gate reference (self-error
  5.5e-3 rms J against the yardstick).
- *Octree ladder* in **`tau_max`** (8/4/2/1/0.5 at depth 7), fitted slope in
  `tau_max` ∈ [0.3, 1.0] (measured ~0.6); **depth saturation** asserted instead
  of a depth-ladder fit (d3 ≡ d5 to 2.4e-4, d4 grid identical to d5); `h0`
  **insensitivity** asserted (≤ 1e-2 rms J for h0 = 5/2.5 at fixed τ/depth)
  instead of an h0 convergence fit.
- *Unigrid ladder* `u12 → u48` against **u96 of the same family** (slope
  ∈ [0.6, 1.6] in h, measured 1.0) — never differenced against the octree for
  the fit, because the cross-family gap does not vanish (F11).
- *Anchor 3 stated per family*: pure absorber vs analytic `exp(-kappa s)` —
  octree sweep and deterministic march exact to 1e-12, unigrid `chain` pinned
  in a violation band (rms 0.05–0.25, max 0.10–0.55).
- *F11 pinned*: unigrid chain-vs-exact-march rms-J band [0.03, 0.05] at
  u12/24/48 with an explicit **non-shrinking** assertion; u96-vs-reference gap
  band [0.03, 0.06]; octree reference asserted to sit at the march limit
  (mean J agrees with the u24 exact march to ≤ 0.5 %).
- Ranking at equal cost: octree ≥ chain ≥ sc (hypothesis of
  `plan/transport-ranking-plan.md`, enforced by assertion; the margin is
  structural — the unigrid transports sit on the F11 plateau).
- δ-M reported on/off at the reference rung: field shift small (measured
  4.5e-4, *below* the refinement self-error), iteration effect large (90 vs 17
  on u24), plateau δ-M-independent (no cancellation).
- MC order measured by **seed splitting** (two seeds per N, differenced): fit
  slope ∈ [−0.65, −0.35] over 4 history counts; *not* error-vs-reference, which
  F11's 4 % bias would swamp at any affordable N.

Every rung solved with `n_iter=400, tol=1e-8` and asserts `converged` — an
early probe's ranking numbers were iteration error, not discretization error.

Original spec, kept for the record: ladder `u24 → u48 (→ u96 slow)`, `octree
d2 → d3 → d4 → d5 (τ_max 0.5, 50 380 leaves — the gate reference; the
642-ordinate reference stays in the report script)`, error vs reference
shrinking monotonically along every ladder, fitted slope ∈ [0.6, 1.6], the same
ranking assertion, and MC order `N^-0.5` measured against the reference.

---

## 4. What already exists and must not be duplicated

201 test functions already cover per-component behaviour (`tests/test_transport.py`,
`test_short_characteristics.py`, `test_octree.py`, `test_jit.py`, `test_ground.py`,
`test_delta_m.py`, `test_accelerator.py`, `test_solver.py`, `test_mc_*.py`). The new suite
deliberately adds only what is missing: **(a)** solver-level (not unit-level) equivalence
across the *full* option matrix, **(b)** the deterministic SC ≡ MC identity test
parametrized over δ-M/beam/grid, **(c)** a statistical methodology with an explicit
noise-validity mask, **(d)** the refinement-ladder ranking with convergence-order fits,
**(e)** the octree energy closure and the MC ground capability. Where an existing test
already pins an anchor, the matrix tests *reuse* the case fixtures instead of re-deriving
them, and no existing test file is rewritten.

## 5. New files

```
tests/matrix_cases.py      case builders + grid factories (cloud/slab, levels cached by
                           fixture) + the memoized converged-solve helper
tests/matrix_stats.py      z-stats, noise-validity mask, pooled chi2, slope fits, band asserts
tests/test_matrix_anchors.py        T0
tests/test_matrix_equivalence.py    T1
tests/test_matrix_crossfamily.py    T2
tests/test_matrix_statistics.py     T3        (mark: mc)
tests/test_matrix_closure.py        T4
tests/test_matrix_ladder.py         T5        (mark: slow, reference)
pytest.ini                        markers, testpaths, --strict-markers
scripts/solver_matrix.py            report runner (# %% cells, caching, Agg, plot/)
plan/solver-test-matrix-plan.md     this plan
docs/solver-matrix.md               results write-up (regenerated from the report)
```

`pytest.ini` (none exists today; markers are currently unregistered):

```ini
[pytest]
testpaths = tests
addopts = -ra --strict-markers
markers =
    slow: refinement ladder / fine reference / high-history MC runs
    mc: Monte Carlo solvers (statistical, seeded)
    reference: needs the fine-octree reference run
```

`n_refine` is a config knob everywhere, recorded in every output; no code assumes 42
(AGENTS.md rule).

---

## 6. Statistical methodology

### 6.1 Seeding protocol
Fixed PCG64 base seeds per test (recorded in the report). Same seed → bit-identical
results asserted; every statistical assertion is made on ≥ 2 independent seeds and
requires **both** to pass, so a single lucky stream cannot mask a regression.

### 6.2 Noise-validity mask for MC-vs-deterministic differencing (fixes F3)
A cell enters the z-statistics only if its tally noise model is trustworthy:
- `sigma_c > sigma_floor` with `sigma_floor = eps_rel · max|I|` (`eps_rel` ≈ 1e-6), and
- **analytic minimum collision count**: from the march segments, the expected number of
  real collisions along the ray through the cell, `E[n_c] = Σ_seg (1−exp(−τ_seg))` (or the
  closed-form `τ_c` for the next-flight estimator), must satisfy `E[n_c] ≥ n_min` (`n_min`
  tuned once, then pinned; ~5–10);
- optionally: exclude cells where the *deterministic* line integral is exactly 0.
The masked and unmasked statistics are **both** reported; the mask is defined by the
ray/τ data, never by the observed deviation (no cherry-picking).

### 6.3 Pooled statistics instead of per-cell maxima
- masked mean z with its own standard error; masked rms z; tail fractions at |z| > 2, 3, 5
  compared with Gaussian expectations;
- pooled `chi2 = Σ z²/N` with a documented acceptance interval;
- aggregate norm comparison `‖MC − SC‖ / ‖SC‖` vs the combined `σ` norm — the headline
  number, since it is insensitive to the tail problem;
- for MC-vs-MC: combined σ `sqrt(σ_1² + σ_2²)` and the `mc_compare.py` convention of
  reporting z per quantity.

### 6.4 Convergence-order fits
Least-squares log-log slope over ladder points (grid h for SC, history count for MC) with
pinned acceptance intervals (§3 T5). Fits, not point comparisons, are what make the
ranking meaningful.

---

## 7. Phase M6 — Lambertian ground in the MC family (closes F1)

Motivation: without it the physically richest case (all-sky, emissive/reflecting ground —
the MANUAL §11 case) can never be cross-validated between solver families, and the matrix
would need two disjoint case setups forever. Scope mirrors SC phase 5
(`rt3d/transport.py:ground_inflow`, `bottom_mask`, `beam_irradiance_ground`):

1. **Shared inflow helper** — `rt3d/mc/ground.py` (new):
   `ground_radiance(grid, bc, F_down, beam_irr)` = `eps_g·B_g + A_g·(F_down + F_beam)/π`
   per bottom column, reusing `rt3d.transport.ground_inflow` and
   `beam_irradiance_ground` so the SC and MC ground boundary conditions are *the same
   code*, by construction.
2. **A1 explicit variant** — analog boundary scattering at `z = z0`: on a bottom-face
   crossing, sample a Lambertian (cosine-weighted, via `mc.sources.face_inflow_directions`)
   re-emission with weight `× A_g` (absorbed otherwise), plus a dedicated ground-emission
   population seeded on the bottom face with intensity `eps_g·B_g`
   (`w = π A ε B_g / n`) and a ground-reflected beam population with intensity
   `A_g F0 |μ_sun| exp(−τ_b,g)/π`. Self-consistent inside one history — no outer iteration.
3. **A1 lambda variant** — the ground inflow is rebuilt between iterations from the
   previous iteration's bottom-cell downward irradiance `F_down` (the exact twin of the SC
   sweep ordering: downward ordinates first, then the ground inflow). The bottom-face
   population then needs a *spatially varying* intensity → new weighted face seeding
   (2-D analogue of `mc.sources.sample_births`, categorical over bottom cells with weights
   `I_g,c · A_c`).
4. **A2 stage 2** — the fixed-source line integral needs a *given* ground radiance: new
   kwarg `ground_I=(n_bottom,)` (or accept `bc.ground` + an `F_down` field), applied at the
   bottom-face exits of the backward marches exactly where `bc.inflow` is applied today.
5. **Adjoint B** — adjoint of the boundary operator: on a ground hit, score
   `wt · [eps_g B_g + A_g F_beam(x_g)/π]`, then continue with `wt ×= A_g` and a
   cosine-weighted inward direction (the adjoint counterpart of the forward cosine-
   weighted seeding convention; the existing exit-score pattern `score += wt · I_in` stays
   the template).
6. Guards stay: variable `ground.height` → `NotImplementedError` everywhere.
7. Numba parity (`rt3d/mc/jitmc.py`) for the ground branch, NumPy fallback kept, both
   tested.

Verification of M6 (before the matrix re-uses the all-sky case):
- SC phase-5 anchors re-run for MC: pure-absorber ground Beer-Lambert with albedo A
  (`test_ground.py:63` analogue), analytic ground emission `I = eps B_g` in vacuum,
  reflected-beam irradiance `A F0 |μ_sun| exp(−τ_b)/π`;
- MC all-sky at the cloud-top pixel vs SC all-sky (`diagnostics.allsky`) within the same
  bands as T3;
- energy closure with ground: the bottom-face power includes the reflected + emitted part;
- the whole T1–T5 matrix then runs a **second** case family: the full all-sky + ground
  case, so the §1 capability matrix collapses to "everything supported except variable
  ground".

**Status (2026-11-16): implemented and verified.**  Delivered as designed:
`rt3d/mc/ground.py` (shared law via `ground_inflow`/`beam_irradiance_ground`),
fixed ground populations + analog reflection + per-iteration re-seeding in
`j_mc` (`n_ground_histories`), the `down_irradiance_from_segments` tally in
`track_histories`, stage-2 `ground_I=`, and the adjoint ground score/continuation
in `backward.py`; variable-height ground still raises everywhere.  Ground
reflection sits in the generation loop between collision-sampler calls, so the
jit and numpy sampling paths share it by construction (parity asserted).
Verification module: `tests/test_mc_ground.py` (13 tests: vacuum emission
J = B_g/2 and F_z = π B_g per the angle-averaged MC convention; Beer-Lambert +
albedo ground vs continuum analytic at A ∈ {0, 0.3, 0.9} within 2.5σ; adjoint
cones over the ground; reflected-beam population J = A F0 μ/(2π) with exact
J_direct = F0/4π and `n_ground_reflections = 0`; stage-2 `ground_I` vs the SC
solve — z-entry ordinate 7.5e-3, obliques in the F11 band).  The all-sky
cross-family comparison moved into T3 §4 (`test_matrix_statistics.py`): with
conventions matched (MC 'J_total' includes the uncollided beam, SC 'J' does
not — the F12 fix was prerequisite) MC power is −2.15 % vs SC, per-cell
rms z 0.80.  T2 gained positive ground smoke tests for all three entry points
(replacing the old rejection guard).  Remaining deviation from the §7 sketch:
the cloud-top-pixel `diagnostics.allsky` comparison is replaced by the T3 §4
observables + masked-z form; a full bottom-face power *closure* with ground
is left to the T4 extension round.

## 8. Octree-capable energy closure (closes F4)

`rt3d/diagnostics.py`: extend `flux_divergence` / `energy_balance` (or add
`energy_balance_octree`) to build the closure from leaf face areas and volumes
(`cell_volumes`, `half`), summing the net `F_z` over the leaves touching each face, with
the same second-order face extrapolation idea as the unigrid version; document the order
of the closure error (the ad-hoc cell-center estimate in `mc_compare.txt` gives −0.62 % on
the cloud vs −6.96 % on the slab: that difference is boundary extrapolation, not physics,
and the new helper must state its own error). Test: closure residual on the conservative
`omega = 1, B = 0` case must → 0 as the grid refines, on **both** grids.

## 9. Report script `scripts/solver_matrix.py`

`# %%` cell style like `compare_transport.py` / `mc_compare.py`, `matplotlib.use("Agg")`,
outputs under `plot/solver_matrix/`, per-run solve cache in
`plot/solver_matrix/_solve_cache/` (gitignored, keyed by variant id + `n_dir`, stale
entries re-solved):

- `matrix.csv` — one row per solver variant: `n_iter`, converged flag, wall time, field
  size, `J`/`F`/all-sky scalars, closure residual, guards hit;
- `ranking.csv` — every variant's max/mean error vs the reference for `J`, angular
  variance, TOA `F+` (same common-lattice sampling trick as `compare_transport.py`);
- `convergence.csv` — ladder errors + fitted slopes, MC `N^-0.5` fits;
- `mc_stats.csv` — masked/unmasked z stats, tail fractions, per-seed spread, σ/I;
- figures: `j_err_maps.png`, `j_convergence.png`, `err_bars.png`, `z_hist.png` (masked vs
  unmasked z distributions — the F3 story in one picture), `mc_convergence.png`,
  `closure.png`, `capability_matrix.png`;
- optional `scene_*.html` pyvista/vtk.js scenes (trame-pyvista is installed).

Runtime: first run ≈ 40–60 min (the 642-ordinate reference octree run needs ~22 GB per the
docs — keep it opt-in via `--reference-ordinates 3`; the default reference is the
42-ordinate depth-5 octree, ≈ 1.4 GB). Cached reruns: seconds.

## 10. Runtime budget for the gate (measured, extrapolated)

| tier | estimate |
|---|---|
| T0 + T1 + T2 | ≈ 8–12 min |
| T3 + T4 (MC with ~2·10⁵ histories, masked stats) | ≈ 15–25 min |
| T5 ladder (`slow`, incl. depth-5 reference at 42 ordinates) | ≈ 30–50 min |
| M6 ground re-runs of the all-sky matrix (`slow`) | ≈ +20–30 min |
| total default run | ≈ 25–40 min |
| total with `-m "slow or reference"` | ≈ 1.5–2 h |

## 11. Execution order (each step verified before the next)

1. `pytest.ini` + `tests/matrix_cases.py` + `tests/matrix_stats.py`; record a baseline run
   of the existing 201 tests (no baseline has ever been recorded — the earlier attempt was
   interrupted).  **Status: done** (helpers, `pytest.ini`, baseline run).
2. T0, T1 (deterministic only, no MC) — should be green immediately; any red is a real
   finding to investigate before proceeding.  **Status: done** — T0 43 tests / 2 s,
   T1 20 tests / 30 s; findings F6 (Ng) and the oblique-ordinate bands recorded above.
3. T2 cross-family identity + its negative control — the δ-M feed (F2) must be encoded
   here, not papered over.  **Status: done** — 12 tests / 105 s; identity measured at
   1.2e-12 (octree, same source) and 5e-9 (vs the solver's own I at tol 1e-8), negative
   control 9.5e-2 / 5.8e-2; the unigrid is a pinned two-sided band (0.93 vs chain,
   0.90 vs SC).
4. §8 octree closure, then T4.  **Status: done** — `cell_volumes` / `face_powers` /
   octree `flux_divergence` added (unigrid path bit-unchanged), T4 18 tests / 3:41 green
   including the `slow` convergence test; finding F7 pinned (closure residual = face
   extrapolation + midpoint volume rule; octree closure tracks `h0`, not `max_depth`).
5. Mask/χ² helpers validated against synthetic Gaussian data + the measured F3 numbers;
   then T3.  **Status: done** — 13 tests / 3:13 green (fixed seeds, two-seed rule, masked
   z for A2 next-flight and for the adjoint all-sky, aggregate error = aggregate sigma to
   0.2 %, MC order 0.500 for a 4x flight increase); finding F9 recorded above.
6. Ladder + reference, then a **threshold-pinning pass**: fill in every pinned band from a
   real run and record the numbers in §2 of this plan and in `docs/solver-matrix.md`.
   **Status: done** (`docs/solver-matrix.md` written in step 8) — T5 10 tests / 14:37 green (module
   docstring + bands carry the measured numbers: yardstick self-error 6.8e-3 rms J at the
   τ-saturation floor, REF rung 5.5e-3 / 5.3e-2, τ-ladder slope 0.6 in `tau_max` with
   max-J and mean-J floors pinned as bands, unigrid Cauchy slope 1.03 with the
   vs-u96 distance only monotone+banded (F11), anchor 3 per family (octree/march 1e-15,
   chain 1.2–1.4e-1), MC order −0.49 by seed splitting, cost ranking oct ≥ chain ≥ sc,
   δ-M shift 4.5e-4 ≪ rung self-error).
7. Phase M6 (MC ground) with its own verification (§7), then re-enable the all-sky case
   across all tiers and update the capability matrix.  **Status: M6 done** —
   `tests/test_mc_ground.py` (13 tests) green; T2 rejection guard replaced by positive
   smoke tests (T2 now 12 tests, green); T3 §4 all-sky cross-family MC-vs-SC green
   (T3 now 15 tests).  Prerequisite fixes: **F12** (uninitialised `tf` buffer
   made every axis-parallel-direction MC run nondeterministic; one-line fix in
   `rt3d/mc/tracking.py`, full MC regression green after it: 60 tests) and
   **F13** (unwrapped periodic exit points in the ground-column lookup broke
   the march-vs-solver ground identity at 1.2e-1; `wrap_xy` fold in
   `rt3d/mc/forward.py`/`backward.py`, guarded by
   `test_ground_stage2_march_identity`).  Step 7 complete: the ground variant
   of the T4 closure is pinned (`test_closure_ground_case_band`: relative
   residual -3.5e-3 oct2 / -2.2e-3 oct3 / +8.7e-2 u24, bottom-face power
   guard, oct2 > oct3 improvement per the F7 ordering).
8. `scripts/solver_matrix.py` + `docs/solver-matrix.md`; refresh AGENTS.md (its "there is
   **no test runner installed**" line and its statement that the `rt3d/` package "does
   **not** exist yet" are both stale) and add decisions-log rows to `plan/plan.md`
   (M6 + the test matrix).  **Status: done** — report runner + full report green
   (17 matrix / 12 ranking / 10 convergence / 6 mc_stats rows, 7 figures, ~10 min full
   run on the M4 Pro; A1 lambda MC dominates at 228 s); AGENTS.md refreshed; plan.md
   decisions-log rows added (M6, F12, F13, F14, test matrix).  Report-run additions
   found on the way: **F13** (§2) and the λ-vs-explicit convention rule (the lambda
   variant never tallies the uncollided beam; add `J_direct_analytic` before any
   SC comparison — −2.41 % vs −2.35 % after matching, was −8.7 % before).  Adjoint
   probe leaves must sit **inside the cloud core** (a near-vacuum leaf puts the z
   denominator wholly in the F14 gap: rms z 6.3 there vs 1.6–2.2 in the core).

## 12. Risks and open items

- **Statistical flakiness**: mitigated by fixed seeds, the two-seed requirement, and pinned
  bands instead of bare 2σ rules; F3 means some quantities can only be validated with
  masked statistics — the test states that out loud rather than hiding it.
- **Reference sufficiency**: the depth-5 octree saturates (`tau_max = 0.5` adds no leaves,
  per `transport-ranking-plan.md`), so it is a *discretization-limit* reference, not the
  continuum limit; the 642-ordinate variant stays opt-in in the report script.
- **Memory**: octree march tables scale as ~34 MB/ordinate on 50 k leaves; the gate keeps
  42 ordinates, the 642-ordinate reference is opt-in.
- **δ-M on the octree**: refinement runs on physical `κ_e` outside the solver — intended
  design — so δ-M and grid-refinement errors do not cancel; the ladder reports δ-M on and
  off for the reference rung.
- MC λ-variant + `crn` divergence on inhomogeneous media (F5): the matrix will not use
  that combination, and a guard/assert documents it.
- **λ-variant uncollided beam (resolved by the report run)**: the λ variant deliberately
  does not tally the uncollided column (its `Lambda_bar` needs removal-only transport,
  so the direct-beam `I_in e^-tau` term cannot be sampled) and ships the analytic
  column as `J_direct_analytic` instead. Any SC comparison must add it: uncorrected λ
  reads −8.7 % on the all-sky power, −2.41 % corrected, matching explicit's −2.35 %.
  `mc_j_total()` in the report applies it; the T3 §4 fixture runs only the explicit
  variant (where `J_total` already carries the tallied column), so no correction is
  needed there.  The rule is item 6 of `docs/solver-matrix.md` §6.
- `energy_balance` on the octree is only as good as its face extrapolation: pin its
  residual as a band, and fit its *improvement* under refinement rather than its absolute
  value.

## 13. Definition of done

- `pytest -m "not slow"` green in ≤ 40 min; full `pytest` green in ≤ 2 h.
- Every solver variant in §1 appears in at least one assertion, and `matrix.csv` has a row
  for each; the capability matrix contains no gap except variable ground.
- The SC ≡ MC deterministic identity holds at ≤ 1e-8 for every parametrization, and its
  negative control still fails as expected.
- Every pinned band is traceable to a measured run recorded in `docs/solver-matrix.md`.
