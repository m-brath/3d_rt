# Solver test matrix — tiers, reference ranges, findings

The full cross-solver acceptance check for `rt3d`: every solver variant of both
families (short characteristics and Monte Carlo) runs on one physical setup,
correctness is established against a **fine-octree reference and refinement
ladders** (not just pairwise agreement), and every accepted imperfection is
**recorded as a reference range** with a measured cause.

Design document (decisions, probe history): `plan/solver-test-matrix-plan.md`.
This page is the readable summary: what each tier asserts, which reference ranges are
recorded, and what the findings F1–F14 mean for interpreting the numbers.

Two front ends share the same helpers (`tests/matrix_cases.py`,
`tests/matrix_stats.py`), so acceptance check and report cannot drift apart:

| front end | what it is |
|---|---|
| `pytest tests/test_matrix_*.py` | the acceptance check (T0–T5), markers `slow` / `mc` / `reference` (`pytest.ini`) |
| `scripts/solver_matrix.py` | the report: CSVs + figures under `plot/solver_matrix/`, disk solve cache in `plot/solver_matrix/_solve_cache/` (gitignored) |

```sh
~/miniforge3/envs/pyarts/bin/python -m pytest -q                     # everything
~/miniforge3/envs/pyarts/bin/python -m pytest -q -m "not slow"       # acceptance check without ladders
~/miniforge3/envs/pyarts/bin/python scripts/solver_matrix.py         # report (~10 min)
```

## 1. The case

The matrix case is the MANUAL §11 worked example: the `class_test.py` Gaussian
cloud (`omega_cap = 0.9` → max single-scattering albedo 0.9, max `g` 0.925,
τ ≈ 2 through the core) in a 10 × 10 × 10 m periodic box, icosahedral
refinement 1 = 42 ordinates, solar beam at 30° zenith, and the full all-sky
boundary: top inflow `I = 1` plus a Lambertian ground (albedo 0.3,
emissivity 1, `B_g = 1`). Since phase M6 **every family runs every case**;
the only unsupported configuration anywhere is a variable-height ground
(guarded by `NotImplementedError` in both families).

Grid rungs (development sizes): `u12³/u24³/u48³` unigrids; octrees with base
cell `h0 = 1.25` refined to `tau_max ∈ {4, 2, 1, 0.5}`; the **reference rung**
`REF = tau_max 0.5, depth 5` (50 380 leaves). T5's yardstick (not stored in
outputs, used to bound the reference's own error): `tau_max 0.25, depth 7`
(107 108 leaves).

## 2. Tiers

| tier | file | asserts | scale |
|---|---|---|---|
| T0 | `test_matrix_invariants.py` | AGENTS.md invariants 1–3 per variant (direction weights, HG normalization incl. MC samplers, Beer-Lambert on identical rays for chain/sc/octree/jit/numpy/MC cell+global/adjoint/beam-τ) | 43 tests, ~2 s |
| T1 | `test_matrix_equivalence.py` | deterministic variant equivalence: jit vs numpy and n_jobs bit-identical; δ-M 1/2 vs 0 ranges; Ng fixed point; chain vs sc range; ground/beam/periodic structure; guards | 20 tests, ~30 s |
| T2 | `test_matrix_crossfamily.py` | zero-noise identity: MC deterministic line integral fed the SC's own source **≡** the SC octree sweep (1.2e-12 max, δ-M 0/1/2, beam on/off, both majorant modes, jit on/off), negative control (physical κ under δ-M fails by ≥ 1e-2), **ground stage-2 identity (F13 guard)**, unigrid recorded as reference range (F11), positive M6 ground smoke tests for all three MC entry points | 13 tests, ~2 min |
| T3 | `test_matrix_statistics.py` | statistical consistency under the noise model (§6 of the plan): A1 explicit vs λ, A1 vs SC, seed-pair pooled z, next-flight F3 noise masking, adjoint masked z, **all-sky ground MC vs SC cross-family** (section 4) | 15 tests, ~3 min |
| T4 | `test_matrix_closure.py` | invariant 4 per family with the octree-capable closure: face powers vs `∫4πκ_a(B−J)dV`, analytic-integral identities, the ground-case closure (bottom-face power includes reflected + emitted parts), cross-family absorbed power (SC-42 vs SC-162 vs MC A1) | 21 tests, ~4 min |
| T5 | `test_matrix_ladder.py` | refinement ladders per family, reference ranking, order fits, per-family invariant-3, MC `N^-1/2` by seed splitting | 10 tests, ~15 min |
| M6 | `test_mc_ground.py` | the MC ground law against closed forms: vacuum emission, Beer-Lambert + albedo continuum, adjoint cones, reflected-beam populations, stage-2 vs SC, determinism pairings | 13 tests, ~25 s |

## 3. Recorded reference ranges (measured thresholds)

Everything below is *measured*, not assumed; a drift outside a reference range
fails the acceptance check with the range name in the message.

**Deterministic iteration / equivalence (T1).** δ-M 1/2 vs δ-M 0 converged-field
gap ≤ 1.5e-2 (J) / 6e-2 (I) on u24³ (measured 8.4e-3 / 4.1e-2 and 8.5e-3 /
3.1e-2); octree 2.5e-2 / 5e-2. Ng vs plain lambda: ΔJ/J = 3.8e-7. jit/numpy
and n_jobs: bit-identical.

**Cross-family identity (T2).** Octree march vs SC chain: 1.2e-12 (same
source); vs the solver's own `I`: ≤ 5e-9 at tol 1e-8. With ground: same
identity to 1e-6 (measured 8.6e-10; F13 guard). Negative control feeding
physical κ_e under δ-M: dev 9.5e-2 (M1) / 5.8e-2 (M2). Unigrid: march vs
chain 0.93 (I) recorded as a reference range — F11, not a bug.

**Statistics (T3).** Pooled per-cell z of two independent A1 runs:
rms ∈ [0.65, 1.05] (measured 0.83–0.86 — batch σ is mildly conservative),
|mean z| ≤ 0.10, |z| > 3 ≈ 0. All-sky ground seed pair (120k histories ×
2 seeds, merged ground/beam tallies): rms z 0.635, tail3 0, so the merged
tally σ is valid; seed-split aggregate power noise 1.8× the per-cell σ
quadrature (range [0.5, 3]). Adjoint masked z vs SC columns: rms ∈
[0.60, 1.40] (measured 0.97) with the F3 noise mask; unmasked the z blows
to ~1e25 on τ ≈ 0 rays — the mask is part of the estimator contract.

**All-sky ground MC ↔ SC (T3 §4).** Conventions matched first: MC `J_total`
includes the uncollided-beam column, SC `J` does not; SC gets the analytic
`F0 exp(-tau_b)/4pi` column added (worth +6.8 % of the power). Then
absorbed power MC − SC ∈ [−0.05, 0] % (measured **−2.15 %**), leaf-mean J
∈ [−0.04, 0] (−1.69 %), per-cell rms z vs the SC column ∈ [0.5, 1.2]
(0.795, mean −0.30 — a coherent quarter-σ offset, part of it F14).

**Closures (T4).** Relative invariant-4 residual: conservative test medium
3.3e-5 (u24) → 6.6e-7 (u48), −1.4e-3 (octree h0 1.25) → −1.3e-4 (h0 0.625);
matrix case −1.0e-2 (oct2) / −9.7e-3 (oct3), unigrid +4.2e-2 (chain, F11)
and −2.2e-1 (sc stencil); **ground case** −3.5e-3 (oct2) / −2.2e-3 (oct3) /
+8.7e-2 (u24), with the bottom-face power guard (`out_bottom < −100`: a
dropped ground term fails loudly). F7: the residual is dominated by the
face-extrapolation error, which improves with base-cell size — the
midpoint-cell-center volume rule is not the limiter.

**Ladders and orders (T5).** REF self-error (5→7 rungs): 5.5e-3 rms J /
5.3e-2 max J — at the τ-saturation floor (a 7e-3 rms **J** corresponds to
~2.5 J in per-cell absolute terms in optically thick cells); REF vs u96:
0.046 rms / 0.22 max — F11's plateau, not grid coarseness. Octree τ-ladder
errors ~2e-2/8e-3/6e-3 (τ4/2/1) with fitted slope 0.6 in `tau_max`;
`J_max`/`mean_J` recorded as floor ranges (they saturate, F10). Unigrid Cauchy
pairs 2.8e-2 → 1.4e-2 → 6.7e-3, fitted slope 1.03 — internally textbook
first order. MC power seed-split error ∝ `N^-0.49`.

## 4. Findings F1–F14 — how to read the numbers

* **F1** (closed by M6): the MC family used to reject the Lambertian ground;
  every §2/§4 ground row in the capability matrix now reads "supported".
* **F2**: the stage-2 march fed the solver's own source must use
  `kappa_e_delta` (δ-M), and the source must stay direction-resolved;
  physical κ fails the identity by ≥ 1e-2 (T2 negative control).
* **F3**: next-flight per-cell σ is invalid (zero/undefined) where flights
  never score; every MC-vs-reference z statistic goes through
  `matrix_stats.next_flight_mask` — unmasked z is meaningless (1e25 tails).
* **F5**: common random numbers couple populations — CRN off for
  inhomogeneous fields.
* **F6**: Ng acceleration is *not* a speed win on this cloud (iteration
  counts recorded, "no pathological slowdown" asserted).
* **F7**: octree closure residual = face-extrapolation error; `h0`, not
  `depth`, sets closure accuracy (see T4 ranges).
* **F9**: batch σ is honest *per cell* (A2 masked rms z ≈ 1.06 at 100
  flights) but the **aggregate** per-cell σ quadrature underestimates
  whole-domain error 2–3× (track correlation). Aggregate statements use
  seed splitting, never the quadrature alone.
* **F10**: the octree's three resolution knobs are not interchangeable —
  the ladder runs in `tau_max` at non-binding depth; the depth ladder
  saturates; `h0` barely moves cell-centered fields (it moves *faces*).
* **F11**: unigrid `chain`/`sc` carry an **h-independent** grazing-ordinate
  transport error (~12 % rms vs analytic `exp(-tau)` in the pure-absorber
  limit; octree/march exact to 1e-15). The unigrid family plateaus ~4 %
  rms J from the octree reference at every resolution. Recorded as a finding
  (solver unchanged); invariant 3 is asserted **per family**.
* **F12** (fixed 2026-11-16): the MC segment marcher filled face-crossing
  times into `np.empty`, leaving `da == 0` entries (beam `dy = 0` exactly,
  z-poles) as uninitialized heap garbage → identical seeds gave different
  runs (±5 % power) on any case with an axis-parallel direction. Fixed
  (`np.full(..., np.inf)` in `rt3d/mc/tracking.py`); guarded by equal-seed
  pairings in `test_mc_ground.py` at 1e-12. All pre-fix all-sky probe
  numbers were artifacts.
* **F13** (fixed 2026-11-16): the stage-2 march and adjoint looked up the
  ground column of a bottom exit at the **unwrapped** periodic exit point;
  `ground_column_lookup` silently clamped to boundary columns, breaking the
  F2 march-vs-solver identity by 1.2e-1 on the octree ground case (only
  with per-column varying `ground_I` — constant ground radiance hid it in
  all unit tests). Fixed with `periodic_box.wrap_xy` before the lookup;
  guarded by `test_ground_stage2_march_identity`. Post-fix: 5.6e-17 /
  8.6e-10 identity, adjoint bias gone.
* **F14** (documented design difference, not a bug): SC normalizes the beam
  first-scatter phase function on the direction grid
  (`sum_i w_i Phi_i = 1`), the MC families sample the **analytic** HG. On
  42 ordinates the raw quadrature defect is 4–8× at cloud-core
  `g = 0.9–0.925`, so **beam-case cross-family per-ordinate comparisons are
  angle-limited, not noise-limited** (adjoint vs SC columns: coherent 6–15 %
  deviations, masked rms z up to 30). It contributes to the −2.15 % T3 §4
  power offset. Refining the direction grid (T5 study) drives the two
  conventions together; on 42 ordinates every beam per-ordinate range must
  carry this caveat.

## 5. Report artifacts

`scripts/solver_matrix.py` (plan §9) writes, under `plot/solver_matrix/`:

* `matrix.csv` — one row per variant: iterations, converged flag, wall time,
  field size, mean J, absorbed power, top/bottom face powers, closure
  residual, error vs REF, notes;
* `ranking.csv` — every variant's max/rms J error, mean-J and power error vs
  REF (REF itself at 642 ordinates is opt-in via `--reference-ordinates 3`);
* `convergence.csv` — unigrid Cauchy pairs + slope, octree τ-rungs vs REF,
  MC power seed-split vs `N` + slope;
* `mc_stats.csv` — masked/unmasked z stats per MC estimator (the F3 story),
  with the F9/F14 caveats in the row notes;
* figures: `j_err_maps.png`, `j_convergence.png`, `err_bars.png`,
  `z_hist.png`, `mc_convergence.png`, `closure.png`, `capability_matrix.png`.

`--quick` smoke-runs the machinery on a reduced variant set. The solve cache
is keyed by variant id and direction level; stale entries re-solve.

## 6. Interpretation rules (summary)

1. Never difference unmasked MC z statistics (F3).
2. Never use the per-cell σ quadrature for aggregate-domain error (F9);
   split seeds instead.
3. Cross-family octree identities are exact (F2) — anything above 1e-6 in a
   same-grid same-source comparison is a bug, as F13 shows.
4. Unigrid ↔ octree gaps ≈ 4 % rms J are F11 plateau, not convergence.
5. Beam-case per-ordinate MC ↔ SC gaps on 42 ordinates are F14-limited.
6. MC `J` is angle-averaged; MC `J_total` includes the uncollided beam, SC
   `J` does not — add `_beam_j_analytic` to SC before differencing. The
   *lambda* MC variant never tallies the uncollided column at all (the beam
   enters its fixed population only through first scatter) and ships the
   analytic column as `J_direct_analytic` — add it to the λ field before
   comparing, or λ reads ~6.5 % low (measured: −8.7 % uncorrected, −2.4 %
   corrected, matching explicit's −2.4 %).
7. Adjoint all-sky probe leaves go **inside the cloud core** (max
   `kappa_a` per z-slab): at a near-vacuum leaf the z denominator sits
   wholly in the F14 quadrature gap (rms z 6.3 vs 1.6–2.2 in the core).
