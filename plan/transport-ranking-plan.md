# Plan — accuracy ranking of the transport-scheme comparison ("which run is best?")

Follow-up to `plan/compare-transport-plan.md`. The six-run comparison is purely
*pairwise*: without a reference solution it cannot say which run is closest to
the (angle-discretized) exact solution. This extension adds a reference run and
grid-refinement runs to `scripts/compare_transport.py`, and ranks every run by
its error against the reference.

## 1. Why pairwise comparison is not enough

chain↔SC differ by 31% in J and unigrid↔octree by 18% — but a difference
between two approximations says nothing about which one is *right*. Both are
discretization errors of the same continuous problem; ranking them requires a
solution that is demonstrably closer to the continuum limit.

## 2. New runs (4 + 1 reference)

Same case, same 42 ordinates, same Ng/tol settings as the six base runs
(except the two reference-grid runs — see the addendum at the end).

| key | grid | transport | delta-M | role |
|---|---|---|---|---|
| `uni_chain48_dM1` | unigrid 48³ (h = 0.208 m) | chain | 1 | chain grid-refinement (24³ → 48³) |
| `uni_sc48_dM1`    | unigrid 48³ | sc | 1 | SC grid-refinement |
| `octree_d4_dM1`   | octree tau_max = 1.0, max_depth 4 (34 476 leaves) | exact march | 1 | octree refinement step d3 → d4 |
| `octree_ref_dM1`  | octree tau_max = 0.5, max_depth 5 (50 380 leaves, h_min = 0.078 m) | exact march | 1 | fine-grid δ-M equivalence + efficiency candidate |
| `octree_ref_dM0`  | same octree as `octree_ref_dM1` | exact march | 0 | **THE REFERENCE** |

- The reference is run with **delta-M off** so it carries no O(tau_cell)
  explicit-term quadrature gap; at h_min = 0.078 m it is the finest
  discretization the solver can build on this case (the kappa_e-driven
  refinement saturates: tau_max < 0.5 no longer adds leaves at depth 5).
- `octree_ref_dM1` vs `octree_ref_dM0` re-verifies the delta-M equivalence at
  the finest level (expected: smaller than the 1.3% seen at depth 3).
- delta-M = 1 for the refinement runs: it does not change the converged
  solution (established in the base study) and cuts their cost ~3–4×.

## 3. Ranking diagnostics

All runs (11 total) sampled pointwise onto the same 64³ lattice
(h = 0.15625 m, exact for piecewise-constant fields; the reference's 0.078 m
cells are sampled pointwise, not averaged).

- **Error vs reference**: max/mean relative difference of J, sigma^2, F+ (TOA)
  for every run against `octree_ref_dM0` → `ranking.csv` + printed table.
- **Convergence evidence** (this is what makes the ranking meaningful):
  - chain 24³ → 48³: error vs reference should shrink ~2× (first-order h),
  - sc 24³ → 48³: same, from a larger base,
  - octree d3 → d4 → d5: shrinking error,
  - scalars <J> and mean F_z @TOA approaching the reference values.
- Figures: `j_err_maps.png` (|J - J_ref| per run), `j_convergence.png`
  (band-mean J profiles grouped by scheme + reference),
  `err_bars.png` (max/mean J error per run, log scale),
  reference curves added to `j_profiles.png` / `toa_mu_spectra.png` families
  via the new figures; scenes: `scene_octree_ref_dM0.html` + three
  |dJ|-vs-reference scenes.

## 4. Deliverables

- extended `scripts/compare_transport.py` (run matrix, ranking block, new figures)
- new outputs in `plot/compare_transport/`: `ranking.csv`, `j_err_maps.png`,
  `j_convergence.png`, `err_bars.png`, extra scenes
- new section "Which run is best?" in `docs/compare-transport.md`, refreshed
  wall times (rerun on Windows), updated findings

## 5. Expected outcome (hypothesis, to be verified by the numbers)

- accuracy: octree (finest) > unigrid chain > unigrid SC at equal cost budget;
  refinement errors shrink monotonically toward the reference, confirming all
  pairwise gaps are discretization, not physics;
- best accuracy per wall time: `octree_ref_dM1` / `octree_dM1` for accuracy,
  `uni_chain_dM1` for cheap-and-acceptable;
- delta-M on is strictly better than off at identical accuracy (same converged
  fields, 2–4× less wall time) — so among the original six, the dM1 variants
  dominate their dM0 twins.

## 6. Verification

- reference self-consistency: `octree_ref_dM1` vs `octree_ref_dM0` ≤ the
  depth-3 δ-M gap (≤1.3% max);
- octree d3 vs d4 error vs reference < octree d3 pairwise gap pattern;
- chain/sc 48³ errors ≈ half the 24³ errors (first-order h);
- TOA mean F_z of refinement runs moves monotonically toward the reference.

## 7. Addendum (2026-10-15) — 642 ordinates for the two reference-grid runs

The two runs on the fine reference octree (`octree_ref_dM0`, `octree_ref_dM1`)
are re-run with icosahedral refinement 3 = **642 ordinates** instead of 42, so
the reference is finer in *angle* as well as in space. Consequences:

- the ranking errors of all 42-ordinate runs now *include* their angular
  (ordinate-resolution) error — the previous "the 42-ordinate angular error
  cancels in every comparison" no longer holds against the reference;
- comparisons *among* the 42-ordinate runs (the six base pairs, the unigrid
  refinement pairs) are unchanged;
- the δ-M equivalence check `octree_ref_dM1` vs `octree_ref_dM0` stays valid
  (both at 642 ordinates);
- the octree convergence sequence d3 → d4 → ref is no longer purely spatial
  (the last step changes grid *and* angle);
- cost: the reference runs' sweeps scale ~linearly in N_dir (15.3× per
  iteration) and the octree march precomputation stores ~34 MB per ordinate
  on the 50 380-leaf grid (~22 GB for 642 ordinates) — needs a ≥32 GB machine.
