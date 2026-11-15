# Plan — Monte Carlo solvers (forward two-stage + backward adjoint), with null collisions

Status: **M0 + M1 + M2 + M3 implemented** (M0: 2026-10-24, M1: 2026-11-13,
M2: 2026-11-14, M3:
2026-11-15): `rt3d/mc/`
package (`tracking.py` Woodcock core + unigrid/octree DDA walkers + analog
history chains, `sources.py` HG inverse-CDF + birth/face/beam samplers,
`estimators.py` track-length tallies, `forward.py` stage-1 `j_mc` both
variants + stage-2 `radiance_from_source` / `stage2_source`, `backward.py`
adjoint `backward_detector_mc` / `backward_all_sky`), shared case
`rt3d/cases/slab_1d.py`,
`scripts/mc_phase0_check.py`, `scripts/mc_forward_slab.py`,
`scripts/mc_forward_cloud.py`, `scripts/mc_backward.py`,
`tests/test_mc_tracking.py` (9) + `tests/test_mc_sources.py` (12) +
`tests/test_mc_forward.py` (13) + `tests/test_mc_radiance.py` (5) +
`tests/test_mc_backward.py` (8). All M0 validations green. M1 slab
validations green (J vs 1D-DO within 2 sigma + discretization allowance,
variant agreement, Ng 14 vs 89 iterations at omega = 0.9, global closure
|rel| ~ 1.3% — better than the SC-42 4.7%). M3 adjoint validations green
(closed-form absorber 3 sigma, slab/beam/octree vs SC per-node 2 sigma + 8%,
cloud all-sky at node directions 21/21 inside 2 sigma + 10% rms-sky with
hemispheric upward-flux closure adjoint −2.2% vs the SC-162 reference (SC-42
itself −2.0%), narrow-sensor
efficiency figure `plot/mc_backward.png`). Adds a second solver family next
to the short-characteristics solver (`rt3d/`, phases 0-6 done). This plan
reuses the octree (`rt3d/octree.py`), unigrid (`rt3d/geometry.py`),
direction grid (`rt3d/directions.py`), ground BC
(`rt3d/transport.ground_surface`), and the SC results of `lambda_iteration`
as cross-validation reference.

M1 implementation notes (how the code differs from the sketch above, all
verified by `tests/test_mc_forward.py` and `scripts/mc_forward_slab.py`):

- **Emission-rate convention**: the swept equation is
  `Omega.grad I + kappa_e I = kappa_e * S`, so all MC birth rates must be
  `kappa_e * S` (thermal: `kappa_a B`, already the rate).  Omitting the
  `kappa_e` factor rescales the MC `Lambda` operator by `1/kappa_e`; for
  `kappa_e < omega-less-than-1` domains the lambda iteration then diverges
  (observed: `kappa_e = 0.5`, `omega = 0.8` blows up to `J ~ 1e66`).
- **Track-length tally** (`estimators.py`): `J_c = sum_k w_k ell_{c,k} /
  (4 pi V_c)`, `F_c = sum_k w_k ell_{c,k} Omega_k / V_c` (laid out
  `(n_cells, 3)` like `rt3d.source.flux_vector`).  Error bars from batch
  sums: `est = sum(sums)`, `sigma = sqrt(n_batches) * std(sums, ddof=1)`.
- **Lambda-variant sampling defaults**: `crn=False`, `birth="source"`
  (fresh photon set per iteration, births from `kappa_e S_n`).  Stable on
  slabs, beams and the gaussian cloud, but the residual stalls at the tally
  noise floor instead of `tol`.  `crn=True` (frozen stream + `birth="uniform"`)
  converges cleanly and enables Ng **only for mild `kappa_e` variation**; on
  the cloud the frozen uniform-birth operator has too few effective samples
  per high-`kappa_e` cell and its spectral radius exceeds 1 — the iteration
  diverges (diagnostic: `s_max_history`).
- **Beam bookkeeping** (lambda variant): exactly **one** first-scatter child
  per primary (collision pdf `kappa_e e^{-kappa_e s}` = the Beer-Lambert
  attenuation); the primary is then done — deeper collision orders come from
  the `omega * J` source update, mirroring how the SC solver re-scatters
  `beam_scat` without ever multiplying it by `omega` twice.  The iterated
  source `S` must **not** also carry the angle-averaged first-scatter source:
  the bfs tally already *is* its transported J, adding both double-counts.
  The explicit variant follows unsplit particle physics and reports the
  unscattered beam track separately as `J_direct` (matches `F0 e^{-tau_b}/4pi`).
- **Stage 1 has no direction grid / `n_refine`** — no angular quadrature in
  MC stage 1; the icosahedral grid enters with stage-2 tally binning (M2).
- **Anchor 4 for MC** is checked globally (net downward face flux vs
  `4 pi kappa_a int J dV`): pointwise `div F` vs heating is truncation-
  dominated at the topmost slab interfaces for *both* solver families
  (MC ~ 12-18%, SC ~ 8% of peak heating), while the global closure is
  ~1.1-1.3% for MC vs 4.7% for SC-42.
- Ground BC (`bc.ground`) raises `NotImplementedError` in M1 (out of phase
  scope); `kappa_max` modes, octree DDA and periodic wrap are exercised on
  the cloud.

M2 implementation notes (stage-2 radiance, `radiance_from_source` +
`stage2_source` in `forward.py`, verified by `tests/test_mc_radiance.py` and
`scripts/mc_forward_cloud.py`):

- **Estimator deviation from the sketch**: the next-flight tally scores
  `S(cell, d)` at the first *real* collision with **no `/kappa_max`
  division** — the real-collision density `kappa_e e^{-tau}` already
  reproduces `int kappa_e S e^{-tau} ds` exactly in both majorant modes
  (nulls are a tracker detail, not an estimator factor).
- **Deterministic == SC on the octree, bit for bit**: the `S (1 -
  e^{-tau_seg}) e^{-tau_before}` segment recursion is identical to
  `sweep_direction_octree_exact`, so feeding the SC's converged source
  reproduces `lambda_iteration`'s `I` (and quadrature `J`) at 0.0 rel err —
  provided the feed is faithful: `ng=None` (the Ng-accelerated `S` is not
  the source that produced `I`) and the **direction-resolved** `S` passed
  per ordinate (with per-cell g the `(n_cells, N_dir)` columns differ;
  collapsing to one column costs ~12% on the cloud).
  On the unigrid the SC `chain`/`sc` sweeps are different path stencils;
  there the deterministic estimator is validated against the closed-form
  Beer-Lambert integral instead.
- **Statistics criterion**: aggregate z-scores (mean ~ 0, rms ~ 1, coverage)
  on sigma-resolved cells, not raw pointwise 2-sigma bands — the binary
  tally's batch sigma collapses to 0 in optically thick cells and is ~10–20%
  low at 10 batches (see decisions log; measured rms z = 1.04–1.07,
  sigma ~ 1/sqrt(n_hist)).
- Uncollided boundary term from the exact backward-march `tau_tot` with the
  inflow of the face the march exits (`bc.inflow` / `_entry_I_in`
  convention); Lambertian ground raises `NotImplementedError`.

M3 implementation notes (backward adjoint solver, `rt3d/mc/backward.py`,
validated by `tests/test_mc_backward.py` and `scripts/mc_backward.py`):

- **Score deviation from the plan sketch (important)**: the adjoint walk
  scores `(1-omega) B + omega F0 e^{-tau_b} Phi_HG(Omega_sun -> Omega_f)`
  at every *real* collision and carries the diffuse rescattering by the
  walk continuation itself (HG reciprocity = same HG about the reversed
  direction, weight 1).  The sketch's `omega J` in the score **plus** HG
  continuation would re-expand the already-scattered stage-1 field at all
  orders and over-count.  Consequence: solver B needs **no stage-1 `J`** —
  it solves the closed RTE (B, bc, beam) directly and is a genuinely
  independent family next to solver A.  Because the real-event density is
  `kappa_e e^{-tau}`, scores carry no `kappa_e` factor
  (`kappa_e (1-omega) B = kappa_a B`, `kappa_e omega Phi = kappa_s Phi`);
  the walk scores the face inflow when it exits the domain.
- Detector: point or plane aperture + cone about `omega0`; uniform aperture
  and cap sampling make the output the aperture/cone-**averaged radiance**
  (no extra `Delta-omega` factor).  `backward_all_sky` runs one equal-area
  icosahedral cap per upward ordinate (caps tile the hemisphere).
- Validation anchors: closed-form Beer-Lambert slab (3 sigma), closed
  scattering slab and beam slab vs converged `lambda_iteration` I per
  upward ordinate (2 sigma + 8%), octree slab, global-mode nulls, all-sky
  consistency at **exact node directions** (`cone_scale = 0` default:
  slab 2 sigma + 5% rms-sky, cloud 2 sigma + 10% rms-sky + hemispheric
  upward-flux closure vs SC-42 and an SC refine-2 / 162-ordinate strong
  reference — see decision log: SC-42 *per-ordinate* radiance in the
  `g ~ 0.55` cloud has angular errors far beyond the per-cell J errors;
  full icosahedral caps cross the horizon and are not node-comparable).
- Efficiency (measured, honest): at equal *single-ray* cost the fixed-source
  next-flight tally scores once per history and has slightly lower
  variance per collision than the walk; the adjoint win is the whole-image
  cost — one all-sky image costs `N_up x n_seeds` walks instead of
  `n_leaf x N_dir x n_h` forward rays (17 000x fewer rays on the cloud) —
  and the absence of the whole stage-1 iteration.

Decisions taken with the user up front:

1. **Stage 1 exists in both variants** — explicit-scattering MC (one run, no outer
   iteration) *and* an MC lambda iteration mirroring the DOIT structure of the SC
   solver; selectable via `stage1="explicit" | "lambda"`.
2. **Null-collision scope**: Woodcock tracking with per-cell `kappa_max = kappa_e(cell)`
   as default (zero nulls on a piecewise-constant octree, best efficiency), plus a
   deliberately implemented and tested **global-`kappa_max` mode** in which real null
   collisions occur; the null-collision machinery is first-class, not incidental.
3. **Both spatial grids from the start**: unigrid DDA and octree DDA traversal kernels.

## 1. Physics / MC formulation

Same monochromatic LTE RTE as plan/plan.md section 1 — the MC family must solve
exactly the same problem so the two families can be differenced:

    Omega . grad I + kappa_e I = kappa_s * int Phi_HG(Omega'->Omega) I dOmega' + kappa_a B

with the same solar-beam split (`omega_sun`, `F0`), the same boundary conditions
(top `I_diff = 0`, Lambertian+emissive ground via `boundary_conditions(ground=...)`,
periodic in x/y), and the same derived quantities (`kappa_a = kappa_e - kappa_s`,
`omega = kappa_s / kappa_e` guarded against `kappa_e = 0`).

MC identities used throughout:

- **Flight sampling**: path length from `exp(-int kappa_max ds)` with
  `kappa_max >= kappa_e` (Woodcock). Event type at each collision by comparing
  `R * kappa_max` against cumulative rates: scatter `kappa_s`, absorption `kappa_a`,
  **null `kappa_max - kappa_e`** (direction and weight unchanged, flight continues).
- **Scatter direction**: HG inverse-CDF sampling, no quadrature, no delta-M.
  With `u ~ U(0,1)`: for `g != 0`, `q = 2 g u / (1 - g^2) + 1/(1 + g)`,
  `cos(theta) = (1 + g^2 - 1/q^2) / (2 g)`; azimuth uniform. Limit `g -> 0`:
  isotropic (`cos(theta) = 2u - 1`). Must reproduce `hg_matrix` moments to
  quadrature accuracy (anchor 2 companion test).
- **Next-flight (collision) estimator**: a collision sampled with density
  `kappa_e e^{-tau(s)}` at `r_c` contributes `S(r_c) / kappa_e(r_c)` to the line
  integral of `S e^{-tau}` to *every downstream point* of that ray
  (`E[1_{s_c<s} S/kappa_e] = int_0^s S e^{-tau} ds'`). This is what makes both the
  Stage-2 radiance tally and the backward tally O(collisions), not O(segments).
- Reciprocity `Phi_HG(Omega'.Omega) = Phi_HG(Omega.Omega')` is what makes the
  adjoint (backward) transport identical physics run reversed — no separate adjoint
  phase function needed.

HG note: MC transport has **no 42-ordinate resolution limit for `g > 0.5`** — the
transport is continuous in angle; the icosahedral grid enters only as tally binning
in Stage 2 / backward solvers. Delta-M has no meaning for MC and is not supported.

## 2. Null-collision design (`rt3d/mc/tracking.py`)

The tracker is the shared core of all solvers. One code path, two modes:

- **mode="cell"** (default): on entry to a cell (unigrid or octree leaf) set
  `kappa_max = kappa_e(cell)`; the piecewise-constant profile makes nulls exactly
  zero — every collision is real — while the code remains a Woodcock tracker.
- **mode="global"**: `kappa_max = max_domain(kappa_e)` (or a supplied cap): real
  null collisions in low-opacity/vacuum regions. Enables a fully vectorized NumPy
  photon-batch loop (advance all live photons one flight at a time, cheap nulls in
  vacuum) and is the reference implementation of the null-collision concept.
- Both modes share the same event-type decision logic; the unit tests assert that
  cell and global mode agree statistically (same seed, different stream) on the
  transmittance anchor.

Null collisions are also load-bearing in two further places:

- **Birth-point sampling from inhomogeneous rates** (thermal source `kappa_a B`,
  and the fixed source `S_n` in the lambda-iteration variant): sample positions by
  Woodcock against a source-rate profile `kappa_src_max >= S_n(r)` with null source
  events; flight afterwards uses the ordinary `kappa_max` tracker. Two chained
  Woodcock processes (birth + flight), one tracker class.
- **Solar beam as a particle stream**: beam photons are seeded at the top entry
  face along `omega_sun` with weight `F0`, fly with the same tracker, split into
  diffuse HG directions at collisions with probability `omega`, otherwise true
  absorption terminates them. One unified loop for thermal/beam/diffuse particles
  instead of a separate analytic beam sampler (the analytic
  `beam_scattering_source` / `beam_optical_depth_octree` stay as cross-checks).

Traversal kernels (both from the start, numba JIT + NumPy reference fallback per the
`rt3d/jit.py` pattern):

- `mc/unigrid`: DDA over the `rt3d.geometry.unigrid` lattice with periodic wrap
  (reuse the face-snap fix from the phase-5 bugfix), per-cell kappa lookup O(1).
- `mc/octree`: DDA over the octree using the same per-level lattice tables as
  `octree_grid.locate`; step from cell to cell, `kappa_max` from the leaf.

Variance-reduction switches (M4): implicit capture (scatter/absorb roulette with
weight `omega` on scatter), Russian roulette with weight floor, optional splitting
of beam collisions. RNG: NumPy `PCG64` (reference path), reproducible seeds; numba
path uses its own seeded generator; `seed` and `n_histories` recorded in outputs.

## 3. Solver A — forward MC, two stages (`rt3d/mc/forward.py`)

### Stage 1: source field J -> S

Entry point `j_mc(grid, kappa_e, kappa_s, B, bc, g=0.0, stage1=..., ...) -> J, sigma_J`.

- **stage1="explicit"**: classic generation-based MC. Thermal photons born at rate
  `kappa_a B` (null-collision birth sampling), beam stream as above, physical
  scattering regenerating diffuse photons, Russian-roulette chain termination.
  `J` tallied per cell with the **exponential-path estimator**: each flight between
  cells `a -> b` contributes its (attenuated) path length weighted by
  `4pi`-normalized history weight to the crossed cells. No outer iteration; `J`
  and its history-batch standard error come out of one run.
- **stage1="lambda"**: MC transport operator applied to the current source.
  Loop `S_0 = (1-omega) B + beam_scat`; each iteration: sample photons born from
  `S_n(r)` (null-collision source sampling), fly with true absorption only (no
  particle scattering), tally `J_n` by the same exponential-path estimator;
  `S_{n+1} = (1-omega) B + omega J_n + beam_scat`; converge on `||dS||/||S||`.
  This is the DOIT-structure twin of `rt3d.solver.lambda_iteration`.
  **Ng coupling** (`rt3d.accelerator.ng1d`) is *allowed but opt-in*: it only behaves
  with **common random numbers** (identical photon streams per iteration, fixed
  base seed) so the `Lambda_bar` history pairs see correlated noise; documented as
  experimental, plain-LI is the default for this variant.
- The two variants must agree within 2 sigma on the slab case (consistency anchor).

### Stage 2: radiance from the fixed source (`radiance_from_source`)

Entry point `radiance_from_source(oct_or_grid, dg, kappa_e, S_total, bc, ...) -> I[(n_cells, N_dir)], sigma_I`.

- Ray set = **identical to the SC octree marches**: one ray per leaf center per
  icosahedral ordinate `Omega_i` (`precompute_octree_march`-style segment lists,
  `node_major_segments` deposition). Unigrid twin: per-cell ray per ordinate
  through cell centers with the lattice geometry. This makes Stage 2 output
  directly differencable against `lambda_iteration` output, cell by cell,
  ordinate by ordinate.
- Per ray, per history: sample collision points by the Woodcock flight sampler;
  each collision at `r_c` deposits `S_total(r_c) / kappa_max(local)` into every
  downstream cell of that ray (next-flight estimator; division by `kappa_max` makes
  the estimator unbiased in global mode). Uncollided rays pick up the boundary
  inflow `I_in(Omega)` (top = 0/background, ground radiance from
  `ground_inflow`, periodic entry) at the ray origin.
- `S_total = (1-omega) B + omega J + beam first-scatter` — assembled by reusing
  `rt3d.source.scattering_source`-side bookkeeping in scalar (cell) form.
- Reference estimator **`estimator="deterministic"`**: because cells are
  piecewise constant, the line integral `int S e^{-tau} ds` is exactly summable
  per segment with `(1 - e^{-tau_seg})/tau_seg` weights — that is the zero-noise
  limit of the tally. The MC ("next-flight") estimator is studied against it
  (M2 variance study: measured `sigma_I` vs. observed scatter, ~1/sqrt(N_hist)).

## 4. Solver B — backward (adjoint) MC (`rt3d/mc/backward.py`)

Entry point `backward_detector_mc(grid, dg, kappa_e, kappa_s, S_total, tau_b, detector, ...) -> I_sensor, sigma`.

- **Detector** spec: `detector = {"point"/"plane": ..., "omega0": ..., "cone": ..., "n_seeds": ...}` —
  e.g. a top-of-domain pixel viewing along `Omega_i` within a cone; seeds are
  launched **backwards** from the detector: position uniform over the detector
  aperture, direction uniform within the cone (cone weight `* Delta-omega` in the
  estimator so the output is radiance, not flux), weight 1.
- Adjoint flight: identical tracker run along `-Omega`; at each collision sample
  the (adjoint) scatter direction from the HG CDF (reciprocity); the estimator at
  collision point `r` scores
  `w * [ (1-omega) B + omega J + kappa_s Phi(Omega -> Omega_sun) F0 e^{-tau_b(r)} ] / kappa_max`
  using the Stage-1 `J` and `tau_b` from `beam_optical_depth_octree` (beam coupling
  without beam particles in the adjoint run).
- **Consistency check**: seeds on the top face over the full lower hemisphere and
  icosahedral cone bins must reproduce Solver A Stage 2's top-face radiance
  (all-sky image) within 2 sigma.
- **Efficiency demonstration**: narrow sensor cone — backward MC variance vs.
  forward MC variance at equal cost (forward needs per-pixel tallies over all
  ordinates/histories; backward concentrates on the detector). This is the
  headline argument for Solver B and gets a dedicated figure.
- Optional extension (out of scope for M0-M4, note only): a backward run whose
  detector is "everywhere" (adjoint flux weighting `4pi`) would give the
  heating-rate response field — not implemented.

## 5. Estimators and error accounting (`rt3d/mc/estimators.py`)

- **History batching**: histories grouped into `n_batches` (default 100); per
  batch tallies -> mean, sample std, 2-sigma band. Every MC quantity in outputs
  carries its `sigma`.
- Tallies: `J` per cell (Stage 1); `I[(cell, ordinate)]` (Stage 2); sensor
  radiance per (pixel, cone bin) (Solver B); heating-rate tally
  `4 pi kappa_a (B - J)` for the energy anchor.
- Outputs follow the existing traceability rules: `n_refine`, `n_histories`,
  `seed`, `mode`, `stage1`, grid/octree spec recorded in every output file
  (xarray / h5py, same convention as the SC runs).

## 6. Package layout

```
rt3d/mc/
  __init__.py
  tracking.py     Woodcock core: flight + null-collision event decision,
                  birth-point sampler; unigrid + octree DDA kernels
                  (numba JIT + NumPy fallback, rt3d/jit.py pattern)
  sources.py      thermal birth sampling, HG inverse-CDF, beam seeding,
                  fixed-source (lambda variant) birth sampling
  estimators.py   exponential-path J tally, next-flight I tally,
                  deterministic segment reference tally, history batching
  forward.py      j_mc(stage1="explicit"|"lambda"), radiance_from_source(estimator=...)
  backward.py     backward_detector_mc + detector spec, all-sky consistency mode
scripts/
  mc_phase0_check.py      anchors: transmittance, HG CDF vs hg_matrix, cell-vs-global mode
  mc_forward_slab.py      stage 1 both variants vs phase2 slab reference
  mc_forward_cloud.py     stage 1+2 on the gaussian cloud vs lambda_iteration
  mc_backward.py          sensor/all-sky backward runs, efficiency figure
  mc_compare.py           SC vs MC(A-stage2) vs MC(B) differencing report
tests/
  test_mc_tracking.py     flight distribution, null mode equivalence, transmittance anchor
  test_mc_sources.py      HG CDF vs hg_matrix moments, birth sampling vs analytic rates
  test_mc_forward.py      slab J, energy closure, stage-1 variant agreement
  test_mc_backward.py     top-face consistency vs stage 2, cone-bin efficiency trend
```

No new dependencies (numba, numpy, pytest already in `pyarts`).

## 7. Phases and validation

| # | Deliverable | Validation |
|---|---|---|
| M0 | tracker (both grids, both `kappa_max` modes), event decision, HG CDF | `kappa_s=0, B=0`: MC beam transmittance = `exp(-tau)` per ray (anchor 3, 2 sigma); HG samples reproduce `hg_matrix` moments to quadrature accuracy (anchor 2 companion); cell-mode and global-mode (nulls) agree within combined 2 sigma; periodic wrap + ground entry identical to `march_lattice`/`march_octree` segment counts |
| M1 | Stage 1 `J` both variants | 1D slab `tau=5`, `g=0`, `omega` in {0.5, 0.9}: `J(z)` vs phase2 Chandrasekhar/two-stream reference within 2 sigma; explicit vs lambda variant agree within 2 sigma; heating rate `4 pi kappa_a (B - J)` vs `div F` of the SC reference (anchor 4) |
| M2 | Stage 2 radiance on the gaussian cloud | `I(cell, ordinate)` vs converged `lambda_iteration` within 2 sigma per cell (aggregate); measured error bars vs observed deviations (~1/sqrt(N)); deterministic-reference estimator agrees with SC sweep structurally |
| M3 | Backward MC | top-face all-sky vs Solver A Stage 2 within 2 sigma per cone bin; beam coupling check: `kappa_a = 0` single-scatter regime vs analytic single-scatter beam source |
| M4 | performance + variance reduction | numba fast path (jit.py pattern, numpy fallback tested); implicit capture + Russian roulette; wall-time MC vs SC table (the MC family is expected to win on Stage-2/backward-style queries, SC on J — reported honestly) |
| M5 | integration pass | `mc_compare.py` report in `plot/`, energy-balance closure over the full MC chain (A1 -> A2 -> B), decisions logged in plan/plan.md |

Every phase uses `direction_grid(n_refine)` with the development grid `n_refine=1`
(42 ordinates) for tallies; no code may assume `N_dir`. Higher levels only in a
final tally-convergence note (binning noise, not transport noise).

## 8. Decisions log entries to add to plan/plan.md when implementation starts

| Item | Decision |
|---|---|
| MC family | two solvers: A forward two-stage (J then radiance from fixed source), B backward adjoint from a detector; both Woodcock with null collisions; SC results remain the reference |
| Stage 1 | two selectable variants: explicit-scattering MC (no outer iteration) and MC lambda iteration (fixed-source transport, DOIT twin); Ng on the lambda variant opt-in, requires common random numbers, experimental |
| Null collisions | tracker always Woodcock `kappa_max >= kappa_e`; default `kappa_max = kappa_e(cell)` (zero nulls on piecewise-constant cells), global-`kappa_max` mode with real null events implemented and tested (vectorized NumPy batch + birth-point sampling); next-flight estimator uses `/kappa_max` so both modes are unbiased |
| Beam in MC | beam as a particle stream (seeded top, splits to diffuse with prob omega at collisions), unified with the thermal/beam/diffuse tracker; analytic `beam_scattering_source`/`tau_b` kept as cross-checks and for the backward coupling term |
| Stage 2 rays | reuse the SC leaf-center-per-ordinate march set so MC and SC radiance are cell-and-ordinate differencable |
| Backward MC | HG reciprocity used; detector-seeded; scores the Stage-1 full source `(1-omega)B + omega J + kappa_s Phi(->Omega_sun) F0 e^{-tau_b}` along adjoint collisions; validated against Stage 2 on the top face |
| Delta-M in MC | not supported / not needed — HG is sampled exactly, quadrature limits do not apply to MC transport |
| Grids | unigrid and octree DDA kernels from the start (M0), both numba + NumPy fallback |
| Error accounting | history batching, every MC output carries 2-sigma; `seed`, `n_histories`, `n_refine`, tracker mode recorded |
