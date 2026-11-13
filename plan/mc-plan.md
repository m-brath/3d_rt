# Plan — Monte Carlo solvers (forward two-stage + backward adjoint), with null collisions

Status: **M0 implemented** (2026-10-24): `rt3d/mc/` package (`tracking.py`
Woodcock core + unigrid/octree DDA walkers, `sources.py` HG inverse-CDF),
`scripts/mc_phase0_check.py`, `tests/test_mc_tracking.py` (9) +
`tests/test_mc_sources.py` (12). All M0 validations green: transmittance
anchor |z| <= 1.3 sigma on both grids and both modes, cell-vs-global
agreement within combined sigma, walker segments bit-identical to
`march_lattice`/`march_octree`. Numba fast path arrives with M4 per the
rt3d/jit.py pattern. M1-M5 not started. Adds a second solver family next
to the short-characteristics solver (`rt3d/`, phases 0-6 done). This plan
reuses the octree (`rt3d/octree.py`), unigrid (`rt3d/geometry.py`),
direction grid (`rt3d/directions.py`), ground BC
(`rt3d/transport.ground_surface`), and the SC results of `lambda_iteration`
as cross-validation reference.

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
