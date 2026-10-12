# Plan — 3D all-sky monochromatic short-characteristics radiative transfer solver

Status: phases 0-6 done (rt3d/directions.py, geometry.py, characteristics.py,
transport.py, source.py, solver.py, octree.py, accelerator.py, diagnostics.py,
cases/;
scripts/phase0_check.py, phase1_beer.py, phase2_slab.py, phase3_octree.py,
phase4_ng.py, phase5_ground.py, phase6_directions.py pass;
tests/ green (80) — pytest 9.1.1 installed into pyarts). Direction grid for all
development: **icosahedral refinement 1 = 42 ordinates**, configurable
via `n_refine` (see section 2).
Environment: mamba env `pyarts` (`C:\Users\fredd\miniforge3\envs\pyarts\python.exe`) —
Python 3.14.6, numpy 2.5.2, scipy 1.18.0, matplotlib 3.11.1, xarray, h5py, pyarts3 3.0.0.dev10.
No numba, no pyvista.

## 1. Physics / formulation

Monochromatic, LTE, no polarization, no refraction:

    Omega . grad I_nu + kappa_e I_nu
        = kappa_s * int Phi_HG(Omega' -> Omega) I_nu dOmega' + kappa_a B_nu(T)

with `kappa_a = kappa_e - kappa_s`, `omega = kappa_s / kappa_e`, and

    Phi_HG = (1 - g^2) / (4 pi (1 + g^2 - 2 g cos(Theta))^(3/2))

Inputs: extinction coefficient `kappa_e`, scattering coefficient `kappa_s`,
temperature `T`, asymmetry parameter `g`.

### Solar beam split

The direct beam is a delta in direction, so it cannot live on the icosahedral
quadrature. Split `I = I_diff + I_beam`:

- `I_beam = F0 * exp(-tau_b(r))` from one dedicated sweep along `Omega_sun`
  (plus ground reflection if enabled).
- Its first-scattering contribution
  `kappa_s * Phi_HG(Omega_sun -> Omega_i) * F0 * exp(-tau_b)`
  enters the source function as a fixed term, never iterated on the quadrature.

### Boundaries

- **Top:** `I_diff = 0` (optionally a background field).
- **Bottom (flat ground at z = z0):** Lambertian + emissive,
  `I_up = A_g/pi * F_down + eps_g * B(T_g)`, computed by sweeping downward ordinates first.
  Beam reflected Lambertian as well. Surface is flat for now; keep a hook for a
  variable surface later.
- **Sides:** periodic in `x` and `y` by wrapping ray exit -> entry.

## 2. Direction grid

Reuse `scripts/icosphere_grid.py`, and add:

- **Solid-angle weights** `w_i` from spherical-triangle areas (L'Huilier) distributed to
  vertices. Icosphere vertices are *not* equal-area, so `4*pi/N` is only an approximation.
- **Precomputed HG matrix** `Phi[i,j] = Phi_HG(Omega_j -> Omega_i)`, `N x N`. The whole
  scattering integral becomes one BLAS matmul over all cells:
  `(n_cells x N) @ (N x N)`. If `g` varies spatially, bucket cells by `g` and do one
  matmul per bucket.
- A **vectorized rewrite** of `refine_icosahedric_grid` — the current dedup loop is
  O(N^2) pure Python. It is fine for `n_refine = 1` (42 ordinates, the development grid)
  but stalls at 162/642, so the rewrite is needed only for the phase 6 convergence study.
- **Rotate the grid so no ordinate has exactly `Omega_z = 0`** — those form cyclic
  characteristics under `x,y` periodicity.

### Grid size: 42 ordinates is the development grid

The icosahedral grid is parameterized by refinement level `n_refine`, with
`N_dir = 10 * 4**n_refine + 2`:

| `n_refine` | `N_dir` | role |
|---|---|---|
| 0 | 12 | smoke tests only |
| **1** | **42** | **the development grid — every phase of this plan uses it** |
| 2 | 162 | refinement study / production |
| 3 | 642 | refinement study / production |

**All development, all test cases, all benchmarks and all validation runs in phases 0-6 use
`n_refine = 1` (42 ordinates).** Higher levels exist only as a posterior convergence check
(phase 6), never as the working configuration. Rationale: the sweep cost scales linearly in
`N_dir`, so 42 ordinates keeps every iteration loop fast enough to iterate on the physics,
and a 42-ordinate `S_N` set is adequate for the 1D slab benchmarks.

`n_refine` must be a **first-class option everywhere**, never a hard-coded constant:

- `rt3d.directions.icosahedric_grid(n_refine=1)` — the only place the default appears.
- `rt3d.solver.SolverConfig(n_refine=1, ...)` — stored on the config, logged at run start,
  and recorded in every output file so results are always traceable to their direction grid.
- Every case in `rt3d/cases/` and every runner in `scripts/` accepts `--n-refine` /
  a module-level `N_REFINE` and forwards it unchanged.
- Nothing in the code may assume a specific `N_dir`: no fixed-size arrays, no hard-coded
  42/162, no level-specific index arithmetic. The only `N_dir`-dependent precomputations
  are the weights `w_i` and the `N_dir x N_dir` HG matrix, both rebuilt from `n_refine`.

Known limitation to record in phase 6: 42 ordinates under-resolves strongly forward-peaked
scattering (`g > ~0.5`) and the solar beam's scattering lobe. The convergence study
(42 -> 162 -> 642) quantifies this; it is accepted for development, not hidden.

## 3. Octree

Flat struct-of-arrays octree in numpy (cubic cells, 2:1 balance, max-depth cap).
Per leaf: center, half-size, level, and cell-averaged `kappa_e, kappa_s, g, T`.
Point location is O(1) via a per-level lattice -> leaf-index table.

**Split criterion driven by extinction (input).** Refine a cell when *either*

- `kappa_e * h > tau_max` (optical depth across the cell — the accuracy-relevant
  measure for short characteristics), or
- `(max(kappa_e) - min(kappa_e)) / mean(kappa_e) > eps` (resolves gradients / cloud edges).

Fields are sampled either from an analytic callable or from the
`RegularGridInterpolator` fields already used in `scripts/class_test.py`.

## 4. Short characteristics + transport

Per ordinate `Omega`, one characteristic per leaf through its center. Build a
**predecessor chain**: for leaf `c`, march backwards from `c`'s center, record the upwind
leaf `u` and the sub-segments (cell id, length) of the straight path
`u_center -> c_center`. Explicit sub-segments are what makes the octree case correct when
`u` and `c` differ in size (Bipolar / Rybicki-Hummer approach).

Transport recursion per segment, exact for piecewise-constant sources:

    I_c = I_u * exp(-tau_tot) + sum_sub S_sub * (1 - exp(-tau_sub)) * exp(-tau_after)

Each node has exactly one parent, so the per-direction dependency graph is a forest
rooted at boundary-entry cells. Process by BFS waves from the roots, vectorized with
numpy fancy indexing. This handles the periodic wrap automatically (the unwrapped
ordering is implicit in the chain).

## 5. Iteration loop (DOIT-style) + Ng acceleration

```
S0 = (1 - omega) * B + beam_term
repeat:
  1. scattering integral  J = <I>  ->  S = (1 - omega) * B + omega * J + beam_scat   # incoming radiation
  2. transport sweep over all ordinates + top/ground/periodic BCs                    # transport
  3. moments J_new, F, heating rate
  4. residual R = J_new - S ;  dS = (1 - omega * Lambda_bar)^-1 R ;  S = S + dS      # Ng
```

`Lambda_bar` is a 1D angle-averaged operator in `z` built from horizontally averaged
`kappa_e(z), omega(z)`, assembled once per outer iteration into an `Nz x Nz` dense matrix
and solved directly (`Nz ~ 50-200`, trivial). This is the ARTS DOIT accelerator.

## 6. Package layout

```
rt3d/
  directions.py       icosphere grid (n_refine -> 12/42/162/642), area weights, HG matrix,
                      vectorized refinement
  geometry.py         cuboid, ray-box intersection, periodic wrap, ground-surface hook
  octree.py           flat SoA octree, tau/gradient refinement, O(1) point location
  characteristics.py  predecessor chains + sub-segments per ordinate
  transport.py        SC sweep, top/ground/periodic BCs, dedicated beam sweep
  source.py           scattering integral (BLAS matmul), Planck, source assembly
  accelerator.py      1D Lambda_bar assembly + Ng update
  solver.py           iteration loop, convergence monitor
  diagnostics.py      J, F, heating rate, all-sky images, plots
  cases/              slab_1d.py, gaussian_cloud.py, clear_sky.py
scripts/              runner scripts (existing two files left untouched)
tests/                pytest (needs installing pytest into pyarts)
```

## 7. Phases and validation

| # | Deliverable | Validation |
|---|---|---|
| 0 | package skeleton, directions (`n_refine=1` -> 42 ordinates) + weights + HG matrix, geometry | `sum(w_i) = 4*pi`; `int Phi dOmega = 1`; grid rebuilds correctly for `n_refine` 0/1/2/3 |
| 1 | **uniform Cartesian** SC sweep, beam + thermal, `kappa_s = 0` | Beer-Lambert analytic transmittance |
| 2 | add scattering, plain lambda iteration | 1D slab vs. Chandrasekhar / two-stream |
| 3 | octree + predecessor chains | octree vs. uniform on the `class_test.py` Gaussian cloud |
| 4 | 1D `Lambda_bar` Ng acceleration | convergence rate vs. plain LI as `omega -> 1` |
| 5 | periodic sides, ground reflection, all-sky output images | `div F` = heating rate (energy conservation) |
| 6 | direction-grid convergence study 42 -> 162 -> 642, performance pass, variable-surface hook | angular convergence of J and top-of-domain flux |

Every phase runs at `n_refine = 1` (42 ordinates) unless the row explicitly says otherwise.

## 8. Decisions log

| Item | Decision |
|---|---|
| Octree refinement | both `kappa_e * h > tau_max` **and** relative `kappa_e` gradient > eps |
| Thermal emission | `(1-omega) B + omega J`, switchable off for shortwave-only runs |
| Ng acceleration | 1D column-averaged `Lambda_bar` in z, dense `Nz x Nz` solve |
| Spatial discretization | uniform Cartesian prototype first, octree swapped in afterwards |
| Ground BC | Lambertian + emissive, Lambertian beam reflection, flat for now |
| Directions | **icosphere refinement 1 = 42 ordinates for all development**, spherical-Voronoi area weights, `n_refine` configurable everywhere (0/1/2/3 -> 12/42/162/642) |
| Predecessor chain (uniform grid) | bipolar center-to-center connection realized as one integer lattice stencil per ordinate (offset minimizing angle to Omega, s_z != 0); identical sub-segment pattern for all cells -> fully vectorized sweep. Octree (phase 3) uses the general backward-march predecessor search |
| Stencil search radius | scales with grid anisotropy: `max_offset=None` -> `~3 * max(h)/min(h)` per axis (capped at 24). A fixed radius 3 leaves large, n_refine-independent stencil errors on anisotropic lattices (hz = hx/4: ~36 deg) |
| Octree predecessor chain | per leaf, march backwards along -Omega over a limited hop budget (8 own-cell widths) and pick the traversed leaf whose center-to-center vector is best aligned with Omega (cos_min = 0.9); leaves with no aligned candidate are **rooted** and carry an exact march to the domain entry (true ray). Dependency cycles resolved by topological peeling + rooting. Sub-segments of each hop by exact DDA march (vectorized over leaves, per-ray path budget, periodic minimum-image). **Superseded for the solver** - see "Octree transport = exact characteristics" below; the chain code remains for path-accuracy checks |
| Octree gradient criterion floor | the relative-gradient split applies only where `kappa_e * h > 1e-2 * tau_max`: without the floor the vacuum tails of a Gaussian cloud (huge relative gradients, negligible optical depth) refine everywhere |
| Octree 2:1 balance | face neighbors probed on a 3x3 grid per face (transverse offsets -h/2, 0, +h/2); face-center probing alone misses small cells touching a big face and the balance loop terminates with violations |
| HG quadrature row normalization | `hg_matrix(g)` rows are rescaled to unit sum (`normalize=True` default). The analytic HG integrates to 1 (anchor 2), but the 42-ordinate icosahedral quadrature gives row sums 17-22 at g = 0.95, making the lambda-iteration scattering operator supercritical (diverges already at omega ~ 0.7). Row normalization enforces exact discrete conservation of isotropic radiation; at g = 0 rows already sum to exactly 1, so isotropic results are unchanged |
| Octree transport = exact characteristics | the octree sweep marches one ray per leaf center backwards along -Omega to the domain entry (DDA `march_octree`), precomputes the segments once per ordinate, and assembles each sweep by vectorized per-ray cumsums (`precompute_octree_march` + `sweep_direction_octree_exact`). The best-aligned center-to-center chain is **dropped from the solver**: its cos_min = 0.9 hop tolerance makes the path error scale-invariant - measured tau error ~4 on the Gaussian cloud, not shrinking with refinement (unigrid lattice stencil: transmittance wrong by up to ~5e4 for grazing ordinates, chain: ~5). Exact march costs ~1-3 s per ordinate once, sweeps stay ~0.05 s |
| Anchor 5 interpretation (phase 3) | on the Gaussian cloud at 24^3 the octree (exact characteristics) and the uniform grid (lattice stencil) differ by ~18% in J below the cloud even on an **identical piecewise-constant kappa field** - this is the uniform grid's grazing-ordinate stencil path error, not an octree error (octree sweep reproduces `exp(-tau)` at machine precision, and its eps_grad series self-converges 11% -> 7%). The two discretizations converge to the same exact solution only as h -> 0; the octree gets there much faster. Top-of-domain flux agrees to 0.01% (global energy balance) |
| Ng implementation (phase 4) | `Lambda_bar` probed by plane-averaged history pairs (x^m, y^m) = (sigma^m - sigma^(m-1), S^(m+1) - S^(m)) - swept-source and consecutive-LI-update differences; this pair identity `Lambda_bar x = y` holds **exactly even for accelerated iterates** (pairing consecutive LI *step* differences instead breaks once acceleration starts and made the iteration blow up). Minimum-norm dense Nz x Nz fit; geometric-series correction sigma^(n+1) = S^(n+1) + (I - Lambda_bar)^-1 Lambda_bar d^(n+1) applied per z bin; safeguard: plain-LI step if rho(Lambda_bar) >= 1. Measured (slab tau=5, g=0): omega 0.5: 22 -> 10 iters; 0.9: 87 -> 13; 0.99: 178 -> 14 (12.7x). On the omega <= 0.9 cloud LI is already fast (34 iters, domain mostly transparent) and Ng is neutral there |
| Ground BC implementation (phase 5) | `geometry.ground_surface(albedo, emissivity, B_g)` - monochromatic: ground emission is the input Planck value `B_g`, not `T_g` (no Planck function anywhere). Attached as `boundary_conditions(ground=...)`. Radiance `I_g = eps_g B_g + A_g (F_down + F0 |mu_sun| e^-tau_b,ground) / pi`; `F_down` sampled at bottom-layer cell/leaf centers (first-order to the face). Solver sweeps downward ordinates first, rebuilds `I_g`, then sweeps upward with per-cell `bottom_I`. Unigrid: chains with \|s_z\| > 1 root **above** the bottom layer, so the ground radiance (a column field) is tiled over all z-layers before indexing roots. Octree: `precompute_octree_march` stores `entry_cell` = bottom-face leaf containing each upward ray's exact ground entry point (`oct.locate`), so `I_g` is evaluated at the true entry column |
| march_lattice periodic-wrap stall (phase 5 bugfix) | a ray exiting within float noise of a periodic face re-exited immediately with a zero-length step and stalled until max_steps **without ever recording an exit** (exit face stayed at the zero default -> root got `side_I` instead of top/bottom inflow). Fixed by snapping the wrapped point to the opposite face with a 1e-9 tolerance (both directions). Affected all phases but only fired on unlucky float coincidences; phase 1-4 numbers unchanged |
| Beam first-scatter HG normalization (phase 5) | `beam_scattering_source` rescales the phase vector so `sum_i w_i Phi(Omega_sun -> Omega_i) = 1` (same convention as `hg_matrix(normalize=True)`): beam first-scatter power now passes through the ordinate set exactly instead of carrying the raw HG quadrature error at large g |
| Anchor 4 measurement (phase 5) | pointwise `div F` (finite differences on cell-center fluxes) vs `4 pi kappa_a (B - J)` on the converged 24^3 cloud+ground+beam run: rms 1.6%, max 14% (cloud-edge FD truncation). Global closures: diffuse 3.0%, beam 0.44%, total 3.2% - floor set by the 2nd-order face extrapolation of F_z and quadrature noise on 24^3. `diagnostics.energy_balance` uses net F_z (both hemispheres) linearly extrapolated to the faces |
| Angular convergence study (phase 6) | `scripts/phase6_directions.py`, 42 -> 162 -> 642 ordinates. **Cloud 24^3 + ground + beam** (g <= 0.952 only in the small cloud core): x,y-mean J profile 0.47% then 0.36% (ratio 1.32), cell-level max\|dJ\| 13.2% -> 4.7%, top-of-domain flux 0.77% -> 0.03% - angular error is small where the field is smooth/diffuse. **Uniform slab tau=10, omega=0.9, g=0.952** (HG lobe everywhere, half-width ~9 deg vs 42-ordinate spacing ~25 deg): consecutive differences are **non-monotone** (1.86% then 4.08%) because the 42-ordinate solution benefits from error cancellation (`hg_matrix` row normalization redistributes the unresolved lobe isotropically); the sequence still converges to the 642 solution - differences vs the finest grid shrink monotonically: 42 vs 642 = 5.9% in J / 5.8% in top flux, 162 vs 642 = 4.1% / 3.6%. Conclusion: 42 ordinates carry ~6% J error at g ~ 0.95 (plan section 2 limitation, quantified); even 642 is not fully converged at g = 0.952. `tests/test_directions.py::test_angular_convergence_slab` (g=0.6) asserts monotone shrinking of consecutive differences |
| Performance pass (phase 6) | unigrid transport sweep scales **linearly in N_dir**: 24^3 sweep pass 0.09 s / 0.36 s / 1.34 s at 42/162/642 ordinates (ratio 15.3 vs N_dir ratio 15.3), ~2.1-2.3 ms per ordinate per 24^3 sweep; chain build also linear (0.3 s / 1.1 s / 4.0 s, one-time). Full Ng cloud run at 642 ordinates: 40 iters, 84 s. **numba not needed** at these sizes (section 9 fallback stays unused) |
| Variable-surface hook (phase 6) | `geometry.ground_surface(..., height=...)`: scalar (flat, default) or callable f(x, y) -> z; `is_variable()` reports which. Transport still implements only the flat ground at the domain bottom - `lambda_iteration` raises `NotImplementedError` for a variable surface (hook + guard, plan section 1 "variable surface later") |
| Delta-M scaling (Wiscombe) | Option `delta_M` on `lambda_iteration` (default 0 = off; `DELTA_M` config cell in `scripts/phase2_slab.py` / `phase4_ng.py` / `phase6_directions.py`; plan/delta-m-plan.md). Split defined on the **row-normalized** `hg_matrix`: `direction_grid.hg_delta_matrices(g, M)` gives `Phi_w == Phi_w_delta + sum_n (2n+1) g^n P_n(Omega_i . Omega_j) w_j / 4pi` exactly (Legendre recurrence), so delta-M changes only iteration + transport scaling, **not the converged discrete solution**. **Only the scattering part of the extinction is scaled**: `kappa_e_delta = kappa_e - kappa_s g^M` (`source.delta_extinction`), used for **all diffuse** sweeps. Full-extinction scaling `kappa_e (1 - g^M)` tried and **rejected**: (a) the explicit `-g^M I` term breaks anchor 3 (exact Beer-Lambert at kappa_s = 0) at O(tau_cell) - measured 7.6e-2 at tau_cell = 0.1; (b) LI isotropic-mode gain `(omega - g^M)/(1 - g^M)` makes plain lambda iteration **diverge for omega < g^M** (measured at omega = 0.5, g = 0.9). With scattering-only scaling anchor 3 is exact and the explicit-term gain `omega g^M (1-g)/(1 - omega g^M) < 1` for all omega < 1. Source: `scattering_source(..., delta_M=M)` -> `S = [(1-omega)B + omega*(I @ Phi_w_delta.T + sum_n I @ moments[n].T) - omega g^M I] / (1 - omega g^M)` (per-cell). The explicit `-omega g^M I` term carries an O(tau_cell) short-characteristics quadrature error: delta-on vs delta-off converged J differs 3.9e-2 (nz=8) -> 9.2e-3 (nz=16) -> 2.1e-3 (nz=32) on the tau=5 slab, shrinking ~linearly with h (same solution, not an inconsistency). Beam keeps **physical** attenuation: `tau_b` and ground beam irradiance use physical `kappa_e` (beam not on the quadrature, moment subtraction applies to the diffuse field only); `beam_scattering_source` albedo factor becomes `omega / (1 - omega g^M)`. Octree refinement stays on physical `kappa_e`. Result dict records `delta_M` and `kappa_e_delta`. Measured (tau=5 slab, nz=16, omega=0.9, tol=1e-8, 42 ordinates): plain LI g=0.5: 75 -> 45, g=0.8: 67 -> 22, g=0.9: 69 -> 14, g=0.95: 65 -> 11 (M=2 at g=0.9: 77 -> 23); Ng g=0.9: 90 -> 15, g=0.95: 82 -> 11, but delta-M **slows Ng at small g** (g=0.5: 31 -> 59) -> recommend `delta_M` only for g >~ 0.5, M=1. Acceleration test (nz=8, tol=1e-6, g=0.95): 62 -> 12. `tests/test_delta_m.py` (30 tests: operator identities, RHS exactness `(1 - omega g^M) S_delta + omega g^M I == S_undeltaed`, solver equivalence, acceleration, anchors 3 + 4 with delta on) |
| Unigrid SC transport (`transport="sc"`) | Second, independent unigrid algorithm alongside the lattice-stencil chains (plan/plan-sc-unigrid.md, DOIT-style): `rt3d/short_characteristics.py` (`sc_stencil` / `build_sc_chain` / `sweep_direction_sc` / `sweep_all_sc`), option `transport="chain" | "sc"` on `lambda_iteration` (default `"chain"`, existing behaviour unchanged; octree + `"sc"` raises). Per ordinate: entry axis a = argmax |Omega_a|/h_a, single upwind layer, bilinear 4-point stencil (offsets + weights identical for every cell, 4-ring bound holds on anisotropic grids), layer sweep along a vectorized over the layer plane; periodic x/y entry axes solved by wrap-around Gauss-Seidel passes (wrap_tol on the wrap feed). **Two-segment recursion** (amendment, plan-sc-unigrid.md section 6): each node value is first propagated along +Omega by t_f through the node cell (gap segment) before the bilinear combination - without it the nodes all sit strictly upwind of the face point, the interpolation is an extrapolation, and the transmittance error is O(1) **scale-invariant** (tau halved for z-entry, error ratio 1.000 under refinement 8->16->32). With the gap: z-entry ordinates exact vs exp(-tau) (anchor 3 holds for them); oblique ordinates keep a bounded, resolution-independent angular stencil error (effective direction set by the bilinear weights). Pass cap `max(4 n_a, 64)` not n_a (pure-shift exactness argument fails for the bilinear stencil; measured ring gain ~0.30 at n_x=24 with ~50 passes to 1e-12 in vacuum; kappa_e > 0 in the ring -> 1-3 passes). Ground/beam/Ng/delta-M reused unchanged (bottom_I tiled like the chain sweep). Perf at 24^3 / 42 ordinates, kappa_e > 0: SC build 0.13 s vs chain 0.15 s, SC sweep 0.086 s vs chain 0.059 s, SC precompute ~25 MB vs ~47 MB. Tests: `tests/test_short_characteristics.py` (15: stencil, naive scalar reference, exact slab recursion, transmittance, ring residual + dense solve, cloud SC-vs-chain convergence, solver integration); docs: sc-transport.md section 12 |

## 9. Known risk

The sweep is `N_dir x N_leaf` steps. At the development grid this is `42` ordinates x
`1e5` leaves = `4.2e6` steps per sweep, which is still slow in pure Python but roughly 4x
cheaper than the 162-ordinate grid. First mitigation is BFS-wave numpy vectorization;
`numba` (`mamba install -n pyarts numba`) is the fallback in phase 6 **(not needed: the
phase 6 performance pass measured linear scaling in `N_dir`, ~2.1 ms per ordinate per
24^3 sweep)**.

A second, smaller risk: 42 ordinates is a coarse `S_N` set. It is expected to be adequate
for the phase 1-4 benchmarks, but it will under-resolve `g >~ 0.5` and the beam scattering
lobe. Phase 6 quantifies this with the 42 -> 162 -> 642 convergence study rather than
pretending the development grid is production-accurate.
