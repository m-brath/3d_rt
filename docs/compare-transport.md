# Transport-scheme comparison — chain vs. SC vs. octree, with and without delta-M, ranked against a fine reference

Comparison of the three transport configurations of the `rt3d` solver on the
MANUAL §11 worked example, each run with and without delta-M (Wiscombe)
scaling, **plus a fine-octree reference solution and grid-refinement runs
that rank every configuration by accuracy** (plan:
`plan/compare-transport-plan.md` + `plan/transport-ranking-plan.md`, incl.
its addendum §7); all figures and CSVs live in `plot/compare_transport/`.
The two runs on the fine reference octree use **642 ordinates** (icosahedral
refinement 3) instead of the 42-ordinate development grid, so the reference
is finer in *angle* as well as in space and the ranking errors of the
42-ordinate runs include their angular error (comparisons *among* the
42-ordinate runs still cancel it). Measured 2026-10-15 on macOS/arm64
(Apple M4 Pro, 64 GB) with the `pyarts` interpreter; physics numbers are
machine-independent, wall times are not (the previous version of this report
was measured on Windows — wall times differ by ~2–4× between the two
machines; Ng-accelerated iteration counts can also shift by a few iterations
with BLAS rounding, e.g. SC δ-off: 44 → 51). The wall times here predate the
numba fast paths of `rt3d/jit.py` (landed after this run started); reruns
with them will be faster.

Reproduce with:

```sh
~/miniforge3/envs/pyarts/bin/python scripts/compare_transport.py
```

(The 642-ordinate reference runs need ~22 GB RAM for their octree march
tables — ~34 MB per ordinate on the 50 380-leaf grid. 3D HTML scenes need
`trame-pyvista`; installed in the macOS `pyarts` env on 2026-10-13, so all
scenes and `vol_j.png` are current. Per-run solutions are cached in
`plot/compare_transport/_solve_cache/` (gitignored) — reruns skip the solves;
the cache records the ordinate count and stale entries are re-solved.)

---

## 1. Purpose and run matrix

The solver currently offers three transport algorithms and an optional
delta-M split of the scattering operator. This study answers four questions
on one physically non-trivial case (strongly forward-scattering cloud,
solar beam, reflecting ground):

1. **chain vs. SC** — how large is the pure *transport-algorithm* error when
   both run on *identical cells*?
2. **unigrid vs. octree** — the grid-refinement error (validation anchor 5).
3. **delta-M on/off** — does it accelerate convergence, and does the
   converged solution stay the same (the equivalence claim of
   `plan/delta-m-plan.md`)?
4. **which run is best?** — pairwise differences alone cannot answer this
   (a gap between two approximations says nothing about which is closer to
   the exact solution), so the study adds a fine reference solution and
   ranks every run by its error against it (section 5).

Six comparison runs plus five ranking runs (N_dir column: icosahedral
refinement 1 = 42 ordinates for all runs except the two on the fine
reference octree, which use refinement 3 = 642 ordinates):

| run key | grid | transport | delta-M | N_dir | role |
|---|---|---|---|---|---|
| `uni_chain_dM0` | unigrid 24³ | lattice-stencil chains | off | 42 | comparison |
| `uni_chain_dM1` | unigrid 24³ | lattice-stencil chains | M = 1 | 42 | comparison |
| `uni_sc_dM0`    | unigrid 24³ | single-cell short characteristics | off | 42 | comparison |
| `uni_sc_dM1`    | unigrid 24³ | single-cell short characteristics | M = 1 | 42 | comparison |
| `octree_dM0`    | octree (25 264 leaves) | exact characteristic marches | off | 42 | comparison |
| `octree_dM1`    | octree (25 264 leaves) | exact characteristic marches | M = 1 | 42 | comparison |
| `uni_chain48_dM1` | unigrid 48³ (h = 0.208 m) | chains | M = 1 | 42 | chain refinement |
| `uni_sc48_dM1`    | unigrid 48³ | short characteristics | M = 1 | 42 | SC refinement |
| `octree_d4_dM1`   | octree τ_max = 1.0, depth 4 (34 476 leaves) | exact marches | M = 1 | 42 | octree refinement |
| `octree_ref_dM1`  | octree τ_max = 0.5, depth 5 (50 380 leaves) | exact marches | M = 1 | **642** | fine-grid δ-M check |
| `octree_ref_dM0`  | same octree as `octree_ref_dM1` | exact marches | off | **642** | **the reference** |

---

## 2. The setups

### 2.1 Common physical case (MANUAL §11, unchanged)

Gaussian cloud + Lambertian ground + solar beam on the domain
x, y ∈ [−5, 5], z ∈ [0, 10] m, periodic in x and y, space at the top,
ground at the bottom. Units: lengths m, κ in m⁻¹, radiometric quantities
normalised to the incident beam flux F₀ = 1.

- **Extinction** κ_e = C·gauss(σ = 1 m, μ = (0, 0, 4) m), C scaled so the
  maximum vertical column optical depth is τ = 10 (peak κ_e ≈ 3.9 m⁻¹,
  mean free path ≈ 0.26 m).
- **Scattering** κ_s = ω_local·κ_e with ω_local = gauss/max(gauss), capped
  at ω ≤ 0.9 (`omega_cap=0.9`) so plain lambda iteration stays tractable.
- **Asymmetry** g_local = gauss/(1.05·max) → peak g ≈ 0.952 in the cloud
  core (HG phase function) — the showcase regime for delta-M.
- **Thermal emission** B = 0 (pure scattering/beam problem).
- **Solar beam**: direct collimated beam, 30° from zenith, propagating in
  the x–z plane: ω_sun = [sin 30°, 0, −cos 30°], F₀ = 1.
- **Boundary conditions**: isotropic background I = 1 on the top face;
  Lambertian ground, albedo 0.3, emissivity 0 (no thermal emission).

### 2.2 Direction grids and iteration scheme

- Icosahedral direction grids, solid-angle weights from spherical-triangle
  areas: refinement 1 → **42 ordinates** (development grid, all comparison
  and refinement runs, Σw − 4π = −7.1e−15) and refinement 3 → **642
  ordinates** for the two runs on the fine reference octree
  (Σw − 4π = +7.1e−15).
- Lambda iteration with **1D Ng acceleration**, convergence on the relative
  source change, tol = 1e−6, cap 60 iterations.
- The 42-ordinate *angular* error cancels in all comparisons *among* the
  42-ordinate runs (the six base pairs, the unigrid refinement pairs), but
  **not** against the 642-ordinate reference: the ranking errors of the
  42-ordinate runs include their angular error, and the reference's own
  angular error is ~15× better resolved (section 5.2).

### 2.3 Grids

- **Unigrid 24³**: h = 0.4167 m, 13 824 cells; **unigrid 48³**: h = 0.2083 m,
  110 592 cells.
- **Octree** base cell h₀ = 1.25 m, refinement driven by the *physical* κ_e,
  2:1 balanced:
  - depth 3 (τ_max = 2.0): **25 264 leaves** (L0: 316, L1: 788, L2: 3 680,
    L3: 20 480), cell size 1.25 m in the vacuum tails down to 0.156 m;
  - depth 4 (τ_max = 1.0): **34 476 leaves** (finest cells still 0.156 m —
    the extra refinement acts on the intermediate levels);
  - depth 5 (τ_max = 0.5): **50 380 leaves**, finest cells **0.078 m**
    (τ_cell ≤ 0.5 in the cloud core). κ_e-driven refinement *saturates*
    here: τ_max < 0.5 or depth > 5 adds no leaves, so this is the finest
    discretization the solver builds on this case — hence the reference.

### 2.4 Transport algorithms

- **`chain` (unigrid)** — lattice-stencil chain sweeps
  (`rt3d/characteristics.py`, DOIT-style): per ordinate one integer stencil
  s, characteristics run center-to-center across the whole grid in
  multi-cell chains. Exact for piecewise-constant sources along the chain;
  carries the known *grazing-ordinate stencil error* on coarse grids
  (MANUAL §6, anchor 5).
- **`sc` (unigrid)** — single-cell short characteristics
  (`rt3d/short_characteristics.py`, `plan/plan-sc-unigrid.md`):
  gap segment + target segment of equal length, bilinear 4-point face
  interpolation, wrap-around passes on periodic entry axes. Local and
  gather-only; more diffusive on coarse grids (path error ~ linear in cell
  size).
- **Octree** — exact characteristic marches
  (`transport.precompute_octree_march` + `sweep_direction_octree_exact`):
  one backward ray march per leaf per ordinate along the *true* ordinate
  direction, no lattice-stencil approximation. Short characteristics are
  *not available* on the octree (the solver raises `ValueError` for
  `transport="sc"` there).

### 2.5 delta-M scaling (`delta_M=1`, delta-Eddington)

Wiscombe split of the HG phase function, defined on the *row-normalized*
discrete operator (MANUAL §9, `plan/delta-m-plan.md`):

```
Phi_delta = Phi_HG - (1/4pi) * sum_{n<1} (2n+1) g^n P_n(Omega.Omega')
kappa_e_delta = kappa_e - kappa_s * g^M        (diffuse transport only)
```

with the explicit `-omega·g^M·I/(1 - omega·g^M)` term in the source. Only
the scattering part of the extinction is scaled; the direct beam keeps its
physical attenuation, and octree refinement stays driven by the physical
κ_e. delta-M changes the *iteration* (fewer iterations at g ≈ 0.95), not
the converged discrete solution — the on/off comparison below verifies
this.

### 2.6 The reference solution

`octree_ref_dM0`: the depth-5 octree (h_min = 0.078 m ≈ 3.3 cells per mean
free path in the cloud core) at **642 ordinates**, exact marches, **delta-M
off** so it carries no O(τ_cell) quadrature gap of the explicit −ω g^M I
term. It is the finest space-angle discretization available on this case;
its own remaining error is bounded by the depth-3 → depth-4 → depth-5
*space* convergence sequence and by the 42 → 642 *angle* step (section 5.2:
the 42-ordinate reference scalars differ from the 642-ordinate ones by
~1.6% in ⟨J⟩ and ~1% in the TOA flux — the reference's remaining angular
error is correspondingly smaller), and by the fine-grid δ-on↔δ-off check
(0.19% max, section 4.1).

---

## 3. Diagnostics

All cell fields are sampled onto one **common 64³ lattice** at the baseline
octree cell size (h = 0.15625 m) — exact pointwise sampling for
piecewise-constant fields, the finer reference cells are sampled, not
averaged — so every pair of runs is directly pointwise comparable.

- **Mean intensity** J = (1/4π)∫I dΩ.
- **Angular variance** σ²(r) = (1/4π)∫(I − J)² dΩ
  (`rt3d.diagnostics.angular_variance`) — zero for an isotropic field,
  measures the angular anisotropy of the radiation field.
- **Upward radiance at TOA**: the upward flux map
  F⁺(x, y) = ∫_{μ>0} I μ dΩ at the top layer, and the domain-mean upward
  radiance per upward ordinate (21 of the 42, and 321 of the 642 ordinates
  have μ > 0; per-ordinate spectra are written per direction grid to
  `toa_mean_radiance.csv` and `toa_mean_radiance_642.csv`).

Six comparison pairs (B − A): SC↔chain per δ, δ-on↔δ-off per method,
octree↔unigrid-chain per δ. Plus the **accuracy ranking**: every run
against `octree_ref_dM0` (`ranking.csv`).

---

## 4. Results

### 4.1 Convergence and efficiency

| run | transport | δ-M | N_dir | n_iter | converged | wall [s] | ⟨J⟩ | mean F_z @TOA |
|---|---|---|---|---|---|---|---|---|
| `uni_chain_dM0` | chain | off | 42 | 32 | ✓ | 1.6 | 0.5188 | −2.5215 |
| `uni_chain_dM1` | chain | M=1 | 42 | **12** | ✓ | 0.7 | 0.5188 | −2.5214 |
| `uni_sc_dM0`    | sc    | off | 42 | 51 | ✓ | 16.6 | 0.4632 | −2.5935 |
| `uni_sc_dM1`    | sc    | M=1 | 42 | **12** | ✓ | 4.1 | 0.4633 | −2.5933 |
| `octree_dM0`    | octree march | off | 42 | 30 | ✓ | 37.9 | 0.4014 | −2.5312 |
| `octree_dM1`    | octree march | M=1 | 42 | **12** | ✓ | 20.7 | 0.4016 | −2.5308 |
| `uni_chain48_dM1` | chain 48³ | M=1 | 42 | 13 | ✓ | 5.4 | 0.5194 | −2.5223 |
| `uni_sc48_dM1`    | sc 48³    | M=1 | 42 | 12 | ✓ | 15.9 | 0.4640 | −2.5927 |
| `octree_d4_dM1`   | octree march d4 | M=1 | 42 | 12 | ✓ | 33.0 | 0.4114 | −2.5319 |
| `octree_ref_dM1`  | octree march d5 | M=1 | 642 | 34 | ✓ | 3393.7 | 0.3778 | −2.5073 |
| `octree_ref_dM0`  | octree march d5 | off | 642 | 23 | ✓ | 2367.4 | 0.3778 | −2.5074 |

- **delta-M is a large accelerator at 42 ordinates**: 32 → 12 (chain),
  51 → 12 (SC), 30 → 12 (octree d3) iterations; wall time drops by ~2×
  everywhere and by ~4× for SC.
- **but not at 642 ordinates**: on the reference grid δ-M *increases* the
  iteration count (23 off → 34 on) and hence the wall time (2367 → 3394 s).
  The acceleration benefit is angular-grid dependent — at 42 ordinates the
  row-normalized HG operator at g ≈ 0.95 is extremely forward-peaked and
  removing its g¹ moment pays; at 642 ordinates it no longer does.
- The unigrid TOA mean F_z reproduces the MANUAL §11 numbers exactly
  (−2.5215 unigrid, −2.5312 octree d3).
- **the 42 → 642 ordinate step moves the reference itself**: ⟨J⟩
  0.3840 → 0.3778 (−1.6%) and mean F_z @TOA −2.5320 → −2.5074 (+1.0%) —
  the 42-ordinate angular error inflates in-domain J and makes the net TOA
  flux too negative even on the well-resolved octree.
- **δ-M equivalence at the finest grid**: `octree_ref_dM1` vs
  `octree_ref_dM0` differ by 0.19% max / 0.005% mean in J — half the
  depth-3 gap (1.3%), confirming the O(τ_cell) scaling of the explicit
  term and that the reference is δ-M-independent.

### 4.2 Mean intensity J

![J slices](../plot/compare_transport/j_slices.png)

Band-mean J profiles (8 base-cell bands, `profiles.csv`; δ-on/off pairs
agree to ≤3e−4, δ-off values shown):

| z [m] | chain 24³ | SC 24³ | octree d3 | chain 48³ | SC 48³ | octree d4 | reference (642 ord.) |
|---|---|---|---|---|---|---|---|
| 0.625 | 0.4372 | 0.3680 | 0.3095 | 0.4382 | 0.3660 | 0.3102 | 0.3388 |
| 3.125 | 0.4529 | 0.3629 | 0.3431 | 0.4539 | 0.3636 | 0.3587 | 0.3096 |
| 4.375 | 0.4964 | 0.4077 | 0.4154 | 0.4974 | 0.4105 | 0.4143 | 0.3186 |
| 6.875 | 0.5870 | 0.5631 | 0.5377 | 0.5872 | 0.5642 | 0.5443 | 0.5517 |
| 9.375 | 0.5893 | 0.5712 | 0.5850 | 0.5894 | 0.5709 | 0.5849 | 0.5840 |

Note the reference column: in the cloud core (z = 3.1–4.4 m) the reference
J is far *lower* (0.31–0.32) than every depth-3-or-coarser run
(0.41–0.50) — the coarse grids overestimate in-cloud radiation trapping;
only the depth-4/5 octree resolution comes close. Compared with the
42-ordinate reference of the previous study version, the 642-ordinate
reference lowers the cloud-core bands further (0.319/0.330 → 0.310/0.319)
and raises the bottom band (0.319 → 0.339): part of the old
octree-d3↔reference agreement in the bottom band was angular-error
cancellation.

![J difference maps](../plot/compare_transport/j_diff.png)

- **chain vs. SC on identical cells is the largest pairwise discrepancy of
  the six-run study** (max 31%, mean 9%): SC's extra numerical diffusion
  lets more radiation penetrate the optically thick core, where it is
  absorbed (ω = 0.9, τ = 10) — J is lower everywhere below and inside the
  cloud (0.368 vs 0.437 in the bottom band).
- **unigrid-chain vs. octree** (18% max, 2.8% mean) reproduces the
  anchor-5 pattern of the manual: the unigrid grazing-ordinate stencil
  error inflates J below the cloud (0.437 vs 0.310 in the bottom band,
  the manual's 29%), with a ring structure around the cloud in the
  difference maps.
- **delta-M pairs are flat** (≤1.3% max, ~0.01–0.02% mean): the converged
  solution is independent of δ-M, as designed; the residual pointwise gap
  is the documented O(τ_cell) quadrature error of the explicit
  −ω g^M I term.

### 4.3 Angular variance

![Angular variance slices](../plot/compare_transport/var_slices.png)
![Angular variance difference maps](../plot/compare_transport/var_diff.png)

σ² is high in the vacuum (collimated beam + background = strongly
anisotropic field) and collapses toward zero inside the cloud core where
multiple scattering isotropises the field — the "hole" in all six panels.
Scheme differences show up mostly in the *shadow region below the cloud*
and in the beam path: SC's diffusion smooths the beam shadow (47.8% max
variance difference vs. chain), while the octree resolves the shadow
boundary at 0.156 m. delta-M changes σ² by ≤0.3%.

### 4.4 Upward radiance at TOA

![TOA upward flux maps](../plot/compare_transport/toa_fup_maps.png)
![TOA upward flux difference maps](../plot/compare_transport/toa_fup_diff.png)
![TOA radiance vs mu](../plot/compare_transport/toa_mu_spectra.png)

- The **chain** runs show a *bright* spot above the cloud (the scattering
  layer seen from above), the **SC** runs a deep *shadow* blob — the
  TOA-integrated signature of SC's extra diffusion (max F⁺ difference
  35%). The octree map is coarse by construction (top-layer cells are
  1.25–2.5 m, the cloud is far below and the refinement is κ_e-driven).
- Per-ordinate domain-mean radiance (`toa_mean_radiance.csv`): chain and
  octree agree closely at all μ; SC collapses at grazing ordinates
  (μ = 0.10: 0.032 vs 0.088 chain / 0.041 octree) — the SC wrap-ring /
  grazing-path error — and lies between chain and octree at μ ≳ 0.7.
- Domain-mean upward radiance: chain 0.179, octree 0.170, SC 0.142,
  reference (642 ord.) 0.168 [F₀ sr⁻¹]; consistently, the TOA net flux is
  least upward-escaping for SC (mean F_z −2.5935 vs −2.5215/−2.5312,
  reference −2.5074).

### 4.5 Pair statistics (from `pair_stats.csv`)

Relative differences of B − A, normalised by max|A| of the reference run
(max over the common lattice / mean over the domain):

| pair | J max | J mean | σ² max | σ² mean | F⁺ max | F⁺ mean |
|---|---|---|---|---|---|---|
| SC − chain, δ off | 31.1% | 9.1% | 47.8% | 6.2% | 35.2% | 9.8% |
| SC − chain, δ on  | 31.1% | 9.1% | 47.8% | 6.2% | 35.2% | 9.8% |
| δ1 − δ0, unigrid chain | 0.9% | 0.01% | 0.2% | 0.02% | 0.1% | 0.02% |
| δ1 − δ0, unigrid SC    | 0.7% | 0.01% | 0.2% | 0.01% | 0.0% | 0.02% |
| δ1 − δ0, octree        | 1.3% | 0.02% | 0.3% | 0.04% | 0.1% | 0.05% |
| octree − unigrid chain, δ off | 18.2% | 2.8% | 23.6% | 3.2% | 23.0% | 7.6% |

---

## 5. Which run is best? — accuracy ranking vs. the reference

All pairwise numbers above compare approximations *to each other*. Ranking
needs a solution that is demonstrably closer to the continuum limit: the
fine-octree, **642-ordinate** reference `octree_ref_dM0` (section 2.6).
Errors below are relative to max|·| of the reference field on the common
lattice (`ranking.csv`, `err_bars.png`, `j_err_maps.png`). Because the
reference resolves angle 15× better than the development grid, the errors
of the 42-ordinate runs below are *space + angle* errors together — which
is exactly what one wants when asking how far a practical run is from the
truth.

### 5.1 Ranking

| run | wall [s] | J err max | J err mean | σ² err max | F⁺ err max | ΔF_z @TOA |
|---|---|---|---|---|---|---|
| `octree_ref_dM1` | 3393.7 | 0.2% | **0.005%** | 0.1% | 0.0% | +0.0001 |
| `octree_d4_dM1` | 33.0 | 11.9% | **1.85%** | 12.9% | 14.1% | −0.0245 |
| `octree_dM1` | 20.7 | 11.9% | **1.92%** | 14.1% | 14.9% | −0.0235 |
| `octree_dM0` | 37.9 | 11.9% | 1.91% | 14.0% | 14.9% | −0.0238 |
| `uni_chain48_dM1` | 5.4 | 16.7% | **3.46%** | 17.7% | 20.7% | −0.0150 |
| `uni_chain_dM1` | 0.7 | 20.6% | **3.47%** | 24.7% | 20.6% | −0.0140 |
| `uni_chain_dM0` | 1.6 | 20.6% | 3.46% | 24.7% | 20.6% | −0.0142 |
| `uni_sc48_dM1` | 15.9 | 36.2% | **6.18%** | 43.4% | 37.6% | −0.0853 |
| `uni_sc_dM1` | 4.1 | 33.2% | **6.29%** | 44.2% | 37.5% | −0.0860 |
| `uni_sc_dM0` | 16.6 | 33.2% | 6.30% | 44.2% | 37.6% | −0.0861 |

(reference `octree_ref_dM0` = 0 by construction; ΔF_z relative to −2.5074:
chain −0.6%, SC −3.4%, octree d3 −0.9%; Δ⟨J⟩: chain +0.141, SC +0.086,
octree d3 +0.024)

![J error maps vs reference](../plot/compare_transport/j_err_maps.png)
![error bars](../plot/compare_transport/err_bars.png)

**Accuracy order: octree > unigrid chain > unigrid SC** (mean J error
1.9% vs 3.5% vs 6.3%). The gaps are much *narrower* than against the
42-ordinate reference of the previous study version (0.5% vs 2.7% vs
6.9%): with the angular error no longer cancelling, all 42-ordinate runs
sit near a common **angular error floor**, and the octree's spatial
advantage (which is real — see 5.2) is partly masked by it.

### 5.2 Convergence evidence (why the ranking is meaningful)

![J convergence](../plot/compare_transport/j_convergence.png)
![TOA convergence](../plot/compare_transport/toa_convergence.png)

- **Octree** d3 → d4 (both 42 ordinates): mean J error 1.91% → 1.85%,
  max 11.9% → 11.9% — *flat*, in stark contrast to the 0.47% → 0.22%
  halving measured against the 42-ordinate reference. Against the
  642-ordinate reference the octree's 42-ordinate **angular** error
  (~1.9% mean, ~12% max) dominates its remaining cell-size error.
- **Unigrid chain** 24³ → 48³ (8× cells): mean error 3.46% → 3.46%,
  max 20.6% → 16.7% — the mean is pinned at the angular floor; the
  remaining error is dominated by the **grazing-ordinate stencil
  direction error** (an angular error that does not shrink with h at
  fixed 42 ordinates) plus the common quadrature error.
- **Unigrid SC** 24³ → 48³: mean 6.30% → 6.18%, max 33.2% → 36.2% —
  likewise dominated by the angular grazing-ordinate wrap-ring error
  (section 4.4), not by cell size.
- **The reference's own angular error**: moving the reference grid from
  42 to 642 ordinates changed ⟨J⟩ by 1.6% and TOA F_z by 1.0% — so the
  42-ordinate quadrature error is a ~1–2% effect on scalars even for the
  octree, and the 642-ordinate reference's remaining angular error is
  correspondingly smaller again.

Putting the two rankings together: the octree's *cell-size* error really
does halve per refinement level (0.47% → 0.22% → 0 when the angle is held
at 42 ordinates), but at 42 ordinates every scheme additionally carries a
scheme-dependent *angular* error (octree ~1.9%, chain ~3.5%, SC ~6.3%
mean) that only N_dir removes. The unigrid schemes' angular error is the
grazing-ordinate stencil/wrap error of section 4.4 — much larger than the
octree's pure quadrature error, which is why the octree still wins the
ranking and why refining the unigrid alone does not close the gap.

### 5.3 Verdict

- **Most accurate** (of the original six): `octree_dM1` — 1.9% mean J
  error against the 642-ordinate reference (0.47% against the
  42-ordinate one), 20.7 s. The fine reference grid itself
  (`octree_ref_dM0`, 2367 s) is the right choice when the answer itself
  is the goal; its δ-M twin reproduces it to 0.005% mean but is *slower*
  (3394 s) because δ-M no longer accelerates at 642 ordinates.
- **Best accuracy per cost**: `octree_dM1` for physically trustworthy
  fields; `uni_chain_dM1` (0.7 s, 3.5% mean error) is the cheap workhorse
  when 4% is acceptable — but its error is angular and will not shrink
  with grid refinement at 42 ordinates.
- **delta-M on dominates off at 42 ordinates** (same converged fields,
  ≤1.3% max, 2–4× less wall time), but **not at 642** (34 vs 23
  iterations on the reference grid). At cloud-core g ≈ 0.95 with the
  development grid it should be the default; on finely-resolved angular
  grids it is at best neutral.
- **SC is dominated on both axes**: less accurate *and* more expensive
  than chain at identical grids (24³: 6.3% @ 4.1 s vs 3.5% @ 0.7 s).

---

## 6. Findings

1. **delta-M (M = 1) does what it should at 42 ordinates**: it cuts the
   iteration count to 12 for every scheme (from 32/51/30), and the
   converged fields agree with the δ-off runs to ≤1.3% max / ≤0.05% mean
   — the residual is the known O(τ_cell) explicit-term quadrature gap
   (it halves to 0.19% at the finest grid), not a physics change. At the
   cloud-core g ≈ 0.95 with the development grid it should be the default
   setting. **New**: at 642 ordinates the acceleration disappears (23 →
   34 iterations on the reference grid) — the benefit is tied to the
   coarse, strongly row-normalized 42-ordinate HG operator.
2. **Against the 642-ordinate reference, the accuracy order is octree >
   chain > SC** (mean J error 1.9% vs 3.5% vs 6.3%) — but all three sit
   near a common *angular error floor* of the 42-ordinate development
   grid: refining the octree d3 → d4 or the unigrid 24³ → 48³ barely
   moves the mean error any more. The octree's *cell-size* error still
   converges normally (0.47% → 0.22% → 0 against the 42-ordinate
   reference); what the new ranking exposes is that at 42 ordinates the
   angular part dominates for every scheme — smallest for the octree
   (pure quadrature error), much larger for the unigrid schemes
   (grazing-ordinate stencil/wrap errors).
3. **SC is systematically more diffusive**: lower J inside and below the
   cloud, deeper TOA shadow, collapsed grazing-ordinate radiance, and a
   ~3.4% too-small net upward flux at TOA. It is also ~5× slower per
   iteration than chain on the unigrid (wrap-around ring passes).
4. **The octree buys both resolution and path fidelity**: finer cells
   where κ_e demands them (0.078 m in the reference cloud core) *and*
   marching along the true ordinate directions; its exact-march transport
   removes the chain's grazing-ordinate stencil error, which is why
   unigrid-chain vs. octree is the manual's anchor-5 pattern.
5. **δ-M is scheme- and grid-agnostic in the converged solution**: the
   δ-on↔δ-off differences are ~2 orders of magnitude smaller than any
   scheme comparison, so the comparison is not confounded by the
   scattering-operator split (its *iteration* benefit is not
   angular-grid-agnostic — see 1).
6. **Angular error is now measured, not assumed**: the 42 → 642 ordinate
   step on the reference grid moves ⟨J⟩ by 1.6% and TOA F_z by 1.0%, and
   the ranking gaps of the 42-ordinate runs are dominated by their
   angular error. The full direction-convergence study (42/162/642 for
   all schemes, phase 6) should re-run this ranking to complete the
   picture.

---

## 7. Interactive 3D scenes

Self-contained vtk.js HTML (open directly in a browser — orbit, actor
visibility toggling, colour maps; no server, no Python needed). One scene
per run (κ_e isosurfaces at 0.5/1/2/3 m⁻¹ + J and σ² cut planes at
x = 0, y = 0, z = 4) plus difference scenes (|ΔJ| isosurfaces + slices),
all regenerated 2026-10-15 on macOS:

```
plot/compare_transport/scene_uni_chain_dM0.html   scene_uni_chain_dM1.html
plot/compare_transport/scene_uni_sc_dM0.html      scene_uni_sc_dM1.html
plot/compare_transport/scene_octree_dM0.html      scene_octree_dM1.html
plot/compare_transport/scene_octree_ref_dM0.html
plot/compare_transport/scene_diff_chain_vs_sc_dM0.html
plot/compare_transport/scene_diff_uni_vs_octree_dM0.html
plot/compare_transport/scene_diff_ref_vs_chain24_dM0.html
plot/compare_transport/scene_diff_ref_vs_sc24_dM0.html
plot/compare_transport/scene_diff_ref_vs_octree3_dM0.html
```

The three `ref_vs_*` scenes are the ranking story made visible: where each
coarse run gets J wrong against the reference (chain: below/around the
cloud; SC: everywhere diffuse; octree d3: no longer just the cloud core —
a broad few-percent pattern strongest in the anisotropic vacuum around
the cloud, the angular error floor of the 42-ordinate ordinates).

(pyvista's vtk.js export cannot carry volume mappers, hence slices +
isosurfaces; the static volume overview is `vol_j.png`.)

## 8. Output files

| file | content |
|---|---|
| `summary.csv` | run table (grid, N_dir, iterations, wall, ⟨J⟩, TOA F_z), all 11 runs |
| `ranking.csv` | per-run error vs the reference (J, σ², F⁺, scalars) |
| `profiles.csv` | band-mean J profiles, all 11 runs |
| `toa_mean_radiance.csv` | per-ordinate domain-mean upward radiance at TOA, 42-ordinate runs |
| `toa_mean_radiance_642.csv` | same for the two 642-ordinate reference runs |
| `pair_stats.csv` | max/mean relative differences of all six pairs × J, σ², F⁺ |
| `j_slices.png`, `j_profiles.png`, `j_diff.png` | mean-intensity figures |
| `j_err_maps.png` | \|J − J_ref\| at the cloud plane, every run vs reference |
| `j_convergence.png` | band-mean J: refinement sequences + reference |
| `err_bars.png` | max relative error vs reference, all runs (log scale) |
| `var_slices.png`, `var_diff.png` | angular-variance figures |
| `toa_fup_maps.png`, `toa_fup_diff.png`, `toa_mu_spectra.png` | TOA radiance figures |
| `toa_convergence.png` | TOA μ-spectra: refinement sequences + reference |
| `vol_j.png` | static volume renderings, base six runs + reference |
| `scene_*.html` | interactive 3D scenes (gitignored, regenerate by rerunning) |
| `_solve_cache/` | per-run solved fields (gitignored; reruns resume) |
