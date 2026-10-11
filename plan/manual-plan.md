# Plan — Create `MANUAL.md` for 3d_rt_sandbox

Deliverable: a single detailed manual **`MANUAL.md`** in the repo root
(`C:\Users\fredd\projekte\3d_rt_sandbox\MANUAL.md`), in **English**, with a table of
contents. Theory depth: physics + numerics (key formulas, no exhaustive proofs).
The manual documents the **actual code** in `rt3d/` (phases 0–6 complete, 80 tests
green), not the aspirational names in `plan/plan.md` (e.g. the entry point is
`solver.lambda_iteration(...)`, there is no `SolverConfig`).

## Structure

### Part I — Theory (~40%)

1. **Scope & physical assumptions** — monochromatic, LTE, scalar RTE (no
   polarization), no refraction, Henyey–Greenstein phase function only, periodic
   x/y, top = space, bottom = flat ground, solar radiation as a direct beam.
2. **Governing equation** —
   `Omega·grad I + kappa_e I = kappa_s ∮ Phi_HG I dOmega' + kappa_a B(T)`;
   derived quantities `kappa_a = kappa_e − kappa_s`, `omega = kappa_s/kappa_e`
   with the `kappa_e → 0` guard (vacuum cells); HG phase function
   `Phi_HG = (1−g²)/(4π(1+g²−2g cosΘ)^{3/2})`.
3. **Solar beam split** — `I = I_diff + I_beam`; why a direction-delta cannot
   live on the icosahedral quadrature; dedicated beam sweep `I_beam = F0·exp(−tau_b)`;
   first-scattering term `kappa_s·Phi(Omega_sun→Omega_i)·F0·exp(−tau_b)` as a
   fixed (never iterated) source contribution; beam first-scatter HG row
   normalization convention.
4. **Angular discretization** — icosahedral grids, `N_dir = 10·4^n_refine + 2`
   (12/42/162/642), 42 ordinates as the development grid; solid-angle weights
   from spherical-triangle areas (not 4π/N); precomputed HG matrix as one BLAS
   matmul; **row normalization** and why (raw 42-ordinate HG row sums 17–22 at
   g=0.95 make lambda iteration supercritical; normalized rows conserve
   isotropic radiation exactly).
5. **Spatial discretization** — uniform `unigrid` vs. flat struct-of-arrays
   octree; split criteria `kappa_e·h > tau_max` and relative kappa_e gradient > eps
   with the `1e-2·tau_max` floor (vacuum-tail rationale); 2:1 balance via 3×3 face
   probing; O(1) point location; why the solver uses **exact characteristics**
   (per-leaf-center backward march to domain entry, DDA) instead of the
   best-aligned center-to-center chain (scale-invariant ~4 tau path error).
6. **Short-characteristics transport** — one characteristic per leaf/cell center;
   explicit sub-segments; recursion exact for piecewise-constant sources:
   `I_c = I_u·exp(−tau_tot) + Σ_sub S_sub·(1−exp(−tau_sub))·exp(−tau_after)`;
   BFS-wave / cumsum vectorization.
7. **Boundary conditions** — top `I_diff = 0` (+ optional background); Lambertian
   + emissive ground `I_up = A_g/pi·F_down + eps_g·B_g` (+ Lambertian beam
   reflection), downward ordinates swept first; periodic sides by ray
   exit→entry wrap (incl. the wrap-stall bugfix note).
8. **Iteration & Ng acceleration** — DOIT-style loop (source → transport →
   residual → Ng update); 1D angle-averaged `Lambda_bar` in z from horizontally
   averaged `kappa_e(z), omega(z)`, dense Nz×Nz fit; history-pair identity
   `Lambda_bar x = y` with swept-source/consecutive-update differences (why
   step-difference pairing breaks under acceleration); rho(Lambda_bar) ≥ 1
   safeguard; measured speedups (ω=0.5: 22→10, ω=0.9: 87→13, ω=0.99: 178→14 iters).

### Part II — Usage (~60%)

9. **Environment setup** — mamba env `pyarts`, interpreter by absolute path
   `C:\Users\fredd\miniforge3\envs\pyarts\python.exe`; **BLAS/openblas PATH
   gotcha** prominently (prepend `...\envs\pyarts\Library\bin` or every matmul
   hard-crashes with exit code -1066598273, no traceback); package versions;
   pytest installed, no numba/pyvista.
10. **Module tour** — one section per `rt3d` module with real signatures:
    - `directions.py` — `icosahedric_grid(n_refine=1)`, `direction_grid`,
      `solid_angle_weights`, `hg`, `hg_matrix`
    - `geometry.py` — `cuboid`, `unigrid`, `ray_box_exit`, `march_lattice`,
      `ground_surface(albedo, emissivity, B_g, height=...)`
    - `octree.py` — `octree_grid`, `march_octree`, `node_major_segments`
    - `characteristics.py` — `stencil_for_direction`, `build_chain` (unigrid),
      `build_chain_octree` (reference only, not used by solver)
    - `transport.py` — `boundary_conditions`, `sweep_all`, `sweep_all_octree_exact`,
      `precompute_octree_march`, `beam_optical_depth(_octree)`, ground helpers
    - `source.py` — `single_scattering_albedo`, `mean_intensity`, `flux_vector`,
      `scattering_source`, `beam_scattering_source`
    - `accelerator.py` — `ng1d`
    - `solver.py` — `lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, g,
      periodic_box, beam, n_iter, tol, ng)` + full result-dict reference
      (`I, S, J, F, omega, history, n_iter, converged`)
    - `diagnostics.py` — `heating_rate`, `flux_divergence`, `energy_balance`,
      `allsky`
    - `cases/` — `gaussian_cloud`, `gauss` (note: `slab_1d`/`clear_sky` from the
      plan do not exist as case modules; the slab lives in `scripts/phase2_slab.py`)
11. **Worked end-to-end example** — Gaussian cloud + ground + solar beam:
    build grid (unigrid and octree variants), sample fields, set BCs via
    `boundary_conditions(ground=...)`, run `lambda_iteration(..., ng=True)`,
    interpret the result dict, plot J / all-sky images with `matplotlib.use("Agg")`
    into `plot/`. The example will be **executed with the pyarts interpreter
    (PATH fix applied) and verified** before inclusion.
12. **Runner scripts** — table of `scripts/phase0_check.py` … `phase6_directions.py`:
    what each validates and how to run it; note `class_test.py` /
    `icosphere_grid.py` are read-only legacy reference.
13. **Condensed API reference** — signature + one-line description tables for
    all ~48 public functions/classes.
14. **Validation anchors** — the 5 physics anchors from AGENTS.md with measured
    numbers: `sum(w_i)=4π`; `∮Phi dΩ=1`; transmittance = exp(−tau) at machine
    precision (octree exact march); anchor 4 div F vs heating rate (rms 1.6%,
    max 14%, global closure 3.2%); anchor 5 octree-vs-unigrid 18% J difference
    interpretation (unigrid grazing-ordinate stencil error, not octree error).
15. **Known limitations & performance** — 42 ordinates ≈6% J error at g≈0.95,
    even 642 not fully converged at g=0.952, non-monotone angular convergence
    at large g (error cancellation); variable surface = hook only,
    `NotImplementedError` in transport; sweep cost ~2.1–2.3 ms per ordinate per
    24³ sweep, linear in N_dir; full Ng cloud run at 642 ordinates ≈84 s;
    numba not needed.

## Method

1. Read every `rt3d/` module fully to extract exact signatures, defaults,
   return values and docstrings — API sections must match code, not plan.md.
2. Draft the worked example, run it with
   `$env:Path = "C:\Users\fredd\miniforge3\envs\pyarts\Library\bin;" + $env:Path`
   then `& C:\Users\fredd\miniforge3\envs\pyarts\python.exe <example>`; include
   only verified outputs/numbers.
3. Write `MANUAL.md` in one pass; keep style consistent with repo
   (English prose, code blocks with real imports).
4. Do not modify any other file; `plan/plan.md` stays the decisions-log source
   of truth (manual cross-references it rather than duplicating the log).

## Open choices (defaults assumed)

- Location: repo root `MANUAL.md` (single file). Alternative: split under `docs/`.
- No git operations (no repo in this directory).
