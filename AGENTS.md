# AGENTS.md — 3d_rt_sandbox

Guidance for coding agents working in this repository.

## What this project is

A 3D all-sky, **monochromatic** short-characteristics radiative-transfer solver written in
Python. Physical assumptions (do not change them without being asked):

- **LTE**, local source function `S = (1 - omega) * B_nu(T) + omega * J_nu`
- **Monochromatic** — one frequency at a time, no line structure, no frequency coupling
- **No polarization** (scalar RTE only). Stokes machinery in `scripts/icosphere_grid.py`
  (`rotater_mat`, `rotatepoles`) is legacy and unused by the solver — do not extend it.
- **No refraction** — rays are straight lines
- **Incoming solar radiation as a direct beam** (collimated, single direction)
- **Periodic boundaries** in `x` and `y`; top of the cuboid is space, bottom is ground
- **Phase function:** Henyey–Greenstein only
- **Spatial discretization:** octree, refinement driven by the extinction coefficient
- **Iterative scheme:** lambda iteration with 1D Ng acceleration, structured like the DOIT
  solver of ARTS (compute incoming radiation → transport → next iteration)

See `plan/plan.md` for the agreed design, implementation phases and the decisions log.
That plan is the source of truth when a decision is already recorded there; if you change
a design decision, update the decisions log in `plan/plan.md` rather than contradicting it
silently. The solver package `rt3d/` described there does **not** exist yet — it is created
in phase 0.

**All plans belong in the project's `plan/` folder.** Whenever you create a plan
(work plans for new tasks, follow-up plans like `plan/manual-plan.md`), write it to
`plan/` in this repository — never to external or tool-managed plan directories.

## Environment

Mamba/conda environment **`pyarts`**. There is no `mamba`/`conda` on `PATH` in a plain
shell — call the environment interpreter by absolute path:

```powershell
& C:\Users\fredd\miniforge3\envs\pyarts\python.exe <script.py>
```

**BLAS gotcha:** the env's numpy links against the `libcblas.dll` shim, which
*delay-loads* `openblas.dll` from `...\envs\pyarts\Library\bin`. Running the
interpreter by absolute path **without activating the env** leaves that directory
off `PATH`, and every matrix product then hard-crashes the process with exit code
`-1066598273` (`0xC06D007F`, delay-load failure — no traceback). Prepend it to
`PATH` before running anything that does a matmul:

```powershell
$env:Path = "C:\Users\fredd\miniforge3\envs\pyarts\Library\bin;" + $env:Path
```

Other envs on this machine (`astro`, `rvg`, `base`) are **not** for this project.

Installed and available in `pyarts`:

| Package | Version |
|---|---|
| Python | 3.14.6 |
| numpy | 2.5.2 |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| xarray | 2026.7.0 |
| h5py | 3.16.0 |
| pyarts3 | 3.0.0.dev10 |

**Not installed:** `pytest`, `numba`, `pyvista`. Do not import them without first
installing them into `pyarts` and saying so explicitly.

## Repository layout

```
3d_rt_sandbox/
  scripts/        existing standalone research scripts (see below)
  inputdata/      currently empty — input fields / grids go here
  plot/           currently empty — figure and image output goes here
  plan/           all plans live here (plan.md: agreed design, phase breakdown
                  and decisions log; additional plans as separate files)
  AGENTS.md       this file
  .gitignore      Visual Studio template
  .gitattributes  "* text=auto" (LF normalization)
```

`scripts/icosphere_grid.py` and `scripts/class_test.py` are pre-existing research code
from an earlier project (author header: Manfred Brath). **Treat them as read-only
reference unless explicitly asked to modify them.** They define the conventions to reuse:

- `icosphere_grid.py` — icosahedron construction (`make_icosahedric_grid`), geodesic
  refinement (`refine_icosahedric_grid`), cart→sphere (`vertices_cart2sphere`), vertex
  rotation (`rotate_vertices`), barycentric interpolation on facets.
  Note: `refine_icosahedric_grid` uses O(N²) pure-Python dedup loops and is very slow at
  refinement level ≥ 3. Write a vectorized replacement rather than optimizing it in place.
- `class_test.py` — `domain` / `periodic_domain` classes holding `extcoeff`, `albedo`, `g`
  as `scipy.interpolate.RegularGridInterpolator` fields, the normalized 3D Gaussian helper
  `gauss(...)`, the Gaussian-cloud test case, and `point_on_cuboid(...)`.

## Input conventions

The four physical inputs are **extinction coefficient** `extcoeff` (κ_e), **scattering
coefficient** (κ_s), **temperature** `T`, and **asymmetry parameter** `g`.

Derived quantities, always computed, never stored as separate inputs:

```
kappa_a = kappa_e - kappa_s          # absorption coefficient
omega   = kappa_s / kappa_e          # single scattering albedo, guard kappa_e == 0
```

The older `class_test.py` stores `albedo` instead of `kappa_s`. New code takes `kappa_s`.
Always guard `kappa_e -> 0` (vacuum cells) before dividing.

Direction grid is **always** an icosahedral (geodesic) grid — never a polar/azimuthal
product grid. Solid-angle weights come from spherical-triangle areas, not `4*pi/N`.

The development grid is **icosahedral refinement 1 = 42 ordinates** (`N_dir = 10*4**n_refine + 2`).
All development, test cases, benchmarks and validation runs use it. `n_refine` is a
configurable option that must be exposed on the solver config, on every case and every
runner script, and recorded in every output file — but do not raise it just because a run
looks cheap. Higher levels (162, 642) belong to the phase 6 convergence study. No code may
assume a specific `N_dir`: no hard-coded 42/162, no fixed-size direction arrays, no
level-specific index arithmetic.

## Code style

- Follow the existing style: NumPy-style docstrings, `numpy as np`, 4-space indent,
  snake_case functions, `lowercase_class_names` (`class domain:`) as in `class_test.py`.
- Keep the existing files' author headers intact.
- Scripts in `scripts/` use `# %%` cell markers (VS Code / Spyder interactive workflow).
  Match that when adding runner scripts.
- Heavy loops are the main bottleneck. Prefer vectorized NumPy; batch the scattering
  integral as a single BLAS matmul over all cells instead of looping over cells.
- No network access is assumed. Do not add dependencies without asking.

## Running and checking

This directory **is a git repository** (branch `main`). There is **no test runner
installed**.

Quick sanity check that the environment works:

```powershell
& C:\Users\fredd\miniforge3\envs\pyarts\python.exe -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

Plots must not block: use `matplotlib.use("Agg")` and `fig.savefig("plot/...")` for
anything run non-interactively, and write figures into `plot/`.

## Subagents

Hard rules when delegating to subagents:

- **At most 2 subagents running in parallel.** Never launch a third concurrent subagent.
  If more work needs parallelizing, batch it into two subagents or do it inline.
- **Never reuse a subagent.** Always start a fresh subagent conversation — do not pass a
  `sessionID` to continue an earlier one, and do not treat a previous subagent's final
  answer as authoritative context for a new task. Each subagent call must carry all the
  context it needs in its own prompt.

## Physics validation anchors

When changing transport, source, or iteration code, at least these must still hold:

1. `sum(w_i) == 4*pi` over the direction grid
2. `int Phi dOmega == 1` for the HG phase function
3. `kappa_s = 0`, `B = 0` → transmittance equals `exp(-tau)` along every ray
4. `div F` equals the heating rate from the intensity field (energy conservation)
5. Octree result converges to the uniform-grid result on the `class_test.py` Gaussian cloud
