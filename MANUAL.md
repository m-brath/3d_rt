# MANUAL.md — 3d_rt_sandbox

User manual for **`rt3d`**, a 3D all-sky, monochromatic radiative-transfer
solver with two solver families on shared grids and boundary conditions:

- **short characteristics (SC)** — deterministic λ-iteration on icosahedral
  ordinate sets (unigrid chain / single-cell SC sweeps, octree exact
  characteristics), with optional numba fast paths on an all-NumPy core;
- **Monte Carlo (MC)** — Woodcock null-collision transport with
  angle-exact HG sampling (`rt3d/mc/`), forward solvers A1/A2 and an adjoint
  detector solver B.

This manual documents the **actual code** in `rt3d/` (SC phases 0–6 and the
MC family M0–M6 complete, **341 tests green** — 330 in the default tiers,
re-verified on macOS/arm64 for this update; plus the 11 `slow`-tier tests).
The agreed design and the decisions log live in
[`plan/plan.md`](plan/plan.md), the MC design in
[`plan/mc-plan.md`](plan/mc-plan.md), and the test matrix with its findings
log F1–F14 in [`plan/solver-test-matrix-plan.md`](plan/solver-test-matrix-plan.md)
— those files stay the source of truth for *why* a decision was made; this
manual explains *what the code does and how to use it* and cross-references
the plans instead of duplicating the log.

**Contents**

- [Part I — Theory](#part-i--theory)
  1. [Scope and physical assumptions](#1-scope-and-physical-assumptions)
  2. [Governing equation](#2-governing-equation)
  3. [The solar beam split](#3-the-solar-beam-split)
  4. [Angular discretization — icosahedral direction grids](#4-angular-discretization--icosahedral-direction-grids)
  5. [Spatial discretization — unigrid and octree](#5-spatial-discretization--unigrid-and-octree)
  6. [Short-characteristics transport](#6-short-characteristics-transport)
  7. [Boundary conditions](#7-boundary-conditions)
  8. [Iteration scheme and 1D Ng acceleration](#8-iteration-scheme-and-1d-ng-acceleration)
  9. [Monte Carlo solver family](#9-monte-carlo-solver-family)
- [Part II — Usage](#part-ii--usage)
  10. [Environment setup](#10-environment-setup)
  11. [Module tour](#11-module-tour)
  12. [Worked end-to-end example](#12-worked-end-to-end-example)
  13. [Runner scripts](#13-runner-scripts)
  14. [Condensed API reference](#14-condensed-api-reference)
  15. [Verification invariants and the test matrix](#15-verification-invariants-and-the-test-matrix)
  16. [Known limitations and performance](#16-known-limitations-and-performance)

---

# Part I — Theory

## 1. Scope and physical assumptions

`rt3d` solves the **scalar radiative transfer equation (RTE)** in a 3D
rectangular domain under the following fixed physical assumptions (plan
section 1; do not change them without updating `plan/plan.md`):

| Assumption | Meaning in the code |
|---|---|
| **Monochromatic** | One frequency at a time. All intensities, $B$, $B_g$, $F_0$ are values at the working frequency. No line structure, no frequency coupling. |
| **LTE** | Local source function $S = (1-\omega)B + \omega J$ (plus scattering anisotropy, see §2). |
| **Scalar RTE** | No polarization. The Stokes machinery in `scripts/icosphere_grid.py` (`rotater_mat`, `rotatepoles`) is legacy and unused. |
| **No refraction** | Rays are straight lines; the DDA marches of `rt3d/geometry.py` and `rt3d/octree.py` traverse straight rays. |
| **Henyey–Greenstein phase function only** | The only scattering phase function; asymmetry parameter $g$ may be scalar or per-cell. |
| **Periodic x and y** | Rays exiting a side face re-enter the opposite face (`periodic_box` argument throughout). |
| **Top = space, bottom = flat ground** | Top inflow is a constant background `top_I` (default 0); the bottom face may carry a Lambertian + emissive ground (`ground_surface`). |
| **Solar radiation as a direct beam** | A collimated beam in one direction, handled separately from the diffuse field (§3). |

The spatial domain is an axis-aligned **cuboid** (`rt3d.geometry.cuboid`):
$z = z_1$ is space (top), $z = z_0$ is the ground plane (bottom), $x$ and $y$
wrap periodically.

Both solver families implement exactly these assumptions on the same grids,
fields and boundary conditions, so their results are differenced cell-by-cell
(the pytest matrix, §15):

| family | entry points | angular treatment | typical question |
|---|---|---|---|
| **short characteristics** | `rt3d.solver.lambda_iteration` | icosahedral quadrature (§4) | full $I$/J/F field of a source field |
| **Monte Carlo, forward** | `rt3d.mc.forward.j_mc` (stage 1), `radiance_from_source` (stage 2) | continuous — HG inverse-CDF, no quadrature in transport | same fields by particle transport; angle-exact at large $g$ |
| **Monte Carlo, adjoint** | `rt3d.mc.backward.backward_detector_mc` | continuous; quadrature only as detector-cone binning | radiance **at a detector** (all-sky at one point) |

## 2. Governing equation

For monochromatic intensity $I(\mathbf{r}, \boldsymbol{\Omega})$, LTE,
scalar RTE:

$$
\boldsymbol{\Omega}\cdot\nabla I(\mathbf{r},\boldsymbol{\Omega})
\;+\; \kappa_e(\mathbf{r})\, I(\mathbf{r},\boldsymbol{\Omega})
\;=\; \kappa_s(\mathbf{r}) \oint \Phi_{\mathrm{HG}}(\boldsymbol{\Omega}'\to\boldsymbol{\Omega})\, I(\mathbf{r},\boldsymbol{\Omega}')\, d\Omega'
\;+\; \kappa_a(\mathbf{r})\, B(\mathbf{r})
$$

with the derived quantities **always computed, never stored as inputs**
(AGENTS.md input conventions):

$$
\kappa_a = \kappa_e - \kappa_s
\qquad\text{(absorption coefficient)},
\qquad
\omega = \frac{\kappa_s}{\kappa_e}
\qquad\text{(single scattering albedo)}
$$

$\omega$ is computed by `rt3d.source.single_scattering_albedo` with a
**vacuum guard**: where $\kappa_e = 0$ (vacuum cells) the division is skipped
and $\omega = 0$ is used, so vacuum cells neither scatter nor absorb.

The **Henyey–Greenstein** phase function (`rt3d.directions.hg`) is
normalized so that its exact integral over solid angle is 1:

$$
\Phi_{\mathrm{HG}}(\cos\Theta)
= \frac{1-g^2}{4\pi\left(1+g^2-2g\cos\Theta\right)^{3/2}},
\qquad -1 < g < 1
$$

where $\Theta$ is the scattering angle between incoming $\boldsymbol{\Omega}'$
and outgoing $\boldsymbol{\Omega}$. With
$S = (1-\omega)B + \omega\oint \Phi\, I\, d\Omega'$ the RTE takes the form

$$
\frac{dI}{ds} = \kappa_e\left(S - I\right)
$$

along a ray, which is the form the transport recursion of §6 integrates
exactly.

## 3. The solar beam split

The incoming solar radiation is a **direct beam**: a collimated bundle in one
unit direction $\boldsymbol{\Omega}_{\mathrm{sun}}$ (must have
$\Omega_{\mathrm{sun},z} < 0$) with irradiance $F_0$ at the domain top. A
direction-delta distribution
$I_{\mathrm{beam}} \propto \delta(\boldsymbol{\Omega}-\boldsymbol{\Omega}_{\mathrm{sun}})$
**cannot live on the icosahedral quadrature** of §4 — no ordinate carries it,
and projecting it onto the ordinates would corrupt the scattering integral.
The solver therefore splits

$$
I = I_{\mathrm{diff}} + I_{\mathrm{beam}}
$$

and treats the two parts separately:

```mermaid
flowchart TD
    A["solar beam: direction omega_sun, irradiance F0"] --> B["beam_optical_depth / beam_optical_depth_octree: exact backward ray, tau_b at every cell/leaf center"]
    B --> C["I_beam = F0 * exp(-tau_b) — analytic, never on the quadrature"]
    B --> D["beam_scattering_source: omega * Phi_HG(Omega_sun -> Omega_i) * F0 * exp(-tau_b), row-normalized"]
    D --> E["fixed source term added to S in every iteration (never iterated)"]
    B --> F["beam_irradiance_ground: F0 * abs(mu_sun) * exp(-tau_b) at the ground face"]
    F --> G["Lambertian beam reflection in the ground inflow (section 7)"]
```

1. **Beam sweep (no angular quadrature).** The beam intensity at every cell
   center is the analytic

   $$
   I_{\mathrm{beam}}(\mathbf{r}) = F_0\, e^{-\tau_b(\mathbf{r})},
   \qquad
   \tau_b(\mathbf{r}) = \int_{\mathrm{entry}}^{\mathbf{r}} \kappa_e\, ds
   $$

   where the beam optical depth $\tau_b$ is accumulated along the *exact*
   backward ray from each cell center to the domain entry point:
   `rt3d.transport.beam_optical_depth` (unigrid) and
   `beam_optical_depth_octree` (octree). The beam is never transported on the
   ordinate set.

2. **First-scattering source (fixed, never iterated).** The beam is scattered
   once into the diffuse field, contributing to the source function

   $$
   S_{\mathrm{beam}}(\mathbf{r},\Omega_i)
   = \omega(\mathbf{r})\,\Phi_{\mathrm{HG}}(\boldsymbol{\Omega}_{\mathrm{sun}}\to\Omega_i)\,
     F_0\, e^{-\tau_b(\mathbf{r})}
   $$

   (`rt3d.source.beam_scattering_source`). This term is a *fixed* source
   contribution: it is added to $S$ in every iteration but never rebuilt from
   an iterated intensity, because the beam itself is converged analytically
   from the start.

3. **Beam first-scatter HG row normalization.** The phase-function row
   $\Phi_i = \Phi_{\mathrm{HG}}(\boldsymbol{\Omega}_{\mathrm{sun}}\to\Omega_i)$
   is rescaled so that $\sum_i w_i \Phi_i = 1$ exactly — the same discrete
   quadrature convention as the row-normalized scattering matrix (§4).
   Without it, the beam first-scatter power passing through a coarse ordinate
   set would carry the raw HG quadrature error (at 42 ordinates and $g = 0.952$
   the raw sum is ≈ 0.27, i.e. a factor ~3.7 error).

The ground also sees the beam: its Lambertian reflection
$A_g F_0 |\mu_{\mathrm{sun}}| e^{-\tau_{b,\mathrm{ground}}}/\pi$ is part of
the fixed ground inflow (§7).

## 4. Angular discretization — icosahedral direction grids

The direction grid is **always** an icosahedral (geodesic) grid — never a
polar/azimuthal product grid. Construction (`rt3d.directions`):

- base icosahedron (12 vertices, 20 spherical triangles),
- geodesic refinement: bisect every edge, project to the sphere, split each
  triangle into four (vectorized edge-indexed implementation; replaces the
  $O(N^2)$ dedup loops of `scripts/icosphere_grid.py`),
- $N_{\mathrm{dir}} = 10\cdot 4^{n_{\mathrm{refine}}} + 2$ →
  **12 / 42 / 162 / 642** ordinates for
  $n_{\mathrm{refine}} = 0 / 1 / 2 / 3$.

**42 ordinates ($n_{\mathrm{refine}} = 1$) is the development grid**: all
development, test cases, benchmarks and validation runs use it.
$n_{\mathrm{refine}}$ is configurable everywhere
(`icosahedric_grid(n_refine=...)`) and recorded in every output file; no code
assumes a specific $N_{\mathrm{dir}}$. Levels 162/642 belong to the phase 6
convergence study.

**Solid-angle weights** come from **spherical-triangle areas**, not $4\pi/N$:
each triangle's spherical excess (L'Huilier's formula) is split equally among
its three corner vertices (`solid_angle_weights`). Icosphere vertices are not
equal-area, so this distinction matters. The weights satisfy
$\sum_i w_i = 4\pi$ to machine precision (verification invariant 1).

**Equator guard.** On construction the grid is rotated about a generic axis
(`direction_grid._rotate_off_poles`) so that **no ordinate has exactly
$\Omega_z = 0$**. Ordinates with $\Omega_z = 0$ form cyclic characteristics
under x/y periodicity (a ray never reaches the top or bottom boundary). The
guard is $\min_i|\Omega_z| > 10^{-6}$; measured $\min_i|\Omega_z|$ is
0.42 / 0.105 / 0.034 / 0.013 for 12 / 42 / 162 / 642 ordinates.

**Precomputed HG matrix.** The scattering integral over *all* cells is one
BLAS matmul `(n_cells × N_dir) @ (N_dir × N_dir)`.
`direction_grid.hg_matrix(g)` returns
$W_{ij} = \Phi_{\mathrm{HG}}(\Omega_j\to\Omega_i)\, w_j$ (weights of the
*incoming* direction folded in).

**Row normalization and why.** The analytic HG function integrates to 1 over
the sphere, but the coarse icosahedral quadrature breaks this at large $g$:
at 42 ordinates and $g = 0.952$ the raw weighted row sums are **16.9–21.6**
(not 1). The λ-iteration scattering operator then has spectral radius ≫ 1 —
it is *supercritical* and the iteration diverges for $\omega$ not far below 1.
`hg_matrix(g, normalize=True)` (the default) rescales each row so that
$\sum_j W_{ij} = 1$ exactly. This enforces exact discrete conservation of
isotropic radiation; at $g = 0$ the raw row sums are already exactly 1, so
isotropic results are unchanged. The same convention is used for the beam
first-scatter row (§3). The price is a modified discrete phase function; the
angular error it introduces is part of the 42-ordinate error budget (§16).

**The MC family does not transport on the quadrature.** `rt3d.mc.sources`
samples the HG phase function *exactly* by inverse CDF (§9), so MC transport
has no 42-ordinate resolution limit at large $g$ and needs no delta-M; the
icosahedral grid enters the MC solvers only as **tally binning** (stage-2
radiance columns, all-sky detector cones) — and the difference between the
SC quadrature-normalized and the analytic MC beam source is a measured,
documented 42-ordinate effect (finding F14, §9 and §16).

## 5. Spatial discretization — unigrid and octree

Two spatial grids, both exposing cell/leaf centers, boxes and O(1) point
location to the transport:

**`rt3d.geometry.unigrid`** — uniform Cartesian lattice defined by
cell-center coordinate arrays $x, y, z$. Cell $(i, j, k)$ has flat index
$k\, n_y n_x + j\, n_x + i$ ($i$ fastest); the domain box extends half a cell
beyond the outermost centers. A single-cell dimension gets a nominal unit
width, so a plane-parallel slab is $n_x = n_y = 1$ with periodic sides.

**`rt3d.octree.octree_grid`** — flat struct-of-arrays octree over the cuboid,
built from cubic base cells of size $h_0$ (must divide all side lengths),
cubic cells, max-depth cap, node arrays `centers_all / half_all /
level_all / children_all`. Leaf-indexed public arrays (`centers`, `half`,
`level`, `kappa_e`, `kappa_s`, `g`, `T`) are frozen by `finalize()`.

**Refinement criteria** (`octree_grid.refine`, driven by the *input*
extinction field, sampled at the 8 cell corners):

$$
\kappa_{e,\max}\,\Delta x > \tau_{\max}
\qquad\text{(optical depth per cell)}
$$

$$
\frac{\kappa_{e,\max}-\kappa_{e,\min}}{\langle\kappa_e\rangle} > \varepsilon_{\mathrm{grad}}
\qquad\text{(cloud edges)}
$$

The gradient criterion is applied **only where the cell is not effectively
transparent**, gated by
$\kappa_{e,\max}\,\Delta x > 10^{-2}\,\tau_{\max}$.
Rationale: in the vacuum tails of a cloud the *relative* $\kappa_e$ gradient
is large but physically irrelevant — without the floor the octree would
refine the exponentially small Gaussian tails down to the depth cap.

**2:1 face balance** (`balance_21`): a leaf may differ from any face neighbor
by at most one level. Each face is probed at a **3×3 grid** of points
(transverse offsets $-h/2, 0, +h/2$), which is guaranteed to hit a deeper
neighbor of any allowed size; the loop repeats until no splits remain.

**O(1) point location** (`locate`): vectorized descent from the base lattice
(`floor` to base cell, then octant selection per level) — no per-level dense
tables.

**Why the solver uses exact characteristics on the octree.** The phase-3
reference implementation (`characteristics.build_chain_octree`) connects each
leaf to the *best-aligned center-to-center predecessor* (the octree analogue
of the unigrid lattice stencil, §6). In heterogeneous 3D fields its hop
alignment tolerance makes the **path error scale-invariant**: on the Gaussian
cloud the characteristic $\tau$ error stays ≈ 4 no matter how much the grid
is refined — refining the grid does not converge the ray path. The solver
therefore uses **exact characteristics** instead:
`transport.precompute_octree_march` marches one ray *backwards from every
leaf center along $-\boldsymbol{\Omega}$ to the domain entry point* (DDA
through the octree, `octree.march_octree`), once per ordinate, and reuses the
segments in every iteration. The only remaining path error is the octree's
own resolution of the piecewise-constant field. On the unigrid the
lattice-stencil chain is kept (its error is the documented invariant-5 effect,
§15).

> **Performance note** (plan/optimization-plan.md): with `numba` installed
> (auto-detected, `rt3d.jit`), the octree path *fuses* the DDA march with the
> formal solution — tau and source contributions are accumulated per ray in
> compiled parallel code and the segment arrays are never materialised
> (measured on a 489k-leaf octree: precompute 443 s → 5.5 s for 42 ordinates,
> 18 GB segment memory → ~0). The NumPy segment march above remains the
> reference implementation (`jit=False`), and all compiled kernels are
> bit-comparable to it (marches, unigrid chain/SC sweeps) or agree to
> transcendental rounding (fused octree sweep) — `tests/test_jit.py`.

## 6. Short-characteristics transport

> Detailed derivation, flowcharts and schematic figures for both grids:
> [`docs/sc-transport.md`](docs/sc-transport.md) (figures generated by
> `scripts/sc_transport_figures.py` into `plot/`).

One characteristic per cell/leaf center per ordinate. Along a characteristic
the medium is piecewise constant per traversed cell, and the formal solution

$$
I_c = I_u\, e^{-\tau_{\mathrm{tot}}}
\;+\; \sum_{\mathrm{sub}} S_{\mathrm{sub}}\,
       \left(1 - e^{-\tau_{\mathrm{sub}}}\right)\,
       e^{-\tau_{\mathrm{after}}}
$$

($\tau_{\mathrm{sub}}$ = optical depth of sub-segment *sub*,
$\tau_{\mathrm{after}}$ = optical depth between that sub-segment and the
target center) is **exact** for piecewise-constant sources. The code uses
`-expm1(-tau)` for $1 - e^{-\tau}$ accuracy at small $\tau$.

```mermaid
flowchart TD
    A["one ordinate Omega_i"] --> B{"grid + transport option?"}
    B -->|"unigrid, transport='chain' (default)"| C["stencil_for_direction + subsegments_uniform + build_chain (once per ordinate)"]
    B -->|"unigrid, transport='sc'"| K["build_sc_chain: single-cell SC stencil + gap/target segments (once per ordinate)"]
    B -->|"octree"| D["precompute_octree_march: exact backward DDA march from every leaf center to domain entry (once per ordinate)"]
    C --> E["sweep_direction: boundary-entry root chains, then topological BFS waves"]
    K --> L["sweep_direction_sc: layer loop along the entry axis, bilinear 4-point interpolation at the face point"]
    D --> F["sweep_direction_octree_exact: block cumsum prefixes + np.add.at"]
    E --> G["I at all cell/leaf centers for this ordinate"]
    L --> G
    F --> G
    G --> H["sweep_all / sweep_all_sc / sweep_all_octree_exact loop over all ordinates"]
```

**Unigrid (`characteristics.build_chain` + `transport.sweep_direction`).**
For one ordinate $\boldsymbol{\Omega}$ the upwind cell of $c$ is $c - s$ with
a single integer **lattice stencil** $s$ per ordinate
(`stencil_for_direction`: the offset with $s_z \neq 0$ whose lattice
direction $s\cdot h$ is closest to $\boldsymbol{\Omega}$; the search radius
auto-scales with grid anisotropy, $\sim 3\max(h)/\min(h)$). The straight path
$u_{\mathrm{center}} \to c_{\mathrm{center}}$ is split into **explicit
sub-segments** (`subsegments_uniform`) whose relative offsets and lengths are
the *same* for every cell. Since $s_z \neq 0$ always (§4 equator guard) and
$z$ is the only non-periodic dimension, every chain terminates at a
z-boundary: the dependency graph is a forest, processed in **topological BFS
waves** (`chain["waves"]`) — each wave is a handful of vectorized gathers.
Chain roots carry an exact backward march (`march_lattice`) from the root
center to the domain entry point.

**Octree (`transport.precompute_octree_march` +
`sweep_direction_octree_exact`).** Segments of each backward ray are stored
ray-contiguous (`leaf`, `length`, `offsets`, ordered *center → entry*). The
recursion is then fully vectorized: per-segment optical depths
$\tau_{\mathrm{seg}} = \kappa_e[\mathrm{leaf}]\cdot \ell$, block-wise
cumulative sums via `_block_prefixes` (a single `np.cumsum` plus per-block
offset correction — the "cumsum vectorization"), and `np.add.at` accumulation
onto leaf centers. The march depends only on grid + direction, so it is built
once per ordinate and reused by every iteration. With numba available (the
default, `jit=None`) the march and this recursion are instead **fused per ray
in compiled parallel code** (`rt3d.jit`): no segment arrays are materialised
and each sweep re-marches — cheaper overall than storing and re-reading tens
of millions of segments per ordinate.

**Unigrid single-cell SC (`rt3d.short_characteristics`,
`lambda_iteration(transport="sc")`).** A second, independent unigrid
transport — the DOIT-style single-cell short characteristics of
`plan/plan-sc-unigrid.md`. For one ordinate the upwind node is chosen on the
axis $a = \arg\max|\Omega_a|/h_a$; the characteristic into the cell center is
split into a **gap segment** from the four upwind stencil nodes to the entry
face point $p$ plus the **target segment** $p \to$ center, each integrated
with the piecewise-constant formal solution, and the four propagated node
values are combined by bilinear 4-point interpolation at $p$
(`build_sc_chain` + `sweep_direction_sc`, vectorized as a layer loop along
the entry axis). The gap segment is essential: without it the interpolation
extrapolates, the face→upwind-center half cell is never integrated, and the
transmittance error is $O(1)$ and scale-invariant. With it, z-entry
characteristics reproduce $e^{-\tau}$ to machine precision; oblique (x/y-entry)
ordinates keep a bounded angular stencil error of the same character as the
chain method's grazing-ordinate error (§15, invariant 5). The scheme is
unigrid-only (`transport="sc"` with an octree grid raises
`ValueError`); `scripts/compare_transport.py` runs chain vs sc vs octree
side by side (plan/compare-transport-plan.md).

`transport.sweep_all` / `sweep_all_octree_exact` loop over all ordinates;
the source $S$ may be direction-dependent (`(n_cells, N_dir)`), which is what
anisotropic scattering requires.

## 7. Boundary conditions

`rt3d.transport.boundary_conditions` holds constant background intensities
`top_I`, `bottom_I`, `side_I` (sides ignored when periodic) plus an optional
`ground`:

- **Top:** $I_{\mathrm{diff}} = \texttt{top\_I}$ (default 0 = space) for
  ordinates entering through $z = z_1$.
- **Bottom without ground:** $I_{\mathrm{diff}} = \texttt{bottom\_I}$.
- **Lambertian + emissive ground** (`ground_surface(albedo, emissivity, B_g,
  height=...)`): upward ordinates entering through the bottom face get the
  ground radiance

  $$
  I_{\uparrow}
  = \varepsilon_g\, B_g
  + \frac{A_g}{\pi}\left(
      F_{\downarrow}
      + F_0\,|\mu_{\mathrm{sun}}|\,e^{-\tau_{b,\mathrm{ground}}}
    \right),
  \qquad
  F_{\downarrow} = \sum_{\Omega_z < 0} w\,|\Omega_z|\, I
  $$

  where $F_{\downarrow}$ is the downward irradiance at ground level (rebuilt
  from the just-swept downward ordinates every iteration), and the last term
  is the **Lambertian reflection of the solar beam** (fixed part).
  Monochromatic convention: $B_g$ is the Planck value at the working
  frequency, given as an input like every other $B$.

  ```mermaid
  flowchart LR
      A["sweep all downward ordinates (Omega_z negative)"] --> B["downward_irradiance: F_down at ground level"]
      B --> C["assemble ground radiance: eps_g*B_g + A_g*(F_down + beam)/pi"]
      C --> D["sweep all upward ordinates (Omega_z positive) with bottom_I = ground radiance at each ray entry point"]
  ```

  In the solver loop the **downward ordinates are swept first**, the ground
  radiance is assembled, then the upward ordinates are swept with that
  per-entry-point inflow (`bottom_I=` argument of the sweeps; on the octree
  the entry leaf comes from `march["entry_cell"]`, on the unigrid the $k = 0$
  ground radiance tiles over all layers because chains with $|s_z| > 1$ root
  above the bottom layer).
- **Periodic sides:** implemented in the ray marches
  (`geometry.march_lattice`, `octree.march_octree`) as ray exit→entry wrap:
  a ray crossing a periodic x/y face is wrapped (`cuboid.wrap_xy`) and keeps
  marching; only top/bottom exits terminate it.
  *Wrap-stall bugfix note:* a ray that exits exactly (or within float noise)
  on a periodic face would be mapped by `mod()` back **onto the same face**,
  re-exit immediately with a zero-length step, and stall the march until
  `max_steps` without ever recording an exit. `march_lattice` therefore snaps
  such points to the *opposite* face (see the comment block in
  `geometry.march_lattice`).
- **Variable surface hook:** `ground_surface(height=f(x, y))` is accepted
  (`is_variable()`), but the transport supports only a flat ground at the
  domain bottom — `lambda_iteration` raises `NotImplementedError` for a
  variable surface (phase 6 hook, see §16).

The MC family (§9) implements the **same** ground law — `rt3d.mc.ground`
calls `rt3d.transport.ground_inflow`, so SC and MC share the boundary
condition by construction — with the same variable-surface guard.

## 8. Iteration scheme and 1D Ng acceleration

The solver (`rt3d.solver.lambda_iteration`) is structured like DOIT
(compute incoming radiation → transport → next iteration):

```mermaid
flowchart TD
    A["S = (1-omega)*B + S_beam — fixed parts only"] --> B["transport sweep over all ordinates (chains/marches built once)"]
    B --> C["S_new = (1-omega)*B + omega * mean(Phi I) + S_beam — incoming radiation"]
    C --> D["history: rel = max abs(S_new - S) / max abs(S_new)"]
    D --> E{"ng active?"}
    E -->|no| F["next source = S_new (plain lambda iteration)"]
    E -->|yes| G["ng1d.step: append history pair, fit Lambda_bar (Nz x Nz)"]
    G --> H{"rho(Lambda_bar) >= 1 or non-finite correction?"}
    H -->|yes, safeguard| F
    H -->|no| I["geometric-series extrapolation per z bin"]
    F --> J{"rel < tol?"}
    I --> J
    J -->|no| B
    J -->|yes| K["return I, S, J, F, omega, history, n_iter, converged"]
```

The initial source is $S^{0} = (1-\omega)B + S_{\mathrm{beam}}$ (fixed parts
only — no radiation yet). The convergence history measures
$\max|S^{\mathrm{new}} - S| / \max|S^{\mathrm{new}}|$ — the plain-LI residual
of the *swept* source, also when Ng is active.

**1D Ng acceleration (`rt3d.accelerator.ng1d`).** Ng (1982) acceleration with
a plane-averaged ($z$) estimate of the scattering operator $\bar{\Lambda}$
(plan section 5). It needs **no extra transport sweeps**: the λ-iteration
iterate differences themselves probe $\bar{\Lambda}$. With $\sigma^n$ the
source *actually swept* at iteration $n$ and
$S^{n+1} = \Lambda\,\sigma^n + Q$ the plain-LI update ($Q$ = fixed source
parts), the history pairs

$$
x^{(m)} = \sigma^{m} - \sigma^{m-1},
\qquad
y^{(m)} = \Lambda\,\sigma^{m} - \Lambda\,\sigma^{m-1}
        = S^{m+1} - S^{m}
\qquad (Q \text{ cancels})
$$

satisfy $\bar{\Lambda}\, x^{(m)} = y^{(m)}$ exactly for the 1D operator.
$\bar{\Lambda} \in \mathbb{R}^{N_z \times N_z}$ is the minimum-norm
least-squares fit of the **horizontally plane-averaged** pairs
(`plane_average`: unigrid — one bin per z layer; octree — one bin per base
cell z level, leaves weighted by their horizontal cross-section). The
accelerated source is the geometric-series extrapolation

$$
\sigma^{n+1} = S^{n+1} + \left(I - \bar{\Lambda}\right)^{-1}\bar{\Lambda}\, d^{n+1},
\qquad
d^{n+1} = S^{n+1} - \sigma^{n}
$$

applied per z bin. (Scalar check: for $\Lambda = \lambda$ this reproduces the
fixed point in one step.)

Two subtleties encoded in the code:

- **Swept-source / consecutive-update pairing.** $x$ must be the difference
  of the sources *actually swept*, and $y$ the difference of *consecutive LI
  updates* — not the naive difference of successive iterates. Once
  acceleration is active, the swept source is no longer the plain LI iterate,
  and step-difference pairing would probe $(I-A)^{-1}\Lambda$ instead of
  $\Lambda$ and break the fit.
- **Spectral-radius safeguard.** A bad fit with
  $\rho(\bar{\Lambda}) \geq 1$ would make the geometric series blow up;
  `ng1d.step` checks $\max_i |\mathrm{eig}_i(\bar{\Lambda})| \geq 1$ and falls back to plain LI for
  that step (also on non-finite corrections). Warmup: `n_start = 3` history
  pairs of plain LI before acceleration starts; at most
  `max_hist = 3·Nz` pairs are kept.

The scattering part of the source is direction-averaged in the history
(scalar RTE, HG scattering ⇒ the accelerated part is isotropic), so the
history is stored as `(n_cells,)` direction-averaged vectors.

Measured iteration counts to $\mathrm{tol} = 10^{-6}$ on the
$\tau = 5$ slab, $g = 0$ (phase 4, `scripts/phase4_ng.py`):
**$\omega = 0.5$: 22 → 10, $\omega = 0.9$: 87 → 13,
$\omega = 0.99$: 178 → 14** (LI → Ng).

### Delta-M scaling (`delta_M`)

**What it is.** For strongly forward-peaked scattering ($g \gtrsim 0.5$)
the HG lobe is unresolved on the 42-ordinate grid, the effective optical
depth is overestimated by the un-split forward peak, and the
λ-iteration scattering operator converges slowly (§4). **Delta-M
(Wiscombe) scaling** subtracts the first $M$ Legendre moments of the HG
phase function (its moments are $g^n$) from the scattering operator and
reduces the *scattering part* of the extinction accordingly:

$$
\Phi_\delta(\cos\Theta)
= \Phi_{\mathrm{HG}}(\cos\Theta)
- \frac{1}{4\pi}\sum_{n=0}^{M-1}(2n+1)\,g^n P_n(\cos\Theta),
\qquad
\kappa_{e,\delta} = \kappa_e - \kappa_s\, g^M
$$

The subtracted moments are added back into the source function exactly,
together with an explicit $-\omega g^M I$ term and the scale
$1/(1 - \omega g^M)$ (`rt3d.source.scattering_source(..., delta_M=M)`),
while all diffuse transport sweeps use $\kappa_{e,\delta}$
(`rt3d.source.delta_extinction`). The split is defined on the
row-normalized `hg_matrix` via
`direction_grid.hg_delta_matrices(g, M)`, so
`Phi_w == Phi_w_delta + sum(moments)` holds exactly as matrices —
delta-M changes the iteration and the transport scaling, **not the
converged discrete solution** (§4). The reduced diffuse extinction
shortens the effective mean free path per scattering event, so the
λ-iteration contracts far faster at large $g$.

**How to use it.** Pass `delta_M=1` to `lambda_iteration` (default
`delta_M = 0` = off). The phase runner scripts have a `DELTA_M = 0`
config cell (`scripts/phase2_slab.py`, `scripts/phase4_ng.py`,
`scripts/phase6_directions.py`). $M = 1$ (delta-Eddington) is the
recommended, validated choice; $M = 2$ is implemented and tested as
well.

**When to use it.** $g \gtrsim 0.5$ — where the HG lobe is unresolved
and plain LI suffers (the Gaussian cloud core has $g \approx 0.952$).
At small $g$ delta-M can *slow* Ng-accelerated runs (table below), so
leave it off for $g \lesssim 0.5$.

**What it does not change.**

- **The converged solution.** Delta-on and delta-off converge to the
  same discrete solution up to the $O(\tau_{\mathrm{cell}})$
  short-characteristics quadrature error of the explicit
  $-\omega g^M I$ term — measured on the $\tau = 5$ slab: max relative
  $\Delta J$ 3.9e-2 ($n_z = 8$) → 9.2e-3 ($n_z = 16$) → 2.1e-3
  ($n_z = 32$), shrinking ~linearly with cell size (agreement on the
  same solution, not an inconsistency).
- **Beam attenuation.** $\tau_b$ and the ground beam irradiance keep
  the **physical** $\kappa_e$ (the beam is not on the quadrature; the
  moment subtraction applies to the diffuse field only, §3). Only the
  first-scatter albedo factor becomes
  $\omega/(1 - \omega g^M)$ in `beam_scattering_source`.
- **Octree refinement.** Still driven by the physical $\kappa_e$ — it
  runs outside the solver (§5).
- **Verification invariants.** Invariant 3 (exact Beer–Lambert at
  $\kappa_s = 0$) holds with delta on — this is *why* only the
  scattering part of the extinction is scaled: full-extinction scaling
  $\kappa_e (1 - g^M)$ breaks invariant 3 at $O(\tau_{\mathrm{cell}})$ and
  makes plain LI diverge for $\omega < g^M$ (plan decisions log,
  `plan/delta-m-plan.md`).

Measured iteration counts on the $\tau = 5$ slab, $n_z = 16$,
$\omega = 0.9$, $\mathrm{tol} = 10^{-8}$, 42 ordinates (plain LI →
delta-M $M=1$, Ng → Ng + delta-M $M=1$):

| $g$ | plain LI | Ng |
|---|---|---|
| 0.5 | 75 → 45 | 31 → **59** (slower) |
| 0.8 | 67 → 22 | — |
| 0.9 | 69 → 14 | 90 → 15 |
| 0.95 | 65 → 11 | 82 → 11 |

($M = 2$ at $g = 0.9$: 77 → 23 plain LI; the regression test
`tests/test_delta_m.py::test_solver_acceleration_delta_M1` at
$n_z = 8$, $\mathrm{tol} = 10^{-6}$, $g = 0.95$: 62 → 12.)

## 9. Monte Carlo solver family

The `rt3d.mc` package (plan/mc-plan.md, phases M0–M6) transports the same
problem — same grids, same `boundary_conditions` + `ground`, same
$\kappa_e/\kappa_s/g$ conventions — by **Woodcock null-collision particle
transport** instead of ordinate sweeps. Transport is **angle-exact** (HG
inverse-CDF; no quadrature in the collision physics), every run is seeded
(PCG64) and carries 1-$\sigma$ batch error bars, and every solver is
differenced cell-by-cell against the SC family in the test matrix (§15).

```mermaid
flowchart TD
    A1["solver A stage 1 — j_mc(stage1='lambda' or 'explicit'): diffuse J, F by particle transport"] --> S2["stage2_source(J) = (1-omega)B + omega J + beam first scatter"]
    S2 --> A2["solver A stage 2 — radiance_from_source: I(cell, ordinate) on the SC ray set"]
    A1 -.->|"independent closed-RTE solve, no stage-1 J"| B["solver B — backward_detector_mc / backward_all_sky: adjoint radiance at a detector"]
    A2 --> X["differencing vs lambda_iteration (matrix tiers T2/T3)"]
    B --> X
```

**Tracking core (`rt3d.mc.tracking`, M0).** One flight sampler and event
decision shared by every solver: sample a distance from a majorant
$\kappa_{\max} \ge \kappa_e$, DDA through the grid, then `event_decision`
classifies the collision as scatter / true absorption / **null** (null:
direction and weight unchanged, flight continues). Two majorant modes:
`mode="cell"` (default) sets $\kappa_{\max} = \kappa_e(\text{cell})$ — on a
piecewise-constant grid every collision is real, but the code stays a
Woodcock tracker; `mode="global"` uses a domain-wide majorant so vacuum
regions produce *real* null collisions and the whole photon batch advances
one vectorized flight at a time. The DDA walkers (`walk_lattice`,
`walk_octree`) reproduce the SC marches `geometry.march_lattice` /
`octree.march_octree` **segment-for-segment** (same lattice formulas, same
periodic-wrap snap), so MC and SC see exactly the same geometry.

**Sources and sampling (`rt3d.mc.sources`, M0/M1).** The HG phase function
is sampled *exactly* by inverse CDF:

$$
q = \frac{2\,g\,u}{1-g^2} + \frac{1}{1+g},
\qquad
\cos\Theta = \frac{1+g^2-q^{-2}}{2\,g},
\qquad
\text{azimuth} \sim U[0,2\pi),
$$

monotone in $u$ for either sign of $g$, isotropic limit at $g = 0$. This is
the companion of verification invariant 2: HG samples reproduce the
`hg_matrix` quadrature moments to quadrature accuracy
(`tests/test_mc_sources.py`). Birth positions from inhomogeneous source
rates come from `sample_births` (categorical over cells, or chained *null
source events* — provably the same density); face populations use
cosine-weighted inward directions (`face_inflow_directions`, the
crossing-rate measure $|\mu|\,d\Omega$); the beam seeds on its entry face
(`beam_seed`).

**Tallies and error bars (`rt3d.mc.estimators`).** Track-length tally of
the fluence,

$$
J_c = \frac{1}{4\pi V_c}\sum_k w_k\,\ell_{c,k},
\qquad
F_c = \frac{1}{V_c}\sum_k w_k\,\ell_{c,k}\,\boldsymbol{\Omega}_k,
$$

exact for a particle population killed at true absorption. Weights are
*powers* (intensity × area): face source $w = \pi A\,I_{\mathrm{in}}/n$,
beam $w = F_0\,|\boldsymbol\Omega_{\mathrm{sun}}\cdot
\mathbf n_{\mathrm{face}}|\,A/n$, volume source $w = 4\pi Q/n$ — the tally
then carries intensity units. Histories are grouped into `n_batches`
contiguous batches and $\sigma$ is the batch-mean standard error
(`ddof=1`); the plan's acceptance bands are 2σ. Note finding **F3**:
*per-cell* MC z-scores are heavy-tailed — the matrix tests use pooled
statistics (`tests/matrix_stats.py`); never read a single-cell z-score as a
test.

**Solver A stage 1 (`rt3d.mc.forward.j_mc`, M1).** Two variants of the same
physics:

- `stage1="lambda"` (default) — the DOIT twin of `lambda_iteration`: the
  transport operator is applied to the current scalar source, particles fly
  with *removal only* (the scattering is done by the outer iteration),
  $S^{n+1} = (1-\omega)B + \omega J^n +$ fixed boundary/beam terms; same
  history/tol bookkeeping, and `ng=True` applies the same 1D Ng
  accelerator. **Convergence caveat (finding F5):** with fresh sampling per
  iteration the residual stalls at the tally-noise floor (`history[-1]` ≈
  0.2 on the cloud octree) — validate this variant on the agreement of its
  $J$ field, never on `converged`. `crn=True, birth="uniform"` freezes the
  birth stream (common random numbers) but can diverge on inhomogeneous
  fields.
- `stage1="explicit"` — classic generation-based analog MC
  (`rt3d.mc.tracking.track_histories`): physical scattering regenerates the
  diffuse field inside the history, no outer iteration; the unscattered
  beam is tallied into a separate `J_direct`, so `J` stays the *diffuse*
  mean intensity like the SC field.

The beam is a **particle stream** seeded on its entry face (plus first-
scatter children in the λ variant), not an analytic source term;
`rt3d.source.beam_scattering_source` and $\tau_b$ stay available as
analytic cross-checks. The result dict mirrors the SC one — `J, sigma_J,
F, sigma_F, heating, history, n_iter, converged` — plus traceability
(`seed`, `n_histories`, `n_null`, `n_collisions`, …).

**Solver A stage 2 (`radiance_from_source` + `stage2_source`, M2).**
Radiance $I(\text{cell},\Omega_i)$ of a *fixed* source on a ray set
identical to the SC sweeps — one backward ray per cell/leaf center per
icosahedral ordinate:

- `estimator="next_flight"` (default): `n_histories` backward flights per
  ray score $S_{\text{total}}$ at their first *real* collision (nulls
  continue); unbiased in both majorant modes, no $/\kappa_{\max}$ factor.
- `estimator="deterministic"`: the exact per-segment line integral — the
  zero-noise limit, and on the octree **identical to the SC sweep to
  1.2e-12** (tier T2, finding F2). This is the hinge that separates
  transport discretization from statistical noise in every SC-vs-MC
  comparison.

**Solver B — adjoint detector MC (`rt3d.mc.backward`, M3).** Answers "what
reaches this detector": adjoint histories fly *backwards* from a point or
plane-aperture detector (`detector = {"point": ...}` or
`{"plane": {...}}` + `omega0` + `cone` + `n_seeds`) along $-\Omega$; at
every real collision of the closed-RTE walk a history scores

$$
w\left[(1-\omega)B(\mathbf r)
+\omega\,F_0\,e^{-\tau_b(\mathbf r)}\,
\Phi_{\mathrm{HG}}(\boldsymbol\Omega_{\mathrm{sun}}\to\boldsymbol\Omega_f)\right],
$$

with the *forward* intensity direction $\boldsymbol\Omega_f$ (HG
reciprocity: the adjoint kernel is the same HG about the reversed
direction); the walk continues at scattering and terminates at true
absorption, and scores the face inflow where it exits the domain. No
stage-1 $J$ is used — solver B is an **independent solve of the closed
problem** (the plan sketch's "$\omega J$ at every collision" would
re-expand the already-scattered field at all orders and over-count —
deviation logged in `plan/plan.md`). `backward_all_sky` runs one detector
per upward icosahedral ordinate; keep `cone_scale` small (default 0 =
exact node direction, the mode that differences against `I[cell,
ordinate]`): full equal-area caps cross the horizon near $\mu \approx 0$
and mix in downward radiance (measured 0.3 vs 0.02 at $\mu \approx 0.1$).

**Ground (phase M6, `rt3d.mc.ground`).** `ground_radiance` calls
`rt3d.transport.ground_inflow` — SC and MC share the ground law **by
construction**. The MC side adds the sampling machinery: ground-emission
and reflected-beam face populations seeded from the per-column radiance
(`sample_ground_face`), $F_\downarrow$ measured from flight segments
(`down_irradiance_from_segments`, the twin of
`transport.downward_irradiance`), and albedo feedback either as analog
boundary scattering in the explicit chain or as a re-seeded iterated
inflow in the λ variant (the exact twin of the SC "downward ordinates
first" ordering). The adjoint walk scores the ground source and continues
cosine-weighted upward with $\times A_g$.

**Variance reduction and the numba path (M4).** `implicit_capture=True`
never terminates a walk at absorption — every real collision multiplies
the weight by the collision albedo $\omega$ (unchanged expectation,
measured $\sigma$ gain ≈ 3× at $\omega = 0.5$, ≈ 1.8× at $\omega = 0.9$);
`roulette_floor` (Russian roulette) requires it. The compiled per-photon
fast path is `rt3d.mc.jitmc.sample_collisions_jit`, selected with
**`jit=True` — opt-in here**, unlike the SC family's auto-default
`jit=None`: it is statistically equivalent but uses a *different RNG
stream* (numba's generator, not PCG64), so a seeded `jit=True` run does
not reproduce the NumPy-path numbers.

**Measured cross-family facts** (findings log in
`plan/solver-test-matrix-plan.md`): the T2 identity above; **F11** — the
unigrid `chain` transport carries an $h$-independent path error (volume-
mean $J$ +3.1% over the exact-march value), which MC-vs-SC comparisons
must not blame on the particles; **F14** — SC's quadrature-normalized beam
first-scatter source sits a factor $1/s(g)$ below the MC analytic-HG one at
large $g$ (raw quadrature defect $s(g) = 4.0$–8.9 at $g = 0.9$–0.925 on 42
ordinates), so beam-case per-ordinate SC-vs-MC comparisons on the
development grid are F14-limited, not noise-limited; wall-time
expectation (M4 table, `scripts/mc_phase4_walltimes.py`): SC wins
whole-domain stage-1 fields, MC wins stage-2/all-sky-at-a-point queries.

---

# Part II — Usage

## 10. Environment setup

Mamba/conda environment **`pyarts`**. There is no `mamba`/`conda` on `PATH`
in a plain shell — call the environment interpreter by absolute path:

```sh
# macOS / Linux
~/miniforge3/envs/pyarts/bin/python <script.py>
```

```powershell
# Windows
& "$env:USERPROFILE\miniforge3\envs\pyarts\python.exe" <script.py>
```

> **⚠ BLAS/openblas PATH gotcha — Windows only, read this first.**
> On Windows the env's numpy links against the `libcblas.dll` shim, which
> *delay-loads* `openblas.dll` from `...\envs\pyarts\Library\bin`. Running
> the interpreter by absolute path **without activating the env** leaves
> that directory off `PATH`, and every matrix product then hard-crashes the
> process with exit code **`-1066598273`** (`0xC06D007F`, delay-load
> failure) — **no traceback**. Prepend it before running anything that does
> a matmul (which is everything in `rt3d`):
>
> ```powershell
> $env:Path = "$env:USERPROFILE\miniforge3\envs\pyarts\Library\bin;" + $env:Path
> ```
>
> On macOS/Linux the absolute interpreter path works as-is — no `PATH` fix
> needed.

Installed packages (reference versions — they may differ slightly between
machines; confirm with the sanity check below):

| Package | Version |
|---|---|
| Python | 3.14.6 |
| numpy | 2.5.2 |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| xarray | 2026.7.0 |
| h5py | 3.16.0 |
| pyarts3 | 3.0.0.dev10 |
| pytest | 9.1.1 |
| numba | 0.66.0 (+ llvmlite 0.48.0) |
| pyvista | 0.49.1 (with vtk) + trame-pyvista 0.2.1 |
| markdown | 3.11.1 (md → HTML for the browser print-to-PDF doc regeneration) |

`numba` drives the compiled fast paths of `rt3d.jit` (SC) and
`rt3d.mc.jitmc` (MC): auto-detected (`rt3d.jit.HAVE_NUMBA`), with a
bit-identical NumPy reference fallback for every kernel (§16). `pyvista` is
optional (3D figures in §12, HTML scene export via trame). The solver core
itself needs only numpy (+ scipy in `ng1d`); never add a new dependency
without installing it into `pyarts` first.

Quick sanity check:

```sh
# macOS / Linux
~/miniforge3/envs/pyarts/bin/python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

Running the tests — **341 tests** in `tests/` (per-module suites plus the
cross-solver matrix tiers T0–T5, §15; `pytest.ini` defines the markers
`slow`, `mc`, `reference`):

```sh
# macOS / Linux
~/miniforge3/envs/pyarts/bin/python -m pytest -q -m "not slow"  # 330 tests, ≈ 29 min measured (macOS/arm64)
~/miniforge3/envs/pyarts/bin/python -m pytest -q                # adds the 11 slow-tier tests (ladder, fine reference)
```

```powershell
# Windows (PATH fix first!)
& "$env:USERPROFILE\miniforge3\envs\pyarts\python.exe" -m pytest -q
```

Per-tier runtime budgets (measured/extrapolated, plan/solver-test-matrix-plan.md
§10): T0+T1+T2 ≈ 8–12 min, T3+T4 ≈ 15–25 min, the `slow` T5 ladder ≈ 30–50 min;
`-m "slow or reference"` extends the full run to ≈ 1.5–2 h.
`tests/conftest.py` puts the repo root on `sys.path`, so `import rt3d` works
from the repo root in scripts and tests alike. Plots must not block: use
`matplotlib.use("Agg")` and `fig.savefig("plot/...")`; figure output goes to
`plot/`.

## 11. Module tour

One section per `rt3d` module, with the real signatures.

### `rt3d.directions` — direction grids and HG matrices

- `make_icosahedric_grid() -> (vertices (12,3), triangles (20,3))` — base
  icosahedron (same construction as `scripts/icosphere_grid.py`).
- `refine_icosahedric_grid(vertices, triangles, n_refine=1) -> (vertices, triangles)`
  — vectorized geodesic refinement (edge-indexed midpoint table; exact, fast
  enough for $n_{\mathrm{refine}} = 3$).
- `solid_angle_weights(vertices, triangles) -> weights (N,)` — per-vertex
  solid angles from spherical-triangle areas (L'Huilier);
  $\sum w = 4\pi$.
- `hg(cos_theta, g) -> Phi` — HG phase function,
  $\oint\Phi\,d\Omega = 1$ analytically.
- `class direction_grid(n_refine=1)` — attributes `n_refine`,
  `directions (N_dir,3)`, `weights (N_dir,)`, `triangles`,
  `down_idx` / `up_idx` (indices with $\Omega_z < 0$ / $> 0$; a partition,
  since no ordinate has $\Omega_z = 0$), property `n_dir`.
  Methods: `cos_theta() -> (N_dir,N_dir)`;
  `hg_matrix(g, with_weights=True, normalize=True) -> W` with
  $W_{ij} = \Phi_{\mathrm{HG}}(\Omega_j\to\Omega_i)\,w_j$ and unit row sums
  when `normalize=True` (§4);
  `hg_delta_matrices(g, M) -> (Phi_w_delta, moments)` — exact delta-M
  split of the row-normalized matrix,
  `moments[n][i, j] = (2n+1) g^n P_n(Omega_i . Omega_j) w_j / (4 pi)`,
  `Phi_w == Phi_w_delta + sum(moments)` (§8).
- `icosahedric_grid(n_refine=1) -> direction_grid` — the single place the
  default $n_{\mathrm{refine}} = 1$ (42 ordinates) appears.

### `rt3d.geometry` — domain, unigrid, ray marching, ground

- `class cuboid(x0, x1, y0, y1, z0, z1, periodic_x=True, periodic_y=True)` —
  properties `Lx, Ly, Lz`; `contains(points)`, `wrap_xy(points)`.
- `class unigrid(x, y, z)` — cell-center arrays; attributes `hx, hy, hz`,
  `nx, ny, nz, n_cells`, `centers (n_cells,3)` (flat-index order, $i$
  fastest), `box`. Methods `cell_box(idx)`, `unravel(idx)`, `ravel(i,j,k)`,
  `locate(points)` (O(1), clamped).
- `ray_box_exit(origin, direction, box) -> (t_exit, exit_axis, exit_side)` —
  slab-method exit distance/face for points inside a box.
- `march_lattice(origin, direction, grid, max_steps=None, periodic_box=None)
  -> (steps, exit_t, exit_axis, exit_side)` — vectorized Amanatides–Woo DDA
  of K rays sharing one direction through the unigrid; `steps` is
  step-major `(ray_idx, cell_idx, length)`; with `periodic_box`, x/y exits
  wrap and only z exits terminate (§7 incl. wrap-stall fix).
- `class ground_surface(albedo=0.0, emissivity=1.0, B_g=0.0, height=0.0)` —
  flat Lambertian + emissive ground; `height(x, y)` / `is_variable()` are
  the variable-surface hook (transport: flat only, §7).

### `rt3d.octree` — octree grid and DDA march

- `class octree_grid(cuboid, h0)` — cubic base cells, $h_0$ must divide all
  side lengths. Build: `refine(kappa_e_fn, tau_max=0.5, eps_grad=0.3,
  max_depth=6)`, `balance_21(max_depth=6)`, `finalize()` (freezes leaf
  arrays `centers, half, level, n_leaf` and allocates `kappa_e, kappa_s, g,
  T` slots); `sample_fields(kappa_e=…, kappa_s=…, g=…, T=…)` samples
  callables $(x,y,z) \mapsto$ array at leaf centers; `locate(points)`
  vectorized descent; `split(nodes)`, `leaves_all()`.
- `march_octree(origin, direction, oct, path_len, periodic_box=None) ->
  (seg_ray, seg_leaf, seg_len)` — vectorized DDA, per-ray directions and
  per-ray path-length budgets, segments in traversal order per ray.
- `node_major_segments(seg_ray, seg_leaf, seg_len, n_nodes) ->
  (seg_ray, seg_leaf, seg_len, offsets)` — regroup step-major segments into
  node-major contiguous blocks (stable sort preserves traversal order).

### `rt3d.characteristics` — predecessor chains (unigrid + octree reference)

- `stencil_for_direction(omega, h, max_offset=None) -> (s, vhat)` — integer
  lattice stencil per ordinate (§6); $s_z \neq 0$; automatic search radius
  $\min\!\big(24, \max(3, \lceil 3\max(h)/\min(h)\rceil)\big)$.
- `subsegments_uniform(s, h) -> (rel (M,3), lengths (M,))` — exact
  sub-segments of the center-to-center path on the lattice.
- `build_chain(grid, omega, periodic_box=None, max_offset=None) -> chain` —
  unigrid chain dict: `s`, `vhat`, `parent`, `roots`, `waves`,
  `rel (M,3) int16` + `sub_delta (M,) int64` + `periodic` (the sub-segment
  cells of a non-root cell are `cell + rel[m]`, flat index `wave +
  sub_delta[m]`; replaced the old `(M, n_cells)` int64 table, which cost
  ~1.5 GB at 100³), `sub_len`, `root_steps` = `(ridx, cell, length)` arrays
  (step-major backward march segments from root centers to entry),
  `root_exit_t`, `root_exit_axis`, `root_exit_side`. Used by the solver on
  the unigrid.
- `build_chain_octree(oct, omega, periodic_box=None, cos_min=0.9,
  max_hops=8) -> chain` — **reference only, not used by the solver** (§5):
  best-aligned center-to-center connection with cycle peeling; kept for
  phase-3 validation (`scripts/phase3_octree.py` cell 1).

### `rt3d.short_characteristics` — unigrid single-cell SC (`transport="sc"`)

Second, independent unigrid transport (§6, plan/plan-sc-unigrid.md); the
solver selects it with `lambda_iteration(transport="sc")`.

- `sc_stencil(omega, h) -> dict` — entry axis $a = \arg\max|\Omega_a|/h_a$,
  propagation sense `sigma`, face-segment length `t_f = h_a/(2|\Omega_a|)`,
  the 4 upwind bilinear node `offsets` and `weights` (sum 1).
- `build_sc_chain(grid, omega, periodic_box=None) -> sc` — per-ordinate
  chain dict (`a`, `sigma`, `t_f`, `weights`, layer-grouped `order` +
  `slices`, `node_idx`, `node_face`, `wrap`, `n_layers`), built once per
  ordinate like `build_chain`.
- `sweep_direction_sc(grid, sc, kappa_e, S, bc, bottom_I=None,
  wrap_tol=1e-12, max_passes=None) -> I (n_cells,)` — one ordinate sweep as
  a layer loop along the entry axis (§6); on periodic x/y entry axes the
  layer loop runs ring passes until the wrap boundary converges
  (`max_passes` cap, default `max(4 n_a, 64)` — the bilinear stencil's
  ring operator needs more than $n_a$ passes for grazing ordinates in
  near-vacuum rings).
- `sweep_all_sc(grid, direction_grid, kappa_e, S, bc, periodic_box=None,
  chains=None, n_jobs=1) -> (I, chains)` — all ordinates; compiled via
  `rt3d.jit.sc_sweep_jit` when numba is available.

### `rt3d.transport` — sweeps, boundary conditions, beam

- `class boundary_conditions(top_I=0.0, bottom_I=0.0, side_I=0.0, ground=None)`
  — face backgrounds + optional ground (§7); `inflow(exit_axis, exit_side)`.
- `bottom_mask(grid) -> bool array` — cells/leaves touching the bottom face.
- `downward_irradiance(I, dg) -> F_down` —
  $\sum_{\Omega_z<0} w\,|\Omega_z|\, I$ per cell (`I` restricted to
  `dg.down_idx` order).
- `ground_inflow(F_down, ground, beam_irr=0.0) -> I_up` —
  $\varepsilon_g B_g + A_g (F_{\downarrow} + \text{beam}_{\mathrm{irr}})/\pi$.
- `beam_irradiance_ground(grid, omega_sun, kappa_e, tau_b, F0, periodic_box=None)`
  — $F_0|\mu_{\mathrm{sun}}|e^{-\tau_b}$ extended through the bottom cell to
  the ground face.
- `sweep_direction(grid, chain, kappa_e, S, bc, bottom_I=None) -> I (n_cells,)`
  — one unigrid ordinate sweep (recursion of §6).
- `sweep_all(grid, direction_grid, kappa_e, S, bc, periodic_box=None,
  chains=None, n_jobs=1) -> (I (n_cells,N_dir), chains)` — all ordinates;
  pass prebuilt `chains` to reuse across iterations. With numba available
  each ordinate sweeps via the compiled `rt3d.jit.chain_sweep_jit` kernel
  (bit-identical to `sweep_direction`); `n_jobs > 1` runs the kernels on a
  thread pool (njit releases the GIL).
- `beam_optical_depth(grid, omega_sun, kappa_e, periodic_box=None) -> tau_b`
  — exact backward-beam $\tau$ at cell centers (unigrid); raises unless
  $\Omega_{\mathrm{sun},z} < 0$.
- `precompute_octree_march(oct, omega, periodic_box=None, jit=None) -> march`
  — exact backward march per leaf center for one ordinate (§6). `jit=None`
  (default) uses the fused numba path when available: dict `fused=True`,
  `omega`, `budget`, `entry_cell`, `periodic_box` — no segment arrays, the
  sweep re-marches in compiled code. `jit=False` gives the reference
  segment dict: `leaf`, `length`, `offsets`, `ray_of`, `omega`,
  `entry_cell` (bottom-face leaf of each upward ray's ground entry point,
  for the ground BC).
- `sweep_direction_octree_exact(oct, march, kappa_e, S, bc, bottom_I=None) -> I`
  — one octree ordinate sweep along exact characteristics (fused numba
  march+sweep when `march["fused"]`).
- `sweep_all_octree_exact(oct, direction_grid, kappa_e, S, bc,
  periodic_box=None, marches=None, n_jobs=1) -> (I, marches)` — all
  ordinates. **This is the octree transport used by the solver.**
- `sweep_direction_octree(oct, chain, kappa_e, S, bc)` and
  `sweep_all_octree(...)` — sweeps along `build_chain_octree` chains
  (reference only).
- `beam_optical_depth_octree(oct, omega_sun, kappa_e, periodic_box=None,
  jit=None) -> tau_b`.

### `rt3d.jit` — numba fast paths (plan/optimization-plan.md)

All kernels reproduce their NumPy reference operation-for-operation and
degrade to it when numba is missing (`rt3d.jit.HAVE_NUMBA`); parity is
locked by `tests/test_jit.py`.

- `march_octree_jit(origin, direction, oct, path_len, periodic_box=None,
  merge=False)` — per-ray compiled DDA, ray-major segments, bit-comparable
  to `octree.march_octree`.
- `sweep_direction_octree_exact_jit(oct, omega, budget, kappa_e, S, I_in,
  periodic_box=None)` — fused march + formal solution, parallel over rays,
  no segment arrays.
- `beam_optical_depth_octree_jit(...)`, `locate_jit(points, oct)` — fused
  beam tau; parallel point location (also used by `balance_21`).
- `chain_sweep_jit(grid, chain, kappa_e, S_d, bc, bottom_I=None)`,
  `sc_sweep_jit(grid, sc, kappa_e, S_d, bc, ...)` — compiled unigrid
  sweeps, bit-identical to `sweep_direction` / `sweep_direction_sc`;
  used automatically by the solver and the `sweep_all*` functions.
- `octree_grid.march_arrays()` — cached plain-array node tables + scalar
  parameters consumed by the kernels.

### `rt3d.source` — source function assembly

- `single_scattering_albedo(kappa_e, kappa_s) -> omega` — with the
  $\kappa_e \to 0$ guard (§2).
- `mean_intensity(I, dg) -> J` — $J = \frac{1}{4\pi}\oint I\,d\Omega$.
- `flux_vector(I, dg) -> F (n_cells,3)` — $F = \oint I\,\boldsymbol{\Omega}\,d\Omega$.
- `delta_extinction(kappa_e, kappa_s, g, M) -> kappa_e - kappa_s * g**M`
  — delta-M transport extinction (only the scattering part scaled;
  `M = 0` returns `kappa_e` unchanged) (§8).
- `scattering_source(I, dg, omega, B, Phi_w=None, g=0.0, beam_scat=None,
  g_cells=None, g_decimals=2, Phi_cache=None, delta_M=0) -> S (n_cells,N_dir)` —
  $S_{ic} = (1-\omega_c)B_c + \omega_c \sum_j w_j \Phi(\Omega_j\to\Omega_i) I_{cj} + \text{beam\_scat}$.
  Spatially varying $g$ (`g_cells`) is bucketed by `round(g, g_decimals)`
  with one BLAS matmul per bucket and an optional `{g: Phi_w}` `Phi_cache`
  (§2/§4). With `delta_M >= 1` the delta-M form is used instead —
  `Phi_w_delta` + moment operators + explicit $-\omega g^M I$ term, scaled
  by $1/(1 - \omega g^M)$; splits are cached under the tuple key
  `(g_value, delta_M)` (§8).
- `beam_scattering_source(dg, omega_sun, F0, tau_b, omega, g, g_cells=None,
  delta_M=0) -> S_beam (n_cells,N_dir)` — fixed beam first-scatter term
  with normalized HG row (§3); with `delta_M >= 1` the albedo factor
  $\omega$ becomes $\omega/(1 - \omega g^M)$, $\tau_b$ stays physical (§8).

### `rt3d.accelerator` — Ng acceleration

- `class ng1d(grid, max_hist=None, n_start=3)` — 1D Ng state (§8); methods
  `plane_average(v)`, `lambda_bar()` ($N_z\times N_z$ least-squares fit),
  `step(S_swept, S_new) -> next source`.

### `rt3d.solver` — the iteration loop

- `lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, g=0.0,
  periodic_box=None, beam=None, n_iter=100, tol=1e-8, ng=None,
  delta_M=0, transport="chain", jit=None, n_jobs=1) -> result`
  — the entry point. `grid` is a `unigrid` or `octree_grid`; $g$ scalar or
  `(n_cells,)`; `beam = {"omega_sun": (3,) unit, "F0": float}` or `None`;
  `ng` is `None`/`False` (plain LI), `True` (build `ng1d(grid)`) or a
  prepared `ng1d`; `delta_M = 0` off, `>= 1` delta-M scaling (§8);
  `transport` selects the **unigrid** sweep scheme (§6): `"chain"` (the
  lattice-stencil chain, default) or `"sc"` (single-cell SC; raises with an
  octree grid);
  `jit=None` auto-selects the fused numba octree march (§5 performance
  note); `n_jobs > 1` sweeps ordinates in parallel threads
  (plan/optimization-plan.md P3).
  Returns the result dict:

  | key | shape / type | meaning |
  |---|---|---|
  | `I` | `(n_cells, N_dir)` | converged center intensities (diffuse field) |
  | `S` | `(n_cells, N_dir)` | converged source function |
  | `J` | `(n_cells,)` | mean intensity |
  | `F` | `(n_cells, 3)` | flux vector |
  | `omega` | `(n_cells,)` | single scattering albedo |
  | `history` | list of float | relative source change per iteration |
  | `n_iter` | int | iterations performed |
  | `converged` | bool | `history[-1] < tol` |
  | `delta_M` | int | delta-M order used (0 = off) |
  | `kappa_e_delta` | `(n_cells,)` | extinction used for the diffuse sweeps; equals `kappa_e` when `delta_M = 0` (§8) |
  | `transport` | str | unigrid transport used (`"chain"` or `"sc"`) |
  | `jit` | bool | fused numba octree march/sweep used (octree only) |
  | `n_jobs` | int | thread count used for the per-ordinate sweeps |

  Note there is **no `SolverConfig`** class (that name exists only in
  `plan/plan.md`); all configuration is via the keyword arguments above.

### `rt3d.mc` — Monte Carlo solver family (theory §9, plan/mc-plan.md)

- `rt3d.mc.tracking`:
  `sample_collisions(starts, directions, budgets, grid, kappa_e, kappa_s,
  g=0.0, mode="cell", kappa_max=None, periodic_box=None, seed=0,
  scatter=True, max_collisions=20000, path_cap=None, jit=False)` — the
  shared Woodcock flight sampler: per-history collision records
  `(hist_idx, cell_idx, length)` + collision type/direction arrays;
  `scatter=False` implements the removal-only transport operator of the
  λ-variant. `walk_lattice(starts, directions, grid, budgets=None,
  periodic_box=None)` / `walk_octree(starts, directions, oct, budgets,
  periodic_box=None)` — pure geometric DDA traversals (no collisions),
  segment-for-segment identical to the SC marches. Helpers:
  `flight_distance(u, kappa_max)`, `event_decision(u, kappa_max, kappa_s,
  kappa_e)`, `flight_budget(p, d, grid, periodic_box=None, path_cap=None)`,
  `flight_segments(starts, directions, ends, grid, periodic_box=None)`,
  `path_integral_from_segments(seg_ray, seg_cell, seg_len, cell_field,
  n_rays)`; event codes `EV_EXIT/EV_SCATTER/EV_ABSORB/EV_NULL`.
  `track_histories(starts, directions, grid, kappa_e, kappa_s, tally,
  ..., g, mode, implicit_capture, roulette_floor, jit, ground, ...)` —
  analog generation chain (stage1="explicit", §9).
- `rt3d.mc.sources`: `hg_cos_theta(u, g)` / `hg_directions(d_in, u_cos,
  u_az, g)` / `hg_scatter_directions(d_in, cells, g, rng)` — HG inverse-CDF
  (§9 formula); `isotropic_directions(n, rng)`,
  `face_inflow_directions(n, axis, side, rng)` (cosine-weighted inward),
  `face_points(n, box, axis, side, rng, eps=None)`, `face_area(box, axis)`,
  `sample_births(n, grid, rate, method="categorical"|"null_collision",
  seed=0)` — birth positions from a per-cell rate, with the null-source-
  event count returned as diagnostic; `beam_seed(n, box, omega_sun, rng)`.
- `rt3d.mc.estimators`: `class track_tally(n_cells, n_histories,
  n_batches=100, flux=True)` — track-length accumulator: `add(hist, cell,
  length, weight, direction=None)` deposits a flight's segments,
  `merge(other)` combines chains, `result(volumes)` returns
  `J/sigma_J` (and `F/sigma_F`) with history-batch error bars (§9);
  `cell_volumes(grid)`,
  `n_cells_of(grid)`; `heating_rate(kappa_a, B, J)` — **argument order
  swapped vs `rt3d.diagnostics.heating_rate(kappa_a, J, B)`** (finding F8
  in the findings log — do not swap by reflex).
- `rt3d.mc.forward`:
  `j_mc(grid, kappa_e, kappa_s, B, bc, g=0.0, periodic_box=None, beam=None,
  stage1="lambda", n_histories=20000, n_boundary_histories=20000,
  n_beam_histories=20000, n_ground_histories=None, n_batches=100, seed=0,
  mode="cell", kappa_max=None, n_iter=200, tol=1e-8, ng=None, crn=False,
  birth="source", max_collisions=20000, path_cap=None, tally_flux=True,
  implicit_capture=False, roulette_floor=None, jit=False) -> result` —
  stage 1 (§9); result keys `J, sigma_J, F, sigma_F, heating, history,
  s_max_history, n_iter, converged, J_direct, J_direct_analytic, J_total,
  J_fixed, n_null, n_null_source, n_generations, n_collisions, capped,
  n_truncated` + traceability (`stage1, mode, seed, n_histories, ...`).
  `stage2_source(grid, dg, kappa_e, kappa_s, B, J, g=0.0, beam=None,
  periodic_box=None)` — fixed source `(n_cells,)` or `(n_cells, N_dir)`
  with beam; `radiance_from_source(grid, dg, kappa_e, S_total, bc,
  periodic_box=None, estimator="next_flight"|"deterministic",
  n_histories=25, n_batches=10, seed=0, mode="cell", kappa_max=None,
  jit=False, ground_I=None) -> {"I", "sigma_I", "J", ...}` — stage 2 (§9).
- `rt3d.mc.backward`:
  `backward_detector_mc(grid, kappa_e, kappa_s, B, bc, detector, g=0.0,
  beam=None, tau_b=None, periodic_box=None, n_batches=100, seed=0,
  mode="cell", kappa_max=None, max_collisions=20000, path_cap=None,
  implicit_capture=False, roulette_floor=None, jit=False)` — adjoint
  radiance at one detector (`detector` spec in §9), returns `I`/`sigma_I`
  + walk statistics; `backward_all_sky(..., point, n_seeds=20000,
  cone_scale=0.0, ...)` — one detector per upward ordinate, returns
  per-ordinate `I/sigma_I` on `up_idx`; `all_sky_cone_half(dg, i)` —
  equal-area cap half-angle of ordinate $i$.
- `rt3d.mc.ground` (M6): `bottom_columns(grid)`,
  `ground_radiance(grid, ground, F_down=None, beam_irr=0.0)` (calls
  `rt3d.transport.ground_inflow`), `ground_beam_irradiance(...)`,
  `sample_ground_face(n, grid, I_col, rng)`, `ground_face_weight(...)`,
  `down_irradiance_from_segments(rec, weights, cols)`,
  `ground_column_lookup(grid, points, cols)`, `ground_or_raise(bc)`.
- `rt3d.mc.jitmc`: `jit_mc_usable(grid, periodic_box=None)`,
  `sample_collisions_jit(...)` — compiled clone of `sample_collisions`
  (statistically equivalent, different RNG stream, §9).

### `rt3d.diagnostics` — diagnostics

- `heating_rate(kappa_a, J, B)` — net heating rate per volume,
  $4\pi\kappa_a (B - J)$ (positive = radiation gains energy from matter).
  (The MC twin `rt3d.mc.estimators.heating_rate` takes `(kappa_a, B, J)`.)
- `cell_volumes(grid) -> (n_cells,)` — `hx hy hz` on the unigrid,
  `(2 half)^3` per octree leaf (the leaves tile the domain).
- `flux_divergence(grid, F) -> per-cell div F` — central differences,
  periodic wrap in x/y, one-sided at top/bottom; **works on both grids**
  (octree: side-leaf location along each axis, F7).
- `face_powers(grid, I, dg) -> dict` — outward power `out_top` /
  `out_bottom` through the horizontal faces (net $F_z$ second-order
  extrapolated from the boundary layers on the unigrid, per-boundary-leaf
  extrapolation on the octree) plus `flux_top` / `flux_bottom` / `area`.
- `angular_variance(I, dg) -> (n_cells,)` —
  $\frac{1}{4\pi}\sum_i w_i (I_i - J)^2$, the angular anisotropy of the
  field on the quadrature.
- `energy_balance(grid, I, dg, kappa_a, B) -> dict` — global closure on
  **any grid** (F7): `out_top`, `out_bottom` (via `face_powers`), `heating`
  ($\int 4\pi\kappa_a (B-J)\,dV$ with `cell_volumes`),
  `residual = out_top + out_bottom − heating`, `relative` = residual scaled
  by the larger of face power and heating. With a beam, subtract the
  first-scatter input $\int \kappa_s F_0 e^{-\tau_b}\,dV$ from the residual
  to close the diffuse-field balance (§12, §15 invariant 4).
- `allsky(grid, I, dg, face="top") -> dict` — outgoing intensity images at a
  horizontal face for every leaving ordinate: `omega (n_out,3)`,
  `weights (n_out,)`, `images (n_out, ny, nx)`; `face="top"` uses upward
  ordinates at the top layer, `face="bottom"` downward ordinates at the
  bottom layer (uniform unigrid only).

### `rt3d.cases` — test cases

- `gauss(x, y, z, sigma=1.0, mu=(0,0,4))` — normalized 3D Gaussian
  (integral 1), as in `scripts/class_test.py`.
- `gaussian_cloud(sigma=1.0, mu=(0,0,4), tau_column_max=10.0, omega_cap=None)
  -> case dict` — the `class_test.py` case as callables: `box`
  ($x,y \in [-5,5]$, $z \in [0,10]$) and `kappa_e`, `kappa_s`, `g`, `T`
  fields $(x,y,z) \mapsto$ array. Peak column $\tau$ 10; local $\omega$
  peaks at 1.0 (uncapped, faithful to `class_test.py`) or `omega_cap`; local
  $g$ peaks ≈ 0.952. `class_test.py` stores `albedo`; this case converts to
  $\kappa_s = \omega_{\mathrm{local}}\cdot\kappa_e$.
- `slab_1d` module (used by both `scripts/phase2_slab.py` and
  `scripts/mc_forward_slab.py`, so SC and MC see exactly the same slab):
  `slab_grid(tau=5.0, kappa_e=1.0, nz=50, nxy=4) -> (grid, box, hz)` —
  cubic-cell slab, $n_z$ layers of $\tau/n_z$ each, periodic x/y;
  `slab_profile(grid, nz)` — vertical profile selector (column (1,1) of the
  flat k-major field); references `d1o_J(kappa_e, omega, hz, nz, top_I=1.0,
  n_iter=4000, tol=1e-12)` — 1D discrete ordinates on the *same* z cells
  with 64 Gauss–Legendre $\mu$ levels (the strong angular reference) and
  `two_stream_J(kappa_e, omega, z0, z1, zc, top_I=1.0)` — analytic
  Eddington two-stream with Marshak BCs (context curve only).
- Note: `clear_sky` from `plan/plan.md` still **does not exist as a case
  module**.

## 12. Worked end-to-end example

Gaussian cloud + Lambertian ground + solar beam, on both grids, Ng
accelerated. The script below was **executed with the pyarts interpreter and
its output verified**; the printed numbers are the actual measured values.
Figures are written to `plot/`.

**Units.** The solver itself is unit-agnostic — it only ever sees consistent
numbers — so the example fixes a convention once: **lengths in m**, hence
$\kappa_e$ and $\kappa_s$ in m$^{-1}$, and radiative quantities normalised to
the incident beam flux $F_0 = 1$, i.e. $J/F_0$ and $I/F_0$ [sr$^{-1}$]. $g$ is
dimensionless. With `tau_column_max = 10` over the 10 m box this gives a peak
$\kappa_e \approx 3.9$ m$^{-1}$ (mean free path ≈ 0.26 m). The `scripts/phase*`
runners use the same metre convention.

```mermaid
flowchart LR
    A["case: gaussian_cloud()"] --> B["grid: unigrid or octree_grid + refine/balance"]
    A --> C["fields: kappa_e, kappa_s, g sampled at centers"]
    B --> D["boundary_conditions(top_I, ground)"]
    C --> E["lambda_iteration(..., beam=..., ng=True)"]
    D --> E
    E --> F["result dict: I, S, J, F, omega, history, n_iter, converged"]
    F --> G["diagnostics: energy_balance, flux_divergence, heating_rate"]
    F --> H["2D plots into plot/ (matplotlib Agg)"]
    H --> H1["input fields: pcolormesh of kappa_e, kappa_s, g"]
    H --> H2["octree slice: PolyCollection over the leaves"]
    H --> H3["J slice + band-mean profiles, all-sky images"]
    F --> I["3D view (pyvista, optional): kappa_e isosurfaces + J volume"]
```

```python
# gaussian_cloud_example.py — run from the repo root:
#   Windows:      $env:Path = "$env:USERPROFILE\miniforge3\envs\pyarts\Library\bin;" + $env:Path
#                 & "$env:USERPROFILE\miniforge3\envs\pyarts\python.exe" gaussian_cloud_example.py
#   macOS/Linux:  ~/miniforge3/envs/pyarts/bin/python gaussian_cloud_example.py
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, Normalize

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.directions import icosahedric_grid
from rt3d.geometry import unigrid, ground_surface
from rt3d.octree import octree_grid
from rt3d.transport import boundary_conditions, beam_optical_depth
from rt3d.solver import lambda_iteration
from rt3d.diagnostics import (allsky, energy_balance, flux_divergence,
                              heating_rate)

# Units: the solver is unit-agnostic, so the example fixes them once — lengths
# in m, hence kappa_e, kappa_s in m^-1; radiative quantities are normalised to
# the incident beam flux F0 = 1, i.e. J / F0 and I / F0 [sr^-1].  g is
# dimensionless.
LBL_X, LBL_Y, LBL_Z = "x  [m]", "y  [m]", "z  [m]"

# ---------------------------------------------------------------- case setup
case = gaussian_cloud(omega_cap=0.9)          # capped core: Ng makes it tractable
box = case["box"]                              # x,y in [-5,5], z in [0,10] (m)

dg = icosahedric_grid(n_refine=1)              # 42 ordinates (development grid)
print(f"N_dir = {dg.n_dir},  sum(w) - 4pi = {dg.weights.sum() - 4*np.pi:.3e}")

# solar beam: 30 deg from zenith, propagating in the x-z plane
mu_sun = np.cos(np.radians(30.0))
beam = {"omega_sun": np.array([np.sqrt(1.0 - mu_sun**2), 0.0, -mu_sun]),
        "F0": 1.0}

bc = boundary_conditions(top_I=1.0,
                         ground=ground_surface(albedo=0.3, emissivity=0.0, B_g=0.0))

# ---------------------------------------------------------------- unigrid run
n = 24
xs = np.linspace(box.x0, box.x1, n + 1)[:-1] + (box.Lx / n) / 2
grid = unigrid(xs, xs, np.linspace(box.z0, box.z1, n + 1)[:-1] + (box.Lz / n) / 2)
c = grid.centers
kappa_e = case["kappa_e"](*c.T)
kappa_s = case["kappa_s"](*c.T)
g_cells = case["g"](*c.T)                      # per-cell g -> bucketed matmul
B = np.zeros(grid.n_cells)

t0 = time.perf_counter()
res = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc,
                       g=g_cells, periodic_box=box, beam=beam,
                       n_iter=60, tol=1e-6, ng=True)
t_uni = time.perf_counter() - t0
print(f"unigrid:  n_iter={res['n_iter']}, converged={res['converged']}, "
      f"wall={t_uni:.2f} s")

# ---------------------------------------------------------------- octree run
oct_ = octree_grid(box, h0=1.25)
oct_.refine(case["kappa_e"], tau_max=2.0, eps_grad=1.0, max_depth=3)
oct_.balance_21(max_depth=3)
oct_.finalize()
oct_.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"],
                   g=case["g"], T=case["T"])
print(f"octree:   n_leaf={oct_.n_leaf}, max_level={oct_.max_level}")
lvl_cnt = np.bincount(oct_.level, minlength=oct_.max_level + 1)
print("leaves per level:", ", ".join(f"L{l}: {k}" for l, k in enumerate(lvl_cnt)))

t0 = time.perf_counter()
res_oct = lambda_iteration(oct_, dg, oct_.kappa_e, oct_.kappa_s,
                           np.zeros(oct_.n_leaf), bc,
                           g=oct_.g, periodic_box=box, beam=beam,
                           n_iter=60, tol=1e-6, ng=True)
t_oct = time.perf_counter() - t0
print(f"octree:   n_iter={res_oct['n_iter']}, converged={res_oct['converged']}, "
      f"wall={t_oct:.2f} s")

# ---------------------------------------------------------------- interpretation
J4 = res["J"].reshape(grid.nz, grid.ny, grid.nx)     # k-major -> (k, j, i)

# mean-J profile on the 8 coarse base-cell bands (both grids)
nband = int(round(box.Lz / oct_.h0))
zband = np.linspace(box.z0, box.z1, nband + 1)
zb_uni = np.array([J4[(grid.z >= zband[b]) & (grid.z < zband[b + 1])].mean()
                   for b in range(nband)])
zi = np.clip(((oct_.centers[:, 2] - box.z0) / oct_.h0).astype(int), 0, nband - 1)
zb_oct = np.array([res_oct["J"][zi == b].mean() for b in range(nband)])

zc = 0.5 * (zband[:-1] + zband[1:])
print("z      J_unigrid  J_octree  rel.diff")
for b in (0, 2, 4, 5, 7):
    d = abs(zb_oct[b] - zb_uni[b]) / max(zb_uni[b], 1e-12)
    print(f"{zc[b]:5.2f}  {zb_uni[b]:9.4f}  {zb_oct[b]:8.4f}  {d*100:7.1f}%")

top_layer = oct_.centers[:, 2] + oct_.half >= box.z1 - 1e-9
print(f"top-of-domain mean F_z: unigrid "
      f"{res['F'][-grid.ny*grid.nx:, 2].mean():.4f}, "
      f"octree {res_oct['F'][top_layer, 2].mean():.4f}")

# diffuse-field energy closure (phase-5 recipe, §15 invariant 4)
kappa_a = kappa_e - kappa_s
bal = energy_balance(grid, res["I"], dg, kappa_a, B)
tau_b = beam_optical_depth(grid, beam["omega_sun"], kappa_e, periodic_box=box)
V = grid.hx * grid.hy * grid.hz
beam_scat = (kappa_s * beam["F0"] * np.exp(-tau_b)).sum() * V
res_dif = bal["residual"] - beam_scat
in_diff = np.sum(dg.weights[dg.down_idx]
                 * np.abs(dg.directions[dg.down_idx, 2])) * box.Lx * box.Ly
print(f"  diffuse closure residual {res_dif:9.2f}"
      f"  ({abs(res_dif) / (in_diff + beam_scat) * 100:.2f}% of diffuse input)")

# pointwise invariant 4: div F vs heating rate in the interior
divF = flux_divergence(grid, res["F"])
heat = heating_rate(kappa_a, res["J"], B)
kk = np.arange(grid.n_cells) // (grid.ny * grid.nx)
interior = (kk > 0) & (kk < grid.nz - 1)
err_rms = np.sqrt(np.mean((divF[interior] - heat[interior]) ** 2)) / np.abs(heat).max()
err_max = np.abs(divF[interior] - heat[interior]).max() / np.abs(heat).max()
print(f"pointwise div F vs heating (interior): rms {err_rms*100:.1f}%, "
      f"max {err_max*100:.1f}% of max|heating|")

# ---------------------------------------------------------------- plots
# cell-edge coordinates so pcolormesh draws the actual cell boxes
xe = np.concatenate([grid.x - grid.hx / 2, [grid.x[-1] + grid.hx / 2]])
ze = np.concatenate([grid.z - grid.hz / 2, [grid.z[-1] + grid.hz / 2]])
XU, ZU = np.meshgrid(xe, ze)
jy = grid.ny // 2


def uni_slice(arr):
    """y = const slice of a k-major cell field, as (nz, nx)."""
    return arr.reshape(grid.nz, grid.ny, grid.nx)[:, jy, :]


# (a) the three input fields
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (arr, ttl, clab, cm) in zip(
    axes,
    ((kappa_e, r"$\kappa_e$  (extinction)", r"$\kappa_e$  [m$^{-1}$]", "magma"),
     (kappa_s, r"$\kappa_s$  (scattering)", r"$\kappa_s$  [m$^{-1}$]", "viridis"),
     (g_cells, r"$g$  (HG asymmetry)", r"$g$  [-]", "cividis"))):
    im = ax.pcolormesh(XU, ZU, uni_slice(arr), cmap=cm, shading="flat")
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Z)
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=clab)
fig.suptitle(rf"input fields on the ${grid.nx}^3$ unigrid, "
             rf"$y = {grid.y[jy]:.3f}$ m slice (nearest row to $y = 0$;  "
             rf"$\kappa_s = \omega_{{\rm local}}\,\kappa_e$, $\omega \leq 0.9$)")
fig.tight_layout()
fig.savefig("plot/manual_example_fields.png", dpi=140)
plt.close(fig)

# (b) the same slice on the octree leaves.  A plane through y = 0 coincides with
# a cell face at *every* refinement level (the base lattice is invarianted at -5
# with h0 = 1.25), which would either leave holes or draw overlapping leaves;
# offsetting by a quarter of the finest cell puts the plane strictly inside one
# leaf per (x, z) column at all levels.
h_min = 2.0 * oct_.half.min()
y_slice = 0.0 + h_min / 4.0
sel = np.abs(oct_.centers[:, 1] - y_slice) < oct_.half - 1e-12
cs, hs = oct_.centers[sel], oct_.half[sel]
x0, x1 = cs[:, 0] - hs, cs[:, 0] + hs
z0, z1 = cs[:, 2] - hs, cs[:, 2] + hs
verts = np.stack([np.column_stack([x0, z0]), np.column_stack([x1, z0]),
                  np.column_stack([x1, z1]), np.column_stack([x0, z1])], axis=1)
print(f"octree slice at y = {y_slice:.4f} m: {sel.sum()} leaves "
      f"(finest cell {h_min:.4f} m)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
im0 = axes[0].pcolormesh(XU, ZU, uni_slice(kappa_e), cmap="magma", shading="flat")
axes[0].set_title(rf"$\kappa_e$ on the unigrid  (${grid.nx}^3$)")
axes[0].set_xlabel(LBL_X)
axes[0].set_ylabel(LBL_Z)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04,
             label=r"$\kappa_e$  [m$^{-1}$]")

pc = PolyCollection(verts, array=oct_.kappa_e[sel], cmap="magma",
                    edgecolors="0.55", linewidths=0.25, antialiaseds=True)
axes[1].add_collection(pc)
axes[1].set_xlim(box.x0, box.x1)
axes[1].set_ylim(box.z0, box.z1)
axes[1].set_title(rf"$\kappa_e$ on the octree leaves ({sel.sum()} in slice)")
axes[1].set_xlabel(LBL_X)
axes[1].set_ylabel(LBL_Z)
fig.colorbar(pc, ax=axes[1], fraction=0.046, pad=0.04,
             label=r"$\kappa_e$  [m$^{-1}$]")

norm_lvl = BoundaryNorm(np.arange(oct_.max_level + 2) - 0.5, oct_.max_level + 1)
pl_ = PolyCollection(verts, array=oct_.level[sel], cmap="viridis", norm=norm_lvl,
                     edgecolors="0.55", linewidths=0.25, antialiaseds=True)
axes[2].add_collection(pl_)
axes[2].set_xlim(box.x0, box.x1)
axes[2].set_ylim(box.z0, box.z1)
axes[2].set_title(rf"octree refinement level ($h = h_0/2^\ell$, $h_0 = {oct_.h0}$ m)")
axes[2].set_xlabel(LBL_X)
axes[2].set_ylabel(LBL_Z)
cb = fig.colorbar(pl_, ax=axes[2], fraction=0.046, pad=0.04,
                  ticks=np.arange(oct_.max_level + 1), label="level  \u2113")
cb.set_ticklabels([f"{l}  ({oct_.h0 / 2**l:g} m)" for l in range(oct_.max_level + 1)])
fig.suptitle(rf"octree slice at $y = {y_slice:.3f}$ m   "
             rf"($\tau_{{\max}} = 2.0$, $\varepsilon = 1.0$, "
             rf"$n_{{\rm leaf}} = {oct_.n_leaf}$)")
fig.tight_layout()
fig.savefig("plot/manual_example_octree.png", dpi=140)
plt.close(fig)

# (c) mean intensity J: slice + band-mean profiles
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im = axes[0].pcolormesh(XU, ZU, J4[:, jy, :], cmap="viridis", shading="flat")
axes[0].set_xlabel(LBL_X)
axes[0].set_ylabel(LBL_Z)
axes[0].set_title(r"$J$, $y = 0$ slice (unigrid)")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label=r"$J\,/F_0$")
axes[1].plot(zb_uni, zc, label="unigrid 24$^3$")
axes[1].plot(zb_oct, zc, "--", label="octree")
axes[1].set_xlabel(r"band-mean $J\,/F_0$")
axes[1].set_ylabel(LBL_Z)
axes[1].legend()
fig.tight_layout()
fig.savefig("plot/manual_example_j.png", dpi=120)
plt.close(fig)

# (d) all-sky images, one per upward ordinate, with a shared colour scale
sky = allsky(grid, res["I"], dg, face="top")
n_out = sky["omega"].shape[0]
ncol, nrow = 7, int(np.ceil(n_out / 7))
fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol + 1.2, 1.9 * nrow + 1.0))
norm_sky = Normalize(vmin=0, vmax=max(sky["images"].max(), 1e-12))
ext = [box.x0, box.x1, box.y0, box.y1]
last = None
for r in range(nrow):
    for cc in range(ncol):
        ax, k = axes[r, cc], r * ncol + cc
        if k >= n_out:
            ax.axis("off")
            continue
        last = ax.imshow(sky["images"][k], origin="lower", norm=norm_sky,
                         cmap="inferno", extent=ext)
        ax.set_title(rf"$\mu = \cos\vartheta = {sky['omega'][k, 2]:.2f}$", fontsize=7)
        if r < nrow - 1:
            ax.set_xticklabels([])
        if cc > 0:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6)
axes[0, 0].set_ylabel(LBL_Y, fontsize=8)
axes[-1, 0].set_xlabel(LBL_X, fontsize=8)
fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.25)
cax = fig.add_axes([0.90, 0.08, 0.02, 0.80])
fig.colorbar(last, cax=cax, label=r"$I\,/F_0$  [sr$^{-1}$]")
fig.suptitle(rf"all-sky outgoing intensity at the top face "
             rf"({n_out} upward ordinates of the {dg.n_dir})")
fig.savefig("plot/manual_example_allsky.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- 3D view
# pyvista is optional:  mamba install -n pyarts -c conda-forge pyvista
# (pulls vtk).  Everything above runs without it.
try:
    import pyvista as pv
except ImportError:
    pv = None
    print("pyvista not installed - skipping the 3D figure")

if pv is not None:
    # sample the piecewise-constant octree fields on the lattice of finest-cell
    # centres: exact, because every lattice point lies strictly inside one leaf
    nl = (np.array([box.Lx, box.Ly, box.Lz]) / h_min).astype(int)
    gx = box.x0 + (np.arange(nl[0]) + 0.5) * h_min
    gy = box.y0 + (np.arange(nl[1]) + 0.5) * h_min
    gz = box.z0 + (np.arange(nl[2]) + 0.5) * h_min
    P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    li = oct_.locate(P)
    lat = pv.ImageData(dimensions=tuple(int(v) for v in nl),
                       origin=(gx[0], gy[0], gz[0]), spacing=(h_min,) * 3)
    lat.point_data["J"] = res_oct["J"][li]
    lat.point_data["kappa_e"] = oct_.kappa_e[li]
    print(f"3D lattice {tuple(int(v) for v in nl)} at {h_min:g} m: "
          f"J {lat['J'].min():.3f}..{lat['J'].max():.3f}, "
          f"kappa_e {lat['kappa_e'].min():.3f}..{lat['kappa_e'].max():.3f}")

    iso_levels = [0.5, 1.0, 2.0, 3.0]
    iso_cols = [(0.75, 0.85, 1.0), (0.55, 0.7, 1.0),
                (0.35, 0.5, 0.95), (0.2, 0.3, 0.85)]
    pv.OFF_SCREEN = True
    views = []
    for az, el, what in ((40, 18, "iso"), (-140, 14, "vol")):
        plotter = pv.Plotter(off_screen=True, window_size=(760, 640))
        if what == "iso":
            for lev, col in zip(iso_levels, iso_cols):
                plotter.add_mesh(lat.contour(isosurfaces=[lev], scalars="kappa_e"),
                                 color=col, opacity=0.35, specular=0.5,
                                 specular_power=25)
        else:
            # opacity increases toward low J, so the cloud shadow is the dense
            # part of the volume; the scalar bar is added by matplotlib below
            plotter.add_volume(lat, scalars="J", cmap="inferno",
                               opacity=np.linspace(0.10, 0.0, 24), shade=False,
                               mapper="smart", show_scalar_bar=False)
        plotter.add_mesh(pv.Box(bounds=(box.x0, box.x1, box.y0, box.y1,
                                        box.z0, box.z1)),
                         style="wireframe", color="black", line_width=1.0)
        plotter.reset_camera()
        plotter.camera.azimuth, plotter.camera.elevation = az, el
        plotter.camera.zoom(0.92)
        views.append(plotter.screenshot(return_img=True))
        plotter.close()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    for ax, img in zip(axes, views):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title(r"$\kappa_e$ isosurfaces at 0.5, 1, 2, 3 m$^{-1}$"
                      "\n(octree staircase visible)")
    axes[1].set_title(r"$J$ volume rendering"
                      "\n(opacity increases toward low $J$)")
    fig.subplots_adjust(right=0.88, wspace=0.05)
    cax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
    smap = plt.cm.ScalarMappable(cmap="inferno",
                                 norm=Normalize(vmin=lat["J"].min(),
                                                vmax=lat["J"].max()))
    fig.colorbar(smap, cax=cax, label=r"$J\,/F_0$")
    fig.savefig("plot/manual_example_3d.png", dpi=140)
    plt.close(fig)
```

Verified output (measured on macOS/arm64 with the pyarts interpreter; the
iteration counts and all physics numbers are machine-independent, wall times
are not — the same run on the Windows machine takes 4.2 s / 107.0 s):

```text
N_dir = 42,  sum(w) - 4pi = -7.105e-15
unigrid:  n_iter=32, converged=True, wall=1.64 s
octree:   n_leaf=25264, max_level=3
leaves per level: L0: 316, L1: 788, L2: 3680, L3: 20480
octree:   n_iter=30, converged=True, wall=38.75 s
z      J_unigrid  J_octree  rel.diff
 0.62     0.4372    0.3095     29.2%
 3.12     0.4529    0.3431     24.2%
 5.62     0.5589    0.4788     14.3%
 6.88     0.5870    0.5377      8.4%
 9.38     0.5893    0.5850      0.7%
top-of-domain mean F_z: unigrid -2.5215, octree -2.5312
  diffuse closure residual      9.49  (2.96% of diffuse input)
pointwise div F vs heating (interior): rms 1.6%, max 13.9% of max|heating|
octree slice at y = 0.0391 m: 796 leaves (finest cell 0.1562 m)
3D lattice (64, 64, 64) at 0.15625 m: J 0.064..0.608, kappa_e 0.000..3.876
```

### Figures produced

**Input fields** — $\kappa_e$, $\kappa_s$ and $g$ on the $y = 0.208$ m row of
the $24^3$ unigrid (the nearest cell row to $y = 0$), drawn as `pcolormesh`
over the *cell edges* so the cell structure is visible. $\kappa_s =
\omega_{\mathrm{local}}\kappa_e$ with the $\omega \le 0.9$ cap flattens the
scattering panel at the cloud core; $\kappa_e$ peaks at $\approx 3.9$ m$^{-1}$
(column $\tau = 10$) and $g$ at $\approx 0.95$ ($= 1/1.05$):

![Input fields kappa_e, kappa_s and g on the unigrid, y = 0 slice, with units](plot/manual_example_fields.png)

**The octree grid itself** — the same $\kappa_e$ field on the actual leaves.
Left: unigrid. Middle: the 796 leaves cut by the plane $y = 0.039$ m, drawn as
one `PolyCollection`. Right: the refinement level $\ell$ of those leaves, with
the colourbar labelled by the cell size $h_0/2^\ell$. The staircase is the
octree doing its job — $h = 1.25$ m in the vacuum tails, $0.156$ m inside the
cloud, for 25 264 leaves instead of the $24^3 = 13824$ unigrid cells:

![kappa_e on the unigrid, kappa_e on the octree leaves, and the octree refinement level, same y slice](plot/manual_example_octree.png)

**Mean intensity $J$** — left: $y = 0$ slice through the cloud on the unigrid,
colourbar $J/F_0$; right: band-mean $J$ profiles, unigrid $24^3$ vs octree (the
difference is the unigrid grazing-ordinate stencil error, invariant 5):

![Mean intensity J: y=0 slice (unigrid, left) and band-mean J profiles unigrid vs octree (right)](plot/manual_example_j.png)

**All-sky outgoing intensity at the top face** — the 21 upward ordinates of the
42-ordinate development grid, each panel labelled with $\mu = \cos\vartheta$,
one shared colour scale $I/F_0$ [sr$^{-1}$], axes in m. The dark blobs are the
cloud shadow seen at the corresponding slant; the bright diagonal bands are the
forward-scattering lobe of the $30^\circ$ beam ($g \approx 0.95$):

![All-sky outgoing intensity at the top face, one image per upward ordinate, with mu labels and a labelled colourbar](plot/manual_example_allsky.png)

**3D view of the simulation** (pyvista, off-screen; install command in the code
block above) —
left: $\kappa_e$ isosurfaces at 0.5, 1, 2, 3 m$^{-1}$, nested, with the octree
staircase still visible on the surfaces. Right: volume rendering of $J$ with
the opacity transfer function *inverted* (low $J$ $\to$ high opacity), so the
dense dark core is the radiation shadow of the cloud and the beam path lights
up around it. Both panels come from the octree solution resampled onto the
$64^3$ lattice of finest-cell centres (§12 code), domain box outlined:

![3D view: kappa_e isosurfaces and a volume rendering of J from the octree solution](plot/manual_example_3d.png)

Interpreting the result dict and the diagnostics:

- **`n_iter` / `history`**: Ng converges the near-conservative cloud
  ($\omega \le 0.9$, $g \approx 0.95$ core) in ~30 iterations; plain LI would
  need several hundred (§8). `history[k]` is the relative source change at
  iteration $k$.
- **unigrid vs octree J**: the ~10–30% band-mean differences are the
  *unigrid* lattice-stencil path error for grazing ordinates in the strongly
  heterogeneous cloud (verification invariant 5, §15) — the octree
  exact-characteristic result is the better one; the top-of-domain flux,
  dominated by near-vertical paths, agrees to 0.4%.
- **energy closure**: `energy_balance` alone leaves the beam first-scatter
  input unaccounted; after subtracting
  $\int \kappa_s F_0 e^{-\tau_b}\,dV$ the global diffuse balance closes to
  **2.96%** of the incident diffuse power — the discretization error of the
  42-ordinate / $24^3$ solve, not a conservation bug.
- **pointwise div F vs heating**: rms 1.6% of max|heating| — invariant 4.
- Figures: `plot/manual_example_fields.png` (input fields $\kappa_e$, $\kappa_s$,
  $g$), `plot/manual_example_octree.png` (same slice on the octree leaves + the
  refinement levels), `plot/manual_example_j.png` (J slice + profile),
  `plot/manual_example_allsky.png` (21 upward-ordinate all-sky images at the
  top face), `plot/manual_example_3d.png` (pyvista $\kappa_e$ isosurfaces +
  $J$ volume). All axes and colourbars carry the units of the convention above.

### Minimal MC example (verified)

The same MC machinery on the phase-2 slab — the smallest end-to-end check of
§9: stage-1 `j_mc` in both variants against the 64-level Gauss–Legendre
discrete-ordinates reference from `rt3d.cases.slab_1d`. This script was
**executed with the pyarts interpreter**; the printed numbers are its actual
output (seeded, NumPy path — MC wall times scale with history count, not
cell count).

```python
# mc_slab_example.py — run from the repo root
import time

import numpy as np

from rt3d.cases.slab_1d import d1o_J, slab_grid
from rt3d.mc.forward import j_mc
from rt3d.transport import boundary_conditions

grid, box, hz = slab_grid(tau=5.0, kappa_e=1.0, nz=50)
bc = boundary_conditions(top_I=1.0)
kappa_e = np.full(grid.n_cells, 1.0)
kappa_s = 0.9 * kappa_e
B = np.zeros(grid.n_cells)


def col(v):                     # vertical profile out of a flat cell field
    return np.asarray(v).reshape(grid.nz, grid.nx, grid.ny)[:, 1, 1]


J_ref = d1o_J(1.0, 0.9, hz, 50)          # 64 GL levels, angle-exact reference

common = dict(g=0.0, periodic_box=box, n_histories=20_000,
              n_boundary_histories=20_000, n_batches=200)
t0 = time.perf_counter()
r_lam = j_mc(grid, kappa_e, kappa_s, B, bc, stage1="lambda",
             n_iter=120, tol=1e-8, seed=7, **common)
t_lam = time.perf_counter() - t0
t0 = time.perf_counter()
r_exp = j_mc(grid, kappa_e, kappa_s, B, bc, stage1="explicit",
             seed=8, **common)
t_exp = time.perf_counter() - t0

for tag, r, wall in (("lambda", r_lam, t_lam), ("explicit", r_exp, t_exp)):
    dev = np.abs(col(r["J"]) - J_ref)
    band = 2.0 * r["sigma_J"].max() + 0.03 * J_ref.max()
    print(f"MC stage1={tag}: max|J-J_ref|/max J_ref = "
          f"{dev.max()/J_ref.max()*100:5.2f}% (band "
          f"{band/J_ref.max()*100:5.2f}%), mean J {col(r['J']).mean():.4f} "
          f"vs ref {J_ref.mean():.4f}, wall {wall:.1f} s")
```

Verified output (macOS/arm64; the band = 2σ + 3% reference cell
discretization, the acceptance rule of `scripts/mc_forward_slab.py` — the
band is dominated by the per-cell σ of near-vacuum cells, finding F3):

```text
MC stage1=lambda: max|J-J_ref|/max J_ref =  4.12% (band  8.98%), mean J 0.2403 vs ref 0.2350, wall 16.8 s
MC stage1=explicit: max|J-J_ref|/max J_ref =  2.89% (band  6.53%), mean J 0.2323 vs ref 0.2350, wall 1.4 s
```

Both variants sit inside the band; the λ variant needs one fresh photon set
*per iteration* (hence the 10× wall time and the F5 noise floor in
`history`), while the explicit variant finishes the whole field in one shot
with ≈ 40 scattering generations (`n_generations`). For the
production chain A1 → A2 → B (stage-2 radiance and adjoint all-sky
differenced against SC) run `scripts/mc_compare.py`.

## 13. Runner scripts

All scripts in `scripts/` use `# %%` cell markers (VS Code / Spyder workflow),
insert the repo root on `sys.path`, use `matplotlib.use("Agg")`, and write
figures to `plot/`. Run any of them as

```sh
# macOS / Linux
~/miniforge3/envs/pyarts/bin/python scripts/phase0_check.py
```

```powershell
# Windows (PATH fix first!)
& "$env:USERPROFILE\miniforge3\envs\pyarts\python.exe" scripts/phase0_check.py
```

**SC phase scripts** (phases 0–6):

| script | what it validates |
|---|---|
| `phase0_check.py` | direction grid rebuilds for $n_{\mathrm{refine}}$ 0–3 (12/42/162/642), $\sum w = 4\pi$, $\oint\Phi\,d\Omega = 1$, no $\Omega_z = 0$ ordinate, ray-box exit + DDA march consistency (invariants 1–2). |
| `phase1_beer.py` | uniform-grid short characteristics: beam $I = F_0 e^{-\tau_b}$, diffuse transmittance vs Beer–Lambert for every ordinate/cell (invariant 3), isothermal $I = B$ exactness. |
| `phase2_slab.py` | scattering + plain λ-iteration on a 1D slab ($\tau = 5$, $\omega = 0.5/0.9$) vs an independent 64-level Gauss–Legendre discrete-ordinates solver and the two-stream approximation. Case + references shared via `rt3d.cases.slab_1d`. |
| `phase3_octree.py` | octree vs unigrid on the Gaussian cloud (invariant 5): chain-vs-exact-path sanity (`build_chain_octree` reference), octree self-convergence, column $\tau \to 10$, scattering run at $\omega \le 0.7$. |
| `phase4_ng.py` | 1D Ng acceleration: LI vs Ng iteration counts as $\omega \to 1$ (slab), cloud on unigrid ($\omega \le 0.9$, per-cell $g$) and on the octree. |
| `phase5_ground.py` | periodic sides + Lambertian ground + all-sky output; analytic ground Beer–Lambert checks; pointwise div F vs heating and global energy balance incl. beam/ground terms (invariant 4). |
| `phase6_directions.py` | direction-grid convergence 42 → 162 → 642 on the cloud, $g = 0.952$ slab (known 42-ordinate limitation), performance pass (chain build + sweep time vs $N_{\mathrm{dir}}$), npz record with `n_refine`. |

**MC phase scripts** (M0–M6, §9):

| script | what it validates |
|---|---|
| `mc_phase0_check.py` | M0 tracker: flight distribution + event fractions, HG samples vs `hg_matrix` moments (invariant-2 companion), MC beam transmittance $= e^{-\tau}$ on both grids and both majorant modes, cell vs global mode agreement. |
| `mc_forward_slab.py` | M1 stage-1 $J$ on the phase-2 slab: λ variant (fresh / CRN / Ng) and explicit chains vs the 64-GL discrete-ordinates reference, with 2σ bands (shared case `rt3d.cases.slab_1d`). |
| `mc_forward_cloud.py` | M2 stage-2 radiance $I(\text{cell}, \text{ordinate})$ on the cloud **octree**, where MC and SC share the exact marches — purely statistical differencing. |
| `mc_backward.py` | M3 adjoint detector MC: all-sky consistency (one detector per upward ordinate at exact node direction) vs SC columns on slab + cloud, and the efficiency argument (point query at fixed cost). |
| `mc_compare.py` | M5 integration pass: full chain A1 → A2 → B vs SC on one setup, energy closure over the whole MC chain (the head-to-head report). |
| `mc_phase4_walltimes.py` | M4 wall-time table: SC vs MC per query type, NumPy vs numba, analog vs variance-reduced (MC wins stage-2/backward queries, SC wins stage-1 fields). |

**Comparison / performance / report scripts:**

| script | what it does |
|---|---|
| `compare_transport.py` | transport-scheme comparison (unigrid chain vs single-cell sc vs octree, ± delta-M) on the §12 worked example; caches solves in `plot/compare_transport/_solve_cache/` (plan/compare-transport-plan.md). |
| `solver_matrix.py` | the solver-matrix report runner (plan/solver-test-matrix-plan.md §9): one physical setup, every variant → `matrix.csv`, `ranking.csv`, `convergence.csv`, `mc_stats.csv` + figures under `plot/solver_matrix/`; cached reruns are seconds. |
| `bench_optimization.py` | reproduces the optimization-plan baseline table (march cost/memory, sweep costs, scattering matmul). |
| `bench_compare_walltimes.py` | before/after wall-clock of the solver optimizations vs the cached `compare_transport.py` solutions (validation gate: max relative $\Delta J$). |
| `sc_transport_figures.py` | schematic figures for `docs/sc-transport.md` (§6). |
| `report_figures.py` | vector PDF figures + LaTeX table fragments for the project report (`report/`). |

`scripts/class_test.py` and `scripts/icosphere_grid.py` are **read-only
legacy reference** (pre-existing research code); do not modify them.

## 14. Condensed API reference

Signatures + one-liners for all public functions/classes of `rt3d`
(docstrings in the modules are authoritative for defaults).

### `rt3d.directions`

| signature | description |
|---|---|
| `make_icosahedric_grid()` | base icosahedron: 12 unit vertices, 20 triangles |
| `refine_icosahedric_grid(vertices, triangles, n_refine=1)` | vectorized geodesic refinement |
| `solid_angle_weights(vertices, triangles)` | per-vertex solid angles from spherical-triangle areas |
| `hg(cos_theta, g)` | HG phase function, $\oint\Phi\,d\Omega = 1$ |
| `direction_grid(n_refine=1)` | icosahedral direction set with weights, up/down index split |
| `.cos_theta()` | $(N,N)$ matrix of direction cosines |
| `.hg_matrix(g, with_weights=True, normalize=True)` | weighted, row-normalized HG matrix |
| `.hg_delta_matrices(g, M)` | exact delta-M split of the row-normalized HG matrix |
| `icosahedric_grid(n_refine=1)` | build a `direction_grid` (default 42 ordinates) |

### `rt3d.geometry`

| signature | description |
|---|---|
| `cuboid(x0, x1, y0, y1, z0, z1, periodic_x=True, periodic_y=True)` | domain box with periodic x/y |
| `.contains(points)` / `.wrap_xy(points)` | inside test / periodic wrap |
| `unigrid(x, y, z)` | uniform Cartesian cell lattice from cell centers |
| `.cell_box(idx)` / `.unravel(idx)` / `.ravel(i,j,k)` / `.locate(points)` | cell boxes, index maps, O(1) point location |
| `ray_box_exit(origin, direction, box)` | exit distance + face of a box along a direction |
| `march_lattice(origin, direction, grid, max_steps=None, periodic_box=None)` | vectorized DDA march through the unigrid |
| `ground_surface(albedo=0.0, emissivity=1.0, B_g=0.0, height=0.0)` | flat Lambertian + emissive ground (variable surface = hook only) |

### `rt3d.octree`

| signature | description |
|---|---|
| `octree_grid(cuboid, h0)` | flat struct-of-arrays octree, cubic base cells |
| `.refine(kappa_e_fn, tau_max=0.5, eps_grad=0.3, max_depth=6)` | top-down refinement on $\kappa_e\tau$ + gradient criteria |
| `.balance_21(max_depth=6)` | 2:1 face balance via 3×3 face probing |
| `.finalize()` / `.sample_fields(...)` | freeze leaf arrays / sample callable fields at leaf centers |
| `.locate(points)` / `.split(nodes)` / `.leaves_all()` | vectorized descent point location / node ops |
| `.march_arrays()` | cached plain-array node tables for the numba kernels (`rt3d.jit`) |
| `march_octree(origin, direction, oct, path_len, periodic_box=None)` | vectorized DDA march through the octree (reference) |
| `node_major_segments(seg_ray, seg_leaf, seg_len, n_nodes)` | regroup segments into node-major blocks |

### `rt3d.characteristics`

| signature | description |
|---|---|
| `stencil_for_direction(omega, h, max_offset=None)` | best integer lattice stencil for one ordinate |
| `subsegments_uniform(s, h)` | exact sub-segments of the stencil center-to-center path |
| `build_chain(grid, omega, periodic_box=None, max_offset=None)` | unigrid predecessor chain (stencil + waves + root marches) |
| `build_chain_octree(oct, omega, periodic_box=None, cos_min=0.9, max_hops=8)` | octree center-to-center chain — **reference only, not used by the solver** |

### `rt3d.short_characteristics` (unigrid single-cell SC, `transport="sc"`)

| signature | description |
|---|---|
| `sc_stencil(omega, h)` | entry axis, sense, face-segment length, 4 bilinear nodes + weights |
| `build_sc_chain(grid, omega, periodic_box=None)` | per-ordinate SC chain (layer-grouped order, node indices/faces) |
| `sweep_direction_sc(grid, sc, kappa_e, S, bc, bottom_I=None, wrap_tol=1e-12, max_passes=None)` | one SC ordinate sweep (layer loop, ring passes on periodic entry axes) |
| `sweep_all_sc(grid, direction_grid, kappa_e, S, bc, periodic_box=None, chains=None, n_jobs=1)` | all-ordinate SC transport |

### `rt3d.transport`

| signature | description |
|---|---|
| `boundary_conditions(top_I=0.0, bottom_I=0.0, side_I=0.0, ground=None)` | face backgrounds + optional ground |
| `.inflow(exit_axis, exit_side)` | background inflow for a chain entry face |
| `bottom_mask(grid)` | cells/leaves touching the bottom face |
| `downward_irradiance(I, dg)` | $\sum_{\Omega_z<0} w\|\Omega_z\| I$ at ground level |
| `ground_inflow(F_down, ground, beam_irr=0.0)` | Lambertian ground radiance |
| `beam_irradiance_ground(grid, omega_sun, kappa_e, tau_b, F0, periodic_box=None)` | direct-beam irradiance at the ground face |
| `sweep_direction(grid, chain, kappa_e, S, bc, bottom_I=None)` | one unigrid ordinate sweep (reference) |
| `sweep_all(grid, direction_grid, kappa_e, S, bc, periodic_box=None, chains=None, n_jobs=1)` | all-ordinate unigrid transport (numba kernels when available) |
| `precompute_octree_march(oct, omega, periodic_box=None, jit=None)` | exact backward march per leaf center, one ordinate (default: fused numba mode, no segment arrays) |
| `sweep_direction_octree_exact(oct, march, kappa_e, S, bc, bottom_I=None)` | one octree ordinate sweep, exact characteristics |
| `sweep_all_octree_exact(oct, direction_grid, kappa_e, S, bc, periodic_box=None, marches=None, n_jobs=1)` | all-ordinate octree transport (solver path) |
| `sweep_direction_octree(oct, chain, kappa_e, S, bc)` / `sweep_all_octree(...)` | chain-based octree sweeps (reference only) |
| `beam_optical_depth(grid, omega_sun, kappa_e, periodic_box=None)` | beam $\tau$ at unigrid cell centers |
| `beam_optical_depth_octree(oct, omega_sun, kappa_e, periodic_box=None, jit=None)` | beam $\tau$ at octree leaf centers |

### `rt3d.jit` (numba fast paths)

| signature | description |
|---|---|
| `HAVE_NUMBA` | numba availability flag; all entry points fall back to NumPy references |
| `march_octree_jit(origin, direction, oct, path_len, periodic_box=None, merge=False)` | compiled per-ray DDA (bit-comparable segments) |
| `sweep_direction_octree_exact_jit(oct, omega, budget, kappa_e, S, I_in, periodic_box=None)` | fused octree march + formal solution, parallel over rays |
| `beam_optical_depth_octree_jit(oct, omega_sun, kappa_e, budget, periodic_box=None)` | fused beam-$\tau$ march |
| `locate_jit(points, oct)` | parallel point location |
| `chain_sweep_jit(grid, chain, kappa_e, S_d, bc, bottom_I=None)` | compiled unigrid chain sweep (bit-identical) |
| `sc_sweep_jit(grid, sc, kappa_e, S_d, bc, bottom_I=None, wrap_tol=1e-12, max_passes=None)` | compiled SC sweep (bit-identical) |

### `rt3d.source`

| signature | description |
|---|---|
| `single_scattering_albedo(kappa_e, kappa_s)` | $\omega = \kappa_s/\kappa_e$ with vacuum guard |
| `mean_intensity(I, dg)` | $J = \frac{1}{4\pi}\oint I\,d\Omega$ |
| `flux_vector(I, dg)` | $F = \oint I\,\boldsymbol{\Omega}\,d\Omega$ |
| `scattering_source(I, dg, omega, B, Phi_w=None, g=0.0, beam_scat=None, g_cells=None, g_decimals=2, Phi_cache=None, delta_M=0)` | direction-dependent source $S = (1-\omega)B + \omega\langle\Phi I\rangle + \text{beam\_scat}$; `delta_M >= 1` uses the delta-M form |
| `beam_scattering_source(dg, omega_sun, F0, tau_b, omega, g, g_cells=None, delta_M=0)` | fixed beam first-scatter source, normalized HG row |
| `delta_extinction(kappa_e, kappa_s, g, M)` | delta-M transport extinction $\kappa_e - \kappa_s g^M$ |

### `rt3d.accelerator`

| signature | description |
|---|---|
| `ng1d(grid, max_hist=None, n_start=3)` | 1D plane-averaged Ng acceleration state |
| `.plane_average(v)` / `.lambda_bar()` / `.step(S_swept, S_new)` | z-bin averages / $\bar{\Lambda}$ fit / next source |

### `rt3d.solver`

| signature | description |
|---|---|
| `lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, g=0.0, periodic_box=None, beam=None, n_iter=100, tol=1e-8, ng=None, delta_M=0, transport="chain", jit=None, n_jobs=1)` | the SC solver; returns `I, S, J, F, omega, history, n_iter, converged, delta_M, kappa_e_delta, transport, jit, n_jobs` |

### `rt3d.mc.*` (Monte Carlo family, §9)

| signature | description |
|---|---|
| `j_mc(grid, kappa_e, kappa_s, B, bc, g=0.0, periodic_box=None, beam=None, stage1="lambda", n_histories=20000, n_boundary_histories=20000, n_beam_histories=20000, n_ground_histories=None, n_batches=100, seed=0, mode="cell", kappa_max=None, n_iter=200, tol=1e-8, ng=None, crn=False, birth="source", max_collisions=20000, path_cap=None, tally_flux=True, implicit_capture=False, roulette_floor=None, jit=False)` | MC stage 1: `J, sigma_J, F, sigma_F, heating, history, n_iter, converged` + direct-beam parts + diagnostics |
| `stage2_source(grid, dg, kappa_e, kappa_s, B, J, g=0.0, beam=None, periodic_box=None)` | fixed source $(1-\omega)B + \omega J$ + beam first scatter |
| `radiance_from_source(grid, dg, kappa_e, S_total, bc, periodic_box=None, estimator="next_flight", n_histories=25, n_batches=10, seed=0, mode="cell", kappa_max=None, jit=False, ground_I=None)` | MC stage 2 radiance `I, sigma_I` on the SC ray set (`"deterministic"` = zero-noise) |
| `backward_detector_mc(grid, kappa_e, kappa_s, B, bc, detector, g=0.0, beam=None, tau_b=None, periodic_box=None, n_batches=100, seed=0, mode="cell", kappa_max=None, max_collisions=20000, path_cap=None, implicit_capture=False, roulette_floor=None, jit=False)` | adjoint radiance at one detector: `I, sigma_I` + walk statistics |
| `backward_all_sky(grid, kappa_e, kappa_s, B, bc, dg, point, g=0.0, beam=None, tau_b=None, periodic_box=None, n_seeds=20000, n_batches=100, seed=0, mode="cell", kappa_max=None, cone_scale=0.0, jit=False, implicit_capture=False, roulette_floor=None)` | adjoint all-sky at a point, one detector per upward ordinate |
| `all_sky_cone_half(dg, i)` | equal-area cap half-angle of ordinate $i$ |
| `sample_collisions(starts, directions, budgets, grid, kappa_e, kappa_s, ...)` | shared Woodcock flight sampler (both grids, both majorant modes) |
| `walk_lattice(starts, directions, grid, budgets=None, periodic_box=None)` / `walk_octree(starts, directions, oct, budgets, periodic_box=None)` | pure geometric DDA traversals (segment-identical to the SC marches) |
| `track_histories(starts, directions, grid, kappa_e, kappa_s, tally, ...)` | analog generation chain (`stage1="explicit"`) |
| `hg_cos_theta(u, g)` / `hg_directions(...)` / `hg_scatter_directions(...)` | HG inverse-CDF direction sampling |
| `isotropic_directions(n, rng)` / `face_inflow_directions(n, axis, side, rng)` / `face_points(...)` / `face_area(...)` / `beam_seed(...)` | source population samplers |
| `sample_births(n, grid, rate, method="categorical", seed=0, volumes=None)` | birth positions from a per-cell source rate (null source events) |
| `track_tally(n_cells, n_histories, n_batches=100, flux=True)` | track-length tally with history-batch error bars |
| `cell_volumes(grid)` / `n_cells_of(grid)` / `heating_rate(kappa_a, B, J)` | grid helpers (note the `heating_rate` arg order vs diagnostics) |
| `ground_radiance(grid, ground, F_down=None, beam_irr=0.0)` / `bottom_columns(grid)` / `sample_ground_face(...)` / `down_irradiance_from_segments(...)` | MC ground machinery (calls `transport.ground_inflow`) |
| `jit_mc_usable(grid, periodic_box=None)` / `sample_collisions_jit(...)` | MC numba fast path (statistically equivalent, different stream) |

### `rt3d.diagnostics`

| signature | description |
|---|---|
| `heating_rate(kappa_a, J, B)` | $4\pi\kappa_a (B - J)$ per cell |
| `cell_volumes(grid)` | cell/leaf volumes ($h_x h_y h_z$ / $(2\,\mathrm{half})^3$) |
| `flux_divergence(grid, F)` | div F by central differences (unigrid and octree) |
| `face_powers(grid, I, dg)` | outward power through top/bottom faces (both grids) |
| `angular_variance(I, dg)` | angular anisotropy $\frac{1}{4\pi}\sum_i w_i (I_i-J)^2$ per cell |
| `energy_balance(grid, I, dg, kappa_a, B)` | global face-flux vs heating closure (any grid, adds `relative`) |
| `allsky(grid, I, dg, face="top")` | outgoing intensity images per leaving ordinate at top/bottom face (unigrid) |

### `rt3d.cases`

| signature | description |
|---|---|
| `gauss(x, y, z, sigma=1.0, mu=(0,0,4))` | normalized 3D Gaussian |
| `gaussian_cloud(sigma=1.0, mu=(0,0,4), tau_column_max=10.0, omega_cap=None)` | class_test.py cloud as box + callable fields |
| `slab_1d.slab_grid(tau=5.0, kappa_e=1.0, nz=50, nxy=4)` | cubic-cell periodic slab lattice → `(grid, box, hz)` |
| `slab_1d.slab_profile(grid, nz)` | vertical profile selector for flat slab fields |
| `slab_1d.d1o_J(kappa_e, omega, hz, nz, top_I=1.0, ...)` | 64 Gauss–Legendre 1D discrete-ordinates reference $J$ |
| `slab_1d.two_stream_J(kappa_e, omega, z0, z1, zc, top_I=1.0)` | analytic Eddington two-stream (context only) |

## 15. Verification invariants and the test matrix

The five physics invariants from AGENTS.md, with measured numbers (all
re-verified for this manual unless marked "phase record"). They are enforced
continuously by the pytest suite in `tests/` — 341 tests: per-module suites
plus the six-tier **cross-solver matrix** (full spec + findings log F1–F14
in `plan/solver-test-matrix-plan.md`, shared case `tests/matrix_cases.py`,
pooled-statistics helpers `tests/matrix_stats.py`):

| tier | module | what it pins |
|---|---|---|
| T0 | `test_matrix_invariants.py` | invariants swept over every solver variant, no full solve |
| T1 | `test_matrix_equivalence.py` | deterministic variant equivalence (chain / sc / octree, ±delta-M, ±Ng) |
| T2 | `test_matrix_crossfamily.py` | **zero-noise identity**: SC octree sweep ≡ MC `radiance_from_source(estimator="deterministic")` to 1.2e-12 (F2) |
| T3 | `test_matrix_statistics.py` (mark `mc`) | MC ↔ SC and MC ↔ MC consistency with the pooled noise model (fixes F3) |
| T4 | `test_matrix_closure.py` | global invariant-4 closures per family, octree-capable (F7) |
| T5 | `test_matrix_ladder.py` (mark `slow`) | refinement ladders, fine-octree reference ranking, convergence orders |

The MC family also has direct companions of the invariants (per-module
`tests/test_mc_*.py`): invariant 2 — the HG inverse-CDF reproduces the
`hg_matrix` quadrature moments to quadrature accuracy; invariant 3 — MC
beam transmittance $= e^{-\tau}$ on **both grids and both majorant modes**;
invariant 4 — `j_mc`'s track-tallied $F$/heating close globally in T4.
Invariant 5 has a sharper statement from the matrix (F11): the unigrid
`chain` transport carries an $h$-**independent** path error (volume-mean
$J$ +3.1% above the exact-march value, unchanged from $u12^3$ to $u24^3$)
— the octree exact-characteristic limit is the converged one.

1. **$\sum_i w_i = 4\pi$ over the direction grid.**
   $\left|\sum_i w_i - 4\pi\right|$ = 3.6e-15 / 1.8e-15 / 7.1e-15 / 1.4e-14
   for 12 / 42 / 162 / 642 ordinates — machine precision.

2. **$\oint \Phi\, d\Omega = 1$ for the HG phase function.**
   Analytically exact by construction. Discretely at 42 ordinates: raw
   weighted row sums are exactly 1 at $g = 0$, 1.000–1.002 at $g = 0.5$, and
   **16.9–21.6 at $g = 0.952$** — the reason for row normalization (§4);
   after normalization the row sums are 1 to ≤ 3.3e-16.

3. **$\kappa_s = 0$, $B = 0$ → transmittance $= e^{-\tau}$ along every ray.**
   Octree exact march (uniform $\kappa_e = 0.7$, periodic box, all 42
   ordinates): $\max|I - e^{-\tau}| = 9.7\times 10^{-16}$, max $\tau$ error
   $7.1\times 10^{-14}$ — machine precision.
   (For contrast, the unigrid lattice stencil on the same test has path
   errors up to $\tau \approx 34$ for grazing ordinates — the documented
   stencil error, not a bug; see invariant 5.)

4. **div $F$ equals the heating rate from the intensity field.**
   Cloud + ground + beam run (phase 5 setup; reproduced in §12): pointwise
   div $F$ vs $4\pi\kappa_a(B-J)$ in the interior —
   **rms 1.6%, max 13.9%** of $\max|$heating$|$. Global closure (face fluxes
   + gas absorption − beam first-scatter input): **2.96%** of the incident
   diffuse power (phase 5 record: 3.2%). Both are discretization errors of
   the 42-ordinate/$24^3$ solve, not conservation failures.

5. **Octree converges to the uniform-grid result on the Gaussian cloud.**
   With identical piecewise-constant fields, the $J$ difference between
   octree (exact characteristics) and unigrid (lattice stencil) is ≈ **18%**
   (phase 3 record; §12 band means show 0.7–29% depending on height).
   Interpretation: this is the **unigrid grazing-ordinate stencil error** —
   the best-aligned center-to-center characteristic of a 42-ordinate grazing
   ray deviates strongly from the true ray on a $24^3$ lattice — *not* an
   octree error. The octree self-convergence series (phase 3, cell 2b) and
   the column-$\tau \to 10$ check confirm the octree side; the coarse
   uniform reference under-resolves the cloud (column $\tau$ 9.58).

## 16. Known limitations and performance

**Angular resolution.** 42 ordinates is the development grid but not a
converged solution at large $g$: the HG lobe at $g \approx 0.95$ is
under-resolved (≈ 6% $J$ error in the cloud case, phase 6 record), and even
**642 ordinates are not fully converged at $g = 0.952$**; angular
convergence is **non-monotone at large $g$** (error cancellation between
ordinates — do not extrapolate monotonically). The row normalization (§4)
deliberately trades phase-function fidelity for discrete energy conservation.

**Variable surface.** `ground_surface(height=f(x,y))` is a hook only; the
transport supports a flat ground at the domain bottom, and every solver —
`lambda_iteration` (SC) and `j_mc` / `radiance_from_source` /
`backward_detector_mc` (MC, `mc.ground.ground_or_raise`) — raises
`NotImplementedError` otherwise (plan phase 6 hook).

**Octree chain transport.** `build_chain_octree` / `sweep_all_octree` are
kept as phase-3 reference only: their center-to-center connection has a
scale-invariant path error ($\tau \approx 4$ on the cloud) that refinement
does not remove. The solver always uses exact characteristics
(`precompute_octree_march`). The unigrid `chain` scheme likewise carries an
$h$-independent path error (F11, §15); compare against the octree or the
single-cell `sc` scheme, not against `chain`, when transport accuracy
matters.

**Performance** (this machine, phase 6 record + re-measured):

- Transport sweep cost ≈ **2.1–2.3 ms per ordinate per $24^3$ sweep**
  (re-measured for this manual: 2.6 ms/ordinate including loop overhead),
  **linear in $N_{\mathrm{dir}}$**; chain/march precomputation is built once
  and reused.
- Full Ng cloud run ($24^3$, ground + beam) at **642 ordinates ≈ 84 s**
  (phase 6 record); the §12 example: unigrid 42 ordinates ≈ 4 s, octree
  (25 264 leaves) ≈ 100 s — both on the Windows machine; the §12 macOS run is
  ~2.5× faster.
- The octree DDA march cost grows with the *finest* cell size (the step
  budget scales as $\text{path}/h_{\min}$), so very deep octrees
  (max_depth ≥ 4 with small $h_0$) get disproportionately expensive — the
  documented cloud runs use $h_0 = 1.25$, max_depth = 3.
- **numba is installed and on by default for the SC octree** (`jit=None`):
  the fused march + formal solution turns a 443 s precompute into ≈ 5.5 s
  at 489k leaves (§5 performance note); the unigrid chain/`sc` sweeps run
  through bit-identical compiled kernels, and `jit=False` forces the pure
  NumPy references (parity locked by `tests/test_jit.py`). The MC fast path
  is **opt-in** (`jit=True`, §9). Before/after tables:
  `scripts/bench_compare_walltimes.py`, `scripts/mc_phase4_walltimes.py`.

**MC-specific caveats** (details in the findings log,
`plan/solver-test-matrix-plan.md`):

- Every MC result is statistical — always carry the 1-$\sigma$ batch error
  bars. Per-cell z-scores are heavy-tailed (F3): use the pooled / masked
  statistics (`tests/matrix_stats.py`), never a per-cell maximum.
- The λ variant's `converged` flag is **not meaningful** (F5): with fresh
  sampling per iteration the residual floors at the tally noise. Validate
  its $J$ field instead; `crn=True, birth="uniform"` can stabilize the
  residual but freezes a noisy operator on inhomogeneous fields.
- Every SC-vs-MC per-ordinate comparison *with the beam* on the 42-ordinate
  development grid is **F14-limited, not noise-limited** (SC's
  quadrature-normalized vs MC's analytic HG first-scatter source: defect
  $s(g) = 4.0$–8.9 at $g = 0.9$–0.925); the families only come together as
  $n_{\mathrm{refine}}$ grows (T5).
- Adjoint cone averages are comparable to quadrature node values only with
  `cone_scale` small (default 0); full equal-area caps cross the horizon
  (§9).
- MC neither uses nor needs delta-M — its transport is angle-exact — which
  means MC and SC carry *different angular errors* at fixed
  $n_{\mathrm{refine}}$; expect the two families to converge toward each
  other with the direction ladder, not at the development grid.
- The history budgets (`n_histories`, `n_boundary_histories`,
  `n_beam_histories`, `n_ground_histories`) are separate populations; cost
  is linear in their sum, per iteration for the λ variant.

**Other caveats.**

- `allsky` is unigrid-only; `flux_divergence`, `face_powers` and
  `energy_balance` work on both grids (octree: per-boundary-leaf face
  extrapolation, F7).
- `energy_balance`'s `residual` does not include the beam first-scatter
  input; subtract $\int \kappa_s F_0 e^{-\tau_b}\,dV$ for the diffuse
  closure (§12).
- A single-cell unigrid dimension has nominal width 1.0 (slab convention);
  keep $n_x = n_y = 1$ only with periodic sides.
- The convergence history is the relative change of the *swept* source; with
  Ng active the swept source is the extrapolated one, so `history` measures
  the plain-LI residual of the accelerated iterate.
- `rt3d.mc.estimators.heating_rate(kappa_a, B, J)` and
  `rt3d.diagnostics.heating_rate(kappa_a, J, B)` share a name but **swap
  the last two arguments** (finding F8) — always pass by keyword if unsure.
