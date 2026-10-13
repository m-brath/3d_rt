# Plan — transport-scheme comparison: chain vs. SC vs. octree, with/without delta-M

Comparison of the three transport configurations of the solver on the MANUAL §11
worked example, each run with and without delta-M (Wiscombe) scaling, and a
diagnostic set of mean intensity **J**, angular **variance** of the radiance and
the **upward radiance at TOA**. Interactive 3D views are exported as vtk.js
HTML scenes (pyvista `plotter.save("*.html")`).

## 1. Run matrix (6 runs)

Foundation = MANUAL §11 example, identical setup: `gaussian_cloud(omega_cap=0.9)`,
24³ unigrid, octree `h0=1.25, tau_max=2.0, eps_grad=1.0, max_depth=3` +
`balance_21`, icosahedral `n_refine=1` (42 ordinates), 30° solar beam,
`top_I=1`, Lambert ground albedo 0.3, `lambda_iteration(..., ng=True,
n_iter=60, tol=1e-6)`.

| # | grid | `transport` | `delta_M` |
|---|---|---|---|
| 1–2 | unigrid 24³ | `chain` (lattice-stencil chains) | 0 / 1 |
| 3–4 | unigrid 24³ | `sc` (single-cell short characteristics) | 0 / 1 |
| 5–6 | octree | octree exact marches (`chain`; `sc` is unigrid-only) | 0 / 1 |

Pairings: chain↔SC isolates the transport-algorithm error on identical cells;
unigrid↔octree is validation anchor 5; δ-on↔δ-off checks the delta-M
equivalence claim (same converged solution, fewer iterations at core g≈0.95).

## 2. Diagnostics

- **J**: y=0 slices (all runs sampled onto a common 64×64 raster at the finest
  octree cell, h=0.15625 m), band-mean z profiles (8 base-cell bands, 6 curves),
  pairwise relative-difference maps at the z=4 cloud plane:
  chain↔SC per δ, δ-on↔δ-off per method, unigrid-chain↔octree-chain per δ.
- **Angular variance** σ²(r) = (1/4π)∫(I−J)² dΩ — new reusable diagnostic
  `rt3d.diagnostics.angular_variance(I, dg)` (+ unit tests in
  `tests/test_diagnostics.py`: isotropic field → 0; I = J + a·μ → ≈ a²/3 on the
  icosahedral grid; non-negativity). Same plot family as J.
- **Upward radiance at TOA**: unigrid via `diagnostics.allsky(face="top")`;
  octree via rasterization of the top-layer leaves onto the same 64×64 grid
  (`oct_.locate` at z = z1 − h/2). Outputs: F↑(x,y) maps for all 6 runs +
  difference maps, domain-mean upward radiance vs μ per run, per-ordinate mean
  radiance table (CSV).
- **Efficiency table**: `n_iter`, wall time, converged flag per run
  (printed + `summary.csv`).

## 3. Interactive 3D — vtk.js HTML

All runs are sampled onto one common 64³ lattice (piecewise-constant fields,
exact via cell/leaf lookup), so scenes are directly comparable. pyvista's
vtk.js export does **not** carry volume mappers, so scenes use slice meshes +
isosurfaces (toggleable actors in the viewer sidebar):

- one scene per run: κ_e isosurfaces (0.5/1/2/3 m⁻¹) + J slices (x=0, y=0, z=4)
  + σ² slices (y=0, z=4) + domain wireframe,
- two difference scenes: |ΔJ| isosurfaces + slices for chain↔SC (δ off) and
  unigrid-chain↔octree-chain (δ off),
- static PNG volume renderings (matplotlib-embedded, manual §11 style) as the
  overview figures.

## 4. Deliverables

- `scripts/compare_transport.py` — `# %%` cell runner (setup → 6 runs → table →
  figures → HTML), Agg backend, all output into `plot/compare_transport/`
- `rt3d/diagnostics.py`: `angular_variance()`; `tests/test_diagnostics.py`
- Figures `plot/compare_transport/*.png`, scenes `plot/compare_transport/scene_*.html`
  (+ companion `.vtkjs`), CSVs `summary.csv`, `toa_mean_radiance.csv`,
  `profiles.csv`

## 5. Verification

- `pytest tests/` full suite green (new diagnostics tests included)
- sanity: δ-on vs δ-off converged J agree (delta-M equivalence), TOA mean F_z
  matches the manual §11 numbers (−2.52 unigrid), anchor-5 pattern reproduced
  (octree vs unigrid J differ at the grazing-ordinate level, not the scheme level)
