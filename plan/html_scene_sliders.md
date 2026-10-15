# Plan — fancy interactive 3D HTML scenes with slice/iso sliders

Target: `scripts/compare_transport.py`, interactive 3D scene section (~lines 654–741).

## Background / findings

- The 12 exported scenes (`plot/compare_transport/scene_*.html`) are static vtk.js
  OfflineLocalView files produced by `pl.trame.export_html()`. Geometry is baked in:
  4 fixed `kappa_e` isosurfaces, 6 fixed slices (J and var at x=0, y=0, z=4), wireframe box.
- Exported HTML tail is exactly `OfflineLocalView.load(container, { base64Str });` and the
  bundled viewer sets `global.renderWindow` after synchronizing the scene → custom JS
  injected into the exported HTML can toggle actor visibility at runtime and re-render.
- `OfflineLocalView.load` does not return the proxy; use `window.renderWindow` +
  `getRenderers()[0]` actors in add_mesh order (validated in browser during implementation).

## Decisions (user, 2026-10-15)

- Precomputed-snapshot sliders (self-contained offline HTML), **higher resolution**:
  ~25 slice positions per axis, ~15 iso levels.
- **Also a z-slice slider** (replaces the fixed z = 4 plane; default stays z = 4).

## Implementation

1. Scene construction (`scene_run`, `scene_diff`):
   - Slices: 25 positions per axis for x, y and z, both scalars (J viridis, var inferno;
     dJ magma in diff scenes). Slice grids stride-2 subsampled to keep files ~7–9 MB.
   - Isosurfaces: 15 precomputed levels (kappa_e levels spanning the cloud; |dJ|
     quantile levels for diffs), nicer shaded material.
   - Default visible state reproduces the old view: x=0, y=0, z=4 slices + mid iso level.
   - Fancy polish: dark gradient background, improved contour lighting/specular, cleaner
     scalar bars, title overlay with run key, same camera.
2. HTML post-processing helper: after `pl.trame.export_html()`, replace the
   `OfflineLocalView.load(...)` tail with load + injected dark DOM overlay panel:
   - sliders x / y / z slice position (live readout in m)
   - slider isosurface level (readout in m^-1 or |dJ|)
   - J/var scalar radio (run scenes), visibility checkboxes slices / isosurfaces / box
   - JS waits for `window.renderWindow`, toggles precomputed actor visibility by group
     (manifest JSON embedded by Python), calls `renderWindow.render()`.
3. Verification: regenerate scenes, open one exported HTML in the browser tool, exercise
   the sliders to confirm vtk.js actor toggling works (validates actor-order/manifest).

Caveat: slider steps are discrete snapshots — inherent to the offline export.

## Status (2026-10-16): DONE

- Implemented as planned (N_SLICE=25/axis incl. z, N_ISO=15, stride-2 slices,
  injected dark control panel; validated in browser for run + diff scenes).
- Fixes during testing: boundary slice planes slice empty (positions kept
  inside); pyvista dedupes scalar bars sharing a lookup table (actor-order
  bookkeeping); manifest self-check in `_finish`.
- Environment: `numba 0.68.0` + `llvmlite 0.50.0` installed into `pyarts` on
  the Windows machine (were missing despite AGENTS.md; without them the
  642-ordinate octree reference solve OOMs on the NumPy fallback path).
- Both `octree_ref` runs re-solved at 642 ordinates and cached; all 12
  `plot/compare_transport/scene_*.html` regenerated (run scenes 2.5–3.1 MB,
  diff scenes 5.7–22.7 MB).
