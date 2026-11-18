# Plan — Update `MANUAL.md` for the MC family, test matrix, and current environment

## Why the manual is stale

`MANUAL.md` documents the SC-only solver ("phases 0–6 complete, 110 tests
green", "numba/pyvista not installed", "pure NumPy"). Since then the repo grew:

- `rt3d/mc/` — Monte Carlo solver family, phases M0–M6 (Woodcock tracking,
  stage-1 `j_mc` λ/explicit, stage-2 `radiance_from_source`, adjoint
  `backward_detector_mc`/`backward_all_sky`, Lambertian ground, M4 variance
  reduction + `rt3d.mc.jitmc` fast path), plan/mc-plan.md
- `rt3d/short_characteristics.py` — second unigrid transport (single-cell SC,
  plan/plan-sc-unigrid.md), selected by `lambda_iteration(transport="sc")`
- `rt3d/cases/slab_1d.py` — the slab case now **does** exist as a case module
  (the manual claims it does not)
- `tests/` — cross-solver pytest matrix T0–T5 (341 tests collected), markers
  `slow`/`mc`/`reference`, findings log F1–F14 in plan/solver-test-matrix-plan.md
- `rt3d/jit.py` numba fast paths are **installed and on by default**
  (`jit=None`); §9/§15 of the manual still say numba is not installed
- `rt3d/diagnostics.py` — `energy_balance`/`flux_divergence` are now
  octree-capable; new `face_powers`, `angular_variance`, `cell_volumes`
- many runner scripts: `mc_*`, `solver_matrix.py`, `compare_transport.py`,
  `sc_transport_figures.py`, `bench_*`, `report_figures.py`
- environment: numba 0.66.0 + pyvista 0.49.1 installed; macOS/Linux paths
  should appear next to the Windows ones

## Changes to `MANUAL.md`

1. **Intro + TOC** — both solver families, 341 tests, numba note; TOC gains
   §9 and Part II renumbers 9–15 → 10–16; fix all internal § cross-refs.
2. **§1** — solver-family overview paragraph.
3. **§4** — MC uses the icosahedral grid for tally binning only; HG sampled
   exactly by inverse CDF (cross-ref §9).
4. **§6** — add the unigrid single-cell SC scheme (`rt3d.short_characteristics`,
   `transport="sc"`); update flowchart.
5. **§7** — one-line cross-ref: the MC family implements the same ground law.
6. **new §9 "Monte Carlo solver family"** — Woodcock cell/global modes, HG
   inverse-CDF, solver A stage 1 (λ / explicit), stage 2 next-flight +
   deterministic estimators, solver B adjoint detector MC, track-length
   tallies + batch error bars, beam/ground as particle populations, M4
   variance reduction (implicit capture, roulette, CRN), measured facts
   (T2 zero-noise identity 1.2e-12; F5 λ-residual floor; F14 beam-HG
   normalization difference), pointers to plans, no findings-log duplication.
7. **§10 environment** — dual-platform commands; package table extended
   (numba/pyvista/trame-pyvista/markdown); "not installed" paragraph replaced
   by "installed, auto-detected fast paths"; test suite = 341 tests with tier
   runtimes from plan/solver-test-matrix-plan.md §10; `pytest -m "not slow"`.
8. **§11 module tour** — new subsections `rt3d.short_characteristics` and
   `rt3d.mc` (per-file + key signatures); fix the `slab_1d` note (it exists:
   `slab_grid`, `slab_profile`, `d1o_J`, `two_stream_J`); extend diagnostics
   tour (`face_powers`, `angular_variance`, octree-capable closures).
9. **§12 worked example** — keep verified SC example unchanged; add a short
   **verified** MC example (slab, `j_mc` λ + explicit vs `d1o_J` reference);
   fix § refs.
10. **§13 runner scripts** — table extended with the mc_*, matrix, comparison
    and figure scripts.
11. **§14 condensed API reference** — tables for `rt3d.short_characteristics`
    and `rt3d.mc.*`; updated `rt3d.solver` (transport/jit/n_jobs),
    `rt3d.diagnostics`, `rt3d.cases` lines.
12. **§15 verification invariants** — intro now points at the T0–T5 pytest
    matrix and findings log; MC companions of invariants 2–4 mentioned.
13. **§16 limitations/performance** — numba is installed + default (`jit=None`);
    closures no longer unigrid-only (`allsky` still is); MC caveats: statistical
    noise + F3 pooled statistics, F5 convergence floor, adjoint cone/`cone_scale`,
    F14 42-ordinate beam cross-family limit, delta-M unnecessary in MC.

## Method

1. Extract exact signatures from `rt3d/` (module tours/API tables must match
   code).
2. Run the new MC slab example with the pyarts interpreter; include only
   verified numbers.
3. Keep style: NumPy-style API tables, mermaid flowcharts, `§` cross-refs.
4. No changes to `rt3d/`, tests, or the plans (except this file); the manual
   cross-references `plan/mc-plan.md` and `plan/solver-test-matrix-plan.md`
   instead of duplicating the findings log.
