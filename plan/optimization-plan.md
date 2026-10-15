# Plan — performance optimization of the `rt3d/` solver

Analysis of the solver package for speed. Based on profiling on this machine
(14 cores, pyarts env) with the `gaussian_cloud` case (omega_cap 0.7, 42
ordinates, Ng on) — see the measured baseline below.

## 0. Environment note

`numba 0.66.0` (+ `llvmlite 0.48.0`) is **already installed** in `pyarts` and
JITs correctly with Python 3.14 / numpy 2.5 (verified by a jit smoke test).
The AGENTS.md package table lists numba as "not installed" — that entry is
outdated and should be corrected when this plan is executed. No *further* new
dependencies are needed.

## 1. Measured baseline (this machine)

| Stage | Cost | Notes |
|---|---|---|
| `precompute_octree_march` | **5–8 s per ordinate → 4–6 min for 42 dirs** | octree, 489k leaves (h0=1, max_depth 4) |
| octree march memory | **~26 GB** for 42 dirs | 374 MB/dir, 26–37M segments/dir |
| octree exact sweep | ~1.1 s per ordinate → ~45 s/iteration | memory-bound on the segment arrays |
| `balance_21` | 11 s at 489k leaves | 54 `locate()` probes/leaf/pass |
| unigrid chain sweep (42 dirs) | 0.07 s @30³, 0.21 s @50³ | already efficient |
| unigrid SC sweep (42 dirs) | 0.43 s @30³, 1.14 s @50³ | ~5× slower than chain |
| `build_chain` memory | 228 MB @50³, ~1.5 GB @100³ | `sub_cells` int64 (M, n_cells) |
| scattering matmul, scalar g | 0.05 s/iter | BLAS — fine |
| scattering matmul, `g_cells` | 0.27 s/iter | 90 buckets × matmul + masks |

Micro-benchmarks (30M-element arrays): `np.add.at` 0.05 s, `np.exp` 0.10 s,
one full segment body (gather ×2 + expm1 + exp + scatter) ~0.5 s — i.e. the
octree sweep cost is ~8 such passes per ordinate per iteration.

## 2. P1 — rewrite the octree march (`octree.py: march_octree`, `locate`)

**Root cause.** The DDA loop re-gathers all active rays every step, and after
every face crossing calls `oct.locate(pp)` — a *full descent from the base
lattice* (max_depth+1 gather passes over all K = n_leaf rays), although a ray
only ever needs to descend from its own current node. Total cost ≈
M × depth × n_leaf gathers (30M × 4 × 489k).

**Fix A (preferred): numba.** Per-ray compiled DDA with *incremental*
descent/ascend (track the current node; cross one face → step to the
neighbouring node in the tree). Additionally **fuse march + formal-solution
accumulation**: sweep tau/contrib directly into the per-leaf outputs, so the
segment arrays (`leaf/length/ray_of/offsets`, 26 GB for 42 dirs) are never
materialised — precompute becomes ~seconds and its memory ~zero.
Keep the existing NumPy march as reference implementation; gate the numba
path behind a flag with automatic fallback.

**Fix B (fallback, pure NumPy):** keep per-ray node state; descend only from
the current node; bucket rays by current node so gathers touch only the
active subset.

**Validation:** segment lists (or fused tau/contrib) bit-comparable to the
reference march on the phase-3/5 octree tests; anchor 3
(`kappa_s = 0, B = 0 → exp(-tau)`).

## 3. P2 — segment-array size and sweep fusion

- **Merge consecutive segments in the same leaf** (the DDA re-locates after
  every internal plane crossing; collinear same-leaf runs are currently
  stored as separate segments). Reduces M measurably on deep octrees.
- **int32 / float32** for `leaf`/`offsets`/`length` where accuracy allows
  (tau accumulation stays float64) — halves the sweep's memory traffic.
- **Fuse the sweep expression** in `sweep_direction_octree_exact`
  (`tau_seg`, `expm1`, two `exp`, two `add.at`) into fewer passes with
  preallocated buffers (no per-call temporaries of M elements).

## 4. P3 — parallelize over ordinates

The 42 (or 162/642) ordinates are fully independent in `sweep_all*`,
`precompute_octree_march` and the solver's per-iteration direction loop.
- Threads (numpy releases the GIL for the big elementwise/gather ops) with
  shared read-only chain/march arrays and disjoint output columns `I[:, d]`;
  or `multiprocessing` for the precompute (chains/marches are picklable
  arrays).
- Expected near-linear up to ~8–12× on this 14-core machine; applies to
  octree sweeps (45 s/iter → ~5 s/iter) and to the unigrid sweeps too.
- Keep a `n_jobs` option on the solver config; default 1 to preserve
  current behaviour, record it in outputs.

## 5. P4 — unigrid chain memory (`characteristics.build_chain`)

`sub_cells` stores absolute int64 indices, shape (M, n_cells): 228 MB @50³,
~1.5 GB @100³. The sub-segment offsets are the *same relative offsets* for
every cell (already computed as `rel`): store `rel` as int16 plus per-cell
base flat indices, and form absolute indices in the sweep as
`base[wave] + row_delta` (row_delta = ravel of rel offset — one small
precomputed int per sub-segment row). Cuts chain memory ~4–20× with the same
sweep structure.

## 6. P5 — SC sweep (`short_characteristics.sweep_direction_sc`)

- `np.any(face)` + up to 3 `np.where` per stencil node per layer *every
  pass*: faces are static per chain — precompute per-chain boundary node
  lists (or a packed int8 code) and apply boundary values only at the few
  boundary nodes instead of full-array `where`.
- Wrap-pass loop re-sweeps all layers; keep, but the hoisted boundary
  handling removes most of the per-pass overhead. Target: SC ≈ chain cost.

## 7. P6 — `g_cells` scattering source (`source.scattering_source`)

90 buckets × (matmul + boolean mask) per iteration (0.27 s at 489k cells).
The HG operator is spectrally low-rank: its Legendre moments are exactly
gⁿ (already used by `hg_delta_matrices`). Replace the bucket loop by a
truncated moment expansion:
`S_scat ≈ sum_{n<K} (2n+1) P_n(cosθ) applied via (I @ L_n)` with per-cell
gⁿ factors — K dense (n_cells, N_dir) @ (N_dir, K) matmuls, K ~ 10–20 for
g ≤ 0.95 at 1e-6. Keep the bucket path as reference; switch by option.

## 8. Minor items

- `stencil_for_direction`: pure-Python loop over (2·r+1)³ (r up to 24 →
  121³ ≈ 1.8M iterations/ordinate). Vectorize with meshgrid + lexsort, or
  numba. (0.02 s at r=12, but minutes at r=24 on anisotropic grids.)
- `balance_21`: 54 full-length `locate()` probes per leaf per pass; the
  3×3 transverse probe grid can be reduced (edge/corner probes only) or the
  probes batched into one `locate` call per (axis, side) instead of 54.
- `ng1d`: `eigvals` per iteration is O(Nz³) — negligible at Nz ≤ 64, keep.
- AGENTS.md: correct the numba row after this plan lands.

## 9. Suggested execution order

1. P1 numba march (+ reference-parity tests) — biggest single win
   (precompute 4–6 min → ~10 s; 26 GB → ~0).
2. P2 segment merging + sweep fusion.
3. P3 parallel sweeps (largest remaining multiplier).
4. P4 chain memory, P5 SC sweep, P6 g_cells source.
5. Re-run `scripts/compare_transport.py` wall-clock table as the before/
   after benchmark.

## 10. Validation gates (all steps)

- Full `pytest tests/` suite green after each step.
- Physics anchors of AGENTS.md: sum(w)=4π; HG ∫ΦdΩ=1; exp(−tau) transmittance;
  div F = heating; octree → unigrid convergence on the Gaussian cloud.
- J-field bit/close comparison against stored `compare_transport` results
  (same seeds/settings) — optimization must not change converged solutions
  beyond float32-level effects where float32 is introduced (keep float64
  for tau accumulation).
