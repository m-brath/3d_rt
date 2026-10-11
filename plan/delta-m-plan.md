# Plan — delta-M scaling for strongly forward-peaked phase functions

Delta-M (Wiscombe) scaling of the Henyey–Greenstein phase function: subtract the
first `M` Legendre moments from the scattering operator and reduce the transport
extinction accordingly. Purpose: cases with `g ≳ 0.5` (cloud core `g = 0.952`) —
the HG lobe is unresolved on the 42-ordinate grid, the effective optical depth is
overestimated by the un-split forward peak, and the lambda-iteration scattering
operator converges slowly (the `hg_matrix` row-normalization hack, decisions log
"HG quadrature row normalization", only keeps it subcritical).

## 1. Math (exact transformation, HG phase function)

HG Legendre moments are `g^n`. Delta-M of order `M` subtracts the first `M`
moments:

```
Phi_delta(Theta) = Phi_HG(Theta) - (1/4pi) * sum_{n=0}^{M-1} (2n+1) g^n P_n(cos Theta)
kappa_e_delta    = kappa_e - kappa_s * g^M      # used by ALL diffuse transport
```

**Only the scattering part of the extinction is scaled.** The phase function
applies to scattering alone, so keeping absorption unscaled

- leaves pure absorption (`kappa_s = 0`) exactly Beer-Lambert — validation
  anchor 3 holds with delta on (full-extinction scaling `kappa_e (1 - g^M)`
  breaks it at O(tau_cell) via the explicit term), and
- keeps the explicit term's LI gain `omega g^M (1 - g)/(1 - omega g^M) < 1`
  for all `omega < 1` — full-extinction scaling has isotropic-mode gain
  `(omega - g^M)/(1 - g^M)` and **diverges for `omega < g^M`** (measured).

The RTE `dI/ds = -kappa_e I + kappa_a B + kappa_s int Phi I dOmega` is **exactly**
equivalent to

```
dI/ds = -kappa_e_delta * (I - S_delta)

S_delta = [ (1-omega) B
            + omega * sum_{n<M} (2n+1)/(4pi) int P_n(Omega.Omega') I(Omega') dOmega'
            + omega * int Phi_delta I dOmega'
            - omega * g^M * I(Omega) ] / (1 - omega * g^M)
```

(per-cell factors — `omega` varies by cell). Sanity checks:
- isotropic field: `kappa_e (S - I) = -kappa_a I` (correct absorption),
- `kappa_s = 0`: `kappa_e_delta = kappa_e` and the explicit term vanishes —
  anchor 3 is *exact*, not approximate.

`M = 1` (delta-Eddington): the moment sum is just `J` (already computed by
`mean_intensity`). `M = 2` adds `3 Omega.F / 4pi` (already computed by
`flux_vector`). The `-omega g^M I / (1 - omega g^M)` term is explicit (uses the
swept `I`, same as the scattering integral) — standard in delta-M discrete-
ordinate codes. Its short-characteristics quadrature error is O(tau_cell):
delta-on vs delta-off converged J differs by 3.9e-2 (nz=8) -> 9.2e-3 (nz=16)
-> 2.1e-3 (nz=32) on the tau=5 slab, shrinking ~linearly with h.

## 2. Key design decision: split the *existing normalized* operator

`hg_matrix(g, normalize=True)` (row-normalized) stays the definition of the
discrete truth. The delta split is defined **on the normalized matrix**:

```
Phi_w_delta = dg.hg_matrix(g) - sum_{n<M} (2n+1) g^n * P_n(Omega_i.Omega_j) w_j / (4pi)
```

so `Phi_w = Phi_w_delta + moments` exactly. Consequences:

- delta-M changes convergence and transport scaling, **not** the converged
  discrete solution — delta-on and delta-off must agree at convergence;
- at `M = 1` the residual operator has row sums ≈ 0 (vs 1 today), which removes
  the supercritical-scattering problem the row-normalization hack only papered
  over at `g ≈ 0.95`.

## 3. Code changes (as implemented)

- **`rt3d/directions.py`**: `direction_grid.hg_delta_matrices(g, M)` returning
  `(Phi_w_delta, [P_hat_n_w for n < M])` with `moments[n][i,j] =
  (2n+1) g^n P_n(Omega_i.Omega_j) w_j / (4pi)` (classical Wiscombe weights),
  `P_n` via Legendre recurrence on `cos_theta()`. `hg_matrix` untouched.
- **`rt3d/source.py`**:
  - `delta_extinction(kappa_e, kappa_s, g, M)` -> `kappa_e - kappa_s * g^M`
    (per-cell for `g_cells`, guarded `|g| < 1`; `kappa_s = 0` -> unchanged).
  - `scattering_source(..., delta_M=0)` gains the delta path: per-cell scale
    `1/(1 - omega g^M)`, moment terms, `Phi_w_delta` matmul, and the
    `-omega g^M I` term. Bucketing / `Phi_cache` machinery reused (splits
    cached under tuple key `(g_value, delta_M)`).
  - `beam_scattering_source(..., delta_M=0)`: albedo factor becomes
    `omega / (1 - omega g^M)` (it is `kappa_s Phi F0 e^-tau_b / kappa_e_delta`);
    `tau_b` stays the **physical** beam optical depth (the direct beam
    attenuates with total extinction); the phase vector stays the full
    normalized `Phi` (the beam is not on the quadrature, the moment
    subtraction applies only to the diffuse field).
- **`rt3d/solver.py`**: option `delta_M=0` (0 = off). When `> 0`: compute
  `kappa_e_delta` once and pass it to **all diffuse** sweeps; the beam keeps
  physical attenuation (`beam_optical_depth(_octree)` and
  `beam_irradiance_ground` use physical `kappa_e`); split operators built once
  (per bucket); result dict records `delta_M` and `kappa_e_delta`. Octree
  refinement stays driven by the **physical** `kappa_e` (conservative, no
  octree changes).
- **Ng accelerator**: no code change (it acts on source history); covered by a
  test. Measured: Ng + delta-M is the fastest combination at large g
  (g = 0.95: Ng-only 82 iters, Ng+delta 11); at small g delta-M slightly slows
  Ng (g = 0.5: 31 -> 59) — use delta-M for `g >~ 0.5` only.
- **Runner scripts**: `DELTA_M = 0` config cell in `phase2_slab.py`,
  `phase4_ng.py`, `phase6_directions.py` (scripts are `# %%` cell scripts, no
  argparse); `gaussian_cloud` (core `g = 0.952`) is the showcase case.

## 4. Tests (`tests/test_delta_m.py`)

1. **Equivalence**: slab, `g = 0.8/0.9`, `omega = 0.5/0.9`: `delta_M=1` vs off ->
   same converged `J` (tight tolerance). Correctness anchor.
2. **Acceleration**: slab `g = 0.95`, `omega = 0.9`: iteration count with
   `delta_M=1` far below plain LI.
3. **Anchor 3 with delta on**: `kappa_s = 0` -> transmittance `exp(-tau)` with
   physical `kappa_e` (the `-g^M I` term must restore it).
4. **Anchor 4**: `div F` = heating rate on a delta-on run.
5. `M=2` equivalence + moment-operator unit test (`P_hat_0,w` row sums = 1,
   `P_hat_1,w` row sums ≈ 0).

## 5. Docs

- `plan/plan.md`: decisions-log entry (delta-M formulation, split-of-normalized-
  operator choice, explicit `-g^M I` term, octree refinement on physical
  `kappa_e`, measured iteration counts).
- `MANUAL.md`: section on `delta_M`, recommendation to enable for `g ≳ 0.5`.

## 6. Choices

- **Default**: `delta_M = 0` off, opt-in — keeps every existing test/number
  reproducible.
- **Order**: implement general `M` (cheap via Legendre recurrence), validate and
  recommend `M = 1`; test `M = 2` as well.

## 7. Work breakdown with subagents

Delegation follows the AGENTS.md subagent rules: **at most 2 subagents in
parallel**, **never reuse a subagent** (fresh conversation per task, every
prompt fully self-contained — the subagent gets the relevant math from section 1,
the API sketch from section 3, and the environment notes below, not this
conversation).

Dependency DAG and schedule:

| Step | Subagent | Scope | Runs |
|---|---|---|---|
| 1 | `general` A — **core operators** | `rt3d/directions.py` (`hg_delta_matrices`), `rt3d/source.py` (`delta_extinction`, delta path in `scattering_source`, `beam_scattering_source` scale factor); runs its own quick sanity checks | alone |
| 2a | `general` B — **solver integration** | `rt3d/solver.py` `delta_M` option: `kappa_e_delta` into all sweeps + beam tau + ground beam irradiance, result dict keys; runner-script `--delta-M` flags | in parallel with 2b |
| 2b | `general` C — **tests** | `tests/test_delta_m.py` (all five tests of section 4); writes against A's frozen API | in parallel with 2a |
| 3 | main agent (inline) | run the full pytest suite, verify anchors 1–4, measure iteration counts delta-on vs off on the slab and the Gaussian cloud | — |
| 4 | `general` D — **docs** | `plan/plan.md` decisions-log entry (with the measured numbers from step 3), `MANUAL.md` `delta_M` section | alone, after 3 |

Rules of engagement:

- Step 1 must finish before 2a/2b start: A freezes the public API
  (`hg_delta_matrices(g, M) -> (Phi_w_delta, [P_hat_n_w])`,
  `delta_extinction(kappa_e, kappa_s, g, M)`, `scattering_source` /
  `beam_scattering_source` `delta_M` kwargs) and both
  2a and 2b prompts quote it verbatim.
- Each subagent prompt includes: the delta-M math of section 1, the design
  decision of section 2 (split of the *normalized* `hg_matrix`), the repo
  conventions (NumPy docstrings, `# %%` cells in `scripts/`, no new
  dependencies), and the environment recipe — run
  `C:\Users\fredd\miniforge3\envs\pyarts\python.exe` with
  `C:\Users\fredd\miniforge3\envs\pyarts\Library\bin` prepended to `PATH`
  (BLAS delay-load gotcha), `matplotlib.use("Agg")`, figures to `plot/`.
- Subagents do not commit; the main agent reviews diffs, runs the suite and
  commits after step 3.
- Fallback: if the equivalence test (section 4.1) fails, the main agent
  debugs inline rather than re-delegating (single source of truth for the
  math).
