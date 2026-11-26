# Plan — heating rates from the local estimator + analytic beam split

Status: **planned, NOT implemented** (written 2026-11-26). This is the execution plan
for `plan/heating-beam-split-plan.md` (findings F15–F18, decisions D1–D5), incorporating
the review corrections to `docs/heating-rate-estimators.md`. No code is changed by this
plan.

Companion documents:

- `docs/heating-rate-estimators.md` — theory: moment identity, local estimator, noise
  budget, beam split, F14 caveat.
- `plan/heating-beam-split-plan.md` — findings F15–F18, decisions D1–D5, acceptance
  criteria H1–H5.
- `plan/solver-test-matrix-plan.md` / `docs/solver-matrix.md` — findings F1–F14.

---

## 1. Goal

Produce the physically complete gas heating

$$
H(\mathbf r) \;=\; \underbrace{4\pi\kappa_a\big(B-J_d\big)}_{\text{diffuse, already computed}}
\;-\;\underbrace{\kappa_a F_0 e^{-\tau_b}}_{\text{analytic, missing}}
$$

from both solver families, with per-cell `sigma`, tests and matrix reporting. Keep the
existing `heating` key meaning **diffuse** for back-compatibility; add `heating_beam`
and `heating_total` alongside.

The estimator rule is fixed by `docs/heating-rate-estimators.md` §11: the production
heating rate is the local form `4πκ_a(B−J)` from the fluence tally, never `∇·F` of an MC
flux field.

---

## 2. Current state

| piece | where | state |
|---|---|---|
| diffuse `J` + `sigma_J` | `rt3d/mc/estimators.py:track_tally` | done |
| diffuse `heating` | `rt3d/mc/forward.py:_common_output:830` | emitted, no `sigma`, no test |
| SC diffuse `heating` | `rt3d/diagnostics.py:heating_rate`, `energy_balance` | done |
| beam optical depth | `rt3d/transport.py:beam_optical_depth{,_octree}` | done |
| beam ground irradiance | `rt3d/transport.py:beam_irradiance_ground` | done |
| hand-written closures | `scripts/phase5_ground.py:174-208` | done, not a function |

Missing: `heating_beam`, `heating_total`, `sigma_heating`, SC
`energy_balance(beam=...)`, tests, matrix columns.

---

## 3. Design decisions

- **D1** — `beam_heating_rate` lives in `rt3d/diagnostics.py`, next to `heating_rate`;
  re-export from `rt3d.mc.estimators` only if the MC outputs need it locally.
- **D2** — `energy_balance(..., beam=None)` is in scope (H3).
- **D3** — δ-M: derive from the code's source convention, verify with H5; do not apply
  silently for `M ≥ 1`.
- **D4** — do **not** rename `heating` in this change.
- **D5** — additive keys only; no existing key changes meaning.
- **F8** — settle the two `heating_rate` argument orders before adding a third helper.

### Review corrections already applied to the theory document

These are folded into this plan so it does not inherit the earlier errors:

- §5.1 diffuse source is `κ_a B`, not `κ_a 4πB` (the 4π appears only after the angular
  integral).
- §5.3 "vacuum above a cloud" corrected to "tenuous air just above a cloud" — in true
  vacuum `κ_a = 0` and the beam term vanishes.
- §7 SC normalisation clarified: the row-normalisation closes the *incoming* integral
  needed by the source evaluation; the *outgoing* integral needed by the moment identity
  is closed only up to the direction spread of the row sums (measured 4.01–5.05 at
  `g = 0.9` on 42 ordinates, 22 %).
- §3.3 `implicit_capture` rewrite: deposit `w_old(1−ω)` at each collision (or
  `κ_a w ds` per segment), not a "true absorption" event.
- §5.6 δ-M wording: it approximates the forward peak as unscattered in the *diffuse*
  transport; it does not add it to the analytic beam term.
- §6 `(L/h)^2` history-count statement qualified by the diffusion-regime
  `(κ_tr L)^2` factor.
- §9 band noise adds in quadrature only for independently sampled bands (CRN
  correlates them).

---

## 4. Steps

### Step 0 — canonical helpers and the F8 guard

Add to `rt3d/diagnostics.py`:

```python
def beam_heating_rate(kappa_a, F0, tau_b):
    """-kappa_a * F0 * exp(-tau_b).  Positive = radiation gains from matter."""
    return -np.asarray(kappa_a, float) * float(F0) * np.exp(-np.asarray(tau_b, float))
```

Settle F8 with the smallest safe change. `rt3d.mc.estimators.heating_rate` is used only
at `rt3d/mc/forward.py:830` and re-exported at `rt3d/mc/__init__.py:27`; no test imports
it. Recommended: make it a thin alias of the diagnostics helper with the same
`(kappa_a, J, B)` order, update `forward.py:830`, and add a sign-pinning test. Fallback
if zero API change is required: keep both helpers, add a test that asserts
`heating_rate(kappa_a, J, B) == 4πκ_a(B−J)` for each, and give `beam_heating_rate` a
distinct name so the hazard does not grow.

### Step 1 — MC: compute `tau_b` once, emit the new keys

**Thread `tau_b`:**

- *Explicit variant*: at the start of the `if beam is not None:` block
  (`forward.py:~710`), compute
  `tau_b = _beam_tau(grid, beam, kappa_e, periodic_box)` and reuse it for the ground
  irradiance (`~737`) and for `heating_beam`.
- *λ variant*: `_beam_j_analytic` already computes `tau_b`. Change it to return
  `(J_direct, tau_b)`, or compute `tau_b` once before `_common_output`. Then reorder:

```python
J_dir, tau_b = _beam_j_analytic(...)
out = _common_output(res_tot, kappa_a, B,
                     heating_beam=beam_heating_rate(kappa_a, beam["F0"], tau_b))
out["J_direct_analytic"] = J_dir
```

**Extend `_common_output` (`forward.py:823`):**

```python
def _common_output(res, kappa_a, B, heating_beam=None):
    heating = heating_rate(kappa_a, res["J"], B)   # diffuse, unchanged
    out = {
        "J": res["J"],
        "sigma_J": res["sigma_J"],
        "J_total": res["J"],
        "sigma_J_total": res["sigma_J"],
        "heating": heating,
        "sigma_heating": 4.0 * np.pi * np.asarray(kappa_a) * res["sigma_J"],
    }
    if heating_beam is not None:
        out["heating_beam"] = heating_beam
        out["heating_total"] = heating + heating_beam
        out["sigma_heating_total"] = out["sigma_heating"]   # beam term is noise-free
    ...
```

Vacuum guard: `κ_a = 0` gives `heating_beam = 0`; no division.

### Step 2 — SC: `energy_balance(..., beam=None)`

Extend `rt3d/diagnostics.energy_balance` (`diagnostics.py:258`). When `beam` is given:

```python
tau_b = beam_optical_depth{,_octree}(grid, beam["omega_sun"], kappa_e, ...)
beam_scat = float(np.sum(kappa_s * beam["F0"] * np.exp(-tau_b) * vol))
gas_beam  = float(np.sum(kappa_a * beam["F0"] * np.exp(-tau_b) * vol))
beam_at_ground = float(np.sum(beam_irradiance_ground(...)) * area)
heating_beam = -kappa_a * beam["F0"] * np.exp(-tau_b)   # per cell
heating_total = heating + heating_beam                   # per cell
```

Return `beam_scat`, `gas_beam`, `beam_at_ground`, `heating_total`, and the two residuals
(`res_dif`, `res_bim`) from `plan/heating-beam-split-plan.md` §3. Keep `heating` diffuse.
Then `scripts/phase5_ground.py` can call the function instead of its hand-written block.

### Step 3 — tests

| test | file | assertion |
|---|---|---|
| **H1a** | `tests/test_mc_forward.py` | slab, `B=0`, no ground, beam on, `κ_a>0`: `Σ(heating_total−heating)·V == −∫κ_a F0 e^{−τ_b}dV` to machine precision; `Σ heating_total·V` equals `−(out_top+out_bottom)` within the existing slab range |
| **H1b** | `tests/test_mc_forward.py` | closed form `κ_a F0 μ_0^{-1}(e^{−τ_0}−e^{−τ_top})` per unit area |
| **H1c** | `tests/test_mc_forward.py` | `sigma_heating == 4πκ_a sigma_J` exactly |
| **H1d** | `tests/test_mc_forward.py` | explicit vs λ: `heating_total` within combined `2σ`, better than `heating` alone |
| **H2** | `tests/test_mc_forward.py` | `J_direct` vs `J_direct_analytic` regression guard |
| **H3** | `tests/test_matrix_closure.py` | matrix case, `beam=True`, octree d2: beam closure `≤ 0.5 %` of `in_beam`; diffuse closure unchanged |
| **H4** | `tests/test_mc_forward.py` | `κ_e=0` cells: `heating_beam==0`, finite, no NaN in `heating_total` |
| **H5** | `tests/test_delta_m.py` | `delta_M∈{0,1}`, cloud + beam: beam closure within the `delta_M=0` range; record any growth |
| **F8** | `tests/test_diagnostics.py` | sign and argument-order pin for both `heating_rate` helpers and `beam_heating_rate` |

### Step 4 — matrix wiring

- `tests/matrix_cases.py` / `tests/matrix_stats.py`: expose the two closures and
  `heating_total` so pytest and the report share them.
- `scripts/solver_matrix.py`: add `heating`, `heating_beam`, `heating_total` columns to
  the CSV; one figure of the solar column.

### Step 5 — docs and decisions log

- Update `docs/heating-rate-estimators.md` §11 coverage map (remove "missing (F15)" once
  done).
- Add the three decisions from `plan/heating-beam-split-plan.md` §8 to `plan/plan.md`.
- Mark `plan/heating-beam-split-plan.md` status when implemented.

---

## 5. Acceptance criteria

The H1a–H5 table above, plus:

- `heating` still means diffuse everywhere (existing plots/reference ranges unchanged).
- `heating_total` is finite everywhere, including vacuum and the octree `κ_e→0` case.
- `sigma_heating_total == sigma_heating` exactly.
- No `div F` of an MC flux field is used as an estimator.
- No second `tau_b` computation in a beam run.

---

## 6. Non-goals (unchanged)

Face-current divergence tally, next-event fluence estimator, adjoint heating response,
band/k-distribution summation, renaming `heating` → `heating_diffuse`, and any
deposition-style heating tally.

---

## 7. Risks / open questions

1. **F8 API**: `rt3d.mc.estimators.heating_rate` is re-exported. Decide
   alias-with-same-order vs keep-both-with-a-test before H1.
2. **`_beam_j_analytic` signature**: only one call site (`forward.py:681`), so returning
   `(J, tau_b)` is contained; verify no script imports it.
3. **Explicit variant with no ground**: `tau_b` is currently computed only for the
   ground branch; hoist it to the beam block.
4. **δ-M**: the derivation predicts the beam term survives unchanged, but H5 is the gate;
   do not apply for `M ≥ 1` without it.
5. **Sign convention**: keep the repository's positive = radiation gains (gas cooling);
   the beam term is negative. The F8 test should pin this.
