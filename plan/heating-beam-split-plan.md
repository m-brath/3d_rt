# Plan — direct (flux-free) heating rates and the analytic uncollided-beam split

Status: **planned, NOT implemented** (written 2026-11-26). No file outside `plan/`
and `docs/` is touched by this plan. The theory derivation lives in
`docs/heating-rate-estimators.md`; this file is the implementation spec, the
findings that motivated it, and the acceptance criteria.

---

## 1. The question this plan came from

Heating rates are conventionally obtained by differencing radiation fluxes.  For a
Monte Carlo solve that is the wrong estimator: by the first moment of the RTE the
divergence is a *local* quantity,

$$

\nabla\cdot\mathbf F \;=\; 4\pi\kappa_a (B - J)
$$

which needs only $\int\Phi\,d\Omega = 1$ and holds at arbitrary single-scattering
albedo.  So MC never needs fluxes, cell faces or finite differences to produce a
heating-rate field.

Inspecting the current outputs against that claim turned up a bookkeeping gap
(F15 below) whose physical impact is larger than the variance argument that
prompted the question.

---

## 2. Findings

Follows the numbering of `plan/solver-test-matrix-plan.md` (F1–F14 are taken).
Nothing here is measured on a run yet — F15/F16/F17 are **code-inspection**
findings; F18 is the one that needs a measurement.

**Finding F15 — both families report *diffuse-field* heating; the uncollided-beam
absorption is missing from the field.**

- `rt3d/mc/forward.py:_common_output` (lines 823–835) reports
  `heating = heating_rate(kappa_a, B, res["J"])`, i.e. `4π κ_a (B − J)` from the
  **diffuse** tally only.
  - λ variant: the uncollided beam is never tallied — `J_direct` is zeros
    (`forward.py:677`) and only `J_direct_analytic` (`forward.py:681`, from
    `_beam_j_analytic`, `forward.py:838`) exists, and it never reaches `heating`.
  - explicit variant: uncollided beam tracks go to `tally_direct` and reach only
    `J_total` / `F_total` (`forward.py:779–785`), never `heating`.
- Same split on the SC side: `rt3d/solver.lambda_iteration` transports the diffuse
  field (`I = I_diff`, beam handled analytically through
  `rt3d.source.beam_scattering_source`), so
  `rt3d.diagnostics.energy_balance` (`diagnostics.py:258`) integrates diffuse
  heating.
- The **total** field is (derivation in `docs/heating-rate-estimators.md` §5)

$$
\nabla\cdot(\mathbf F_d+\mathbf F_b)
\;=\; 4\pi\kappa_a (B - J_d)\;-\;\kappa_a F_0 e^{-\tau_b}
$$

  and the second term — absorption of the uncollided beam — is analytic,
  noise-free, and already used by hand in `scripts/phase5_ground.py:184`
  (`gas_beam = (kappa_a * F0 * np.exp(-tau_b)).sum() * V`).
- Size: it is the **dominant** term exactly where the diffuse field is weak —
  clear air, optically thin cells, the tenuous air just above the cloud — i.e. where a
  solar-driven heating profile matters most.  (In true vacuum $\kappa_a=0$ and the term
  vanishes identically; the dominance is over the *diffuse* term where the air is thin
  but still absorbing.)

**Finding F16 — the MC `heating` output has neither a test nor a `sigma`.**

- `tests/test_mc_*.py` contains **zero** references to `heating` (grep,
  2026-11-26). The only heating assertion in the matrix is
  `tests/test_matrix_closure.py:218`, which runs at `kappa_a = 0` — the point
  where F15's missing term vanishes identically. Nothing in the matrix can see
  the gap.
- `sigma_heating` is not reported at all, contradicting `plan/mc-plan.md` §5
  ("Every MC quantity in outputs carries its `sigma`").

**Finding F17 — sign/order hazard inherited from F8.**

`rt3d.diagnostics.heating_rate(kappa_a, J, B)` and
`rt3d.mc.estimators.heating_rate(kappa_a, B, J)` have **reversed** argument order
(F8). Any beam term must not introduce a third spelling or a third argument order.

**Finding F18 — how big is it? (to be measured in H1.)**

Expected: volume-integrated, `∫κ_a F0 e^{−τ_b} dV` is a large fraction of the total
absorbed solar power in a low-`ω` or optically thin domain, and in the slab case it
is available in closed form (`κ_a F0 μ_0^{-1}(e^{−τ_0} − e^{−τ_top})` per unit area).
H1 measures it on the matrix case.

---

## 3. Target bookkeeping

The codebase keeps two independent closures; the new output must reproduce both
identities (they are already the hand-written logic of `scripts/phase5_ground.py`,
lines 174–208):

| closure | identity | today |
|---|---|---|
| diffuse | `out_top + out_bottom + gas_diff = beam_scat` | ✓ phase5 by hand, not a function |
| beam | `in_beam = gas_beam + beam_scat + beam_at_ground` | ✓ phase5 by hand, `rt3d.transport.beam_irradiance_ground` |
| **field** | `heating_total = 4πκ_a(B − J_d) − κ_a F0 e^{−τ_b}` | ✗ missing (F15) |

with `gas_diff = −∫4πκ_a(B − J_d)dV`, `beam_scat = ∫κ_s F0 e^{−τ_b}dV`,
`gas_beam = ∫κ_a F0 e^{−τ_b}dV`.

Sign convention is kept as in `rt3d/diagnostics.py:20`: **positive = radiation
gains energy from matter** (gas cooling). The beam term is therefore negative.

---

## 4. Design

### H1 — `heating_beam` / `heating_total` / sigmas (core step)

- New helper in `rt3d/mc/estimators.py` (or `rt3d/diagnostics.py`, see D1):

  ```
  beam_heating_rate(kappa_a, F0, tau_b) -> -kappa_a * F0 * exp(-tau_b)
  ```

  single definition, one argument order, docstring states the F15 closure and the
  sign convention explicitly.
- `rt3d/mc/forward.py:_common_output` gains an optional `heating_beam=None`
  argument and emits
  - `heating` — **unchanged meaning** (diffuse field), so every existing call
    site, plot and reference range stays valid;
  - `heating_total = heating + heating_beam`;
  - `sigma_heating = 4π κ_a σ_J` (closes F16's second half);
  - `sigma_heating_total = sigma_heating` — the beam term has *identically zero*
    noise; this must be stated in the docstring because it is the whole point.
- `tau_b` is already computed inside `j_mc` (`_beam_tau` / `_beam_j_analytic`);
  **reuse it**, do not call `beam_optical_depth{_octree}` a second time.
- Vacuum guard: `κ_a = 0` ⇒ `heating_beam = 0` with no division — the guard AGENTS.md
  requires around `κ_s/κ_e` is not needed here.

### H2 — explicit variant: `J_direct` becomes a cross-check

`J_direct` (MC) is now redundant with `J_direct_analytic` for the heating field.
Keep the tally (it is the existing validation of the beam seeding and the DDA
marcher), and add a docstring line: *`J_direct` is a consistency check of
`J_direct_analytic`, it carries `O(1/√N_beam)` noise and must never be used for
heating.*  A `tally_direct=False` switch that skips the work is recorded as future
work (§9), not part of this plan.

### H3 — SC parity (see D2)

`rt3d.diagnostics.energy_balance(grid, I, dg, kappa_a, B, beam=None)`:
when `beam` is given, additionally return `beam_scat`, `gas_beam`,
`beam_at_ground` and both closure residuals of §3, so the phase5 hand-written
block becomes a tested function. `heating` keeps its diffuse meaning; add
`heating_total` alongside.

### H4 — matrix wiring

- `tests/matrix_cases.py` / `matrix_stats.py`: expose the two closures so both
  front ends (`pytest tests/test_matrix_*.py` and `scripts/solver_matrix.py`)
  share the helpers — the rule that keeps acceptance and report from drifting.
- `scripts/solver_matrix.py`: add the heating columns to the CSV and one figure
  (profile of `heating`, `heating_beam`, `heating_total` for a solar column).

---

## 5. Explicit non-goals (discussed, deferred — see §9)

Recorded so nobody re-litigates them mid-implementation:

1. **Next-event / forced-collision fluence estimator** for `J` (every history
   contributes to every voxel its next flight crosses, weighted by `e^{−τ}`; kills
   zero-inflation on fine octree leaves).
2. **Face-current divergence tally** (`±w` on cell-face crossings, net power / V_c —
   conservative by construction, no derivative, immune to the graded-octree O(h)
   problem of `_flux_divergence_octree`, `diagnostics.py:193`).
3. **Adjoint heating-rate response** (volume detector with adjoint source
   `4πκ_a χ_c`); already flagged as "not implemented" in `plan/mc-plan.md:389–391`.
4. **Frequency / k-distribution band summation** — the solver stays monochromatic.
5. Fixing **F8** (the duplicated `heating_rate` helpers) — still a follow-up,
   but it must be settled *before* H1 adds a third heating function, else the
   hazard doubles.

---

## 6. Acceptance criteria

Statistics follow the rules already recorded in the matrix plan, in particular:

- **F3**: near-vacuum / optically thick cells must stay behind the mask — a pointwise
  2σ test there is meaningless;
- **F9**: per-point z is the unit, pooled `mean z` needs the correlation allowance
  (reference range `|mean z| ≤ 0.10`), and path parity must be tested at
  *different* seeds.

| step | assertion |
|---|---|
| H1a | 1D slab, `B = 0`, no ground, `beam` on, `κ_a > 0`: `Σ(heating_total − heating)·V == −∫κ_a F0 e^{−τ_b}dV` to machine precision, and `Σ heating_total·V` equals `−(out_top + out_bottom)` within the *existing* slab closure range |
| H1b | the same slab against the closed form `κ_a F0 μ_0^{-1}(e^{−τ_0} − e^{−τ_top})` per unit area |
| H1c | `sigma_heating == 4π κ_a sigma_J` exactly (identity test, not statistical) |
| H1d | explicit vs λ: `heating_total` agrees within combined `2σ` (the beam term being common to both, the agreement must be *better* than for `heating`) |
| H2 | `J_direct` vs `J_direct_analytic` unchanged from today (regression guard only) |
| H3 | matrix case with `beam=True`, octree d2: beam closure residual `≤ 0.5 %` of `in_beam`, diffuse closure unchanged from the recorded T-range |
| H4 | vacuum / `κ_e = 0` cells: `heating_beam == 0` and finite everywhere; no NaN in `heating_total` for the `κ_e → 0` octree case |
| H5 | `delta_M ∈ {0, 1}` on the cloud with beam: beam closure residual stays within the range recorded for `delta_M = 0`; any growth is recorded as a reference range with cause, per the matrix convention |

Figures → `plot/solver_matrix/`, `matplotlib.use("Agg")`.

---

## 7. Decisions to sign off before coding

| id | question | recommendation |
|---|---|---|
| **D1** | Where does `beam_heating_rate` live? | `rt3d/diagnostics.py` next to `heating_rate` (it is a diagnostic, and the MC package should not own the sign convention). Re-export from `rt3d.mc.estimators` if the MC outputs need it locally. |
| **D2** | `energy_balance(beam=...)` in or out? | **In**, as H3 — it converts `scripts/phase5_ground.py`'s hand-written block into a tested function and is the cheapest way to *prove* H1's identity. If the answer is "out", H3/H5 move to a follow-up and the plan's proof rests on H1a/H1b alone. |
| **D3** | delta-M | Derive from the code's own conventions (below), then verify numerically (H5). Do **not** silently apply the term for `M ≥ 1` without H5. |
| **D4** | Rename `heating` → `heating_diffuse`? | **No** in this step — it breaks plots and reference ranges. Do it together with the F8 fix in a separate change. |
| **D5** | Back-compat of `sigma_*` additions | additive keys only; no existing key changes meaning. |

**Delta-M position (D3 derivation).** The swept equation is
`Ω̂·∇I + κ_e^δ I = κ_e^δ S` with `κ_e^δ = κ_e − κ_s g^M`
(`rt3d/source.py:296–304`, `rt3d/solver.py:85–104`). `beam_scattering_source` with
`delta_M ≥ 1` uses `ω/(1 − ω g^M)`, which is exactly
`κ_s Φ F0 e^{−τ_b} / κ_e^δ` in that source convention — i.e. **the physical
first-scatter source is unchanged by delta-M**; only its representation in `S`
changes. `τ_b` deliberately stays the physical optical depth. Prediction, to be
confirmed by H5: the beam-side term stays exactly `−κ_a F0 e^{−τ_b}` for all `M`,
and the only delta-M artifacts in a heating comparison are (i) the quadrature
normalisation of Φ (finding **F14**, up to 4–8.9× at `g → 0.9` between SC and MC)
and (ii) the SC discretisation of the explicit `−ω g^M I` term, which `solver.py`
records as vanishing ~linearly with cell size.

---

## 8. Decisions-log entries to add to `plan/plan.md` on implementation

1. *Heating rate in MC is the local form `4πκ_a(B − J)`, never `div F` of an MC flux
   field; `div F` remains a verification invariant only (invariant 4).*
2. *Reported MC/SC `heating` is the **diffuse** field; the total gas heating is
   `heating + heating_beam` with `heating_beam = −κ_a F0 e^{−τ_b}` analytic.*
3. *`heating_beam` has identically zero MC noise; `sigma_heating = 4πκ_a σ_J`.*

---

## 9. Future work (recorded, not scheduled)

- `tally_direct=False` switch in the explicit chain (skips uncollided-beam
  scoring once §4/H2 has established it as a cross-check).
- Items 1–3 of §5, in the order: face-current divergence (cheapest structural win
  on the graded octree) → next-event fluence (biggest noise win on fine leaves) →
  adjoint heating response (per-ROI).
- F8 cleanup bundled with `heating` → `heating_diffuse` renaming (D4).
- k-distribution / band summation if the solver ever leaves monochromatic.
