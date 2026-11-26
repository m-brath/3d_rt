# Heating rates in a 3D radiation field — theory of flux-free (local) estimators

Why the divergence of the radiative flux — the quantity a heating rate *is* — can be
estimated in Monte Carlo without ever constructing a flux, what that costs and buys in
variance, and how the direct solar beam has to be bookkepted for the answer to be
correct rather than just quiet.

Companion implementation spec and findings: `plan/heating-beam-split-plan.md`.
Related documentation: `docs/sc-transport.md` (SC sweeps), `docs/solver-matrix.md`
(the cross-solver test matrix and findings F1–F14).

| code under discussion | role |
|---|---|
| `rt3d/mc/estimators.py` | `track_tally` (track-length fluence → `J`, `F`), `heating_rate` |
| `rt3d/mc/forward.py` | `j_mc` both variants, `_common_output`, `_beam_j_analytic` |
| `rt3d/mc/tracking.py`, `rt3d/mc/jitmc.py` | Woodcock tracker, analog chains, `implicit_capture` (in `tracking.py`); `jitmc.py` is the numba fast path |
| `rt3d/diagnostics.py` | `heating_rate`, `flux_divergence`, `_flux_divergence_octree`, `energy_balance` |
| `rt3d/source.py` | `beam_scattering_source` (first-scatter source, δ-M, Φ renormalisation) |
| `rt3d/transport.py` | `beam_optical_depth{,_octree}`, `beam_irradiance_ground` |
| `scripts/phase5_ground.py` | the hand-written two-closure balance this document explains |

**Marker convention.** Statements are labelled as one of
**[code]** (read off the repository, with file:line),
**[measured]** (a number produced by a run in this repository, with its source),
**[theory]** (standard radiative-transfer / Monte Carlo theory, derived here), or
**[predicted]** (a consequence the document derives that has *not* yet been measured
here). Nothing in §2–§5 needs a measurement to be true; the numbers in §4.3 and §7
are the repository's own.

---

## 1. Setting and conventions

The solver is monochromatic, scalar (no polarisation), straight-ray, LTE. At one
frequency, with $\kappa_a = \kappa_e - \kappa_s$ and $\omega = \kappa_s/\kappa_e$:

$$
\hat{\Omega}\cdot\nabla I(\mathbf r,\hat{\Omega}) + \kappa_e I
\;=\; \kappa_e S,
\qquad
S = (1-\omega)B + \omega\!\int \Phi(\hat{\Omega}'\to\hat{\Omega})\,I(\hat{\Omega}')\,d\Omega' + S_{\text{beam}}
$$

**[code]** The "swept equation" convention $\hat{\Omega}\cdot\nabla I + \kappa_e I = \kappa_e S$
is the one the MC birth rates must respect (`plan/mc-plan.md`, M1 note: omitting the
$\kappa_e$ factor rescales the $\Lambda$ operator and makes the λ iteration diverge).
It matters below in §5.4, because it decides how a source term looks in δ-M.

**Sign convention.** `rt3d/diagnostics.py:20` and `rt3d/mc/estimators.py:213` both
return $4\pi\kappa_a(B-J)$ and define *positive = radiation gains energy from the
matter* (gas cooling). The meteorological heating rate (gas warming) is its negative.
This document keeps the repository's sign and says so at each point where the sign
could flip. **[code]** (And note finding F8: the two `heating_rate` helpers have
*reversed* argument order while returning the same expression.)

Finally: "$\Phi$" is the Henyey–Greenstein phase function, normalised analytically so
that $\int\Phi\,d\Omega = 1$. Whether that integral equals 1 *on a discrete ordinate
set* is a separate and very consequential question (§7).

---

## 2. The moment identity

Take the first angular moment of the transfer equation, i.e. integrate over solid
angle. Term by term:

$$
\int \hat{\Omega}\cdot\nabla I\,d\Omega \;=\; \nabla\cdot\!\int \hat{\Omega} I\,d\Omega \;=\; \nabla\cdot\mathbf F
$$

$$
-\kappa_e\!\int I\,d\Omega = -4\pi\kappa_e J
\qquad
\kappa_a 4\pi B \text{ is angle-independent}
$$

and the scattering term, swapping the order of integration,

$$
\kappa_s\!\int\!\!\int \Phi(\hat{\Omega}'\to\hat{\Omega}) I(\hat{\Omega}')\,d\Omega'\,d\Omega
= \kappa_s\!\int I(\hat{\Omega}')\underbrace{\left[\int\Phi(\hat{\Omega}'\to\hat{\Omega})\,d\Omega\right]}_{=1}d\Omega'
= 4\pi\kappa_s J .
$$

Adding,

$$
\boxed{\;\nabla\cdot\mathbf F \;=\; 4\pi\kappa_a B - 4\pi\kappa_e J + 4\pi\kappa_s J \;=\; 4\pi\kappa_a\,(B - J)\;}
$$

**What the derivation actually uses, and what it therefore does not care about:**

| used | not used |
|---|---|
| $\int\Phi\,d\Omega = 1$ (scattering conserves photon number) | the *shape* of $\Phi$ — HG, Mie, anything normalised |
| locality of $\kappa_a,\kappa_s,B,J$ in $\mathbf r$ | the angular distribution of $I$ (anisotropy, beams, discontinuities) |
| the transfer equation itself | spatial structure, grid, $\omega$ magnitude, optical depth |
| straight rays (so that $\int\hat{\Omega}\cdot\nabla I = \nabla\cdot\mathbf F$) | boundary conditions, sources, geometry (see §8) |

So the identity is **exact at any single-scattering albedo, any asymmetry parameter,
any spatial inhomogeneity, and to any order of multiple scattering** — all of that is
buried inside $J$. **[theory]**

Two consequences that matter for the codebase:

1. The divergence is a **local product of cell quantities**. Computing it does not
   require fluxes, cell faces, finite differences, or a stencilled derivative. The
   "usual" route — difference the fluxes — is an *indirect* evaluation of a quantity
   that is available *directly*.
2. $4\pi\kappa_a(B-J)$ and $\nabla\cdot\mathbf F$ are equal for the **exact**
   solution, not for a discrete one. For a discretised solver they differ by its
   truncation and quadrature error. That is precisely what verification invariant 4
   tests (`rt3d/diagnostics.py:6-12`), and it is why the tolerances there are
   percentages and not zeros. The moment identity also tells you *which* side is the
   definition: the local form needs no extra discretisation, so it is the primary;
   $\nabla\cdot\mathbf F$ is the derived quantity and the cross-check.

---

## 3. The Monte Carlo fluence tally as a heating estimator

### 3.1 Track-length (exponential-path) fluence

**[code]** `rt3d/mc/estimators.py` accumulates, per cell $c$,

$$
\text{fluence}_c = \frac{1}{V_c}\sum_k w_k\,\ell_{c,k},
\qquad
J_c = \frac{\text{fluence}_c}{4\pi},
\qquad
\mathbf F_c = \frac{1}{V_c}\sum_k w_k\,\ell_{c,k}\,\hat{\Omega}_k ,
$$

with $\ell_{c,k}$ the path length history $k$ spends in $c$ and $w_k$ a *power*
(intensity × area), so the tally has intensity units.

**[theory] Why the expectation is right.** A particle that is alive at path length $s$
(has not been truly absorbed) has survival probability $e^{-\tau_a(s)}$, so
$\mathbb{E}[\ell_c] = \int \mathbb{1}_c\,e^{-\tau_a}\,ds$. Summed over the generations
of a scattering chain, those surviving paths reproduce the Neumann (collision) series
of the transfer equation, term by term. Equivalently, in operator form, the track
tally estimates $\int_{V_c}\!\int I\,d\Omega\,dV$ — the *angle-integrated fluence* in
the cell — and dividing by $4\pi V_c$ gives the volume-averaged mean intensity. This
is the classical track-length (path-length) fluence estimator of neutral-particle
Monte Carlo.

### 3.2 Cell averages commute with $\kappa_a$

The tally gives a *volume average*, and the local identity needs $\kappa_a(\mathbf r)$
inside the average. On a grid where $\kappa_a$ is **piecewise constant per cell** —
unigrid cells and octree leaves alike **[code]** — the commutation is exact:

$$
\big\langle 4\pi\kappa_a (B - J)\big\rangle_c \;=\; 4\pi\kappa_{a,c}\big(B_c - \langle J\rangle_c\big).
$$

So `heating_rate(kappa_a, B, J)` applied to a per-cell tally is *exactly* the
cell-averaged divergence, with no commutation or sub-grid error. **[theory + code]**
If $\kappa_a$ were ever sampled continuously within a cell (e.g. a sub-grid profile),
this step would need the $\kappa_a$-weighted track tally $\sum_k w_k\kappa_a\ell_{c,k}$
instead of $\kappa_{a,c}\sum_k w_k\ell_{c,k}$.

### 3.3 The deposition (collision) form, and the `implicit_capture` trap

An alternative estimator of the same absorption term: deposit the power a history
*loses* at each true absorption in the cell where it is absorbed. For an analog chain
that kills on absorption, the two have identical expectation — the expected weight
removed in $c$ per flight equals $\kappa_{a,c}$ times the expected track length in $c$
— but different variance (the collision form is Bernoulli-thin: it fires only at
collision points; the track form spreads over every crossed cell). **[theory]**

**[predicted, important]** The repository supports `implicit_capture`
(`rt3d/mc/tracking.py`, `plan/mc-plan.md` M4), where absorption does **not** remove the
history: at every real collision the weight is multiplied by $\omega$
(`tracking.py:989-999`), and the removed fraction $w\,(1-\omega)$ is never deposited.
A deposition-style heating tally that waits for a "true absorption" event is therefore
*wrong* (no such event exists under implicit capture). The correct rewrite is to deposit
$w_{\text{old}}(1-\omega)$ at each collision — equivalently the per-segment
$\kappa_a w\,ds$ — not to wait for an absorption event. The $\kappa_a\cdot$track-length
form used here is immune, because $\kappa_a$ is a per-cell property and $J$ is a
per-cell tally. Any future "optimisation" into a deposition tally must re-derive this.

### 3.4 What the local form buys, structurally

- The emission term $4\pi\kappa_a B$ is **analytic and carries zero noise**. The only
  stochastic term is the absorption.
- Cells with $\kappa_a = 0$ (vacuum, pure scattering) get heating **exactly zero**,
  not a noise floor around zero. A differenced flux field cannot do this: it produces
  a non-zero noise field in exactly those cells.
- No angular quadrature (MC stage 1 has none — `plan/mc-plan.md` line 72), so no
  closure error from the ordinate set (§7).
- No face areas, no extrapolation to faces, no stencil — hence no graded-mesh
  accuracy question on the octree (§8).

---

## 4. Three ways to get $\nabla\cdot\mathbf F$, with variance analysis

### 4.1 (A) Local absorption — the recommended estimator

$$
\hat H^{(A)}_c = 4\pi\kappa_{a,c}\big(B_c - J^{\text{MC}}_c\big),
\qquad
\sigma_{H} = 4\pi\kappa_{a,c}\,\sigma_J .
$$

Noise enters only through $J$, and is prefactored by $\kappa_a$.

### 4.2 (B) Face-current divergence — conservative by construction

Instead of a track tally per cell, tally the **net power crossing each cell face**:
deposit $\pm w$ when a history crosses the interface between two cells. Then

$$
\hat H^{(B)}_c = \frac{1}{V_c}\sum_{\text{faces of }c}\!\!\pm\,(\text{net power through face}).
$$

This is the divergence theorem applied per cell — a finite-volume Monte Carlo.

Properties **[theory]**:
- Summing $\hat H^{(B)}_c$ over all cells **telescopes**: every interior face appears
  twice with opposite sign, so the volume sum equals the boundary power *identically*
  — global closure holds by construction, not by convergence.
- No derivative of a noisy field, and no extrapolation to the domain faces, which is
  where the SC/octree closure error currently lives (`rt3d/diagnostics.py:58-92`,
  `_octree_face_flux`; M5 measured the closure residual to be *only* this
  face-extrapolation error).
- But: emission and absorption both enter through the same noisy crossing statistics,
  so (B) has **no analytic term** — in this respect it is strictly worse than (A).
  Its win is conservation and grid-independence, not noise.

### 4.3 (C) Flux tally + finite differences — the estimator to avoid

`rt3d.diagnostics.flux_divergence` applied to an MC `F` field. With cell flux estimates
$\mathbf F_c = \mathbf F_c^{\text{true}} + \varepsilon_c$, and $\varepsilon$ treated as
independent between cells, a central difference over $2h$ gives

$$
\sigma(\hat\nabla\!\cdot\!\mathbf F) \;\approx\; \frac{\sqrt{2}\,\sigma_F}{2h},
\qquad
\frac{\sigma(\hat\nabla\!\cdot\!\mathbf F)}{|\nabla\cdot\mathbf F|}
\;\approx\; \frac{1}{\sqrt{2}}\,\frac{L}{h}\,\frac{\sigma_F}{|F|}
$$

where $L$ is the gradient scale of the field. Differencing amplifies the flux noise by
$L/h$ **[theory]**. Two penalties compound in the optically thick, scattering-dominated
interior, where $\mathbf F$ is itself the small residual of a nearly isotropic fluence:
there $|F| \ll$ fluence, so $\sigma_F/|F|$ is already large *before* the $L/h$.

**[measured]** The repository already sees this shape: `plan/mc-plan.md` lines 74–78 —
pointwise $\nabla\cdot\mathbf F$ versus heating is truncation-dominated at the slab
interfaces, **MC 12–18 %, SC ≈8 % of peak heating**, while the *global* closure is
1.1–1.3 % (MC) vs 4.7 % (SC-42). The pointwise number is the derivative; the global
one, which needs no derivative, is an order of magnitude better.

**[theory]** The independence assumption behind the $\sqrt2$ is optimistic: one history
contributes to many cells, so neighbouring-cell noise is correlated on the scale of a
mean free path. That correlation lets part of it cancel in a difference (helping (C)
somewhat), but it also breaks the usual $\sqrt n$ averaging when you smooth a field.
Measured in this repository as finding **F9**: the spread of pooled per-cell `mean z`
over independent seed pairs is 0.023 where $\text{std}/\sqrt n$ claims 0.010 — a
factor 2.2.

### 4.4 Comparison

| | (A) local absorption | (B) face currents | (C) flux + FD |
|---|---|---|---|
| needs fluxes | no | no | yes |
| derivative of a noisy field | no | no | **yes** ($\times L/h$) |
| analytic (noise-free) term | **yes** ($4\pi\kappa_a B$) | no | no |
| $\kappa_a=0$ cells | exactly 0 | noisy | noisy |
| global closure | by convergence | **exact by construction** | by convergence |
| per-cell conservation | no | **yes** | no |
| angular quadrature error (MC) | none | none | none |
| needs face areas / extrapolation | no | yes (interior faces only) | yes, incl. domain faces |
| accuracy on graded octree | grid-limited only | grid-limited only | **O(h)** (`diagnostics.py:160-167`) |
| requires $\kappa_a$ field | yes | no | no |

Last two rows of the left column are the honest counter-arguments to (A): it needs
$\kappa_a$ (so it cannot validate an *unknown* absorption field) and it is only as
spatially resolved as the grid, being a volume average per cell.

### 4.5 Illustrative arithmetic

Not a run — an order-of-magnitude reading of the formulae above, to size the effect.
A cloud with gradient scale $L = 500$ m resolved at $h = 10$ m gives $L/h = 50$. If the
flux tally reaches 10 % relative error (achievable), the differenced divergence is at
$\approx 0.7\times 50\times 10\% \approx 350\%$ relative error — and in the diffusion
interior $\sigma_F/|F|$ is typically worse than 10 % for the same history count. Route
(A) at the same count is at the level of $\sigma_J/J$ scaled by the cancellation ratio
(§6). **[theory]**

---

## 5. The direct solar beam: why the local form alone is not enough

This is the part of the question that turned out to be a correctness issue rather than
a variance issue. See `plan/heating-beam-split-plan.md` finding F15.

### 5.1 The split the solvers actually use

**[code]** Both families transport the **diffuse** field only: `I = I_d + I_b`, with the
beam handled analytically (`rt3d/solver.py` feeds the diffuse sweeps through
`rt3d.source.beam_scattering_source`; `rt3d/mc/forward.py`'s λ variant never tallies
uncollided beam tracks, and its explicit variant routes them to a separate
`tally_direct` that reaches `J_total` but not `heating`).

The two transport equations:

$$
\hat{\Omega}\cdot\nabla I_b = -\kappa_e I_b,
\qquad
\hat{\Omega}\cdot\nabla I_d = -\kappa_e I_d + \kappa_a B + \kappa_s\!\int\!\Phi I_d \;+\; \kappa_s\,\Phi(\hat{\Omega}_{\text{sun}}\to\hat{\Omega})\,F_0 e^{-\tau_b}
$$

with $\tau_b$ the beam optical depth from the entry face along $\hat{\Omega}_{\text{sun}}$
(`rt3d/transport.beam_optical_depth{,_octree}`).

### 5.2 The identities

Take moments. For the diffuse field, using $\int\Phi(\hat{\Omega}_{\text{sun}}\to\hat{\Omega})d\Omega = 1$ on the last term:

$$
\nabla\cdot\mathbf F_d \;=\; 4\pi\kappa_a (B - J_d) \;+\; \kappa_s F_0 e^{-\tau_b}
\tag{5.1}
$$

For the beam, with $\hat{\Omega}_{\text{sun}}\cdot\nabla\tau_b = \kappa_e$ and
$\mathbf F_b = F_0 e^{-\tau_b}\hat{\Omega}_{\text{sun}}$:

$$
\nabla\cdot\mathbf F_b \;=\; -\kappa_e F_0 e^{-\tau_b}
\tag{5.2}
$$

Adding (and using $J_b = F_0e^{-\tau_b}/4\pi$):

$$
\boxed{\;\nabla\cdot(\mathbf F_d+\mathbf F_b) \;=\; 4\pi\kappa_a (B - J_d) \;-\; \kappa_a F_0 e^{-\tau_b} \;=\; 4\pi\kappa_a (B - J_d - J_b)\;}
\tag{5.3}
$$

**[theory]** Equation (5.3) is the whole point of the split: the diffuse-field local
form $4\pi\kappa_a(B - J_d)$ is missing exactly the absorption of the uncollided beam,
$-\kappa_a F_0e^{-\tau_b}$. That term is **analytic, exact, and noise-free** — every
ingredient is available at machine precision (`kappa_a`, and `F0*exp(-tau_b)` which the
solver already computes for `beam_scattering_source`).

Equation (5.1) also explains what the scattering does to the bookkeeping: it returns
$\kappa_sF_0e^{-\tau_b}$ as a *source* to the diffuse field, which the MC transports as
first-scatter children (`_beam_first_scatter`). So scattering never changes the beam
term — it only moves the rest of the energy into the part you pay noise for (§5.6).

### 5.3 The two closures — matching what the code already does by hand

**[code]** `scripts/phase5_ground.py:174-208` already implements (5.1) and (5.2) as two
independent power balances:

| closure | identity | code |
|---|---|---|
| diffuse | `out_top + out_bottom + gas_diff = beam_scat` | `res_dif`, `gas_diff = -bal["heating"]` |
| beam | `in_beam = gas_beam + beam_scat + beam_at_ground` | `res_bim`, `gas_beam = (kappa_a*F0*exp(-tau_b)).sum()*V` |
| field | `heating_total = 4πκ_a(B − J_d) − κ_a F0 e^{−τ_b}` | **missing** — F15 |

with $\text{gas\_diff} = \int -4\pi\kappa_a(B-J_d)\,dV$, $\text{beam\_scat} = \int \kappa_s F_0e^{-\tau_b}dV$,
$\text{gas\_beam} = \int \kappa_a F_0 e^{-\tau_b}dV$. Both scalar closures are right; the
**field** form is missing — which is exactly the kind of gap that survives when a balance is
written once in a script and never in a library. `rt3d.transport.beam_irradiance_ground`
already supplies `beam_at_ground`.

Magnitude **[predicted, to be measured as H1 of the plan]**: in clear air, optically
thin cells, and the tenuous air just above a cloud, the diffuse term is small and $-\kappa_a
F_0e^{-\tau_b}$ is the dominant contribution — i.e. the missing term is largest exactly
where a solar-driven heating profile is usually reported. (In true vacuum $\kappa_a=0$
and both terms vanish identically; the beam term dominates where the air is optically
thin but still absorbing.) In a slab it has the closed
form per unit area $\kappa_a F_0\mu_0^{-1}(e^{-\tau_0}-e^{-\tau_{\text{top}}})$.

### 5.4 δ-M: why the term is (predicted to) survive unchanged

**[code]** With `delta_M = M ≥ 1` the diffuse transport uses
$\kappa_e^\delta = \kappa_e - \kappa_s g^M$ (`rt3d/source.delta_extinction`; only the
scattering part is scaled, so $\kappa_s = 0$ is untouched), and the beam first-scatter
albedo factor becomes $\omega/(1-\omega g^M)$ (`rt3d/source.py:296-304`). `rt3d/solver.py:93-96`
states that $\tau_b$ deliberately stays the **physical** beam optical depth.

**[theory]** Because the swept equation is $\hat{\Omega}\cdot\nabla I + \kappa_e^\delta I = \kappa_e^\delta S$,
a physical source term $q$ must appear as $S = q/\kappa_e^\delta$. And indeed

$$
\frac{\omega}{1-\omega g^M}\,F_0e^{-\tau_b}\Phi
\;=\;\frac{\kappa_e\,\omega}{\kappa_e - \kappa_s g^M}\,F_0e^{-\tau_b}\Phi
\;=\;\frac{\kappa_s\,F_0e^{-\tau_b}\Phi}{\kappa_e^\delta},
$$

so **the physical first-scatter source is invariant under δ-M**; only its representation
in $S$ changes. Since (5.1)–(5.3) were derived from physical source terms, **[predicted]**
the beam term $-\kappa_aF_0e^{-\tau_b}$ remains correct for $M\ge 1$, with the remaining
δ-M artifacts in any *heating comparison* coming from (i) the quadrature renormalisation
of $\Phi$ (§5.5, finding F14) and (ii) the short-characteristics discretisation of the
explicit $-\omega g^M I$ term, which `rt3d/solver.py:97-100` records as vanishing
~linearly with cell size. The plan's H5 test is the numeric confirmation; the
derivation is explicitly *not* treated as a substitute for it.

### 5.5 The angular-normalisation caveat (finding F14)

**[measured, F14]** `rt3d.source.beam_scattering_source` rescales the per-cell beam
phase function so that $\sum_i w_i\Phi_i = 1$ on the ordinate set, deliberately: the
beam first-scatter power then passes through the ordinates exactly. The MC solvers
sample the **analytic** HG. On the 42-ordinate development grid the raw quadrature
defect $s(g)=\sum_iw_i\Phi_{\text{analytic}}$ is ≈1.000 up to $g=0.5$, 1.06–1.11 at
$g=0.7$, and **4.0–8.9 at $g=0.9$–0.925** — so inside a dense cloud core the two
families' beam sources differ by a factor ≈4–9.

Consequence for this document: the **beam term** of (5.3) is unaffected (it involves no
$\Phi$), but the **diffuse field** $J_d$ that the two families produce differs through
F14. Every beam-case, cross-family, heating comparison at $g\to0.9$ on 42 ordinates is
**F14-limited, not noise-limited**. Do not read such a difference as a heating-estimator
error.

### 5.6 Forward scattering, and why δ-M is also a variance lever

With large $g$, forward scattering keeps energy near the beam direction, so
$e^{-\tau_b}$-attenuated direct radiation — and hence the analytic term
$-\kappa_aF_0e^{-\tau_b}$ — remains a large fraction of the absorption *deep* inside
the cloud. Simultaneously, the exact split counts a strongly forward-scattered photon
as "diffuse" immediately, so the noisy $J_d$ carries a lot of physically near-beam
radiation. δ-M approximates that forward peak as unscattered in the *diffuse*
transport — it reduces the diffuse effective extinction to $\kappa_e-\kappa_s g^M$ and
rescales the diffuse source — which is why it is both the standard cure for high-$g$
transport (§5.4) and, in this context, a variance-reduction lever for heating rates —
provided the beam bookkeeping above is right. It does **not** add the forward peak to
the analytic beam term: $\tau_b$ stays physical and the beam source keeps the full
normalised HG (§5.4).

---

## 6. Noise budget: what the split buys and what it cannot

Define the **cancellation ratio**

$$
C \;=\; \frac{J}{|B - J|},
$$

which is $O(1)$ when the radiation field is far from the local source function and very
large near radiative equilibrium (the typical state of a real atmosphere, and of any
weakly absorbing slab illuminated from outside). Then, with $N_c$ effectively
independent contributions (crossings) in the cell:

$$
\frac{\sigma_H}{|H|}\;\approx\;\frac{C}{\sqrt{N_c}}
\qquad\text{(local form)}
$$

**[theory]** The local form's absolute noise is $\sigma_H = 4\pi\kappa_a\sigma_J$: as
$\omega\to1$ ($\kappa_a\to0$) both the signal and the noise vanish together, and the
emission term never contributes. This is the *best* behaviour any estimator can have.

For the flux-difference route, combine §4.3's $\frac{1}{\sqrt2}\frac{L}{h}\frac{\sigma_F}{|F|}$
with the fact that $|\nabla\cdot\mathbf F| = 4\pi\kappa_a|B-J|$ and that in the diffusion
regime $|F|/\text{fluence}\sim 1/(\kappa_{tr}L)$: the relative error becomes

$$
\frac{\sigma(\hat\nabla\cdot\mathbf F)}{|\nabla\cdot\mathbf F|}
\;\sim\; \frac{1}{\sqrt2}\,\frac{L}{h}\,\kappa_{tr}L\,\frac{1}{\sqrt{N_c}}\,
$$

i.e. it pays for the diffusion cancellation *and* the differencing, on top of $C$.
**[theory]** The practical statement is a history-count ratio: in the regime
$|F|\sim$ fluence, differencing needs roughly $(L/h)^2$ times more histories than the
local form; in the diffusion regime the full expression above adds a further
$(\kappa_{tr}L)^2$ factor. (The local form's $C^2$ requirement is intrinsic to the
problem and is shared — no estimator escapes near-equilibrium cancellation.)

**Zero-inflation in optically thin or finely resolved cells.** The track tally reports
$J_c$ only where histories actually pass, so fine octree leaves in tenuous media can be
sampled by few or no histories: the field is zero-inflated and the batch $\sigma$
collapses to zero in cells nothing entered (for the track tally this is immediate: a
cell with no track length has zero in every batch; the related next-flight/binary-tally
pathology is measured in `plan/mc-plan.md` M2 note and finding F3). The analytic beam
term is immune — it exists
independently of sampling. The general cure is a next-event (forced-collision) fluence
estimator, which is deliberately out of scope of the current plan and recorded as future
work.

**Correlated noise.** Because one history deposits in many cells, the noise field of
`heating` is spatially correlated over a mean free path: smoothing the field over $n$
cells does *not* reduce noise by $\sqrt n$ (see the 2.2× factor in F9). This is also why
the $\sqrt 2$ in §4.3 understates the true differencing penalty in some regions and
overstates it in others.

**Feedback noise.** In a radiative-equilibrium loop, $H\to T\to B\to H$. MC noise in $H$
is then feedback noise, and a noise-free component of $H$ improves stability, not just
plot quality. This is the historical reason the energy-deposition ("factor") schemes of
coupled Monte Carlo radiation-hydrodynamics reformulated absorption as a local,
implicit-friendly quantity rather than a flux difference.

---

## 7. Scattering strength: exact vs quadrature-limited

**[theory + code]** The moment identity needs $\int\Phi d\Omega=1$ and nothing else about
$\Phi$. So:

- **MC has no angular quadrature at stage 1** (`plan/mc-plan.md` line 72) and samples
  the HG direction from its exact inverse CDF. Scattering is exactly photon-conserving
  event by event. (M5 uses a closed-form single-scattering invariant that "locks the
  adjoint walk as exact-HG".) For MC, $4\pi\kappa_a(B-J) = \nabla\cdot\mathbf F$ holds
  as a statement about the *continuous* transport problem; the only errors are
  statistical plus spatial (grid, DDA, path sampling).
- **SC normalises on the ordinate set.** `hg_matrix(normalize=True)` row-normalises the
  weighted HG matrix, so the *incoming* integral that the source evaluation needs,
  $\sum_j w_j\Phi_{ij}=1$, holds exactly; `beam_scattering_source` similarly normalises
  the beam phase vector so $\sum_i w_i\Phi_i=1$. The *outgoing* integral that the moment
  identity needs, $\sum_i w_i\Phi_{ij}=1$, is a different sum: because the row sums
  $s_i=\sum_j w_j\Phi_{ij}$ vary across ordinates (measured here: $s_i=4.01$–$5.05$ at
  $g=0.9$ on 42 ordinates, a 22 % spread), the renormalised matrix is not symmetric and
  the outgoing integral is closed only up to that spread. AGENTS.md's invariant 2
  normalises HG analytically; the raw *discrete* defect $s(g)=\sum_iw_i\Phi_i\ne1$ is
  measured in F14 (≈1.000 at $g\le0.5$; 1.06–1.11 at 0.7; 4.0–8.9 at 0.9–0.925 on 42
  ordinates). Tighter grids drive $s(g)\to1$ (development level `n_refine=1` → 42
  ordinates, then 162, 642 — the phase-6 convergence study).

Therefore, **for a strongly forward-scattering atmosphere, the MC local heating field is
the better-defined object**: it samples the analytic HG and has no angular discretisation
error at all, while the SC field carries one that grows steeply with $g$ — the SC
renormalisation keeps the source evaluation conservative but changes the phase function,
and the direction-dependence of that renormalisation is what breaks the outgoing integral.
Symmetrically, the SC field has no *statistical*
error, so the sensible production/diagnostic pairing is: MC for the heating field, SC for
the noise-free structure and the cross-check — with F14 as the reading rule whenever a
beam and $g\gtrsim0.7$ are present.

---

## 8. Geometry, grids, and boundaries

**[theory]** The identity $\nabla\cdot\mathbf F = 4\pi\kappa_a(B-J)$ is a coordinate-free
local statement. It does not care about the coordinate system, so **on a spherical shell
it is no harder than in a Cartesian box**, whereas $\nabla\cdot\mathbf F$ evaluated by
finite differences would need a spherical stencil. Same for the periodic $x$/$y$ faces
(`rt3d/geometry.periodic_box`): the local form never touches them; the flux route has to
wrap them (`diagnostics.py:183-184`) and the octree route probes neighbours across them.

**[code]** On the octree the differences are concrete:

- Local form: needs $\kappa_{a,c}$ and $V_c=(2h_c)^3$ per leaf. Grid-accuracy-limited
  and nothing else.
- `flux_divergence` on the octree (`diagnostics.py:151-228`): finite differences between
  located neighbour centers; on a 2:1 graded mesh the difference quotient is centred
  between the two neighbor centers, so the truncation error is **O(h)** rather than
  O(h²) — its own docstring says so, and `energy_balance` therefore bypasses it and
  integrates face powers instead.
- Face currents (§4.2) would remove both the graded-stencil issue and the
  face-extrapolation term from the closure — recorded as future work.
- The Lambertian ground (`rt3d/mc/ground.py`, `rt3d/transport.ground_surface`) enters
  only through $J$ and through the beam term at the surface; the identity is unaffected.

---

## 9. From a monochromatic column to an atmospheric profile

**[theory]** A real heating-rate profile is a frequency (or band) sum. With a
correlated-k / k-distribution treatment, band $g$ with weight $w_g$,

$$
H(\mathbf r) \;=\; \sum_g w_g\,4\pi\kappa_{a,g}\big(B_g(T) - J_g\big),
\qquad
H_{\text{beam}}(\mathbf r) = -\sum_g w_g\kappa_{a,g}F_{0,g}e^{-\tau_{b,g}},
$$

and everything above applies per band unchanged: the band sum is linear, so the
*emission* part is still exactly known and the *absorption* part adds noise in
quadrature across bands (a band-count noise cost, not a conceptual one — provided the
bands are sampled independently; with common random numbers the band errors are
correlated and need not add in quadrature). The solver here
is monochromatic by design (AGENTS.md), so this is a statement about how the estimator
would generalise, not a claim about current capability.

Limits of the physics that bound what such a profile may be used for:

| assumption | where it bites |
|---|---|
| LTE, $S=(1-\omega)B+\omega J$ | fine below the mesosphere; breaks where level populations decouple |
| no polarisation | irrelevant for heating (scalar moment), relevant for radiances |
| straight rays, no refraction | matters for long tangential paths — and note that with refraction $\int\hat\Omega\cdot\nabla I\,d\Omega = \nabla\cdot\mathbf F$ itself needs a phase-space formulation, so the identity is only exact in a non-refracting medium |
| scalar, single frequency | see the band sum above |

---

## 10. The adjoint viewpoint: heating rate as a response functional

**[theory]** A heating rate in a region of interest $h(\mathbf r)$ is a *linear response*
of the field to the sources:

$$
R \;=\; \int h(\mathbf r)\,4\pi\kappa_a(\mathbf r)\,J(\mathbf r)\,dV ,
$$

whose adjoint problem has an adjoint source $q^*\propto h\,\kappa_a$ distributed over the
ROI. Evaluating the response as $\langle$adjoint fluence, forward sources$\rangle$ scores
exactly the same three objects the existing adjoint walk already scores — thermal
$\kappa_a B$, beam first scatter $\kappa_s\Phi F_0e^{-\tau_b}$, and boundary inflow — but
seeded from a **volume** (the ROI) instead of a detector cone. That is the most literal
"simulate the divergence directly" available: it concentrates the sampling where the
answer is needed. `plan/mc-plan.md:389-391` flags precisely this extension as not
implemented.

Two honest caveats: (i) it is per-ROI (one run per response, or per basis function), so
it is a query tool, not a field generator; (ii) the *uncollided beam* is not part of the
diffuse adjoint phase space, which is a second, independent reason to keep the beam
contribution analytic (as the plan does) and let the adjoint carry only the diffuse
response.

---

## 11. Rules for this codebase

1. **The production heating rate is the local form** $4\pi\kappa_a(B-J)$ from the fluence
   tally — never $\nabla\cdot\mathbf F$ of an MC flux field.
2. **`heating` means the diffuse field.** The physically complete gas heating is
   `heating + heating_beam` with `heating_beam = -κ_a F0 exp(-tau_b)`, analytic and with
   `sigma = 0`.
3. **`sigma_heating = 4π κ_a sigma_J`**, and it must be reported (F16, and
   `plan/mc-plan.md` §5's rule that every MC quantity carries its sigma).
4. **`div F` is a diagnostic and a test** (invariant 4), not an estimator. Its tolerance
   is the discretisation statement, and on the graded octree it is O(h) by construction.
5. **Read beam-case cross-family differences through F14** (analytic vs
   quadrature-normalised HG) before reading them as estimator error.
6. **Never convert the heating estimator into a weight-deposition tally** without
   re-deriving it under `implicit_capture` (§3.3).

### Coverage map

| quantity | production | test that covers it today |
|---|---|---|
| $J$ (MC) | `track_tally` | T0–T5 matrix, `test_mc_forward.py`, statistics per F3/F9 |
| heating (diffuse) | `heating_rate` | **none** for MC (F16); `test_matrix_closure.py:218` only at $\kappa_a=0$ |
| heating (total, with beam) | missing (F15) | none — this is what `plan/heating-beam-split-plan.md` H1–H5 adds |
| $\nabla\cdot\mathbf F$ | diagnostic only | `test_ground.py:156`, `test_delta_m.py:249`, `test_matrix_closure.py:571` |
| global closure | `energy_balance` | matrix T-ranges; phase5 hand-written beam closure |

---

## References

Cited as context; every derivation above is self-contained and none of the claims
depend on these sources.

- S. Chandrasekhar, *An Introduction to the Study of Stellar Structure* / *Radiative
  Transfer* — moments of the transfer equation, the $K$–$J$ structure.
- J. B. Carlson and T. D. Godson, "Monte Carlo solution of transfer problems",
  *J. Nuclear Energy* 12 (1961) — the first radiation-transport Monte Carlo; track and
  collision tallies.
- J. Spanier and E. M. Gelbard, *Monte Carlo Principles and Neutron Transport Problems*
  (1969) — fluence/track-length estimators, next-event estimation, variance reduction.
- J. D. Fleck Jr., "An improved Monte Carlo sampling scheme for coupled radiation
  transport and thermal hydrodynamics", *Nucl. Sci. Eng.* 57 (1976) — local
  energy-deposition ("factor") formulation instead of flux differencing.
- R. Siegel and J. R. Howell, *Thermal Radiation Heat Transfer* — collision /
  next-event estimators for absorption in participating media.
- W. J. Wiscombe, *J. Appl. Meteor.* 16 (1977) and Joseph, Wiscombe & Warner,
  *Q. J. R. Met. Soc.* 102 (1976) — the δ-scaling of strongly forward-peaked phase
  functions (δ-M, δ-Eddington) implemented in `rt3d/source.py`.
- In-repo: `plan/mc-plan.md` (MC family, tallies, error accounting), `plan/plan.md`
  (decisions log), `plan/solver-test-matrix-plan.md` (findings F1–F14, in particular
  F8 sign hazard, F3/F9 statistics rules, F14 HG quadrature defect),
  `plan/heating-beam-split-plan.md` (this document's implementation spec).
