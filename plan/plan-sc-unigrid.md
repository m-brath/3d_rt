# plan/plan-sc-unigrid.md — Short characteristics on the uniform grid (DOIT-style)

Goal: a **second, independent transport algorithm** for the unigrid, modelled on the
DOIT solver of ARTS. The existing lattice-stencil chain transport
(`rt3d/characteristics.py` + `rt3d/transport.py`, `build_chain` / `sweep_direction`)
stays **untouched**; the new method is added alongside it and selected through a
solver option.

Decisions (agreed 2026-10-13):
- single-cell short characteristics with **bilinear 4-point interpolation** at the
  cell-face crossing (no quadratic option for now),
- periodic entry axes (x/y) handled by **wrap-around passes** (repeat ring sweep
  until the wrap-boundary change is below tolerance, capped at `nx` passes),
- direction grid: the same icosahedral `direction_grid` used for the scattering
  integral (no change),
- solver option `transport="chain" | "sc"`, default `"chain"` (existing behaviour
  unchanged).

## 1. Algorithm

Per ordinate Omega and target cell (i, j, k) the characteristic is only one cell
long: march backwards from the cell center along -Omega to the first cell face it
crosses, interpolate the incoming intensity at that face point from the four
neighbouring grid points, and integrate the RTE over the short segment.

### 1.1 Entry face and face point (precomputed once per ordinate)

On a uniform grid the first face hit by the backward ray from a cell center is the
same for every cell:

```
entry axis  a   = argmax_a |Omega_a| / h_a        (axes with Omega_a = 0 skipped)
entry sign      = sign(Omega_a)
face distance t_f = h_a / (2 |Omega_a|)
face point    p = c_center - t_f * Omega
upwind cell   u = (i, j, k) - sign(Omega_a) * e_a     (single immediate neighbour)
```

Bound on the lateral offset of p: since a maximises |Omega_a|/h_a,

```
|p - center|_lateral = t_f |Omega_b| <= (h_a/2) * (|Omega_b|/h_b) / (|Omega_a|/h_a) <= h_b / 2
```

so p always lies inside the 4-ring of cell centers around u in the face plane —
the bilinear stencil is always well-defined, also on anisotropic grids (hz != hx).
No ordinate has Omega_z = 0 (direction-grid rotation), so every characteristic
chain terminates in z.

### 1.2 Interpolation at the face point ("between the neighbors")

p is generally not a grid point. The incoming intensity there is the bilinear
4-point interpolation of the already-swept intensities at the four cell centers
of the upwind layer surrounding p (u plus its diagonal neighbours in the face
plane). The four offsets and weights are identical for every cell and are
precomputed per ordinate, exactly like the chain precomputation. Weights sum to 1.

### 1.3 Formal solution over the short segment

The segment p -> center lies entirely inside cell c; the source is piecewise
constant (same recursion style as the existing sweeps):

```
I_c = I_interp(p) * exp(-kappa_e,c * t_f)
      + S_c * (1 - exp(-kappa_e,c * t_f))
```

### 1.4 Sweep ordering

The whole stencil lies in the already-computed upwind layer, so the sweep is a
layer loop along the entry axis a in propagation order, vectorized over the
layer plane:

- entry axis z (non-periodic): trivially well-ordered, nz vectorized steps.
- entry axis x or y (periodic, ~2/3 of the icosahedral ordinates): the layer
  dependency forms a ring across the wrap. Handle with **wrap-around passes**:
  sweep the ring repeatedly until the change at the wrap boundary is below a
  tolerance, capped at nx (or ny) passes. The composed ring operator has spectral
  radius < 1 (rays always drift in z and leave through top/bottom), so the passes
  converge; the result is exact after nx passes (pure-shift case kappa_e = 0), and
  geometric convergence for kappa_e > 0.

### 1.5 Boundary conditions

- Boundary-layer cells along the entry axis (upwind neighbour outside the domain):
  the stencil points outside the domain take `bc.top_I / bottom_I / side_I`
  (same `rt3d.transport.boundary_conditions` class, `root_exit_axis/side`
  equivalents stored per boundary cell in the precompute).
- Ground (phase 5): for upward z-entry ordinates the bottom-layer inflow is the
  Lambertian ground radiance of the entry column, passed as `bottom_I` exactly as
  the chain sweep does. Solver ordering (downward ordinates first, rebuild I_g)
  is unchanged.

## 2. Code layout (nothing existing is removed)

New file `rt3d/short_characteristics.py`:

- `sc_stencil(omega, h)` -> dict: entry axis + sign, `t_f`, the 4 upwind-layer
  offsets and bilinear weights.
- `build_sc_chain(grid, omega, periodic_box)` -> per-ordinate precompute dict
  (same convention as `build_chain`): layer index order along the entry axis,
  boundary-layer cell masks with entry-face info, wrapped lateral indices for
  periodic x/y, wrap-pass bookkeeping.
- `sweep_direction_sc(grid, sc, kappa_e, S, bc, bottom_I=None)` -> `I (n_cells,)`:
  layer loop, 4 gathers + weights per layer, boundary/stencil replacement,
  short-segment recursion; wrap-around passes for x/y entry axes.
- `sweep_all_sc(grid, dg, kappa_e, S, bc, periodic_box, scs=None)` ->
  `I (n_cells, N_dir), scs`: same signature/return convention as `sweep_all`.

`rt3d/solver.py`:

- `lambda_iteration(..., transport="chain" | "sc")`, default `"chain"`.
- The precompute list `chains` becomes `chains`/`scs` depending on the option;
  ground ordering, Ng, delta-M, beam optical depth, scattering matmul are
  method-agnostic and reused unchanged.

## 3. Tests (`tests/test_short_characteristics.py`)

1. Stencil: entry axis = argmax |Omega_a|/h_a, exact `t_f`, weights sum to 1,
   face point inside the 4-ring on anisotropic grids.
2. Homogeneous medium + constant inflow: `I = I0 * exp(-kappa * sum(t_f))` along
   the staircase path exactly.
3. `kappa_s = 0`, `B = 0`: transmittance vs. true `exp(-tau)` — the SC path is a
   staircase, so the deviation is O(h) and shrinks under refinement (validation
   anchor 3 is exact only for true-ray schemes; document this).
4. Slab + Gaussian cloud: SC converges toward the reference as h -> 0; SC vs.
   chain agreement at fixed grid.
5. Periodic ring: wrap-pass convergence, exactness after nx passes in vacuum.
6. Solver integration with `transport="sc"`: ground, beam, Ng, delta-M cases.

## 4. Docs

- New section in `docs/sc-transport.md` (algorithm + figures optional).
- Decisions-log entry in `plan/plan.md`: single-cell SC with bilinear 4-point
  face interpolation; wrap-around passes for periodic entry axes; `transport`
  solver option.

## 5. Expected behaviour / trade-offs

- Local, gather-heavy; no BLAS matmul needed for the sweep itself.
- Single-cell characteristics + bilinear face interpolation is more diffusive
  than the chain method (classic short-characteristics behaviour); expect
  slightly larger path errors on coarse grids, converging ~linearly with cell
  size.

## 6. Amendments (2026-10-13, during implementation)

Two agreed deviations from sections 1/3, both forced by measurements made
while implementing the tests (42 ordinates, homogeneous slab, refinement
8 -> 16 -> 32):

1. **Two-segment recursion (gap segment added).** The section 1.3 recursion
   as written integrates only the face-point -> target-center segment
   (`I_c = I_interp(p) * exp(-kappa_e,c t_f) + ...`). All four bilinear
   nodes lie strictly *upwind* of p, so the interpolation is an
   extrapolation and the entry-face -> upwind-center half cell is never
   integrated: the transmittance error is O(1) and **scale-invariant**
   (tau effectively halved for z-entry ordinates; error ratio 1.000 under
   refinement - section 3 test 3 as written cannot pass). Fix: propagate
   each node value along +Omega by t_f through the node's own cell (the
   gap segment, which lands exactly in the entry-face plane) before the
   bilinear combination. With the gap segment, z-entry ordinates
   reproduce exp(-kappa * true path) to machine precision; oblique
   (x/y-entry) ordinates keep a bounded, resolution-independent angular
   stencil error (the discrete recursion's effective direction is set by
   the bilinear weights, not by Omega). Section 3 test 3 is replaced by
   that documented behaviour.
2. **Wrap-pass cap raised from n_a to max(4 n_a, 64)** (parameter
   `max_passes`). The "exact after nx passes (pure-shift case)" argument
   does not hold for the bilinear 4-point stencil: the ring-circuit
   operator is non-normal and its vacuum gain (1 - s_z)**n_a approaches 1
   for grazing ordinates (measured: rho ~ 0.30 at n_x = 24, ~50 passes to
   1e-12). With kappa_e > 0 in the ring the gain carries an extra
   exp(-2 kappa t_f n_a) and 1-3 passes suffice; the cap only binds in
   near-vacuum rings, where it truncates the multi-circuit contribution.

Section 3 test 5 "exactness after nx passes in vacuum" is replaced by
fixed-point residual + dense-linear-solve agreement (the discrete system
the passes converge to).
