# -*- coding: utf-8 -*-
"""
T0 of the solver-matrix plan (plan/solver-test-matrix-plan.md section 3):
physics invariants re-run **across every solver variant**, so a new backend,
transport or sampler cannot silently drop one of the AGENTS.md invariants.

Invariants covered here:

1. direction-grid measure: ``sum(w) == 4 pi``, unit ordinates, no ordinate on
   the equator, and the measure of the two Monte Carlo direction samplers
   (isotropic ``1/4pi`` and cosine ``|mu|/pi``) plus the HG inverse-CDF;
2. HG phase function: analytic normalization, quadrature row sums and first
   moment ``g``, and the sampled Legendre moments ``E[P_n] = g**n``;
3. Beer-Lambert ``kappa_s = 0, B = 0``: transmittance ``exp(-tau)`` on the
   same rays for the lattice-stencil chain, the single-cell short
   characteristics, the exact octree march (NumPy and fused numba), the
   Woodcock tracker (both majorant modes), and the adjoint walk;
4. the analytic direct-beam optical depth on the unigrid and the octree.
"""

import numpy as np
import pytest
from scipy.special import eval_legendre

from matrix_cases import direction_grid
from matrix_stats import assert_banded

from rt3d.characteristics import build_chain
from rt3d.directions import hg
from rt3d.geometry import cuboid, unigrid
from rt3d.jit import HAVE_NUMBA, chain_sweep_jit, sc_sweep_jit
from rt3d.mc.backward import backward_detector_mc
from rt3d.mc.sources import (
    face_inflow_directions,
    hg_cos_theta,
    isotropic_directions,
)
from rt3d.mc.tracking import EV_EXIT, flight_budget, sample_collisions
from rt3d.octree import octree_grid
from rt3d.short_characteristics import (
    build_sc_chain,
    sweep_all_sc,
    sweep_direction_sc,
)
from rt3d.transport import (
    beam_optical_depth,
    beam_optical_depth_octree,
    boundary_conditions,
    precompute_octree_march,
    sweep_all,
    sweep_all_octree_exact,
    sweep_direction,
    sweep_direction_octree_exact,
)

# =============================================================================
# 1. direction-grid measure
# =============================================================================


@pytest.mark.parametrize("n_refine", [0, 1, 2, 3])
def test_solid_angle_measure(n_refine):
    """Invariant 1: the icosahedral weights sum to 4 pi at every level."""
    dg = direction_grid(n_refine)
    assert dg.n_dir == 10 * 4**n_refine + 2
    assert abs(dg.weights.sum() - 4.0 * np.pi) < 1e-12
    assert np.all(dg.weights > 0.0)
    # unit ordinates, no duplicate, none on the equator (the octree exact
    # march and the chain lattice both need omega_z != 0)
    assert np.allclose(np.linalg.norm(dg.directions, axis=1), 1.0, atol=1e-12)
    assert np.abs(dg.directions[:, 2]).min() > 1e-3
    assert np.unique(np.round(dg.directions, 12), axis=0).shape[0] == dg.n_dir


def test_isotropic_sampler_measure():
    """The MC isotropic sampler realises the 1/(4 pi) measure."""
    rng = np.random.default_rng(7)
    d = isotropic_directions(2_000_000, rng)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    # E[Omega_i**2] = 1/3, E[Omega_i]**~0
    for ax in range(3):
        assert abs(d[:, ax].mean()) < 2e-3
        assert abs(np.mean(d[:, ax] ** 2) - 1.0 / 3.0) < 1e-3


@pytest.mark.parametrize("side", [-1, 1])
def test_cosine_face_measure(side):
    """The face-source sampler realises the crossing-rate measure |mu| dOmega."""
    rng = np.random.default_rng(11)
    d = face_inflow_directions(2_000_000, 2, side, rng)
    mu = np.abs(d[:, 2])                    # inward cosine (>= 0)
    # cosine law: E[mu**n] = 2/(n+2)
    for n, expected in ((0, 1.0), (1, 2.0 / 3.0), (2, 0.5), (3, 0.4)):
        assert abs(np.mean(mu**n) - expected) < 2e-3
    assert np.all(np.sign(d[:, 2]) == -side)   # inward hemisphere
    assert abs(d[:, 0].mean()) < 2e-3 and abs(d[:, 1].mean()) < 2e-3


@pytest.mark.parametrize("g", [-0.5, 0.0, 0.3, 0.7, 0.9])
def test_hg_sampler_moments(g):
    """The HG inverse-CDF reproduces the Legendre moments g**n (invariant 2)."""
    rng = np.random.default_rng(3)
    mu = hg_cos_theta(rng.random(400_000), g)
    assert mu.min() >= -1.0 and mu.max() <= 1.0
    for n in (1, 2, 3):
        est = float(np.mean(eval_legendre(n, mu)))
        # var(P_n) <= 1, so the standard error is <= 1.6e-3
        assert abs(est - g**n) < 8e-3


# =============================================================================
# 2. phase-function normalization
# =============================================================================


@pytest.mark.parametrize("g", [0.0, -0.3, 0.3, 0.7, 0.9, 0.99])
@pytest.mark.parametrize("n_refine", [1, 2])
def test_hg_quadrature_normalization(g, n_refine):
    """Row-normalized quadrature: row sums exactly 1, first moment near g.

    The row sums are exact by construction (dg.hg_matrix(normalize=True));
    the *moment* is a quadrature error of the discrete ordinate set and is
    fixed to the measured values (plan section 2): the 42-ordinate grid
    lags badly at strong forward peaking (g = 0.9 -> moment 0.981), the
    162-ordinate grid recovers it (0.940).
    """
    dg = direction_grid(n_refine)
    Phi = dg.hg_matrix(g)                    # weights folded in, normalized
    assert np.allclose(Phi.sum(axis=1), 1.0, atol=1e-12)
    cos = dg.directions @ dg.directions.T
    first = float(np.mean(np.einsum("ij,ij->i", Phi, cos)))
    # physical bounds: the discrete moment cannot exceed 1 and must not fall
    # far below the continuous g
    assert first <= 1.0 + 1e-12
    assert first >= g - 0.1
    # fixed values: measured 2026-11-16 on this solver (plan section 2, T0 pass)
    pins = {(1, 0.0): 0.0, (1, -0.3): -0.3, (1, 0.3): 0.3, (1, 0.7): 0.730,
            (1, 0.9): 0.981, (1, 0.99): 1.0,
            (2, 0.0): 0.0, (2, -0.3): -0.3, (2, 0.3): 0.3, (2, 0.7): 0.701,
            (2, 0.9): 0.940, (2, 0.99): 1.0}
    assert_banded(first, pins[(n_refine, g)] - 0.01,
                  pins[(n_refine, g)] + 0.01,
                  f"HG first moment n_refine={n_refine}, g={g}")


@pytest.mark.parametrize("g,tol", [(-0.5, 1e-8), (0.0, 1e-12), (0.5, 1e-8),
                                   (0.9, 1e-7), (0.99, 5e-5)])
def test_hg_analytic_normalization(g, tol):
    """The continuous HG phase function integrates to 1 (analytic, invariant 2)."""
    mu = np.linspace(-1.0, 1.0, 2_000_001)
    integral = 2.0 * np.pi * np.trapezoid(hg(mu, g), mu)
    assert abs(integral - 1.0) <= tol


# =============================================================================
# 3. Beer-Lambert transmittance across every transport
# =============================================================================

KAPPA = 0.7          # constant extinction of the transmittance box


@pytest.fixture(scope="module")
def box_and_grids():
    """Constant-kappa 3D box (periodic x/y) on the unigrid and the octree."""
    box = cuboid(0.0, 4.0, 0.0, 2.0, 0.0, 3.0)
    x = (np.arange(8) + 0.5) * 0.5
    y = (np.arange(4) + 0.5) * 0.5
    z = (np.arange(6) + 0.5) * 0.5
    grid = unigrid(x, y, z)

    oct_ = octree_grid(box, h0=0.5)
    oct_.finalize()
    oct_.kappa_e = np.full(oct_.n_leaf, KAPPA)
    return box, grid, oct_


def _analytic_transmittance(centers, dg, kappa, z1):
    """I = I_top exp(-kappa L) for downward ordinates, 0 for upward."""
    vz = dg.directions[:, 2]
    path = np.where(
        vz < 0.0,
        (z1 - centers[:, 2, None]) / np.abs(vz)[None, :],
        0.0,
    )
    return np.where(vz < 0.0, np.exp(-kappa * path), 0.0)


def _entry_axis(dg, grid):
    """Lattice/SC entry axis: a = argmax |Omega_a| / h_a, per ordinate."""
    h = np.array([grid.hx, grid.hy, grid.hz], dtype=float)
    return np.argmax(np.abs(dg.directions) / h[None, :], axis=1)


def test_transmittance_chain_numpy(box_and_grids):
    """
    Lattice-stencil chain: bounded staircase path error, never amplification.

    Measured (plan section 2): max |I - exp(-tau)| = 0.371 and mean 0.047 on
    this 0.5 m box at kappa = 0.7.  The error is the *stencil path length*
    error of oblique ordinates (the chain walks cell-to-cell along the
    lattice, not along the characteristic) and shrinks with h (T5 ladder).
    """
    box, grid, _oct = box_and_grids
    dg = direction_grid(1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    kappa = np.full(grid.n_cells, KAPPA)
    I, _chains = sweep_all(grid, dg, kappa, np.zeros(grid.n_cells), bc,
                           periodic_box=box)
    ref = _analytic_transmittance(grid.centers, dg, KAPPA, box.z1)
    dev = np.abs(I - ref)
    assert_banded(dev.max(), 0.0, 0.40, "chain transmittance max abs deviation")
    assert_banded(dev.mean(), 0.0, 0.07, "chain transmittance mean abs deviation")
    assert np.all(I <= 1.0 + 1e-12)          # never amplified above I_in


def test_transmittance_sc_numpy(box_and_grids):
    """
    Single-cell short characteristics, two regimes (documented in
    rt3d/short_characteristics.py):

    * **z-entry** ordinates (``argmax |Omega_a|/h_a == 2``) reproduce
      ``exp(-tau)`` to machine precision -- the gap segment makes them exact;
    * **x/y-entry** (oblique) ordinates carry a bounded angular stencil
      error (measured max 0.265 absolute, mean 0.10 on this box), the same
      class of error as the chain's grazing-ordinate path error.
    """
    box, grid, _oct = box_and_grids
    dg = direction_grid(1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    kappa = np.full(grid.n_cells, KAPPA)
    ref = _analytic_transmittance(grid.centers, dg, KAPPA, box.z1)
    I = np.zeros_like(ref)
    for d in range(dg.n_dir):
        sc = build_sc_chain(grid, dg.directions[d], periodic_box=box)
        I[:, d] = sweep_direction_sc(grid, sc, kappa, np.zeros(grid.n_cells), bc)
    assert np.array_equal(I, sweep_all_sc(grid, dg, kappa,
                                          np.zeros(grid.n_cells), bc,
                                          periodic_box=box)[0])

    ax = _entry_axis(dg, grid)
    z_entry = ax == 2
    assert np.count_nonzero(z_entry) > 0
    assert np.allclose(I[:, z_entry], ref[:, z_entry], atol=1e-12)
    dev = np.abs(I[:, ~z_entry] - ref[:, ~z_entry])
    assert_banded(dev.max(), 0.0, 0.30, "SC oblique transmittance max dev")
    assert_banded(dev.mean(), 0.0, 0.12, "SC oblique transmittance mean dev")
    assert np.all(I <= 1.0 + 1e-12)


def test_transmittance_vertical_all_transports(box_and_grids):
    """
    Exactness invariant on the vertical direction (not a grid member: the
    icosahedral grid is rotated off the poles, which is why the transmittance
    invariants above run over the grid's own ordinates).

    Every transport must reproduce exp(-kappa (z1 - z)) to machine precision
    for a vertical characteristic.
    """
    box, grid, oct_ = box_and_grids
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    d = np.array([0.0, 0.0, -1.0])
    ref_u = np.exp(-KAPPA * (box.z1 - grid.centers[:, 2]))
    ref_o = np.exp(-KAPPA * (box.z1 - oct_.centers[:, 2]))
    zeros_u = np.zeros(grid.n_cells)
    zeros_o = np.zeros(oct_.n_leaf)

    chain = build_chain(grid, d, periodic_box=box)
    I_chain = sweep_direction(grid, chain, np.full(grid.n_cells, KAPPA),
                              zeros_u, bc)
    sc = build_sc_chain(grid, d, periodic_box=box)
    I_sc = sweep_direction_sc(grid, sc, np.full(grid.n_cells, KAPPA),
                              zeros_u, bc)
    march = precompute_octree_march(oct_, d, periodic_box=box, jit=False)
    I_oct = sweep_direction_octree_exact(oct_, march,
                                         np.full(oct_.n_leaf, KAPPA),
                                         zeros_o, bc)
    assert np.allclose(I_chain, ref_u, atol=1e-12)
    assert np.allclose(I_sc, ref_u, atol=1e-12)
    assert np.allclose(I_oct, ref_o, atol=1e-12)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not available")
def test_transmittance_jit_sweeps_match_numpy(box_and_grids):
    """Fused numba sweeps reproduce the NumPy transmittance bit-for-bit."""
    box, grid, _oct = box_and_grids
    dg = direction_grid(1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    S0 = np.zeros(grid.n_cells)
    kappa = np.full(grid.n_cells, KAPPA)

    I_ref, chains = sweep_all(grid, dg, kappa, S0, bc, periodic_box=box)
    for d in range(dg.n_dir):
        I_jit = chain_sweep_jit(grid, chains[d], kappa, S0, bc)
        assert np.array_equal(I_jit, I_ref[:, d])

    for d in range(dg.n_dir):
        sc = build_sc_chain(grid, dg.directions[d], periodic_box=box)
        I_np = sweep_direction_sc(grid, sc, kappa, S0, bc)
        I_jit = sc_sweep_jit(grid, sc, kappa, S0, bc)
        assert np.array_equal(I_jit, I_np)


def test_transmittance_octree_numpy(box_and_grids):
    """Exact characteristic march: transmittance to floating-point accuracy."""
    box, _grid, oct_ = box_and_grids
    dg = direction_grid(1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    S0 = np.zeros(oct_.n_leaf)
    I, _marches = sweep_all_octree_exact(oct_, dg, oct_.kappa_e, S0, bc,
                                         periodic_box=box)
    ref = _analytic_transmittance(oct_.centers, dg, KAPPA, box.z1)
    assert np.allclose(I, ref, atol=1e-12)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not available")
def test_transmittance_octree_fused_matches_reference(box_and_grids):
    """Fused numba march+sweep equals the NumPy reference march result."""
    box, _grid, oct_ = box_and_grids
    dg = direction_grid(1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    S0 = np.zeros(oct_.n_leaf)
    I_ref, _ = sweep_all_octree_exact(oct_, dg, oct_.kappa_e, S0, bc,
                                      periodic_box=box)
    marches = [
        precompute_octree_march(oct_, dg.directions[d],
                                periodic_box=box, jit=None)
        for d in range(dg.n_dir)
    ]
    assert all(m.get("fused", False) for m in marches)
    I_fused, _ = sweep_all_octree_exact(oct_, dg, oct_.kappa_e, S0, bc,
                                        periodic_box=box, marches=marches)
    assert np.allclose(I_fused, I_ref, atol=1e-12)


# =============================================================================
# 4. Monte Carlo tracker and adjoint walk
# =============================================================================


@pytest.mark.parametrize("mode", ["cell", "global"])
@pytest.mark.parametrize("jit", [False, True])
def test_mc_walker_transmittance(box_and_grids, mode, jit):
    """Woodcock tracker: P(no real collision) = exp(-tau) in both modes."""
    if jit and not HAVE_NUMBA:
        pytest.skip("numba not available")
    box, grid, _oct = box_and_grids
    rng = np.random.default_rng(5)
    n = 200_000
    starts = np.column_stack([
        rng.uniform(0.1, 3.9, n),
        rng.uniform(0.1, 1.9, n),
        rng.uniform(0.1, 2.9, n),
    ])
    d = np.tile(np.array([0.3, 0.2, -0.9]) / np.linalg.norm([0.3, 0.2, -0.9]),
                (n, 1))
    kappa = np.full(grid.n_cells, KAPPA)
    budget = flight_budget(starts, d, grid, periodic_box=box)
    res = sample_collisions(starts, d, budget, grid, kappa, np.zeros_like(kappa),
                            mode=mode, periodic_box=box, jit=jit)
    exited = res["coll_event"] == EV_EXIT
    p_est = exited.mean()
    # constant kappa: the per-flight survival probability is exp(-tau) with
    # tau of that flight's budget
    p_ref = float(np.mean(np.exp(-KAPPA * budget)))
    sigma = np.sqrt(p_ref * (1.0 - p_ref) / n)
    z = (p_est - p_ref) / sigma
    assert abs(z) < 4.0, f"transmittance z = {z:+.2f} ({mode}, jit={jit})"


def test_adjoint_pure_absorber_transmittance(box_and_grids):
    """
    Adjoint walk with omega = 0 scores I_in exp(-tau) at the detector.

    The measured direction is **downward** (light from the top inflow), so
    the adjoint flight direction is -omega0 = +z and the walk travels from
    the detector up to the top face, where the exit-inflow score is applied.
    """
    box, _grid, oct_ = box_and_grids
    dg = direction_grid(1)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    pt = (2.0, 1.0, 0.25)                    # inside the medium, near the bottom
    kappa = np.full(oct_.n_leaf, KAPPA)
    zeros = np.zeros(oct_.n_leaf)
    r = backward_detector_mc(oct_, kappa, zeros, zeros, bc,
                             {"point": pt, "omega0": np.array([0.0, 0.0, -1.0]),
                              "cone": 0.0, "n_seeds": 200_000},
                             periodic_box=box, n_batches=50, seed=17, jit=True)
    ref = np.exp(-KAPPA * (box.z1 - pt[2]))
    assert r["sigma_I"] > 0.0
    z = (r["I"] - ref) / r["sigma_I"]
    assert abs(z) < 4.0, f"adjoint transmittance z = {z:+.2f} (I={r['I']:.5f}, " \
                         f"ref={ref:.5f})"
    # the batching must resolve the Bernoulli variance of the tally
    assert_banded(r["sigma_I"], 0.2 * np.sqrt(ref * (1 - ref) / 200_000),
                  4.0 * np.sqrt(ref * (1 - ref) / 200_000),
                  "adjoint transmittance sigma")


# =============================================================================
# 5. direct-beam optical depth
# =============================================================================


@pytest.mark.parametrize("sun_deg,periodic", [(0.0, True), (0.0, False),
                                               (30.0, True)])
def test_beam_optical_depth_matches_analytic(box_and_grids, sun_deg, periodic):
    """
    Beam tau equals kappa (z1 - z)/|mu_sun| exactly on both grids.

    A vertical beam is analytic for periodic and non-periodic sides; the
    30 deg case is only analytic with periodic sides (a non-periodic beam
    from a cell near the x edge leaves through the side face, which the
    analytic vertical column ignores -- that is not a solver property).
    """
    box, grid, oct_ = box_and_grids
    if not periodic:
        box = cuboid(0.0, 4.0, 0.0, 2.0, 0.0, 3.0,
                     periodic_x=False, periodic_y=False)
    mu = np.cos(np.radians(sun_deg))
    omega_sun = np.array([np.sqrt(1.0 - mu * mu), 0.0, -mu])

    tau_u = beam_optical_depth(grid, omega_sun,
                               np.full(grid.n_cells, KAPPA),
                               periodic_box=box if periodic else None)
    tau_o = beam_optical_depth_octree(
        oct_, omega_sun, np.full(oct_.n_leaf, KAPPA),
        periodic_box=box if periodic else None,
    )
    ref_u = KAPPA * (box.z1 - grid.centers[:, 2]) / mu
    ref_o = KAPPA * (box.z1 - oct_.centers[:, 2]) / mu
    assert np.allclose(tau_u, ref_u, atol=1e-12)
    assert np.allclose(tau_o, ref_o, atol=1e-12)
