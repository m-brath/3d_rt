# -*- coding: utf-8 -*-
"""Delta-M (Wiscombe) scaling of the HG phase function (plan/delta-m-plan.md).

Operator level: the split Phi_w == Phi_w_delta + sum(moments) is exact on the
row-normalized discrete HG matrix, so the transport RHS is unchanged
((1 - omega g**M) S_delta + omega g**M I == S_undeltaed) and only the
iteration / transport scaling changes.  Only the *scattering* part of the
extinction is scaled: kappa_e_delta = kappa_e - kappa_s g**M, so pure
absorption (kappa_s = 0) is exactly unchanged and the explicit term's
iteration gain stays < 1 for all omega < 1.

Solver level: delta_M = 1/2 must reproduce the delta_M = 0 converged solution
up to the short-characteristics quadrature error of the explicit -g**M I term
(percent-level on these coarse slabs, shrinking with grid refinement),
converge in far fewer iterations at large g, and keep validation anchors 3
(Beer-Lambert with the *physical* kappa_e) and 4 (div F = heating rate).
"""

import numpy as np
import pytest

from rt3d.characteristics import build_chain
from rt3d.diagnostics import flux_divergence, heating_rate
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, ground_surface, unigrid
from rt3d.solver import lambda_iteration
from rt3d.source import delta_extinction, scattering_source
from rt3d.transport import boundary_conditions


# module-level development grid (42 ordinates)
DG = icosahedric_grid(n_refine=1)


def _slab(nz, hz, hxy=None):
    """Uniform kappa_e = 1 slab: 4x4 xy cells, nz z-layers of thickness hz,
    periodic x/y box (pattern of tests/test_accelerator.py)."""
    if hxy is None:
        hxy = hz
    x = (np.arange(4) + 0.5) * hxy
    y = (np.arange(4) + 0.5) * hxy
    z = (np.arange(nz) + 0.5) * hz
    grid = unigrid(x, y, z)
    box = cuboid(0.0, 4 * hxy, 0.0, 4 * hxy, grid.box[2][0], grid.box[2][1])
    kappa_e = np.full(grid.n_cells, 1.0)
    return grid, box, kappa_e


# ------------------------------------------------------------- operator level


@pytest.mark.parametrize("g", [0.5, 0.9, 0.95])
@pytest.mark.parametrize("M", [1, 2])
def test_operator_split_reconstructs_hg_matrix(g, M):
    """hg_matrix(g) == Phi_w_delta + sum(moments) exactly (the split is
    defined on the normalized matrix, plan/delta-m-plan.md section 2)."""
    Phi_w = DG.hg_matrix(g)
    Phi_w_delta, moments = DG.hg_delta_matrices(g, M)
    assert len(moments) == M
    assert np.allclose(Phi_w, Phi_w_delta + sum(moments), atol=1e-12)


@pytest.mark.parametrize("g", [0.5, 0.9, 0.95])
@pytest.mark.parametrize("M", [1, 2])
def test_operator_scattering_integral_split(g, M):
    """The scattering integral splits exactly for any intensity array:
    I @ Phi_w.T == I @ Phi_w_delta.T + sum_n I @ moments[n].T."""
    rng = np.random.default_rng(17)
    I = rng.uniform(0.1, 2.0, (8, DG.n_dir))
    Phi_w = DG.hg_matrix(g)
    Phi_w_delta, moments = DG.hg_delta_matrices(g, M)
    scat = I @ Phi_w_delta.T
    for Pn_w in moments:
        scat += I @ Pn_w.T
    assert np.allclose(I @ Phi_w.T, scat, atol=1e-12)


@pytest.mark.parametrize("g", [0.5, 0.9, 0.95])
def test_operator_moment_row_sums(g):
    """P_hat_0,w rows sum to 1 (isotropic mode removed exactly by M = 1);
    P_hat_1,w rows sum ~ 0 (the icosahedral first moment vanishes)."""
    _, moments = DG.hg_delta_matrices(g, 2)
    assert np.allclose(moments[0].sum(axis=1), 1.0, atol=1e-14)
    assert np.allclose(moments[1].sum(axis=1), 0.0, atol=1e-12)


def test_operator_delta_extinction():
    """kappa_e_delta = kappa_e - kappa_s * g**M; M = 0 leaves kappa_e
    unchanged; vacuum cells stay 0; bad g / M raise."""
    kappa_e = np.array([0.0, 1.0, 2.5])
    kappa_s = 0.5 * kappa_e
    assert np.array_equal(delta_extinction(kappa_e, kappa_s, 0.9, 0), kappa_e)
    assert np.allclose(delta_extinction(kappa_e, kappa_s, 0.9, 1),
                       kappa_e - kappa_s * 0.9)
    assert np.allclose(delta_extinction(kappa_e, kappa_s, 0.9, 2),
                       kappa_e - kappa_s * 0.81)
    # per-cell g
    g_cells = np.array([0.0, 0.5, 0.9])
    assert np.allclose(delta_extinction(kappa_e, kappa_s, g_cells, 1),
                       kappa_e - kappa_s * g_cells)
    # pure absorption: extinction exactly unchanged (anchor 3 with delta on)
    assert np.allclose(delta_extinction(kappa_e, np.zeros_like(kappa_e), 0.95, 2),
                       kappa_e)
    # vacuum guard: kappa_e = 0 stays 0 for any g
    assert delta_extinction(np.array([0.0]), np.array([0.0]), 0.95, 3)[0] == 0.0
    with pytest.raises(ValueError):
        delta_extinction(kappa_e, kappa_s, 1.0, 1)
    with pytest.raises(ValueError):
        delta_extinction(kappa_e, kappa_s, -1.0, 1)
    with pytest.raises(ValueError):
        delta_extinction(kappa_e, kappa_s, 0.9, -1)


@pytest.mark.parametrize("bad", [(1.0, 1), (-1.0, 1), (0.0, 0), (0.0, 1.5)])
def test_operator_hg_delta_matrices_validation(bad):
    """hg_delta_matrices requires a positive integer M and -1 < g < 1."""
    g, M = bad
    with pytest.raises(ValueError):
        DG.hg_delta_matrices(g, M)


# ------------------------------------------------------- transport-RHS exactness


@pytest.mark.parametrize("M", [1, 2])
def test_rhs_source_exactness_scalar_g(M):
    """Core correctness anchor: the delta-M transport RHS equals the
    undeltaed one elementwise,
    (1 - omega g**M) S_delta + omega g**M I == S_undeltaed."""
    rng = np.random.default_rng(23)
    n_cells = 8
    I = rng.uniform(0.1, 2.0, (n_cells, DG.n_dir))
    omega = rng.uniform(0.0, 1.0, n_cells)
    B = rng.uniform(0.0, 3.0, n_cells)
    g = 0.9
    S0 = scattering_source(I, DG, omega, B, g=g, delta_M=0)
    Sd = scattering_source(I, DG, omega, B, g=g, delta_M=M)
    gM = g**M
    om = omega[:, None]
    assert np.allclose((1.0 - om * gM) * Sd + om * gM * I, S0, atol=1e-12)


def test_rhs_source_exactness_g_cells():
    """Same exactness for the spatially varying g path (per-cell scaling
    1/(1 - omega g**M) with per-cell omega * g**M in the explicit term)."""
    rng = np.random.default_rng(29)
    n_cells = 8
    I = rng.uniform(0.1, 2.0, (n_cells, DG.n_dir))
    omega = rng.uniform(0.0, 1.0, n_cells)
    B = rng.uniform(0.0, 3.0, n_cells)
    g_cells = np.array([0.3, 0.3, 0.7, 0.7, 0.9, 0.9, 0.9, 0.5])
    S0 = scattering_source(I, DG, omega, B, g_cells=g_cells, delta_M=0)
    Sd = scattering_source(I, DG, omega, B, g_cells=g_cells, delta_M=1)
    gM = g_cells[:, None]
    om = omega[:, None]
    assert np.allclose((1.0 - om * gM) * Sd + om * gM * I, S0, atol=1e-12)


# --------------------------------------------------------------- solver level


# tau = 5 slab, nz = 16 layers (4x4 xy): the delta-on/off fixed points differ
# only by the SC quadrature error of the explicit -g**M I term (~1e-2 relative
# here, halving under refinement); a wrong kappa_e_delta or a missing -g**M I
# term changes J by orders of magnitude.
EQ_NZ = 16
EQ_TOL = 2.5e-2


@pytest.mark.parametrize("g,omega", [(0.8, 0.9), (0.9, 0.9), (0.9, 0.5)])
def test_solver_equivalence_delta_M1(g, omega):
    """delta_M = 1 vs delta_M = 0 at the same grid/directions: same converged
    J (within the explicit-term quadrature error), and the result records the
    scaled extinction kappa_e - kappa_s * g.  The omega = 0.5 case is stable
    only because the scaling touches the scattering part of the extinction
    alone (full-extinction scaling diverges for omega < g)."""
    grid, box, kappa_e = _slab(EQ_NZ, 5.0 / EQ_NZ)
    kappa_s = np.full(grid.n_cells, omega)
    B = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=1.0)

    res0 = lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc, g=g,
                            periodic_box=box, n_iter=2000, tol=1e-9,
                            ng=None, delta_M=0)
    res1 = lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc, g=g,
                            periodic_box=box, n_iter=2000, tol=1e-9,
                            ng=None, delta_M=1)
    assert res0["converged"]
    assert res1["converged"]

    J0, J1 = res0["J"], res1["J"]
    assert np.max(np.abs(J1 - J0)) / np.max(np.abs(J0)) < EQ_TOL

    assert res0["delta_M"] == 0
    assert np.array_equal(res0["kappa_e_delta"], kappa_e)
    assert res1["delta_M"] == 1
    assert np.allclose(res1["kappa_e_delta"], kappa_e - kappa_s * g)


def test_solver_acceleration_delta_M1():
    """g = 0.95 (unresolved forward lobe): delta_M = 1 needs far fewer
    lambda iterations than plain LI; both converge."""
    grid, box, kappa_e = _slab(8, 5.0 / 8)
    kappa_s = np.full(grid.n_cells, 0.9)
    B = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=1.0)

    res0 = lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc, g=0.95,
                            periodic_box=box, n_iter=2000, tol=1e-6,
                            ng=None, delta_M=0)
    res1 = lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc, g=0.95,
                            periodic_box=box, n_iter=2000, tol=1e-6,
                            ng=None, delta_M=1)
    assert res0["converged"]
    assert res1["converged"]
    assert res1["n_iter"] < 0.5 * res0["n_iter"]


def test_anchor3_transmittance_delta_on():
    """Validation anchor 3 with delta on: kappa_s = 0, B = 0, top_I = 1,
    g = 0.9, delta_M = 1.  Because only the scattering part of the
    extinction is scaled (kappa_e_delta = kappa_e - kappa_s * g = kappa_e)
    and the explicit -omega * g * I term vanishes for omega = 0, the delta-on
    run is identical to the delta-off one: downward-ordinate transmittance at
    cell centers equals exp(-kappa_e * geometric path) with the PHYSICAL
    kappa_e to the same accuracy as the plain sweep."""
    nz = 16
    grid, box, kappa_e = _slab(nz, 0.5 / nz)
    kappa_s = np.zeros(grid.n_cells)
    B = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=1.0)

    res = lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc, g=0.9,
                           periodic_box=box, n_iter=500, tol=1e-12,
                           ng=None, delta_M=1)
    assert res["converged"]
    assert np.array_equal(res["kappa_e_delta"], kappa_e)

    z_top = grid.box[2][1]
    for d in range(DG.n_dir):
        chain = build_chain(grid, DG.directions[d], periodic_box=box)
        vz = chain["vhat"][2]
        if vz >= 0.0:
            continue
        path = (z_top - grid.centers[:, 2]) / abs(vz)
        assert np.allclose(res["I"][:, d], np.exp(-path), atol=1e-2)


def test_anchor4_div_F_heating_delta_on():
    """Validation anchor 4 with delta on: absorbing + scattering slab
    (kappa_e = 1, omega = 0.9, g = 0.9) with smooth nonzero B and a
    Lambertian ground; div F = 4 pi kappa_a (B - J) pointwise in the
    interior, same finite-difference tolerance as tests/test_ground.py."""
    grid, box, kappa_e = _slab(8, 0.25, hxy=0.5)  # z in [0, 2], tau = 2
    kappa_s = np.full(grid.n_cells, 0.9)
    B = 1.0 + 0.5 * grid.centers[:, 2] / 2.0
    bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=0.3))

    res = lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc, g=0.9,
                           periodic_box=box, n_iter=2000, tol=1e-9,
                           ng=None, delta_M=1)
    assert res["converged"]

    kappa_a = 1.0 - res["omega"]
    divF = flux_divergence(grid, res["F"])
    heat = heating_rate(kappa_a, res["J"], B)
    scale = np.max(np.abs(heat))
    # interior only: one-sided differences at top/bottom carry O(h) error
    k = np.arange(grid.n_cells) // (grid.ny * grid.nx)
    interior = (k > 0) & (k < grid.nz - 1)
    assert np.max(np.abs(divF[interior] - heat[interior])) / scale < 0.05


def test_solver_equivalence_delta_M2():
    """delta_M = 2 (delta-P1 split) vs delta_M = 0: same converged J, and
    kappa_e_delta = kappa_e - kappa_s * g**2."""
    grid, box, kappa_e = _slab(EQ_NZ, 5.0 / EQ_NZ)
    kappa_s = np.full(grid.n_cells, 0.9)
    B = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=1.0)

    res0 = lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc, g=0.9,
                            periodic_box=box, n_iter=2000, tol=1e-9,
                            ng=None, delta_M=0)
    res2 = lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc, g=0.9,
                            periodic_box=box, n_iter=2000, tol=1e-9,
                            ng=None, delta_M=2)
    assert res0["converged"]
    assert res2["converged"]

    J0, J2 = res0["J"], res2["J"]
    assert np.max(np.abs(J2 - J0)) / np.max(np.abs(J0)) < EQ_TOL

    assert res2["delta_M"] == 2
    assert np.allclose(res2["kappa_e_delta"], kappa_e - kappa_s * 0.9**2)
