# -*- coding: utf-8 -*-
"""
Single-cell short characteristics on the unigrid (plan/plan-sc-unigrid.md):
stencil correctness, exact homogeneous-medium recursion, transmittance
convergence under refinement, periodic wrap-pass convergence, and solver
integration with transport="sc".
"""

import numpy as np
import pytest

from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid
from rt3d.short_characteristics import (
    build_sc_chain,
    sc_stencil,
    sweep_all_sc,
    sweep_direction_sc,
)
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions, sweep_all


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def naive_sc_sweep(grid, omega, kappa_e, S, bc, bottom_I=None):
    """
    Independent scalar reference: recursive bilinear short characteristics
    with gap-segment propagation on the unigrid (non-periodic only),
    memoized.  Deliberately naive - no layer machinery, no precompute -
    so it cross-checks the vectorized build_sc_chain / sweep_direction_sc
    indexing and boundary handling.
    """
    h = np.array([grid.hx, grid.hy, grid.hz])
    st = sc_stencil(omega, h)
    a, sigma, t_f = st["a"], st["sigma"], st["t_f"]
    offsets, weights = st["offsets"], st["weights"]
    dims = (grid.nx, grid.ny, grid.nz)
    E = np.exp(-kappa_e * t_f)
    src = S * (1.0 - E)

    memo = {}

    def I_of(idx):
        i, j, k = idx
        # out of domain: boundary face value (z faces take precedence,
        # same convention as build_sc_chain); the inflow sits at the
        # domain face itself, no gap segment
        if k < 0:
            return bc.bottom_I if bottom_I is None else bottom_I[grid.ravel(i, j, 0)]
        if k >= grid.nz:
            return bc.top_I
        if i < 0 or i >= grid.nx or j < 0 or j >= grid.ny:
            return bc.side_I
        if idx in memo:
            return memo[idx]
        c = grid.ravel(i, j, k)
        acc = 0.0
        for m in range(4):
            ni = i + offsets[m, 0]
            nj = j + offsets[m, 1]
            nk = k + offsets[m, 2]
            v = I_of((ni, nj, nk))
            if (0 <= ni < grid.nx) and (0 <= nj < grid.ny) and (0 <= nk < grid.nz):
                # gap segment through the node cell
                nc = grid.ravel(ni, nj, nk)
                v = v * E[nc] + src[nc]
            acc += weights[m] * v
        val = acc * E[c] + src[c]
        memo[idx] = val
        return val

    out = []
    for c in range(grid.n_cells):
        ii, jj, kk = grid.unravel(np.array([c]))
        out.append(I_of((int(ii[0]), int(jj[0]), int(kk[0]))))
    return np.array(out)


# ---------------------------------------------------------------------------
# 1. stencil
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [(1.0, 1.0, 1.0), (1.0, 0.7, 0.3), (0.4, 1.0, 2.0)])
def test_sc_stencil_entry_axis_and_weights(h):
    h = np.asarray(h)
    dg = icosahedric_grid(n_refine=1)
    for omega in dg.directions:
        st = sc_stencil(omega, h)
        a, sigma, t_f = st["a"], st["sigma"], st["t_f"]

        # entry axis = argmax |omega_a| / h_a, exact face distance
        ratio = np.abs(omega) / h
        assert a == int(np.argmax(ratio))
        assert sigma == np.sign(omega[a])
        assert np.isclose(t_f, h[a] / (2.0 * abs(omega[a])))

        # weights: non-negative, sum to 1
        w = st["weights"]
        assert np.all(w >= 0.0)
        assert np.isclose(w.sum(), 1.0)

        # offsets: upwind layer along a, +-1 ring laterally
        off = st["offsets"]
        assert np.all(off[:, a] == -sigma)
        lateral = [ax for ax in range(3) if ax != a]
        for ax in lateral:
            assert np.all(np.abs(off[:, ax]) <= 1)

        # face point inside the 4-ring of the upwind layer:
        # |t_f omega_b| <= h_b / 2 (the bound from the plan)
        for ax in lateral:
            assert abs(t_f * omega[ax]) <= h[ax] / 2.0 + 1e-12

        # the four nodes are the bilinear corners of p in the face plane:
        # p - center - h * (lateral offsets of the first node) in [0, 1]^2
        p_lat = np.array([-t_f * omega[ax] for ax in lateral])
        n0 = np.array([off[0, ax] for ax in lateral])
        frac = (p_lat - n0 * h[lateral]) / h[lateral]
        assert np.all(frac >= -1e-12) and np.all(frac <= 1.0 + 1e-12)


def test_sc_stencil_matches_direct_bilinear():
    """Weights are the exact bilinear interpolation weights of the face
    point in the upwind layer."""
    h = np.array([1.0, 0.7, 0.3])
    omega = np.array([0.8, -0.5, -0.3464])
    omega = omega / np.linalg.norm(omega)
    st = sc_stencil(omega, h)
    a, t_f, off, w = st["a"], st["t_f"], st["offsets"], st["weights"]
    lateral = [ax for ax in range(3) if ax != a]
    p = np.array([-t_f * omega[ax] for ax in lateral])  # rel. to cell center
    # direct bilinear: product of 1D linear weights at p
    lo = [np.floor(p[m] / h[lateral[m]]) for m in range(2)]
    fr = [p[m] / h[lateral[m]] - lo[m] for m in range(2)]
    ref_w = np.array(
        [
            (1 - fr[0]) * (1 - fr[1]),
            (1 - fr[0]) * fr[1],
            fr[0] * (1 - fr[1]),
            fr[0] * fr[1],
        ]
    )
    ref_off = np.array(
        [
            [int(lo[0]), int(lo[1])],
            [int(lo[0]), int(lo[1]) + 1],
            [int(lo[0]) + 1, int(lo[1])],
            [int(lo[0]) + 1, int(lo[1]) + 1],
        ]
    )
    order = np.argsort(np.lexsort((ref_off[:, 1], ref_off[:, 0])))
    got = np.lexsort((off[:, lateral[1]], off[:, lateral[0]]))
    assert np.array_equal(off[got][:, lateral], ref_off[order][:, lateral])
    assert np.allclose(w[got], ref_w[order])


def test_build_sc_chain_structure(box3d):
    """order is a permutation of all cells; node_face == 0 exactly for the
    nodes inside the domain, and node_idx then equals the raw offset."""
    grid = box3d
    dg = icosahedric_grid(n_refine=1)
    for omega in dg.directions[:8]:
        sc = build_sc_chain(grid, omega, periodic_box=None)
        assert np.array_equal(np.sort(sc["order"]), np.arange(grid.n_cells))
        total = sum(e - s for s, e in sc["slices"])
        assert total == grid.n_cells
        assert sc["wrap"] is False

        coords = np.column_stack(grid.unravel(sc["order"]))
        st = sc_stencil(omega, [grid.hx, grid.hy, grid.hz])
        for m in range(4):
            raw = coords + st["offsets"][m]
            inside = np.all((raw >= 0) & (raw < (grid.nx, grid.ny, grid.nz)), axis=1)
            assert np.array_equal(sc["node_face"][m] == 0, inside)
            assert np.array_equal(
                sc["node_idx"][m][inside], grid.ravel(*raw[inside].T)
            )


@pytest.fixture
def box3d():
    x = np.linspace(0.5, 5.5, 6)
    y = np.linspace(0.5, 3.5, 4)
    z = np.linspace(0.5, 7.5, 8)
    return unigrid(x, y, z)


# ---------------------------------------------------------------------------
# 2. homogeneous medium + constant inflow: exact staircase recursion
# ---------------------------------------------------------------------------


def test_homogeneous_constant_inflow_exact():
    """Entry-axis-z ordinate, homogeneous kappa, S = 0, constant inflow:
    with the gap segment the staircase path (node gap t_f + target
    segment t_f per hop) equals the true ray path, so the SC sweep
    reproduces I0 exp(-kappa * path) to machine precision."""
    hz = 0.25
    z = np.arange(0.125, 10.0, hz)
    grid = unigrid(np.array([0.0]), np.array([0.0]), z)
    pbox = cuboid(*grid.box[0], *grid.box[1], *grid.box[2])
    theta = np.deg2rad(30.0)
    omega = np.array([np.sin(theta), 0.0, -np.cos(theta)])
    h = np.array([grid.hx, grid.hy, grid.hz])
    assert sc_stencil(omega, h)["a"] == 2  # entry axis z

    kappa = 0.7
    kappa_e = np.full(grid.n_cells, kappa)
    S = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=3.0)
    sc = build_sc_chain(grid, omega, periodic_box=pbox)
    I = sweep_direction_sc(grid, sc, kappa_e, S, bc)

    z_top = grid.box[2][1]
    path = (z_top - grid.centers[:, 2]) / abs(omega[2])
    assert np.allclose(I, 3.0 * np.exp(-kappa * path), atol=1e-13)

    # upward ordinate, bottom inflow: same identity from the bottom face
    omega_up = np.array([-np.sin(theta), 0.0, np.cos(theta)])
    sc_up = build_sc_chain(grid, omega_up, periodic_box=pbox)
    bc_up = boundary_conditions(bottom_I=2.0)
    I_up = sweep_direction_sc(grid, sc_up, kappa_e, S, bc_up)
    z_bot = grid.box[2][0]
    path_up = (grid.centers[:, 2] - z_bot) / omega_up[2]
    assert np.allclose(I_up, 2.0 * np.exp(-kappa * path_up), atol=1e-13)


def test_sc_matches_naive_recursion_nonperiodic(box3d):
    """Full 3D non-periodic box, heterogeneous kappa_e and S, all boundary
    faces: the vectorized layer sweep equals the naive scalar recursion
    for every ordinate."""
    grid = box3d
    rng = np.random.default_rng(11)
    kappa_e = rng.uniform(0.0, 2.0, grid.n_cells)
    S = rng.uniform(0.0, 1.0, grid.n_cells)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.4, side_I=0.2)
    dg = icosahedric_grid(n_refine=1)
    for omega in dg.directions:
        sc = build_sc_chain(grid, omega, periodic_box=None)
        I = sweep_direction_sc(grid, sc, kappa_e, S, bc)
        I_ref = naive_sc_sweep(grid, omega, kappa_e, S, bc)
        assert np.allclose(I, I_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. kappa_s = 0, B = 0: transmittance vs. true exp(-tau), O(h)
# ---------------------------------------------------------------------------


def test_transmittance_staircase_error():
    """kappa_s = 0, B = 0 vs. the true-ray exp(-tau) (validation anchor 3
    is exact only for true-ray schemes - documented in plan/plan-sc-
    unigrid.md section 3/5):  z-entry ordinates are exact to machine
    precision (gap + target segment tile the true path), oblique
    (x/y-entry) ordinates carry a bounded angular stencil error of the
    same character as the chain method's grazing-ordinate error - it
    does NOT shrink with h (the discrete recursion's effective direction
    is set by the bilinear weights, not by Omega)."""
    dg = icosahedric_grid(n_refine=1)
    kappa = 0.5

    def errs(n):
        x = np.linspace(0.5, n - 0.5, n)
        grid = unigrid(x, x, x)
        box = cuboid(0.0, n, 0.0, n, 0.0, n)
        kappa_e = np.full(grid.n_cells, kappa)
        S = np.zeros(grid.n_cells)
        bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
        z_top = grid.box[2][1]
        out = []
        for omega in dg.directions:
            if omega[2] >= 0.0:
                continue  # downward ordinates: entry through the top
            sc = build_sc_chain(grid, omega, periodic_box=box)
            I = sweep_direction_sc(grid, sc, kappa_e, S, bc)
            path = (z_top - grid.centers[:, 2]) / abs(omega[2])
            out.append(np.max(np.abs(I - np.exp(-kappa * path))))
        return np.array(out)

    e8 = errs(8)
    e16 = errs(16)
    za = np.array(
        [sc_stencil(o, np.ones(3))["a"] for o in dg.directions if o[2] < 0]
    )
    # z-entry ordinates: exact (anchor 3 holds)
    assert np.allclose(e8[za == 2], 0.0, atol=1e-12)
    # oblique ordinates: bounded stencil path error, stable under
    # refinement (documented trade-off, not a discretization to be
    # refined away)
    oblique = za != 2
    assert np.all(e8[oblique] > 1e-6)
    assert np.all(e8[oblique] < 0.35)
    assert np.all(e16[oblique] < 0.35)


# ---------------------------------------------------------------------------
# 5. periodic ring: wrap-pass convergence
# ---------------------------------------------------------------------------


def test_periodic_ring_wrap_passes():
    """Vacuum ring with an x-entry ordinate: the wrap-pass sweep reaches
    the fixed point of the discrete SC system (residual ~ 0) and matches
    a dense linear solve of the same system.  The ring-circuit gain is
    ~ w_z0**n_x (w_z0 = same-z-layer bilinear weight) plus a non-normal
    transient, so the pass cap must exceed n_x - see sweep_direction_sc
    (default cap max(4 n_a, 64))."""
    n = 12
    x = np.linspace(0.5, n - 0.5, n)
    y = np.linspace(0.5, 3.5, 4)
    z = np.linspace(0.5, 4.5, 5)
    grid = unigrid(x, y, z)
    box = cuboid(0.0, n, 0.0, 4.0, 0.0, 5.0)
    omega = np.array([0.8, -0.3, -0.5])
    omega = omega / np.linalg.norm(omega)
    st = sc_stencil(omega, [grid.hx, grid.hy, grid.hz])
    assert st["a"] == 0

    kappa_e = np.zeros(grid.n_cells)
    S = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0, side_I=0.0)
    sc = build_sc_chain(grid, omega, periodic_box=box)
    assert sc["wrap"] is True

    I = sweep_direction_sc(grid, sc, kappa_e, S, bc, wrap_tol=1e-13)

    # fixed-point residual: recompute the recursion from the final I
    E = np.exp(-kappa_e * sc["t_f"])
    src = S * (1.0 - E)
    acc = np.zeros(grid.n_cells)
    for m in range(4):
        idx = sc["node_idx"][m]
        face = sc["node_face"][m]
        vals = I[idx] * E[idx] + src[idx]  # gap segment
        vals = np.where(face == 0, vals, bc.inflow(
            np.where(face == 1, 2, np.where(face == 2, 2, 0)),
            np.where(face == 1, 1, -1),
        ))
        acc += sc["weights"][m] * vals
    residual = np.max(np.abs(I - (acc * E + src)))
    assert residual < 1e-10

    # dense solve of the same linear system I = A I + b
    N = grid.n_cells
    A = np.zeros((N, N))
    b = np.zeros(N)
    cells = sc["order"]
    for m in range(4):
        idx = sc["node_idx"][m]
        face = sc["node_face"][m]
        interior = face == 0
        A[cells[interior], idx[interior]] += (
            sc["weights"][m] * E[cells[interior]] * E[idx[interior]]
        )
        np.add.at(
            b,
            cells[interior],
            sc["weights"][m] * E[cells[interior]] * src[idx[interior]],
        )
        np.add.at(
            b,
            cells[~interior],
            sc["weights"][m]
            * E[cells[~interior]]
            * bc.inflow(
                np.full(int((~interior).sum()), 2),
                np.where(face[~interior] == 1, 1, -1),
            ),
        )
    b = b * E + src
    I_dense = np.linalg.solve(np.eye(N) - A, b)
    assert np.allclose(I, I_dense, atol=1e-8)


def test_periodic_ring_vacuum_constant_inflow():
    """Vacuum, constant inflow on all faces, S = 0: the exact discrete
    solution is the constant inflow (ring passes must reproduce it).
    Worst case: n = 24, most-grazing 42-ordinate direction (circuit gain
    ~ 0.30 plus non-normal transient -> ~50 passes to 1e-12, within the
    default cap max(4 n_a, 64))."""
    n = 24
    x = np.linspace(0.5, n - 0.5, n)
    grid = unigrid(x, x, x)
    box = cuboid(0.0, n, 0.0, n, 0.0, n)
    dg = icosahedric_grid(n_refine=1)
    kappa_e = np.zeros(grid.n_cells)
    S = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=2.5, bottom_I=2.5, side_I=2.5)
    for omega in dg.directions:
        sc = build_sc_chain(grid, omega, periodic_box=box)
        I = sweep_direction_sc(grid, sc, kappa_e, S, bc, wrap_tol=1e-13)
        assert np.allclose(I, 2.5, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. slab + Gaussian cloud: SC vs. chain, convergence toward reference
# ---------------------------------------------------------------------------


def _cloud_case(n):
    x = np.linspace(0.5, n - 0.5, n)
    grid = unigrid(x, x, x)
    box = cuboid(0.0, n, 0.0, n, 0.0, n)
    c = grid.centers
    c0 = np.array([n / 2, n / 2, n / 2])
    sig = n / 6.0
    gauss = np.exp(-np.sum((c - c0) ** 2, axis=1) / (2 * sig**2))
    kappa_e = 0.02 + 1.5 * gauss
    kappa_s = 0.9 * kappa_e
    B = np.zeros(grid.n_cells)
    dg = icosahedric_grid(n_refine=1)
    bc = boundary_conditions(top_I=1.0)
    return grid, box, dg, kappa_e, kappa_s, B, bc


def test_sc_vs_chain_and_refinement():
    grids = {}
    for n in (8, 16):
        grid, box, dg, kappa_e, kappa_s, B, bc = _cloud_case(n)
        res_sc = lambda_iteration(
            grid, dg, kappa_e, kappa_s, B, bc, g=0.3, periodic_box=box,
            n_iter=200, tol=1e-7, transport="sc",
        )
        res_ch = lambda_iteration(
            grid, dg, kappa_e, kappa_s, B, bc, g=0.3, periodic_box=box,
            n_iter=200, tol=1e-7, transport="chain",
        )
        assert res_sc["converged"] and res_ch["converged"]
        grids[n] = (res_sc, res_ch)

    def rel(a, b):
        return np.max(np.abs(a - b)) / np.max(b)

    d8 = rel(grids[8][0]["J"], grids[8][1]["J"])
    d16 = rel(grids[16][0]["J"], grids[16][1]["J"])
    # both methods physical: J positive, illuminated from above
    for n in (8, 16):
        assert np.all(grids[n][0]["J"] > 0.0)
    # SC agrees with the chain method at fixed grid, and the difference
    # shrinks as h -> 0 (both converge to the same exact solution)
    assert d16 < d8
    assert d8 < 0.5  # same ballpark on the coarse grid


# ---------------------------------------------------------------------------
# 6. solver integration: transport="sc"
# ---------------------------------------------------------------------------


def test_solver_sc_matches_chain_slab():
    z = np.linspace(0.5, 19.5, 20)
    grid = unigrid(np.array([0.0]), np.array([0.0]), z)
    box = cuboid(-0.5, 0.5, -0.5, 0.5, 0.0, 20.0)
    dg = icosahedric_grid(n_refine=1)
    kappa_e = np.full(grid.n_cells, 0.25)
    kappa_s = 0.9 * kappa_e
    B = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=1.0)
    kw = dict(g=0.0, periodic_box=box, n_iter=300, tol=1e-8, ng=True)
    r_sc = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc,
                            transport="sc", **kw)
    r_ch = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc,
                            transport="chain", **kw)
    assert r_sc["converged"] and r_ch["converged"]
    assert r_sc["transport"] == "sc"
    # plane-parallel conservative slab: both methods reproduce the
    # two-stream-like solution closely (SC's oblique-ordinate path error
    # on the 20-cell tau=5 slab shows up as a few percent near the bottom)
    assert np.max(np.abs(r_sc["J"] - r_ch["J"])) / np.max(r_ch["J"]) < 0.1
    # flux profiles agree between methods (F_z decays with depth through
    # the absorption at omega = 0.9; the decay itself must match)
    fz_sc = r_sc["F"][:, 2]
    fz_ch = r_ch["F"][:, 2]
    assert np.max(np.abs(fz_sc - fz_ch)) / np.max(np.abs(fz_ch)) < 0.1


def test_solver_sc_ground_beam_ng_deltaM():
    """transport="sc" through the full solver machinery: ground, beam,
    Ng, delta-M all method-agnostic and reused unchanged."""
    n = 8
    x = np.linspace(0.5, n - 0.5, n)
    grid = unigrid(x, x, x)
    box = cuboid(0.0, n, 0.0, n, 0.0, n)
    dg = icosahedric_grid(n_refine=1)
    rng = np.random.default_rng(7)
    kappa_e = 0.05 + rng.uniform(0.0, 1.0, grid.n_cells)
    kappa_s = 0.7 * kappa_e
    B = np.full(grid.n_cells, 1.0)
    sun = np.array([0.25, -0.15, -0.957])
    sun = sun / np.linalg.norm(sun)
    beam = {"omega_sun": sun, "F0": 2.0}

    from rt3d.geometry import ground_surface

    ground = ground_surface(albedo=0.4, emissivity=0.8, B_g=1.5)
    bc = boundary_conditions(top_I=0.5, ground=ground)

    r_sc = lambda_iteration(
        grid, dg, kappa_e, kappa_s, B, bc, g=0.6, periodic_box=box,
        beam=beam, n_iter=300, tol=1e-7, ng=True, delta_M=1, transport="sc",
    )
    r_ch = lambda_iteration(
        grid, dg, kappa_e, kappa_s, B, bc, g=0.6, periodic_box=box,
        beam=beam, n_iter=300, tol=1e-7, ng=True, delta_M=1, transport="chain",
    )
    assert r_sc["converged"] and r_ch["converged"]
    assert np.all(np.isfinite(r_sc["I"]))
    assert np.all(r_sc["J"] > 0.0)
    # ground reflection present: upward intensity at the bottom exceeds
    # the pure-boundary value
    mb = np.flatnonzero(np.arange(grid.n_cells) < grid.ny * grid.nx)
    assert np.mean(r_sc["I"][np.ix_(mb, dg.up_idx)]) > 0.5
    # same-solution agreement, loose on the coarse grid
    dJ = np.max(np.abs(r_sc["J"] - r_ch["J"])) / np.max(r_ch["J"])
    assert dJ < 0.35


def test_solver_transport_option_validation():
    z = np.array([0.5, 1.5])
    grid = unigrid(np.array([0.0]), np.array([0.0]), z)
    dg = icosahedric_grid(n_refine=0)
    kappa_e = np.ones(grid.n_cells)
    kappa_s = np.zeros(grid.n_cells)
    B = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=1.0)
    with pytest.raises(ValueError, match="transport"):
        lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, transport="bogus")


def test_sweep_all_sc_convention():
    """sweep_all_sc matches the sweep_all signature/return convention and
    reuses precomputes."""
    grid, box, dg, kappa_e, _, B, bc = _cloud_case(6)
    S = (1.0 - 0.5) * B + 0.5
    I1, scs = sweep_all_sc(grid, dg, kappa_e, S, bc, periodic_box=box)
    assert I1.shape == (grid.n_cells, dg.n_dir)
    assert len(scs) == dg.n_dir
    I2, scs2 = sweep_all_sc(grid, dg, kappa_e, S, bc, periodic_box=box, scs=scs)
    assert np.array_equal(I1, I2)  # deterministic reuse
    # direction-dependent source supported like sweep_all
    S2 = np.broadcast_to(S[:, None], (grid.n_cells, dg.n_dir)).copy()
    I3, _ = sweep_all_sc(grid, dg, kappa_e, S2, bc, periodic_box=box, scs=scs)
    assert np.allclose(I3, I1)
