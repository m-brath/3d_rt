# -*- coding: utf-8 -*-
"""
Parity tests for the numba fast paths of rt3d.jit
(plan/optimization-plan.md).  Every compiled kernel must reproduce its
NumPy reference: segment marches and unigrid sweeps bit-comparably, the
fused octree sweep to transcendental rounding.  With numba absent the
module-level fallbacks keep the reference paths, so these tests skip.
"""

import numpy as np
import pytest

from rt3d.characteristics import build_chain, stencil_for_direction
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid
from rt3d.jit import (
    HAVE_NUMBA,
    beam_optical_depth_octree_jit,
    chain_sweep_jit,
    locate_jit,
    march_octree_jit,
    sc_sweep_jit,
    sweep_direction_octree_exact_jit,
)
from rt3d.octree import march_octree, octree_grid
from rt3d.short_characteristics import build_sc_chain, sweep_direction_sc
from rt3d.source import _bucket_structure, scattering_source
from rt3d.transport import (
    beam_optical_depth_octree,
    boundary_conditions,
    precompute_octree_march,
    sweep_direction,
    sweep_direction_octree_exact,
)

pytestmark = pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")


def make_oct(max_depth=3, h0=1.0):
    """Octree on [0, 4]^3 with a compact cloud (as in tests/test_octree)."""
    box = cuboid(0.0, 4.0, 0.0, 4.0, 0.0, 4.0)

    def kappa_e(x, y, z):
        r = np.sqrt((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2)
        return np.where(r < 1.2, 10.0, 0.05)

    oct = octree_grid(box, h0=h0)
    oct.refine(kappa_e, tau_max=0.5, eps_grad=0.3, max_depth=max_depth)
    oct.balance_21(max_depth=max_depth)
    oct.finalize()
    oct.sample_fields(kappa_e=kappa_e)
    return oct


DIRS = [
    np.array([0.3, -0.5, 0.8]),
    np.array([-0.6, 0.2, -0.75]),
    np.array([0.0, 0.9, 0.4]),
    np.array([0.2, 0.3, 0.9]),
]
DIRS = [d / np.linalg.norm(d) for d in DIRS]


# ----------------------------------------------------------------- locate


def test_locate_jit_bit_identical():
    oct = make_oct()
    rng = np.random.default_rng(0)
    pts = rng.random((5000, 3)) * 3.99 + 0.005
    pts = np.r_[pts, oct.centers, np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])]
    assert np.array_equal(oct.locate(pts), locate_jit(pts, oct))


# ------------------------------------------------------------------ march


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("d", DIRS)
def test_march_jit_bit_identical_per_ray(d, periodic):
    oct = make_oct()
    box = oct.cuboid if periodic else None
    budget = np.full(oct.n_leaf, 6.0)
    sr, sl, sLen = march_octree(oct.centers, d, oct, budget, periodic_box=box)
    jr, jl, jL = march_octree_jit(oct.centers, d, oct, budget, periodic_box=box)

    def per_ray(r, leaf, ln):
        out = {}
        for a, b, c in zip(r, leaf, ln):
            out.setdefault(int(a), []).extend([int(b), float(c)])
        return out

    assert per_ray(sr, sl, sLen) == per_ray(jr, jl, jL)


# ------------------------------------------------------- fused octree sweep


def test_fused_sweep_matches_reference():
    oct = make_oct()
    bc = boundary_conditions(top_I=1.0, bottom_I=0.3)
    rng = np.random.default_rng(1)
    S = rng.random(oct.n_leaf)
    for d in DIRS:
        m_ref = precompute_octree_march(oct, d, periodic_box=oct.cuboid,
                                        jit=False)
        I_ref = sweep_direction_octree_exact(oct, m_ref, oct.kappa_e, S, bc)
        m_jit = precompute_octree_march(oct, d, periodic_box=oct.cuboid)
        assert m_jit["fused"]
        I_jit = sweep_direction_octree_exact(oct, m_jit, oct.kappa_e, S, bc)
        assert np.allclose(I_jit, I_ref, rtol=1e-9, atol=1e-12)


def test_fused_sweep_ground_bc_matches_reference():
    oct = make_oct()
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    rng = np.random.default_rng(2)
    S = rng.random(oct.n_leaf)
    bottom_I = rng.random(oct.n_leaf)
    d = DIRS[3]  # omega_z > 0
    m_ref = precompute_octree_march(oct, d, periodic_box=oct.cuboid, jit=False)
    m_jit = precompute_octree_march(oct, d, periodic_box=oct.cuboid)
    assert np.array_equal(m_ref["entry_cell"], m_jit["entry_cell"])
    # bottom_I convention: per-ray inflow at the ray's ground entry point
    I_ref = sweep_direction_octree_exact(oct, m_ref, oct.kappa_e, S, bc,
                                         bottom_I=bottom_I)
    I_jit = sweep_direction_octree_exact(oct, m_jit, oct.kappa_e, S, bc,
                                         bottom_I=bottom_I)
    assert np.allclose(I_jit, I_ref, rtol=1e-9, atol=1e-12)


def test_fused_sweep_beer_lambert_anchor3():
    """kappa_s = 0, S = 0: fused sweep reproduces exp(-tau) (anchor 3)."""
    dg = icosahedric_grid(n_refine=1)
    box = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    oct = octree_grid(box, h0=0.5)
    oct.finalize()
    oct.sample_fields(kappa_e=lambda x, y, z: np.full(x.size, 1.0))
    bc = boundary_conditions(top_I=1.0, bottom_I=1.0)
    S = np.zeros(oct.n_leaf)
    z = oct.centers[:, 2]
    for d in dg.directions:
        march = precompute_octree_march(oct, d, periodic_box=box)
        assert march["fused"]
        I = sweep_direction_octree_exact(oct, march, oct.kappa_e, S, bc)
        t_e = np.where(d[2] < 0.0, (2.0 - z) / (-d[2]), (z - 0.0) / d[2])
        assert np.allclose(I, np.exp(-t_e), rtol=1e-12, atol=1e-15)


def test_beam_tau_jit_matches_reference():
    oct = make_oct()
    ws = np.array([0.4, 0.0, -np.sqrt(1.0 - 0.16)])
    tau_ref = beam_optical_depth_octree(oct, ws, oct.kappa_e, oct.cuboid,
                                        jit=False)
    tau_jit = beam_optical_depth_octree(oct, ws, oct.kappa_e, oct.cuboid)
    assert np.array_equal(tau_ref, tau_jit)


# --------------------------------------------------------- unigrid sweeps


def _mini_grids():
    box = cuboid(-5.0, 5.0, -5.0, 5.0, 0.0, 10.0)
    xs = np.linspace(-5.0, 5.0, 21)[:-1] + 0.5
    zs = np.linspace(0.0, 10.0, 13)[:-1] + (10.0 / 13) / 2
    grid = unigrid(xs, xs, zs)
    return box, grid


@pytest.mark.parametrize("periodic", (False, True))
def test_chain_sweep_jit_bit_identical(periodic):
    box, grid = _mini_grids()
    dg = icosahedric_grid(n_refine=1)
    rng = np.random.default_rng(3)
    k_e = np.abs(rng.normal(1.0, 0.5, grid.n_cells))
    S = rng.random(grid.n_cells)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.4, side_I=0.2)
    pb = box if periodic else None
    bottom_I = rng.random(grid.n_cells)
    for d in (dg.directions[2], dg.directions[17], dg.directions[40]):
        chain = build_chain(grid, d, periodic_box=pb)
        I_ref = sweep_direction(grid, chain, k_e, S, bc)
        I_jit = chain_sweep_jit(grid, chain, k_e, S, bc)
        assert np.array_equal(I_ref, I_jit)
        if periodic:
            I_ref = sweep_direction(grid, chain, k_e, S, bc, bottom_I=bottom_I)
            I_jit = chain_sweep_jit(grid, chain, k_e, S, bc, bottom_I=bottom_I)
            assert np.array_equal(I_ref, I_jit)


@pytest.mark.parametrize("periodic", (False, True))
def test_sc_sweep_jit_bit_identical(periodic):
    box, grid = _mini_grids()
    dg = icosahedric_grid(n_refine=1)
    rng = np.random.default_rng(4)
    k_e = np.abs(rng.normal(1.0, 0.5, grid.n_cells))
    S = rng.random(grid.n_cells)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.4, side_I=0.2)
    pb = box if periodic else None
    bottom_I = rng.random(grid.n_cells)
    for d in (dg.directions[5], dg.directions[21], dg.directions[36]):
        sc = build_sc_chain(grid, d, periodic_box=pb)
        I_ref = sweep_direction_sc(grid, sc, k_e, S, bc)
        I_jit = sc_sweep_jit(grid, sc, k_e, S, bc)
        assert np.array_equal(I_ref, I_jit)
        if periodic:
            I_ref = sweep_direction_sc(grid, sc, k_e, S, bc,
                                       bottom_I=bottom_I)
            I_jit = sc_sweep_jit(grid, sc, k_e, S, bc, bottom_I=bottom_I)
            assert np.array_equal(I_ref, I_jit)


# ------------------------------------------------- source bucket structure


def test_g_cells_buckets_bit_identical():
    """The cached bucket-contiguous matmul schedule matches the naive
    per-bucket boolean-mask loop bit-for-bit (P6)."""
    dg = icosahedric_grid(n_refine=1)
    rng = np.random.default_rng(5)
    n = 4000
    I = rng.random((n, dg.n_dir))
    omega = rng.random(n) * 0.7
    B = rng.random(n)
    g_cells = np.round(np.clip(rng.random(n) * 0.9, 0.0, 0.9), 2)

    cache = {}
    S_new = scattering_source(I, dg, omega, B, g_cells=g_cells,
                              Phi_cache=cache)
    gv = np.round(g_cells, 2)
    S_ref = np.zeros((n, dg.n_dir))
    for gv_u in np.unique(gv):
        rows = gv == gv_u
        Phi_u = dg.hg_matrix(float(gv_u))
        S_ref[rows] = (
            (1.0 - omega[rows])[:, None] * B[rows, None]
            + omega[rows, None] * (I[rows] @ Phi_u.T)
        )
    assert np.array_equal(S_new, S_ref)

    # delta-M path too
    cache = {}
    S_new = scattering_source(I, dg, omega, B, g_cells=g_cells,
                              Phi_cache=cache, delta_M=1)
    S_ref = np.zeros((n, dg.n_dir))
    for gv_u in np.unique(gv):
        rows = gv == gv_u
        Pwd, moms = dg.hg_delta_matrices(float(gv_u), 1)
        gM = float(gv_u)
        scat = I[rows] @ Pwd.T
        for P in moms:
            scat += I[rows] @ P.T
        denom = 1.0 - omega[rows] * gM
        S_ref[rows] = (
            (1.0 - omega[rows])[:, None] * B[rows, None]
            + omega[rows, None] * scat
            - omega[rows, None] * gM * I[rows]
        ) / denom[:, None]
    assert np.array_equal(S_new, S_ref)


# --------------------------------------------------------- stencil search


def test_stencil_vectorized_matches_reference_loop():
    import itertools

    def old_stencil(omega, h, max_offset):
        best_s, best_cos, best_norm2 = None, -np.inf, np.inf
        for s in itertools.product(range(-max_offset, max_offset + 1),
                                   repeat=3):
            if s == (0, 0, 0) or s[2] == 0:
                continue
            v = np.asarray(s, dtype=float) * h
            c = float(v @ omega) / np.linalg.norm(v)
            n2 = int(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
            if (c > best_cos + 1e-12
                    or (abs(c - best_cos) <= 1e-12 and n2 < best_norm2)):
                best_s, best_cos, best_norm2 = np.array(s), c, n2
        v = best_s * h
        return best_s, v / np.linalg.norm(v)

    dg = icosahedric_grid(n_refine=1)
    for h in (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.25])):
        for d in dg.directions[::7]:
            for r in (3, 6, 12):
                s1, v1 = old_stencil(d, h, r)
                s2, v2 = stencil_for_direction(d, h, r)
                assert np.array_equal(s1, s2)
                assert np.allclose(v1, v2)
