# -*- coding: utf-8 -*-
"""Phase 3: octree build, point location, DDA march, and octree transport."""

import numpy as np
import pytest

from rt3d.characteristics import build_chain, build_chain_octree
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, march_lattice, ray_box_exit, unigrid
from rt3d.octree import march_octree, octree_grid
from rt3d.source import single_scattering_albedo
from rt3d.transport import (
    beam_optical_depth,
    beam_optical_depth_octree,
    boundary_conditions,
    precompute_octree_march,
    sweep_direction,
    sweep_direction_octree,
    sweep_direction_octree_exact,
)


def make_oct(tau_max=0.5, eps_grad=0.3, max_depth=3, h0=1.0):
    """Octree on [0, 4]^3 with a compact cloud (kappa 10 inside r < 1.2
    around (2, 2, 2), 0.05 outside)."""
    box = cuboid(0.0, 4.0, 0.0, 4.0, 0.0, 4.0)

    def kappa_e(x, y, z):
        r = np.sqrt((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2)
        return np.where(r < 1.2, 10.0, 0.05)

    oct = octree_grid(box, h0=h0)
    oct.refine(kappa_e, tau_max=tau_max, eps_grad=eps_grad, max_depth=max_depth)
    oct.balance_21(max_depth=max_depth)
    oct.finalize()
    oct.sample_fields(kappa_e=kappa_e)
    return oct


# --------------------------------------------------------------------- build


def test_refine_deepens_at_cloud_and_respects_depth():
    oct_coarse = make_oct(tau_max=2.0, max_depth=2)
    oct_fine = make_oct(tau_max=0.5, max_depth=3)
    assert oct_coarse.n_leaf > 64
    assert oct_fine.n_leaf > oct_coarse.n_leaf
    assert oct_fine.level.max() <= 3
    # deepest cells sit at/inside the cloud
    deep = oct_fine.level == oct_fine.level.max()
    r_deep = np.linalg.norm(oct_fine.centers[deep] - np.array([2.0, 2.0, 2.0]),
                            axis=1)
    assert np.all(r_deep < 1.6)
    # vacuum corners stay coarse (only the 2:1 balance cascade reaches them)
    corner = (oct_fine.centers[:, 0] < 1.0) & (oct_fine.centers[:, 2] > 3.0)
    assert corner.sum() > 0
    assert np.all(oct_fine.level[corner] < oct_fine.level.max())


def test_balance_21_face_neighbors():
    oct = make_oct(tau_max=0.5, max_depth=3)
    c, h, lvl = oct.centers, oct.half, oct.level
    for axis in range(3):
        trans = [a for a in range(3) if a != axis]
        for side in (-1.0, 1.0):
            for o1 in (-0.5, 0.0, 0.5):
                for o2 in (-0.5, 0.0, 0.5):
                    probe = c.copy()
                    probe[:, axis] += side * (h + 1e-9 * h + 1e-12)
                    probe[:, trans[0]] += o1 * h
                    probe[:, trans[1]] += o2 * h
                    inside = oct.cuboid.contains(probe)
                    nb = np.where(inside, oct.locate(probe), -1)
                    nb_lvl = np.where(nb >= 0, oct.level[nb], lvl)
                    assert np.all(np.abs(nb_lvl - lvl) <= 1)


# -------------------------------------------------------------------- locate


def test_locate_contains_points():
    oct = make_oct(tau_max=0.5, max_depth=3)
    rng = np.random.default_rng(42)
    p = rng.uniform([0.001, 0.001, 0.001], [3.999, 3.999, 3.999], size=(500, 3))
    leaf = oct.locate(p)
    assert np.all(leaf >= 0) and np.all(leaf < oct.n_leaf)
    d = np.abs(p - oct.centers[leaf])
    assert np.all(d <= oct.half[leaf][:, None] + 1e-9)


# --------------------------------------------------------------------- march


def test_march_segment_lengths_match_budget():
    oct = make_oct(tau_max=0.5, max_depth=3)
    rng = np.random.default_rng(7)
    K = 60
    origin = rng.uniform(0.01, 3.99, size=(K, 3))
    v = rng.normal(size=(K, 3))
    d = v / np.linalg.norm(v, axis=1)[:, None]
    budget = rng.uniform(0.5, 3.0, size=K)
    # rays stop at the domain boundary: clamp budgets to the exit distance
    t_exit, _, _ = ray_box_exit(origin, d, oct.box)
    budget = np.minimum(budget, t_exit)
    sr, sl, sLen = march_octree(origin, d, oct, budget)
    tot = np.zeros(K)
    np.add.at(tot, sr, sLen)
    assert np.allclose(tot, budget)
    assert np.all(sl >= 0) and np.all(sl < oct.n_leaf)
    # every ray is active at least one step
    assert set(sr) == set(range(K))


def test_march_matches_lattice_on_single_level():
    """On a single-level octree the march equals the unigrid DDA march."""
    box = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    oct = octree_grid(box, h0=0.5)
    oct.finalize()
    x = np.linspace(0.25, 1.75, 4)
    grid = unigrid(x, x, x)

    rng = np.random.default_rng(3)
    origin = rng.uniform(0.01, 1.99, size=(20, 3))
    d = np.array([0.5, -0.3, 0.8]) / np.linalg.norm([0.5, -0.3, 0.8])
    # march_lattice runs to the box exit: give the octree march the same budget
    budget, _, _ = ray_box_exit(origin, d, grid.box)

    sr, sl, sLen = march_octree(origin, d, oct, budget)
    steps, _, _, _ = march_lattice(origin, d, grid)
    uni = {}
    for ridx, cell, ln in steps:
        for r, c, l in zip(ridx, cell, ln):
            uni.setdefault(int(r), []).append((int(c), float(l)))
    oct_by_ray = {}
    for r, leaf, ln in zip(sr, sl, sLen):
        oct_by_ray.setdefault(int(r), []).append((int(leaf), ln))
    assert set(oct_by_ray) == set(uni)
    for r in uni:
        # merge consecutive segments in the same cell
        merged = []
        for cell, ln in oct_by_ray[r]:
            if merged and merged[-1][0] == cell:
                merged[-1][1] += ln
            else:
                merged.append([cell, ln])
        assert [c for c, _ in merged] == [c for c, _ in uni[r]]
        assert np.allclose([l for _, l in merged], [l for _, l in uni[r]])


# --------------------------------------------------------- octree transport


def test_single_level_octree_sweep_vs_true_ray():
    """kappa_s = 0, uniform kappa: octree chain path lengths stay close to
    the exact ray path lengths (the unigrid lattice stencil deviates much
    more on grazing ordinates of this coarse 4^3 grid)."""
    dg = icosahedric_grid(n_refine=1)
    box = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    oct = octree_grid(box, h0=0.5)
    oct.finalize()
    oct.sample_fields(kappa_e=lambda x, y, z: np.full(x.size, 1.0))

    x = np.linspace(0.25, 1.75, 4)
    grid = unigrid(x, x, x)
    grid.kappa_e = oct.kappa_e

    bc = boundary_conditions(top_I=1.0, bottom_I=1.0)
    S = np.zeros(oct.n_leaf)

    dev_oct, dev_uni = [], []
    for d in dg.directions:
        chain = build_chain_octree(oct, d, periodic_box=box)
        I_oct = sweep_direction_octree(oct, chain, oct.kappa_e, S, bc)
        chain_u = build_chain(grid, d, periodic_box=box)
        I_uni = sweep_direction(grid, chain_u, grid.kappa_e, S, bc)
        # exact ray path length from the domain entry to each center
        z = oct.centers[:, 2]
        t_e = np.where(d[2] < 0.0, (2.0 - z) / (-d[2]), (z - 0.0) / d[2])
        sr, sl, sLen = march_octree(oct.centers, -d, oct, t_e, periodic_box=box)
        L_true = np.zeros(oct.n_leaf)
        np.add.at(L_true, sr, sLen)
        with np.errstate(divide="ignore"):
            L_oct = -np.log(np.maximum(I_oct, 1e-300))
            L_uni = -np.log(np.maximum(I_uni, 1e-300))
        assert np.all(np.isfinite(L_oct))
        dev_oct.append(np.max(np.abs(L_oct - L_true)))
        dev_uni.append(np.mean(np.abs(L_uni - L_true)))
    # octree chain path deviation bounded on the cubic grid ...
    assert max(dev_oct) < 1.0
    # ... and on average smaller than the unigrid lattice-stencil deviation
    assert np.mean(dev_oct) < np.mean(dev_uni)


# ------------------------------------------------------- exact-march sweep


def test_exact_march_beer_lambert():
    """kappa_s = 0, S = 0, uniform kappa: the exact-march sweep reproduces
    exp(-kappa * t_e) at machine precision for every ordinate (anchor 3),
    including rays wrapping through periodic side faces."""
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
        I = sweep_direction_octree_exact(oct, march, oct.kappa_e, S, bc)
        t_e = np.where(d[2] < 0.0, (2.0 - z) / (-d[2]), (z - 0.0) / d[2])
        assert np.allclose(I, np.exp(-t_e), rtol=1e-12, atol=1e-15)


def test_exact_march_uniform_medium_source():
    """Uniform kappa and uniform S: analytic slab solution along the ray."""
    dg = icosahedric_grid(n_refine=1)
    box = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    oct = octree_grid(box, h0=0.5)
    oct.finalize()
    kappa = 2.0
    oct.sample_fields(kappa_e=lambda x, y, z: np.full(x.size, kappa))
    bc = boundary_conditions(top_I=1.0, bottom_I=0.5)
    S = np.full(oct.n_leaf, 3.0)
    z = oct.centers[:, 2]
    for d in dg.directions:
        march = precompute_octree_march(oct, d, periodic_box=box)
        I = sweep_direction_octree_exact(oct, march, oct.kappa_e, S, bc)
        t_e = np.where(d[2] < 0.0, (2.0 - z) / (-d[2]), (z - 0.0) / d[2])
        I_in = np.where(d[2] < 0.0, bc.top_I, bc.bottom_I)
        expected = I_in * np.exp(-kappa * t_e) \
            + S * (1.0 - np.exp(-kappa * t_e))
        assert np.allclose(I, expected, rtol=1e-12, atol=1e-15)


def test_exact_march_vs_per_ray_reference():
    """Heterogeneous refined octree, heterogeneous S: the vectorized
    sweep matches a plain per-ray entry->center recursion over the raw
    march segments."""
    oct = make_oct(tau_max=0.5, eps_grad=0.3, max_depth=3)
    rng = np.random.default_rng(7)
    S = rng.random(oct.n_leaf) + 0.1
    box = cuboid(0.0, 4.0, 0.0, 4.0, 0.0, 4.0)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.3)
    z = oct.centers[:, 2]
    for d in dg_directions_sample():
        march = precompute_octree_march(oct, d, periodic_box=box)
        I = sweep_direction_octree_exact(oct, march, oct.kappa_e, S, bc)

        budget = np.where(d[2] < 0.0, (box.z1 - z) / (-d[2]),
                          (z - box.z0) / d[2])
        sr, sl, sLen = march_octree(oct.centers, -d, oct, budget,
                                    periodic_box=box)
        order = np.argsort(sr, kind="stable")
        sr, sl, sLen = sr[order], sl[order], sLen[order]
        counts = np.bincount(sr, minlength=oct.n_leaf)
        offs = np.concatenate(([0], np.cumsum(counts)))
        I_in = bc.top_I if d[2] < 0.0 else bc.bottom_I
        I_ref = np.empty(oct.n_leaf)
        for i in range(oct.n_leaf):
            # raw march order is center -> entry; recurse entry -> center
            segs = range(offs[i + 1] - 1, offs[i] - 1, -1)
            Icur = I_in
            for s in segs:
                tau = oct.kappa_e[sl[s]] * sLen[s]
                Icur = Icur * np.exp(-tau) + S[sl[s]] * (-np.expm1(-tau))
            I_ref[i] = Icur
        # cumsum-based assembly vs per-ray loop: float rounding only
        assert np.allclose(I, I_ref, rtol=1e-9, atol=1e-10)


def dg_directions_sample():
    """A few ordinates incl. steep and grazing ones."""
    dg = icosahedric_grid(n_refine=1)
    iz = np.argsort(dg.directions[:, 2])
    return dg.directions[[iz[0], iz[5], iz[20], iz[36], iz[-1]]]


def test_octree_beer_transmittance_vs_uniform_beam():
    """kappa_s = 0 beam anchor: octree tau_b agrees with the unigrid tau_b
    on the same sampled kappa field (both exact along their own rays)."""
    box = cuboid(-5.0, 5.0, -5.0, 5.0, 0.0, 10.0)

    def kappa_fn(x, y, z):
        return 3.0 * np.exp(-((x**2 + y**2 + (z - 4) ** 2) / 2.0))

    oct = octree_grid(box, h0=1.25)
    oct.refine(kappa_fn, tau_max=0.5, eps_grad=0.3, max_depth=3)
    oct.balance_21(max_depth=3)
    oct.finalize()
    oct.sample_fields(kappa_e=kappa_fn)

    x = np.linspace(-4.84375, 4.84375, 32)
    z = np.linspace(0.15625, 9.84375, 32)
    grid = unigrid(x, x, z)
    cx = grid.centers
    grid.kappa_e = kappa_fn(cx[:, 0], cx[:, 1], cx[:, 2])

    omega_sun = np.array([0.3, 0.2, -0.93]) / np.linalg.norm([0.3, 0.2, -0.93])
    tau_oct = beam_optical_depth_octree(oct, omega_sun, oct.kappa_e, box)
    tau_uni = beam_optical_depth(grid, omega_sun, grid.kappa_e, box)

    # compare column depth at the cloud center column (z = 4, x = y = 0)
    i_oct = np.argmin(np.linalg.norm(oct.centers - [0, 0, 4], axis=1))
    i_uni = np.argmin(np.linalg.norm(grid.centers - [0, 0, 4], axis=1))
    assert np.isclose(tau_oct[i_oct], tau_uni[i_uni], rtol=0.05)
    assert np.all(tau_oct >= 0.0)


def test_chain_waves_cover_all_leaves_once():
    dg = icosahedric_grid(n_refine=1)
    oct = make_oct(tau_max=0.5, max_depth=3)
    for d in dg.directions[:6]:
        chain = build_chain_octree(oct, d, periodic_box=oct.cuboid)
        covered = np.concatenate([chain["roots"]] + chain["waves"])
        assert covered.size == oct.n_leaf
        assert np.unique(covered).size == oct.n_leaf
        # parents precede children in wave order
        pos = {int(node): i for i, node in enumerate(covered)}
        for w in chain["waves"]:
            for node in w:
                assert pos[int(chain["parent"][node])] < pos[int(node)]


def test_g_bucketed_source_matches_scalar():
    """scattering_source with per-cell g equal to a constant reproduces the
    scalar-g result."""
    from rt3d.source import scattering_source

    dg = icosahedric_grid(n_refine=0)
    rng = np.random.default_rng(5)
    n = 10
    I = rng.uniform(size=(n, dg.n_dir))
    omega = rng.uniform(size=n)
    B = rng.uniform(size=n)
    g = 0.4
    S1 = scattering_source(I, dg, omega, B, g=g)
    S2 = scattering_source(I, dg, omega, B, g_cells=np.full(n, g))
    assert np.allclose(S1, S2)
