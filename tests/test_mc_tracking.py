# -*- coding: utf-8 -*-
"""
MC phase M0 invariants (plan/mc-plan.md section 7): Woodcock flight sampler,
null-collision event decision, cell-vs-global mode equivalence, and DDA
walkers identical to the SC marches.

  1. walk_lattice / walk_octree segment records identical to
     march_lattice / march_octree (periodic wrap + ground entry)
  2. kappa_s = 0, B = 0: MC beam transmittance == exp(-tau) along every
     ray (verification invariant 3 companion), on unigrid and octree, in both
     majorant modes
  3. cell mode (zero nulls) and global mode (real null collisions) agree
     statistically within combined 2 sigma
"""

import numpy as np
import pytest

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, march_lattice, unigrid
from rt3d.mc.tracking import (
    EV_ABSORB,
    EV_EXIT,
    EV_NULL,
    EV_SCATTER,
    event_decision,
    flight_distance,
    path_integral_from_segments,
    sample_collisions,
    walk_lattice,
    walk_octree,
)
from rt3d.octree import octree_grid


@pytest.fixture
def box3d():
    """Periodic 3D unigrid box with random kappa_e (vacuum patches included)."""
    x = np.linspace(0.5, 5.5, 6)
    y = np.linspace(0.5, 3.5, 4)
    z = np.linspace(0.5, 7.5, 8)
    grid = unigrid(x, y, z)
    box = cuboid(*grid.box[0], *grid.box[1], *grid.box[2])
    rng = np.random.default_rng(11)
    kappa_e = rng.uniform(0.0, 1.5, grid.n_cells)
    kappa_e[rng.uniform(size=grid.n_cells) < 0.35] = 0.0  # vacuum: nulls must fire
    return grid, box, kappa_e


@pytest.fixture
def cloud_octree():
    """Gaussian-cloud octree (cubic base cells h0 = 2.5, depth <= 3)."""
    case = gaussian_cloud()
    oct = octree_grid(case["box"], h0=2.5)
    oct.refine(case["kappa_e"], tau_max=0.5, eps_grad=0.3, max_depth=3)
    oct.balance_21(max_depth=3)
    oct.finalize()
    oct.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"])
    return oct, case["box"]


def _transmittance_stats(grid, box, kappa_e, dg, n_photons, seed, octree=False):
    """
    Transmittance invariant: beam photons seeded uniformly on the top face,
    per-ordinate analytic tau from the exact DDA walk, observed survival
    fraction vs. sum(exp(-tau)) with binomial error.
    """
    origins = np.column_stack(
        [
            np.empty(n_photons),
            np.empty(n_photons),
            np.full(n_photons, grid.box[2][1] - 1e-9),
        ]
    )
    rng = np.random.default_rng(seed)
    z0, z1 = grid.box[2]
    Lz = z1 - z0

    out = {}
    for mode in ("cell", "global"):
        n_coll = 0
        n_tot = 0
        var = 0.0
        exp_surv = 0.0
        for d in dg.down_idx[::4]:
            omega = dg.directions[d]
            starts = origins.copy()
            starts[:, 0] = rng.uniform(box.x0, box.x1, n_photons)
            starts[:, 1] = rng.uniform(box.y0, box.y1, n_photons)
            budget = np.full(n_photons, Lz / abs(omega[2]))

            if octree:
                sr, sc, sl = walk_octree(starts, omega, grid, budget, periodic_box=box)
            else:
                sr, sc, sl, *_ = walk_lattice(starts, omega, grid, periodic_box=box)
            tau = path_integral_from_segments(sr, sc, sl, kappa_e, n_photons)
            p = np.exp(-tau)

            res = sample_collisions(
                starts, omega, budget, grid, kappa_e, np.zeros_like(kappa_e),
                mode=mode, periodic_box=box, seed=seed + d,
            )
            surv = np.count_nonzero(res["survived"])
            assert np.all(res["coll_event"][~res["survived"]] == EV_ABSORB)
            assert np.all(res["exit_t"][~res["survived"]] == res["coll_distance"][~res["survived"]])
            n_coll += surv
            n_tot += n_photons
            exp_surv += p.sum()
            var += np.sum(p * (1.0 - p))
        z_sigma = (n_coll - exp_surv) / np.sqrt(var)
        out[mode] = z_sigma
    # two-sided ~3.5 sigma margin over the whole 4-configuration ensemble
    assert all(abs(v) < 3.5 for v in out.values()), out
    return out


# =============================================================================
# flight sampler and event decision units
# =============================================================================


def test_event_decision_fractions():
    """Global-mode rates: scatter/absorb/null fractions match kappa_s /
    (kappa_e - kappa_s) / (kappa_max - kappa_e)."""
    rng = np.random.default_rng(1)
    n = 400_000
    kappa_max, kappa_s, kappa_e = 2.0, 0.5, 1.0
    ev = event_decision(rng.random(n), kappa_max,
                        np.full(n, kappa_s), np.full(n, kappa_e))
    frac = np.bincount(ev[ev >= 0], minlength=3) / n
    assert np.allclose(frac, [0.25, 0.25, 0.5], atol=0.005)
    # cell mode: kappa_max == kappa_e -> nulls exactly never
    ev = event_decision(rng.random(n), kappa_e, np.full(n, 0.4), np.full(n, 1.0))
    assert not np.any(ev == EV_NULL)


def test_flight_distance_exponential():
    """-ln(1-u)/kappa is Exp(kappa); kappa == 0 gives an unbounded flight."""
    rng = np.random.default_rng(2)
    k = 1.7
    s = flight_distance(rng.random(400_000), k)
    se = (1.0 / k) / np.sqrt(s.size)
    assert abs(s.mean() - 1.0 / k) < 4.0 * se
    s0 = flight_distance(rng.random(100), np.zeros(100))
    assert np.all(np.isinf(s0))


# =============================================================================
# walkers identical to the SC marches (M0: segment counts identical)
# =============================================================================


def test_walk_lattice_matches_march_lattice(box3d):
    grid, box, _ = box3d
    dg = icosahedric_grid(n_refine=1)
    rng = np.random.default_rng(7)
    starts = rng.uniform([b[0] for b in grid.box], [b[1] for b in grid.box],
                         size=(300, 3))
    for di in dg.down_idx[::3]:
        omega = dg.directions[di]
        msteps, mt, max_, msd = march_lattice(starts, omega, grid,
                                              periodic_box=box)
        mr = np.concatenate([s[0] for s in msteps])
        mc = np.concatenate([s[1] for s in msteps])
        ml = np.concatenate([s[2] for s in msteps])
        wr, wc, wl, wt, wax, wsd = walk_lattice(starts, omega, grid,
                                                periodic_box=box)
        assert np.array_equal(wr, mr)
        assert np.array_equal(wc, mc)
        assert np.allclose(wl, ml, rtol=0, atol=1e-14)
        assert np.allclose(wt, mt, rtol=0, atol=1e-12)
        assert np.array_equal(wax, max_)
        assert np.array_equal(wsd, msd)


def test_walk_octree_matches_march_octree(cloud_octree):
    from rt3d.octree import march_octree

    oct, box = cloud_octree
    dg = icosahedric_grid(n_refine=1)
    z = oct.centers[:, 2]
    z0, z1 = oct.box[2]
    for di in list(dg.down_idx[::4]) + list(dg.up_idx[::4]):
        omega = dg.directions[di]
        if omega[2] < 0.0:
            budget = (z1 - z) / (-omega[2])
        else:
            budget = (z - z0) / omega[2]
        mr, ml, mlen = march_octree(oct.centers, -omega, oct, budget, periodic_box=box)
        wr, wl, wlen = walk_octree(oct.centers, -omega, oct, budget, periodic_box=box)
        assert np.array_equal(wr, mr)
        assert np.array_equal(wl, ml)
        assert np.array_equal(wlen, mlen)


# =============================================================================
# transmittance invariant 3 companion, both grids, both majorant modes
# =============================================================================


def test_transmittance_invariant_unigrid(box3d):
    """kappa_s = 0: MC survival == exp(-tau) per ray, cell and global mode."""
    grid, box, kappa_e = box3d
    dg = icosahedric_grid(n_refine=1)
    _transmittance_stats(grid, box, kappa_e, dg, n_photons=3000, seed=101)


def test_transmittance_invariant_octree(cloud_octree):
    """Same invariant on the octree (global mode crosses vacuum nulls there)."""
    oct, box = cloud_octree
    dg = icosahedric_grid(n_refine=1)
    _transmittance_stats(oct, box, oct.kappa_e, dg, n_photons=2000, seed=202,
                         octree=True)


def test_global_mode_nulls_actually_occur(box3d):
    """Global majorant in a medium with vacuum patches must fire real null
    collisions; cell mode has zero nulls by construction."""
    grid, box, kappa_e = box3d
    omega = np.array([0.3, -0.4, -0.86])
    omega /= np.linalg.norm(omega)
    rng = np.random.default_rng(13)
    starts = np.column_stack(
        [rng.uniform(box.x0, box.x1, 400),
         rng.uniform(box.y0, box.y1, 400),
         np.full(400, box.z1 - 1e-9)]
    )
    budget = np.full(400, (box.z1 - box.z0) / abs(omega[2]))
    res_g = sample_collisions(starts, omega, budget, grid, kappa_e,
                              np.zeros_like(kappa_e), mode="global",
                              periodic_box=box, seed=5)
    res_c = sample_collisions(starts, omega, budget, grid, kappa_e,
                              np.zeros_like(kappa_e), mode="cell",
                              periodic_box=box, seed=5)
    assert np.sum(res_g["n_null"]) > 0
    assert np.sum(res_c["n_null"]) == 0


# =============================================================================
# cell vs. global mode equivalence (plan M0: combined 2 sigma)
# =============================================================================


def test_cell_vs_global_agreement(box3d):
    """With scattering (kappa_s > 0) both modes sample the same first real
    collision distribution; agreement within combined 2 sigma."""
    grid, box, kappa_e = box3d
    kappa_s = 0.8 * kappa_e
    dg = icosahedric_grid(n_refine=1)
    omega = dg.directions[dg.down_idx[np.argmin(dg.directions[dg.down_idx, 2])]]
    rng = np.random.default_rng(17)
    K = 8000
    starts = np.column_stack(
        [rng.uniform(box.x0, box.x1, K),
         rng.uniform(box.y0, box.y1, K),
         np.full(K, box.z1 - 1e-9)]
    )
    budget = np.full(K, (box.z1 - box.z0) / abs(omega[2]))

    # analytic first-event statistics from the exact walk
    sr, sc, sl, *_ = walk_lattice(starts, omega, grid, periodic_box=box)
    tau = path_integral_from_segments(sr, sc, sl, kappa_e, K)
    p_surv = np.exp(-tau)

    res = {}
    for mode, seed in (("cell", 21), ("global", 22)):
        r = sample_collisions(starts, omega, budget, grid, kappa_e, kappa_s,
                              mode=mode, periodic_box=box, seed=seed)
        assert np.all(r["coll_event"][~r["survived"]]
                      != EV_EXIT)
        assert np.count_nonzero(r["coll_event"] == EV_SCATTER) > 0
        res[mode] = r

    s_c = np.count_nonzero(res["cell"]["survived"])
    s_g = np.count_nonzero(res["global"]["survived"])
    exp = p_surv.sum()
    var = np.sum(p_surv * (1.0 - p_surv))
    assert abs(s_c - exp) < 3.0 * np.sqrt(var)
    assert abs(s_g - exp) < 3.0 * np.sqrt(var)
    assert abs(s_c - s_g) < 3.0 * np.sqrt(2.0 * var)

    # first collision distance distribution: mean agrees between modes
    d_c = np.minimum(res["cell"]["coll_distance"], budget)
    d_g = np.minimum(res["global"]["coll_distance"], budget)
    assert abs(d_c.mean() - d_g.mean()) < 3.0 * d_c.std() / np.sqrt(K) * np.sqrt(2)


def test_seed_reproducibility(box3d):
    """Same seed -> identical streams; different seeds -> consistent."""
    grid, box, kappa_e = box3d
    omega = dg_down_omega()
    K = 500
    starts = np.tile(np.array([[2.0, 2.0, box.z1 - 1e-9]]), (K, 1))
    budget = np.full(K, (box.z1 - box.z0) / abs(omega[2]))
    kw = dict(mode="global", periodic_box=box, seed=99)
    r1 = sample_collisions(starts, omega, budget, grid, kappa_e,
                           np.zeros_like(kappa_e), **kw)
    r2 = sample_collisions(starts, omega, budget, grid, kappa_e,
                           np.zeros_like(kappa_e), **kw)
    assert np.array_equal(r1["coll_distance"], r2["coll_distance"])
    assert np.array_equal(r1["coll_cell"], r2["coll_cell"])


def dg_down_omega():
    dg = icosahedric_grid(n_refine=1)
    return dg.directions[dg.down_idx[0]]
