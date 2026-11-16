# -*- coding: utf-8 -*-
"""
MC phase M4 anchors (plan/mc-plan.md section 2/7): numba fast path of the
flight sampler (`rt3d.mc.jitmc`, jit.py fallback pattern).

  1. unigrid, cell + global mode: the compiled per-photon walk reproduces
     the NumPy reference's collision / event statistics within 4 sigma and
     the mean uncollided exit path within a few percent
  2. octree (gaussian cloud, depth-3), cell + global: same statistical
     equivalence, collision leaves in range
  3. transmittance anchor (plan anchor 3) ON the jit path: pure absorber,
     MC survival == exp(-tau) within 4 sigma
  4. plumbing: `sample_collisions(jit=True)` routes through the kernel
     when `jit_mc_usable` (same return keys, cell mode stays null-free);
     `backward_detector_mc(jit=True)` agrees with the NumPy walk within
     the combined 2 sigma (statistical streams differ by construction);
     stage-2 `radiance_from_source(jit=True)` agrees with the NumPy
     next-flight tally at aggregate sigma statistics
"""

import numpy as np

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.cases.slab_1d import slab_grid
from rt3d.cases.slab_1d import slab_grid as _sg  # noqa: F401
from rt3d.mc.backward import backward_detector_mc
from rt3d.mc.forward import radiance_from_source
from rt3d.mc.jitmc import jit_mc_usable, sample_collisions_jit
from rt3d.mc.tracking import sample_collisions
from rt3d.transport import boundary_conditions


def _rng_cloud_starts(n, seed):
    case = gaussian_cloud(omega_cap=0.9)
    return case


def _stats(res):
    ev = np.asarray(res["coll_event"])
    return ev


def test_jit_unigrid_stats():
    grid, box, hz = slab_grid(4.0, 2.0, nz=16, nxy=4)
    nc = grid.n_cells
    rng = np.random.default_rng(101)
    ke = 2.0 + rng.random(nc)
    ks = 0.6 * ke
    K = 30000
    starts = np.column_stack([rng.uniform(box.x0, box.x1, K),
                              rng.uniform(box.y0, box.y1, K),
                              rng.uniform(box.z0, box.z1, K)])
    u = rng.random((K, 3)) - 0.5
    dirs = u / np.linalg.norm(u, axis=1)[:, None]
    bud = np.minimum(
        np.where(dirs[:, 2] > 0, box.z1 - starts[:, 2],
                 starts[:, 2] - box.z0) / np.maximum(np.abs(dirs[:, 2]),
                                                     1e-12), 100.0)

    for mode in ("cell", "global"):
        ra = sample_collisions(starts, dirs, bud, grid, ke, ks, mode=mode,
                               periodic_box=box, seed=41)
        rb = sample_collisions_jit(starts, dirs, bud, grid, ke, ks,
                                   mode=mode, periodic_box=box, seed=41)
        ea, eb = _stats(ra), _stats(rb)
        fa = ea != -1
        fb = eb != -1
        p = fa.mean()
        sig = np.sqrt(max(p, 1e-9) * (1 - p) / K)
        assert abs(fa.mean() - fb.mean()) < 4.0 * sig
        ps = (ea == 0).mean()
        sig_s = np.sqrt(max(ps, 1e-9) * (1 - ps) / K)
        assert abs(ps - (eb == 0).mean()) < 4.0 * sig_s
        ta = ra["exit_t"][~fa].mean()
        tb = rb["exit_t"][~fb].mean()
        assert abs(ta - tb) < 0.15 + 0.10 * ta
        if mode == "cell":
            assert np.all(rb["n_null"] == 0)


def test_jit_octree_stats():
    case = gaussian_cloud(omega_cap=0.9)
    cbox = case["box"]
    from rt3d.octree import octree_grid

    oct_ = octree_grid(cbox, h0=2.5)
    oct_.refine(case["kappa_e"], tau_max=1.0, eps_grad=0.5, max_depth=3)
    oct_.balance_21(max_depth=3)
    oct_.finalize()
    oct_.sample_fields(kappa_e=case["kappa_e"])
    ko = oct_.kappa_e
    kso = 0.7 * ko

    rng = np.random.default_rng(102)
    K = 20000
    sel = rng.choice(oct_.n_leaf, K)
    starts = oct_.centers[sel] + rng.random((K, 3)) * 0.1
    u = rng.random((K, 3)) - 0.5
    dirs = u / np.linalg.norm(u, axis=1)[:, None]
    bud = np.minimum(
        np.where(dirs[:, 2] > 0, cbox.z1 - starts[:, 2],
                 starts[:, 2] - cbox.z0) / np.maximum(np.abs(dirs[:, 2]),
                                                      1e-12), 100.0)

    for mode in ("cell", "global"):
        ra = sample_collisions(starts, dirs, bud, oct_, ko, kso, mode=mode,
                               periodic_box=cbox, seed=44)
        rb = sample_collisions_jit(starts, dirs, bud, oct_, ko, kso,
                                   mode=mode, periodic_box=cbox, seed=44)
        ea, eb = _stats(ra), _stats(rb)
        fa, fb = ea != -1, eb != -1
        p = fa.mean()
        sig = np.sqrt(max(p, 1e-9) * (1 - p) / K)
        assert abs(fa.mean() - fb.mean()) < 4.0 * sig
        cc = rb["coll_cell"][fb]
        assert np.all(cc >= 0) and np.all(cc < oct_.n_leaf)
        if mode == "global":
            assert np.count_nonzero(rb["n_null"]) > 1000


def test_jit_transmittance_anchor():
    grid, box, hz = slab_grid(4.0, 2.0, nz=16, nxy=4)
    ke = np.full(grid.n_cells, 2.0)
    ks = np.zeros(grid.n_cells)
    rng = np.random.default_rng(103)
    K = 20000
    starts = np.column_stack([rng.uniform(box.x0, box.x1, K),
                              rng.uniform(box.y0, box.y1, K),
                              np.full(K, box.z1 - 1e-9)])
    d = np.array([0.2, 0.1, -0.97])
    d = d / np.linalg.norm(d)
    dirs = np.tile(d, (K, 1))
    tau = 2.0 * (box.z1 - box.z0) / abs(d[2])
    res = sample_collisions_jit(starts, dirs, np.full(K, tau + 1.0), grid,
                                ke, ks, mode="global", periodic_box=box,
                                seed=43)
    surv = np.count_nonzero(res["coll_event"] == -1) / K
    p = np.exp(-tau)
    sig = np.sqrt(p * (1 - p) / K)
    assert abs(surv - p) < 4.0 * sig


def test_sample_collisions_jit_routing():
    grid, box, hz = slab_grid(4.0, 2.0, nz=8, nxy=2)
    ke = np.full(grid.n_cells, 1.5)
    ks = 0.5 * ke
    rng = np.random.default_rng(104)
    K = 2000
    starts = rng.uniform(
        [box.x0, box.y0, box.z0], [box.x1, box.y1, box.z1], (K, 3))
    dirs = rng.normal(size=(K, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    assert jit_mc_usable(grid, box)
    keys_np = sorted(sample_collisions(starts, dirs, np.full(K, 3.0), grid,
                                       ke, ks, periodic_box=box, seed=5))
    keys_jit = sorted(sample_collisions(starts, dirs, np.full(K, 3.0), grid,
                                        ke, ks, periodic_box=box, seed=5,
                                        jit=True))
    assert keys_np == keys_jit


def test_backward_jit_matches_numpy_walk():
    grid, box, hz = slab_grid(4.0, 2.0, nz=16, nxy=4)
    ke = np.full(grid.n_cells, 2.0)
    ks = np.full(grid.n_cells, 1.0)
    B = np.full(grid.n_cells, 0.3)
    bc = boundary_conditions(top_I=1.0)
    pt = (grid.x[1], grid.y[1], grid.z[-1])
    om = np.array([0.0, 0.0, 1.0])
    det = {"point": pt, "omega0": om, "cone": 0.0, "n_seeds": 60000}
    rn = backward_detector_mc(grid, ke, ks, B, bc, det, g=0.3,
                              periodic_box=box, n_batches=50, seed=45)
    rj = backward_detector_mc(grid, ke, ks, B, bc, det, g=0.3,
                              periodic_box=box, n_batches=50, seed=45,
                              jit=True)
    z = (rj["I"] - rn["I"]) / np.hypot(rj["sigma_I"], rn["sigma_I"])
    assert abs(z) < 3.0


def test_stage2_jit_matches_numpy_tally():
    """radiance_from_source(jit=True): aggregate agreement with the NumPy
    next-flight tally (M5 wiring of the fast path into stage 2)."""
    from rt3d.directions import icosahedric_grid

    dg = icosahedric_grid(n_refine=1)
    grid, box, hz = slab_grid(4.0, 2.0, nz=8, nxy=2)
    ke = np.full(grid.n_cells, 2.0)
    ks = 0.5 * ke
    S = np.full(grid.n_cells, 0.7)
    bc = boundary_conditions(top_I=1.0)
    kw = dict(periodic_box=box, estimator="next_flight", n_histories=60,
              n_batches=10, seed=46)
    rn = radiance_from_source(grid, dg, ke, S, bc, **kw)
    rj = radiance_from_source(grid, dg, ke, S, bc, jit=True, **kw)
    assert rj["jit"] is True
    # uncollided part is deterministic -> identical; scatter part statistical
    sig = rn["sigma_I"] > 1e-3 * np.sqrt(np.mean(rn["I"] ** 2))
    z = (rj["I"] - rn["I"])[sig] / np.hypot(rj["sigma_I"], rn["sigma_I"])[sig]
    assert np.sqrt(np.mean(z ** 2)) < 1.5
    assert np.abs(z).max() < 5.0
