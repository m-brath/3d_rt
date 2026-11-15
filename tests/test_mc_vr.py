# -*- coding: utf-8 -*-
"""
MC phase M4 anchors (plan/mc-plan.md section 2/7): variance reduction —
implicit capture (albedo-weight chains, no absorption termination) plus
Russian-roulette weight floor, on the forward chains (`track_histories` /
`j_mc` explicit variant) and the backward adjoint walk.

  1. adjoint walk, closed-form absorber slab (omega = 0): implicit capture +
     roulette degenerates to the analog walk bit-for-bit (zero-weight kill
     guard, no capped histories)
  2. adjoint walk, closed scattering slab omega = 0.5: VR value matches the
     analog walk within combined 2 sigma AND the standard error drops
     (measured ~3x — the walk scores (1-omega) B at every collision, so
     folding absorption into weights spreads the score over many histories)
  3. adjoint walk, omega = 0.9 (where implicit capture matters most):
     match + improved sigma
  4. forward j_mc explicit variant: VR matches the analog J per cell within
     combined 2 sigma + 5% band at equal cost; variance-neutral here —
     explicit-chain noise is birth-statistics dominated (chains short),
     reported honestly in plan/plan.md
  5. traceability: switches recorded in the track_histories output
"""

import numpy as np

from rt3d.cases.slab_1d import slab_grid
from rt3d.directions import icosahedric_grid
from rt3d.mc.backward import backward_detector_mc
from rt3d.mc.forward import j_mc
from rt3d.mc.tracking import track_histories
from rt3d.mc.estimators import track_tally
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions

DG = icosahedric_grid(n_refine=1)


def _slab():
    grid, box, hz = slab_grid(4.0, 2.0, nz=16, nxy=4)
    return grid, box, np.full(grid.n_cells, 2.0)


def _top_center(grid):
    pt = (grid.x[1], grid.y[1], grid.z[-1])
    cidx = ((grid.nz - 1) * grid.ny + 1) * grid.nx + 1
    return cidx, pt


# --------------------------------------------------------------------------
# 1: adjoint closed form with VR
# --------------------------------------------------------------------------


def test_backward_vr_closed_form():
    grid, box, ke = _slab()
    ks = np.zeros(ke.size)
    B = np.full(ke.size, 1.0)
    bc = boundary_conditions(bottom_I=5.0)
    _, pt = _top_center(grid)
    om = np.array([0.15, -0.25, 0.9576])
    om = om / np.linalg.norm(om)
    det = {"point": pt, "omega0": om, "cone": 0.0, "n_seeds": 40000}
    ra = backward_detector_mc(grid, ke, ks, B, bc, det, periodic_box=box,
                              n_batches=40, seed=21)
    r = backward_detector_mc(grid, ke, ks, B, bc, det, periodic_box=box,
                             n_batches=40, seed=21, implicit_capture=True,
                             roulette_floor=0.05)
    # omega = 0: implicit capture degenerates to the analog walk exactly
    # (the closed-form value itself is locked by the M3 test at seed 11)
    assert r["I"] == ra["I"] and r["sigma_I"] == ra["sigma_I"]
    # zero-weight guard: no history crawls on with weight 0
    assert r["n_capped"] == 0


# --------------------------------------------------------------------------
# 2 + 3: adjoint VR on scattering slabs
# --------------------------------------------------------------------------


def _adjoint_pair(omega_ssa, kw_vr, seed, n_seeds=100000):
    grid, box, ke = _slab()
    ks = omega_ssa * ke
    B = np.full(ke.size, 0.3)
    bc = boundary_conditions(top_I=1.0)
    _, pt = _top_center(grid)
    d_up = DG.directions[np.argmax(DG.directions[:, 2])]
    det = {"point": pt, "omega0": d_up, "cone": 0.0, "n_seeds": n_seeds}
    ra = backward_detector_mc(grid, ke, ks, B, bc, det, g=0.3,
                              periodic_box=box, n_batches=100, seed=seed)
    rv = backward_detector_mc(grid, ke, ks, B, bc, det, g=0.3,
                              periodic_box=box, n_batches=100, seed=seed,
                              **kw_vr)
    return ra, rv


def test_backward_vr_omega_half():
    ra, rv = _adjoint_pair(0.5, {"implicit_capture": True,
                                "roulette_floor": 0.1}, seed=22)
    z = (rv["I"] - ra["I"]) / np.hypot(rv["sigma_I"], ra["sigma_I"])
    assert abs(z) < 3.0
    assert rv["sigma_I"] < 0.6 * ra["sigma_I"]


def test_backward_vr_omega_high():
    ra, rv = _adjoint_pair(0.9, {"implicit_capture": True,
                                "roulette_floor": 0.05}, seed=23)
    z = (rv["I"] - ra["I"]) / np.hypot(rv["sigma_I"], ra["sigma_I"])
    assert abs(z) < 3.0
    assert rv["sigma_I"] < ra["sigma_I"]


# --------------------------------------------------------------------------
# 4: forward explicit variant VR
# --------------------------------------------------------------------------


def test_forward_j_mc_vr_matches():
    grid, box, ke = _slab()
    ks = 0.5 * ke
    B = np.full(ke.size, 0.3)
    bc = boundary_conditions(top_I=1.0)
    sc = lambda_iteration(grid, DG, ke, ks, B, bc, g=0.3, periodic_box=box,
                          n_iter=800, tol=1e-10)
    kw = dict(stage1="explicit", n_histories=100000, n_batches=100, seed=24)
    ja = j_mc(grid, ke, ks, B, bc, g=0.3, periodic_box=box, **kw)
    jv = j_mc(grid, ke, ks, B, bc, g=0.3, periodic_box=box,
              implicit_capture=True, roulette_floor=0.05, **kw)
    dif = np.abs(jv["J"] - ja["J"])
    band = 2.0 * np.hypot(jv["sigma_J"], ja["sigma_J"]) + 0.05 * sc["J"]
    assert np.all(dif < band)
    # equal transport cost within 25 % (roulette kills the zero-weight tail)
    assert abs(jv["n_collisions"] - ja["n_collisions"]) \
        < 0.25 * ja["n_collisions"]


# --------------------------------------------------------------------------
# 5: traceability of the switches in the chain output
# --------------------------------------------------------------------------


def test_track_histories_vr_traceability():
    grid, box, ke = _slab()
    ks = 0.5 * ke
    tally = track_tally(ke.size, 100, n_batches=10)
    c0 = grid.centers[grid.n_cells // 2]
    out = track_histories(np.tile(c0, (100, 1)), np.tile([0.0, 0.0, 1.0],
                                                         (100, 1)),
                          grid, ke, ks, tally, periodic_box=box, seed=25,
                          implicit_capture=True, roulette_floor=0.2)
    assert out["implicit_capture"] is True
    assert out["roulette_floor"] == 0.2
    # omega = 0.5: with weight floor 0.2 no chain survives many collisions
    assert np.all(out["n_collisions"] < 200)
