# -*- coding: utf-8 -*-
"""
MC phase M3 invariants (plan/mc-plan.md section 7): backward (adjoint) detector
MC — radiance at a detector from walks launched backwards from it.

  1. pure absorber + thermal B + bottom inflow, exact-direction point
     detector on the slab: matches the closed-form Beer-Lambert integral
     within 3 sigma
  2. scattering slab (closed problem, no J input): adjoint walk reproduces
     the converged SC I at the detector cell for upward ordinates within
     combined 2 sigma + an 8% angular/quadrature allowance (42-ordinate
     quadrature vs angle-exact MC)
  3. beam coupling: adjoint score carries the continuous-HG first-scatter
     term; agreement vs SC-with-beam within the same band
  4. all-sky node-direction mode (cone = 0, the default) vs SC I columns:
     every upward ordinate inside 2 sigma + 8% rms-sky band, and the
     hemispheric upward-flux closure agrees within 3%; the full-cap mode
     (cone_scale = 1) is a full-sphere aperture whose near-horizon caps
     cross the horizon and mix in downward radiance (documented)
  5. global majorant mode equivalent to cell mode (with real nulls on a
     varying-kappa slab)
  6. octree walk runs and agrees with the SC octree field
  7. guards (ground BC, detector spec) + traceability keys
  8. (M5) closed-form single-scatter invariant at omega -> 0 on the octree
     slab: the exact-HG walk against the deterministic integral of the
     first-scatter series — the walk's boundary-incidence path verified
     independent of any SC quadrature (SC-42 itself is off by ~15% on
     this observable; see decision log MC adjoint pixel-level trust)
"""

import numpy as np
import pytest

from rt3d.cases.slab_1d import slab_grid
from rt3d.directions import icosahedric_grid
from rt3d.geometry import ground_surface
from rt3d.mc.backward import (
    all_sky_cone_half,
    backward_all_sky,
    backward_detector_mc,
)
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions

DG = icosahedric_grid(n_refine=1)


def _slab(tau=4.0, kappa_e=2.0, nz=16, nxy=4):
    grid, box, hz = slab_grid(tau, kappa_e, nz, nxy)
    return grid, box, hz


def _top_center(grid):
    """(flat index, point) of the top-layer cell in column (1, 1)"""
    pt = (grid.x[1], grid.y[1], grid.z[-1])
    cidx = ((grid.nz - 1) * grid.ny + 1) * grid.nx + 1
    return cidx, pt


# --------------------------------------------------------------------------
# 1: pure absorber, closed form
# --------------------------------------------------------------------------


def test_backward_pure_absorber_slab():
    grid, box, hz = _slab()
    ke = np.full(grid.n_cells, 2.0)
    ks = np.zeros(grid.n_cells)
    B = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(bottom_I=5.0)
    _, pt = _top_center(grid)
    om = np.array([0.15, -0.25, 0.9576])
    om = om / np.linalg.norm(om)
    tau = 2.0 * pt[2] / om[2]
    ana = 5.0 * np.exp(-tau) + (1.0 - np.exp(-tau))

    det = {"point": pt, "omega0": om, "cone": 0.0, "n_seeds": 40000}
    r = backward_detector_mc(grid, ke, ks, B, bc, det, periodic_box=box,
                             n_batches=40, seed=11)
    z = (r["I"] - ana) / r["sigma_I"]
    assert abs(z) < 3.0
    assert r["n_absorbed"] + r["n_exited"] == r["n_histories"]


# --------------------------------------------------------------------------
# 2 + 3: closed scattering problem (+ beam) vs SC
# --------------------------------------------------------------------------


def test_backward_scattering_vs_sc():
    grid, box, hz = _slab(tau=4.0)
    ke = np.full(grid.n_cells, 2.0)
    ks = np.full(grid.n_cells, 1.0)          # omega = 0.5
    B = np.full(grid.n_cells, 0.3)
    bc = boundary_conditions(top_I=1.0)
    g = 0.3
    sc = lambda_iteration(grid, DG, ke, ks, B, bc, g=g, periodic_box=box,
                          n_iter=800, tol=1e-10)
    cidx, pt = _top_center(grid)
    up = np.flatnonzero(DG.directions[:, 2] > 0.0)
    devs = []
    for d in up[:8]:
        det = {"point": pt, "omega0": DG.directions[d], "cone": 0.0,
               "n_seeds": 20000}
        r = backward_detector_mc(grid, ke, ks, B, bc, det, g=g,
                                 periodic_box=box, n_batches=20, seed=12)
        Iref = sc["I"][cidx, d]
        band = 2.0 * r["sigma_I"] + 0.08 * abs(Iref)
        devs.append(abs(r["I"] - Iref) / band)
    devs = np.asarray(devs)
    assert np.all(devs < 1.0)
    assert devs.max() < 0.8


def test_backward_beam_vs_sc():
    grid, box, hz = _slab(tau=4.0)
    ke = np.full(grid.n_cells, 2.0)
    ks = np.full(grid.n_cells, 1.0)
    B = np.full(grid.n_cells, 0.3)
    bc = boundary_conditions(top_I=1.0)
    sun = np.array([0.2, 0.1, -0.974679])
    sun = sun / np.linalg.norm(sun)
    beam = {"omega_sun": sun, "F0": 3.0}
    sc = lambda_iteration(grid, DG, ke, ks, B, bc, g=0.0, periodic_box=box,
                          beam=beam, n_iter=800, tol=1e-10)
    cidx, pt = _top_center(grid)
    up = np.flatnonzero(DG.directions[:, 2] > 0.0)
    devs = []
    for d in up[:6]:
        det = {"point": pt, "omega0": DG.directions[d], "cone": 0.0,
               "n_seeds": 30000}
        r = backward_detector_mc(grid, ke, ks, B, bc, det, g=0.0, beam=beam,
                                 periodic_box=box, n_batches=30, seed=13)
        Iref = sc["I"][cidx, d]
        band = 2.0 * r["sigma_I"] + 0.08 * abs(Iref)
        devs.append(abs(r["I"] - Iref) / band)
    assert np.all(np.asarray(devs) < 1.0)


# --------------------------------------------------------------------------
# 4: all-sky cone mode
# --------------------------------------------------------------------------


def test_backward_all_sky_vs_sc():
    grid, box, hz = _slab(tau=4.0)
    ke = np.full(grid.n_cells, 2.0)
    ks = np.full(grid.n_cells, 1.0)
    B = np.full(grid.n_cells, 0.3)
    bc = boundary_conditions(top_I=1.0)
    g = 0.3
    sc = lambda_iteration(grid, DG, ke, ks, B, bc, g=g, periodic_box=box,
                          n_iter=800, tol=1e-10)
    cidx, pt = _top_center(grid)
    rs = backward_all_sky(grid, ke, ks, B, bc, DG, pt, g=g, periodic_box=box,
                          n_seeds=10000, n_batches=20, seed=14)
    up = np.flatnonzero(DG.directions[:, 2] > 0.0)
    assert rs["up_idx"].tolist() == up.tolist()
    # default mode: exact node directions (cone_scale = 0) — directly
    # differencable against the SC I columns
    assert all(r["n_iter"] >= 0 for r in rs["runs"])

    Iref = sc["I"][cidx, up]
    band = 2.0 * rs["sigma_I"] + 0.08 * np.sqrt(np.mean(Iref ** 2))
    inside = np.abs(rs["I"] - Iref) < band
    assert inside.all()
    assert np.median(np.abs(rs["I"] - Iref) / band) < 0.5

    # hemisphere-averaged upward flux closes against the SC quadrature
    mu = DG.directions[up, 2]
    F_adj = float(DG.weights[up] @ (mu * rs["I"]))
    F_sc = float(DG.weights[up] @ (mu * Iref))
    assert abs(F_adj - F_sc) / F_sc < 0.03


def test_backward_all_sky_cone_mode_horizon():
    # full icosahedral caps (cone_scale = 1) are FULL-SPHERE apertures:
    # the near-horizon nodes' caps cross the horizon and pick up the
    # top-lit downward radiance, so their cone averages say nothing about
    # the upward node value — the reason node-direction mode is the
    # differencing default of `backward_all_sky`
    grid, box, hz = _slab(tau=4.0)
    ke = np.full(grid.n_cells, 2.0)
    ks = np.zeros(grid.n_cells)         # pure absorber, no local emission
    B = np.zeros(grid.n_cells)
    bc = boundary_conditions(top_I=1.0)
    _, pt = _top_center(grid)

    # equal-area caps tile the hemisphere
    up = np.flatnonzero(DG.directions[:, 2] > 0.0)
    caps = 2 * np.pi * (1 - np.cos([all_sky_cone_half(DG, i) for i in up]))
    assert np.allclose(caps.sum(), 2 * np.pi)

    # near-horizon node: upward radiance is extinction-depleted (long
    # oblique path to the bottom), the cap mixes in the top-lit downward
    # half and is O(1)
    k_low = int(np.argmin(DG.directions[up, 2]))
    rs = backward_all_sky(grid, ke, ks, B, bc, DG, pt,
                          periodic_box=box, n_seeds=5000, n_batches=10,
                          seed=18, cone_scale=1.0)
    I_flat = backward_detector_mc(
        grid, ke, ks, B, bc,
        {"point": pt, "omega0": DG.directions[up[k_low]], "cone": 0.0,
         "n_seeds": 5000}, periodic_box=box, n_batches=10, seed=18,
    )
    assert I_flat["I"] < 0.02
    assert rs["I"][k_low] > 0.03
    assert rs["I"][k_low] > 2.0 * I_flat["I"]


# --------------------------------------------------------------------------
# 5: global majorant mode with real nulls
# --------------------------------------------------------------------------


def test_backward_global_mode_nulls():
    grid, box, hz = _slab(tau=4.0)
    z = grid.centers[:, 2]
    ke = 2.0 + 1.5 * ((z > 1.6) & (z < 2.4))
    ks = 0.5 * ke
    B = np.full(grid.n_cells, 0.5)
    bc = boundary_conditions(top_I=1.0)
    _, pt = _top_center(grid)
    om = DG.directions[np.argmax(DG.directions[:, 2])]
    det = {"point": pt, "omega0": om, "cone": 0.0, "n_seeds": 30000}
    rc = backward_detector_mc(grid, ke, ks, B, bc, det, g=0.2,
                              periodic_box=box, n_batches=30, seed=15,
                              mode="cell")
    rg = backward_detector_mc(grid, ke, ks, B, bc, det, g=0.2,
                              periodic_box=box, n_batches=30, seed=15,
                              mode="global")
    assert rc["n_null"] == 0
    assert rg["n_null"] > 1000
    zgg = (rg["I"] - rc["I"]) / np.hypot(rg["sigma_I"], rc["sigma_I"])
    assert abs(zgg) < 4.0


# --------------------------------------------------------------------------
# 6: octree walk vs SC octree field
# --------------------------------------------------------------------------


def test_backward_octree_vs_sc():
    grid, box, hz = _slab(tau=4.0)

    def kc(x, y, z):
        return 2.0 + 1.5 * ((z > box.z0 + 0.4 * box.Lz)
                            & (z < box.z0 + 0.6 * box.Lz))

    oct_ = octree_grid(box, h0=box.Lx / 2)
    oct_.refine(kc, max_depth=3)
    oct_.balance_21(max_depth=3)
    oct_.finalize()
    ko = kc(oct_.centers[:, 0], oct_.centers[:, 1], oct_.centers[:, 2])
    Bo = np.full(oct_.n_leaf, 0.7)
    bc = boundary_conditions(top_I=3.0)
    sc = lambda_iteration(oct_, DG, ko, np.full(oct_.n_leaf, 0.5), Bo, bc,
                          g=0.2, periodic_box=box, n_iter=60, tol=1e-9,
                          ng=None, jit=False)
    # detector at the leaf center nearest the axis in the top z layer
    top_z = oct_.centers[:, 2].max()
    cand = np.flatnonzero(np.abs(oct_.centers[:, 2] - top_z) < 1e-9)
    c = cand[np.argmin(np.hypot(oct_.centers[cand, 0],
                                oct_.centers[cand, 1]))]
    pt = tuple(oct_.centers[c])
    devs = []
    for d in np.flatnonzero(DG.directions[:, 2] > 0.0)[:6]:
        det = {"point": pt, "omega0": DG.directions[d], "cone": 0.0,
               "n_seeds": 20000}
        r = backward_detector_mc(oct_, ko, np.full(oct_.n_leaf, 0.5), Bo, bc,
                                 det, g=0.2, periodic_box=box, n_batches=20,
                                 seed=16)
        Iref = sc["I"][c, d]
        band = 2.0 * r["sigma_I"] + 0.08 * abs(Iref)
        devs.append(abs(r["I"] - Iref) / band)
    assert np.all(np.asarray(devs) < 1.2)


# --------------------------------------------------------------------------
# 7: guards + traceability
# --------------------------------------------------------------------------


def test_backward_guards_and_traceability():
    grid, box, hz = _slab(nz=8, nxy=2)
    ke = np.full(grid.n_cells, 1.0)
    ks = 0.5 * ke
    B = np.full(grid.n_cells, 0.2)
    bc = boundary_conditions(top_I=1.0)
    _, pt = _top_center(grid)
    det = {"point": pt, "omega0": DG.directions[0], "cone": 0.1,
           "n_seeds": 500}
    r = backward_detector_mc(grid, ke, ks, B, bc, det, periodic_box=box,
                             n_batches=10, seed=17)
    for key, want in (("seed", 17), ("estimator", "adjoint_walk"),
                      ("mode", "cell"), ("n_histories", 500),
                      ("n_batches", 10)):
        assert r[key] == want
    assert r["n_exited"] + r["n_absorbed"] + r["n_capped"] == 500

    bc_g = boundary_conditions(
        top_I=1.0, ground=ground_surface(albedo=0.5,
                                         height=lambda x, y: 0.0 * x))
    # M6: flat ground is supported; the guard is the variable surface
    with pytest.raises(NotImplementedError):
        backward_detector_mc(grid, ke, ks, B, bc_g, det, periodic_box=box)

    with pytest.raises(ValueError, match="aperture"):
        backward_detector_mc(grid, ke, ks, B, bc,
                             {"omega0": DG.directions[0], "n_seeds": 10},
                             periodic_box=box)


# --------------------------------------------------------------------------
# 8: (M5) closed-form single-scatter invariant, omega -> 0, exact HG
# --------------------------------------------------------------------------


def test_backward_single_scatter_closed_form_octree():
    """omega = 0.01 slab, vertical detector: first-order scattering integral
    evaluated deterministically vs the adjoint walk.  Invariants the walk's
    boundary-incidence path against ANALYSIS (no SC quadrature involved);
    the same observable carries a ~15% SC-42 quadrature bias (M5 report)."""
    from rt3d.directions import hg
    from rt3d.geometry import cuboid
    from rt3d.octree import octree_grid

    g = 0.3
    ke = 2.0
    box = cuboid(0.0, 4.0, 0.0, 2.0, 0.0, 2.0)
    oct_ = octree_grid(box, h0=0.5)
    oct_.refine(lambda x, y, z: np.full_like(x, ke), max_depth=1)
    oct_.balance_21(max_depth=1)
    oct_.finalize()
    top_z = oct_.centers[:, 2].max()
    cand = np.flatnonzero(np.abs(oct_.centers[:, 2] - top_z) < 1e-9)
    c = cand[np.argmin(np.hypot(oct_.centers[cand, 0], oct_.centers[cand, 1]))]
    pt = tuple(oct_.centers[c])

    # I(P, +z) = int_0^{z_P} kappa_s e^{-ke s} A(z_P - s) ds,
    # A(z) = 2 pi int_0^1 Phi(-mu) e^{-ke (z1 - z)/mu} dmu  (incidence from
    # the top face along -mu in +z-scattered direction; cos gamma = -mu)
    zs = np.linspace(0.0, pt[2], 4001)
    mu = np.linspace(1e-6, 1.0, 2001)
    A = 2 * np.pi * np.trapezoid(
        hg(-mu, g)[None, :] * np.exp(-ke * (box.z1 - (pt[2] - zs))[:, None]
                                    / mu[None, :]), mu, axis=1)
    I_ana = 0.01 * ke * np.trapezoid(np.exp(-ke * zs) * A, zs)

    r = backward_detector_mc(oct_, np.full(oct_.n_leaf, ke),
                             np.full(oct_.n_leaf, 0.01 * ke),
                             np.zeros(oct_.n_leaf),
                             boundary_conditions(top_I=1.0),
                             {"point": pt, "omega0": np.array([0.0, 0.0, 1.0]),
                              "cone": 0.0, "n_seeds": 400_000},
                             g=g, periodic_box=box, n_batches=50, seed=92)
    assert abs(r["I"] - I_ana) < 4.0 * r["sigma_I"]
    assert I_ana > 0.0
