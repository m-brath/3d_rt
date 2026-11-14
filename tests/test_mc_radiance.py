# -*- coding: utf-8 -*-
"""
MC phase M2 anchors (plan/mc-plan.md section 7): stage-2 radiance
``I(cell, ordinate)`` of a fixed source, on the SC ray set (one ray per cell
center per icosahedral ordinate, marched backwards to the domain entry).

  1. deterministic estimator == the analytic line integral on the uniform
     slab (closed-form Beer-Lambert + homogeneous source integral)
  2. deterministic estimator on the octree == converged SC ``I`` bit for bit
     when fed the SC's own source (same exact marches, same recursion)
  3. next-flight estimator vs deterministic: aggregate z-scores within 2
     sigma (mean ~ 0, rms ~ 1), on unigrid and octree, cell and global
     majorant mode
  4. sigma_I ~ 1/sqrt(n_histories)
  5. stage-2 guards + traceability keys, and the stage2_source bookkeeping
     (scalar diffuse form; direction-resolved form with beam)
"""

import numpy as np
import pytest

from rt3d.cases.slab_1d import slab_grid
from rt3d.directions import icosahedric_grid
from rt3d.geometry import ground_surface
from rt3d.mc.forward import radiance_from_source, stage2_source
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions

DG = icosahedric_grid(n_refine=1)


def _aggregate_z(diff, sigma, floor=1e-3):
    """z-scores of resolved cells only (binary-tally sigma collapses to 0 in
    optically thick cells, where the 2-sigma test is vacuous)."""
    scale = 1.0 if sigma.max() == 0.0 else float(np.sqrt(np.mean(diff ** 2)))
    mask = sigma > floor * scale
    z = diff[mask] / sigma[mask]
    return z


# --------------------------------------------------------------------------
# 1: unigrid slab, uniform kappa — deterministic == analytic, next-flight
#    statistically identical
# --------------------------------------------------------------------------


def _analytic_slab(grid, ke, S, bc):
    """I = I_in e^-tau + S (1 - e^-tau) in the uniform medium, per ordinate."""
    mu = np.abs(DG.directions[:, 2])
    z = grid.centers[:, 2]
    z1 = grid.box[2][1]
    up = DG.directions[:, 2] > 0.0
    tau = np.where(up, z[:, None], z1 - z[:, None]) / mu[None, :] * ke[:, None]
    I_in = np.where(up, bc.bottom_I, bc.top_I)
    return I_in[None, :] * np.exp(-tau) + S[:, None] * (1.0 - np.exp(-tau))


def test_stage2_unigrid_slab():
    grid, box, hz = slab_grid(tau=4.0, kappa_e=2.0, nz=16, nxy=4)
    ke = np.full(grid.n_cells, 2.0)
    S = np.full(grid.n_cells, 1.0)
    bc = boundary_conditions(top_I=3.0)
    ana = _analytic_slab(grid, ke, S, bc)

    det = radiance_from_source(grid, DG, ke, S, bc, periodic_box=box,
                               estimator="deterministic")
    assert np.max(np.abs(det["I"] - ana)) < 1e-10
    assert np.all(det["sigma_I"] == 0.0)

    mc = radiance_from_source(grid, DG, ke, S, bc, periodic_box=box,
                              estimator="next_flight", n_histories=200,
                              n_batches=20, seed=31)
    z = _aggregate_z(mc["I"] - ana, mc["sigma_I"])
    assert abs(z.mean()) < 0.15
    assert np.sqrt(np.mean(z ** 2)) < 1.3
    assert np.count_nonzero(np.abs(z) <= 2.0) >= 0.92 * z.size


# --------------------------------------------------------------------------
# 2 + 4: octree slab — deterministic == SC I exactly; sigma scaling
# --------------------------------------------------------------------------


def _octree_cloud_slab():
    """refined octree slab with an embedded high-kappa layer (M2 test grid);
    per-cell g so the SC source is **direction-resolved** and must be fed to
    stage 2 per ordinate (collapsing it to one column would silently use the
    wrong source — the g = 0 columns are identical, the g != 0 ones are not)"""
    grid, box, hz = slab_grid(tau=4.0, kappa_e=2.0, nz=16, nxy=4)

    def kc(x, y, z):
        return 2.0 + 1.5 * ((z > box.z0 + 0.4 * box.Lz)
                            & (z < box.z0 + 0.6 * box.Lz))

    oct_ = octree_grid(box, h0=box.Lx / 2)
    oct_.refine(kc, max_depth=4)
    oct_.balance_21(max_depth=4)
    oct_.finalize()
    ko = kc(oct_.centers[:, 0], oct_.centers[:, 1], oct_.centers[:, 2])
    go = np.where(oct_.centers[:, 2] > 0.5 * (box.z0 + box.z1), 0.4, 0.1)
    return box, oct_, ko, go


def test_stage2_octree_equals_sc():
    box, oct_, ko, go = _octree_cloud_slab()
    Bo = np.full(oct_.n_leaf, 0.7)
    bc = boundary_conditions(top_I=3.0)
    sc = lambda_iteration(oct_, DG, ko, np.zeros(oct_.n_leaf), Bo, bc,
                          g=go, periodic_box=box, n_iter=40, tol=1e-10,
                          ng=None, jit=False)
    S = np.asarray(sc["S"])
    assert S.shape == (oct_.n_leaf, DG.n_dir)   # direction-resolved source
    det = radiance_from_source(oct_, DG, ko, S, bc, periodic_box=box,
                               estimator="deterministic")
    # identical recursion on identical marches, per-ordinate source
    assert np.allclose(det["I"], sc["I"], rtol=0.0, atol=1e-12)
    assert np.allclose(det["J"], sc["J"], rtol=0.0, atol=1e-12)


def test_stage2_octree_next_flight_and_scaling():
    box, oct_, ko, go = _octree_cloud_slab()
    Bo = np.full(oct_.n_leaf, 0.7)
    bc = boundary_conditions(top_I=3.0)
    sc = lambda_iteration(oct_, DG, ko, np.zeros(oct_.n_leaf), Bo, bc,
                          g=go, periodic_box=box, n_iter=40, tol=1e-10,
                          ng=None, jit=False)
    S = np.asarray(sc["S"])
    det = radiance_from_source(oct_, DG, ko, S, bc, periodic_box=box,
                               estimator="deterministic")

    mc1 = radiance_from_source(oct_, DG, ko, S, bc, periodic_box=box,
                               estimator="next_flight", n_histories=80,
                               n_batches=10, seed=41, mode="cell")
    z = _aggregate_z(mc1["I"] - det["I"], mc1["sigma_I"])
    assert abs(z.mean()) < 0.15
    assert np.sqrt(np.mean(z ** 2)) < 1.3
    assert np.count_nonzero(np.abs(z) <= 2.0) >= 0.92 * z.size

    mc4 = radiance_from_source(oct_, DG, ko, S, bc, periodic_box=box,
                               estimator="next_flight", n_histories=320,
                               n_batches=10, seed=41, mode="cell")
    both = (mc1["sigma_I"] > 0) & (mc4["sigma_I"] > 0)
    ratio = np.median(mc1["sigma_I"][both] / mc4["sigma_I"][both])
    assert 1.7 < ratio < 2.4


def test_stage2_octree_global_mode():
    box, oct_, ko, go = _octree_cloud_slab()
    Bo = np.full(oct_.n_leaf, 0.7)
    bc = boundary_conditions(top_I=3.0)
    sc = lambda_iteration(oct_, DG, ko, np.zeros(oct_.n_leaf), Bo, bc,
                          g=go, periodic_box=box, n_iter=40, tol=1e-10,
                          ng=None, jit=False)
    S = np.asarray(sc["S"])
    mcg = radiance_from_source(oct_, DG, ko, S, bc, periodic_box=box,
                               estimator="next_flight", n_histories=80,
                               n_batches=10, seed=42, mode="global",
                               kappa_max=float(ko.max()))
    assert mcg["n_null"] > 0
    z = _aggregate_z(mcg["I"] - sc["I"], mcg["sigma_I"])
    assert abs(z.mean()) < 0.15
    assert np.sqrt(np.mean(z ** 2)) < 1.3


# --------------------------------------------------------------------------
# 5: stage2_source bookkeeping + guards + traceability
# --------------------------------------------------------------------------


def test_stage2_source_and_guards():
    grid, box, hz = slab_grid(tau=2.0, kappa_e=1.0, nz=8, nxy=2)
    ke = np.full(grid.n_cells, 1.0)
    ks = 0.5 * ke
    B = np.full(grid.n_cells, 2.0)
    J = np.full(grid.n_cells, 0.25)
    bc = boundary_conditions(top_I=1.0)

    S = stage2_source(grid, DG, ke, ks, B, J)
    assert S.shape == (grid.n_cells,)
    assert np.allclose(S, 0.5 * B + 0.5 * J)       # omega = 0.5

    sun = np.array([0.3, -0.2, -0.9])
    sun = sun / np.linalg.norm(sun)
    beam = {"omega_sun": sun, "F0": 1.5}
    Sd = stage2_source(grid, DG, ke, ks, B, J, g=0.4, beam=beam,
                       periodic_box=box)
    assert Sd.shape == (grid.n_cells, DG.n_dir)
    assert np.all(Sd >= 0.0)
    # angle average of the beam part = omega * F0 exp(-tau_b) / 4 pi
    from rt3d.transport import beam_optical_depth

    tau_b = beam_optical_depth(grid, sun, ke, periodic_box=box)
    om = ks / ke
    avg = Sd @ DG.weights / (4.0 * np.pi)
    assert np.allclose(avg, S + om * 1.5 * np.exp(-tau_b) / (4.0 * np.pi))

    with pytest.raises(ValueError, match="estimator"):
        radiance_from_source(grid, DG, ke, S, bc, periodic_box=box,
                             estimator="adjoint")

    bc_g = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=0.5))
    with pytest.raises(NotImplementedError):
        radiance_from_source(grid, DG, ke, S, bc_g, periodic_box=box)

    r = radiance_from_source(grid, DG, ke, S, bc, periodic_box=box,
                             estimator="next_flight", n_histories=40,
                             n_batches=8, seed=7)
    for key, want in (("seed", 7), ("estimator", "next_flight"),
                      ("mode", "cell"), ("n_batches", 8),
                      ("n_histories", 40), ("n_dir", DG.n_dir),
                      ("n_refine", DG.n_refine)):
        assert r[key] == want
