# -*- coding: utf-8 -*-
"""Phase 4: 1D (plane-averaged) Ng acceleration of the lambda iteration."""

import numpy as np
import pytest

from rt3d.accelerator import ng1d
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions


# ------------------------------------------------------------- 1D model problem


def make_unigrid(nx=2, ny=2, nz=12):
    x = (np.arange(nx) + 0.5) * 0.5
    y = (np.arange(ny) + 0.5) * 0.5
    z = (np.arange(nz) + 0.5) * 0.5
    return unigrid(x, y, z)


def test_ng_solves_1d_linear_iteration_exactly():
    """Synthetic iteration S <- L S + Q with a z-coupled dense L (isotropic
    in x, y, direction).  With the full window the plane-averaged Ng
    coefficient solve exhausts the Krylov space: convergence in ~nz steps,
    matching the direct solve (I - L)^-1 Q."""
    nz = 12
    grid = make_unigrid(nz=nz)
    n_dir = 3
    rng = np.random.default_rng(3)

    # tridiagonal coupling, spectral radius < 1
    L = np.diag(np.full(nz, 0.2)) \
        + np.diag(np.full(nz - 1, 0.4), 1) + np.diag(np.full(nz - 1, 0.4), -1)
    Q = rng.random(nz) + 0.5

    acc = ng1d(grid, max_hist=nz, n_start=3)
    S = np.zeros((grid.n_cells, n_dir))
    layer = np.repeat(np.arange(nz), grid.ny * grid.nx)
    history = []
    for _ in range(nz + 10):
        Sz = S[:, 0].reshape(nz, -1)[:, 0]  # per-layer value (isotropic)
        S_new = (L @ Sz + Q)[layer][:, None] * np.ones((1, n_dir))
        rel = np.max(np.abs(S_new - S))
        history.append(rel)
        S = acc.step(S, S_new)
        if rel < 1e-10:
            break

    S_exact = np.linalg.solve(np.eye(nz) - L, Q)
    assert history[-1] < 1e-10
    assert len(history) <= nz + 4
    assert np.allclose(S[:, 0].reshape(nz, -1)[:, 0], S_exact, atol=1e-9)


def test_ng_warmup_matches_plain_li():
    grid = make_unigrid(nz=6)
    acc = ng1d(grid, n_start=3)
    rng = np.random.default_rng(4)
    S = rng.random((grid.n_cells, 2))
    for _ in range(4):
        S_new = rng.random((grid.n_cells, 2))
        out = acc.step(S, S_new)
        if len(acc.px) <= 3:
            assert np.array_equal(out, S_new)
    assert len(acc.px) == 3


# ------------------------------------------------------------------- physics


SLAB_NZ = 50
SLAB_HZ = 0.1  # kappa = 1, tau = 5 over the slab (same as phase 2)


def slab_setup(omega):
    """Plane-parallel slab, kappa = 1, tau = 5, g = 0, B = 0, top I = 1."""
    hx = SLAB_HZ
    x = (np.arange(4) + 0.5) * hx
    y = (np.arange(4) + 0.5) * hx
    z = (np.arange(SLAB_NZ) + 0.5) * SLAB_HZ
    grid = unigrid(x, y, z)
    box = cuboid(0.0, 4 * hx, 0.0, 4 * hx, grid.box[2][0], grid.box[2][1])
    kappa_e = np.full(grid.n_cells, 1.0)
    kappa_s = np.full(grid.n_cells, omega)
    bc = boundary_conditions(top_I=1.0)
    return grid, box, kappa_e, kappa_s, bc


@pytest.mark.parametrize("omega", [0.9, 0.99])
def test_ng_slab_converges_faster_same_solution(omega):
    """omega -> 1: Ng needs far fewer iterations than plain LI and reaches
    the same converged J profile (validation of plan phase 4)."""
    dg = icosahedric_grid(n_refine=1)
    grid, box, kappa_e, kappa_s, bc = slab_setup(omega)

    res_li = lambda_iteration(
        grid, dg, kappa_e, kappa_s, np.zeros(grid.n_cells), bc,
        periodic_box=box, n_iter=2000, tol=1e-8,
    )
    res_ng = lambda_iteration(
        grid, dg, kappa_e, kappa_s, np.zeros(grid.n_cells), bc,
        periodic_box=box, n_iter=2000, tol=1e-8, ng=True,
    )
    assert res_li["converged"]
    assert res_ng["converged"]
    assert res_ng["n_iter"] < 0.5 * res_li["n_iter"]
    J_li = res_li["J"].reshape(SLAB_NZ, 4, 4).mean(axis=(1, 2))
    J_ng = res_ng["J"].reshape(SLAB_NZ, 4, 4).mean(axis=(1, 2))
    assert np.max(np.abs(J_li - J_ng)) / J_li.max() < 1e-5


# ------------------------------------------------------------------- octree


def test_ng_octree_binning_and_plane_average():
    box = cuboid(0.0, 4.0, 0.0, 4.0, 0.0, 4.0)
    oct = octree_grid(box, h0=1.0)
    oct.refine(lambda x, y, z: np.where(z > 2.0, 5.0, 0.1),
               tau_max=0.5, eps_grad=0.3, max_depth=2)
    oct.balance_21(max_depth=2)
    oct.finalize()

    acc = ng1d(oct)
    assert acc.n_bins == 4
    # constant field: plane average is that constant in every bin
    c = np.full(oct.n_leaf, 3.5)
    prof = acc.plane_average(c)
    assert np.allclose(prof, 3.5)
    # every leaf is binned
    assert np.all(acc.bin_idx >= 0) and np.all(acc.bin_idx < acc.n_bins)
