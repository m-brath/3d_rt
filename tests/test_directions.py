# -*- coding: utf-8 -*-
"""Phase 0 anchors: direction grid, solid-angle weights, HG matrix."""

import numpy as np
import pytest

from rt3d.directions import (
    hg,
    icosahedric_grid,
    make_icosahedric_grid,
    refine_icosahedric_grid,
    solid_angle_weights,
)


@pytest.mark.parametrize("n_refine", [0, 1, 2, 3])
def test_grid_size(n_refine):
    dg = icosahedric_grid(n_refine=n_refine)
    assert dg.n_dir == 10 * 4**n_refine + 2
    assert dg.directions.shape == (dg.n_dir, 3)
    assert dg.weights.shape == (dg.n_dir,)


@pytest.mark.parametrize("n_refine", [0, 1, 2, 3])
def test_weights_sum_4pi(n_refine):
    dg = icosahedric_grid(n_refine=n_refine)
    assert np.isclose(dg.weights.sum(), 4 * np.pi, rtol=1e-12)
    assert np.all(dg.weights > 0)


@pytest.mark.parametrize("n_refine", [1, 2, 3])
@pytest.mark.parametrize("g", [0.0, 0.3, 0.5])
def test_hg_normalized(n_refine, g):
    """int Phi dOmega == 1 within quadrature error of the icosahedral grid."""
    dg = icosahedric_grid(n_refine=n_refine)
    err = np.max(np.abs(dg.hg_matrix(g).sum(axis=1) - 1.0))
    assert err < 5e-3


def test_hg_isotropic_exact():
    dg = icosahedric_grid(n_refine=1)
    W = dg.hg_matrix(0.0)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-14)


@pytest.mark.parametrize("n_refine", [0, 1, 2, 3])
def test_no_ordinate_on_equator(n_refine):
    dg = icosahedric_grid(n_refine=n_refine)
    assert np.min(np.abs(dg.directions[:, 2])) > 1e-6


def test_directions_unit_norm():
    dg = icosahedric_grid(n_refine=2)
    assert np.allclose(np.linalg.norm(dg.directions, axis=1), 1.0)


def test_refine_matches_reference_base():
    """Base grid identical to scripts/icosphere_grid.py construction."""
    v, t = make_icosahedric_grid()
    assert v.shape == (12, 3)
    assert t.shape == (20, 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    w = solid_angle_weights(v, t)
    assert np.allclose(w, np.full(12, 4 * np.pi / 12), atol=1e-14)


def test_vectorized_refine_matches_reference():
    """rt3d refinement must produce the same vertex set as the reference
    implementation in scripts/icosphere_grid.py (n_refine=1)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "icosphere_grid",
        Path(__file__).resolve().parent.parent / "scripts" / "icosphere_grid.py",
    )
    ig = importlib.util.module_from_spec(spec)
    # the read-only reference script predates NumPy 2.0; alias the removed
    # np.NaN while it is imported and executed
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    try:
        spec.loader.exec_module(ig)
        v0, t0 = ig.make_icosahedric_grid()
        v_ref, _ = ig.refine_icosahedric_grid(v0, t0, 1)
    finally:
        del np.NaN
    v_rt, _ = refine_icosahedric_grid(v0, t0, 1)

    assert v_rt.shape == v_ref.shape
    # compare as sorted point sets
    a = np.lexsort((v_ref[:, 2], v_ref[:, 1], v_ref[:, 0]))
    b = np.lexsort((v_rt[:, 2], v_rt[:, 1], v_rt[:, 0]))
    assert np.allclose(v_ref[a], v_rt[b], atol=1e-12)


def test_hg_function_normalization_analytic():
    """HG integrates to 1 analytically: check via dense quadrature."""
    mu = np.linspace(-1, 1, 200001)
    for g in (-0.5, 0.0, 0.5, 0.9):
        integral = np.trapezoid(2 * np.pi * hg(mu, g), mu)
        # trapezoid error is dominated by the forward peak at g -> 1
        assert np.isclose(integral, 1.0, rtol=1e-5)


def test_angular_convergence_slab():
    """Phase 6 anchor: on a scattering slab the J difference between
    consecutive direction levels (42->162->642) must shrink."""
    from rt3d.geometry import cuboid, unigrid
    from rt3d.solver import lambda_iteration
    from rt3d.transport import boundary_conditions

    x = (np.arange(4) + 0.5) * 0.5
    y = (np.arange(4) + 0.5) * 0.5
    z = (np.arange(12) + 0.5) * 2.0 / 12
    grid = unigrid(x, y, z)
    box = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    kappa_e = np.full(grid.n_cells, 1.0)
    kappa_s = 0.5 * kappa_e

    profiles = []
    for n in (1, 2, 3):
        dg = icosahedric_grid(n_refine=n)
        bc = boundary_conditions(top_I=1.0)
        res = lambda_iteration(grid, dg, kappa_e, kappa_s,
                               np.zeros(grid.n_cells), bc, g=0.6,
                               periodic_box=box, n_iter=200, tol=1e-10)
        J = res["J"].reshape(grid.nz, grid.ny, grid.nx).mean(axis=(1, 2))
        profiles.append(J)

    scale = profiles[-1].max()
    d_42_162 = np.max(np.abs(profiles[1] - profiles[0])) / scale
    d_162_642 = np.max(np.abs(profiles[2] - profiles[1])) / scale
    assert d_162_642 < d_42_162
