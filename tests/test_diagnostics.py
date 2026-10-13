# -*- coding: utf-8 -*-
"""Tests for rt3d.diagnostics.angular_variance."""

import numpy as np

from rt3d.diagnostics import angular_variance
from rt3d.directions import icosahedric_grid


def test_isotropic_field_zero():
    dg = icosahedric_grid(n_refine=1)
    J = np.linspace(0.3, 1.7, 5)
    I = np.broadcast_to(J[:, None], (5, dg.n_dir))
    var = angular_variance(I, dg)
    assert np.allclose(var, 0.0, atol=1e-24)


def test_beam_field_a2_over_3():
    # I = J + a * mu: on an isotropic grid sum w mu^2 / 4pi ~ 1/3
    dg = icosahedric_grid(n_refine=1)
    a = 0.4
    J = 1.0
    I = J + a * dg.directions[None, :, 2]
    var = angular_variance(I, dg)
    assert np.allclose(var, a**2 / 3.0, rtol=0.05)


def test_nonnegative_and_shift_invariance():
    rng = np.random.default_rng(42)
    dg = icosahedric_grid(n_refine=1)
    I = rng.uniform(0.0, 2.0, size=(7, dg.n_dir))
    var = angular_variance(I, dg)
    assert np.all(var >= 0.0)
    # adding a constant to every ordinate leaves J shifted equally -> same var
    var_shift = angular_variance(I + 3.7, dg)
    assert np.allclose(var, var_shift)
