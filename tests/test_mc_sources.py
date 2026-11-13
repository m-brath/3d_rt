# -*- coding: utf-8 -*-
"""
MC phase M0 source samplers (plan/mc-plan.md section 1): HG inverse-CDF
sampling must reproduce the continuous HG moments (E[P_n(cos t)] = g^n)
and agree with the `hg_matrix` quadrature moments to quadrature accuracy
(companion of validation anchor 2).

Birth-point sampling vs. analytic rates joins this file with M1.
"""

import numpy as np
import pytest

from rt3d.directions import hg, icosahedric_grid
from rt3d.mc.sources import hg_cos_theta, hg_directions


def _legendre(n, x):
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    p0, p1 = np.ones_like(x), x
    for m in range(1, n):
        p0, p1 = p1, ((2 * m + 1) * x * p1 - m * p0) / (m + 1)
    return p1


def test_hg_cos_theta_monotone_endpoints():
    """Inverse CDF maps [0,1) -> [-1,1] monotonically, both g signs."""
    u = np.linspace(0.0, 1.0 - 1e-12, 1000)
    for g in (-0.9, -0.3, 0.0, 0.3, 0.9):
        c = hg_cos_theta(u, g)
        assert c[0] == pytest.approx(-1.0, abs=1e-9)
        assert c[-1] <= 1.0
        assert np.all(np.diff(c) >= 0.0)
    assert hg_cos_theta(np.array([1.0 - 1e-15]), 0.5)[0] == pytest.approx(1.0, abs=1e-6)


def test_hg_invalid_g():
    with pytest.raises(ValueError):
        hg_cos_theta(np.array([0.5]), 1.0)
    with pytest.raises(ValueError):
        hg_cos_theta(np.array([0.5]), -1.2)


def test_hg_isotropic_limit():
    """g = 0: cos(t) ~ U(-1,1): E = 0, E[cos^2] = 1/3, E[cos^3] = 0."""
    rng = np.random.default_rng(3)
    c = hg_cos_theta(rng.random(400_000), 0.0)
    assert abs(c.mean()) < 4 * np.sqrt(1.0 / 3 / c.size)
    assert abs(np.mean(c**2) - 1.0 / 3.0) < 4 * np.sqrt(np.var(c**2) / c.size)
    assert abs(np.mean(c**3)) < 4 * np.sqrt(np.var(c**3) / c.size)


@pytest.mark.parametrize("g", [-0.5, -0.2, 0.3, 0.6, 0.9])
def test_hg_legendre_moments(g):
    """The continuous HG phase function has Legendre moments g^n; the
    inverse-CDF sampler is exact, so the MC moments must match within the
    sampling noise."""
    rng = np.random.default_rng(11 + int(abs(g) * 100))
    c = hg_cos_theta(rng.random(200_000), g)
    for n in range(1, 4):  # n = 0 (P_0 == 1) is trivial
        pn = _legendre(n, c)
        target = g**n
        se = np.sqrt(np.var(pn) / c.size)
        assert abs(np.mean(pn) - target) < 6.0 * se, (
            f"g={g}, n={n}: MC {np.mean(pn):.5f} vs. g^n {target:.5f}, se {se:.5f}"
        )


@pytest.mark.parametrize("g", [0.0, 0.3])
def test_hg_moments_vs_hg_matrix_quadrature(g):
    """MC sample moments vs. the direction_grid.hg_matrix quadrature
    moments: the sampler is exact, so the deviation must equal the raw
    quadrature error |g^n - quad| up to MC noise."""
    dg = icosahedric_grid(n_refine=1)
    omega0 = dg.directions[0]
    cos = np.clip(dg.directions @ omega0, -1.0, 1.0)
    w = dg.weights

    rng = np.random.default_rng(23)
    c = hg_cos_theta(rng.random(400_000), g)
    for n in range(4):
        quad = np.sum(w * hg(cos, g) * _legendre(n, cos))  # unnormalized anchor-2 quadrature
        mc = np.mean(_legendre(n, c))
        se = np.sqrt(np.var(_legendre(n, c)) / c.size)
        assert abs(mc - quad) <= abs(g**n - quad) + 6.0 * se, (
            f"g={g}, n={n}: MC {mc:.5f}, quad {quad:.5f}, g^n {g**n:.5f}"
        )


def test_hg_directions_unit_and_scatter_angle():
    """Outgoing directions are unit vectors whose cosine with the incoming
    direction equals the sampled cos(theta); the frame choice is invisible
    (azimuth uniform, no preferred transverse direction)."""
    rng = np.random.default_rng(31)
    dg = icosahedric_grid(n_refine=1)
    n = 39_996  # 6 incoming ordinates x 6666 samples
    d_in = np.repeat(dg.directions[:6], n // 6, axis=0)
    u1, u2 = rng.random(2 * n).reshape(2, n)
    for g in (0.0, 0.6):
        d_out = hg_directions(d_in, u1, u2, g)
        assert np.allclose(np.linalg.norm(d_out, axis=1), 1.0, atol=1e-12)
        cos = np.sum(d_in * d_out, axis=1)
        assert np.allclose(cos, hg_cos_theta(u1, g), atol=1e-12)
        # HG first moment: <cos(t)> = g
        se = np.sqrt(1.0 / 3.0 / n)
        assert abs(cos.mean() - g) < 6 * se
        # azimuth symmetry: mean transverse deviation ~ 0
        perp = d_out - cos[:, None] * d_in
        assert np.linalg.norm(perp.mean(axis=0)) < 6 * np.sqrt(0.5 / n)


def test_hg_directions_forward_peaking(g=0.9):
    """Strongly forward HG: most samples scatter within a narrow cone."""
    rng = np.random.default_rng(37)
    d_in = np.tile(np.array([0.0, 0.0, 1.0]), (100_000, 1))
    d_out = hg_directions(d_in, rng.random(100_000), rng.random(100_000), g)
    assert np.mean(d_out[:, 2] > 0.9) > 0.8
    # momentum conservation of the HG kernel: <Omega_out> = g * Omega_in
    assert np.allclose(d_out.mean(axis=0), [0, 0, g], atol=0.02)
