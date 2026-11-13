# -*- coding: utf-8 -*-
"""
MC source samplers (plan/mc-plan.md sections 1-2).

M0 delivers the **HG inverse-CDF scatter sampler**: continuous Henyey-
Greenstein direction sampling with no quadrature and no delta-M (MC
transport is exact in angle; the icosahedral grid enters only as tally
binning).  The sampler must reproduce the `hg_matrix` quadrature moments
to quadrature accuracy — the companion of validation anchor 2, asserted
in tests/test_mc_sources.py.

HG inverse-CDF (with u ~ U(0,1), -1 < g < 1):

    g != 0:  q       = 2 g u / (1 - g^2) + 1/(1 + g)
             cos(t)  = (1 + g^2 - 1/q^2) / (2 g)
    g  = 0:  cos(t)  = 2 u - 1            (isotropic limit)

The formula is monotone in u from cos = -1 (u = 0) to cos = +1 (u = 1)
for both signs of g.  The azimuth is uniform on [0, 2 pi).

Birth-point sampling from inhomogeneous source rates (thermal kappa_a B
and the fixed source S_n of the lambda-iteration variant, both by chained
Woodcock sampling with null source events) arrives with M1 in this file.
"""

import numpy as np


def hg_cos_theta(u, g):
    """
    Inverse-CDF sampling of the HG scattering angle cosine.

    Parameters
    ----------
    u : array_like
        Uniform samples in [0, 1).
    g : float
        Asymmetry parameter, -1 < g < 1.

    Returns
    -------
    cos_theta : ndarray
        Cosine of the scattering angle (between incoming and outgoing
        direction), same shape as u, in [-1, 1].
    """
    if not (-1.0 < g < 1.0):
        raise ValueError(f"g must satisfy -1 < g < 1, got {g}")
    u = np.asarray(u, dtype=float)
    if abs(g) < 1e-12:  # isotropic limit (the CDF formula is 0/0 there)
        return 2.0 * u - 1.0
    q = 2.0 * g * u / (1.0 - g * g) + 1.0 / (1.0 + g)
    return np.clip((1.0 + g * g - 1.0 / (q * q)) / (2.0 * g), -1.0, 1.0)


def hg_directions(d_in, u_cos_theta, u_azimuth, g):
    """
    Sample outgoing HG directions for incoming directions `d_in`.

    The scattering polar angle cosine comes from `hg_cos_theta` applied to
    `u_cos_theta`; the azimuth around the incoming direction is uniform,
    psi = 2 pi u_azimuth.  HG reciprocity
    Phi(Omega'.Omega) = Phi(Omega.Omega') means the same sampler serves
    the adjoint (backward) transport run in reverse (plan section 4).

    Parameters
    ----------
    d_in : (N, 3) or (3,) ndarray
        Incoming (unit) directions.
    u_cos_theta, u_azimuth : (N,) ndarray
        Uniform samples.
    g : float
        Asymmetry parameter.

    Returns
    -------
    d_out : (N, 3) ndarray
        Unit outgoing directions.
    """
    d = np.atleast_2d(np.asarray(d_in, dtype=float))
    d = d / np.linalg.norm(d, axis=1)[:, None]
    n = d.shape[0]

    # orthonormal frame (e1, e2, d): reference axis chosen away from d
    ref = np.where(np.abs(d[:, 2:3]) <= 0.9,
                   np.array([0.0, 0.0, 1.0]),
                   np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(ref, d)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(d, e1)

    c = hg_cos_theta(np.asarray(u_cos_theta, dtype=float), g)
    s = np.sqrt(np.maximum(0.0, 1.0 - c * c))
    psi = 2.0 * np.pi * np.asarray(u_azimuth, dtype=float)
    cp, sp = np.cos(psi), np.sin(psi)

    out = c[:, None] * d + s[:, None] * (cp[:, None] * e1 + sp[:, None] * e2)
    return out
