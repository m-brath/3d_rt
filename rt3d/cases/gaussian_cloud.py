# -*- coding: utf-8 -*-
"""
Gaussian cloud test case — the `scripts/class_test.py` case expressed as
callable input fields (validation anchor 5: octree vs uniform grid).

Conventions copied from class_test.py:

    domain        x, y in [-5, 5], z in [0, 10]
    kappa_e       C * gauss(sigma=1, mu=(0, 0, 4)), C such that the
                  maximum vertical column optical depth is 10
    omega_local   gauss / max(gauss)          (peak 1.0 — conservative core)
    g_local       gauss / (max(gauss) * 1.05) (peak ~0.952)

class_test.py stores `albedo`; new code takes kappa_s = omega_local * kappa_e.
`omega_cap` optionally caps omega_local so that plain lambda iteration
converges tractably (Ng acceleration is phase 4); the uncapped case is the
faithful class_test.py setup.
"""

import numpy as np

from rt3d.geometry import cuboid


def gauss(x, y, z, sigma=1.0, mu=(0.0, 0.0, 4.0)):
    """Normalized 3D Gaussian (integral over all space = 1)."""
    return (
        1.0
        / (2.0 * np.pi * sigma**2)
        * np.exp(
            -(
                (x - mu[0]) ** 2
                + (y - mu[1]) ** 2
                + (z - mu[2]) ** 2
            )
            / (2.0 * sigma**2)
        )
    )


def gaussian_cloud(sigma=1.0, mu=(0.0, 0.0, 4.0), tau_column_max=10.0,
                   omega_cap=None):
    """
    Build the class_test.py Gaussian cloud.

    Parameters
    ----------
    sigma : float
        Cloud standard deviation (same in x, y, z), by default 1.0.
    mu : (3,) tuple
        Cloud center, by default (0, 0, 4).
    tau_column_max : float
        Maximum vertical column optical depth of kappa_e, by default 10.
    omega_cap : float, optional
        Cap on the local single scattering albedo (default None: peak 1.0,
        faithful to class_test.py).

    Returns
    -------
    case : dict with keys
        'box'      rt3d.geometry.cuboid (x, y in [-5, 5], z in [0, 10])
        'kappa_e', 'kappa_s', 'g', 'T' : callables (x, y, z) -> array
    """
    box = cuboid(-5.0, 5.0, -5.0, 5.0, 0.0, 10.0)

    # peak kappa_e from the max vertical column integral:
    # int gauss dz through the center = 1 / (sqrt(2 pi) sigma)
    C = tau_column_max * np.sqrt(2.0 * np.pi) * sigma

    def kappa_e(x, y, z):
        return C * gauss(x, y, z, sigma=sigma, mu=mu)

    def omega_local(x, y, z):
        w = gauss(x, y, z, sigma=sigma, mu=mu)
        w = w / gauss(*mu, sigma=sigma, mu=mu)  # peak 1 at the cloud center
        if omega_cap is not None:
            w = np.minimum(w, omega_cap)
        return w

    def kappa_s(x, y, z):
        return omega_local(x, y, z) * kappa_e(x, y, z)

    def g(x, y, z):
        return gauss(x, y, z, sigma=sigma, mu=mu) / (1.05 * gauss(*mu, sigma=sigma, mu=mu))

    def T(x, y, z):
        return np.zeros_like(np.asarray(x, dtype=float))

    return {
        "box": box,
        "kappa_e": kappa_e,
        "kappa_s": kappa_s,
        "g": g,
        "T": T,
    }
