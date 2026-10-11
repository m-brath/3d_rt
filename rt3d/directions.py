# -*- coding: utf-8 -*-
"""
Icosahedral (geodesic) direction grids with solid-angle weights and
precomputed Henyey-Greenstein phase-function matrices.

Conventions follow scripts/icosphere_grid.py (Manfred Brath). The O(N^2)
pure-Python dedup refinement of that file is replaced here by a vectorized
edge-based refinement (needed for n_refine >= 2, i.e. 162/642 ordinates).

The grid is rotated about the y-axis on construction so that no ordinate has
exactly Omega_z = 0: such ordinates form cyclic characteristics under x,y
periodicity (plan section 2).
"""

import numpy as np


# =============================================================================
# icosahedric grid construction
# =============================================================================


def make_icosahedric_grid():
    """
    Create the base icosahedric grid (12 vertices, 20 triangles).

    Same construction as scripts/icosphere_grid.py:make_icosahedric_grid.

    Returns
    -------
    vertices : (12, 3) ndarray
        Unit-norm cartesian vertex coordinates.
    triangles : (20, 3) ndarray
        Vertex indices of each spherical triangle.
    """
    # golden number
    Z = (1 + np.sqrt(5)) / 2
    N = 0
    X = 1

    vertices = np.array(
        [
            [-X, N, Z],
            [X, N, Z],
            [-X, N, -Z],
            [X, N, -Z],
            [N, Z, X],
            [N, Z, -X],
            [N, -Z, X],
            [N, -Z, -X],
            [Z, X, N],
            [-Z, X, N],
            [Z, -X, N],
            [-Z, -X, N],
        ],
        dtype=float,
    )

    vertices /= np.sqrt(np.sum(vertices**2, axis=1))[:, None]

    triangles = np.array(
        [
            [0, 4, 1], [0, 9, 4], [9, 5, 4], [4, 5, 8], [4, 8, 1],
            [8, 10, 1], [8, 3, 10], [5, 3, 8], [5, 2, 3], [2, 7, 3],
            [7, 10, 3], [7, 6, 10], [7, 11, 6], [11, 0, 6], [0, 1, 6],
            [6, 1, 10], [9, 0, 11], [9, 11, 2], [9, 2, 5], [7, 2, 11],
        ],
        dtype=int,
    )

    return vertices, triangles


def refine_icosahedric_grid(vertices, triangles, n_refine=1):
    """
    Vectorized geodesic refinement: bisect every edge, project midpoints to
    the unit sphere, split every triangle into four.

    Replaces the O(N^2) dedup loop of scripts/icosphere_grid.py with an
    edge-indexed midpoint table (np.unique over sorted index pairs), which is
    exact (no float comparison) and fast enough for n_refine = 3 (642 points).

    Parameters
    ----------
    vertices : (N, 3) ndarray
        Unit-norm vertex coordinates.
    triangles : (M, 3) ndarray
        Vertex indices of each triangle.
    n_refine : int, optional
        Number of refinement levels to apply, by default 1.

    Returns
    -------
    vertices : (10*4**n + N0*..., 3) ndarray
    triangles : (M0*4**n, 3) ndarray
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    for _ in range(n_refine):
        # the three edges of every triangle, as sorted index pairs
        edges = np.concatenate(
            [
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)

        # unique edges -> one midpoint per shared edge
        uniq, inverse = np.unique(edges, axis=0, return_inverse=True)
        mid = vertices[uniq[:, 0]] + vertices[uniq[:, 1]]
        mid /= np.linalg.norm(mid, axis=1)[:, None]

        n_old = vertices.shape[0]
        # inverse is block-ordered (all edge01, then all edge12, then all edge20)
        edge_mid = n_old + inverse.reshape(3, -1).T  # (M, 3): mid of (0,1),(1,2),(2,0)

        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        m_ab, m_bc, m_ca = edge_mid[:, 0], edge_mid[:, 1], edge_mid[:, 2]

        triangles_new = np.concatenate(
            [
                np.column_stack([a, m_ab, m_ca]),
                np.column_stack([m_ab, b, m_bc]),
                np.column_stack([m_ca, m_bc, c]),
                np.column_stack([m_ab, m_bc, m_ca]),
            ]
        )

        vertices = np.concatenate([vertices, mid])
        triangles = triangles_new

    return vertices, triangles


# =============================================================================
# solid-angle weights (spherical triangle areas, L'Huilier)
# =============================================================================


def solid_angle_weights(vertices, triangles):
    """
    Solid-angle weight per vertex from spherical-triangle areas.

    Each spherical triangle area (spherical excess via L'Huilier's formula)
    is distributed equally to its three corner vertices. Icosphere vertices
    are *not* equal-area, so this is not 4*pi/N.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        Unit-norm vertex coordinates.
    triangles : (M, 3) ndarray
        Vertex indices of each spherical triangle.

    Returns
    -------
    weights : (N,) ndarray
        Solid-angle weights, sum(weights) == 4*pi.
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # side lengths as central angles
    a = np.arccos(np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0))
    b = np.arccos(np.clip(np.sum(v2 * v0, axis=1), -1.0, 1.0))
    c = np.arccos(np.clip(np.sum(v0 * v1, axis=1), -1.0, 1.0))

    s = 0.5 * (a + b + c)
    # L'Huilier: tan(E/4) = sqrt(tan(s/2) tan((s-a)/2) tan((s-b)/2) tan((s-c)/2))
    t = np.tan(s / 2) * np.tan((s - a) / 2) * np.tan((s - b) / 2) * np.tan((s - c) / 2)
    excess = 4.0 * np.arctan(np.sqrt(np.clip(t, 0.0, None)))

    weights = np.zeros(vertices.shape[0])
    np.add.at(weights, triangles[:, 0], excess / 3.0)
    np.add.at(weights, triangles[:, 1], excess / 3.0)
    np.add.at(weights, triangles[:, 2], excess / 3.0)

    return weights


# =============================================================================
# Henyey-Greenstein phase function
# =============================================================================


def hg(cos_theta, g):
    """
    Henyey-Greenstein phase function, normalized so int Phi dOmega = 1.

    Phi = (1 - g^2) / (4 pi (1 + g^2 - 2 g cos(Theta))^(3/2))

    Parameters
    ----------
    cos_theta : array_like
        Cosine of the scattering angle.
    g : float
        Asymmetry parameter, -1 < g < 1.
    """
    cos_theta = np.asarray(cos_theta, dtype=float)
    denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5
    return (1.0 - g * g) / (4.0 * np.pi * denom)


class direction_grid(object):
    """
    Icosahedral direction set with solid-angle weights.

    Attributes
    ----------
    n_refine : int
        Icosahedral refinement level; N_dir = 10*4**n_refine + 2.
    directions : (N_dir, 3) ndarray
        Unit direction vectors Omega_i.
    weights : (N_dir,) ndarray
        Solid-angle weights w_i, sum == 4*pi.
    triangles : (M, 3) ndarray
        Connectivity of the underlying geodesic grid.
    """

    def __init__(self, n_refine=1):
        if n_refine < 0 or int(n_refine) != n_refine:
            raise ValueError(f"n_refine must be a non-negative integer, got {n_refine}")
        self.n_refine = int(n_refine)

        vertices, triangles = make_icosahedric_grid()
        vertices, triangles = refine_icosahedric_grid(vertices, triangles, self.n_refine)

        # rotate about a generic axis so that no ordinate has Omega_z == 0
        # (cyclic characteristics under x,y periodicity, plan section 2).
        # Coordinate-axis rotations are excluded: the refined grid contains
        # vertices on the coordinate axes, which such rotations leave fixed.
        vertices = self._rotate_off_poles(vertices)

        self.directions = vertices
        self.triangles = triangles
        self.weights = solid_angle_weights(vertices, triangles)
        # no ordinate has Omega_z == 0 (_rotate_off_poles), so these two
        # index arrays partition all ordinates
        self.down_idx = np.flatnonzero(vertices[:, 2] < 0.0)
        self.up_idx = np.flatnonzero(vertices[:, 2] > 0.0)

    @staticmethod
    def _rotate_off_poles(vertices):
        """
        Rotate the grid about a generic axis so that no ordinate has
        Omega_z == 0 (exact zeros form cyclic characteristics under x,y
        periodicity). Deterministically sweeps candidate (axis, angle) pairs
        and keeps the rotation maximizing min|Omega_z|. Coordinate-axis
        rotations are excluded: the refined grid has vertices on the
        coordinate axes, which such rotations leave fixed.

        Note: for large N_dir the quasi-uniform grid always has some
        ordinate close to the equator (min|Omega_z| ~ 0.01 at 642 ordinates);
        only *exact* zeros are harmful, so the guard is 1e-6.
        """
        best, best_min = None, -1.0
        for axis in ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [3.0, 1.0, 2.0], [1.0, 3.0, 2.0]):
            k = np.asarray(axis) / np.linalg.norm(axis)
            K = np.array(
                [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
            )
            for angle in np.arange(0.05, np.pi, 0.05):
                # Rodrigues rotation matrix
                R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
                cand = vertices @ R.T
                m = np.min(np.abs(cand[:, 2]))
                if m > best_min:
                    best, best_min = cand, m
        if best is None or best_min < 1e-6:
            raise RuntimeError(
                f"could not rotate direction grid off the equator: min|Omega_z| = {best_min:.3e}"
            )
        return best

    @property
    def n_dir(self):
        return self.directions.shape[0]

    def cos_theta(self):
        """(N_dir, N_dir) matrix of cos(Omega_i, Omega_j)."""
        return np.clip(self.directions @ self.directions.T, -1.0, 1.0)

    def hg_matrix(self, g, with_weights=True, normalize=True):
        """
        Precomputed HG phase-function matrix Phi[i, j] = Phi_HG(Omega_j -> Omega_i).

        With with_weights=True the quadrature weights of the *incoming*
        direction are folded in: W[i, j] = Phi_HG(Omega_j -> Omega_i) * w_j,
        so the scattering integral over all cells is one BLAS matmul
        (n_cells x N_dir) @ (N_dir x N_dir).

        With normalize=True (the default) each row is rescaled so that
        sum_j W[i, j] = 1 exactly.  The analytic HG function integrates to 1
        over the sphere (validation anchor 2), but the coarse icosahedral
        quadrature breaks this at large g (42 ordinates, g = 0.952: row sums
        17-22), which makes the lambda-iteration scattering operator
        supercritical and divergent for omega not far below 1.  Row
        normalization enforces exact discrete conservation of isotropic
        radiation; at g = 0 the row sums are already exactly 1, so isotropic
        results are unchanged.

        Parameters
        ----------
        g : float
            Asymmetry parameter.
        with_weights : bool, optional
            Fold in the incoming-direction weights, by default True.
        normalize : bool, optional
            Row-normalize the weighted matrix to unit row sums, by default True.
        """
        W = hg(self.cos_theta(), g)
        if with_weights:
            W = W * self.weights[None, :]
            if normalize:
                W = W / W.sum(axis=1, keepdims=True)
        return W

    def hg_delta_matrices(self, g, M):
        """
        Delta-M split of the row-normalized weighted HG matrix.

        The split is defined *on the normalized matrix* of hg_matrix
        (normalize=True), so that Phi_w == Phi_w_delta + sum(moments)
        exactly (as matrices, to machine precision).  The scattering
        integral therefore splits exactly, for any intensity array I:

            I @ Phi_w.T == I @ Phi_w_delta.T + sum_n I @ moments[n].T

        Delta-M changes only the iteration/transport scaling (the caller
        reduces the transport extinction to kappa_e - kappa_s * g**M, the
        scattering part only; rt3d.source.delta_extinction) and adds the
        explicit -omega * g**M * I term to the source), not the converged
        discrete solution.

        P_n(cos_theta_ij) is evaluated by the Legendre recurrence
        P_0 = 1, P_1 = x, (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}.

        Parameters
        ----------
        g : float
            Asymmetry parameter, -1 < g < 1.
        M : int
            Number of Legendre moments to subtract, positive integer.

        Returns
        -------
        Phi_w_delta : (N_dir, N_dir) ndarray
            hg_matrix(g) minus the first M Legendre moment operators.
        moments : list of (N_dir, N_dir) ndarray
            moments[n][i, j] = (2n+1) g**n P_n(Omega_i . Omega_j) * w_j / (4 pi),
            n = 0 .. M-1, so that
            Phi_w == Phi_w_delta + sum(moments) exactly.
        """
        if M < 1 or int(M) != M:
            raise ValueError(f"M must be a positive integer, got {M}")
        if not (-1.0 < g < 1.0):
            raise ValueError(f"g must satisfy -1 < g < 1, got {g}")

        x = self.cos_theta()
        w4pi = self.weights[None, :] / (4.0 * np.pi)

        # classical Wiscombe weights: HG Legendre moments are g**n, so the
        # subtracted operators carry the factor g**n (unweighted subtraction
        # would over-subtract P_1 at g < 1 and enlarge the M >= 2 residual)
        moments = []
        g_pow = 1.0  # g**n
        P_nm1 = np.ones_like(x)  # P_0(x) = 1
        moments.append(1.0 * g_pow * P_nm1 * w4pi)
        if M >= 2:
            P_n = x.copy()  # P_1(x) = x
            g_pow *= g
            moments.append(3.0 * g_pow * P_n * w4pi)
            for n in range(1, M - 1):
                # (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}
                P_np1 = ((2.0 * n + 1.0) * x * P_n - n * P_nm1) / (n + 1.0)
                g_pow *= g
                moments.append((2.0 * (n + 1) + 1.0) * g_pow * P_np1 * w4pi)
                P_nm1, P_n = P_n, P_np1

        Phi_w_delta = self.hg_matrix(g) - sum(moments)
        return Phi_w_delta, moments


def icosahedric_grid(n_refine=1):
    """
    Build the icosahedral direction grid. The single place the default
    n_refine = 1 (42 ordinates, the development grid) appears.

    Parameters
    ----------
    n_refine : int, optional
        Refinement level: 0/1/2/3 -> 12/42/162/642 ordinates, by default 1.

    Returns
    -------
    direction_grid
    """
    return direction_grid(n_refine=n_refine)
