# -*- coding: utf-8 -*-
"""
1D Ng acceleration of the lambda iteration (plan section 5, phase 4).

Ng (1982) acceleration with a plane-averaged (z) estimate of the scattering
operator Lambda_bar, as decided in the plan (1D column-averaged Lambda_bar
in z, dense Nz x Nz solve).  It needs no extra transport sweeps: the LI
iterate differences themselves probe Lambda_bar.

Notation: sigma^n the source actually swept at iteration n,
S^(n+1) = Lambda sigma^n + Q the plain-LI update (Q = fixed source parts).
The history pairs probe Lambda_bar exactly, even for accelerated iterates:

    x^(m) = sigma^m - sigma^(m-1),   y^(m) = Lambda sigma^m - Lambda sigma^(m-1)
                                    = S^(m+1) - S^(m)   (up to Q, which cancels)

Lambda_bar (Nz x Nz) is the minimum-norm fit of the plane-averaged pairs
y = Lambda_bar x.  With d^(n+1) = S^(n+1) - sigma^n the last LI step, the
accelerated source is the geometric-series extrapolation

    sigma^(n+1) = S^(n+1) + (I - Lambda_bar)^-1 Lambda_bar d^(n+1),

applied per z bin.  Scalar check (Lambda = lambda): d^(n+1) =
(1 - lambda)(S* - sigma^n), the correction is lambda (S* - sigma^n), so
sigma^(n+1) = lambda sigma^n + Q + lambda (S* - sigma^n) = S* exactly.

The scattering part of the source is isotropic (scalar RTE, HG scattering)
and the fixed parts Q cancel in the differences, so the history is stored
as direction-averaged (n_cells,) vectors.
"""

import numpy as np


class ng1d:
    """
    Plane-averaged 1D Ng acceleration state for lambda_iteration.

    Parameters
    ----------
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
        Grid whose z structure defines the 1D bins.  Unigrid: one bin per
        z layer.  Octree: one bin per base-cell z level, leaves weighted
        by their horizontal cross-section.
    max_hist : int, optional
        Maximum number of history pairs kept for the Lambda_bar fit
        (older pairs are dropped).  Default 3 * n_bins.
    n_start : int, optional
        Number of history pairs before acceleration starts (plain-LI
        warmup), by default 3.
    """

    def __init__(self, grid, max_hist=None, n_start=3):
        from rt3d.octree import octree_grid

        if isinstance(grid, octree_grid):
            z0, z1 = grid.box[2]
            nzb = int(round((z1 - z0) / grid.h0))
            edges = np.linspace(z0, z1, nzb + 1)
            zk = grid.centers[:, 2]
            self.bin_idx = np.clip(np.digitize(zk, edges[1:-1]), 0, nzb - 1)
            self.bin_w = (2.0 * grid.half) ** 2
            self.n_bins = nzb
        else:
            self.n_bins = grid.nz
            self.bin_idx = np.repeat(np.arange(grid.nz), grid.ny * grid.nx)
            self.bin_w = np.ones(grid.n_cells)

        self.max_hist = 3 * self.n_bins if max_hist is None else int(max_hist)
        self.n_start = int(n_start)
        self.px = []  # plane averages of swept-source differences
        self.py = []  # plane averages of LI-update differences
        self.prev_swept = None
        self.prev_S = None

    def plane_average(self, v):
        """Horizontal (area-weighted) average of a cell quantity per z bin."""
        num = np.zeros(self.n_bins)
        den = np.zeros(self.n_bins)
        np.add.at(num, self.bin_idx, v * self.bin_w)
        np.add.at(den, self.bin_idx, self.bin_w)
        return num / np.maximum(den, 1e-300)

    def lambda_bar(self):
        """Minimum-norm Nz x Nz fit of the pairs: Lam @ x^(m) = y^(m)."""
        X = np.column_stack(self.px)  # Nz x n_pairs
        Y = np.column_stack(self.py)
        Lam_t, *_ = np.linalg.lstsq(X.T, Y.T, rcond=None)
        return Lam_t.T

    def step(self, S_swept, S_new):
        """
        One acceleration step.

        Parameters
        ----------
        S_swept : (n_cells, N_dir) ndarray
            Source that was just transported (sigma^n).
        S_new : (n_cells, N_dir) ndarray
            Plain-LI update Lambda sigma^n + Q.

        Returns
        -------
        (n_cells, N_dir) ndarray
            Source for the next sweep: S_new during warmup, the Ng
            geometric-series extrapolation afterwards.
        """
        s_now = np.mean(S_swept, axis=1)
        sn_now = np.mean(S_new, axis=1)
        if self.prev_swept is not None:
            self.px.append(self.plane_average(s_now - self.prev_swept))
            self.py.append(self.plane_average(sn_now - self.prev_S))
            if len(self.px) > self.max_hist:
                self.px.pop(0)
                self.py.pop(0)
        self.prev_swept = s_now
        self.prev_S = sn_now

        if len(self.px) <= self.n_start:
            return S_new

        Lam = self.lambda_bar()
        # safeguard: a bad fit with spectral radius >= 1 would make the
        # geometric series blow up - fall back to plain LI for this step
        if np.max(np.abs(np.linalg.eigvals(Lam))) >= 1.0:
            return S_new
        d_z = self.plane_average(sn_now - s_now)
        corr_z = np.linalg.solve(
            np.eye(self.n_bins) - Lam, Lam @ d_z
        )
        if not np.all(np.isfinite(corr_z)):
            return S_new
        return S_new + corr_z[self.bin_idx][:, None]
