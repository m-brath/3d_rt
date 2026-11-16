# -*- coding: utf-8 -*-
"""
MC tallies with history-batch error bars (plan/mc-plan.md section 5).

Stage 1 of the forward MC solver needs one estimator: the **track-length /
exponential-path tally** of the angle-averaged intensity.  A particle history
that carries power ``w`` and is removed by true absorption (or leaves the
domain) deposits its path length in every cell it crosses; the summed,
volume-normalized track length is the fluence, and ``J = fluence / 4 pi``:

    J_c = (1 / (4 pi V_c)) sum_k w_k  ell_{c,k}
    F_c = (1 / V_c)        sum_k w_k  ell_{c,k} Omega_k

(``V_c`` cell volume, ``ell_{c,k}`` path length history ``k`` spends in cell
``c``, ``Omega_k`` its direction during that flight).  The identity is exact
for a particle population that is killed at true absorption: a particle is
alive at path length ``s`` with probability ``exp(-tau_a(s))``, so
``E[ell_c] = int_0^inf 1_c exp(-tau_a) ds``, and the scattering generations of
an analog chain sum to the Neumann series of the RTE (plan section 1).

Weight conventions (per particle, so that ``w`` is a *power*, i.e. intensity
times area — the tally then has intensity units):

- isotropic volume source ``S(r)`` (intensity per length), births sampled with
  pdf proportional to ``S``: ``w = 4 pi Q / n``, ``Q = sum_c S_c V_c``;
- isotropic volume source, births sampled **uniformly in the domain volume**
  (the common-random-number mode of the lambda variant):
  ``w = 4 pi V_tot S(r_k) / n`` with ``V_tot`` the domain volume;
- face source with intensity ``I_in`` (isotropic over the inward hemisphere):
  ``w = pi A I_in / n`` with ``A`` the face area, directions cosine-weighted
  about the inward normal (the crossing-rate measure ``|mu| dOmega``);
- direct solar beam of irradiance ``F0`` (unit direction ``Omega_sun``):
  ``w = F0 |Omega_sun . n_face| A / n``, seeded uniformly over the entry face.

Error accounting follows plan section 5: histories are grouped into
``n_batches`` contiguous batches, every tally is accumulated per batch, and
``sigma`` is the batch-mean standard error (``ddof=1``); the plan's acceptance
bands are 2 sigma.
"""

import numpy as np

from rt3d.octree import octree_grid


def cell_volumes(grid):
    """
    Cell volume of every cell/leaf: ``hx*hy*hz`` on the unigrid,
    ``(2*half)**3`` on the octree.
    """
    if isinstance(grid, octree_grid):
        return (2.0 * grid.half) ** 3
    return np.full(grid.n_cells, grid.hx * grid.hy * grid.hz, dtype=float)


def n_cells_of(grid):
    """Number of cells (unigrid) / leaves (octree)."""
    return grid.n_leaf if isinstance(grid, octree_grid) else grid.n_cells


class track_tally(object):
    """
    Per-cell track-length tally of ``J`` (and optionally ``F``) with
    history-batch error bars.

    Parameters
    ----------
    n_cells : int
        Number of cells / leaves.
    n_histories : int
        Number of particle histories this tally is fed by; history indices
        are 0 .. n_histories-1 and are grouped into contiguous batches.
    n_batches : int, optional
        Number of history batches for the error estimate, by default 100
        (plan section 5); clipped to ``n_histories``.
    flux : bool, optional
        Also accumulate the flux-vector tally ``F``, by default True.
    """

    def __init__(self, n_cells, n_histories, n_batches=100, flux=True):
        self.n_cells = int(n_cells)
        self.n_histories = int(n_histories)
        self.n_batches = int(max(1, min(int(n_batches), self.n_histories)))
        self.flux = bool(flux)

        # contiguous history batches (standard history batching)
        edges = np.linspace(0, self.n_histories, self.n_batches + 1)
        edges = np.round(edges).astype(np.int64)
        self._batch = np.empty(self.n_histories, dtype=np.int64)
        for b in range(self.n_batches):
            self._batch[edges[b]:edges[b + 1]] = b

        self.sums = np.zeros((self.n_batches, self.n_cells))
        self.fsums = np.zeros((3, self.n_batches, self.n_cells)) if self.flux else None

    # ------------------------------------------------------------------ add

    def add(self, hist, cell, length, weight, direction=None):
        """
        Deposit one batch of track-length segments.

        Parameters
        ----------
        hist : (M,) ndarray of int
            History index of each segment.
        cell : (M,) ndarray of int
            Cell/leaf index of each segment.
        length : (M,) ndarray
            Segment length.
        weight : (M,) ndarray or float
            Particle power carried during the segment.
        direction : (M, 3) ndarray, optional
            Particle direction during the segment (needed for the flux
            tally; ignored when it is constant and ``flux`` is False).
        """
        length = np.asarray(length, dtype=float)
        if length.size == 0:
            return
        weight = np.asarray(weight, dtype=float)
        if weight.ndim == 0:
            weight = np.full(length.shape, float(weight))
        flat = self._batch[np.asarray(hist, dtype=np.int64)] * self.n_cells \
            + np.asarray(cell, dtype=np.int64)
        self.sums += np.bincount(
            flat, weights=length * weight, minlength=self.sums.size
        ).reshape(self.sums.shape)

        if self.flux:
            if direction is None:
                raise ValueError("flux tally needs segment directions")
            d = np.asarray(direction, dtype=float)
            for ax in range(3):
                self.fsums[ax] += np.bincount(
                    flat, weights=length * weight * d[:, ax],
                    minlength=self.sums.size,
                ).reshape(self.sums.shape)

    def merge(self, other):
        """
        Add another tally (same batching) — used to combine the fixed source
        contributions (boundary inflow, beam first scatter) with the iterated
        volume-source contribution so the reported ``sigma`` covers all of
        them.
        """
        if other.n_batches != self.n_batches or other.n_cells != self.n_cells:
            raise ValueError("cannot merge tallies with different batching")
        self.sums += other.sums
        if self.flux or other.flux:
            if self.fsums is None:
                self.fsums = np.zeros_like(other.fsums)
            if other.fsums is not None:
                self.fsums += other.fsums
            self.flux = True
        return self

    # --------------------------------------------------------------- result

    def result(self, volumes):
        """
        Batch-means estimate and standard error per cell.

        The estimate is the sum of the per-batch tallies; the standard error
        is ``sqrt(n_batches) * std(batch tallies, ddof=1)``, i.e. each batch
        tally is rescaled by ``n_batches`` so that it alone estimates the
        full-population value (standard history-batching).  Report 2 sigma
        bands as ``2 * sigma`` (plan section 5).

        Parameters
        ----------
        volumes : (n_cells,) ndarray
            Cell volumes (rt3d.mc.estimators.cell_volumes).

        Returns
        -------
        dict with keys
            'J', 'sigma_J' : (n_cells,) mean intensity and its 1-sigma
                standard error
            'F', 'sigma_F' : (n_cells, 3) flux vector and standard error
                (same layout as rt3d.source.flux_vector), only when ``flux``
                is on
            'fluence', 'sigma_fluence' : (n_cells,) track-length fluence
                ``sum w ell / V_c`` and its standard error
            'n_batches', 'n_histories'
        """
        volumes = np.asarray(volumes, dtype=float)
        nb = self.n_batches
        # batch-means estimator on *batch sums*: the tally total is the sum of
        # the per-batch sums, and its standard error is sqrt(nb) * std(batch
        # sums)  (equivalently: rescale each batch sum by nb so that every
        # batch alone estimates the full-population total).
        est = self.sums.sum(axis=0)
        se = (np.sqrt(nb) * self.sums.std(axis=0, ddof=1) if nb > 1
              else np.zeros(self.n_cells))
        out = {
            "fluence": est / volumes,
            "sigma_fluence": se / volumes,
            "J": est / (4.0 * np.pi * volumes),
            "sigma_J": se / (4.0 * np.pi * volumes),
            "n_batches": nb,
            "n_histories": self.n_histories,
        }
        if self.flux:
            fm = self.fsums.sum(axis=1) / volumes[None, :]
            fs = (np.sqrt(nb) * self.fsums.std(axis=1, ddof=1) if nb > 1
                  else np.zeros((3, self.n_cells))) / volumes[None, :]
            # (n_cells, 3) — same layout as rt3d.source.flux_vector, so MC and
            # SC flux fields can be differenced cell-by-cell
            out["F"] = fm.T
            out["sigma_F"] = fs.T
        return out


def heating_rate(kappa_a, B, J):
    """
    Net radiative heating rate ``4 pi kappa_a (B - J)`` — the emission/
    absorption side of verification invariant 4 (``div F`` equals this for a
    converged field).
    """
    return 4.0 * np.pi * np.asarray(kappa_a, dtype=float) * (
        np.asarray(B, dtype=float) - np.asarray(J, dtype=float)
    )
