# -*- coding: utf-8 -*-
"""
MC source samplers (plan/mc-plan.md sections 1-2).

M0 delivers the **HG inverse-CDF scatter sampler**: continuous Henyey-
Greenstein direction sampling with no quadrature and no delta-M (MC
transport is exact in angle; the icosahedral grid enters only as tally
binning).  The sampler must reproduce the `hg_matrix` quadrature moments
to quadrature accuracy — the companion of verification invariant 2, asserted
in tests/test_mc_sources.py.

HG inverse-CDF (with u ~ U(0,1), -1 < g < 1):

    g != 0:  q       = 2 g u / (1 - g^2) + 1/(1 + g)
             cos(t)  = (1 + g^2 - 1/q^2) / (2 g)
    g  = 0:  cos(t)  = 2 u - 1            (isotropic limit)

The formula is monotone in u from cos = -1 (u = 0) to cos = +1 (u = 1)
for both signs of g.  The azimuth is uniform on [0, 2 pi).

Birth-point sampling from inhomogeneous source rates (thermal ``kappa_a B``
and the fixed source ``S_n`` of the lambda-iteration variant) and the seeding
of face / direct-beam particle populations are also here (M1).  On
piecewise-constant cells the chained-Woodcock birth process with **null source
events** reduces exactly to the accept/reject test against the cell-majorant
``rate_max >= rate(cell)`` implemented by `sample_births`
(``method="null_collision"``): the along-flight chain of null source collisions
and the cell-level rejection test sample the same density
``p(r) proportional to rate(cell)``.  The null-source-event count is returned
as a first-class diagnostic, mirroring the flight tracker's ``n_null``.
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


# =============================================================================
# direction and position samplers for particle populations (M1)
# =============================================================================


def hg_scatter_directions(d_in, cells, g, rng):
    """
    HG direction resampling at scattering events, with `g` given as a scalar
    or as a per-cell field (looked up at each event's cell `cells` and
    bucketed by unique value — the `rt3d.source.scattering_source` bucketing
    convention).

    `rng` supplies the uniform deviates, so the caller controls the stream
    (reproducibility / common random numbers).
    """
    d_in = np.atleast_2d(np.asarray(d_in, dtype=float))
    n = d_in.shape[0]
    if np.ndim(g) == 0:
        return hg_directions(d_in, rng.random(n), rng.random(n), float(g))
    gv = np.round(np.asarray(g, dtype=float)[np.asarray(cells, dtype=int)], 6)
    out = np.empty_like(d_in)
    for gu in np.unique(gv):
        m = gv == gu
        k = int(np.count_nonzero(m))
        out[m] = hg_directions(d_in[m], rng.random(k), rng.random(k), float(gu))
    return out


def isotropic_directions(n, rng):
    """
    `n` isotropic unit directions (pdf 1 / 4 pi on the sphere): polar cosine
    uniform in [-1, 1], azimuth uniform in [0, 2 pi).
    """
    mu = 2.0 * rng.random(n) - 1.0
    s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    psi = 2.0 * np.pi * rng.random(n)
    return np.column_stack([s * np.cos(psi), s * np.sin(psi), mu])


def face_inflow_directions(n, axis, side, rng):
    """
    Cosine-weighted inward directions for a face source of given
    ``axis`` (0/1/2) and ``side`` (-1 lower / +1 upper face).

    A face source of constant intensity ``I_in`` injects power
    ``I_in |mu| dA dOmega`` — the crossing-rate measure of the face — so the
    direction pdf is ``|mu| / pi`` over the inward hemisphere: cosine law
    ``|mu| = sqrt(u)``, uniform azimuth.  Combined with the per-particle power
    ``w = pi A I_in / n` (plan/mc-plan.md section 5 conventions) the tally
    reproduces the boundary term of the RTE exactly.
    """
    mu = np.sqrt(rng.random(n))
    s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    psi = 2.0 * np.pi * rng.random(n)
    d = np.zeros((n, 3))
    d[:, axis] = -float(side) * mu          # inward
    tang = [a for a in range(3) if a != axis]
    d[:, tang[0]] = s * np.cos(psi)
    d[:, tang[1]] = s * np.sin(psi)
    return d


def face_points(n, box, axis, side, rng, eps=None):
    """
    `n` points uniformly over the face ``(axis, side)`` of ``box``, offset
    ``eps`` inwards so they lie strictly inside the domain (default
    ``1e-9 * min axis length``).
    """
    lo, hi = box[axis]
    if eps is None:
        eps = 1e-9 * min(b[1] - b[0] for b in box)
    p = np.empty((n, 3))
    for a in range(3):
        if a == axis:
            continue
        p[:, a] = rng.uniform(box[a][0], box[a][1], n)
    p[:, axis] = (lo if side < 0 else hi) - float(side) * eps
    return p


def face_area(box, axis):
    """Area of the face perpendicular to ``axis``."""
    area = 1.0
    for a in range(3):
        if a != axis:
            area *= box[a][1] - box[a][0]
    return float(area)


def sample_births(n, grid, rate, method="categorical", seed=0, volumes=None):
    """
    Sample birth positions from an inhomogeneous, piecewise-constant source
    rate ``rate(cell) >= 0`` (thermal ``kappa_a B``, or the fixed source ``S_n``
    of the lambda variant).  The returned density is exactly
    ``p(r) = rate(cell) / Q`` with ``Q = sum_c rate_c V_c`` (plan section 2).

    Parameters
    ----------
    n : int
        Number of birth points.
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
    rate : (n_cells,) ndarray
        Non-negative source rate per cell.
    method : {"categorical", "null_collision"}
        - ``"categorical"`` (default): draw the cell directly with probability
          ``rate_c V_c / Q`` and the position uniformly inside it — exactly the
          same density, no rejections.
        - ``"null_collision"``: the null-source-event sampler.  Proposals are
          uniform in the domain volume and are **accepted** with probability
          ``rate(cell) / rate_max`` (``rate_max = max rate``); a rejected
          proposal is a null source event and a fresh proposal is drawn
          (plan section 2: chained Woodcock birth sampling with null source
          events — on piecewise-constant cells this is the cell-level
          accept/reject form of that chain).
    seed : int
    volumes : (n_cells,) ndarray, optional
        Cell volumes (default rt3d.mc.estimators.cell_volumes).

    Returns
    -------
    dict with keys 'positions' (n, 3), 'cell' (n,), 'Q' (float, the rate
    integral), 'rate_max', 'n_null_source' (int, null source events — zero by
    construction in categorical mode), 'method', 'seed'.
    """
    from rt3d.mc.estimators import cell_volumes, n_cells_of

    rate = np.asarray(rate, dtype=float)
    nc = n_cells_of(grid)
    if rate.shape != (nc,):
        raise ValueError(f"rate must have shape ({nc},)")
    if np.any(rate < 0.0):
        raise ValueError("source rate must be non-negative")
    Q = float(rate @ cell_volumes(grid) if volumes is None else rate @ volumes)
    if not Q > 0.0:
        raise ValueError("source rate integrates to zero — no births to sample")

    rng = np.random.default_rng(seed)
    rate_max = float(rate.max())

    if method == "categorical":
        V = (cell_volumes(grid) if volumes is None else np.asarray(volumes, float))
        cells = rng.choice(nc, size=n, p=rate * V / Q)
        n_null = 0
    elif method == "null_collision":
        cells = np.full(n, -1, dtype=int)
        n_null = 0
        todo = np.arange(n)
        box = grid.box
        while todo.size:
            prop = np.column_stack(
                [rng.uniform(box[a][0], box[a][1], todo.size) for a in range(3)]
            )
            c = grid.locate(prop)
            acc = rng.random(todo.size) * rate_max < rate[c]
            cells[todo[acc]] = c[acc]
            n_null += int(np.count_nonzero(~acc))
            todo = todo[~acc]
    else:
        raise ValueError(f"unknown birth method {method!r}")

    # position inside the cell: uniform in the cell box
    if hasattr(grid, "cell_box"):          # unigrid
        lo, hi = grid.cell_box(cells)
        pos = lo + rng.random((n, 3)) * (hi - lo)
    else:                                  # octree leaf
        h = grid.half[cells][:, None]
        pos = grid.centers[cells] + (2.0 * rng.random((n, 3)) - 1.0) * h

    return {
        "positions": pos,
        "cell": cells,
        "Q": Q,
        "rate_max": rate_max,
        "n_null_source": int(n_null),
        "method": method,
        "seed": int(seed),
    }


def beam_seed(n, box, omega_sun, rng, eps=None):
    """
    Seed `n` direct-beam particles uniformly over the face the beam enters
    through (top face for ``omega_sun[2] < 0``, bottom face otherwise).

    The per-particle power of the resulting population is
    ``F0 * |omega_sun . face_normal| * A_face / n`` (plan section 5 weight
    convention; `face_area` gives ``A_face``).

    Returns
    -------
    positions : (n, 3) ndarray
    axis, side : int
        Entry face.
    mu : float
        ``|omega_sun[axis]|``.
    """
    d = np.asarray(omega_sun, dtype=float)
    d = d / np.linalg.norm(d)
    axis = 2 if abs(d[2]) > 0.0 else int(np.argmax(np.abs(d)))
    side = -1 if d[axis] > 0.0 else 1        # entering face: the far side
    pos = face_points(n, box, axis, side, rng, eps=eps)
    return pos, axis, side, float(abs(d[axis]))
