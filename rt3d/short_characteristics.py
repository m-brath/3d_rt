# -*- coding: utf-8 -*-
"""
Short characteristics (SC) on the unigrid - a second, independent transport
algorithm alongside the lattice-stencil chain sweeps of rt3d.characteristics
(plan/plan-sc-unigrid.md, DOIT-style).

Single-cell short characteristics on the unigrid: per ordinate Omega and
target cell c the characteristic runs from the upwind layer through the
entry face into the target center in two segments of equal length

    t_f = h_a / (2 |Omega_a|),   a = argmax_a |Omega_a| / h_a,

(1) a *gap segment* from each of the four upwind stencil nodes along
+Omega by t_f - it lands exactly in the plane of the entry face and stays
inside the node's own cell (t_f |Omega_b| <= h_b / 2) - and (2) the
*target segment* from the face point p = c_center - t_f * Omega to the
target center.  Both segments are integrated with the piecewise-constant
formal solution, and the four propagated node values are combined by
bilinear 4-point interpolation at p:

    I_c = exp(-kappa_e,c t_f) * sum_m w_m * Ihat_m
          + S_c (1 - exp(-kappa_e,c t_f)),
    Ihat_m = I_{u_m} exp(-kappa_e,u_m t_f) + S_{u_m} (1 - exp(-kappa_e,u_m t_f)).

The gap segment (1) is essential: without it the four bilinear nodes all
lie strictly upwind of p, the interpolation degenerates to an
extrapolation, and the face -> upwind-center half cell is never
integrated - the transmittance error is then O(1) and *scale-invariant*
(measured: tau halved for z-entry ordinates, unchanged under refinement).
With the gap segment the z-entry characteristics reproduce exp(-tau) to
machine precision; oblique (x/y-entry) ordinates keep a bounded angular
stencil error of the same character as the chain method's grazing-ordinate
error (plan/plan-sc-unigrid.md section 6).

On a uniform grid the entry face, the face point and the bilinear stencil
are the *same* for every cell, so the whole sweep is a layer loop along the
entry axis a, vectorized over the layer plane.
Because a maximises |Omega_a|/h_a, the lateral offset of p obeys

    |p - center|_lateral <= h_b / 2,

so p always lies inside the 4-ring of cell centers around the upwind cell u
in the face plane - the bilinear stencil is always well-defined, also on
anisotropic grids (hz != hx). No ordinate has Omega_z = 0 (direction-grid
rotation), so every characteristic chain terminates in z.

Sweep ordering: the whole stencil lies in the already-computed upwind layer.
Entry axis z (non-periodic) is trivially well-ordered. Entry axis x or y
(periodic, ~2/3 of the icosahedral ordinates) forms a dependency *ring*
across the wrap; it is handled by wrap-around passes - the ring is swept
repeatedly until the change at the wrap boundary drops below a tolerance,
capped at max(4 n_a, 64) passes. The composed ring operator has spectral
radius < 1: every stencil application drifts in z (the z-bilinear weight of
the same-layer node is < 1 because Omega_z != 0), and z is non-periodic, so
repeated ring circuits either attenuate (kappa_e > 0) or leave through the
top/bottom boundary.  With kappa_e > 0 somewhere in the ring the circuit
gain carries an extra exp(-2 kappa t_f n_a) and 1-3 passes suffice; in
near-vacuum rings with grazing ordinates the gain (1 - s_z)**n_a can
approach 1 and many passes are needed - the pass cap then truncates the
multi-circuit contribution (documented trade-off).
"""

import numpy as np

from rt3d.transport import _sweep_dirs


def sc_stencil(omega, h):
    """
    Single-cell short-characteristics stencil for one ordinate.

    Parameters
    ----------
    omega : (3,) ndarray
        Unit ordinate direction (propagation direction of the intensity).
        omega_a = 0 on at most two axes; omega_z != 0 expected (chains
        terminate in z).
    h : (3,) array_like
        Cell sizes (hx, hy, hz).

    Returns
    -------
    stencil : dict with keys
        'a'        int    entry axis = argmax_a |omega_a| / h_a
        'sigma'    int    +/-1, sign of omega along the entry axis
        't_f'      float  center -> entry-face distance h_a / (2 |omega_a|)
        'offsets'  (4, 3) int  the four stencil nodes as offsets *relative
                         to the target cell c*: node = c + offsets[m].
                         offsets[:, a] = -sigma (upwind layer), lateral
                         offsets in {-1, 0, +1}
        'weights'  (4,) float  bilinear weights, sum == 1
    """
    omega = np.asarray(omega, dtype=float)
    h = np.asarray(h, dtype=float)

    # entry axis: largest |omega_a| / h_a (axes with omega_a = 0 skipped)
    ratio = np.where(omega != 0.0, np.abs(omega) / h, -1.0)
    a = int(np.argmax(ratio))
    if omega[a] == 0.0:
        raise ValueError(f"ordinate {omega} has no nonzero direction component")
    sigma = 1 if omega[a] > 0.0 else -1
    t_f = h[a] / (2.0 * abs(omega[a]))

    lateral = [ax for ax in range(3) if ax != a]
    b, c = lateral

    # face point p = center - t_f * omega; lateral offsets from the upwind
    # cell center (u differs from c only along axis a, so the lateral
    # offsets are the same): |delta| <= h/2 because a maximises |w|/h
    delta_b = -t_f * omega[b]
    delta_c = -t_f * omega[c]
    s_b = delta_b / h[b]
    s_c = delta_c / h[c]

    # bilinear nodes/weights in the face plane (s in [-1/2, 1/2] -> the
    # nodes are the 4-ring around u)
    nb_lo = int(np.floor(s_b))
    nc_lo = int(np.floor(s_c))
    wb_hi = s_b - nb_lo
    wc_hi = s_c - nc_lo
    wb = np.array([1.0 - wb_hi, 1.0 - wb_hi, wb_hi, wb_hi])
    wc = np.array([1.0 - wc_hi, wc_hi, 1.0 - wc_hi, wc_hi])

    offsets = np.zeros((4, 3), dtype=int)
    offsets[:, a] = -sigma
    offsets[:, b] = [nb_lo, nb_lo, nb_lo + 1, nb_lo + 1]
    offsets[:, c] = [nc_lo, nc_lo + 1, nc_lo, nc_lo + 1]
    weights = wb * wc

    return {
        "a": a,
        "sigma": sigma,
        "t_f": float(t_f),
        "offsets": offsets,
        "weights": weights,
    }


def build_sc_chain(grid, omega, periodic_box=None):
    """
    Per-ordinate SC precompute on the unigrid (same reuse convention as
    rt3d.characteristics.build_chain).

    Parameters
    ----------
    grid : rt3d.geometry.unigrid
    omega : (3,) ndarray
        Unit ordinate direction.
    periodic_box : rt3d.geometry.cuboid, optional
        If given, x/y stencil nodes wrap; only z exits carry boundary faces.

    Returns
    -------
    sc : dict with keys
        'omega'      (3,) float ordinate direction
        'a'          int    entry axis
        'sigma'      int    +/-1 propagation sense along the entry axis
        't_f'        float  segment length (center -> entry face)
        'weights'    (4,) float bilinear stencil weights
        'order'      (n_cells,) int cell flat indices, grouped by layer in
                       sweep order (upwind boundary layer first)
        'slices'     list of (start, stop) into 'order', one per layer
                       along the entry axis, in sweep order
        'node_idx'   (4, n_cells) int stencil-node flat indices, aligned
                       with 'order'; out-of-domain nodes are clamped to the
                       domain (bottom nodes clamp to k = 0 so their column
                       index stays valid for the ground bottom_I)
        'node_face'  (4, n_cells) int8: 0 = interior (use I[node_idx]),
                       1 = top face, 2 = bottom face, 3 = x/y side face
        'wrap'       bool True when the entry axis is periodic (x/y): the
                       layer loop must run wrap-around passes
        'n_layers'   int number of layers along the entry axis (base for
                       the wrap-pass cap, see sweep_direction_sc)
    """
    h = np.array([grid.hx, grid.hy, grid.hz])
    st = sc_stencil(omega, h)
    a, sigma, t_f = st["a"], st["sigma"], st["t_f"]
    offsets, weights = st["offsets"], st["weights"]

    dims = np.array((grid.nx, grid.ny, grid.nz))
    n_a = int(dims[a])
    lateral = [ax for ax in range(3) if ax != a]
    periodic = periodic_box is not None

    # sweep order along the entry axis: upwind boundary layer first
    layers = np.arange(n_a) if sigma > 0 else np.arange(n_a - 1, -1, -1)

    order_parts = []
    idx_parts = [[] for _ in range(4)]
    face_parts = [[] for _ in range(4)]
    for L in layers:
        g1, g2 = np.meshgrid(
            np.arange(dims[lateral[0]]), np.arange(dims[lateral[1]]), indexing="ij"
        )
        coords = np.empty((g1.size, 3), dtype=int)
        coords[:, a] = L
        coords[:, lateral[0]] = g1.ravel()
        coords[:, lateral[1]] = g2.ravel()
        order_parts.append(grid.ravel(coords[:, 0], coords[:, 1], coords[:, 2]))

        for m in range(4):
            nc = coords + offsets[m]
            if periodic:
                nc[:, 0] = np.mod(nc[:, 0], grid.nx)
                nc[:, 1] = np.mod(nc[:, 1], grid.ny)
            face = np.zeros(nc.shape[0], dtype=np.int8)
            if not periodic:
                side_out = (
                    (nc[:, 0] < 0) | (nc[:, 0] >= grid.nx)
                    | (nc[:, 1] < 0) | (nc[:, 1] >= grid.ny)
                )
                face[side_out] = 3
            # z faces take precedence over side faces at domain corners
            face[nc[:, 2] < 0] = 2
            face[nc[:, 2] >= grid.nz] = 1
            nc = np.clip(nc, 0, dims - 1)
            idx_parts[m].append(grid.ravel(nc[:, 0], nc[:, 1], nc[:, 2]))
            face_parts[m].append(face)

    order = np.concatenate(order_parts)
    slices = []
    pos = 0
    for part in order_parts:
        slices.append((pos, pos + part.size))
        pos += part.size

    return {
        "omega": np.asarray(omega, dtype=float),
        "a": a,
        "sigma": sigma,
        "t_f": t_f,
        "weights": weights,
        "order": order,
        "slices": slices,
        "node_idx": np.stack([np.concatenate(p) for p in idx_parts]),
        "node_face": np.stack([np.concatenate(p) for p in face_parts]),
        "wrap": bool(periodic and a != 2),
        "n_layers": n_a,
    }


def sweep_direction_sc(grid, sc, kappa_e, S, bc, bottom_I=None, wrap_tol=1e-12,
                       max_passes=None):
    """
    One SC transport sweep for one ordinate.

    Parameters
    ----------
    grid : unigrid
    sc : dict from build_sc_chain
    kappa_e : (n_cells,) ndarray
        Extinction coefficient per cell.  Both the target segment (inside
        cell c) and the node gap segments (inside the node cells) enter
        the recursion with their own cell's kappa_e.
    S : (n_cells,) ndarray
        Source function per cell (piecewise constant on the segments).
    bc : rt3d.transport.boundary_conditions
        Stencil nodes outside the domain take bc.top_I / bottom_I / side_I.
    bottom_I : (n_cells,) ndarray, optional
        Per-cell inflow replacing bc.bottom_I for stencil nodes below the
        bottom face (Lambertian ground radiance, phase 5).  Same tiling
        convention as sweep_direction: the k = 0 values tile over all
        layers, so the clamped node index picks the entry column.
    wrap_tol : float, optional
        Relative tolerance on the wrap-boundary change between ring passes
        (periodic x/y entry axes), by default 1e-12.
    max_passes : int, optional
        Safety cap on the number of ring passes; None -> max(4 * n_a, 64).
        The plan's original cap of n_a passes rests on a pure-shift
        exactness argument that does not hold for the bilinear 4-point
        stencil: the ring-circuit operator is non-normal and its vacuum
        gain (1 - s_z)**n_a approaches 1 for grazing ordinates, so
        convergence can need several times n_a passes.  With kappa_e > 0
        in the ring the gain carries an extra exp(-2 kappa t_f n_a) and
        1-3 passes suffice; the cap only binds in near-vacuum rings.

    Returns
    -------
    I : (n_cells,) ndarray
        Intensity at all cell centers along this ordinate.
    """
    n = grid.n_cells
    I = np.zeros(n)

    tau = kappa_e * sc["t_f"]
    E = np.exp(-tau)
    src = S * (-np.expm1(-tau))

    weights = sc["weights"]
    order = sc["order"]
    nidx = sc["node_idx"]
    nface = sc["node_face"]

    def sweep_pass():
        for s0, s1 in sc["slices"]:
            cells = order[s0:s1]
            acc = np.zeros(s1 - s0)
            for m in range(4):
                idx = nidx[m][s0:s1]
                face = nface[m][s0:s1]
                # gap segment: propagate the node value along +Omega by
                # t_f through the node's own cell (lands in the entry-face
                # plane); same formal solution as the target segment
                vals = I[idx] * E[idx] + src[idx]
                if np.any(face):
                    # boundary nodes carry the inflow at the domain face
                    # itself - no gap propagation
                    vals = np.where(face == 1, bc.top_I, vals)
                    if bottom_I is None:
                        vals = np.where(face == 2, bc.bottom_I, vals)
                    else:
                        vals = np.where(face == 2, bottom_I[idx], vals)
                    vals = np.where(face == 3, bc.side_I, vals)
                acc += weights[m] * vals
            I[cells] = acc * E[cells] + src[cells]

    if not sc["wrap"]:
        sweep_pass()
        return I

    # periodic entry axis: dependency ring across the wrap. Each pass
    # resolves one more ring circuit of the dependency chain (Gauss-Seidel
    # along the ring); stop when the wrap feed (the last layer computed,
    # which feeds the first layer of the next pass) stops changing.
    if max_passes is None:
        max_passes = max(4 * sc["n_layers"], 64)
    feed_prev = None
    ls, le = sc["slices"][-1]
    for _ in range(max_passes):
        sweep_pass()
        feed = I[order[ls:le]]
        if feed_prev is not None:
            scale = max(np.max(np.abs(feed)), 1e-30)
            if np.max(np.abs(feed - feed_prev)) <= wrap_tol * scale:
                break
        feed_prev = feed.copy()
    return I


def sweep_all_sc(grid, direction_grid, kappa_e, S, bc, periodic_box=None,
                 scs=None, wrap_tol=1e-12, max_passes=None, n_jobs=1):
    """
    SC transport over all ordinates (same signature/return convention as
    rt3d.transport.sweep_all).

    Parameters
    ----------
    S : (n_cells,) or (n_cells, N_dir) ndarray
        Source function; direction-dependent if 2D.
    scs : list of build_sc_chain dicts, optional
        Prebuilt per-ordinate precomputes to reuse across iterations;
        built here if None.
    n_jobs : int, optional
        Threads for the per-ordinate sweeps (P3), by default 1.  With
        numba available (auto-detected) each SC sweep runs as a compiled
        per-ordinate kernel (bit-identical to the NumPy recursion);
        n_jobs > 1 runs those kernels on a thread pool.

    Returns
    -------
    I : (n_cells, N_dir) ndarray
        Intensities at cell centers, ordinates along axis 1.
    scs : list of sc dicts (one per ordinate), for reuse/diagnostics
    """
    I = np.zeros((grid.n_cells, direction_grid.n_dir))
    if scs is None:
        scs = [
            build_sc_chain(grid, direction_grid.directions[d],
                           periodic_box=periodic_box)
            for d in range(direction_grid.n_dir)
        ]
    S_is_dir = np.ndim(S) == 2
    from rt3d.jit import HAVE_NUMBA, sc_sweep_jit

    def one(d):
        s_d = S[:, d] if S_is_dir else S
        if HAVE_NUMBA:
            I[:, d] = sc_sweep_jit(grid, scs[d], kappa_e, s_d, bc,
                                   wrap_tol=wrap_tol, max_passes=max_passes)
        else:
            I[:, d] = sweep_direction_sc(grid, scs[d], kappa_e, s_d, bc,
                                         wrap_tol=wrap_tol,
                                         max_passes=max_passes)

    _sweep_dirs(one, direction_grid.n_dir, n_jobs)
    return I, scs
