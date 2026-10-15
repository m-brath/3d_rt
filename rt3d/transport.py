# -*- coding: utf-8 -*-
"""
Short-characteristics transport sweep (plan section 4) and the dedicated
solar-beam sweep (plan section 1).

Transport recursion per characteristic, exact for piecewise-constant
sources along the sub-segments:

    I_c = I_u * exp(-tau_tot)
          + sum_sub S_sub * (1 - exp(-tau_sub)) * exp(-tau_after)

Boundary conditions for phase 1 are constant intensities on top / bottom /
side faces; the ground reflection and periodic-BC machinery arrives in
phase 5 and plugs into `boundary_conditions`.
"""

import numpy as np

from rt3d.characteristics import build_chain, build_chain_octree
from rt3d.geometry import march_lattice, ray_box_exit


def _sweep_dirs(fn, n_dir, n_jobs):
    """
    Call fn(d) for d in range(n_dir); with n_jobs > 1 in parallel threads
    (plan/optimization-plan.md P3).  The ordinates are independent; fn
    must write disjoint outputs.  numpy releases the GIL for the large
    elementwise/gather loops of the sweeps, so threads scale there.
    """
    if n_jobs is None or n_jobs <= 1 or n_dir <= 1:
        for d in range(n_dir):
            fn(d)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(int(n_jobs), n_dir)) as ex:
        list(ex.map(fn, range(n_dir)))


class boundary_conditions(object):
    """
    Background intensities on the domain faces + optional ground (phase 5).

    Attributes
    ----------
    top_I : float
        Incoming intensity for ordinates entering through the top face.
    bottom_I : float
        Incoming intensity for ordinates entering through the bottom face
        (used when there is no ground).
    side_I : float
        Incoming intensity for ordinates entering through x/y side faces
        (ignored when the sides are periodic).
    ground : rt3d.geometry.ground_surface or None
        Lambertian + emissive ground at the bottom face.  When set, the
        bottom inflow is replaced by the ground radiance
        `eps_g * B_g + A_g * (F_down + F0 |mu_sun| exp(-tau_b)) / pi`
        evaluated at the ray entry point (phase 5).
    """

    def __init__(self, top_I=0.0, bottom_I=0.0, side_I=0.0, ground=None):
        self.top_I = float(top_I)
        self.bottom_I = float(bottom_I)
        self.side_I = float(side_I)
        self.ground = ground

    def inflow(self, exit_axis, exit_side):
        """Inflow intensity for a chain that entered through the face the
        backward march exited through (arrays -> array)."""
        I = np.where(
            exit_axis == 2,
            np.where(exit_side > 0, self.top_I, self.bottom_I),
            self.side_I,
        )
        return I


def bottom_mask(grid):
    """Cells/leaves touching the bottom face (their columns define the
    ground-level quantities)."""
    from rt3d.octree import octree_grid

    if isinstance(grid, octree_grid):
        z0 = grid.box[2][0]
        return grid.centers[:, 2] - grid.half <= z0 + 1e-12 * grid.h0
    m = np.zeros(grid.n_cells, dtype=bool)
    m[: grid.ny * grid.nx] = True  # k = 0 layer (k-major ordering)
    return m


def downward_irradiance(I, dg):
    """
    Downward irradiance sum_{omega_z<0} w |omega_z| I per cell.

    I : (n_cells, N_dir) or (n_cells, n_down) ndarray restricted to the
        downward ordinates (dg.down_idx order).
    """
    mu = np.abs(dg.directions[dg.down_idx, 2])
    return I @ (dg.weights[dg.down_idx] * mu)


def ground_inflow(F_down, ground, beam_irr=0.0):
    """
    Lambertian ground radiance: eps_g B_g + A_g (F_down + beam_irr) / pi.

    F_down : (n_bottom,) downward irradiance at ground level
    beam_irr : (n_bottom,) | scalar, direct-beam irradiance at the ground
    """
    return (
        ground.emissivity * ground.B_g
        + ground.albedo * (F_down + beam_irr) / np.pi
    )


def beam_irradiance_ground(grid, omega_sun, kappa_e, tau_b, F0,
                           periodic_box=None):
    """
    Direct-beam irradiance F0 |mu_sun| exp(-tau_b_ground) at the ground
    below each bottom-face cell center.  tau_b is the beam optical depth
    at cell centers; it is extended to the ground face through the
    bottom cell itself (kappa_e of that cell over the remaining path).
    """
    omega_sun = np.asarray(omega_sun, dtype=float)
    mu = abs(omega_sun[2])
    mb = bottom_mask(grid)
    z = grid.centers[mb, 2]
    z0 = grid.box[2][0]
    tau_g = tau_b[mb] + kappa_e[mb] * (z - z0) / mu
    return F0 * mu * np.exp(-tau_g)


def sweep_direction(grid, chain, kappa_e, S, bc, bottom_I=None):
    """
    One transport sweep for one ordinate.

    Parameters
    ----------
    grid : unigrid
    chain : dict from rt3d.characteristics.build_chain
    kappa_e : (n_cells,) ndarray
        Extinction coefficient per cell.
    S : (n_cells,) ndarray
        Source function per cell (constant along each sub-segment).
    bc : boundary_conditions
    bottom_I : (n_cells,) ndarray, optional
        Per-cell inflow replacing bc.bottom_I for chains entering through
        the bottom face (Lambertian ground radiance, phase 5).

    Returns
    -------
    I : (n_cells,) ndarray
        Intensity at all cell centers along this ordinate.
    """
    n = grid.n_cells
    I = np.zeros(n)

    roots = chain["roots"]
    parent = chain["parent"]
    rel = chain["rel"]
    sub_delta = chain["sub_delta"]
    periodic = chain["periodic"]
    sub_len = chain["sub_len"]
    nx, ny, nxny = grid.nx, grid.ny, grid.nx * grid.ny

    # --- boundary-entry chains: accumulate from the entry point toward the
    # root center. root_steps is step-major from the center backwards, so
    # the "tau_after" factor of a segment is the prefix sum of the segments
    # already visited (they lie between this segment and the center).
    tau_acc = np.zeros(roots.size)
    src_acc = np.zeros(roots.size)
    rs_r, rs_c, rs_l = chain["root_steps"]
    for ridx, cell, ln in zip(rs_r, rs_c, rs_l):
        tau = kappa_e[cell] * ln
        src_acc[ridx] += S[cell] * (-np.expm1(-tau)) * np.exp(-tau_acc[ridx])
        tau_acc[ridx] += tau
    I_in = bc.inflow(chain["root_exit_axis"], chain["root_exit_side"])
    if bottom_I is not None:
        # ground: per-cell inflow for chains entering through the bottom
        # face (evaluated at the root cell center - consistent with the
        # lattice-stencil path approximation)
        bottom_entry = (
            (chain["root_exit_axis"] == 2) & (chain["root_exit_side"] < 0)
        )
        I_in = np.where(bottom_entry, bottom_I[roots], I_in)
    I[roots] = I_in * np.exp(-tau_acc) + src_acc

    # --- interior cells, BFS waves from the roots
    for wave in chain["waves"]:
        if periodic:
            wi0 = wave % nx
            wj0 = (wave // nx) % ny
            wk0 = wave // nxny
        rows = []
        taus = []
        for m in range(sub_len.shape[0]):
            if periodic:
                cells_m = ((wk0 + rel[m, 2]) * nxny
                           + ((wj0 + rel[m, 1]) % ny) * nx
                           + (wi0 + rel[m, 0]) % nx)
            else:
                cells_m = wave + sub_delta[m]
            rows.append(cells_m)
            taus.append(kappa_e[cells_m] * sub_len[m])
        tau_tot = np.zeros(wave.size)
        for t in taus:
            tau_tot += t
        contrib = np.zeros(wave.size)
        prefix = np.zeros(wave.size)
        for m, tau in enumerate(taus):
            prefix += tau
            cells_m = rows[m]
            contrib += S[cells_m] * (-np.expm1(-tau)) * np.exp(-(tau_tot - prefix))
        I[wave] = I[parent[wave]] * np.exp(-tau_tot) + contrib

    return I


def sweep_all(grid, direction_grid, kappa_e, S, bc, periodic_box=None,
              chains=None, n_jobs=1):
    """
    Transport over all ordinates.

    Parameters
    ----------
    S : (n_cells,) or (n_cells, N_dir) ndarray
        Source function; direction-dependent if 2D (anisotropic scattering).
    chains : list of chain dicts, optional
        Prebuilt chains from build_chain (one per ordinate) to reuse across
        iterations; built here if None.
    n_jobs : int, optional
        Threads for the per-ordinate sweeps (P3), by default 1.  With
        numba available (auto-detected) each sweep runs as a compiled
        per-ordinate kernel (bit-identical to the NumPy recursion, and
        faster even single-threaded); n_jobs > 1 runs those kernels on a
        thread pool (njit releases the GIL; the NumPy gathers hold it,
        so plain threads over the NumPy sweeps do not scale).  Without
        numba the NumPy sweeps run, sequentially.

    Returns
    -------
    I : (n_cells, N_dir) ndarray
        Intensities at cell centers, ordinates along axis 1.
    chains : list of chain dicts (one per ordinate), for reuse/diagnostics
    """
    I = np.zeros((grid.n_cells, direction_grid.n_dir))
    if chains is None:
        chains = [
            build_chain(grid, direction_grid.directions[d], periodic_box=periodic_box)
            for d in range(direction_grid.n_dir)
        ]
    S_is_dir = np.ndim(S) == 2
    from rt3d.jit import HAVE_NUMBA, chain_sweep_jit

    def one(d):
        s_d = S[:, d] if S_is_dir else S
        if HAVE_NUMBA:
            I[:, d] = chain_sweep_jit(grid, chains[d], kappa_e, s_d, bc)
        else:
            I[:, d] = sweep_direction(grid, chains[d], kappa_e, s_d, bc)

    _sweep_dirs(one, direction_grid.n_dir, n_jobs)
    return I, chains


def beam_optical_depth(grid, omega_sun, kappa_e, periodic_box=None):
    """
    Beam optical depth tau_b at every cell center: integrate kappa_e along
    the backward beam ray from the cell center to the domain entry point.

    Parameters
    ----------
    omega_sun : (3,) ndarray
        Unit beam propagation direction (must have omega_z < 0).
    kappa_e : (n_cells,) ndarray
    periodic_box : cuboid, optional

    Returns
    -------
    tau_b : (n_cells,) ndarray
    """
    omega_sun = np.asarray(omega_sun, dtype=float)
    if omega_sun[2] >= 0.0:
        raise ValueError("the solar beam must point downward (omega_z < 0)")

    steps, _, _, _ = march_lattice(
        grid.centers, -omega_sun, grid, periodic_box=periodic_box
    )
    tau = np.zeros(grid.n_cells)
    for ridx, cell, ln in steps:
        tau[ridx] += kappa_e[cell] * ln
    return tau


# =============================================================================
# octree transport (phase 3)
# =============================================================================


def _block_prefixes(tau_seg, offsets):
    """
    Per-segment cumulative optical depths within node-major blocks:
    (prefix_before, prefix_through) where prefix_through includes the
    segment itself.
    """
    counts = np.diff(offsets)
    cs = np.cumsum(tau_seg)
    n_blocks = offsets.size - 1
    prev = np.zeros(n_blocks)
    if cs.size:
        starts = offsets[:-1]
        has_prev = starts > 0
        prev[has_prev] = cs[starts[has_prev] - 1]
    prev_rep = np.repeat(prev, counts)
    return cs - tau_seg - prev_rep, cs - prev_rep


def sweep_direction_octree(oct, chain, kappa_e, S, bc):
    """
    One transport sweep for one ordinate on the octree. Same recursion as
    sweep_direction, but with per-node sub-segment blocks from
    build_chain_octree.
    """
    n = oct.n_leaf
    I = np.zeros(n)

    # --- roots: segments ordered center -> entry; the attenuation of a
    # source segment to the leaf center is exp(-prefix_before)
    sr, sl, sLen, offs = chain["root_seg"]
    R = chain["roots"].size
    tau_seg = kappa_e[sl] * sLen
    tau_tot = np.zeros(R)
    np.add.at(tau_tot, sr, tau_seg)
    pre_before, _ = _block_prefixes(tau_seg, offs)
    src = np.zeros(R)
    np.add.at(src, sr, S[sl] * (-np.expm1(-tau_seg)) * np.exp(-pre_before))
    I_in = bc.inflow(chain["root_exit_axis"], chain["root_exit_side"])
    I[chain["roots"]] = I_in * np.exp(-tau_tot) + src

    # --- interior leaves, BFS waves: segments ordered parent -> leaf;
    # attenuation from a segment to the leaf center is exp(-(tau_tot -
    # prefix_through)).  tau_tot / contrib depend only on the source, not
    # on I, so they are computed once per sweep.
    sr2, sl2, sLen2, offs2 = chain["sub_seg"]
    tau_seg = kappa_e[sl2] * sLen2
    tau_tot = np.zeros(n)
    np.add.at(tau_tot, sr2, tau_seg)
    _, pre_thru = _block_prefixes(tau_seg, offs2)
    contrib = np.zeros(n)
    np.add.at(
        contrib,
        sr2,
        S[sl2] * (-np.expm1(-tau_seg)) * np.exp(-(tau_tot[sr2] - pre_thru)),
    )
    for wave in chain["waves"]:
        I[wave] = I[chain["parent"][wave]] * np.exp(-tau_tot[wave]) + contrib[wave]

    return I


def precompute_octree_march(oct, omega, periodic_box=None, jit=None):
    """
    Exact characteristic march for one ordinate: march one ray backwards
    from every leaf center along -omega to the domain entry point.

    The segments depend only on the geometry, not on the radiation field,
    so they are computed once per ordinate and reused by every transport
    sweep.  Unlike the approximate center-to-center chain
    (build_chain_octree), the ray follows the true characteristic exactly,
    so the only path error is the octree's own field resolution.

    Parameters
    ----------
    oct : rt3d.octree.octree_grid
    omega : (3,) array
        Ordinate direction.  omega_z must be nonzero (the icosahedral
        grids have no ordinate on the equator).
    periodic_box : rt3d.geometry.cuboid, optional
    jit : bool, optional
        None (default) -> use the fused numba march+sweep (rt3d.jit) when
        numba is available: the returned dict carries 'fused' = True and
        only the per-ray geometry (budget, entry_cell); the segment arrays
        below are NOT materialised and sweep_direction_octree_exact
        re-marches each sweep in compiled parallel code
        (plan/optimization-plan.md P1).  False -> reference NumPy march
        with explicit segment arrays.

    Returns
    -------
    march : dict with keys
        'leaf'    (M,) int    leaf index of each segment, ray-contiguous
                              blocks ordered center -> entry
        'length'  (M,) float  segment lengths
        'offsets' (n_leaf + 1,) int  block offsets into leaf/length
        'ray_of'  (M,) int    ray (= leaf) index of each segment
        'omega'   (3,) ndarray ordinate direction
        'entry_cell' (n_leaf,) int or None, for omega_z > 0: bottom-face
                              leaf containing the ray's ground entry
                              point (for the Lambertian ground BC)
        fused mode only: 'fused' True, 'budget' (n_leaf,) float,
        'entry_cell', 'omega', 'periodic_box'
    """
    from rt3d.octree import march_octree

    omega = np.asarray(omega, dtype=float)
    if omega[2] == 0.0:
        raise ValueError("exact octree march needs omega_z != 0")
    z = oct.centers[:, 2]
    z0, z1 = oct.box[2]
    if omega[2] < 0.0:
        budget = (z1 - z) / (-omega[2])
        I_in = None  # filled from bc at sweep time
    else:
        budget = (z - z0) / omega[2]

    if jit is None:
        from rt3d.jit import jit_usable

        jit = jit_usable(oct, periodic_box)
    if jit:
        entry_cell = None
        if omega[2] > 0.0:
            p = oct.centers - budget[:, None] * omega[None, :]
            x0, x1 = oct.box[0]
            y0, y1 = oct.box[1]
            p[:, 0] = x0 + np.mod(p[:, 0] - x0, x1 - x0)
            p[:, 1] = y0 + np.mod(p[:, 1] - y0, y1 - y0)
            p[:, 2] = z0
            entry_cell = oct.locate(p)
        return {
            "fused": True,
            "omega": omega,
            "budget": np.ascontiguousarray(budget),
            "entry_cell": entry_cell,
            "periodic_box": periodic_box,
        }

    sr, sl, sLen = march_octree(oct.centers, -omega, oct, budget,
                                periodic_box=periodic_box)
    # stable sort by ray keeps traversal order (center -> entry) per ray
    order = np.argsort(sr, kind="stable")
    leaf = sl[order].astype(np.int32)
    length = sLen[order]
    counts = np.bincount(sr, minlength=oct.n_leaf)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    ray_of = np.repeat(np.arange(oct.n_leaf, dtype=np.int32), counts)

    entry_cell = None
    if omega[2] > 0.0:
        # ground entry point of each backward ray (x, y wrapped into the
        # periodic box); locate() clamps z to the bottom face, so the
        # result is the bottom-face leaf of the entry column
        p = oct.centers - budget[:, None] * omega[None, :]
        x0, x1 = oct.box[0]
        y0, y1 = oct.box[1]
        p[:, 0] = x0 + np.mod(p[:, 0] - x0, x1 - x0)
        p[:, 1] = y0 + np.mod(p[:, 1] - y0, y1 - y0)
        p[:, 2] = z0
        entry_cell = oct.locate(p)

    return {
        "leaf": leaf,
        "length": length,
        "offsets": offsets,
        "ray_of": ray_of,
        "omega": omega,
        "entry_cell": entry_cell,
    }


def _entry_I_in(oct, omega, bc):
    """Inflow intensity of the face the ray enters through."""
    return bc.top_I if omega[2] < 0.0 else bc.bottom_I


def sweep_direction_octree_exact(oct, march, kappa_e, S, bc, bottom_I=None):
    """
    One transport sweep for one ordinate on the octree, along the exact
    characteristic from precompute_octree_march.

    Segments within each ray block are ordered center -> entry, so the
    attenuation of a segment contribution to the leaf center is
    exp(-prefix_before) (same recursion as the chain roots).

    bottom_I : (n_leaf,) ndarray, optional
        Per-leaf inflow for upward ordinates (omega_z > 0), replacing
        bc.bottom_I: the Lambertian ground radiance evaluated at each
        ray's ground entry point via march['entry_cell'] (phase 5).
    """
    if march.get("fused", False):
        from rt3d.jit import sweep_direction_octree_exact_jit

        if march["omega"][2] > 0.0 and bottom_I is not None:
            # bottom_I is already the per-ray inflow (the caller indexes
            # the ground radiance through march['entry_cell'] — same
            # convention as the segment path below)
            I_in = np.ascontiguousarray(bottom_I)
        else:
            I_in = np.full(oct.n_leaf,
                           _entry_I_in(oct, march["omega"], bc))
        return sweep_direction_octree_exact_jit(
            oct, march["omega"], march["budget"], kappa_e, S, I_in,
            periodic_box=march["periodic_box"],
        )

    leaf = march["leaf"]
    tau_seg = kappa_e[leaf] * march["length"]
    pre_before, pre_thru = _block_prefixes(tau_seg, march["offsets"])

    tau_tot = np.zeros(oct.n_leaf)
    np.add.at(tau_tot, march["ray_of"], tau_seg)

    I = np.zeros(oct.n_leaf)
    np.add.at(
        I,
        march["ray_of"],
        S[leaf] * (-np.expm1(-tau_seg)) * np.exp(-pre_before),
    )
    if march["omega"][2] > 0.0 and bottom_I is not None:
        I += bottom_I * np.exp(-tau_tot)
    else:
        I_in = _entry_I_in(oct, march["omega"], bc)
        I += I_in * np.exp(-tau_tot)
    return I


def sweep_all_octree_exact(oct, direction_grid, kappa_e, S, bc,
                           periodic_box=None, marches=None, n_jobs=1):
    """
    Transport over all ordinates on the octree along exact characteristics.

    S : (n_leaf,) or (n_leaf, N_dir) ndarray
    marches : list of precompute_octree_march dicts, optional (built
        once, reused across iterations).
    n_jobs : int, optional
        Threads for the per-ordinate sweeps (P3), by default 1.  With the
        fused numba march the sweep is already parallel over rays, so
        n_jobs > 1 mainly helps the reference NumPy marches.
    """
    I = np.zeros((oct.n_leaf, direction_grid.n_dir))
    if marches is None:
        marches = [
            precompute_octree_march(oct, direction_grid.directions[d],
                                    periodic_box=periodic_box)
            for d in range(direction_grid.n_dir)
        ]
    S_is_dir = np.ndim(S) == 2

    def one(d):
        s_d = S[:, d] if S_is_dir else S
        I[:, d] = sweep_direction_octree_exact(oct, marches[d], kappa_e, s_d, bc)

    _sweep_dirs(one, direction_grid.n_dir, n_jobs)
    return I, marches


def sweep_all_octree(oct, direction_grid, kappa_e, S, bc, periodic_box=None,
                     chains=None, n_jobs=1):
    """
    Transport over all ordinates on the octree.

    S : (n_leaf,) or (n_leaf, N_dir) ndarray
    chains : list of octree chain dicts, optional (built once, reused).
    n_jobs : int, optional
        Threads for the per-ordinate sweeps (P3), by default 1.
    """
    I = np.zeros((oct.n_leaf, direction_grid.n_dir))
    if chains is None:
        chains = [
            build_chain_octree(oct, direction_grid.directions[d],
                               periodic_box=periodic_box)
            for d in range(direction_grid.n_dir)
        ]
    S_is_dir = np.ndim(S) == 2

    def one(d):
        s_d = S[:, d] if S_is_dir else S
        I[:, d] = sweep_direction_octree(oct, chains[d], kappa_e, s_d, bc)

    _sweep_dirs(one, direction_grid.n_dir, n_jobs)
    return I, chains


def beam_optical_depth_octree(oct, omega_sun, kappa_e, periodic_box=None,
                              jit=None):
    """
    Beam optical depth at every leaf center on the octree.

    jit : bool, optional
        None (default) -> fused numba march when available (rt3d.jit),
        False -> reference NumPy march.
    """
    from rt3d.octree import march_octree

    omega_sun = np.asarray(omega_sun, dtype=float)
    if omega_sun[2] >= 0.0:
        raise ValueError("the solar beam must point downward (omega_z < 0)")
    z = oct.centers[:, 2]
    z0, z1 = oct.box[2]
    if periodic_box is not None:
        budget = (z1 - z) / (-omega_sun[2])
    else:
        budget, _, _ = ray_box_exit(oct.centers, -omega_sun, oct.box)
    if jit is None:
        from rt3d.jit import jit_usable

        jit = jit_usable(oct, periodic_box)
    if jit:
        from rt3d.jit import beam_optical_depth_octree_jit

        return beam_optical_depth_octree_jit(
            oct, omega_sun, kappa_e, budget, periodic_box=periodic_box
        )
    sr, sl, sLen = march_octree(oct.centers, -omega_sun, oct, budget,
                                periodic_box)
    tau = np.zeros(oct.n_leaf)
    np.add.at(tau, sr, kappa_e[sl] * sLen)
    return tau
