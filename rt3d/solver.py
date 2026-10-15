# -*- coding: utf-8 -*-
"""
Iteration loop (plan section 5): plain lambda iteration.

    S0 = (1 - omega) * B + beam_scat
    repeat:
      1. transport sweep over all ordinates (chains built once)   # transport
      2. S_new = (1 - omega) B + omega * <Phi I> + beam_scat      # incoming rad.
      3. converge on relative change of S

Ng acceleration (1D Lambda_bar) is added in phase 4 and will slot in as an
alternative update of S between steps 2 and 3.
"""

import numpy as np

from rt3d.characteristics import build_chain
from rt3d.source import (
    beam_scattering_source,
    delta_extinction,
    flux_vector,
    mean_intensity,
    scattering_source,
    single_scattering_albedo,
)
from rt3d.transport import (
    beam_irradiance_ground,
    beam_optical_depth,
    beam_optical_depth_octree,
    bottom_mask,
    downward_irradiance,
    precompute_octree_march,
    sweep_direction,
    sweep_direction_octree_exact,
)


def lambda_iteration(
    grid,
    dg,
    kappa_e,
    kappa_s,
    B,
    bc,
    g=0.0,
    periodic_box=None,
    beam=None,
    n_iter=100,
    tol=1e-8,
    ng=None,
    delta_M=0,
    transport="chain",
    jit=None,
    n_jobs=1,
):
    """
    Plain lambda iteration for the monochromatic RTE.

    Parameters
    ----------
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
    dg : rt3d.directions.direction_grid
    kappa_e, kappa_s : (n_cells,) ndarray
        Extinction / scattering coefficient per cell.
    B : (n_cells,) ndarray
        Planck function (thermal emission); zeros for pure scattering runs.
    bc : rt3d.transport.boundary_conditions
    g : float or (n_cells,) ndarray, optional
        HG asymmetry parameter; per-cell values are bucketed for the
        scattering matmul (rt3d.source.scattering_source), by default 0.0.
    periodic_box : cuboid, optional
    beam : dict or None, optional
        Solar beam: {"omega_sun": (3,) unit direction, "F0": float}.
        Its first-scattering term enters the source as a fixed contribution.
    n_iter : int, optional
        Maximum number of iterations, by default 100.
    tol : float, optional
        Relative convergence tolerance on the source function, by default 1e-8.
    ng : None, bool or rt3d.accelerator.ng1d, optional
        Ng acceleration (rt3d.accelerator.ng1d): None/False = plain lambda
        iteration, True = build ng1d(grid), or pass a prepared ng1d
        instance.  It replaces the source update between steps 2 and 3
        (plan section 5); the convergence history still measures the
        plain-LI residual of the swept source.
    delta_M : int, optional
        Delta-M (Wiscombe) scaling order, by default 0 (off).  With
        M = delta_M >= 1 every diffuse transport/attenuation quantity
        uses the reduced extinction kappa_e - kappa_s * g**M
        (rt3d.source.delta_extinction - only the scattering part of the
        extinction is scaled, so pure absorption kappa_s = 0 is exactly
        unchanged) and the source uses the exact Legendre-moment split of
        rt3d.source.scattering_source (plan/delta-m-plan.md).
        The direct solar beam keeps its physical attenuation: tau_b and
        the ground beam irradiance use the physical kappa_e (the beam is
        not on the quadrature, the moment subtraction applies only to the
        diffuse field); only its first-scatter albedo factor becomes
        omega / (1 - omega * g**M).  The converged solution is identical
        to delta_M = 0 up to the short-characteristics discretization of
        the explicit -omega * g**M * I term (agreement improves ~linearly
        with cell size); iteration is markedly faster for strongly
        forward-peaked scattering and the effective diffuse optical
        depth changes.  M = 1 is recommended for g >~ 0.5.  Octree
        refinement is unaffected: it runs outside the solver on the
        physical kappa_e.
    transport : {"chain", "sc"}, optional
        Unigrid transport algorithm, by default "chain" (the lattice-
        stencil chain sweeps of rt3d.characteristics).  "sc" selects the
        independent single-cell short-characteristics sweeps of
        rt3d.short_characteristics (plan/plan-sc-unigrid.md): bilinear
        4-point face interpolation, wrap-around passes on periodic entry
        axes.  More diffusive on coarse grids (path error ~linear in
        cell size), but local and gather-only.  Ignored on the octree
        (exact marches); transport="sc" with an octree grid raises
        ValueError.
    jit : bool, optional
        Octree exact marches: None (default) -> fused numba march+sweep
        (rt3d.jit, plan/optimization-plan.md P1) when numba is available:
        no segment arrays are materialised and each sweep re-marches in
        compiled code parallel over rays.  False -> reference NumPy
        march with precomputed segment arrays.  Ignored on the unigrid.
    n_jobs : int, optional
        Number of threads for the per-ordinate sweep loop
        (plan/optimization-plan.md P3), by default 1 (sequential, the
        original behaviour).  The ordinates are independent; the threads
        share the read-only chain/march geometry and write disjoint
        output columns I[:, d].  Most useful for the unigrid chain/SC
        sweeps; the fused octree jit sweep is already parallel over rays
        (numba prange), so n_jobs > 1 adds little there.

    Returns
    -------
    result : dict with keys
        'I'        (n_cells, N_dir) converged center intensities
        'S'        (n_cells, N_dir) converged source function
        'J'        (n_cells,) mean intensity
        'F'        (n_cells, 3) flux vector
        'omega'    (n_cells,) single scattering albedo
        'history'  list of relative source changes per iteration
        'n_iter'   iterations performed
        'converged' bool
        'delta_M'  int, delta-M order used (0 = off)
        'kappa_e_delta' (n_cells,) extinction actually used for the
                   transport; equals kappa_e when delta_M == 0
        'transport' str, transport algorithm used ("chain" or "sc")
        'jit'     bool, fused numba octree march/sweep used on the octree
        'n_jobs'  int, thread count used for the per-ordinate sweeps
    """
    from rt3d.octree import octree_grid

    if ng is True:
        from rt3d.accelerator import ng1d

        ng = ng1d(grid)
    elif ng is False:
        ng = None

    kappa_e = np.asarray(kappa_e, dtype=float)
    kappa_s = np.asarray(kappa_s, dtype=float)
    B = np.asarray(B, dtype=float)
    omega = single_scattering_albedo(kappa_e, kappa_s)
    is_octree = isinstance(grid, octree_grid)
    n_cells = grid.n_leaf if is_octree else grid.n_cells
    g_cells = None if np.ndim(g) == 0 else np.asarray(g, dtype=float)

    # delta-M (Wiscombe) scaling: every diffuse transport/attenuation
    # quantity below uses the reduced extinction kappa_e - kappa_s * g**M
    # (only the scattering part is scaled); octree refinement runs
    # outside the solver on the physical kappa_e (intended design).
    if delta_M == 0:
        kappa_e_delta = kappa_e
    else:
        kappa_e_delta = delta_extinction(
            kappa_e, kappa_s, g if g_cells is None else g_cells, delta_M
        )

    if delta_M >= 1:
        # the delta split replaces the plain HG matrix; the cache is
        # keyed (g_value, delta_M) and built once, reused across iterations
        Phi_w = None
        Phi_cache = {}
    else:
        Phi_w = None if g_cells is not None else dg.hg_matrix(g)
        Phi_cache = {} if g_cells is not None else None

    # fixed source parts: thermal + beam first scattering (never iterated)
    S_fixed = (1.0 - omega) * B
    beam_scat = None
    if beam is not None:
        # the DIRECT beam attenuates with the physical kappa_e (it is not
        # on the quadrature, the moment subtraction applies only to the
        # diffuse field); the delta-M scaling of its first-scatter source
        # enters via beam_scattering_source(delta_M=...) as
        # kappa_s Phi F0 exp(-tau_b) / kappa_e_delta
        if is_octree:
            tau_b = beam_optical_depth_octree(
                grid, beam["omega_sun"], kappa_e, periodic_box=periodic_box
            )
        else:
            tau_b = beam_optical_depth(
                grid, beam["omega_sun"], kappa_e, periodic_box=periodic_box
            )
        beam_scat = beam_scattering_source(
            dg, beam["omega_sun"], beam["F0"], tau_b, omega,
            g if g_cells is None else 0.0, g_cells=g_cells, delta_M=delta_M,
        )

    # transport geometry depends only on grid + directions: build once.
    # The octree sweep follows exact characteristics (precomputed backward
    # marches); the approximate center-to-center chain (build_chain_octree)
    # showed non-converging path errors in heterogeneous 3D fields.
    # On the unigrid the solver option `transport` selects between the
    # lattice-stencil chains ("chain", default) and the independent
    # single-cell short-characteristics sweeps ("sc",
    # rt3d.short_characteristics, plan/plan-sc-unigrid.md).
    if transport not in ("chain", "sc"):
        raise ValueError(f"transport must be 'chain' or 'sc', got {transport!r}")
    if is_octree:
        if transport == "sc":
            raise ValueError(
                "transport='sc' is only available on the unigrid; the "
                "octree sweep follows exact characteristic marches"
            )
        chains = [
            precompute_octree_march(grid, dg.directions[d],
                                    periodic_box=periodic_box, jit=jit)
            for d in range(dg.n_dir)
        ]
    elif transport == "sc":
        from rt3d.short_characteristics import build_sc_chain

        chains = [
            build_sc_chain(grid, dg.directions[d], periodic_box=periodic_box)
            for d in range(dg.n_dir)
        ]
    else:
        chains = [
            build_chain(grid, dg.directions[d], periodic_box=periodic_box)
            for d in range(dg.n_dir)
        ]

    # Lambertian ground (phase 5): fixed inflow parts (emission + beam
    # reflection); the reflected diffuse part F_down is rebuilt from the
    # downward ordinates after every sweep.
    ground = getattr(bc, "ground", None)
    if ground is not None:
        if ground.is_variable():
            raise NotImplementedError(
                "variable ground surface: the transport supports only a "
                "flat ground at the domain bottom (plan section 1, "
                "phase 6 hook)"
            )
        mb = bottom_mask(grid)
        I_g_fixed = np.zeros(n_cells)
        I_g_fixed[mb] = ground.emissivity * ground.B_g
        if beam is not None:
            # direct-beam irradiance at the ground: physical attenuation
            # (tau_b and the bottom-cell extension use physical kappa_e)
            I_g_fixed[mb] += ground.albedo * beam_irradiance_ground(
                grid, beam["omega_sun"], kappa_e, tau_b, beam["F0"]
            ) / np.pi

    # start: no radiation yet -> S = fixed part only
    S = np.broadcast_to(S_fixed[:, None], (n_cells, dg.n_dir)).copy()
    if beam_scat is not None:
        S += beam_scat

    history = []
    converged = False
    I = np.zeros((n_cells, dg.n_dir))

    use_jit_sweeps = not is_octree
    if use_jit_sweeps:
        from rt3d.jit import HAVE_NUMBA

        use_jit_sweeps = HAVE_NUMBA

    def sweep_one(d, bottom_I=None):
        """One ordinate sweep with the selected transport method; the
        ground ordering, Ng, delta-M and the scattering matmul above are
        method-agnostic and shared."""
        if is_octree:
            return sweep_direction_octree_exact(
                grid, chains[d], kappa_e_delta, S[:, d], bc, bottom_I=bottom_I
            )
        if use_jit_sweeps:
            if transport == "sc":
                from rt3d.jit import sc_sweep_jit

                return sc_sweep_jit(grid, chains[d], kappa_e_delta, S[:, d],
                                    bc, bottom_I=bottom_I)
            from rt3d.jit import chain_sweep_jit

            return chain_sweep_jit(grid, chains[d], kappa_e_delta, S[:, d],
                                   bc, bottom_I=bottom_I)
        if transport == "sc":
            from rt3d.short_characteristics import sweep_direction_sc

            return sweep_direction_sc(
                grid, chains[d], kappa_e_delta, S[:, d], bc, bottom_I=bottom_I
            )
        return sweep_direction(
            grid, chains[d], kappa_e_delta, S[:, d], bc, bottom_I=bottom_I
        )

    def bottom_I_for(d, I_g):
        """Per-ordinate ground inflow (Lambertian ground, phase 5)."""
        if is_octree:
            return I_g[chains[d]["entry_cell"]]
        # ground radiance is a column field: chains with
        # |s_z| > 1 root above the bottom layer (and SC stencil
        # nodes clamp to k = 0), so the k = 0 values must tile
        # over all layers for bottom-entry inflow
        return np.tile(I_g[: grid.ny * grid.nx], grid.nz)

    def sweep_group(ds, I_g=None):
        """Sweep the ordinates in ds, filling I[:, d]; with n_jobs > 1
        in parallel threads (independent ordinates, disjoint columns)."""
        if n_jobs is not None and n_jobs > 1 and len(ds) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(n_jobs, len(ds))) as ex:
                futs = [
                    (d, ex.submit(sweep_one, d,
                                  None if I_g is None else bottom_I_for(d, I_g)))
                    for d in ds
                ]
                for d, fut in futs:
                    I[:, d] = fut.result()
        else:
            for d in ds:
                I[:, d] = sweep_one(
                    d, None if I_g is None else bottom_I_for(d, I_g)
                )

    for it in range(n_iter):
        if ground is None:
            sweep_group(range(dg.n_dir))
        else:
            # downward ordinates first: they set the ground inflow
            sweep_group(dg.down_idx)
            I_g = I_g_fixed.copy()
            I_g[mb] += ground.albedo * downward_irradiance(
                I[:, dg.down_idx], dg
            )[mb] / np.pi
            sweep_group(dg.up_idx, I_g)
        S_new = scattering_source(
            I, dg, omega, B, Phi_w=Phi_w, g=g if g_cells is None else 0.0,
            beam_scat=beam_scat, g_cells=g_cells, Phi_cache=Phi_cache,
            delta_M=delta_M,
        )
        scale = max(np.max(np.abs(S_new)), 1e-30)
        rel = np.max(np.abs(S_new - S)) / scale
        history.append(rel)
        S = ng.step(S, S_new) if ng is not None else S_new
        if rel < tol:
            converged = True
            break

    return {
        "I": I,
        "S": S,
        "J": mean_intensity(I, dg),
        "F": flux_vector(I, dg),
        "omega": omega,
        "history": history,
        "n_iter": len(history),
        "converged": converged,
        "delta_M": int(delta_M),
        "kappa_e_delta": kappa_e_delta,
        "transport": transport,
        "jit": bool(is_octree and any(
            c.get("fused", False) for c in chains)),
        "n_jobs": int(n_jobs),
    }
