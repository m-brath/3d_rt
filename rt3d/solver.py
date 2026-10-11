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
    if is_octree:
        chains = [
            precompute_octree_march(grid, dg.directions[d],
                                    periodic_box=periodic_box)
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
    for it in range(n_iter):
        if ground is None:
            for d, chain in enumerate(chains):
                if is_octree:
                    I[:, d] = sweep_direction_octree_exact(
                        grid, chain, kappa_e_delta, S[:, d], bc
                    )
                else:
                    I[:, d] = sweep_direction(
                        grid, chain, kappa_e_delta, S[:, d], bc
                    )
        else:
            # downward ordinates first: they set the ground inflow
            for d in dg.down_idx:
                chain = chains[d]
                if is_octree:
                    I[:, d] = sweep_direction_octree_exact(
                        grid, chain, kappa_e_delta, S[:, d], bc
                    )
                else:
                    I[:, d] = sweep_direction(
                        grid, chain, kappa_e_delta, S[:, d], bc
                    )
            I_g = I_g_fixed.copy()
            I_g[mb] += ground.albedo * downward_irradiance(
                I[:, dg.down_idx], dg
            )[mb] / np.pi
            for d in dg.up_idx:
                chain = chains[d]
                if is_octree:
                    I[:, d] = sweep_direction_octree_exact(
                        grid, chain, kappa_e_delta, S[:, d], bc,
                        bottom_I=I_g[chain["entry_cell"]],
                    )
                else:
                    # ground radiance is a column field: chains with
                    # |s_z| > 1 root above the bottom layer, so the k = 0
                    # values must tile over all layers for bottom-entry
                    # roots
                    I[:, d] = sweep_direction(
                        grid, chain, kappa_e_delta, S[:, d], bc,
                        bottom_I=np.tile(I_g[: grid.ny * grid.nx], grid.nz),
                    )
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
    }
