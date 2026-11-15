# -*- coding: utf-8 -*-
"""
Solver B — backward (adjoint) detector MC (plan/mc-plan.md section 4).

Estimates the radiance ``I(r_det, Omega)`` **at a detector** — a point or
aperture on/inside the domain viewing along ``omega0`` within a cone — by
launching photon histories *backwards* from the detector with the same
Woodcock tracker as the forward solvers.  Where solver A answers "what does
this source field produce everywhere", solver B answers "what reaches this
detector", concentrating the sampling budget on the sensor — the headline
argument for the adjoint family (plan section 4, efficiency figure).

Estimator (collision-density adjoint walk, deviation from the plan sketch
logged in plan/plan.md): an adjoint history starts at the detector and flies
along ``-Omega``; at every **real** collision of the closed-RTE walk it
scores

    w * [ (1 - omega) B(r) + omega(r) F0 e^{-tau_b(r)} Phi_HG(Omega_sun -> Omega_f) ]

with the *forward* intensity direction ``Omega_f = -Omega*`` (HG reciprocity:
the adjoint scattering kernel of the azimuthally symmetric HG is the same HG
about the reversed direction), then **continues** with weight ``w = 1`` when
the event was a scattering (density ``kappa_s``) and terminates after true
absorption (density ``kappa_a``); when the walk exits the domain it scores
the boundary inflow ``I_in`` of the face it leaves.  Because the real-event
density is ``kappa_e e^{-tau}``, the scores carry no ``kappa_e`` factor:
``kappa_e (1 - omega) B = kappa_a B`` (thermal) and ``kappa_e omega Phi =
kappa_s Phi`` (beam first scatter) come out exactly.  The multiple scattering
of the diffuse field is carried by the walk continuation — **not** by an
``omega * J`` score (the plan sketch's ``omega J`` at every collision would
re-expand the already-scattered stage-1 field at all orders and
over-count; the closed-RTE adjoint needs no stage-1 ``J`` at all).  This
makes solver B an *independent* solve of the same closed problem that
`rt3d.mc.forward.j_mc` / `rt3d.solver.lambda_iteration` solve, and it is
validated directly against their radiance fields.

Detector spec (plan section 4):

    detector = {"point": (x, y, z)}                    # point detector
               or {"plane": {"z": z, "rect": ((x0, x1), (y0, y1))}}
               + "omega0": (3,) unit viewing direction (into the domain the
                 adjoint seeds fly along -omega0 ... omega0 is the measured
                 intensity direction),
               + "cone": half angle [rad] (0 or None -> exact direction),
               + "n_seeds": number of adjoint histories

Uniform aperture + cone sampling makes the output the aperture- and
cone-**averaged radiance** (the ``Delta-omega`` cone weight is carried by
the uniform-in-cap pdf, so no extra factor appears — plan section 4).
`backward_all_sky` runs one detector cone per upward icosahedral ordinate
(half angle from the ordinate's solid-angle weight) and returns the all-sky
image differencable against the stage-2 / SC radiance columns.

Not implemented (M3 scope): Lambertian ground BC (`bc.ground` raises),
variance reduction (M4).
"""

import numpy as np

from rt3d.directions import hg
from rt3d.mc.sources import hg_scatter_directions
from rt3d.mc.tracking import (
    EV_EXIT,
    EV_SCATTER,
    _exit_distance,
    _path_cap,
    sample_collisions,
)

__all__ = ["backward_detector_mc", "backward_all_sky", "all_sky_cone_half"]


def _cone_directions(omega0, cone, n, rng):
    """Uniform directions inside the spherical cap of half-angle `cone`
    about `omega0` (pdf 1/(2 pi (1 - cos cone))); cone <= 0 -> exact."""
    om = np.asarray(omega0, dtype=float)
    om = om / np.linalg.norm(om)
    if cone is None or cone <= 0.0:
        return np.tile(om, (n, 1))
    mu = 1.0 - rng.random(n) * (1.0 - np.cos(cone))
    s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    psi = 2.0 * np.pi * rng.random(n)
    # orthonormal basis (e1, e2, om)
    ref = np.array([0.0, 0.0, 1.0]) if abs(om[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(om, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(om, e1)
    return (mu[:, None] * om[None, :]
            + s[:, None] * (np.cos(psi)[:, None] * e1[None, :]
                            + np.sin(psi)[:, None] * e2[None, :]))


def detector_seeds(detector, rng):
    """Seed positions and adjoint flight directions (= -omega0 sampled in
    the cone) of one detector population."""
    n = int(detector["n_seeds"])
    om0 = np.asarray(detector["omega0"], dtype=float)
    if "point" in detector:
        pos = np.tile(np.asarray(detector["point"], dtype=float), (n, 1))
    elif "plane" in detector:
        pl = detector["plane"]
        z = float(pl["z"])
        (x0, x1), (y0, y1) = pl["rect"]
        pos = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n),
                               np.full(n, z)])
    else:
        raise ValueError("detector needs a 'point' or a 'plane' aperture")
    d = _cone_directions(-om0, detector.get("cone"), n, rng)
    return pos, d


def backward_detector_mc(
    grid,
    kappa_e,
    kappa_s,
    B,
    bc,
    detector,
    g=0.0,
    beam=None,
    tau_b=None,
    periodic_box=None,
    n_batches=100,
    seed=0,
    mode="cell",
    kappa_max=None,
    max_collisions=20000,
    path_cap=None,
):
    """
    Adjoint (backward) radiance at one detector — plan/mc-plan.md section 4.

    Parameters
    ----------
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
    kappa_e, kappa_s : (n_cells,) ndarray
        Extinction / scattering coefficient fields (the closed problem's).
    B : (n_cells,) ndarray
        Planck function (thermal source side of the adjoint score).
    bc : rt3d.transport.boundary_conditions
        Face inflows scored when an adjoint walk exits the domain; a
        Lambertian `ground` raises NotImplementedError (M3 scope).
    detector : dict
        See the module docstring; `n_histories` is ``detector['n_seeds']``.
    g : float or (n_cells,) ndarray
        HG asymmetry parameter — adjoint continuation kernel *and* the beam
        phase function of the score (HG reciprocity, same g).
    beam : dict or None
        {"omega_sun": (3,) unit direction, "F0": float}: the beam
        first-scatter term enters the collision score with the **continuous**
        HG phase ``omega F0 e^{-tau_b} Phi_HG(Omega_sun . Omega_f)`` (MC
        samples angles exactly; the SC quadrature-rescaled Phi differs by the
        42-ordinate error only).
    tau_b : (n_cells,) ndarray, optional
        Beam optical depth; computed via rt3d.transport.beam_optical_depth(_octree)
        when `beam` is set and tau_b is None.
    periodic_box, mode, kappa_max, max_collisions, path_cap
        As in the forward solvers / `rt3d.mc.tracking.sample_collisions`.
    n_batches : int
        History batches for sigma_I (seeds grouped in blocks).
    seed : int
        PCG64 base seed, recorded in the output.

    Returns
    -------
    dict with keys
        'I', 'sigma_I' : float
            Cone/aperture-averaged radiance at the detector and its 1-sigma
            batch standard error.
        'estimator' (='adjoint_walk'), 'mode', 'kappa_max', 'seed',
        'n_histories', 'n_batches', 'n_generations', 'n_collisions',
        'n_scatter', 'n_null', 'n_exited', 'n_absorbed', 'capped',
        'n_truncated'   statistics & traceability
    """
    from rt3d.geometry import ray_box_exit
    from rt3d.octree import octree_grid
    from rt3d.source import single_scattering_albedo
    from rt3d.transport import beam_optical_depth, beam_optical_depth_octree

    if getattr(bc, "ground", None) is not None:
        raise NotImplementedError(
            "MC backward implements the face inflows of boundary_conditions; "
            "the self-coupled Lambertian ground is not part of M3 "
            "(plan/mc-plan.md)"
        )

    is_octree = isinstance(grid, octree_grid)
    kappa_e = np.asarray(kappa_e, dtype=float)
    kappa_s = np.asarray(kappa_s, dtype=float)
    B = np.asarray(B, dtype=float)
    n_cells = grid.n_leaf if is_octree else grid.n_cells
    if kappa_e.shape != (n_cells,) or kappa_s.shape != (n_cells,) \
            or B.shape != (n_cells,):
        raise ValueError(f"kappa_e/kappa_s/B must have shape ({n_cells},)")
    omega = single_scattering_albedo(kappa_e, kappa_s)

    if beam is not None:
        om_sun = np.asarray(beam["omega_sun"], dtype=float)
        om_sun = om_sun / np.linalg.norm(om_sun)
        F0 = float(beam["F0"])
        if tau_b is None:
            tau_b = (
                beam_optical_depth_octree(grid, om_sun, kappa_e,
                                          periodic_box=periodic_box)
                if is_octree
                else beam_optical_depth(grid, om_sun, kappa_e,
                                        periodic_box=periodic_box)
            )
        tau_b = np.asarray(tau_b, dtype=float)
    else:
        om_sun, F0, tau_b = None, 0.0, None

    rng = np.random.default_rng(seed)
    pos, d = detector_seeds(detector, rng)
    K = pos.shape[0]

    # adjoint flight direction d: the measured intensity direction is -d;
    # scores evaluate the source/emission for the forward direction -d.
    score = np.zeros(K)
    n_coll = np.zeros(K, dtype=np.int64)
    n_scat = np.zeros(K, dtype=np.int64)
    n_null = np.zeros(K, dtype=np.int64)
    exited = np.zeros(K, dtype=bool)

    live = np.arange(K)
    cap = _path_cap(path_cap, grid)
    n_trunc = 0
    gen = 0
    while live.size and gen < int(max_collisions):
        pl, dl = pos[live], d[live]
        raw = _exit_distance(pl, dl, grid, periodic_box=periodic_box)
        budget = np.minimum(raw, cap)

        res = sample_collisions(pl, dl, budget, grid, kappa_e, kappa_s,
                                mode=mode, kappa_max=kappa_max,
                                periodic_box=periodic_box,
                                seed=seed + 2_000_003 * gen + 1)
        ev = res["coll_event"]
        n_coll[live] += (ev != EV_EXIT).astype(np.int64)
        n_null[live] += res["n_null"]

        real = ev != EV_EXIT
        if np.any(real):
            cell = res["coll_cell"][real]
            om_f = -dl[real]                      # forward intensity direction
            sv = (1.0 - omega[cell]) * B[cell]
            if beam is not None:
                cos_sc = np.clip(om_f @ om_sun, -1.0, 1.0)
                g_ev = float(g) if np.ndim(g) == 0 \
                    else np.asarray(g, dtype=float)[cell]
                sv = sv + omega[cell] * F0 * np.exp(-tau_b[cell]) * hg(cos_sc, g_ev)
            score[live[real]] += sv

        # --- exits: boundary inflow of the face the walk leaves
        exit_ev = ev == EV_EXIT
        if np.any(exit_ev):
            il = live[exit_ev]
            te = res["exit_t"][exit_ev]
            dlx = dl[exit_ev]
            if periodic_box is None:
                tf, ax, side = ray_box_exit(pl[exit_ev], dlx, grid.box)
                reached = te >= tf - 1e-9 * max(grid.hx, grid.hy, grid.hz)
                I_in = np.where(reached, bc.inflow(ax, side), 0.0)
            else:
                side_up = dlx[:, 2] > 0.0
                reached = te >= raw[exit_ev] - 1e-9 * grid.box[2][1]
                I_in = np.where(reached,
                                np.where(side_up, bc.top_I, bc.bottom_I), 0.0)
            score[il] += I_in
            exited[il] = reached
            n_trunc += int(np.count_nonzero(~reached))

        # --- scattering: adjoint continuation (weight unchanged);
        # true absorption terminates (score already deposited), as does an
        # exit or a path-cap truncation
        scat = ev == EV_SCATTER
        live = live[scat]
        if live.size:
            isl = live
            j = np.flatnonzero(scat)
            n_scat[isl] += 1
            cell_s = res["coll_cell"][j]
            pos[isl] = pl[j] + res["coll_distance"][j, None] * dl[j]
            if periodic_box is not None:
                pos[isl] = periodic_box.wrap_xy(pos[isl])
            d[isl] = hg_scatter_directions(dl[j], cell_s, g, rng)
        gen += 1

    capped = np.zeros(K, dtype=bool)
    capped[live] = True
    n_capped = int(np.count_nonzero(capped)) + n_trunc

    # absorbed histories: terminated at a true absorption (score deposited)
    n_absorbed = int(K - np.count_nonzero(exited) - n_capped)

    nb = max(1, min(int(n_batches), K))
    per = K // nb
    if nb > 1 and per * nb == K:
        bs = score.reshape(nb, per).sum(axis=1)
        est = float(bs.sum() / K)
        sig = float(np.sqrt(nb) * bs.std(ddof=1) / K)
    else:
        nb = 1
        est = float(score.mean())
        sig = 0.0

    return {
        "I": est,
        "sigma_I": sig,
        "estimator": "adjoint_walk",
        "mode": mode,
        "kappa_max": kappa_max,
        "seed": int(seed),
        "n_histories": K,
        "n_batches": nb if per * nb == K else 1,
        "n_generations": gen,
        "n_collisions": int(n_coll.sum()),
        "n_scatter": int(n_scat.sum()),
        "n_null": int(n_null.sum()),
        "n_exited": int(np.count_nonzero(exited)),
        "n_absorbed": n_absorbed,
        "n_capped": n_capped,
        "n_truncated": n_trunc,
    }


def all_sky_cone_half(dg, i):
    """Half-angle of the equal-area spherical cap matching the solid-angle
    weight w_i of icosahedral ordinate i: 2 pi (1 - cos theta) = w_i."""
    return float(np.arccos(max(-1.0, 1.0 - dg.weights[i] / (2.0 * np.pi))))


def backward_all_sky(grid, kappa_e, kappa_s, B, bc, dg, point, g=0.0,
                     beam=None, tau_b=None, periodic_box=None,
                     n_seeds=20000, n_batches=100, seed=0, mode="cell",
                     kappa_max=None, cone_scale=0.0):
    """
    All-sky radiance at `point`: one point detector per **upward**
    icosahedral ordinate (viewing direction = ordinate).  The cone
    half-angle is `cone_scale` x the equal-area cap of the ordinate weight;
    the default 0.0 (exact node direction) is the mode that differences
    directly against stage-2 / SC ``I[cell, ordinate]`` columns.  Full
    icosahedral caps (`cone_scale = 1`) are full-sphere apertures: near the
    horizon they legitimately pick up top-lit DOWNWARD radiance (the cap
    crosses the horizon), so cone-averaged values are not comparable to
    hemisphere-only quadrature node values (measured ~0.3 vs ~0.02 at
    mu ~ 0.1 on the cloud) — keep `cone_scale` small for differencing.
    Returns dict with per-ordinate 'I'/'sigma_I' arrays and the ordinate
    indices ('up_idx'), plus the batch statistics of each run.
    """
    up_idx = np.flatnonzero(np.asarray(dg.directions)[:, 2] > 0.0)
    I = np.zeros(up_idx.size)
    sI = np.zeros(up_idx.size)
    diag = []
    for k, i in enumerate(up_idx):
        det = {
            "point": tuple(np.asarray(point, dtype=float)),
            "omega0": np.asarray(dg.directions[i], dtype=float),
            "cone": cone_scale * all_sky_cone_half(dg, i),
            "n_seeds": int(n_seeds),
        }
        r = backward_detector_mc(grid, kappa_e, kappa_s, B, bc, det, g=g,
                                 beam=beam, tau_b=tau_b,
                                 periodic_box=periodic_box,
                                 n_batches=n_batches, seed=seed + 1 + k,
                                 mode=mode, kappa_max=kappa_max)
        I[k] = r["I"]
        sI[k] = r["sigma_I"]
        diag.append({"n_iter": r["n_generations"], "n_null": r["n_null"],
                     "n_capped": r["n_capped"]})
    return {"I": I, "sigma_I": sI, "up_idx": up_idx, "runs": diag,
            "n_seeds": int(n_seeds), "seed": int(seed), "mode": mode}
