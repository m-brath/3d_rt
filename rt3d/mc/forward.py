# -*- coding: utf-8 -*-
"""
Solver A, stage 1 — the mean-intensity / source field by Monte Carlo
(plan/mc-plan.md section 3, phase M1).

Entry point `j_mc(grid, kappa_e, kappa_s, B, bc, g, ...) -> J, sigma_J`
solves the same monochromatic LTE problem as the short-characteristics
`solver.lambda_iteration`, but with MC transport instead of the ordinate
sweeps.  Stage 1 has **no angular quadrature at all**: the icosahedral
direction grid enters only as tally binning in stage 2 / the backward solver,
so `j_mc` takes no direction grid and no `n_refine`.

Two selectable variants (plan section 3):

``stage1="lambda"`` — the DOIT twin of `lambda_iteration`.  The transport
operator ``Lambda`` is applied to the current scalar source ``S_n(r)``:
particles are born in the volume and fly with **removal only** (no particle
scattering — the scattering is done by the outer iteration), their track
lengths give ``J_n``, and ``S_{n+1} = (1 - omega) B + omega J_n`` plus the
fixed boundary / beam first-scatter contributions.  Convergence is measured on
the source, exactly as in the SC solver.

``stage1="explicit"`` — classic generation-based MC: physical scattering
regenerates the diffuse field inside the particle history
(`rt3d.mc.tracking.track_histories`), so there is no outer iteration.

Estimator and weight conventions: `rt3d.mc.estimators.track_tally`
(``J_c = sum_k w_k ell_{c,k} / (4 pi V_c)``).  The boundary condition is a
particle population seeded on the inflow faces (cosine-weighted directions,
``w = pi A I_in / n``) and the solar beam is a particle stream seeded on its
entry face (plan section 2), not an analytic source term — the analytic
`rt3d.source.beam_scattering_source` and ``tau_b`` stay as cross-checks.

Direct / diffuse split: the SC solver transports the *diffuse* field only
(``I = I_diff + I_beam``), so MC ``J`` is the diffuse mean intensity as well.
The lambda variant reproduces that bookkeeping exactly — the beam contributes
only through its first-scatter children and the uncollided beam track is never
tallied.  The explicit variant follows the un-split particle physics, so its
unscattered beam tracks are tallied into a separate 'J_direct' and reported
apart from 'J'.

Not implemented in M1 (plan phase scope): the Lambertian + emissive ground
(`bc.ground` raises `NotImplementedError`), implicit capture / Russian
roulette (M4), stage-2 radiance (M2).
"""

import numpy as np

from rt3d.mc.estimators import cell_volumes, heating_rate, n_cells_of, track_tally
from rt3d.mc.sources import (
    beam_seed,
    face_area,
    face_inflow_directions,
    face_points,
    hg_scatter_directions,
    isotropic_directions,
    sample_births,
)
from rt3d.mc.tracking import (
    EV_EXIT,
    _exit_distance,
    flight_budget,
    flight_segments,
    sample_collisions,
    track_histories,
    walk_lattice,
)
from rt3d.source import single_scattering_albedo


def _inflow_faces(bc, periodic_box):
    """
    Faces carrying a constant inflow intensity: list of (axis, side, I_in).
    Periodic x/y faces are skipped (`bc.side_I` is ignored there — same
    convention as `rt3d.transport.boundary_conditions.inflow`).
    """
    faces = []
    if bc.top_I != 0.0:
        faces.append((2, 1, bc.top_I))
    if bc.bottom_I != 0.0:
        faces.append((2, -1, bc.bottom_I))
    if bc.side_I != 0.0:
        per = {
            0: periodic_box is not None and periodic_box.periodic_x,
            1: periodic_box is not None and periodic_box.periodic_y,
        }
        for axis in (0, 1):
            if not per[axis]:
                faces.append((axis, -1, bc.side_I))
                faces.append((axis, 1, bc.side_I))
    return faces


def _single_flight(starts, directions, grid, kappa_e, mode="cell",
                   kappa_max=None, periodic_box=None, seed=0, path_cap=None):
    """
    One removal-only flight per history — the fixed-source transport operator
    of the lambda variant.  The first *real* event is the removal (null
    collisions leave flight and direction unchanged), so the same tracker code
    serves both majorant modes.

    Returns the segment records of the flights (history index, cell, length,
    plus the per-history direction) and the collision statistics.
    """
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    d = np.asarray(directions, dtype=float)
    if d.ndim == 1:
        d = np.tile(d, (starts.shape[0], 1))
    budget = flight_budget(starts, d, grid, periodic_box=periodic_box,
                           path_cap=path_cap)
    res = sample_collisions(starts, d, budget, grid, kappa_e,
                            np.zeros_like(kappa_e), mode=mode,
                            kappa_max=kappa_max, periodic_box=periodic_box,
                            seed=seed)
    ends = res["exit_t"]
    sh, sc, sl = flight_segments(starts, d, ends, grid, periodic_box=periodic_box)
    return {
        "hist": sh, "cell": sc, "len": sl, "dirs_hist": d,
        "n_histories": starts.shape[0],
        "n_null": int(np.sum(res["n_null"])),
        "coll_cell": res["coll_cell"],
        "coll_distance": res["coll_distance"],
        "coll_event": res["coll_event"],
        "survived": res["survived"],
    }


def _concat_populations(populations, n_cells, n_batches, flux):
    """
    Build one track tally from a list of ``(record, per_history_weight)``
    populations.  History indices are re-offset so that each population owns
    its own slice of the batch index space, which keeps history batching
    meaningful across populations.
    """
    hist, cell, ln, wseg, dseg = [], [], [], [], []
    off = 0
    for rec, w in populations:
        n = rec["n_histories"]
        if n == 0 or rec["hist"].size == 0:
            continue
        h = rec["hist"]
        hist.append(h + off)
        cell.append(rec["cell"])
        ln.append(rec["len"])
        wseg.append(np.asarray(w, dtype=float)[h])
        dseg.append(rec["dirs_hist"][h])
        off += n
    tally = track_tally(n_cells, max(1, off), n_batches, flux=flux)
    if hist:
        tally.add(np.concatenate(hist), np.concatenate(cell),
                  np.concatenate(ln), np.concatenate(wseg),
                  np.concatenate(dseg))
    return tally


def _beam_first_scatter(grid, omega_sun, F0, kappa_e, omega, g, n_histories,
                        mode="cell", kappa_max=None, periodic_box=None, seed=0,
                        path_cap=None):
    """
    Diffuse first-scatter population of the direct beam (plan section 2:
    "beam as a particle stream").

    Each primary carries the beam power ``w_p = F0 |Omega_sun . n_face| A / n``
    and flies the beam direction; its **first** collision is sampled at the
    total rate ``kappa_e`` — uncollided-flux sampling, so the collision pdf
    ``kappa_e exp(-kappa_e s) ds`` *is* the Beer-Lambert attenuation and
    exactly one diffuse child per primary is born at that point, with power
    ``w_p * omega(cell)`` in an HG direction about ``Omega_sun``.  The child
    production density is therefore ``w_p kappa_s exp(-tau_b)`` per unit
    path — the first-scattering source ``kappa_s Phi(->Omega_sun) F0
    exp(-tau_b)`` of the SC solver with the HG pattern sampled continuously
    instead of on the ordinate set.  The primary is then done: any deeper
    collision orders are generated by the outer lambda iteration through the
    ``omega * J`` source update (exactly as the SC solver re-scatters
    ``beam_scat``), not by continuing the primary, which would tally the
    first-scatter source unattenuated.  The primary's own track is never
    tallied: the direct beam is not part of the diffuse ``J``.

    Returns a single-flight record set for the children with the spawn power
    per child ('child_w') and the splitting-loop diagnostics.
    """
    d = np.asarray(omega_sun, dtype=float)
    d = d / np.linalg.norm(d)
    pos, axis, side, mu = beam_seed(n_histories, grid.box, d,
                                    np.random.default_rng(seed))
    wp = float(F0) * mu * face_area(grid.box, axis) / n_histories

    # first collision of every primary: one removal-only flight at kappa_e
    prim = _single_flight(pos, d, grid, kappa_e, mode=mode, kappa_max=kappa_max,
                          periodic_box=periodic_box, seed=seed + 1,
                          path_cap=path_cap)
    hit = ~prim["survived"]
    n_null = prim["n_null"]
    n_lost = int(np.count_nonzero(prim["survived"]))   # primaries that left without colliding

    if not np.any(hit):
        empty = np.zeros(0, dtype=np.int64)
        return {"hist": empty, "cell": empty, "len": np.zeros(0),
                "dirs_hist": np.zeros((0, 3)), "n_histories": 0,
                "n_null": n_null, "child_w": np.zeros(0), "n_generations": 0,
                "n_spawned": 0, "n_lost": n_lost}

    cc = prim["coll_cell"][hit]
    cp = pos[hit] + prim["coll_distance"][hit, None] * d
    if periodic_box is not None:
        cp = periodic_box.wrap_xy(cp)
    cw = wp * omega[cc]

    rng = np.random.default_rng(seed + 7)
    cdir = hg_scatter_directions(np.tile(d, (cp.shape[0], 1)), cc, g, rng)

    rec = _single_flight(cp, cdir, grid, kappa_e, mode=mode,
                         kappa_max=kappa_max, periodic_box=periodic_box,
                         seed=seed + 91, path_cap=path_cap)
    rec["child_w"] = cw
    rec["n_null"] += n_null
    rec["n_generations"] = 1
    rec["n_spawned"] = int(cp.shape[0])
    rec["n_lost"] = n_lost
    return rec


def _domain_points(n, grid, rng):
    """`n` points uniform in the domain box volume."""
    return np.column_stack(
        [rng.uniform(grid.box[a][0], grid.box[a][1], n) for a in range(3)]
    )


def j_mc(
    grid,
    kappa_e,
    kappa_s,
    B,
    bc,
    g=0.0,
    periodic_box=None,
    beam=None,
    stage1="lambda",
    n_histories=20000,
    n_boundary_histories=20000,
    n_beam_histories=20000,
    n_batches=100,
    seed=0,
    mode="cell",
    kappa_max=None,
    n_iter=200,
    tol=1e-8,
    ng=None,
    crn=False,
    birth="source",
    max_collisions=20000,
    path_cap=None,
    tally_flux=True,
    implicit_capture=False,
    roulette_floor=None,
    jit=False,
):
    """
    MC stage 1: mean intensity ``J`` (and flux ``F``) of the diffuse field.

    Parameters
    ----------
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
    kappa_e, kappa_s : (n_cells,) ndarray
        Extinction / scattering coefficient per cell (same guard as the SC
        solver: ``omega = kappa_s / kappa_e``, ``omega = 0`` in vacuum).
    B : (n_cells,) ndarray
        Planck function (thermal emission), may be zeros.
    bc : rt3d.transport.boundary_conditions
        Face inflows ``top_I`` / ``bottom_I`` / ``side_I`` (periodic faces are
        skipped).  A Lambertian `ground` is not part of M1 and raises.
    g : float or (n_cells,) ndarray
        HG asymmetry parameter (sampled exactly by the HG inverse-CDF, so MC
        has no 42-ordinate resolution limit for ``g > 0.5``).
    periodic_box : rt3d.geometry.cuboid, optional
    beam : dict or None, optional
        Solar beam ``{"omega_sun": (3,) unit direction, "F0": float}``,
        handled as a particle stream (see `_beam_first_scatter`).
    stage1 : {"lambda", "explicit"}
        Outer-iteration (fixed-source transport) or generation-based MC.
    n_histories : int
        Volume-source histories: thermal births for "explicit", source births
        per iteration for "lambda".
    n_boundary_histories, n_beam_histories : int
        Histories of the face-inflow and direct-beam populations.
    n_batches : int
        History batches for the error bars (plan section 5).
    seed : int
        Base PCG64 seed; recorded in the output.
    mode : {"cell", "global"}, kappa_max : float, optional
        Woodcock majorant mode of the tracker (plan section 2).
    n_iter, tol : int, float
        Iteration cap / relative source convergence criterion of the "lambda"
        variant (same definition as `rt3d.solver.lambda_iteration`).
    ng : None, bool or rt3d.accelerator.ng1d
        Optional 1D Ng acceleration of the "lambda" variant — **opt-in and
        experimental**, it requires ``crn=True`` because the ``Lambda_bar``
        history pairs only see correlated noise when the photon stream is
        frozen (plan section 3).
    crn : bool
        Common random numbers for the "lambda" variant: birth points, flight
        directions and flight paths are drawn **once** and re-weighted by
        ``kappa_e * S_n`` every iteration, which makes ``Lambda`` a fixed
        (noisy) linear operator — with mild ``kappa_e`` variation (slabs) the
        iteration then converges cleanly to ``tol`` and Ng applies.  **Use
        with care in inhomogeneous media**: with a frozen uniform-birth photon
        set the effective sample count per high-``kappa_e`` cell is small, and
        on the Gaussian cloud the noisy frozen operator was measured to have
        a spectral radius above 1 — the iteration diverges (``s_max_history``
        exposes this).  Default ``crn=False`` samples a fresh photon set per
        iteration (plain LI): unconditionally stable, but the residual stalls
        at the tally noise floor instead of reaching ``tol``.
    birth : {"source", "uniform"}
        Birth sampling of the "lambda" variant's volume source, always from
        the *emission rate* ``kappa_e * S_n`` (solver convention).  "source"
        (default) resamples from that density each iteration (`sample_births`,
        plan section 2) — lowest noise per sample.  "uniform" draws birth
        points uniformly in the domain volume and weights them by
        ``kappa_e S_n(cell)`` (zero-weight vacuum births are dropped before
        transport); it is the only mode compatible with `crn` and the right
        choice for near-homogeneous media (Ng path).
    max_collisions : int
        Generation cap of the explicit scattering chains / the beam splitting
        loop; histories still live at the cap are reported in 'capped'.
    path_cap : float, optional
        Flight-length cap (default 100 domain diagonals) guarding the marching
        loop against near-grazing outliers; 'n_truncated' counts the cap hits.
    tally_flux : bool
        Also tally the flux vector ``F`` (validation anchor 4).
    implicit_capture, roulette_floor : M4 variance reduction
        See `rt3d.mc.tracking.track_histories` — albedo-weight chains (no
        absorption termination) and a Russian-roulette weight floor.  Only
        the ``explicit`` variant uses the chains; the lambda variant's
        single-flight populations are removal-only and unaffected.
    jit : bool
        Route the flight sampling through the numba fast path
        (`rt3d.mc.jitmc`) when usable — statistically equivalent, different
        random stream.

    Returns
    -------
    dict with keys
        'J', 'sigma_J'      (n_cells,) diffuse mean intensity and its 1-sigma
                            standard error (the plan's acceptance bands are
                            2 sigma)
        'F', 'sigma_F'      (n_cells, 3) flux vector and standard error
        'heating'           ``4 pi kappa_a (B - J)`` (anchor-4 side)
        'history'           relative source change per iteration ("lambda")
        's_max_history'    max |S| per iteration (divergence diagnostics)
        'n_iter', 'converged'
        'J_direct', 'sigma_J_direct'
                            unscattered-beam mean intensity (explicit variant
                            with a beam); 'J_direct_analytic' for the lambda
                            variant (``F0 exp(-tau_b) / 4 pi``)
        'J_total', 'sigma_J_total'
                            'J' plus the direct-beam part
        'J_fixed', 'sigma_J_fixed'
                            the non-iterated source contributions of the
                            "lambda" variant (boundary inflow + beam)
        'n_null', 'n_null_source', 'n_generations', 'n_truncated', 'capped',
        'n_collisions'      tracker diagnostics (null collisions, null source
                            events, generations, path-cap hits, ...)
        'stage1', 'mode', 'kappa_max', 'birth', 'crn', 'ng', 'seed',
        'n_histories', 'n_boundary_histories', 'n_beam_histories', 'n_batches'
                            full traceability (plan section 5)
    """
    if stage1 not in ("lambda", "explicit"):
        raise ValueError(f"stage1 must be 'lambda' or 'explicit', got {stage1!r}")
    if birth not in ("uniform", "source"):
        raise ValueError(f"birth must be 'uniform' or 'source', got {birth!r}")
    if getattr(bc, "ground", None) is not None:
        raise NotImplementedError(
            "MC stage 1 implements the face inflows of boundary_conditions; "
            "the self-coupled Lambertian ground is not part of M1 "
            "(plan/mc-plan.md)"
        )
    if ng is not None and stage1 == "lambda" and not crn:
        raise ValueError(
            "Ng acceleration of the MC lambda iteration requires crn=True "
            "(common random numbers, plan/mc-plan.md section 3)"
        )
    if crn and stage1 == "lambda" and birth != "uniform":
        raise ValueError(
            "crn requires birth='uniform': source-sampled births must be "
            "resampled from S_n every iteration, which is the opposite of a "
            "frozen photon stream"
        )

    nc = n_cells_of(grid)
    kappa_e = np.asarray(kappa_e, dtype=float)
    kappa_s = np.asarray(kappa_s, dtype=float)
    B = np.asarray(B, dtype=float)
    for name, v in (("kappa_e", kappa_e), ("kappa_s", kappa_s), ("B", B)):
        if v.shape != (nc,):
            raise ValueError(f"{name} must have shape ({nc},)")
    if np.any(kappa_e < 0.0) or np.any(kappa_s < 0.0) or np.any(kappa_s > kappa_e):
        raise ValueError("need 0 <= kappa_s <= kappa_e in every cell")
    if np.any(B < 0.0):
        raise ValueError("B must be non-negative")

    omega = single_scattering_albedo(kappa_e, kappa_s)
    kappa_a = kappa_e - kappa_s
    vol = cell_volumes(grid)
    v_tot = float(np.prod([b[1] - b[0] for b in grid.box]))

    if ng is True:
        from rt3d.accelerator import ng1d

        ng = ng1d(grid)
    elif ng is False:
        ng = None

    faces = _inflow_faces(bc, periodic_box)
    per_face = max(1, int(n_boundary_histories) // len(faces)) if faces else 0
    trkw = dict(mode=mode, kappa_max=kappa_max, periodic_box=periodic_box,
                path_cap=path_cap)

    # ================================================================= lambda
    if stage1 == "lambda":
        populations = []
        n_null = 0

        # fixed source 1: face inflows (removal-only transport of I_in)
        rng = np.random.default_rng(seed + 101)
        for fi, (axis, side, I_in) in enumerate(faces):
            pos = face_points(per_face, grid.box, axis, side, rng)
            dirs = face_inflow_directions(per_face, axis, side, rng)
            rec = _single_flight(pos, dirs, grid, kappa_e, seed=seed + 31 * fi,
                                 **trkw)
            populations.append(
                (rec, np.full(per_face, np.pi * face_area(grid.box, axis)
                              * I_in / per_face))
            )
            n_null += rec["n_null"]

        # fixed source 2: beam first scatter (plan section 2 particle stream)
        beam_rec = None
        if beam is not None:
            beam_rec = _beam_first_scatter(
                grid, beam["omega_sun"], beam["F0"], kappa_e, omega, g,
                int(n_beam_histories), mode=mode, kappa_max=kappa_max,
                periodic_box=periodic_box, seed=seed + 555, path_cap=path_cap,
            )
            populations.append((beam_rec, beam_rec["child_w"]))
            n_null += beam_rec["n_null"]

        fixed = _concat_populations(populations, nc, n_batches, tally_flux)
        fixed_res = fixed.result(vol)

        # NOTE on the beam bookkeeping (mirrors rt3d.solver.lambda_iteration):
        # the SC solver sweeps the fixed source
        #     beam_scat(Omega_i) = omega * Phi(->Omega_sun) F0 exp(-tau_b)
        # every iteration and re-scatters only its *transported* J via the
        # omega * <I> term — beam_scat itself never gets multiplied by omega
        # again.  Here that fixed sweep is the bfs particle population above,
        # whose tally J_fixed already IS <sweep(beam_scat)> (children born at
        # the first beam collision with power w_p * omega(cell), then removal-
        # only transport).  So the iterated source S must NOT also carry the
        # angle-averaged first-scatter source omega F0 exp(-tau_b) / 4 pi:
        # transporting it through Lambda would count the beam twice.
        #
        # the iterated operator Lambda: volume births -> removal-only flight.
        # Births sample the *emission rate* kappa_e * S (solver convention:
        # Omega.grad I + kappa_e I = kappa_e * S).  Omitting kappa_e rescales
        # the Lambda operator by 1/kappa_e and makes the lambda iteration
        # diverge for kappa_e < omega-less-than-1 domains.
        rng = np.random.default_rng(seed)
        var_hist = int(n_histories)
        crn_rec = [None]
        S = ((1.0 - omega) * B).copy()

        def _variable(it):
            """(records, per-history weights) of the volume source at iteration it."""
            emit = kappa_e * S
            if birth == "uniform":
                if crn and crn_rec[0] is None:
                    pos = _domain_points(var_hist, grid, rng)
                    bcell = grid.locate(pos)
                    keep = emit[bcell] > 0.0
                    dirs = isotropic_directions(int(keep.sum()), rng)
                    crn_rec[0] = (_single_flight(pos[keep], dirs, grid, kappa_e,
                                                 seed=seed + 7, **trkw),
                                  bcell[keep], 0)
                if crn:
                    rec, bcell, nn = crn_rec[0]
                else:
                    pos = _domain_points(var_hist, grid, rng)
                    bcell = grid.locate(pos)
                    keep = emit[bcell] > 0.0
                    dirs = isotropic_directions(int(keep.sum()), rng)
                    rec = _single_flight(pos[keep], dirs, grid, kappa_e,
                                         seed=seed + 7 + 1_000_003 * it, **trkw)
                    bcell, nn = bcell[keep], rec["n_null"]
                return rec, 4.0 * np.pi * v_tot * emit[bcell] / var_hist, nn
            b = sample_births(var_hist, grid, emit, method="categorical",
                              seed=seed + 13 + it, volumes=vol)
            rec = _single_flight(b["positions"],
                                 isotropic_directions(var_hist, rng),
                                 grid, kappa_e, seed=seed + 17 + it, **trkw)
            return rec, np.full(var_hist, 4.0 * np.pi * b["Q"] / var_hist), rec["n_null"]

        history = []
        s_max = []
        converged = False
        res_tot = None
        n_iter_done = 0
        for it in range(int(n_iter)):
            if np.max(kappa_e * S) > 0.0:
                rec, w, nn = _variable(it)
                n_null += nn
                var = track_tally(nc, rec["n_histories"], n_batches,
                                  flux=tally_flux)
                var.add(rec["hist"], rec["cell"], rec["len"], w[rec["hist"]],
                        rec["dirs_hist"][rec["hist"]])
            else:
                var = track_tally(nc, var_hist, n_batches, flux=tally_flux)
            merged = track_tally(nc, fixed.n_histories, n_batches, flux=tally_flux)
            merged.merge(fixed)
            merged.merge(var)
            res_tot = merged.result(vol)
            n_iter_done = it + 1

            S_new = (1.0 - omega) * B + omega * res_tot["J"]
            scale = max(np.max(np.abs(S_new)), 1e-30)
            rel = np.max(np.abs(S_new - S)) / scale
            history.append(float(rel))
            s_max.append(float(np.max(np.abs(S_new))))
            S = ng.step(S[:, None], S_new[:, None]).ravel() if ng is not None else S_new
            if rel < tol:
                converged = True
                break

        out = _common_output(res_tot, kappa_a, B)
        out.update({
            "history": history,
            "s_max_history": s_max,
            "seed": int(seed),
            "mode": mode,
            "kappa_max": kappa_max,
            "n_batches": fixed.n_batches,
            "n_iter": n_iter_done,
            "converged": converged,
            "stage1": "lambda",
            "birth": birth,
            "crn": bool(crn),
            "ng": ng is not None,
            "n_histories": var_hist,
            "n_boundary_histories": per_face * len(faces),
            "n_beam_histories": int(n_beam_histories) if beam is not None else 0,
            "n_null": int(n_null),
            "n_null_source": 0,
            "n_generations": (beam_rec or {}).get("n_generations", 0),
            "n_truncated": 0,
            "capped": 0,
            "J_fixed": fixed_res["J"],
            "sigma_J_fixed": fixed_res["sigma_J"],
            "J_direct": np.zeros(nc),
            "sigma_J_direct": np.zeros(nc),
        })
        if beam is not None:
            out["J_direct_analytic"] = _beam_j_analytic(
                grid, beam["omega_sun"], kappa_e, beam["F0"], periodic_box
            )
        return out

    # ============================================================== explicit
    starts, dirs, weights, direct = [], [], [], []
    n_null_source = 0

    rate_th = kappa_a * B
    Q_th = float(rate_th @ vol)
    rng = np.random.default_rng(seed)
    if Q_th > 0.0:
        b = sample_births(int(n_histories), grid, rate_th, method="categorical",
                          seed=seed + 3, volumes=vol)
        n_null_source += b["n_null_source"]
        n_th = b["positions"].shape[0]
        starts.append(b["positions"])
        dirs.append(isotropic_directions(n_th, rng))
        weights.append(np.full(n_th, 4.0 * np.pi * Q_th / n_th))
        direct.append(np.zeros(n_th, dtype=bool))

    for axis, side, I_in in faces:
        starts.append(face_points(per_face, grid.box, axis, side, rng))
        dirs.append(face_inflow_directions(per_face, axis, side, rng))
        weights.append(np.full(per_face, np.pi * face_area(grid.box, axis)
                               * I_in / per_face))
        direct.append(np.zeros(per_face, dtype=bool))

    if beam is not None:
        pos, axis, side, mu = beam_seed(int(n_beam_histories), grid.box,
                                        beam["omega_sun"], rng)
        d = np.asarray(beam["omega_sun"], dtype=float)
        starts.append(pos)
        dirs.append(np.tile(d / np.linalg.norm(d), (pos.shape[0], 1)))
        weights.append(np.full(pos.shape[0], float(beam["F0"]) * mu
                               * face_area(grid.box, axis) / pos.shape[0]))
        direct.append(np.ones(pos.shape[0], dtype=bool))

    if not starts:
        raise ValueError("no source population: B, bc inflows and beam are all zero")

    K = sum(s.shape[0] for s in starts)
    starts = np.concatenate(starts)
    dirs = np.concatenate(dirs)
    weights = np.concatenate(weights)
    direct = np.concatenate(direct)

    tally = track_tally(nc, K, n_batches, flux=tally_flux)
    tally_dir = track_tally(nc, K, n_batches, flux=tally_flux) if beam is not None else None

    tr = track_histories(starts, dirs, grid, kappa_e, kappa_s, tally,
                         weights=weights, g=g, tally_direct=tally_dir,
                         is_direct=direct, max_collisions=max_collisions,
                         periodic_box=periodic_box, mode=mode,
                         kappa_max=kappa_max, path_cap=path_cap, seed=seed,
                         implicit_capture=implicit_capture,
                         roulette_floor=roulette_floor, jit=jit)

    res_tot = tally.result(vol)
    out = _common_output(res_tot, kappa_a, B)
    Jd = np.zeros(nc)
    sJd = np.zeros(nc)
    if tally_dir is not None:
        rd = tally_dir.result(vol)
        Jd, sJd = rd["J"], rd["sigma_J"]
        out["J_total"] = res_tot["J"] + Jd
        out["sigma_J_total"] = np.hypot(res_tot["sigma_J"], sJd)
        if "F" in res_tot:
            out["F_total"] = res_tot["F"] + rd["F"]
    out.update({
        "history": [],
        "n_iter": 1,
        "converged": True,
        "stage1": "explicit",
        "seed": int(seed),
        "mode": mode,
        "kappa_max": kappa_max,
        "n_batches": tally.n_batches,
        "birth": "categorical",
        "crn": False,
        "ng": False,
        "n_histories": int(np.count_nonzero(~direct)),
        "n_boundary_histories": per_face * len(faces),
        "n_beam_histories": int(n_beam_histories) if beam is not None else 0,
        "n_null": int(np.sum(tr["n_null"])),
        "n_null_source": int(n_null_source),
        "n_generations": int(tr["n_generations"]),
        "n_truncated": int(tr["n_truncated"]),
        "capped": int(np.count_nonzero(tr["capped"])),
        "n_collisions": int(np.sum(tr["n_collisions"])),
        "implicit_capture": bool(implicit_capture),
        "roulette_floor": roulette_floor,
        "jit": bool(jit),
        "J_direct": Jd,
        "sigma_J_direct": sJd,
        "J_fixed": None,
        "sigma_J_fixed": None,
    })
    return out


def _common_output(res, kappa_a, B):
    """Shared J / F / heating block of both variants."""
    out = {
        "J": res["J"],
        "sigma_J": res["sigma_J"],
        "J_total": res["J"],
        "sigma_J_total": res["sigma_J"],
        "heating": heating_rate(kappa_a, B, res["J"]),
    }
    if "F" in res:
        out["F"] = res["F"]
        out["sigma_F"] = res["sigma_F"]
    return out


def _beam_j_analytic(grid, omega_sun, kappa_e, F0, periodic_box=None):
    """
    Analytic uncollided-beam mean intensity ``F0 exp(-tau_b) / 4 pi`` — the
    reference for the explicit variant's 'J_direct' tally and the beam field
    the lambda variant never tallies.  tau_b from the SC transport machinery
    (rt3d.transport.beam_optical_depth / _octree).
    """
    from rt3d.octree import octree_grid
    from rt3d.transport import beam_optical_depth, beam_optical_depth_octree

    tau_b = (
        beam_optical_depth_octree(grid, omega_sun, kappa_e, periodic_box=periodic_box)
        if isinstance(grid, octree_grid)
        else beam_optical_depth(grid, omega_sun, kappa_e, periodic_box=periodic_box)
    )
    return float(F0) * np.exp(-tau_b) / (4.0 * np.pi)


# =============================================================================
# Solver A, stage 2 — radiance from a fixed source (plan/mc-plan.md section 3,
# phase M2)
# =============================================================================


def stage2_source(grid, dg, kappa_e, kappa_s, B, J, g=0.0, beam=None,
                  periodic_box=None):
    """
    Assemble the stage-2 fixed source in the solver convention,

        S_total = (1 - omega) * B + omega * J  [+ beam first scatter]

    reusing the SC bookkeeping.  Returns the scalar form ``(n_cells,)`` when
    there is no beam and the direction-resolved form ``(n_cells, N_dir)``
    when there is one — the beam first-scatter source
    ``omega * Phi_HG(->omega_sun) F0 exp(-tau_b)`` of
    `rt3d.source.beam_scattering_source` is strongly anisotropic, so stage-2
    radiance comparisons against `rt3d.solver.lambda_iteration` need the
    per-ordinate values (not just its angle average, which is what stage 1
    uses via the bfs particle stream).
    """
    from rt3d.octree import octree_grid
    from rt3d.source import beam_scattering_source
    from rt3d.transport import beam_optical_depth, beam_optical_depth_octree

    omega = single_scattering_albedo(kappa_e, kappa_s)
    S = (1.0 - omega) * B + omega * J
    if beam is None:
        return S
    if isinstance(grid, octree_grid):
        tau_b = beam_optical_depth_octree(
            grid, beam["omega_sun"], kappa_e, periodic_box=periodic_box
        )
    else:
        tau_b = beam_optical_depth(
            grid, beam["omega_sun"], kappa_e, periodic_box=periodic_box
        )
    g_cells = None if np.ndim(g) == 0 else np.asarray(g, dtype=float)
    beam_scat = beam_scattering_source(
        dg, beam["omega_sun"], beam["F0"], tau_b, omega,
        g if g_cells is None else 0.0, g_cells=g_cells,
    )
    return S[:, None] + beam_scat


def radiance_from_source(
    grid,
    dg,
    kappa_e,
    S_total,
    bc,
    periodic_box=None,
    estimator="next_flight",
    n_histories=25,
    n_batches=10,
    seed=0,
    mode="cell",
    kappa_max=None,
    jit=False,
):
    """
    MC stage 2 — the radiance field ``I(cell, ordinate)`` of a **fixed**
    source, on a ray set identical to the SC sweeps: one ray per cell/leaf
    center per icosahedral ordinate, marched **backwards** from the center to
    the domain entry face (exactly `precompute_octree_march` on the octree,
    the `walk_lattice` DDA on the unigrid), with the boundary inflow applied
    at the face the backward march exits (the `boundary_conditions.inflow`
    convention of the SC sweeps).

    For a target point P at path length ``s_P`` from the entry,

        I(P) = I_in exp(-tau(0, s_P)) + int_0^{s_P} kappa_e S e^{-tau(s, s_P)} ds

    - ``estimator="next_flight"`` (default): for every ray, ``n_histories``
      backward flights are sampled with the M0 Woodcore tracker (``mode``
      "cell"/"global"); each flight scores the source **intensity**
      ``S_total(cell, d)`` at the point of its first *real* collision
      (nulls continue; flights exiting without a collision score 0).  The
      real-collision density ``kappa_e e^{-tau}`` times the source intensity
      reproduces the line integral exactly — the estimator is unbiased in
      both majorant modes with no ``/kappa_max`` factor (deviation from the
      plan sketch, logged).  Error bars: the flights of one ray form
      ``n_batches`` history batches; ``sigma_I`` is the batch-sums standard
      error of the ray mean.
    - ``estimator="deterministic"``: the same line integral evaluated
      exactly per march segment with ``S (1 - exp(-tau_seg)) exp(-tau_before)``
      weights — the zero-noise limit of the tally and the same recursion the
      SC octree sweep runs (so on the octree this is the SC intensity to
      floating-point round-off).

    Parameters
    ----------
    grid : rt3d.geometry.unigrid or rt3d.octree.octree_grid
    dg : rt3d.directions.direction_grid
        Ordinate set the radiance is tallied on (tally binning only; MC
        collision sampling itself stays continuous in angle).
    kappa_e : (n_cells,) ndarray
    S_total : (n_cells,) or (n_cells, N_dir) ndarray
        Fixed source function (intensity units), e.g. `stage2_source`.
    bc : rt3d.transport.boundary_conditions
        Face inflows; a Lambertian `ground` raises NotImplementedError (M2
        scope — the self-coupled surface needs the SC phase-5 machinery).
    periodic_box : rt3d.geometry.cuboid, optional
    estimator : {"next_flight", "deterministic"}
    n_histories : int
        Flights per ray (per cell and ordinate) of the next-flight tally.
    n_batches : int
        History batches for sigma_I.
    seed : int
        Base PCG64 seed (stream offset per ordinate), recorded in the output.
    mode, kappa_max : str, float
        Woodcock majorant mode of the tracker.
    jit : bool
        Route the next-flight sampling through the numba fast path
        (`rt3d.mc.jitmc`) when usable — statistically equivalent, different
        random stream (M4/M5).

    Returns
    -------
    dict with keys
        'I'       (n_cells, N_dir) intensities at cell centers
        'sigma_I' (n_cells, N_dir) 1-sigma standard errors (zeros for the
                  deterministic estimator)
        'J'       (n_cells,) quadrature mean intensity of the tally
        'estimator', 'mode', 'kappa_max', 'seed', 'n_refine', 'n_dir',
        'n_histories', 'n_batches', 'n_null', 'jit'   traceability / diagnostics
    """
    from rt3d.geometry import ray_box_exit
    from rt3d.octree import octree_grid
    from rt3d.transport import _block_prefixes, precompute_octree_march

    if estimator not in ("next_flight", "deterministic"):
        raise ValueError(
            f"estimator must be 'next_flight' or 'deterministic', got {estimator!r}"
        )
    if getattr(bc, "ground", None) is not None:
        raise NotImplementedError(
            "MC stage 2 implements the face inflows of boundary_conditions; "
            "the self-coupled Lambertian ground is not part of M2 "
            "(plan/mc-plan.md)"
        )

    is_octree = isinstance(grid, octree_grid)
    nc = n_cells_of(grid)
    kappa_e = np.asarray(kappa_e, dtype=float)
    if kappa_e.shape != (nc,):
        raise ValueError(f"kappa_e must have shape ({nc},)")
    S_is_dir = np.ndim(S_total) == 2
    S_total = np.asarray(S_total, dtype=float)
    if S_total.shape != ((nc, dg.n_dir) if S_is_dir else (nc,)):
        raise ValueError("S_total must have shape (n_cells,) or (n_cells, N_dir)")

    n_h = int(n_histories) if estimator == "next_flight" else 0
    nb = max(1, min(int(n_batches), n_h)) if n_h else 1

    centers = grid.centers
    I = np.zeros((nc, dg.n_dir))
    sI = np.zeros((nc, dg.n_dir))
    n_null_total = 0

    for d in range(dg.n_dir):
        om = np.asarray(dg.directions[d], dtype=float)
        u = -om
        s_d = S_total[:, d] if S_is_dir else S_total

        # ---- backward march geometry for every center + uncollided data
        if is_octree:
            march = precompute_octree_march(grid, om,
                                            periodic_box=periodic_box, jit=False)
            seg_cell = march["leaf"]
            seg_len = march["length"]
            offsets = march["offsets"]
            ray_of = march["ray_of"]
            budget = np.zeros(nc)
            np.add.at(budget, ray_of, seg_len)
            I_in = np.full(nc, bc.top_I if om[2] < 0.0 else bc.bottom_I)
        else:
            if periodic_box is None:
                budget, ex_ax, ex_side = ray_box_exit(centers, u, grid.box)
                I_in = bc.inflow(ex_ax, ex_side)
            else:
                budget = _exit_distance(centers, np.tile(u, (nc, 1)), grid,
                                        periodic_box=periodic_box)
                I_in = np.full(nc, bc.top_I if om[2] < 0.0 else bc.bottom_I)
            # segment records of the backward march: needed for the exact
            # uncollided attenuation on the inhomogeneous lattice in **both**
            # estimators (and for the deterministic source integral)
            seg_ray, seg_cell, seg_len, *_ = walk_lattice(
                centers, u, grid, budgets=budget, periodic_box=periodic_box
            )
            order = np.argsort(seg_ray, kind="stable")
            seg_ray, seg_cell, seg_len = seg_ray[order], seg_cell[order], seg_len[order]
            ends = np.append(np.flatnonzero(np.diff(seg_ray)), seg_ray.size - 1)
            offsets = np.append(np.append(0, ends[:-1] + 1), seg_ray.size)
            ray_of = np.repeat(np.arange(nc), np.diff(offsets))

        # ---- per-segment attenuation (both estimators need the uncollided term)
        tau_seg = kappa_e[seg_cell] * seg_len
        pre_before, _ = _block_prefixes(tau_seg, offsets)
        tau_tot = np.zeros(nc)
        np.add.at(tau_tot, ray_of, tau_seg)
        uncollided = I_in * np.exp(-tau_tot)

        if estimator == "deterministic":
            src = np.zeros(nc)
            np.add.at(src, ray_of,
                      s_d[seg_cell] * (-np.expm1(-tau_seg)) * np.exp(-pre_before))
            I[:, d] = uncollided + src
            continue

        # ---- next-flight tally: n_h backward flights per ray
        K = nc * n_h
        starts = np.repeat(centers, n_h, axis=0)
        buds = np.repeat(budget, n_h)
        res = sample_collisions(starts, np.tile(u, (K, 1)), buds, grid, kappa_e,
                                kappa_e, mode=mode, kappa_max=kappa_max,
                                periodic_box=periodic_box, seed=seed + 1 + d,
                                jit=jit)
        real = res["coll_event"] != EV_EXIT
        dep = np.zeros(K)
        dep[real] = s_d[res["coll_cell"][real]]
        n_null_total += int(np.sum(res["n_null"]))

        # history batches per ray: flight f of ray c sits in batch of f
        edges = np.round(np.linspace(0, n_h, nb + 1)).astype(np.int64)
        batch_of_f = np.repeat(np.arange(nb), np.diff(edges))
        ray_idx = np.repeat(np.arange(nc), n_h)
        flat = batch_of_f[np.arange(K) % n_h] * nc + ray_idx
        sums = np.bincount(flat, weights=dep, minlength=nb * nc).reshape(nb, nc)
        est = sums.sum(axis=0)
        se = (np.sqrt(nb) * sums.std(axis=0, ddof=1) / n_h if nb > 1
              else np.zeros(nc))
        I[:, d] = uncollided + est / n_h
        sI[:, d] = se

    Jq = I @ dg.weights / (4.0 * np.pi)
    return {
        "I": I,
        "sigma_I": sI,
        "J": Jq,
        "estimator": estimator,
        "mode": mode,
        "kappa_max": kappa_max,
        "seed": int(seed),
        "n_refine": getattr(dg, "n_refine", None),
        "n_dir": int(dg.n_dir),
        "n_histories": n_h,
        "n_batches": nb,
        "n_null": n_null_total,
        "jit": bool(jit),
    }
