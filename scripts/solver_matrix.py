# -*- coding: utf-8 -*-
"""
Solver-matrix report (plan/solver-test-matrix-plan.md section 9).

One runner that exercises **every solver variant of the matrix case on one
physical setup** and writes the report artifacts:

    matrix.csv          one row per variant: iterations, wall time, J/F/all-sky
                        scalars, energy-closure residual, error vs reference
    ranking.csv         every variant's error against the fine-octree reference
                        (field max/rms J, mean J, absorbed power, out-top power)
    convergence.csv     refinement ladders (unigrid Cauchy, octree tau_max) and
                        the Monte Carlo N^-1/2 fit, with the fitted slopes
    mc_stats.csv        masked / unmasked z statistics and tail fractions for
                        the MC estimators (the F3 noise story, plan section 6)
    plots               j_err_maps.png, j_convergence.png, err_bars.png,
                        z_hist.png, mc_convergence.png, closure.png,
                        capability_matrix.png

The physical setup, grids, direction grid and solve options are the ones the
pytest gate uses (`tests/matrix_cases.py`), so the numbers in this report are
the *same* numbers the pinned bands were measured from — the report and the
gate cannot drift apart.

Reference: octree tau_max 0.5, max_depth 5 (the plan's REF rung).  The
642-ordinate reference (`--reference-ordinates 3`) is opt-in: it is ~15x the
memory and time and per plan section 10 belongs to the convergence study, not
the default report.

The all-sky ground case (top inflow + solar beam + Lambertian/emissive
ground) runs for **every** family now that phase M6 has landed; the F12 fix
(`rt3d/mc/tracking.py` face-time buffer) is a precondition of the MC numbers
being reproducible at all.

Run from the repo root:
    macOS/Linux:  ~/miniforge3/envs/pyarts/bin/python scripts/solver_matrix.py
    Windows:      $env:Path = "$env:USERPROFILE\\miniforge3\\envs\\pyarts\\Library\\bin;" + $env:Path
                  & "$env:USERPROFILE\\miniforge3\\envs\\pyarts\\python.exe" scripts/solver_matrix.py

Add ``--quick`` for a reduced (faster, coarser) variant set used to smoke the
report machinery itself.
"""

# %% imports and configuration
import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matrix_cases import (
    cloud_beam,
    cloud_case,
    cloud_fields,
    cloud_octree,
    cloud_unigrid,
    direction_grid,
)
from matrix_stats import (
    field_errors,
    next_flight_mask,
    ray_optical_depth_octree,
    rel_max_dev,
    rel_rms_dev,
    z_stats,
)

import rt3d
from rt3d.diagnostics import (
    angular_variance,
    cell_volumes,
    energy_balance,
    face_powers,
    heating_rate,
)
from rt3d.mc.backward import backward_all_sky
from rt3d.mc.forward import j_mc, radiance_from_source
from rt3d.solver import lambda_iteration
from rt3d.source import mean_intensity

OUT = "plot/solver_matrix"
CACHE = os.path.join(OUT, "_solve_cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true",
                help="reduced variant set to smoke the report machinery")
ap.add_argument("--reference-ordinates", type=int, default=1,
                help="icosahedral refinement of the REF run (1 = 42 ordinates, "
                     "3 = 642: opt-in, huge)")
ARGS = ap.parse_args()

BOX = cloud_case()["box"]
DG = direction_grid(1)
DG_REF = direction_grid(ARGS.reference_ordinates)

REF_TAU, REF_DEPTH = 0.5, 5

print(f"solver-matrix report | {rt3d.__version__ if hasattr(rt3d, '__version__') else 'rt3d'} "
      f"| quick={ARGS.quick} | ref ordinates={DG_REF.n_dir}")


# =============================================================================
# solve cache (plan section 9: keyed by variant id, stale entries re-solved)
# =============================================================================

def _cache_path(key):
    return os.path.join(CACHE, key.replace("/", "_").replace(" ", "_") + ".npz")


def cached_solve(key, fn):
    """fn() -> dict of arrays; cached to disk under `key`."""
    p = _cache_path(key)
    if os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
            if "wall" in out:
                out["wall"] = float(out["wall"])
            return out, True
    t0 = time.perf_counter()
    res = fn()
    wall = time.perf_counter() - t0
    res = dict(res)
    res["wall"] = wall
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in res.items()
                              if not isinstance(v, str)})
    return res, False


# =============================================================================
# the deterministic variants (SC family) — all on the all-sky ground case
# =============================================================================

def sc_solve(grid, transport="chain", delta_M=1, ng=True, jit=None,
             n_jobs=None, dg=None):
    kappa_e, kappa_s, g, B = cloud_fields(grid)
    bc = _bc_ground()
    extra = dict(n_jobs=n_jobs) if n_jobs is not None else {}
    return lambda_iteration(
        grid, dg or DG, kappa_e, kappa_s, B, bc, g=g, periodic_box=BOX,
        beam=cloud_beam(), transport=transport, delta_M=delta_M, ng=ng,
        jit=jit, n_iter=400, tol=1e-8, **extra,
    )


def _bc_ground(albedo=0.3, emissivity=1.0, B_g=1.0):
    from rt3d.geometry import ground_surface
    from rt3d.transport import boundary_conditions

    return boundary_conditions(
        top_I=1.0,
        ground=ground_surface(albedo=albedo, emissivity=emissivity, B_g=B_g))


# variant id, grid factory, sc kwargs, notes
if ARGS.quick:
    SC_VARIANTS = [
        ("u24/chain", lambda: cloud_unigrid(24), {}, ""),
        ("u24/sc", lambda: cloud_unigrid(24), dict(transport="sc"), "F11 plateau"),
        ("oct-d2/chain", lambda: cloud_octree(2.0, 2), {}, ""),
    ]
else:
    SC_VARIANTS = [
        ("u12/chain", lambda: cloud_unigrid(12), {}, "ladder rung"),
        ("u24/chain", lambda: cloud_unigrid(24), {}, "ladder rung"),
        ("u48/chain", lambda: cloud_unigrid(48), {}, "ladder rung"),
        ("u24/sc", lambda: cloud_unigrid(24), dict(transport="sc"),
         "F11 plateau (chain 4.1e-2 vs sc 9.4e-2 rms J)"),
        ("u24/chain/jit-false", lambda: cloud_unigrid(24), dict(jit=False),
         "NumPy reference backend"),
        ("u24/chain/njobs1", lambda: cloud_unigrid(24), dict(n_jobs=1),
         "threading off"),
        ("oct-d2(tau2)/chain", lambda: cloud_octree(2.0, 2), {}, ""),
        ("oct-d3(tau2)/chain", lambda: cloud_octree(2.0, 3), {},
         "ladder rung"),
        ("oct-d5(tau1)/chain", lambda: cloud_octree(1.0, REF_DEPTH), {},
         "ladder rung"),
        ("oct-REF/ng-off", lambda: cloud_octree(REF_TAU, REF_DEPTH),
         dict(ng=False), "plain lambda iteration"),
        ("oct-REF/dM2", lambda: cloud_octree(REF_TAU, REF_DEPTH),
         dict(delta_M=2), "delta-M 2"),
    ]

REF_KEY = "oct-REF/chain"


def solve_ref():
    grid = cloud_octree(REF_TAU, REF_DEPTH)
    res = sc_solve(grid, dg=DG_REF)
    assert res["converged"], "reference solve did not converge"
    return {"J": res["J"], "I": res["I"].astype(np.float32),
            "n_iter": res["n_iter"], "converged": res["converged"]}


ref, hit = cached_solve(REF_KEY, solve_ref)
print(f"reference ({'cached' if hit else 'solved'}): {ref['wall']:.1f} s, "
      f"{ref['I'].shape[0]} cells, {DG_REF.n_dir} ordinates")


# =============================================================================
# run the SC variants, fill matrix.csv / ranking.csv
# =============================================================================

def sc_scalars(grid, res, dg):
    kappa_e, kappa_s, g, B = cloud_fields(grid)
    vol = cell_volumes(grid)
    fp = face_powers(grid, res["I"], dg)
    eb = energy_balance(grid, res["I"], dg, kappa_e - kappa_s, B)
    return {
        "n_cells": len(res["J"]),
        "n_iter": res["n_iter"],
        "converged": bool(res["converged"]),
        "mean_J": float(np.average(res["J"], weights=vol)),
        "power": float(4 * np.pi * np.sum((kappa_e - kappa_s) * res["J"] * vol)),
        "out_top": float(fp["out_top"]),
        "out_bottom": float(fp["out_bottom"]),
        "closure_rel": float(eb["relative"]),
        "angvar": float(np.mean(angular_variance(res["I"], dg))),
    }


# the reference grid object (rebuild once; locate() only needs the tree)
_ref_grid = cloud_octree(REF_TAU, REF_DEPTH)


matrix_rows, ranking_rows = [], []


def add_variant(vid, grid, res, family, transport, notes, dg=None):
    dg = dg or DG
    s = sc_scalars(grid, res, dg)
    e_max, e_rms = field_errors(grid, res["J"], _ref_grid, ref["J"])
    smean = float(np.average(res["J"], weights=cell_volumes(grid)))
    sref = float(np.average(ref["J"], weights=cell_volumes(_ref_grid)))
    p_sc = s["power"]
    p_ref = float(4 * np.pi * np.sum((_ref_grid.kappa_e - _ref_grid.kappa_s)
                                     * ref["J"] * cell_volumes(_ref_grid)))
    matrix_rows.append(dict(
        id=vid, family=family, grid=type(grid).__name__, transport=transport,
        n_dir=dg.n_dir, n_iter=s["n_iter"], converged=s["converged"],
        wall_s=round(float(res["wall"]), 2), n_cells=s["n_cells"],
        mean_J=f"{smean:.5f}", power=f"{p_sc:.2f}",
        out_top=f"{s['out_top']:.2f}", out_bottom=f"{s['out_bottom']:.2f}",
        closure_rel=f"{s['closure_rel']:.3e}",
        J_err_max=f"{e_max:.3e}", J_err_rms=f"{e_rms:.3e}", notes=notes,
    ))
    ranking_rows.append(dict(
        id=vid, wall_s=round(float(res["wall"]), 2),
        J_max_err=f"{e_max:.3e}", J_rms_err=f"{e_rms:.3e}",
        mean_J_err=f"{(smean - sref) / sref:+.3e}",
        power_err=f"{(p_sc - p_ref) / p_ref:+.3e}",
    ))
    print(f"  {vid:24s} wall {res['wall']:7.2f}s  J_rms_err {e_rms:.3e}  "
          f"closure {s['closure_rel']:.1e}")


_sc_live = {}
for vid, mk, kw, note in SC_VARIANTS:
    g = mk()
    t0 = time.perf_counter()
    r = sc_solve(g, **kw)
    r["wall"] = time.perf_counter() - t0
    assert r["converged"], f"{vid} did not converge"
    _sc_live[vid] = (g, r)
    add_variant(vid, g, r, "SC", kw.get("transport", "chain"), note)

# the REF row itself (zero error against itself)
matrix_rows.append(dict(
    id=REF_KEY, family="SC", grid="octree", transport="chain",
    n_dir=DG_REF.n_dir, n_iter=int(ref["n_iter"]),
    converged=bool(ref["converged"]), wall_s=round(float(ref["wall"]), 2),
    n_cells=ref["J"].size,
    mean_J=f"{np.average(ref['J'], weights=cell_volumes(_ref_grid)):.5f}",
    power=f"{4 * np.pi * np.sum((_ref_grid.kappa_e - _ref_grid.kappa_s) * ref['J'] * cell_volumes(_ref_grid)):.2f}",
    out_top="", out_bottom="", closure_rel="",
    J_err_max="0", J_err_rms="0", notes="reference rung (tau 0.5, depth 5)"))
ranking_rows.append(dict(id=REF_KEY, wall_s=round(float(ref["wall"]), 2),
                         J_max_err="0", J_rms_err="0", mean_J_err="0",
                         power_err="0"))


# =============================================================================
# the Monte Carlo variants (MC family) on the same case
# =============================================================================
# j_mc has no angular quadrature in stage 1: J comparisons use J_total
# (incl. the uncollided-beam tally), and SC comparisons must add the
# analytic uncollided column to the SC side (convention: tests §4 header).

mc_rows, mc_stats_rows = [], []
HIST = 5_000 if ARGS.quick else 120_000
BATCHES = 50 if ARGS.quick else 300


def mc_j_total(r):
    """MC mean-intensity field on the SC-convention basis: J_total already
    carries the MC-collided field plus the *tallied* uncollided beam
    (explicit variant); the lambda variant never tallies the uncollided
    column but ships the analytic one -- add it (convention: T3 §4)."""
    J = r["J_total"] if "J_total" in r else r["J"] + r.get("J_direct", 0.0)
    jd = r.get("J_direct_analytic")
    if isinstance(jd, np.ndarray):
        J = J + jd
    return J


oct2 = cloud_octree(2.0, 2)
ke2, ks2, g2, B2 = cloud_fields(oct2)
vol2 = cell_volumes(oct2)
w2 = 4 * np.pi * (ke2 - ks2) * vol2
beam = cloud_beam()

# SC on the SAME grid, conventions matched (SC J + analytic direct column)
res2 = sc_solve(oct2)
from rt3d.mc.forward import _beam_j_analytic
J_sc2 = res2["J"] + _beam_j_analytic(oct2, beam["omega_sun"], ke2, beam["F0"],
                                     periodic_box=BOX)

A1_RUNS = [("A1/explicit", "explicit", {}),
           ("A1/lambda", "lambda", {}),
           ("A1/explicit/global", "explicit", dict(mode="global"))]
a1 = {}
for vid, stage, kw in (A1_RUNS if not ARGS.quick else A1_RUNS[:1]):
    nh_var = (min(HIST, 30_000) if stage == "lambda" else 0)
    extra = dict(n_iter=100, tol=1e-8) if stage == "lambda" else {}
    t0 = time.perf_counter()
    r = j_mc(oct2, ke2, ks2, B2, _bc_ground(), g=g2, beam=beam,
             periodic_box=BOX, stage1=stage,
             n_histories=nh_var,
             n_boundary_histories=HIST, n_beam_histories=HIST,
             n_ground_histories=HIST, n_batches=BATCHES, seed=101,
             jit=True, **dict(kw, **extra))
    wall = time.perf_counter() - t0
    r["wall"] = wall
    a1[vid] = r
    J = mc_j_total(r)
    P = float(np.sum(w2 * J))
    P_sc = float(np.sum(w2 * J_sc2))
    matrix_rows.append(dict(
        id=vid, family="MC", grid="octree-d2", transport=f"stage1={stage}",
        n_dir=0, n_iter=r.get("n_iter", 1), converged=True,
        wall_s=round(wall, 2), n_cells=J.size,
        mean_J=f"{np.mean(J):.5f}", power=f"{P:.2f}",
        out_top="", out_bottom="", closure_rel="n/a (no ordinate set)",
        J_err_max="", J_err_rms="",
        notes=f"power vs SC(same grid) {(P - P_sc) / P_sc * 100:+.2f} %"))
    print(f"  {vid:24s} wall {wall:7.2f}s  power {P:.1f} "
          f"({(P - P_sc) / P_sc * 100:+.2f} % vs SC)")

# seed-pair per-cell z (validity of the merged-tally sigma, F9/F12 guard)
r_b = j_mc(oct2, ke2, ks2, B2, _bc_ground(), g=g2, beam=beam, periodic_box=BOX,
           stage1="explicit", n_histories=0, n_boundary_histories=HIST,
           n_beam_histories=HIST, n_ground_histories=HIST,
           n_batches=BATCHES, seed=202, jit=True)
from matrix_stats import combined_sigma, z_scores
sig = combined_sigma(a1["A1/explicit"]["sigma_J_total"], r_b["sigma_J_total"])
zz = z_scores(a1["A1/explicit"]["J_total"] - r_b["J_total"], sig,
              np.zeros_like(sig))
ok = sig > 1e-6 * np.max(np.abs(a1["A1/explicit"]["J_total"]))
for name, mask in (("masked", ok), ("unmasked", np.ones_like(ok))):
    st = z_stats(zz[mask])
    mc_stats_rows.append(dict(quantity="A1 seed pair (J_total)", mask=name,
                              n=st["n"], mean=f"{st['mean']:.3f}",
                              rms=f"{st['rms']:.3f}",
                              tail2=f"{st['tail2']:.4f}",
                              tail3=f"{st['tail3']:.4f}",
                              max_abs=f"{st['max_abs']:.1f}"))

# A2 next-flight stage 2 on oct2 with the SC-converged source of the SAME
# grid; the stage-2 ground inflow must be handed in closed (M6 convention):
# ground_radiance with the converged F_down and the direct beam ground term.
from rt3d.mc.ground import ground_radiance
from rt3d.transport import bottom_mask, downward_irradiance

NF_FLIGHTS = 25 if ARGS.quick else 100
dg_nf = DG
mb2 = bottom_mask(oct2)
F_down2 = downward_irradiance(res2["I"][:, dg_nf.down_idx], dg_nf)[mb2]
from rt3d.transport import beam_optical_depth_octree
tau_b2 = beam_optical_depth_octree(oct2, beam["omega_sun"], ke2,
                                   periodic_box=BOX)
from rt3d.mc.ground import ground_beam_irradiance
Fb2 = ground_beam_irradiance(oct2, beam, ke2, tau_b2, periodic_box=BOX)
I_g2 = ground_radiance(oct2, _bc_ground().ground, F_down2, beam_irr=Fb2)

det = radiance_from_source(oct2, dg_nf, res2["kappa_e_delta"], res2["S"],
                           _bc_ground(), periodic_box=BOX,
                           estimator="deterministic", ground_I=I_g2)
nf = radiance_from_source(oct2, dg_nf, estimator="next_flight",
                          kappa_e=res2["kappa_e_delta"], S_total=res2["S"],
                          bc=_bc_ground(), periodic_box=BOX, ground_I=I_g2,
                          n_histories=NF_FLIGHTS, n_batches=min(25, NF_FLIGHTS),
                          seed=11, jit=True)
tau_ray = ray_optical_depth_octree(oct2, dg_nf, res2["kappa_e_delta"], BOX)
keep = next_flight_mask(nf["sigma_I"], nf["I"], tau_ray,
                        n_histories=NF_FLIGHTS,
                        min_scoring=0.16 * NF_FLIGHTS)
zn = z_scores(nf["I"], nf["sigma_I"], det["I"])
for name, mask in (("masked", keep), ("unmasked", np.ones_like(keep))):
    st = z_stats(zn[mask])
    mc_stats_rows.append(dict(quantity=f"A2 next-flight vs det ({NF_FLIGHTS} fl.)",
                              mask=name, n=st["n"], mean=f"{st['mean']:.3f}",
                              rms=f"{st['rms']:.3f}",
                              tail2=f"{st['tail2']:.4f}",
                              tail3=f"{st['tail3']:.4f}",
                              max_abs=f"{st['max_abs']:.1f}"))
matrix_rows.append(dict(
    id="A2/next-flight", family="MC", grid="octree-d2",
    transport="stage2 next_flight", n_dir=dg_nf.n_dir, n_iter=1, converged=True,
    wall_s=round(float(nf.get("wall", 0.0)), 2), n_cells=nf["I"].shape[0],
    mean_J=f"{np.mean(mean_intensity(nf['I'], dg_nf)):.5f}", power="",
    out_top="", out_bottom="", closure_rel="n/a (stage-2 only)",
    J_err_max="", J_err_rms="",
    notes=f"agg dev / agg sigma = {rel_rms_dev(nf['I'], det['I']) / max(np.sqrt(np.mean(nf['sigma_I'] ** 2)) / np.sqrt(np.mean(det['I'] ** 2)), 1e-30):.3f}"))

# B adjoint all-sky at a few leaves, vs the SC columns (masked z per seed).
# Pick leaves INSIDE the cloud core (highest kappa_a along each z-slab) --
# like the T3 gate's fixed probe points; a near-vacuum leaf would put the
# z denominator entirely in the F14 quadrature-normalization gap.
sel = np.argsort(oct2.centers[:, 2])
leaf_pts = []
for frac in (0.35, 0.5):
    zs = oct2.centers[sel, 2]
    z_target = np.quantile(zs, frac)
    band = np.flatnonzero(np.abs(zs - z_target) < 0.3)
    cand = [sel[i] for i in band]
    leaf_pts.append(max(cand, key=lambda i: ke2[i] - ks2[i]))
zbands = []
for ip, leaf in enumerate(leaf_pts):
    pt = oct2.centers[leaf]
    n_adj = 2_000 if ARGS.quick else 20_000
    ab = backward_all_sky(oct2, ke2, ks2, B2, _bc_ground(), dg_nf, tuple(pt),
                          g=g2, beam=beam, periodic_box=BOX,
                          n_seeds=n_adj, n_batches=50, seed=14 + ip)
    Icol_sc = det["I"][leaf, ab["up_idx"]]
    zc = z_scores(ab["I"], ab["sigma_I"], Icol_sc)
    okc = next_flight_mask(ab["sigma_I"], ab["I"], tau_ray[leaf, ab["up_idx"]],
                           n_histories=n_adj,
                           min_scoring=3.0 if ARGS.quick else 10.0)
    st = z_stats(zc[okc])
    zbands.append(st)
    mc_stats_rows.append(dict(quantity=f"B adjoint all-sky leaf {ip}", mask="masked",
                              n=st["n"], mean=f"{st['mean']:.3f}",
                              rms=f"{st['rms']:.3f}",
                              tail2=f"{st['tail2']:.4f}",
                              tail3=f"{st['tail3']:.4f}",
                              max_abs=f"{st['max_abs']:.1f}"))
    print(f"  B adjoint leaf {ip}: masked rms z {st['rms']:.2f}, "
          f"|z|>3 {st['tail3']:.3f}")
matrix_rows.append(dict(
    id="B/adjoint-all-sky", family="MC", grid="octree-d2",
    transport="adjoint (exact-HG)", n_dir=dg_nf.n_dir, n_iter=1, converged=True,
    wall_s="", n_cells=len(leaf_pts), mean_J="", power="", out_top="",
    out_bottom="", closure_rel="n/a (detector estimator)",
    J_err_max="", J_err_rms="",
    notes=f"masked rms z vs SC columns {[round(s['rms'], 2) for s in zbands]}"
          " -- per-ordinate gap is F14-limited (quadrature-normalized vs"
          " analytic beam HG), not noise-limited"))


# =============================================================================
# convergence ladders -> convergence.csv
# =============================================================================

conv_rows = []

# unigrid Cauchy ladder (chain): successive rung differences + slope
cauchy_ns = (12, 24, 48) if not ARGS.quick else (12, 24)
sols = {}
for n in cauchy_ns:
    g = cloud_unigrid(n)
    t0 = time.perf_counter()
    r = sc_solve(g)
    r["wall"] = time.perf_counter() - t0
    sols[n] = (g, r, r["wall"])
for a, b in zip(cauchy_ns, cauchy_ns[1:]):
    ga, ra, _ = sols[a]
    gb, rb, _ = sols[b]
    _em, erms = field_errors(ga, ra["J"], gb, rb["J"])
    conv_rows.append(dict(family="unigrid", kind="Cauchy pair",
                          rung_a=f"u{a}", rung_b=f"u{b}",
                          err=erms, note="pairwise rms J"))
if len(cauchy_ns) >= 3:
    # pair i compares rung u_a (coarse) with u_b = 2 u_a: the error scale is
    # the coarse cell width h = 1/n_a; slope of log(err) vs log(h) ~ 1
    hs = [1.0 / n for n in cauchy_ns[:-1]]
    errs = [float(r["err"]) for r in conv_rows[-(len(cauchy_ns) - 1):]]
    slope = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    conv_rows.append(dict(family="unigrid", kind="fitted slope (Cauchy)",
                          rung_a="", rung_b="", err=slope,
                          note="expect ~1 (F11: plateau vs octree stays)"))

# octree tau_max ladder against the REF grid (d5)
oct_ladder = (4.0, 2.0, 1.0) if not ARGS.quick else (2.0,)
for tau in oct_ladder:
    g = cloud_octree(tau, REF_DEPTH)
    r = sc_solve(g)
    _em, erms = field_errors(g, r["J"], _ref_grid, ref["J"])
    conv_rows.append(dict(family="octree", kind="tau rung vs REF",
                          rung_a=f"tau{tau}", rung_b="REF", err=erms,
                          note="d5 fixed; tau ladder saturates below tau ~1 (F10)"))

# MC power seed-split vs N -> N^-0.5 fit.  THREE seeds per rung: a
# two-seed split is itself a ~50 %-noisy estimate of the sd and cannot
# support a slope fit (measured: a lucky 80k pair faked a 4x drop).
mc_ns = (20_000, 80_000, 320_000) if not ARGS.quick else (5_000, 20_000)
mc_err = []
for Nh in mc_ns:
    Ps = []
    for seed in (101, 202, 303):
        r = j_mc(oct2, ke2, ks2, B2, _bc_ground(), g=g2, beam=beam,
                 periodic_box=BOX, stage1="explicit", n_histories=0,
                 n_boundary_histories=Nh, n_beam_histories=Nh,
                 n_ground_histories=Nh, n_batches=100, seed=seed, jit=True)
        Ps.append(float(np.sum(w2 * mc_j_total(r))))
    e = float(np.std(Ps, ddof=1)) / abs(float(np.mean(Ps)))
    mc_err.append((Nh, e))
    conv_rows.append(dict(family="MC", kind="power seed-std rel err",
                          rung_a=f"N={Nh}", rung_b="", err=e,
                          note=f"explicit, 3 seeds: {Ps}"))
slope_mc = float(np.polyfit(np.log([n for n, _ in mc_err]),
                            np.log([e for _, e in mc_err]), 1)[0])
conv_rows.append(dict(family="MC", kind="fitted slope (power err vs N)",
                      rung_a="", rung_b="", err=slope_mc,
                      note="expect ~ -0.5"))
print(f"MC power seed-split slope: {slope_mc:.2f}")


# =============================================================================
# write CSVs
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)


write_csv(os.path.join(OUT, "matrix.csv"), matrix_rows)
write_csv(os.path.join(OUT, "ranking.csv"), ranking_rows)
write_csv(os.path.join(OUT, "convergence.csv"), conv_rows)
write_csv(os.path.join(OUT, "mc_stats.csv"), mc_stats_rows)
print(f"csv: {len(matrix_rows)} matrix rows, {len(ranking_rows)} ranking rows, "
      f"{len(conv_rows)} convergence rows, {len(mc_stats_rows)} mc_stats rows")


# =============================================================================
# figures
# =============================================================================

def raster_yz(grid, field, ny=120, nz=120, x0=0.0, halfw=0.35):
    """Weighted-leaf raster of a per-cell scalar on the yz plane at x0."""
    c = grid.centers
    sel = np.abs(c[:, 0] - x0) < halfw
    ys, zs, f = c[sel, 1], c[sel, 2], np.asarray(field)[sel]
    # leaf face area in yz: cells are cubic (unigrid and octree leaves),
    # so area = vol^(2/3) -- the robust form for both grid types
    areas = cell_volumes(grid)[sel] ** (2.0 / 3.0)
    yi = np.clip(((ys + 5.0) / 10.0 * ny).astype(int), 0, ny - 1)
    zi = np.clip((zs / 10.0 * nz).astype(int), 0, nz - 1)
    num = np.zeros((nz, ny)); den = np.zeros((nz, ny))
    np.add.at(num, (zi, yi), f * areas)
    np.add.at(den, (zi, yi), areas)
    out = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    return out


# j_err_maps.png -------------------------------------------------------------
plot_vids = [v for v, *_ in SC_VARIANTS[:4]]
fig, axes = plt.subplots(1, len(plot_vids) + 1, figsize=(4.2 * (len(plot_vids) + 1), 3.6))
jref_map = raster_yz(_ref_grid, ref["J"])
im = axes[0].imshow(jref_map, origin="lower", extent=[-5, 5, 0, 10],
                    cmap="viridis", vmin=0, vmax=np.nanmax(jref_map))
axes[0].set_title(f"REF  J")
plt.colorbar(im, ax=axes[0], fraction=0.046)
for ax, vid in zip(axes[1:], plot_vids):
    g, r = _sc_live[vid]
    from matrix_stats import sample_at
    err = (r["J"] - sample_at(g, _ref_grid, ref["J"])) / np.maximum(
        np.abs(sample_at(g, _ref_grid, ref["J"])), 1e-3)
    m = raster_yz(g, err)
    im = ax.imshow(m, origin="lower", extent=[-5, 5, 0, 10], cmap="coolwarm",
                   vmin=-0.15, vmax=0.15)
    ax.set_title(f"{vid}  rel J err")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("J error maps vs fine-octree reference (all-sky ground case)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "j_err_maps.png"), dpi=130)
plt.close(fig)

# z_hist.png -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].hist(np.clip(zn[keep], -8, 8), bins=80, color="tab:blue")
axes[0].set_title("A2 next-flight z: MASKED (F3 noise mask on)")
axes[0].set_xlabel("z")
axes[1].hist(np.clip(zn[np.isfinite(zn)], -8, 8), bins=80, color="tab:red")
axes[1].set_title("A2 next-flight z: UNMASKED")
axes[1].set_xlabel("z")
fig.suptitle("the F3 story: per-cell noise invalidity wrecks naive z", y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "z_hist.png"), dpi=130)
plt.close(fig)

# mc_convergence.png ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ns = [n for n, _ in mc_err]; es = [e for _, e in mc_err]
ax.loglog(ns, es, "o-", label="seed-split power err")
ax.loglog(ns, [es[0] * (n / ns[0]) ** -0.5 for n in ns], "--", label="N^-0.5")
ax.set_xlabel("histories per population N"); ax.set_ylabel("rel. power error")
ax.set_title(f"MC convergence (slope {slope_mc:.2f})")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "mc_convergence.png"), dpi=130)
plt.close(fig)

# j_convergence.png -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.0, 3.6))
pair_rows = [r for r in conv_rows if r["kind"] == "Cauchy pair"]
xs = [12 * 2 ** i for i in range(len(pair_rows))]
ax.loglog(xs, [r["err"] for r in pair_rows], "o-", label="unigrid Cauchy")
tau_rows = [r for r in conv_rows if r["kind"] == "tau rung vs REF"]
ax.loglog([4, 2, 1, 0.5][:len(tau_rows)][::-1],
          [r["err"] for r in tau_rows][::-1], "s-", label="octree vs REF")
ax.set_xlabel("unigrid n  /  octree tau_max")
ax.set_ylabel("rms J error"); ax.set_title("ladder errors (log-log)")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "j_convergence.png"), dpi=130)
plt.close(fig)

# closure.png ------------------------------------------------------------------
cl = [(row["n_cells"], float(row["closure_rel"]), row["id"])
      for row in matrix_rows
      if row["family"] == "SC" and row["closure_rel"] not in ("", None)]
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.loglog([c[0] for c in cl], [max(c[1], 1e-6) for c in cl], "o")
for x, y, vid in cl:
    ax.annotate(vid.split("/")[0], (x, max(y, 1e-6)), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("cells"); ax.set_ylabel("relative closure residual")
ax.set_title("anchor-4 closure residual (F7: face extrapolation dominated)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "closure.png"), dpi=130)
plt.close(fig)

# err_bars.png --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.6))
P_sc = float(np.sum(w2 * J_sc2))
labels, vals, bars = ["SC (oct2)"], [P_sc], [0.0]
for vid in a1:
    r = a1[vid]
    J = mc_j_total(r)
    P = float(np.sum(w2 * J))
    sig_p = float(np.sqrt(np.sum((w2 * r["sigma_J_total"]) ** 2)))
    labels.append(vid.replace("A1/", "")); vals.append(P); bars.append(sig_p)
ax.bar(range(len(vals)), np.array(vals) - P_sc, yerr=bars, capsize=4)
ax.set_xticks(range(len(labels)), labels, rotation=20, fontsize=8)
ax.set_ylabel("absorbed power − SC (same grid)")
ax.set_title("MC variants vs SC (per-cell sigma only: F9 aggregate caveat)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "err_bars.png"), dpi=130)
plt.close(fig)

# capability_matrix.png -------------------------------------------------------
CAP = [
    ("thermal B", "ok", "ok", "ok", "ok", "ok", "ok"),
    ("solar beam", "ok", "ok", "ok", "ok", "ok", "ok"),
    ("Lambertian ground", "ok", "ok", "ok", "ok (M6)", "ok (ground_I)", "ok (M6)"),
    ("variable ground", "raises", "raises", "raises", "raises", "raises", "raises"),
    ("angular quadrature", "42-642", "42-642", "42-642", "none", "binning", "exact HG"),
    ("energy closure", "ok", "ok", "ok (octree)", "n/a (J only)", "ok (stage2 I)", "n/a"),
]
fig, ax = plt.subplots(figsize=(9.5, 3.2))
ax.axis("off")
tbl = ax.table(cellText=[[c for c in row[1:]] for row in CAP],
               rowLabels=[r[0] for r in CAP],
               colLabels=["SC chain", "SC sc", "SC octree", "MC A1", "MC A2", "MC B"],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
ax.set_title("capability matrix after phase M6 (F1 closed)", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "capability_matrix.png"), dpi=130)
plt.close(fig)

print(f"done -> {OUT}/")
