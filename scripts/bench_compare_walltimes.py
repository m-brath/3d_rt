# -*- coding: utf-8 -*-
"""
Before/after wall-clock benchmark of the solver optimizations
(plan/optimization-plan.md section 9.5).

Re-solves the cached `scripts/compare_transport.py` runs (the 42-ordinate
configurations) with the current solver and reports

    old wall  (stored in plot/compare_transport/_solve_cache/*.npz,
               measured pre-optimization on this machine)
    new wall  (this run)
    max relative J difference vs the stored solution (validation gate:
    optimization must not change converged solutions beyond
    transcendental-rounding level)

Run from the repo root:
    macOS/Linux:  ~/miniforge3/envs/pyarts/bin/python scripts/bench_compare_walltimes.py
"""

# %% imports and configuration
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.directions import icosahedric_grid
from rt3d.geometry import unigrid, ground_surface
from rt3d.octree import octree_grid
from rt3d.transport import boundary_conditions
from rt3d.solver import lambda_iteration

OUT = str(Path(__file__).resolve().parent.parent / "plot" / "compare_transport")
CACHE_DIR = os.path.join(OUT, "_solve_cache")

case = gaussian_cloud(omega_cap=0.9)
box = case["box"]
dg = icosahedric_grid(n_refine=1)

mu_sun = np.cos(np.radians(30.0))
beam = {"omega_sun": np.array([np.sqrt(1.0 - mu_sun**2), 0.0, -mu_sun]),
        "F0": 1.0}
bc = boundary_conditions(top_I=1.0,
                         ground=ground_surface(albedo=0.3, emissivity=0.0,
                                               B_g=0.0))
N_ITER, TOL = 60, 1e-6


def make_unigrid(n):
    xs = np.linspace(box.x0, box.x1, n + 1)[:-1] + (box.Lx / n) / 2
    zs = np.linspace(box.z0, box.z1, n + 1)[:-1] + (box.Lz / n) / 2
    return unigrid(xs, xs, zs)


def make_octree(tau_max, max_depth):
    o = octree_grid(box, h0=1.25)
    o.refine(case["kappa_e"], tau_max=tau_max, eps_grad=1.0,
             max_depth=max_depth)
    o.balance_21(max_depth=max_depth)
    o.finalize()
    o.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"],
                    g=case["g"], T=case["T"])
    return o


GRIDS = {
    "u24":  make_unigrid(24),
    "u48":  make_unigrid(48),
    "oct3": make_octree(2.0, 3),
    "oct4": make_octree(1.0, 4),
}
IS_OCT = {k: k.startswith("oct") for k in GRIDS}

FIELDS = {}
for gname, g in GRIDS.items():
    c = g.centers
    if IS_OCT[gname]:
        FIELDS[gname] = (g.kappa_e, g.kappa_s, g.g, np.zeros(g.n_leaf))
    else:
        FIELDS[gname] = (case["kappa_e"](*c.T), case["kappa_s"](*c.T),
                         case["g"](*c.T), np.zeros(g.n_cells))

# the cached 42-ordinate compare_transport runs
RUNS = [
    ("uni_chain_dM0",   "u24",  "chain", 0),
    ("uni_chain_dM1",   "u24",  "chain", 1),
    ("uni_sc_dM0",      "u24",  "sc",    0),
    ("uni_sc_dM1",      "u24",  "sc",    1),
    ("octree_dM0",      "oct3", "chain", 0),
    ("octree_dM1",      "oct3", "chain", 1),
    ("uni_chain48_dM1", "u48",  "chain", 1),
    ("uni_sc48_dM1",    "u48",  "sc",    1),
    ("octree_d4_dM1",   "oct4", "chain", 1),
]

# %% re-solve and compare
print(f"{'run':16s} {'n_iter':>6s} {'old wall':>9s} {'new wall':>9s} "
      f"{'speedup':>8s} {'dJ/J':>10s}")
tot_old = tot_new = 0.0
for key, gname, transport, dM in RUNS:
    cf = os.path.join(CACHE_DIR, f"{key}.npz")
    if not os.path.exists(cf):
        print(f"{key:16s}  (no cached reference — skipped)")
        continue
    z = np.load(cf, allow_pickle=True)
    old_wall = float(z["wall"])
    k_e, k_s, g_c, B_f = FIELDS[gname]
    t0 = time.perf_counter()
    r = lambda_iteration(
        GRIDS[gname], dg, k_e, k_s, B_f, bc,
        g=g_c, periodic_box=box, beam=beam,
        n_iter=N_ITER, tol=TOL, ng=True, delta_M=dM, transport=transport,
    )
    new_wall = time.perf_counter() - t0
    J_ref = z["J"]
    dJ = np.max(np.abs(r["J"] - J_ref)) / max(np.max(np.abs(J_ref)), 1e-30)
    tot_old += old_wall
    tot_new += new_wall
    print(f"{key:16s} {r['n_iter']:6d} {old_wall:9.2f} {new_wall:9.2f} "
          f"{old_wall / new_wall:8.2f} {dJ:10.2e}")
print(f"{'TOTAL':16s} {'':6s} {tot_old:9.2f} {tot_new:9.2f} "
      f"{tot_old / tot_new:8.2f}")
