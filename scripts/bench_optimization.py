# -*- coding: utf-8 -*-
"""
Performance benchmark for the rt3d solver (plan/optimization-plan.md).

Reproduces the measured baseline table of the optimization plan:

    octree precompute (march) cost + segment memory        (h0 = 1, depth 4)
    octree exact-sweep cost per ordinate
    balance_21 build cost
    unigrid chain / SC sweep cost (30^3, 50^3) + chain memory
    scattering matmul, scalar g and g_cells
    stencil_for_direction cost

Every stage is timed; memory is measured from the array sizes.  Options
select which stages run so the script is usable as a quick regression check.

Run from the repo root:
    macOS/Linux:  ~/miniforge3/envs/pyarts/bin/python scripts/bench_optimization.py
"""

# %% imports and configuration
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.characteristics import build_chain, stencil_for_direction
from rt3d.directions import icosahedric_grid
from rt3d.geometry import unigrid
from rt3d.octree import octree_grid
from rt3d.short_characteristics import (
    build_sc_chain,
    sweep_all_sc,
    sweep_direction_sc,
)
from rt3d.source import scattering_source
from rt3d.transport import (
    boundary_conditions,
    precompute_octree_march,
    sweep_all,
    sweep_direction_octree_exact,
)

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all",
                choices=["all", "octree", "unigrid", "source", "stencil"])
ap.add_argument("--dirs", type=int, default=6,
                help="number of ordinates to time the octree march for "
                     "(extrapolated to 42)")
args = ap.parse_args()

case = gaussian_cloud(omega_cap=0.7)
box = case["box"]
dg = icosahedric_grid(n_refine=1)
print(f"N_dir = {dg.n_dir}")

bc = boundary_conditions(top_I=1.0)


def timeit(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def arr_mb(a):
    return a.nbytes / 1e6


# %% octree build + balance
if args.stage in ("all", "octree"):
    t0 = time.perf_counter()
    oct = octree_grid(box, h0=1.0)
    oct.refine(case["kappa_e"], tau_max=0.5, eps_grad=0.3, max_depth=4)
    t_refine = time.perf_counter() - t0
    oct._compact()
    t0 = time.perf_counter()
    oct.balance_21(max_depth=4)
    t_balance = time.perf_counter() - t0
    oct.finalize()
    oct.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"],
                      g=case["g"], T=case["T"])
    print(f"octree: n_leaf={oct.n_leaf}, max_level={oct.max_level}")
    print(f"  refine     {t_refine:8.2f} s")
    print(f"  balance_21 {t_balance:8.2f} s")

    # %% octree march precompute + sweep (reference vs fused numba)
    k_e = oct.kappa_e
    S = np.random.default_rng(0).random(oct.n_leaf)
    n_probe = min(args.dirs, dg.n_dir)
    mt, st = 0.0, 0.0
    M_tot = 0
    for d in range(n_probe):
        m, dt = timeit(precompute_octree_march, oct, dg.directions[d],
                       periodic_box=box, jit=False)
        mt += dt
        M_tot += m["leaf"].size
        seg_mb = (arr_mb(m["leaf"]) + arr_mb(m["length"])
                  + arr_mb(m["ray_of"]) + arr_mb(m["offsets"]))
        I, dt = timeit(sweep_direction_octree_exact, oct, m, k_e, S, bc)
        st += dt
    per_dir = mt / n_probe
    print(f"octree march precompute (NumPy ref): {per_dir:6.2f} s/dir  "
          f"-> {per_dir * dg.n_dir:7.1f} s for {dg.n_dir} dirs")
    print(f"  segments/dir ~{M_tot / n_probe / 1e6:5.1f}M   "
          f"memory/dir ~{seg_mb:6.0f} MB -> {42 * seg_mb / 1e3:5.1f} GB for 42 dirs")
    print(f"octree exact sweep (NumPy ref):  {st / n_probe:6.3f} s/dir  "
          f"-> {st / n_probe * dg.n_dir:6.1f} s/iteration")

    mtj, stj = 0.0, 0.0
    for d in range(n_probe):
        m, dt = timeit(precompute_octree_march, oct, dg.directions[d],
                       periodic_box=box)
        mtj += dt
        I, dt = timeit(sweep_direction_octree_exact, oct, m, k_e, S, bc)
        stj += dt
    print(f"octree fused numba: precompute {mtj / n_probe:6.4f} s/dir "
          f"-> {mtj / n_probe * dg.n_dir:6.1f} s for {dg.n_dir} dirs, "
          f"sweep {stj / n_probe:6.3f} s/dir -> {stj / n_probe * dg.n_dir:6.1f} s/iter")

# %% unigrid chain + sc
if args.stage in ("all", "unigrid"):
    for n in (30, 50):
        xs = np.linspace(box.x0, box.x1, n + 1)[:-1] + (box.Lx / n) / 2
        grid = unigrid(xs, xs, xs)
        c = grid.centers
        k_e = case["kappa_e"](*c.T)
        S = np.random.default_rng(0).random(grid.n_cells)

        bt = 0.0
        chains = []
        for d in range(dg.n_dir):
            ch, dt = timeit(build_chain, grid, dg.directions[d],
                            periodic_box=box)
            bt += dt
            chains.append(ch)
        chain_mb = arr_mb(chains[0]["rel"]) + arr_mb(chains[0]["sub_delta"])
        _, ct = timeit(sweep_all, grid, dg, k_e, S, bc,
                       periodic_box=box, chains=chains)
        _, ct8 = timeit(sweep_all, grid, dg, k_e, S, bc,
                        periodic_box=box, chains=chains, n_jobs=8)
        print(f"unigrid {n}^3: build_chain(42) {bt:6.2f} s, "
              f"chain geom {chain_mb:7.1f} MB/dir, "
              f"chain sweep_all {ct:6.3f} s/iter (n_jobs=8: {ct8:6.3f})")

        bt = 0.0
        scs = []
        for d in range(dg.n_dir):
            sc, dt = timeit(build_sc_chain, grid, dg.directions[d],
                            periodic_box=box)
            bt += dt
            scs.append(sc)
        _, st_ = timeit(sweep_all_sc, grid, dg, k_e, S, bc,
                        periodic_box=box, scs=scs)
        _, st8 = timeit(sweep_all_sc, grid, dg, k_e, S, bc,
                        periodic_box=box, scs=scs, n_jobs=8)
        print(f"unigrid {n}^3: build_sc_chain(42) {bt:6.2f} s, "
              f"sc sweep_all {st_:6.3f} s/iter (n_jobs=8: {st8:6.3f})")

# %% scattering source
if args.stage in ("all", "source"):
    n = oct.n_leaf if args.stage == "all" else 489_180
    n = max(n, 1)
    rng = np.random.default_rng(1)
    I = rng.random((n, dg.n_dir))
    omega = rng.random(n) * 0.7
    B = np.zeros(n)
    g_c = np.clip(rng.random(n) * 0.95, 0.0, 0.952)
    cache = {}
    _, dt = timeit(scattering_source, I, dg, omega, B, g=0.6, Phi_cache=cache)
    print(f"scattering matmul scalar g ({n} cells): {dt:.3f} s")
    cache = {}
    _, dt = timeit(scattering_source, I, dg, omega, B, g_cells=g_c,
                   Phi_cache=cache)
    print(f"scattering matmul g_cells ({n} cells): {dt:.3f} s  "
          f"({len(np.unique(np.round(g_c, 2)))} buckets)")

# %% stencil search
if args.stage in ("all", "stencil"):
    omega = dg.directions[7]
    for r in (12, 24):
        _, dt = timeit(stencil_for_direction, omega, np.array([1.0, 1.0, 0.2]),
                       r)
        print(f"stencil_for_direction r={r}: {dt:.3f} s")
