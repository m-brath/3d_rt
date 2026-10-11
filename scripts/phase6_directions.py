# -*- coding: utf-8 -*-
"""
Phase 6 validation (plan section 7): direction-grid convergence study
42 -> 162 -> 642 ordinates, plus the performance pass.

Cells:
  1. Gaussian cloud on the uniform 24^3 grid with ground albedo, direct
     beam and top illumination (phase 5 setup, Ng) run on n_refine =
     1 / 2 / 3: angular convergence of the x,y-mean J profile and of the
     top-of-domain flux.
  2. Uniform slab with strongly forward-peaked scattering (g = 0.952,
     the cloud-core value): quantifies the known limitation that 42
     ordinates under-resolves the HG lobe (plan section 2).
  3. Performance pass: chain-build time and one full transport sweep
     wall time vs N_dir (expected linear scaling).
  4. Plots + npz record (n_refine recorded in every output).
"""

# %% imports and setup
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.characteristics import build_chain
from rt3d.diagnostics import energy_balance
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, ground_surface, unigrid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions, sweep_direction

N_REFINE = (1, 2, 3)  # 42 / 162 / 642 ordinates


def rel_diff(a, b):
    """max|a - b| / max|b| (b = finer grid)."""
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def profile(res, grid):
    """x,y-mean J profile vs z."""
    return res["J"].reshape(grid.nz, grid.ny, grid.nx).mean(axis=(1, 2))


# %% 1. cloud 24^3 + ground + beam on 42 / 162 / 642 ordinates
print("=" * 70)
print("1. Gaussian cloud 24^3, ground albedo 0.3, beam + top_I (Ng):")
print("   angular convergence of J and top-of-domain flux")

case = gaussian_cloud(omega_cap=0.9)
box = case["box"]
x = np.linspace(-4.7916667, 4.7916667, 24)
y = np.linspace(-4.7916667, 4.7916667, 24)
z = np.linspace(0.2083333, 9.7916667, 24)
grid = unigrid(x, y, z)
gc = grid.centers
grid.kappa_e = case["kappa_e"](gc[:, 0], gc[:, 1], gc[:, 2])
grid.kappa_s = case["kappa_s"](gc[:, 0], gc[:, 1], gc[:, 2])
grid.g = case["g"](gc[:, 0], gc[:, 1], gc[:, 2])

A_g = 0.3
sun = np.array([0.3, 0.2, -0.9])
sun = sun / np.linalg.norm(sun)
beam = {"omega_sun": sun, "F0": 1.0}

cloud = {}
for n in N_REFINE:
    dg = icosahedric_grid(n_refine=n)
    bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=A_g))
    t0 = time.time()
    res = lambda_iteration(grid, dg, grid.kappa_e, grid.kappa_s,
                           np.zeros(grid.n_cells), bc, g=grid.g, beam=beam,
                           periodic_box=box, n_iter=1000, tol=1e-6, ng=True)
    wall = time.time() - t0
    bal = energy_balance(grid, res["I"], dg, grid.kappa_e - grid.kappa_s,
                         np.zeros(grid.n_cells))
    cloud[n] = {
        "J_prof": profile(res, grid),
        "J": res["J"],
        "out_top": bal["out_top"],
        "out_bottom": bal["out_bottom"],
        "gas": -bal["heating"],
        "n_iter": res["n_iter"],
        "wall": wall,
    }
    print(f"  n_refine={n} ({dg.n_dir:4d} ord): {res['n_iter']:3d} iters, "
          f"{wall:6.1f}s   out_top {bal['out_top']:8.3f}  "
          f"gas {bal['heating']:8.3f}  bottom {bal['out_bottom']:8.3f}")

print("  angular convergence (relative to next-finer grid):")
for a, b in zip(N_REFINE[:-1], N_REFINE[1:]):
    dJ = rel_diff(cloud[a]["J_prof"], cloud[b]["J_prof"])
    dJf = np.max(np.abs(cloud[a]["J"] - cloud[b]["J"])) / np.max(cloud[b]["J"])
    dF = abs(cloud[a]["out_top"] - cloud[b]["out_top"]) / abs(
        cloud[b]["out_top"])
    print(f"    {10 * 4 ** a + 2:4d} -> {10 * 4 ** b + 2:4d}:  "
          f"max|dJ(z)|/max J = {dJ * 100:5.2f}%   "
          f"max|dJ(cell)|/max J = {dJf * 100:5.2f}%   "
          f"d(out_top)/out_top = {dF * 100:5.2f}%")
print("  differences vs finest grid (642):")
for a in N_REFINE[:-1]:
    dJ = rel_diff(cloud[a]["J_prof"], cloud[3]["J_prof"])
    dJf = np.max(np.abs(cloud[a]["J"] - cloud[3]["J"])) / np.max(
        cloud[3]["J"])
    dF = abs(cloud[a]["out_top"] - cloud[3]["out_top"]) / abs(
        cloud[3]["out_top"])
    print(f"    {10 * 4 ** a + 2:4d} vs 642:  max|dJ(z)|/max J = "
          f"{dJ * 100:5.2f}%   max|dJ(cell)|/max J = {dJf * 100:5.2f}%   "
          f"d(out_top)/out_top = {dF * 100:5.2f}%")
d42_162 = rel_diff(cloud[1]["J_prof"], cloud[2]["J_prof"])
d162_642 = rel_diff(cloud[2]["J_prof"], cloud[3]["J_prof"])
print(f"  ratio d(42->162)/d(162->642) = {d42_162 / d162_642:.2f}")

# %% 2. forward-peaked slab (g = 0.952): HG lobe under-resolution
print("=" * 70)
print("2. uniform slab tau=10, omega=0.9, g=0.952, top_I=1:")
print("   42 ordinates under-resolve the HG lobe (plan section 2)")

xs = (np.arange(4) + 0.5) * 0.5
ys = (np.arange(4) + 0.5) * 0.5
zs = (np.arange(24) + 0.5) * 10.0 / 24
sgrid = unigrid(xs, ys, zs)
sbox = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 10.0)
ke = np.full(sgrid.n_cells, 1.0)
ks = 0.9 * ke
sg = np.full(sgrid.n_cells, 0.952)

slab = {}
for n in N_REFINE:
    dg = icosahedric_grid(n_refine=n)
    bc = boundary_conditions(top_I=1.0)
    t0 = time.time()
    res = lambda_iteration(sgrid, dg, ke, ks, np.zeros(sgrid.n_cells), bc,
                           g=sg, periodic_box=sbox, n_iter=1000, tol=1e-8,
                           ng=True)
    wall = time.time() - t0
    bal = energy_balance(sgrid, res["I"], dg, ke - ks,
                         np.zeros(sgrid.n_cells))
    slab[n] = {
        "J_prof": profile(res, sgrid),
        "out_top": bal["out_top"],
        "n_iter": res["n_iter"],
        "wall": wall,
    }
    print(f"  n_refine={n} ({dg.n_dir:4d} ord): {res['n_iter']:3d} iters, "
          f"{wall:6.1f}s   out_top {bal['out_top']:8.4f}")

print("  angular convergence:")
for a, b in zip(N_REFINE[:-1], N_REFINE[1:]):
    dJ = rel_diff(slab[a]["J_prof"], slab[b]["J_prof"])
    dF = abs(slab[a]["out_top"] - slab[b]["out_top"]) / abs(
        slab[b]["out_top"])
    print(f"    {10 * 4 ** a + 2:4d} -> {10 * 4 ** b + 2:4d}:  "
          f"max|dJ(z)|/max J = {dJ * 100:5.2f}%   "
          f"d(out_top)/out_top = {dF * 100:5.2f}%")
print("  differences vs finest grid (642):")
for a in N_REFINE[:-1]:
    dJ = rel_diff(slab[a]["J_prof"], slab[3]["J_prof"])
    dF = abs(slab[a]["out_top"] - slab[3]["out_top"]) / abs(
        slab[3]["out_top"])
    print(f"    {10 * 4 ** a + 2:4d} vs 642:  max|dJ(z)|/max J = "
          f"{dJ * 100:5.2f}%   d(out_top)/out_top = {dF * 100:5.2f}%")
d42_162s = rel_diff(slab[1]["J_prof"], slab[2]["J_prof"])
d162_642s = rel_diff(slab[2]["J_prof"], slab[3]["J_prof"])
print(f"  ratio d(42->162)/d(162->642) = {d42_162s / d162_642s:.2f} "
      f"(non-monotone: 42-ordinate error cancellation at g=0.952)")

# %% 3. performance pass: chain build + one full sweep vs N_dir
print("=" * 70)
print("3. performance pass (cloud 24^3 grid): chain build + one sweep")

S0 = np.ones(grid.n_cells)
bc = boundary_conditions(top_I=1.0)
perf = {}
for n in N_REFINE:
    dg = icosahedric_grid(n_refine=n)
    t0 = time.time()
    chains = [build_chain(grid, dg.directions[d], periodic_box=box)
              for d in range(dg.n_dir)]
    t_chain = time.time() - t0
    t0 = time.time()
    for d in range(dg.n_dir):
        sweep_direction(grid, chains[d], grid.kappa_e, S0, bc)
    t_sweep = time.time() - t0
    perf[dg.n_dir] = (t_chain, t_sweep)
    print(f"  {dg.n_dir:4d} ord: chains {t_chain:6.2f}s   "
          f"sweep pass {t_sweep:6.2f}s   "
          f"({t_sweep / dg.n_dir * 1e3:5.2f} ms/ordinate)")
print(f"  sweep time ratio 642/42 = {perf[642][1] / perf[42][1]:.2f} "
      f"(N_dir ratio = {642 / 42:.2f}, linear expected)")

# %% 4. record + plot
np.savez(Path("plot/phase6_directions.npz"),
         n_refine=np.array(N_REFINE),
         n_dir=np.array([10 * 4 ** n + 2 for n in N_REFINE]),
         z_cloud=grid.z,
         J_cloud=np.array([cloud[n]["J_prof"] for n in N_REFINE]),
         out_top_cloud=np.array([cloud[n]["out_top"] for n in N_REFINE]),
         z_slab=sgrid.z,
         J_slab=np.array([slab[n]["J_prof"] for n in N_REFINE]),
         out_top_slab=np.array([slab[n]["out_top"] for n in N_REFINE]))

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(2, 3, 1)
for n in N_REFINE:
    ax.plot(cloud[n]["J_prof"], grid.z, label=f"{10 * 4 ** n + 2} ord")
ax.set_xlabel("x,y-mean J")
ax.set_ylabel("z [m]")
ax.set_title("cloud: J profiles")
ax.legend(fontsize=8)

ax = fig.add_subplot(2, 3, 2)
ax.plot(np.abs(cloud[2]["J_prof"] - cloud[1]["J_prof"]), grid.z,
        label="|J162 - J42|")
ax.plot(np.abs(cloud[3]["J_prof"] - cloud[2]["J_prof"]), grid.z,
        label="|J642 - J162|")
ax.set_xlabel("|dJ|")
ax.set_ylabel("z [m]")
ax.set_title("cloud: angular differences")
ax.legend(fontsize=8)

ax = fig.add_subplot(2, 3, 3)
ax.bar(["42->162", "162->642"],
       [d42_162 * 100, d162_642 * 100], color=["steelblue", "darkorange"])
ax.set_ylabel("max|dJ(z)| / max J  [%]")
ax.set_title("cloud: convergence")

ax = fig.add_subplot(2, 3, 4)
for n in N_REFINE:
    ax.plot(slab[n]["J_prof"], sgrid.z, label=f"{10 * 4 ** n + 2} ord")
ax.set_xlabel("x,y-mean J")
ax.set_ylabel("z [m]")
ax.set_title("slab g=0.952: J profiles")
ax.legend(fontsize=8)

ax = fig.add_subplot(2, 3, 5)
ax.plot(np.abs(slab[2]["J_prof"] - slab[1]["J_prof"]), sgrid.z,
        label="|J162 - J42|")
ax.plot(np.abs(slab[3]["J_prof"] - slab[2]["J_prof"]), sgrid.z,
        label="|J642 - J162|")
ax.plot(np.abs(slab[3]["J_prof"] - slab[1]["J_prof"]), sgrid.z, ":",
        label="|J642 - J42|")
ax.set_xlabel("|dJ|")
ax.set_ylabel("z [m]")
ax.set_title("slab: angular differences (HG lobe)")
ax.legend(fontsize=8)

ax = fig.add_subplot(2, 3, 6)
nd = np.array(sorted(perf))
ax.loglog(nd, [perf[k][1] for k in nd], "o-", label="sweep pass")
ax.loglog(nd, [perf[k][0] for k in nd], "s--", label="chain build")
ax.loglog(nd, perf[42][1] * nd / 42, ":", label="linear in N_dir")
ax.set_xlabel("N_dir")
ax.set_ylabel("wall time [s]")
ax.set_title("performance (24^3)")
ax.legend(fontsize=8)

fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07,
                    wspace=0.35, hspace=0.45)
fig.savefig("plot/phase6_directions.png", dpi=130)
print("plot written to plot/phase6_directions.png")
