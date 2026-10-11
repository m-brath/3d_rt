# -*- coding: utf-8 -*-
"""
Phase 4 validation (plan section 7): 1D Lambda_bar Ng acceleration -
convergence rate vs plain lambda iteration as omega -> 1.

Cells:
  1. Plane-parallel slab (g = 0), omega in {0.5, 0.9, 0.99}: iteration
     counts LI vs Ng, convergence histories, J agreement.
  2. Gaussian cloud on the uniform 24^3 grid, omega capped at 0.9 (higher
     than the phase 3 run at 0.7 - Ng makes the near-conservative core
     tractable), g varying per cell: LI vs Ng counts, J profiles.
  3. Same cloud on the phase 3 octree (exact-characteristic sweeps) with
     Ng: count + agreement with the uniform Ng solution (within the
     documented unigrid stencil error).
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

from rt3d.accelerator import ng1d
from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions

N_REFINE = 1
DELTA_M = 0  # delta-M order (0 = off; probes Ng x delta-M interaction in 1.)
dg = icosahedric_grid(n_refine=N_REFINE)

# %% 1. slab: omega -> 1
print("=" * 70)
print("1. slab (tau=5, g=0): LI vs Ng iteration counts as omega -> 1")

KAPPA = 1.0
TAU = 5.0
NZ = 50
HZ = TAU / KAPPA / NZ
hx = HZ
x = (np.arange(4) + 0.5) * hx
y = (np.arange(4) + 0.5) * hx
z = (np.arange(NZ) + 0.5) * HZ
sgrid = unigrid(x, y, z)
zs = z
sbox = cuboid(0.0, 4 * hx, 0.0, 4 * hx, sgrid.box[2][0], sgrid.box[2][1])
sbc = boundary_conditions(top_I=1.0)

hist_li99 = hist_ng99 = None
J_slab = {}
print("  omega   LI iters   Ng iters   speedup   max|dJ|/J_max")
for omega in (0.5, 0.9, 0.99):
    ke = np.full(sgrid.n_cells, KAPPA)
    ks = np.full(sgrid.n_cells, omega)
    t0 = time.time()
    r_li = lambda_iteration(sgrid, dg, ke, ks, np.zeros(sgrid.n_cells), sbc,
                            periodic_box=sbox, n_iter=3000, tol=1e-8,
                            delta_M=DELTA_M)
    t_li = time.time() - t0
    t0 = time.time()
    r_ng = lambda_iteration(sgrid, dg, ke, ks, np.zeros(sgrid.n_cells), sbc,
                            periodic_box=sbox, n_iter=3000, tol=1e-8, ng=True,
                            delta_M=DELTA_M)
    t_ng = time.time() - t0
    J1 = r_li["J"].reshape(NZ, 4, 4).mean(axis=(1, 2))
    J2 = r_ng["J"].reshape(NZ, 4, 4).mean(axis=(1, 2))
    rel = np.max(np.abs(J1 - J2)) / J1.max()
    print(f"  {omega:5.2f}   {r_li['n_iter']:8d}   {r_ng['n_iter']:8d}"
          f"   {r_li['n_iter'] / r_ng['n_iter']:7.1f}   {rel * 100:.3f}%"
          f"   ({t_li:.0f}s / {t_ng:.0f}s)")
    J_slab[omega] = (J1, J2)
    if omega == 0.99:
        hist_li99, hist_ng99 = r_li["history"], r_ng["history"]

# %% 2. Gaussian cloud, uniform 24^3, omega cap 0.9, g varying
print("=" * 70)
print("2. Gaussian cloud 24^3, omega <= 0.9, g varying: LI vs Ng")

case = gaussian_cloud(omega_cap=0.9)
box = case["box"]
bc = boundary_conditions(top_I=1.0)
x = np.linspace(-4.7916667, 4.7916667, 24)
y = np.linspace(-4.7916667, 4.7916667, 24)
z = np.linspace(0.2083333, 9.7916667, 24)
grid = unigrid(x, y, z)
gc = grid.centers
grid.kappa_e = case["kappa_e"](gc[:, 0], gc[:, 1], gc[:, 2])
grid.kappa_s = case["kappa_s"](gc[:, 0], gc[:, 1], gc[:, 2])
grid.g = case["g"](gc[:, 0], gc[:, 1], gc[:, 2])

t0 = time.time()
r_li = lambda_iteration(grid, dg, grid.kappa_e, grid.kappa_s,
                        np.zeros(grid.n_cells), bc, g=grid.g,
                        periodic_box=box, n_iter=1000, tol=1e-6)
print(f"  LI:  {r_li['n_iter']} iters, converged={r_li['converged']}"
      f"  ({time.time() - t0:.0f}s)")
t0 = time.time()
r_ng = lambda_iteration(grid, dg, grid.kappa_e, grid.kappa_s,
                        np.zeros(grid.n_cells), bc, g=grid.g,
                        periodic_box=box, n_iter=1000, tol=1e-6, ng=True)
print(f"  Ng:  {r_ng['n_iter']} iters, converged={r_ng['converged']}"
      f"  ({time.time() - t0:.0f}s)")

p_li = r_li["J"].reshape(grid.nz, grid.ny, grid.nx).mean(axis=(1, 2))
p_ng = r_ng["J"].reshape(grid.nz, grid.ny, grid.nx).mean(axis=(1, 2))
rel = np.max(np.abs(p_ng - p_li)) / p_li.max()
print(f"  cloud J profile: max|dJ|/J_max = {rel * 100:.3f}%")
F_top_li = r_li["F"][-grid.ny * grid.nx:, 2].mean()
F_top_ng = r_ng["F"][-grid.ny * grid.nx:, 2].mean()
print(f"  mean F_z top layer: LI {F_top_li:.4f}, Ng {F_top_ng:.4f}")

# %% 3. octree (exact characteristics) + Ng
print("=" * 70)
print("3. octree tau_max=2, eps=1, depth 3 + Ng")

oct = octree_grid(box, h0=1.25)
oct.refine(case["kappa_e"], tau_max=2.0, eps_grad=1.0, max_depth=3)
oct.balance_21(max_depth=3)
oct.finalize()
oct.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"],
                  g=case["g"])

t0 = time.time()
r_oct = lambda_iteration(oct, dg, oct.kappa_e, oct.kappa_s,
                         np.zeros(oct.n_leaf), bc, g=oct.g,
                         periodic_box=box, n_iter=1000, tol=1e-6, ng=True)
print(f"  octree Ng: {r_oct['n_iter']} iters, converged={r_oct['converged']}"
      f"  ({time.time() - t0:.0f}s)")

zedges = np.linspace(0.0, 10.0, 25)
zk = oct.centers[:, 2]
idx = np.clip(np.digitize(zk, zedges[1:-1]), 0, len(zedges) - 2)
w = (2.0 * oct.half) ** 2
prof = np.zeros(len(zedges) - 1)
cnt = np.zeros(len(zedges) - 1)
np.add.at(prof, idx, r_oct["J"] * w)
np.add.at(cnt, idx, w)
prof /= np.maximum(cnt, 1e-300)
m = cnt > 0
rel = np.max(np.abs(prof[m] - p_ng[m])) / p_ng[m].max()
print(f"  octree-Ng vs uniform-Ng: max|dJ|/J_max = {rel * 100:.2f}%"
      "  (dominated by the unigrid stencil error, see phase 3)")

# %% 4. plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

it = np.arange(1, len(hist_li99) + 1)
axes[0].semilogy(it, hist_li99, label=f"LI ({len(hist_li99)} iters)")
it = np.arange(1, len(hist_ng99) + 1)
axes[0].semilogy(it, hist_ng99, label=f"Ng ({len(hist_ng99)} iters)")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("rel. source change")
axes[0].set_title("slab, $\\omega$ = 0.99")
axes[0].legend()

for omega in (0.5, 0.99):
    J1, J2 = J_slab[omega]
    axes[1].plot(J1, zs, "-", color="k" if omega == 0.5 else "C1")
    axes[1].plot(J2, zs, "--x", color="k" if omega == 0.5 else "C1",
                 label=f"$\\omega$={omega}: LI solid, Ng x")
axes[1].set_xlabel("J (slab mean)")
axes[1].set_ylabel("z [m]")
axes[1].set_title("slab J: LI vs Ng indistinguishable")
axes[1].legend()

axes[2].plot(p_li, grid.z, "k-", label="cloud 24$^3$ LI")
axes[2].plot(p_ng, grid.z, "--x", label="cloud 24$^3$ Ng")
axes[2].plot(prof[m], zedges[:-1][m] + 0.2083, ":", label="octree Ng")
axes[2].set_xlabel("x,y-mean J")
axes[2].set_ylabel("z [m]")
axes[2].set_title("Gaussian cloud, $\\omega \\leq$ 0.9")
axes[2].legend()

fig.tight_layout()
fig.savefig("plot/phase4_ng.png", dpi=130)
print("plot written to plot/phase4_ng.png")
