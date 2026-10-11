# -*- coding: utf-8 -*-
"""
Phase 3 validation (plan section 6, validation anchor 5):
octree vs uniform grid on the class_test.py Gaussian cloud.

The octree transport sweep follows EXACT characteristics: one backward
DDA march per leaf center and ordinate, precomputed once and reused by
every iteration (rt3d.transport.precompute_octree_march).  The
approximate center-to-center chain (build_chain_octree) is still built
and checked in cell 1, but was dropped from the solver: its hop
alignment tolerance makes its path error in heterogeneous 3D fields
scale-invariant (tau error ~4 on this cloud, not shrinking with
refinement), while the unigrid lattice stencil is worse by orders of
magnitude for grazing ordinates (transmittance factor ~5e4 vs ~5).

Checks:
  1. Single-level octree chain vs exact ray path, kappa_s = 0: the chain
     path lengths stay close to the true ray (chain code sanity).
  2. Gaussian cloud, kappa_s = 0:
     a) octree (exact characteristics) on the SAME piecewise-constant
        kappa field as the uniform grid -> the J difference is the
        uniform grid's lattice-stencil path error;
     b) octree on the continuous field (eps_grad series): octree
        self-convergence (successive series members agree) and column
        tau -> 10 (the coarse uniform reference under-resolves the
        cloud, column tau 9.58).
  3. Gaussian cloud with scattering (omega capped at 0.7 so plain lambda
     iteration converges tractably; Ng is phase 4), same piecewise-constant
     fields on both grids: J profiles and top-of-domain flux.

Grids: uniform 24^3 (h = 0.4167) reference; octree base h0 = 1.25 with
tau_max = 2 and eps_grad in {2, 1, 0.5} at max_depth 3.
"""

# %% imports and case setup
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.characteristics import build_chain, build_chain_octree
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid
from rt3d.octree import march_octree, octree_grid
from rt3d.solver import lambda_iteration
from rt3d.source import mean_intensity
from rt3d.transport import (
    boundary_conditions,
    precompute_octree_march,
    sweep_direction,
    sweep_direction_octree,
    sweep_direction_octree_exact,
)

N_REFINE = 1
dg = icosahedric_grid(n_refine=N_REFINE)
case = gaussian_cloud()
box = case["box"]
bc = boundary_conditions(top_I=1.0)

# %% 1. single-level octree vs exact ray path, kappa_s = 0
print("=" * 70)
print("1. single-level octree vs exact ray path, kappa_s = 0, uniform kappa")

small_box = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
oct1 = octree_grid(small_box, h0=0.5)
oct1.finalize()
oct1.sample_fields(kappa_e=lambda x, y, z: np.ones(x.size))

bc2 = boundary_conditions(top_I=1.0, bottom_I=1.0)
S0 = np.zeros(oct1.n_leaf)
dev = []
for d in dg.directions:
    chain = build_chain_octree(oct1, d, periodic_box=small_box)
    I_oct = sweep_direction_octree(oct1, chain, oct1.kappa_e, S0, bc2)
    z = oct1.centers[:, 2]
    t_e = np.where(d[2] < 0.0, (2.0 - z) / (-d[2]), (z - 0.0) / d[2])
    sr, sl, sLen = march_octree(oct1.centers, -d, oct1, t_e, periodic_box=small_box)
    L_true = np.zeros(oct1.n_leaf)
    np.add.at(L_true, sr, sLen)
    with np.errstate(divide="ignore"):
        L_oct = -np.log(np.maximum(I_oct, 1e-300))
    dev.append(np.max(np.abs(L_oct - L_true)))
print(f"  max chain-vs-ray path deviation over all ordinates: {max(dev):.3f} m")
print(f"  mean over ordinates: {np.mean(dev):.3f} m")

dev_exact = []
for d in dg.directions:
    march = precompute_octree_march(oct1, d, periodic_box=small_box)
    I_oct = sweep_direction_octree_exact(oct1, march, oct1.kappa_e, S0, bc2)
    z = oct1.centers[:, 2]
    t_e = np.where(d[2] < 0.0, (2.0 - z) / (-d[2]), (z - 0.0) / d[2])
    dev_exact.append(np.max(np.abs(I_oct - np.exp(-t_e))))
print(f"  exact-march sweep vs exp(-t_e): max dev {max(dev_exact):.2e}")

# %% 2. Gaussian cloud grids
print("=" * 70)
print("2. Gaussian cloud: uniform 24^3 reference + octree series")

x = np.linspace(-4.7916667, 4.7916667, 24)
y = np.linspace(-4.7916667, 4.7916667, 24)
z = np.linspace(0.2083333, 9.7916667, 24)
grid = unigrid(x, y, z)
gc = grid.centers
grid.kappa_e = case["kappa_e"](gc[:, 0], gc[:, 1], gc[:, 2])
grid.kappa_s = case["kappa_s"](gc[:, 0], gc[:, 1], gc[:, 2])
grid.g = case["g"](gc[:, 0], gc[:, 1], gc[:, 2])

octs = {}
configs = [(2.0, 2.0, 3), (2.0, 1.0, 3), (2.0, 0.5, 3)]
for tau_max, eps, depth in configs:
    t0 = time.time()
    oct = octree_grid(box, h0=1.25)
    oct.refine(case["kappa_e"], tau_max=tau_max, eps_grad=eps, max_depth=depth)
    oct.balance_21(max_depth=depth)
    oct.finalize()
    oct.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"],
                      g=case["g"])
    octs[(tau_max, eps)] = oct
    print(f"  octree tau_max={tau_max}, eps={eps}, depth={depth}: "
          f"{oct.n_leaf} leaves (build {time.time() - t0:.1f}s)")


def zprofile_oct(oct, values, zedges):
    """x,y-area-averaged profile of a leaf quantity on the octree
    (leaf horizontal cross-section as weight - a plain leaf average would
    over-weight finely resolved regions).

    Returns (profile, leaf count per bin); bins outside the octree's
    z-coverage (coarsest base-cell centers sit half a cell below the
    top / above the bottom face) are empty and must be excluded.
    """
    zk = oct.centers[:, 2]
    idx = np.clip(np.digitize(zk, zedges[1:-1]), 0, len(zedges) - 2)
    w = (2.0 * oct.half) ** 2
    prof = np.zeros(len(zedges) - 1)
    cnt = np.zeros(len(zedges) - 1)
    np.add.at(prof, idx, values * w)
    np.add.at(cnt, idx, w)
    return prof / np.maximum(cnt, 1e-300), cnt


def zprofile_uni(grid, values):
    """x,y-averaged profile on the unigrid, per cell layer."""
    return values.reshape(grid.nz, grid.ny, grid.nx).mean(axis=(1, 2))


def uni_sampler(grid, values):
    """Callable returning the unigrid's piecewise-constant cell value of
    `values` at any point - lets the octree discretize the SAME field."""
    V = np.asarray(values).reshape(grid.nz, grid.ny, grid.nx)

    def f(x, y, z):
        i = np.clip(np.floor((x - (grid.x[0] - grid.hx / 2)) / grid.hx).astype(int),
                    0, grid.nx - 1)
        j = np.clip(np.floor((y - (grid.y[0] - grid.hy / 2)) / grid.hy).astype(int),
                    0, grid.ny - 1)
        k = np.clip(np.floor((z - (grid.z[0] - grid.hz / 2)) / grid.hz).astype(int),
                    0, grid.nz - 1)
        return V[k, j, i]

    return f


zedges = np.linspace(0.0, 10.0, 25)
zmid_uni = grid.z

# %% 3. kappa_s = 0 transport comparison
print("=" * 70)
print("3. kappa_s = 0: octree vs uniform grid")

S0 = np.zeros(grid.n_cells)
t0 = time.time()
I_uni0 = np.zeros((grid.n_cells, dg.n_dir))
for d_i, d in enumerate(dg.directions):
    chain = build_chain(grid, d, periodic_box=box)
    I_uni0[:, d_i] = sweep_direction(grid, chain, grid.kappa_e, S0, bc)
J_uni0 = mean_intensity(I_uni0, dg)
print(f"  uniform sweep: {time.time() - t0:.1f}s")
prof_uni0 = zprofile_uni(grid, J_uni0)

# 3a. identical piecewise-constant kappa field: the octree sweep is the
#     exact characteristic on this field, so the difference measures the
#     uniform grid's lattice-stencil path error
print("  3a. same piecewise-constant kappa field "
      "(octree exact characteristic vs unigrid stencil)")
ke_same = uni_sampler(grid, grid.kappa_e)
oct_same = octree_grid(box, h0=1.25)
oct_same.refine(case["kappa_e"], tau_max=2.0, eps_grad=1.0, max_depth=3)
oct_same.balance_21(max_depth=3)
oct_same.finalize()
oct_same.sample_fields(kappa_e=ke_same)
t0 = time.time()
I_oct = np.zeros((oct_same.n_leaf, dg.n_dir))
for d_i, d in enumerate(dg.directions):
    march = precompute_octree_march(oct_same, d, periodic_box=box)
    I_oct[:, d_i] = sweep_direction_octree_exact(
        oct_same, march, oct_same.kappa_e, np.zeros(oct_same.n_leaf), bc
    )
J = mean_intensity(I_oct, dg)
prof, cnt = zprofile_oct(oct_same, J, zedges)
m = cnt > 0
rel = np.max(np.abs(prof[m] - prof_uni0[m])) / max(prof_uni0[m].max(), 1e-30)
print(f"    octree {oct_same.n_leaf} leaves: max|dJ|/J_max = {rel * 100:.2f}%  "
      f"({time.time() - t0:.0f}s)")

# 3b. continuous field series: octree self-convergence (successive
#     members of the series) and the field-sampling difference vs the
#     coarse uniform reference
print("  3b. continuous kappa field (octree series)")
prof_oct0 = {}
for (tau_max, eps), oct in octs.items():
    t0 = time.time()
    I_oct = np.zeros((oct.n_leaf, dg.n_dir))
    for d_i, d in enumerate(dg.directions):
        march = precompute_octree_march(oct, d, periodic_box=box)
        I_oct[:, d_i] = sweep_direction_octree_exact(
            oct, march, oct.kappa_e, np.zeros(oct.n_leaf), bc
        )
    J = mean_intensity(I_oct, dg)
    prof, cnt = zprofile_oct(oct, J, zedges)
    prof_oct0[(tau_max, eps)] = (prof, cnt)
    m = cnt > 0
    rel = np.max(np.abs(prof[m] - prof_uni0[m])) / max(prof_uni0[m].max(), 1e-30)
    print(f"    octree tau_max={tau_max}, eps={eps}: max|dJ|/J_max = {rel * 100:.2f}%  "
          f"({time.time() - t0:.0f}s)")

keys = sorted(prof_oct0)
for a, b in zip(keys, keys[1:]):
    pa, ca = prof_oct0[a]
    pb, cb = prof_oct0[b]
    mm = (ca > 0) & (cb > 0)
    r = np.max(np.abs(pa[mm] - pb[mm])) / max(pa[mm].max(), 1e-30)
    print(f"    series self-diff eps={a[1]} -> eps={b[1]}: "
          f"max|dJ|/J_max = {r * 100:.2f}%")

# %% 4. scattering, lambda iteration (omega capped at 0.7)
print("=" * 70)
print("4. scattering (omega <= 0.7, g varying): octree vs uniform LI")

case_c = gaussian_cloud(omega_cap=0.7)
grid.kappa_s = case_c["kappa_s"](gc[:, 0], gc[:, 1], gc[:, 2])

t0 = time.time()
res_uni = lambda_iteration(
    grid, dg, grid.kappa_e, grid.kappa_s, np.zeros(grid.n_cells), bc,
    g=grid.g, periodic_box=box, n_iter=200, tol=1e-6,
)
print(f"  uniform LI: {res_uni['n_iter']} iters, converged={res_uni['converged']}"
      f"  ({time.time() - t0:.0f}s)")

# same piecewise-constant fields on the octree: isolates solver differences
oct_same.sample_fields(kappa_e=uni_sampler(grid, grid.kappa_e),
                       kappa_s=uni_sampler(grid, grid.kappa_s),
                       g=uni_sampler(grid, grid.g))
t0 = time.time()
res_oct = lambda_iteration(
    oct_same, dg, oct_same.kappa_e, oct_same.kappa_s,
    np.zeros(oct_same.n_leaf), bc,
    g=oct_same.g, periodic_box=box, n_iter=200, tol=1e-6,
)
print(f"  octree LI: {res_oct['n_iter']} iters, converged={res_oct['converged']}"
      f"  ({time.time() - t0:.0f}s)")

p_uni = zprofile_uni(grid, res_uni["J"])
p_oct, cnt_oct = zprofile_oct(oct_same, res_oct["J"], zedges)
m = cnt_oct > 0
rel = np.max(np.abs(p_oct[m] - p_uni[m])) / max(p_uni[m].max(), 1e-30)
print(f"  scattering: max|dJ|/J_max = {rel * 100:.2f}%")

# top-of-domain flux comparison
F_uni_top = res_uni["F"][grid.nz * grid.ny * grid.nx - grid.ny * grid.nx:, 2].mean()
top_leaves = oct_same.centers[:, 2] > 9.0
F_oct_top = res_oct["F"][top_leaves, 2].mean()
print(f"  mean F_z near top: uniform {F_uni_top:.4f}, octree {F_oct_top:.4f}")

# %% 5. plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# kappa_e slice (uniform grid, y=0 layer)
kimg = grid.kappa_e.reshape(grid.nz, grid.ny, grid.nx)[:, grid.ny // 2, :]
axes[0].imshow(kimg.T, origin="lower", extent=[-5, 5, 0, 10],
               cmap="magma", aspect="auto")
axes[0].set_title("kappa_e (uniform 24^3)")
axes[0].set_xlabel("x [m]")

# kappa_s slice
simg = grid.kappa_s.reshape(grid.nz, grid.ny, grid.nx)[:, grid.ny // 2, :]
im1 = axes[1].imshow(simg.T, origin="lower", extent=[-5, 5, 0, 10],
                     cmap="magma", aspect="auto")
axes[1].set_title("kappa_s (omega <= 0.7)")
axes[1].set_xlabel("x [m]")
fig.colorbar(im1, ax=axes[1], fraction=0.046)

axes[2].plot(prof_uni0, zmid_uni, "k-", label="uniform 24$^3$, $\\kappa_s=0$")
zmid_oct = zedges[:-1] + 0.2083
for (tau_max, eps) in sorted(prof_oct0):
    prof, cnt = prof_oct0[(tau_max, eps)]
    axes[2].plot(prof[cnt > 0], zmid_oct[cnt > 0], "--",
                 label=f"octree $\\tau_{{max}}$={tau_max}, $\\varepsilon$={eps}")
axes[2].plot(p_uni, zmid_uni, "k.", label="uniform, scattering")
axes[2].plot(p_oct[m], zmid_oct[m], "x", label="octree, scattering")
axes[2].set_xlabel("mean intensity J (x,y averaged)")
axes[2].set_ylabel("z [m]")
axes[2].set_title("J profiles")
axes[2].legend(fontsize=7)

fig.suptitle(f"phase 3: octree vs uniform, Gaussian cloud "
             f"(N_dir = {dg.n_dir})")
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent.parent / "plot" / "phase3_octree.png",
            dpi=140)
print("plot written to plot/phase3_octree.png")
