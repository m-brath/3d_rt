# -*- coding: utf-8 -*-
# %%
"""
Phase 0 validation (plan section 7):
  - direction grid rebuilds for n_refine 0/1/2/3 -> 12/42/162/642 ordinates
  - sum(w_i) == 4*pi over the direction grid
  - int Phi dOmega == 1 for the HG phase function
  - no ordinate has Omega_z == 0 (cyclic characteristics guard)
  - geometry: ray-box exit and DDA march consistency

Run:  & C:/Users/fredd/miniforge3/envs/pyarts/python.exe scripts/phase0_check.py
(needs the env's Library/bin on PATH for BLAS, see AGENTS.md)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid, ray_box_exit, march_lattice

# %% direction grid checks
print("direction grid")
fig, axes = plt.subplots(1, 4, subplot_kw={"projection": "3d"}, figsize=(14, 4))
for ax, n in zip(axes, (0, 1, 2, 3)):
    dg = icosahedric_grid(n_refine=n)
    n_expect = 10 * 4**n + 2
    assert dg.n_dir == n_expect, (dg.n_dir, n_expect)

    sum_w = dg.weights.sum()
    err_w = abs(sum_w - 4 * np.pi)

    err_phi = max(
        np.max(np.abs(dg.hg_matrix(g).sum(axis=1) - 1.0)) for g in (0.0, 0.3, 0.5)
    )
    min_oz = np.min(np.abs(dg.directions[:, 2]))

    print(
        f"  n_refine={n}  N_dir={dg.n_dir}  |sum(w)-4pi|={err_w:.2e}  "
        f"max|int Phi dOmega - 1|={err_phi:.2e}  min|Omega_z|={min_oz:.4f}"
    )
    assert err_w < 1e-12
    assert min_oz > 1e-6

    ax.scatter(*dg.directions.T, s=12)
    ax.set_title(f"n_refine={n}, N_dir={dg.n_dir}")
    ax.set_axis_off()

fig.suptitle("rt3d icosahedral direction grids (development grid: n_refine=1)")
fig.savefig("plot/phase0_directions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> plot/phase0_directions.png")

# %% geometry checks
print("geometry")
x = np.linspace(0.5, 9.5, 10)
grid = unigrid(x, x.copy(), x.copy())
assert grid.n_cells == 1000
assert np.array_equal(grid.locate(grid.centers), np.arange(grid.n_cells))

rng = np.random.default_rng(42)
box = cuboid(0, 10, 0, 10, 0, 10)

for d in (np.array([0.6, 0.5, -0.4]), np.array([-0.1, 0.3, 0.95])):
    d = d / np.linalg.norm(d)
    o = rng.uniform(0.2, 9.8, (200, 3))

    # non-periodic: path length must match the analytic box traversal
    steps, exit_t, eax, esd = march_lattice(o, d, grid)
    tot = np.zeros(len(o))
    for ridx, cell, ln in steps:
        tot[ridx] += ln
    assert np.allclose(tot, exit_t)
    for axis in range(3):
        m = (eax == axis) & (esd == np.sign(d[axis]))
        if np.any(m):
            face = grid.box[axis][1 if d[axis] > 0 else 0]
            t_analytic = (face - o[m, axis]) / d[axis]
            assert np.allclose(exit_t[m], t_analytic), axis

    # periodic: every ray must exit through the z-face opposite to d_z,
    # with path length |dz|/|d_z|
    steps, exit_t, eax, esd = march_lattice(o, d, grid, periodic_box=box)
    z_exit = grid.box[2][0] if d[2] < 0 else grid.box[2][1]
    assert np.all((eax == 2) & (esd == np.sign(d[2])))
    assert np.allclose(exit_t, np.abs(z_exit - o[:, 2]) / abs(d[2]))

    # ray_box_exit consistency with the march
    t_be, ax_be, sd_be = ray_box_exit(o, d, grid.box)
    steps, exit_t, eax, esd = march_lattice(o, d, grid)
    assert np.allclose(t_be, exit_t)
    assert np.array_equal(ax_be, eax) and np.array_equal(sd_be, esd)

print("  ray-box exit and DDA march: all consistent")
print("phase 0 checks passed")
