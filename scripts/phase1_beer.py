# -*- coding: utf-8 -*-
# %%
"""
Phase 1 validation (plan section 7): uniform Cartesian short-characteristics
sweep, beam + thermal source, kappa_s = 0, against the Beer-Lambert law.

  1. beam:      I_beam = F0 * exp(-tau_b), tau_b analytic
  2. diffuse:   I(Omega, x) = I_entry * exp(-kappa * path), analytic path
                length along each characteristic, checked for every ordinate
                and every cell (validation anchor 3)
  3. isothermal: kappa_s = 0, B = const, I_in = B on top and bottom
                -> I = B exactly everywhere (transport + source balance)

Run:  & C:/Users/fredd/miniforge3/envs/pyarts/python.exe scripts/phase1_beer.py
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
from rt3d.geometry import cuboid, unigrid
from rt3d.characteristics import build_chain
from rt3d.transport import beam_optical_depth, boundary_conditions, sweep_direction

N_REFINE = 1  # development grid

# %% setup: homogeneous slab, periodic sides (plane-parallel via 1 cell in x,y)
nz = 40
z = np.linspace(0.5, 10.0 - 0.5, nz)
grid = unigrid(np.array([0.0]), np.array([0.0]), z)
box = cuboid(-0.5, 0.5, -0.5, 0.5, 0.0, 10.0)
kappa = 0.7  # 1/km
kappa_e = np.full(grid.n_cells, kappa)
z_top, z_bot = grid.box[2][1], grid.box[2][0]

dg = icosahedric_grid(n_refine=N_REFINE)
print(f"grid: {grid.nx}x{grid.ny}x{grid.nz}, N_dir={dg.n_dir}, kappa_e={kappa}")

# %% 1. beam: Beer-Lambert optical depth
for label, sun in (("vertical", np.array([0.0, 0.0, -1.0])),
                   ("oblique", np.array([0.5, 0.2, -0.8426]))):
    sun = sun / np.linalg.norm(sun)
    tau_b = beam_optical_depth(grid, sun, kappa_e, periodic_box=box)
    tau_exact = kappa * (z_top - grid.centers[:, 2]) / abs(sun[2])
    err = np.max(np.abs(tau_b - tau_exact))
    print(f"beam {label}: max|tau_b - tau_analytic| = {err:.3e}")
    assert err < 1e-10

# vertical beam: I_beam = F0 exp(-tau_b)
F0 = 1.0
sun = np.array([0.0, 0.0, -1.0])
tau_b = beam_optical_depth(grid, sun, kappa_e, periodic_box=box)
I_beam = F0 * np.exp(-tau_b)
I_beam_exact = F0 * np.exp(-kappa * (z_top - grid.centers[:, 2]))
print("beam transmittance: max|I - I_exact| =", np.max(np.abs(I_beam - I_beam_exact)))
assert np.max(np.abs(I_beam - I_beam_exact)) < 1e-12

# %% 2. diffuse field, kappa_s = 0, B = 0: pure attenuation
bc = boundary_conditions(top_I=1.0, bottom_I=0.3, side_I=0.0)
S = np.zeros(grid.n_cells)

max_err = 0.0
for d in range(dg.n_dir):
    chain = build_chain(grid, dg.directions[d], periodic_box=box)
    I = sweep_direction(grid, chain, kappa_e, S, bc)
    vz = chain["vhat"][2]
    # every characteristic enters through a z-face (periodic sides)
    entry_I = np.where(vz < 0, bc.top_I, bc.bottom_I)
    z_entry = np.where(vz < 0, z_top, z_bot)
    path = np.abs(z_entry - grid.centers[:, 2]) / abs(vz)
    I_exact = entry_I * np.exp(-kappa * path)
    err = np.max(np.abs(I - I_exact))
    max_err = max(max_err, err)
print(f"diffuse attenuation: max err over {dg.n_dir} ordinates = {max_err:.3e}")
assert max_err < 1e-12

# %% 3. isothermal equilibrium: S = B, I_in = B on both z-faces -> I = B
B = 2.5
bc = boundary_conditions(top_I=B, bottom_I=B)
S = np.full(grid.n_cells, B)
max_err = 0.0
for d in range(dg.n_dir):
    chain = build_chain(grid, dg.directions[d], periodic_box=box)
    I = sweep_direction(grid, chain, kappa_e, S, bc)
    max_err = max(max_err, np.max(np.abs(I - B)))
print(f"isothermal equilibrium: max|I - B| = {max_err:.3e}")
assert max_err < 1e-12

# %% plot: transmittance through the slab
bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
S = np.zeros(grid.n_cells)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(I_beam, grid.centers[:, 2], "k-", label="beam (vertical)")
for d in range(0, dg.n_dir, 7):
    chain = build_chain(grid, dg.directions[d], periodic_box=box)
    I = sweep_direction(grid, chain, kappa_e, S, bc)
    ax.plot(I, grid.centers[:, 2], ".", ms=3, label=f"ord. {d} (mu={chain['vhat'][2]:+.2f})")
ax.set_xlabel("intensity")
ax.set_ylabel("z  [km]")
ax.set_title(f"phase 1: Beer-Lambert, kappa_e={kappa}, {dg.n_dir} ordinates")
ax.legend(fontsize=7)
fig.savefig("plot/phase1_beer_lambert.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> plot/phase1_beer_lambert.png")

print("phase 1 checks passed")
