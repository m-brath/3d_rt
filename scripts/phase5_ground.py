# -*- coding: utf-8 -*-
"""
Phase 5 validation (plan section 7): periodic sides + Lambertian ground +
all-sky output; energy conservation (validation anchor 4).

Cells:
  1. Beer-Lambert slab with Lambertian ground albedo: per-ordinate
     analytic check on unigrid (lattice-stencil characteristic) and on
     the octree (exact characteristics); ground-emission case.
  2. Gaussian cloud on the uniform 24^3 grid with ground albedo, direct
     beam and top illumination (Ng): pointwise div F vs heating rate
     4 pi kappa_a (B - J) and global energy balance including the beam
     and ground terms.
  3. All-sky output images at the top face (upward ordinates) saved to
     plot/phase5_allsky.npz.
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
from rt3d.diagnostics import (
    allsky,
    energy_balance,
    flux_divergence,
    heating_rate,
)
from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, ground_surface, unigrid
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.source import flux_vector, mean_intensity
from rt3d.transport import (
    beam_irradiance_ground,
    beam_optical_depth,
    boundary_conditions,
)

N_REFINE = 1
dg = icosahedric_grid(n_refine=N_REFINE)

# %% 1. Beer-Lambert + Lambertian ground (analytic per ordinate)
print("=" * 70)
print("1. Beer-Lambert slab + ground albedo: per-ordinate analytic check")

x = (np.arange(4) + 0.5) * 0.5
y = (np.arange(4) + 0.5) * 0.5
z = (np.arange(8) + 0.5) * 2.0 / 8
sgrid = unigrid(x, y, z)
sbox = cuboid(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
ke = np.full(sgrid.n_cells, 1.0)
ks = np.zeros(sgrid.n_cells)
z1 = sgrid.box[2][1]


def slab_ref(A, vhat_z, z_bot):
    """Discrete analytic I: kappa_s = 0, B = 0, top_I = 1, ground albedo A."""
    mu = dg.directions[:, 2]
    F_down = np.sum(
        dg.weights[dg.down_idx] * np.abs(mu[dg.down_idx])
        * np.exp(-(z1 - z_bot) / np.abs(vhat_z[dg.down_idx]))
    )
    I_g = A * F_down / np.pi
    zc = sgrid.centers[:, 2]
    return np.where(
        vhat_z[None, :] < 0.0,
        np.exp(-(z1 - zc[:, None]) / np.abs(vhat_z[None, :])),
        I_g * np.exp(-zc[:, None] / vhat_z[None, :]),
    )


vz = np.array([
    build_chain(sgrid, dg.directions[d], periodic_box=sbox)["vhat"][2]
    for d in range(dg.n_dir)
])
for A in (0.3, 0.9):
    bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=A))
    r = lambda_iteration(sgrid, dg, ke, ks, np.zeros(sgrid.n_cells), bc,
                         periodic_box=sbox, n_iter=5, tol=1e-12)
    err = np.max(np.abs(r["I"] - slab_ref(A, vz, sgrid.z[0])))
    print(f"  unigrid  A={A}: max|I - analytic| = {err:.3e}")

oct = octree_grid(sbox, h0=0.5)
oct.refine(lambda xx, yy, zz: np.full_like(zz, 1.0), tau_max=1e6,
           eps_grad=1e6, max_depth=0)
oct.balance_21(max_depth=0)
oct.finalize()
oke = np.full(oct.n_leaf, 1.0)
for A in (0.3, 0.9):
    bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=A))
    r = lambda_iteration(oct, dg, oke, np.zeros(oct.n_leaf),
                         np.zeros(oct.n_leaf), bc, periodic_box=sbox,
                         n_iter=5, tol=1e-12)
    mu = dg.directions[:, 2]
    zc = oct.centers[:, 2]
    F_down = np.sum(
        dg.weights[dg.down_idx] * np.abs(mu[dg.down_idx])
        * np.exp(-(z1 - oct.h0 / 2) / np.abs(mu[dg.down_idx]))
    )
    I_g = A * F_down / np.pi
    I_ref = np.where(
        mu[None, :] < 0.0,
        np.exp(-(z1 - zc[:, None]) / np.abs(mu[None, :])),
        I_g * np.exp(-zc[:, None] / mu[None, :]),
    )
    err = np.max(np.abs(r["I"] - I_ref))
    print(f"  octree   A={A}: max|I - analytic| = {err:.3e}")

bc = boundary_conditions(ground=ground_surface(albedo=0.0, B_g=1.0))
r = lambda_iteration(sgrid, dg, ke, ks, np.zeros(sgrid.n_cells), bc,
                     periodic_box=sbox, n_iter=5, tol=1e-12)
zc = sgrid.centers[:, 2]
I_ref = np.where(vz[None, :] > 0.0, np.exp(-zc[:, None] / vz[None, :]), 0.0)
print(f"  ground emission (eps=1, B_g=1): max|I - analytic| = "
      f"{np.max(np.abs(r['I'] - I_ref)):.3e}")

# %% 2. Gaussian cloud + ground + beam: energy conservation
print("=" * 70)
print("2. cloud 24^3, omega <= 0.9, ground albedo 0.3, beam + top_I:")
print("   div F vs heating rate, global energy balance (Ng)")

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
bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=A_g))
sun = np.array([0.3, 0.2, -0.9])
sun = sun / np.linalg.norm(sun)
F0 = 1.0
beam = {"omega_sun": sun, "F0": F0}

t0 = time.time()
res = lambda_iteration(grid, dg, grid.kappa_e, grid.kappa_s,
                       np.zeros(grid.n_cells), bc, g=grid.g, beam=beam,
                       periodic_box=box, n_iter=1000, tol=1e-6, ng=True)
print(f"  Ng: {res['n_iter']} iters, converged={res['converged']}"
      f"  ({time.time() - t0:.0f}s)")

kappa_a = grid.kappa_e - grid.kappa_s
V = (grid.x[1] - grid.x[0]) * (grid.y[1] - grid.y[0]) * (grid.z[1] - grid.z[0])
A_col = (grid.x[1] - grid.x[0]) * (grid.y[1] - grid.y[0])

# pointwise anchor 4: div F = 4 pi kappa_a (B - J)
F = flux_vector(res["I"], dg)
divF = flux_divergence(grid, F)
heat = heating_rate(kappa_a, res["J"], 0.0)
k = np.arange(grid.n_cells) // (grid.ny * grid.nx)
interior = (k > 0) & (k < grid.nz - 1)
scale = np.abs(heat).max()
err_max = np.abs(divF[interior] - heat[interior]).max() / scale
err_rms = np.sqrt(np.mean((divF[interior] - heat[interior]) ** 2)) / scale
print(f"  pointwise div F vs heating (interior): max {err_max * 100:.1f}%, "
      f"rms {err_rms * 100:.1f}% of max|heating|")

# global balance, two independent closures:
#  (1) diffuse field: net outward face flux + gas absorption of the
#      diffuse field = beam first-scatter input (ground exchange is
#      contained in the net bottom face flux)
#  (2) beam: incident power = gas absorption + first scattering +
#      power reaching the ground
bal = energy_balance(grid, res["I"], dg, kappa_a, 0.0)
tau_b = beam_optical_depth(grid, sun, grid.kappa_e, periodic_box=box)
A_dom = grid.nx * grid.ny * A_col
beam_scat = (grid.kappa_s * F0 * np.exp(-tau_b)).sum() * V
gas_beam = (kappa_a * F0 * np.exp(-tau_b)).sum() * V
beam_gnd = beam_irradiance_ground(grid, sun, grid.kappa_e, tau_b, F0)
beam_gnd_pow = beam_gnd.sum() * A_col
in_beam = F0 * abs(sun[2]) * box.Lx * box.Ly
in_diff = np.sum(dg.weights[dg.down_idx]
                 * np.abs(dg.directions[dg.down_idx, 2])) * A_dom
gas_diff = -bal["heating"]
ground_abs = bal["out_bottom"] + (1.0 - A_g) * beam_gnd_pow

res_dif = (bal["out_top"] + bal["out_bottom"] + gas_diff - beam_scat)
res_bim = in_beam - gas_beam - beam_scat - beam_gnd_pow
print("  diffuse closure: out_top + out_bottom + gas(diffuse) - beam_scat")
print(f"    out_top {bal['out_top']:9.2f}  out_bottom {bal['out_bottom']:9.2f}"
      f"  gas(diffuse) {gas_diff:9.2f}  beam_scat {beam_scat:9.2f}")
print(f"    residual {res_dif:9.2f}  ({abs(res_dif) / (in_diff + beam_scat) * 100:.2f}% of input)")
print("  beam closure: in = gas + scattered + ground")
print(f"    in {in_beam:9.2f}  gas {gas_beam:9.2f}  scat {beam_scat:9.2f}"
      f"  ground {beam_gnd_pow:9.2f}   residual {res_bim:9.2f}"
      f"  ({abs(res_bim) / in_beam * 100:.2f}%)")
total_in = in_diff + in_beam
total_out = (bal["out_top"] + in_diff) + gas_diff + gas_beam + ground_abs
print(f"  total: in {total_in:9.2f} (diffuse {in_diff:.2f} + beam {in_beam:.2f})"
      f"   out {total_out:9.2f} (up-through-top {bal['out_top'] + in_diff:.2f}"
      f" + gas + ground)   closure "
      f"{abs(total_in - total_out) / total_in * 100:.2f}%")

# %% 3. all-sky output images (top face, upward ordinates)
print("=" * 70)
print("3. all-sky images at top face")

sky = allsky(grid, res["I"], dg, face="top")
npz = Path("plot/phase5_allsky.npz")
np.savez(npz, omega=sky["omega"], weights=sky["weights"],
         images=sky["images"], n_refine=N_REFINE,
         x=grid.x, y=grid.y, z=grid.z, ground_albedo=A_g)
print(f"  {sky['images'].shape[0]} upward ordinates, images "
      f"{sky['images'].shape[1:]} -> {npz}")

# %% 4. plot
fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(1, 3, 1)
zc = sgrid.centers[:, 2]
d_vert = dg.up_idx[np.argmax(dg.directions[dg.up_idx, 2])]
bc = boundary_conditions(top_I=1.0, ground=ground_surface(albedo=0.9))
r1 = lambda_iteration(sgrid, dg, ke, ks, np.zeros(sgrid.n_cells), bc,
                      periodic_box=sbox, n_iter=5, tol=1e-12)
ref1 = slab_ref(0.9, vz, sgrid.z[0])
col = 0
ax.plot(r1["I"][col::16, d_vert], zc[col::16], "k-", label="solver")
ax.plot(ref1[col::16, d_vert], zc[col::16], "--x", label="analytic")
ax.set_xlabel("I (upward ordinate)")
ax.set_ylabel("z [m]")
ax.set_title("Beer-Lambert + ground A=0.9")
ax.legend()

ax = fig.add_subplot(1, 3, 2)
F4 = np.column_stack([divF, heat]).reshape(grid.nz, grid.ny, grid.nx, 2)
ax.plot(F4[:, 12, 12, 0], grid.z, "k-", label="div F")
ax.plot(F4[:, 12, 12, 1], grid.z, "--x", label="4$\\pi\\kappa_a$(B-J)")
ax.set_xlabel("per volume")
ax.set_ylabel("z [m]")
ax.set_title("anchor 4: div F = heating (column)")
ax.legend()

ax = fig.add_subplot(1, 3, 3)
Fz = F[:, 2].reshape(grid.nz, grid.ny, grid.nx).mean(axis=(1, 2))
ax.plot(Fz, grid.z, "k-")
ax.axvline(0.0, color="gray", lw=0.5)
ax.set_xlabel("x,y-mean F_z")
ax.set_ylabel("z [m]")
ax.set_title("net flux profile (ground + beam)")

mu_up = sky["omega"][:, 2]
pick = np.argsort(-mu_up)[:6]
for n, d in enumerate(pick):
    ax = fig.add_axes([0.08 + n * 0.152, 0.06, 0.13, 0.28])
    im = ax.imshow(sky["images"][d], origin="lower", cmap="inferno")
    ax.set_title(f"$\\mu$={mu_up[d]:.2f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
fig.text(0.08, 0.36, "all-sky images at top face (6 most vertical ordinates)",
         fontsize=9)

fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.42,
                    wspace=0.35, hspace=0.45)
fig.savefig("plot/phase5_ground.png", dpi=130)
print("plot written to plot/phase5_ground.png")
