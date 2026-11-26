# -*- coding: utf-8 -*-
"""
MC phase M3 validation: backward (adjoint) detector MC (plan/mc-plan.md
section 4, phase M2/M3 validation table).

Two parts on the SAME problems the forward family was validated on:

  1. **All-sky consistency** — at a point in the top layer of (a) the 1D
     slab and (b) the octree gaussian cloud, one point detector per upward
     icosahedral ordinate **at the exact node direction** (cone = 0, the
     default of `backward_all_sky`) is run and differenced against the
     converged SC ``lambda_iteration`` radiance of that cell/ordinate,
     plus a hemispheric **upward-flux closure** (adjoint vs SC-42 vs an
     SC refine-2 / 162-ordinate strong reference).  Full icosahedral caps
     cross the horizon at near-horizontal nodes (they are full-sphere
     apertures) and are therefore not differenced against hemisphere-only
     node values.  The adjoint walk solves the *closed* problem (no
     stage-1 J input), so the difference is MC statistics + the SC-42
     angular quadrature error, with an rms-sky-scaled allowance.

  2. **Efficiency at a narrow sensor** — one 3-degree cone detector at the
     cloud-top pixel; the adjoint walk variance-vs-cost is compared with
     the forward stage-2 next-flight tally on the *same single ray* at
     equal collision cost, and the whole-all-sky-image cost is stated: the
     forward stage 2 transports every cell x every ordinate, the adjoint
     only the detector cones (plan section 4 headline argument).
"""

# %% imports and helpers
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.cases.slab_1d import slab_grid
from rt3d.directions import icosahedric_grid
from rt3d.mc.backward import all_sky_cone_half, backward_all_sky, backward_detector_mc
from rt3d.mc.tracking import EV_EXIT, _exit_distance, sample_collisions
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions, precompute_octree_march

dg = icosahedric_grid(n_refine=1)
up_idx = np.flatnonzero(dg.directions[:, 2] > 0.0)
mu_up = dg.directions[up_idx, 2]


def fwd_single_ray_nextflight(oct_, c, d, kappa_e, S_dir, bc, periodic_box,
                              n_h, n_batches, seed):
    """
    Forward stage-2 next-flight estimator restricted to ONE ray (leaf
    center c, ordinate d) — the per-pixel/per-ordinate core of
    `rt3d.mc.forward.radiance_from_source`, used here for the cost
    comparison.  Returns (I, sigma_I, n_collisions).
    """
    om = dg.directions[d]
    march = precompute_octree_march(oct_, om, periodic_box=periodic_box,
                                    jit=False)
    offs = march["offsets"]
    seg = slice(offs[c], offs[c + 1])
    leaf, ln = march["leaf"][seg], march["length"][seg]
    tau = kappa_e[leaf] * ln
    tau_tot = float(tau.sum())
    I_in = bc.top_I if om[2] < 0.0 else bc.bottom_I
    uncol = I_in * np.exp(-tau_tot)

    p0 = oct_.centers[c][None, :].repeat(n_h, axis=0)
    dirs = np.tile(-om, (n_h, 1))
    bud = _exit_distance(p0, dirs, oct_, periodic_box=periodic_box)
    res = sample_collisions(p0, dirs, bud, oct_, kappa_e, kappa_e,
                            mode="cell", periodic_box=periodic_box, seed=seed)
    real = res["coll_event"] != EV_EXIT
    idx = np.where(real, res["coll_cell"], 0)
    dep = np.where(real, S_dir[idx, d], 0.0)
    nb = max(1, min(n_batches, n_h))
    per = n_h // nb
    if nb > 1 and per * nb == n_h:
        bs = dep[:per * nb].reshape(nb, per).sum(axis=1)
        est = float(bs.sum() / n_h)
        sig = float(np.sqrt(nb) * bs.std(ddof=1) / n_h)
    else:
        est, sig = float(dep.mean()), 0.0
    n_coll = int(np.count_nonzero(real))
    return uncol + est, sig, n_coll


# %% part 1a: slab all-sky consistency (exact node directions, cone = 0)
grid, box, hz = slab_grid(4.0, 2.0, nz=16, nxy=4)
nc = grid.n_cells
ke = np.full(nc, 2.0)
ks = np.full(nc, 1.0)                     # omega = 0.5
B = np.full(nc, 0.3)
bc = boundary_conditions(top_I=1.0)
g_slab = 0.3

sc_s = lambda_iteration(grid, dg, ke, ks, B, bc, g=g_slab, periodic_box=box,
                        n_iter=800, tol=1e-10)
pt_s = (grid.x[1], grid.y[1], grid.z[-1])
c_s = ((grid.nz - 1) * grid.ny + 1) * grid.nx + 1
t0 = time.time()
sky_s = backward_all_sky(grid, ke, ks, B, bc, dg, pt_s, g=g_slab,
                         periodic_box=box, n_seeds=20000, n_batches=20,
                         seed=31)
t_sky_s = time.time() - t0
Iref_s = sc_s["I"][c_s, up_idx]
dev_s = np.abs(sky_s["I"] - Iref_s)
rms_s = np.sqrt(np.mean(Iref_s ** 2))
band_s = 2.0 * sky_s["sigma_I"] + 0.05 * rms_s
F_adj_s = float(dg.weights[up_idx] @ (mu_up * sky_s["I"]))
F_sc_s = float(dg.weights[up_idx] @ (mu_up * Iref_s))
print(f"slab all-sky (node dirs): {np.count_nonzero(dev_s <= band_s)}/{up_idx.size} "
      f"inside 2 sigma + 5% rms-sky, max dev/range = "
      f"{(dev_s / band_s).max():.2f} ({t_sky_s:.1f}s, 21 x 20000 seeds)")
print(f"slab hemispheric upward flux at detector: adjoint {F_adj_s:.5f}, "
      f"SC {F_sc_s:.5f}  (rel {100 * (F_adj_s - F_sc_s) / F_sc_s:+.2f}%)")

# %% part 1b: octree cloud all-sky consistency
case = gaussian_cloud(omega_cap=0.9)
cbox = case["box"]
oct_ = octree_grid(cbox, h0=2.5)
oct_.refine(case["kappa_e"], tau_max=1.0, eps_grad=0.5, max_depth=3)
oct_.balance_21(max_depth=3)
oct_.finalize()
oct_.sample_fields(kappa_e=case["kappa_e"])
ko = oct_.kappa_e
ks_o = case["kappa_s"](oct_.centers[:, 0], oct_.centers[:, 1],
                       oct_.centers[:, 2])
g_o = case["g"](oct_.centers[:, 0], oct_.centers[:, 1], oct_.centers[:, 2])
B_o = np.zeros(oct_.n_leaf)
bc_c = boundary_conditions(top_I=1.0)

sc_c = lambda_iteration(oct_, dg, ko, ks_o, B_o, bc_c, g=g_o,
                        periodic_box=cbox, n_iter=600, tol=1e-9, ng=None,
                        jit=False)
# strong angular reference: refine-2 icosahedral set (162 ordinates).  The
# SC-42 per-ordinate top-of-cloud radiance carries large angular
# quadrature error in the strongly anisotropic (g ~ 0.55) cloud — SC42 vs
# SC162 differs by orders of magnitude at near-zero sky pixels — so the
# per-node range uses an rms-sky-scaled absolute floor and the 162-ordinate
# column is shown as reference.  All-sky runs use exact node directions
# (cone_scale = 0): full icosahedral caps (half-angle 24.6 deg) cross the
# horizon at near-horizontal nodes and pick up top-lit DOWNWARD radiance,
# so cone averages are not comparable to hemisphere-only node values.
dg2 = icosahedric_grid(n_refine=2)
sc_c2 = lambda_iteration(oct_, dg2, ko, ks_o, B_o, bc_c, g=g_o,
                         periodic_box=cbox, n_iter=400, tol=1e-8, ng=True,
                         jit=False)
print(f"SC refine-2 (162 ordinates): {sc_c2['n_iter']} it")
S_c = np.asarray(sc_c["S"])
top_z = oct_.centers[:, 2].max()
cand = np.flatnonzero(np.abs(oct_.centers[:, 2] - top_z) < 1e-9)
c_c = cand[np.argmin(np.hypot(oct_.centers[cand, 0], oct_.centers[cand, 1]))]
pt_c = tuple(oct_.centers[c_c])
t0 = time.time()
sky_c = backward_all_sky(oct_, ko, ks_o, B_o, bc_c, dg, pt_c, g=g_o,
                         periodic_box=cbox, n_seeds=20000, n_batches=20,
                         seed=32)
t_sky_c = time.time() - t0
Iref_c = sc_c["I"][c_c, up_idx]
j2 = np.array([int(np.argmax(dg2.directions @ dg.directions[i]))
               for i in up_idx])
Iref_c2 = sc_c2["I"][c_c, j2]
dev_c = np.abs(sky_c["I"] - Iref_c)
band_c = 2.0 * sky_c["sigma_I"] + 0.10 * np.sqrt(np.mean(Iref_c ** 2))
inside_c = np.count_nonzero(dev_c <= band_c)
print(f"cloud all-sky (node dirs): {inside_c}/{up_idx.size} inside "
      f"2 sigma + 10% rms-sky, median dev/range = "
      f"{np.median(dev_c / band_c):.2f} ({t_sky_c:.1f}s)")
F_adj_c = float(dg.weights[up_idx] @ (mu_up * sky_c["I"]))
F_sc_c = float(dg.weights[up_idx] @ (mu_up * Iref_c))
F_sc_c2 = float(dg.weights[up_idx] @ (mu_up * Iref_c2))
print(f"cloud hemispheric upward flux at pixel: adjoint {F_adj_c:.5f}, "
      f"SC-42 {F_sc_c:.5f} ({100 * (F_sc_c - F_sc_c2) / F_sc_c2:+.2f}% vs "
      f"SC-162), SC-162 (node-interp) {F_sc_c2:.5f}  ->  adjoint vs SC-162: "
      f"{100 * (F_adj_c - F_sc_c2) / F_sc_c2:+.2f}%")

# %% part 2: narrow-sensor efficiency (3 deg cone at the cloud-top pixel)
cone = np.deg2rad(3.0)
d_targ = up_idx[np.argmax(mu_up)]           # most vertical upward ordinate
omega0 = dg.directions[d_targ]

adj_cost, adj_sig = [], []
for n_seed in (1_000, 5_000, 25_000, 100_000, 400_000):
    det = {"point": pt_c, "omega0": omega0, "cone": cone, "n_seeds": n_seed}
    r = backward_detector_mc(oct_, ko, ks_o, B_o, bc_c, det, g=g_o,
                             periodic_box=cbox, n_batches=50, seed=33)
    adj_cost.append(r["n_collisions"])
    adj_sig.append(r["sigma_I"] / r["I"])

fwd_cost, fwd_sig = [], []
for n_h in (20, 100, 500, 2_000, 10_000):
    If, sf, ncoll = fwd_single_ray_nextflight(oct_, c_c, d_targ, ko, S_c,
                                              bc_c, cbox, n_h, 50, seed=34)
    fwd_cost.append(ncoll)
    fwd_sig.append(sf / abs(If))
print("narrow-cone 3deg at cloud top "
      f"(SC I = {sc_c['I'][c_c, d_targ]:.4f}):")
for (ac, asg), (fc, fsg) in zip(zip(adj_cost, adj_sig), zip(fwd_cost, fwd_sig)):
    print(f"  adjoint: {ac:>9d} collisions  rel sigma {asg * 100:6.2f}%  |  "
          f"forward 1-ray: {fc:>7d} collisions  rel sigma {fsg * 100:6.2f}%")
print("  reading: at EQUAL single-ray cost the fixed-source next-flight "
      "tally scores once per history and is slightly cheaper per collision;\n"
      "  the adjoint walk's win is the WHOLE-IMAGE cost below (no per-cell x "
      "per-ordinate ray set, no stage-1 J iteration at all).")

# whole-all-sky-image cost at equal per-pixel relative accuracy
n_up = up_idx.size
print(f"all-sky image cost factor: forward stage 2 transports "
      f"n_leaf x N_dir x n_h = {oct_.n_leaf} x {dg.n_dir} x n_h flights "
      f"(= {oct_.n_leaf * dg.n_dir} rays) for ALL cells; the adjoint image "
      f"needs only {n_up} detector cones x n_seeds walks -> "
      f"{oct_.n_leaf * dg.n_dir // n_up}x fewer rays for this pixel column")

# %% figures
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2),
                         gridspec_kw={"height_ratios": [1.4, 1]})

ax = axes[0, 0]
order = np.argsort(mu_up)
x = np.arange(up_idx.size)
ax.errorbar(x, sky_s["I"][order], yerr=2 * sky_s["sigma_I"][order], fmt="o",
            ms=4, mfc="none", color="C0", label="adjoint all-sky (node dirs)")
ax.plot(x, Iref_s[order], "k-x", ms=4, lw=1.2, label="SC 42 ordinates")
ax.set_ylabel("top-layer upward radiance")
ax.set_title("slab: all-sky at top-layer cell center")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.errorbar(x, sky_c["I"][order], yerr=2 * sky_c["sigma_I"][order], fmt="o",
            ms=4, mfc="none", color="C0", label="adjoint all-sky (node dirs)")
ax.plot(x, Iref_c[order], "k-x", ms=4, lw=1.2, label="SC, 42 ordinates")
ax.plot(x, Iref_c2[order], "+", ms=6, color="C3", label="SC, 162 ordinates")
ax.set_ylabel("cloud-top upward radiance")
ax.set_title("octree gaussian cloud: all-sky at cloud-top pixel")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(dev_s / band_s, "o-", ms=4, color="C1", label="slab")
ax.plot(dev_c / band_c, "s-", ms=4, color="C3", label="cloud")
ax.axhline(1.0, color="k", lw=1, ls=":")
ax.set_yscale("log")
ax.set_ylabel("|dev| / (2 sigma + allowance)")
ax.set_xlabel("upward ordinate (sorted by mu)")
ax.set_title("consistency: adjoint vs SC radiance")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1, 1]
ax.loglog(adj_cost, np.array(adj_sig) * 100, "o-", color="C0",
          label="adjoint walk (3$^\\circ$ cone)")
ax.loglog(fwd_cost, np.array(fwd_sig) * 100, "s-", color="C2",
          label="forward stage-2 next-flight (same ray)")
gx = np.array(adj_cost, dtype=float)
ax.loglog(gx, adj_sig[0] * 100.0 * np.sqrt(adj_cost[0] / gx), ls=":",
          color="gray", label=r"$\propto 1/\sqrt{\mathrm{collisions}}$")
ax.set_xlabel("real collisions (cost proxy)")
ax.set_ylabel("relative sigma of sensor radiance  [%]")
ax.set_title("narrow-sensor efficiency at cloud top")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle("MC phase M3: backward adjoint detector MC — all-sky consistency "
             "and narrow-sensor efficiency")
fig.tight_layout()
fig.savefig("plot/mc_backward.png", dpi=150)
print("  -> plot/mc_backward.png")
