# -*- coding: utf-8 -*-
"""
MC phase M2 validation: stage-2 radiance I(cell, ordinate) on the gaussian
cloud (plan/mc-plan.md section 7).

Case identical to the SC phase-3 cloud (`rt3d.cases.gaussian_cloud`, kappa
capped at omega = 0.9, B = 0, I = 1 entering through the top, periodic x/y),
discretized on the **octree** so that both solver families transport along the
*same* exact characteristic marches — the stage-2 comparison is then purely
statistical (the unigrid SC "chain" sweep is a different path discretization,
see the decisions log).

Pipeline (full MC chain vs SC):

    SC:   lambda_iteration  -> J_sc, S_sc, I_sc(cell, ordinate)
    MC:   j_mc (lambda variant, stage 1) -> J_mc
          stage2_source(J_mc)            -> S_mc (scalar, no beam)
          radiance_from_source(S_mc)     -> I_mc(cell, ordinate)

plus a stage-2-only run that feeds the **SC's own converged source** to the
MC estimator, isolating the stage-2 sampling statistics from the stage-1 J
difference.  Validation: aggregate z-scores of (I_mc - I_sc)/sigma_I ~ N(0,1)
(2-sigma coverage >= 95%), sigma_I ~ 1/sqrt(n_histories), and the deterministic
estimator == SC I at round-off (already asserted in tests/test_mc_radiance.py).
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
from rt3d.directions import icosahedric_grid
from rt3d.mc.forward import j_mc, radiance_from_source, stage2_source
from rt3d.octree import octree_grid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions

case = gaussian_cloud(omega_cap=0.9)
box = case["box"]
dg = icosahedric_grid(n_refine=1)
bc = boundary_conditions(top_I=1.0)

oct_ = octree_grid(box, h0=2.5)
oct_.refine(case["kappa_e"], tau_max=1.0, eps_grad=0.5, max_depth=3)
oct_.balance_21(max_depth=3)
oct_.finalize()
oct_.sample_fields(kappa_e=case["kappa_e"])
ke = oct_.kappa_e
ks = case["kappa_s"](oct_.centers[:, 0], oct_.centers[:, 1],
                     oct_.centers[:, 2])
g = case["g"](oct_.centers[:, 0], oct_.centers[:, 1], oct_.centers[:, 2])
B = np.zeros(oct_.n_leaf)
print(f"octree cloud: n_leaf = {oct_.n_leaf}, kappa max = {ke.max():.3f}")

# column nearest to the cloud-centre axis, one leaf per z level, for the
# profile plots
dxy = np.hypot(oct_.centers[:, 0], oct_.centers[:, 1])
zs = np.round(oct_.centers[:, 2], 9)
col_idx = np.array([np.flatnonzero(zs == zl)[np.argmin(dxy[zs == zl])]
                    for zl in np.unique(zs)])
col_idx = col_idx[np.argsort(oct_.centers[col_idx, 2])]


# %% SC reference: lambda_iteration on the octree cloud
# NOTE: must run WITHOUT Ng (ng=None): with Ng the solver returns the
# Ng-accelerated source, which is NOT the source that produced the returned
# I (it breaks on |S_new - S_swept| < tol, then overwrites S via ng.step).
# The deterministic stage-2 estimator reproduces I exactly only when fed the
# source the sweep actually used.
t0 = time.time()
sc = lambda_iteration(oct_, dg, ke, ks, B, bc, g=g, periodic_box=box,
                      n_iter=600, tol=1e-9, ng=None, jit=False)
t_sc = time.time() - t0
S_sc = np.asarray(sc["S"])
# (direction-resolved (n_leaf, N_dir) form: the per-cell g of the cloud
# makes the HG source direction-dependent; radiance_from_source takes it
# per ordinate, so feed it through unmodified)
print(f"SC lambda_iteration: {sc['n_iter']} it, {t_sc:.1f}s, "
      f"max J = {sc['J'].max():.4f}", flush=True)


# %% MC stage 1: mean intensity J (lambda variant, M1 defaults)
t0 = time.time()
mc1 = j_mc(oct_, ke, ks, B, bc, g=g, periodic_box=box, stage1="lambda",
           n_histories=40_000, n_boundary_histories=40_000, n_batches=100,
           seed=71, n_iter=150, tol=1e-7)
t_j = time.time() - t0
dJ = np.abs(mc1["J"] - sc["J"])
print(f"MC j_mc: {mc1['n_iter']} it, {t_j:.1f}s, "
      f"max|dJ|/J_max = {dJ.max() / sc['J'].max() * 100:.2f}%, "
      f"mean sigma_J/J_max = {mc1['sigma_J'].mean() / sc['J'].max() * 100:.2f}%")


# %% MC stage 2: radiance from the fixed source
# (a) feed the MC's own source -> full MC chain
S_mc = stage2_source(oct_, dg, ke, ks, B, mc1["J"])
t0 = time.time()
mc2 = radiance_from_source(oct_, dg, ke, S_mc, bc, periodic_box=box,
                           estimator="next_flight", n_histories=40,
                           n_batches=8, seed=72)
t_i = time.time() - t0

# (b) feed the SC's converged source -> stage-2 statistics only
t0 = time.time()
mc2b = radiance_from_source(oct_, dg, ke, S_sc, bc, periodic_box=box,
                            estimator="next_flight", n_histories=160,
                            n_batches=16, seed=73)
t_i2 = time.time() - t0

# (c) deterministic estimator on the SC source: must equal SC I at round-off
#     (same marches, same recursion, zero noise)
mc2d = radiance_from_source(oct_, dg, ke, S_sc, bc, periodic_box=box,
                            estimator="deterministic")
det_err = np.max(np.abs(mc2d["I"] - sc["I"])) / np.abs(sc["I"]).max()

# (d) deterministic estimator on the MC source: zero-noise radiance of the
#     stage-1 MC source — the fair reference for the MC-chain tally (its
#     deviation from SC I is the propagated stage-1 J error, see below)
mc2dm = radiance_from_source(oct_, dg, ke, S_mc, bc, periodic_box=box,
                             estimator="deterministic")
print(f"deterministic estimator vs SC I: rel max err = {det_err:.2e}  "
      f"({'OK' if det_err < 1e-9 else 'FAIL'})  (fp round-off level)",
      flush=True)
dJ_chain = np.max(np.abs(mc2dm["I"] - sc["I"])) / np.abs(sc["I"]).max()
print(f"stage-1 propagation: max|I(det, S_MC) - I(SC)|/I_max = "
      f"{dJ_chain * 100:.2f}%  (from the stage-1 J noise floor)", flush=True)


# %% aggregate statistics of the stage-2 comparisons
def zstats(tag, I, sig, I_ref):
    dev = I - I_ref
    scale = np.sqrt(np.mean(I_ref ** 2))
    mask = sig > 1e-3 * scale
    z = dev[mask] / sig[mask]
    cov = np.count_nonzero(np.abs(z) <= 2.0) / z.size
    rms = np.sqrt(np.mean(z ** 2))
    ok = abs(z.mean()) < 0.15 and rms < 1.3 and cov >= 0.92
    print(f"  {tag}: n={z.size}, mean z = {z.mean():+.3f}, rms z = {rms:.3f}, "
          f"2-sigma coverage = {cov * 100:.2f}%  ({'OK' if ok else 'FAIL'})")
    return z, mask


print("stage-2 radiance I(cell, ordinate), strict statistical checks:")
zA, maskA = zstats("MC chain vs det(S_MC),     n_h=40 ", mc2["I"],
                   mc2["sigma_I"], mc2dm["I"])
zB, maskB = zstats("SC source vs det(S_SC),   n_h=160 ", mc2b["I"],
                   mc2b["sigma_I"], mc2d["I"])

# sigma ~ 1/sqrt(n_hist): same seed, 2x histories
mc2q = radiance_from_source(oct_, dg, ke, S_sc, bc, periodic_box=box,
                            estimator="next_flight", n_histories=320,
                            n_batches=16, seed=73)
both = (mc2b["sigma_I"] > 0) & (mc2q["sigma_I"] > 0)
ratio = np.median(mc2b["sigma_I"][both] / mc2q["sigma_I"][both])
print(f"sigma_I scaling n_h 160 -> 320: median ratio = {ratio:.3f}  "
      f"(expect ~1.41, {'OK' if 1.25 < ratio < 1.6 else 'FAIL'})")

# mean intensity of the stage-2 quadrature vs SC J
dJq = np.abs(mc2["J"] - sc["J"])
print(f"stage-2 quadrature J (full MC chain) vs SC J: max|dJ|/J_max = "
      f"{dJq.max() / sc['J'].max() * 100:.2f}%  "
      f"(stage-1 noise floor dominates)")


# %% plot: column profiles + z histogram
d_down = int(np.argmin(dg.directions[:, 2]))            # most downward
d_up = int(np.argmax(dg.directions[:, 2]))              # most upward
z_col = oct_.centers[col_idx, 2]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
ax = axes[0]
ax.plot(sc["J"][col_idx], z_col, "k-", lw=1.6, label="SC 42 ordinates")
ax.plot(mc1["J"][col_idx], z_col, "o", ms=4, mfc="none", color="C0",
        label="MC stage 1")
ax.fill_betweenx(z_col, (mc1["J"] - 2 * mc1["sigma_J"])[col_idx],
                 (mc1["J"] + 2 * mc1["sigma_J"])[col_idx], color="C0",
                 alpha=0.2)
ax.set_xlabel("J (column through cloud centre)")
ax.set_ylabel("z [m]")
ax.set_title("stage 1: mean intensity")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

for ax, d, tag in ((axes[1], d_down, f"downward ord d={d_down}"),
                   (axes[2], d_up, f"upward ord d={d_up}")):
    ax.plot(sc["I"][col_idx, d], z_col, "k-", lw=1.6, label="SC I")
    I, s = mc2b["I"][col_idx, d], mc2b["sigma_I"][col_idx, d]
    ax.plot(I, z_col, "o", ms=3.5, mfc="none", color="C1",
            label="MC stage 2 (SC source)")
    ax.fill_betweenx(z_col, I - 2 * s, I + 2 * s, color="C1", alpha=0.2)
    Id, sd = mc2["I"][col_idx, d], mc2["sigma_I"][col_idx, d]
    ax.plot(Id, z_col, "s", ms=3.5, mfc="none", color="C0",
            label="MC chain (MC source)")
    ax.fill_betweenx(z_col, Id - 2 * sd, Id + 2 * sd, color="C0", alpha=0.15)
    ax.set_xlabel("I(cell, ordinate)")
    ax.set_title(tag)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("MC phase M2: stage-2 radiance on the octree gaussian cloud "
             "(ranges = 2 sigma batch)")
fig.tight_layout()
fig.savefig("plot/mc_forward_cloud.png", dpi=150)
print("  -> plot/mc_forward_cloud.png")

# %% z-score histogram of the stage-2-only run
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(zB, bins=41, range=(-5, 5), density=True, histtype="stepfilled",
        alpha=0.5, label=f"MC stage 2 z-scores (n = {zB.size})")
xx = np.linspace(-5, 5, 201)
ax.plot(xx, np.exp(-0.5 * xx ** 2) / np.sqrt(2 * np.pi), "k-", lw=1.2,
        label="N(0, 1)")
ax.set_xlabel(r"$(I_{MC} - I_{SC})/\sigma_I$")
ax.set_title("M2: next-flight radiance tally, aggregate z")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plot/mc_forward_cloud_zscores.png", dpi=150)
print("  -> plot/mc_forward_cloud_zscores.png")
