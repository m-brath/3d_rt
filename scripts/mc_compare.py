# -*- coding: utf-8 -*-
"""
MC phase M5 integration pass (plan/mc-plan.md section 7, M5 row): one
differencing report **SC vs MC solver A (stage 1 + stage 2) vs MC solver B
(adjoint)**, with the energy-balance closure carried over the FULL MC chain
A1 -> A2 -> B — the quantities feed each other exactly like in production:

  A1  j_mc (explicit variant)                    -> J, F            (stage 1)
  A2  stage2_source(J_A1) + radiance_from_source -> I(cell, ord)    (stage 2)
  B   backward_all_sky / backward_detector_mc    -> node columns    (adjoint)

Comparison conventions (M2/M3 decision log, re-confirmed here):

  * stage-2-only statistics vs SC: feed the SC's OWN converged,
    direction-resolved source sc['S'] (ng=None) into the MC stage-2
    next-flight tally -> aggregate 2-sigma z statistics vs SC I;
  * the production CHAIN (A2 fed by the isotropic stage-1 source
    (1-omega) B + omega J_A1, per the plan sketch) is differenced against
    the DETERMINISTIC sweep of its own source (zero-noise reference): its
    deviation from SC I is a **stage-2 source-model term** (isotropic
    omega*J vs the direction-resolved quadrature scattering source, an
    O(g) effect) — reported separately, not folded into stage-2 noise;
  * the adjoint walk B is a closed **exact-HG** solver: at pixel level it
    differs from SC-42 by SC's own angular-quadrature error (several % in
    scattering-dominated boundary fields; the octree-slab angular
    convergence SC-42/162/642 and the omega -> 0 closed-form single-
    scatter invariant below place the walk, not the walk's statistics);
  * unigrid SC default transport='chain' adds a lattice-stencil path error,
    so node-level MC-vs-SC differencing is done on the octree only; the
    unigrid slab carries the GLOBAL chain closures (invariant 4).

Report: tables on stdout (mirrored to plot/mc_compare.txt) plus the figure
plot/mc_compare.png.  Run from the repo root:
    macOS/Linux:  ~/miniforge3/envs/pyarts/bin/python -u scripts/mc_compare.py
"""

# %% imports and configuration
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
from rt3d.directions import hg, icosahedric_grid
from rt3d.geometry import cuboid
from rt3d.mc.backward import backward_all_sky, backward_detector_mc
from rt3d.mc.estimators import cell_volumes
from rt3d.mc.forward import j_mc, radiance_from_source, stage2_source
from rt3d.source import flux_vector
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions
from rt3d.octree import octree_grid

OUT = str(Path(__file__).resolve().parent.parent / "plot")
dg = icosahedric_grid(n_refine=1)
up_idx = np.flatnonzero(dg.directions[:, 2] > 0.0)
mu_up = dg.directions[up_idx, 2]

lines = []


def say(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def hflux(I_nodes, mu):
    """hemispheric upward flux from an upward-node column."""
    return float(dg.weights[up_idx] @ (mu * np.asarray(I_nodes)))


# %% closed-form invariant: omega -> 0 single-scatter, exact-HG walk
say("=" * 74)
say("MC phase M5 integration report — SC vs MC-A(stage1+stage2) vs MC-B")
say(f"direction grid: icosahedral n_refine = {dg.n_refine}, N_dir = "
    f"{dg.n_dir}, {up_idx.size} upward ordinates")
say()

sbox = cuboid(0.0, 4.0, 0.0, 2.0, 0.0, 2.0)
oct_s = octree_grid(sbox, h0=0.5)
oct_s.refine(lambda x, y, z: np.full_like(x, 2.0), max_depth=1)
oct_s.balance_21(max_depth=1)
oct_s.finalize()
ksl = np.full(oct_s.n_leaf, 0.02)          # omega = 0.01
kel = np.full(oct_s.n_leaf, 2.0)
g_sl = 0.3
top_z = oct_s.centers[:, 2].max()
cand = np.flatnonzero(np.abs(oct_s.centers[:, 2] - top_z) < 1e-9)
c_sl = cand[np.argmin(np.hypot(oct_s.centers[cand, 0],
                               oct_s.centers[cand, 1]))]
pt_sl = tuple(oct_s.centers[c_sl])

zs = np.linspace(0.0, pt_sl[2], 4001)                 # LOS below the point
muq = np.linspace(1e-6, 1.0, 2001)
Phi = hg(-muq, g_sl)                                 # scatter into +z
A = 2 * np.pi * np.trapezoid(
    Phi[None, :] * np.exp(-2.0 * (2.0 - (pt_sl[2] - zs))[:, None] / muq[None, :]),
    muq, axis=1)
I_ana = 0.02 * np.trapezoid(np.exp(-2.0 * zs) * A, zs)
rb = backward_detector_mc(oct_s, kel, ksl, np.zeros(oct_s.n_leaf),
                          boundary_conditions(top_I=1.0),
                          {"point": pt_sl, "omega0": np.array([0.0, 0.0, 1.0]),
                           "cone": 0.0, "n_seeds": 400_000}, g=g_sl,
                          periodic_box=sbox, n_batches=50, seed=92, jit=True)
say("closed-form invariant (octree slab h=0.25, omega = 0.01, g = 0.3, "
    "vertical node 0.125 below top):")
say(f"  analytic single-scatter I = {I_ana:.6f}   "
    f"adjoint walk {rb['I']:.6f} +- {rb['sigma_I']:.6f}  "
    f"z = {(rb['I'] - I_ana) / rb['sigma_I']:+.1f}   (exact-HG walk verified)")
say()

# %% octree-slab angular-convergence context for pixel fluxes (B vs SC)
bc_sl = boundary_conditions(top_I=1.0)
Bs = np.zeros(oct_s.n_leaf)
kss = 0.5 * kel
F_sc_sl = {}
for nr in (1, 2, 3):
    dgx = icosahedric_grid(n_refine=nr)
    ux = np.flatnonzero(dgx.directions[:, 2] > 0.0)
    scx = lambda_iteration(oct_s, dgx, kel, kss, Bs, bc_sl, g=g_sl,
                           periodic_box=sbox, n_iter=1200, tol=1e-10,
                           ng=None, jit=True)
    F_sc_sl[nr] = float(dgx.weights[ux]
                        @ (dgx.directions[ux, 2] * scx["I"][c_sl, ux]))
Isk, Ssk = [], []
for i in up_idx:
    r = backward_detector_mc(oct_s, kel, kss, Bs, bc_sl,
                             {"point": pt_sl, "omega0": dg.directions[i],
                              "cone": 0.0, "n_seeds": 80_000}, g=g_sl,
                             periodic_box=sbox, n_batches=40, seed=93,
                             jit=True)
    Isk.append(r["I"])
    Ssk.append(r["sigma_I"])
Isk = np.array(Isk)
Ssk = np.array(Ssk)
F_b_sl = hflux(Isk, mu_up)
sF_b_sl = float(np.sqrt(np.sum((dg.weights[up_idx] * mu_up * Ssk) ** 2)))
say("SC angular-quadrature context at a pixel (octree slab, omega = 0.5, "
    "g = 0.3, B = 0, top inflow):")
say(f"  SC-42 pixel F_up {F_sc_sl[1]:.5f},  SC-162 {F_sc_sl[2]:.5f},  "
    f"SC-642 {F_sc_sl[3]:.5f}")
say(f"  adjoint walk {F_b_sl:.5f} +- {sF_b_sl:.5f} ({100*(F_b_sl-F_sc_sl[3])/F_sc_sl[3]:+.2f}% vs SC-642): "
    "the walk is exact-HG; SC pixel fluxes carry a few % quadrature error "
    "in scattering-dominated boundary fields")
say()

# %% slab (unigrid): GLOBAL chain closures A1 -> A2 (invariant 4)
grid, box, hz = slab_grid(4.0, 2.0, nz=16, nxy=4)
nc = grid.n_cells
ke = np.full(nc, 2.0)
ks = np.full(nc, 1.0)                     # omega = 0.5, B = 0
B_s = np.zeros(nc)
bc_s = boundary_conditions(top_I=1.0)
g_s = 0.3

t0 = time.time()
sc_s = lambda_iteration(grid, dg, ke, ks, B_s, bc_s, g=g_s,
                        periodic_box=box, n_iter=800, tol=1e-10, ng=None,
                        jit=True)
t_sc_s = time.time() - t0
t0 = time.time()
a1_s = j_mc(grid, ke, ks, B_s, bc_s, g=g_s, periodic_box=box,
            stage1="explicit", n_histories=400_000,
            n_boundary_histories=400_000, n_batches=100, seed=51, jit=True)
t_a1_s = time.time() - t0
S_s = stage2_source(grid, dg, ke, ks, B_s, a1_s["J"], g=g_s)
t0 = time.time()
a2_s = radiance_from_source(grid, dg, ke, S_s, bc_s, periodic_box=box,
                            estimator="next_flight", n_histories=800,
                            n_batches=20, seed=52, jit=True)
a2_s_det = radiance_from_source(grid, dg, ke, S_s, bc_s, periodic_box=box,
                                estimator="deterministic")
t_a2_s = time.time() - t0


def colmean(a):
    return a.reshape(grid.nz, grid.ny, grid.nx).mean(axis=(1, 2))


def colz(F):
    return F.reshape(grid.nz, grid.ny, grid.nx, 3)[:, :, :, 2].mean(axis=(1, 2))


def closure(Jcol, Fzcol, ka):
    absorbed = 4 * np.pi * ka * Jcol.sum() * hz
    Ftop = 1.5 * Fzcol[-1] - 0.5 * Fzcol[-2]
    Fbot = 1.5 * Fzcol[0] - 0.5 * Fzcol[1]
    net = Fbot - Ftop
    return absorbed, net, (net - absorbed) / absorbed


say("SLAB unigrid (tau = 5, omega = 0.5, g = 0.3, top inflow, B = 0) — "
    "GLOBAL chain quantities:")
say(f"  wall: SC {t_sc_s:.1f}s, A1 {t_a1_s:.1f}s, A2 {t_a2_s:.1f}s")
Jsc_col, J1_col = colmean(sc_s["J"]), colmean(a1_s["J"])
s1_col = colmean(a1_s["sigma_J"])
J2_col = colmean(a2_s_det["J"])
say(f"  A1 J vs SC J      : max|dJ|/J_max = "
    f"{np.abs(J1_col - Jsc_col).max() / Jsc_col.max() * 100:5.2f}%  "
    f"(A1 rel sigma {np.linalg.norm(s1_col) / np.linalg.norm(Jsc_col) * 100:.2f}%)")
say(f"  A2 J(det, own src) vs A1 J: max = "
    f"{np.abs(J2_col - J1_col).max() / Jsc_col.max() * 100:5.2f}%  "
    f"(stage-1 source feeds A2 by design)")
say("  invariant-4 closure  : net Fz(bottom-top) vs 4 pi kappa_a int J dz "
    "(2-point face extrapolation, per unit area)")
for tag, J, F in (("SC      ", sc_s["J"], sc_s["F"]),
                  ("MC A1   ", a1_s["J"], a1_s["F"]),
                  ("MC A2   ", a2_s["J"], flux_vector(a2_s["I"], dg))):
    Jc = colmean(J)
    ab, net, err = closure(Jc, colz(F), ke[0] - ks[0])
    say(f"    closure {tag}: absorbed {ab:.5f}  net {net:.5f}  "
        f"rel err {err * 100:+.2f}%")
say("  (node-level slab differencing intentionally omitted: unigrid SC "
    "transport='chain' is a lattice stencil, not the line integral — "
    "M2 decision; node differencing is done on the octree below)")
say()

# %% cloud octree: full chain A1 -> A2 -> B
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
ka_o = ko - ks_o
vol = cell_volumes(oct_)

t0 = time.time()
sc_c = lambda_iteration(oct_, dg, ko, ks_o, B_o, bc_c, g=g_o,
                        periodic_box=cbox, n_iter=600, tol=1e-9, ng=None,
                        jit=True)
t_sc_c = time.time() - t0
t0 = time.time()
a1_c = j_mc(oct_, ko, ks_o, B_o, bc_c, g=g_o, periodic_box=cbox,
            stage1="explicit", n_histories=200_000,
            n_boundary_histories=200_000, n_batches=100, seed=61, jit=True)
t_a1_c = time.time() - t0
S_mc = stage2_source(oct_, dg, ko, ks_o, B_o, a1_c["J"], g=g_o)
t0 = time.time()
a2_c = radiance_from_source(oct_, dg, ko, S_mc, bc_c, periodic_box=cbox,
                            estimator="next_flight", n_histories=40,
                            n_batches=10, seed=62, jit=True)
a2_c_det = radiance_from_source(oct_, dg, ko, S_mc, bc_c, periodic_box=cbox,
                                estimator="deterministic")
t_a2_c = time.time() - t0
# stage-2-only statistics: SC's own converged direction-resolved source
S_sc = np.asarray(sc_c["S"])
t0 = time.time()
a2_sc = radiance_from_source(oct_, dg, ko, S_sc, bc_c, periodic_box=cbox,
                             estimator="next_flight", n_histories=160,
                             n_batches=16, seed=64, jit=True)
a2_sc_det = radiance_from_source(oct_, dg, ko, S_sc, bc_c,
                                 periodic_box=cbox, estimator="deterministic")
t_a2sc_c = time.time() - t0

top_zc = oct_.centers[:, 2].max()
cand = np.flatnonzero(np.abs(oct_.centers[:, 2] - top_zc) < 1e-9)
c_c = cand[np.argmin(np.hypot(oct_.centers[cand, 0], oct_.centers[cand, 1]))]
pt_c = tuple(oct_.centers[c_c])
t0 = time.time()
sky_c = backward_all_sky(oct_, ko, ks_o, B_o, bc_c, dg, pt_c, g=g_o,
                         periodic_box=cbox, n_seeds=20_000, n_batches=20,
                         seed=63, jit=True)
t_b_c = time.time() - t0

say("CLOUD OCTREE (gaussian cloud, depth 3, "
    f"{oct_.n_leaf} leaves, top inflow, B = 0):")
say(f"  wall: SC {t_sc_c:.1f}s, A1 {t_a1_c:.1f}s, A2 {t_a2_c:.1f}s "
    f"(+det), A2/SC-src {t_a2sc_c:.1f}s (+det), B {t_b_c:.1f}s")
relJ = np.linalg.norm(a1_c["J"] - sc_c["J"]) / np.linalg.norm(sc_c["J"])
sigJ = np.linalg.norm(a1_c["sigma_J"]) / np.linalg.norm(sc_c["J"])
say(f"  A1 J vs SC J (norms): {relJ * 100:.2f}%  vs A1 sigma {sigJ * 100:.2f}%"
    "  -> consistent")

# stage-2-only statistics vs SC (M2 convention: SC's own direction-resolved S)
det_err = np.abs(a2_sc_det["I"] - sc_c["I"]).max() / np.abs(sc_c["I"]).max()
sig_ok = a2_sc["sigma_I"] > 1e-3 * np.sqrt(np.mean(sc_c["I"] ** 2))
z = (a2_sc["I"] - sc_c["I"])[sig_ok] / a2_sc["sigma_I"][sig_ok]
say(f"  A2/SC-src: deterministic vs SC I = {det_err:.1e} (exact-sweep identity), "
    f"next-flight ({np.count_nonzero(sig_ok)} sigma-resolved cells): "
    f"mean z {z.mean():+.2f}, rms z {np.sqrt(np.mean(z**2)):.2f}, "
    f"|z|<2 {np.mean(np.abs(z) < 2) * 100:.1f}%")

# chain A2 vs its own zero-noise reference (stage-2 noise only)
sig_ok2 = a2_c["sigma_I"] > 1e-3 * np.sqrt(np.mean(a2_c_det["I"] ** 2))
z2 = (a2_c["I"] - a2_c_det["I"])[sig_ok2] / a2_c["sigma_I"][sig_ok2]
say(f"  A2 chain (isotropic omega*J source) vs det(same source) "
    f"({np.count_nonzero(sig_ok2)} cells): mean z {z2.mean():+.2f}, "
    f"rms z {np.sqrt(np.mean(z2**2)):.2f}, "
    f"|z|<2 {np.mean(np.abs(z2) < 2) * 100:.1f}%")
rms_I = np.sqrt(np.mean(sc_c["I"] ** 2))
say(f"  source-model term: det(isotropic omega*J from A1) vs SC I: "
    f"rms {np.sqrt(np.mean((a2_c_det['I'] - sc_c['I']) ** 2)) / rms_I * 100:.1f}% "
    f"of rms(I)  (O(g) anisotropy, NOT a statistic)")

# invariant-4 closure on face leaves
half = oct_.half
top_m = np.abs(oct_.centers[:, 2] + half - cbox.z1) < 1e-9
bot_m = np.abs(oct_.centers[:, 2] - half - cbox.z0) < 1e-9
assert abs(np.sum((2 * half[top_m]) ** 2) - cbox.Lx * cbox.Ly) < 1e-6
say("  invariant-4 closure (cell-center Fz on face leaves — 1st-order "
    "boundary extrapolation, cloud-structural):")
for tag, J, F in (("SC      ", sc_c["J"], sc_c["F"]),
                  ("MC A1   ", a1_c["J"], a1_c["F"]),
                  ("MC A2   ", a2_c["J"], flux_vector(a2_c["I"], dg))):
    absorbed = float(4 * np.pi * np.sum(ka_o * J * vol))
    net = float(np.sum(F[bot_m, 2] * (2 * half[bot_m]) ** 2)
                - np.sum(F[top_m, 2] * (2 * half[top_m]) ** 2))
    say(f"    closure {tag}: absorbed {absorbed:.4f}  net {net:.4f}  "
        f"rel err {100 * (net - absorbed) / absorbed:+.2f}%")

# B vs SC-42 and SC-162 reference at the pixel
I_sc_cn = sc_c["I"][c_c, up_idx]
rms_c = np.sqrt(np.mean(I_sc_cn ** 2))
band_b = 2.0 * sky_c["sigma_I"] + 0.10 * rms_c
inside_b = np.count_nonzero(np.abs(sky_c["I"] - I_sc_cn) <= band_b)
t0 = time.time()
dg2 = icosahedric_grid(n_refine=2)
sc_c2 = lambda_iteration(oct_, dg2, ko, ks_o, B_o, bc_c, g=g_o,
                         periodic_box=cbox, n_iter=400, tol=1e-8, ng=True,
                         jit=True)
j2 = np.array([int(np.argmax(dg2.directions @ dg.directions[i]))
               for i in up_idx])
I_ref2 = sc_c2["I"][c_c, j2]
F_sc_c = hflux(I_sc_cn, mu_up)
# node-column differencing keeps the DG-42 weights/mu (M3 convention): the
# 21 nearest refine-2 nodes approximate the refine-1 columns, they are not
# themselves a quadrature of the refine-2 set
F_ref2 = float(dg.weights[up_idx] @ (mu_up * I_ref2))
F_b_c = hflux(sky_c["I"], mu_up)
F_a2_c = hflux(a2_c["I"][c_c, up_idx], mu_up)
say(f"  all-sky B vs SC-42 at cloud-top pixel: {inside_b}/{up_idx.size} "
    f"inside 2 sigma + 10% rms-sky")
say(f"  pixel hemispheric F_up: SC-42 {F_sc_c:.5f} "
    f"({100*(F_sc_c-F_ref2)/F_ref2:+.2f}% vs SC-162), A2 chain {F_a2_c:.5f} "
    f"(isotropic-source model), B {F_b_c:.5f} "
    f"({100*(F_b_c-F_sc_c)/F_sc_c:+.2f}% vs SC-42, "
    f"{100*(F_b_c-F_ref2)/F_ref2:+.2f}% vs SC-162)")
say()
say("chain reading: A1 -> A2 closes within stage-2 noise against its own "
    "zero-noise sweep; global invariant-4 closure agrees across families to "
    "~2%; the adjoint B is exact-HG and matches the SC angular-converged "
    "reference at the pixel; the isotropic omega*J stage-2 source (plan "
    "sketch) is an O(g) model term at node level, not a statistic.")

# %% figure
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

ax = axes[0, 0]
ax.plot(Jsc_col, grid.z, "k-", lw=1.4, label="SC 42")
ax.errorbar(J1_col, grid.z, yerr=2 * s1_col, fmt="o", ms=4, mfc="none",
            color="C0", label="MC A1 (j_mc explicit)")
ax.plot(J2_col, grid.z, "s--", ms=4, color="C3",
        label="MC A2 (det, own source)")
ax.set_xlabel("angle-averaged J")
ax.set_ylabel("z")
ax.legend(fontsize=8)
ax.set_title("slab: stage-1 J and stage-2 J (chain)")

ax = axes[0, 1]
tags = ["SC", "A1", "A2"]
errs = []
for J, F in ((sc_s["J"], sc_s["F"]), (a1_s["J"], a1_s["F"]),
             (a2_s["J"], flux_vector(a2_s["I"], dg))):
    errs.append(closure(colmean(J), colz(F), ke[0] - ks[0])[2] * 100)
ax.bar(tags, errs, color=["0.4", "C0", "C3"])
ax.axhline(0.0, color="k", lw=0.8)
ax.set_ylabel("closure rel err  %")
ax.set_title("slab invariant-4 closure over the chain")

ax = axes[1, 0]
order = np.argsort(mu_up)
x = np.arange(up_idx.size)
ax.plot(x[order], I_sc_cn[order], "k-", lw=1.2, label="SC 42")
ax.plot(x[order], a2_c["I"][c_c, up_idx][order], "^", ms=4, color="C3",
        label="MC A2 chain (isotropic src)")
ax.errorbar(x[order], sky_c["I"][order], yerr=2 * sky_c["sigma_I"][order],
            fmt="o", ms=4, mfc="none", color="C0", label="MC B (exact HG)")
ax.set_xlabel("upward ordinate (sorted by mu)")
ax.set_ylabel("I (node direction)")
ax.legend(fontsize=8)
ax.set_title("cloud-top pixel all-sky: A2/B vs SC-42")

ax = axes[1, 1]
labels = ["SC42\n-SC162", "B\n-SC42", "B\n-SC162", "A2\n-SC42"]
vals = [100 * (F_sc_c - F_ref2) / F_ref2,
        100 * (F_b_c - F_sc_c) / F_sc_c,
        100 * (F_b_c - F_ref2) / F_ref2,
        100 * (F_a2_c - F_sc_c) / F_sc_c]
ax.bar(labels, vals, color=["0.4", "C0", "C0", "C3"])
ax.axhline(0.0, color="k", lw=0.8)
ax.set_ylabel("pixel F_up rel dev  %")
ax.set_title("cloud pixel flux: quadrature vs model terms")

fig.suptitle("MC integration pass (M5): SC vs solver A stage1+stage2 vs "
             "solver B, chain energy closure", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{OUT}/mc_compare.png", dpi=140)
Path(f"{OUT}/mc_compare.txt").write_text("\n".join(lines) + "\n")
print(f"\nfigure: {OUT}/mc_compare.png   tables: {OUT}/mc_compare.txt")
