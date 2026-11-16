# -*- coding: utf-8 -*-
"""
MC phase M1 validation: stage-1 mean intensity J on the 1D slab.

Case identical to `scripts/phase2_slab.py` (plane-parallel slab, tau = 5,
isotropic scattering g = 0, B = 0, isotropic I = 1 entering through the top
face, vacuum bottom, periodic x/y), solved with the Monte Carlo stage-1
solvers of `rt3d.mc.forward.j_mc` instead of the short-characteristics
sweeps.  Case and references are shared via `rt3d.cases.slab_1d`.

Solvers compared:
  * 1D discrete ordinates, 64 Gauss-Legendre mu levels (strong reference)
  * two-stream (Eddington, Marshak BCs) — context only
  * SC solver, 42 icosahedral ordinates (rt3d phase 2)
  * MC, stage1="lambda": plain LI with fresh photon sets (default) and the
    common-random-number stream (crn=True), with and without 1D Ng
  * MC, stage1="explicit": generation-based analog chains

MC transport is angle-exact; the MC-vs-reference deviations are statistical
(batch sigma) plus the reference's own cell discretization.  Error bars are
2 sigma history-batch bands; energy closure (verification invariant 4) is checked
globally: net downward face flux vs total absorption 4 pi kappa_a int J dz.
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

from rt3d.cases.slab_1d import d1o_J, slab_grid, two_stream_J
from rt3d.directions import icosahedric_grid
from rt3d.mc.forward import j_mc
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions

KAPPA = 1.0  # extinction, uniform
TAU = 5.0  # slab optical thickness
NZ = 50
OMEGAS = (0.5, 0.9)
N_MC = 200_000  # histories per MC population
N_BATCH = 200

grid, box, HZ = slab_grid(TAU, KAPPA, NZ)
bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
z_prof = grid.z
tau_from_top = TAU - KAPPA * z_prof


def col(field):
    """vertical profile out of a flat (n_cells,) / (n_cells, 3) field"""
    a = np.asarray(field)
    if a.ndim == 1:
        return a.reshape(grid.nz, grid.nx, grid.ny)[:, 1, 1]
    return a.reshape(grid.nz, grid.nx, grid.ny, 3)[:, 1, 1, :]


# %% run all solvers for omega = 0.5 and 0.9
results = {}
for omega in OMEGAS:
    kappa_e = np.full(grid.n_cells, KAPPA)
    kappa_s = omega * kappa_e
    B = np.zeros(grid.n_cells)

    J_ref = d1o_J(KAPPA, omega, HZ, NZ)
    J_ts = two_stream_J(KAPPA, omega, grid.box[2][0], grid.box[2][1], z_prof)

    dg = icosahedric_grid(n_refine=1)
    t0 = time.time()
    sc = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, g=0.0,
                          periodic_box=box, n_iter=2000, tol=1e-10)
    t_sc = time.time() - t0
    J_sc = col(sc["J"])

    common = dict(stage1="lambda", g=0.0, periodic_box=box, n_histories=N_MC,
                  n_boundary_histories=N_MC, n_batches=N_BATCH, tol=1e-8,
                  n_iter=120)
    t0 = time.time()
    r_li = j_mc(grid, kappa_e, kappa_s, B, bc, seed=101, **common)
    t_li = time.time() - t0
    t0 = time.time()
    r_crn = j_mc(grid, kappa_e, kappa_s, B, bc, seed=102,
                 crn=True, birth="uniform", **common)
    t_crn = time.time() - t0
    t0 = time.time()
    r_ng = j_mc(grid, kappa_e, kappa_s, B, bc, seed=102, ng=True,
                crn=True, birth="uniform", **common)
    t_ng = time.time() - t0
    t0 = time.time()
    r_ex = j_mc(grid, kappa_e, kappa_s, B, bc, seed=103, stage1="explicit",
                g=0.0, periodic_box=box, n_histories=N_MC,
                n_boundary_histories=N_MC, n_batches=N_BATCH)
    t_ex = time.time() - t0

    results[omega] = dict(J_ref=J_ref, J_ts=J_ts, J_sc=J_sc, sc=sc,
                          r_li=r_li, r_crn=r_crn, r_ng=r_ng, r_ex=r_ex,
                          times=(t_sc, t_li, t_crn, t_ng, t_ex))

    # ---- numerical validation ------------------------------------------
    def report(tag, J, sJ, wall, J_ref, cmp_r):
        dev = np.abs(J - J_ref)
        band = 2.0 * sJ.max() + 0.03 * J_ref.max()
        d12 = np.abs(J - col(cmp_r["J"]))
        s12 = 2.0 * np.hypot(sJ, col(cmp_r["sigma_J"])).max()
        band12 = max(s12, 0.03 * J_ref.max())
        print(
            f"  {tag}: max|dJ|/J_top = {dev.max() / J_ref.max() * 100:6.2f}% "
            f"(band {band / J_ref.max() * 100:5.2f}%) "
            f"{'OK' if dev.max() <= band else 'FAIL'} | "
            f"vs {cmp_r['stage1']} MC: {d12.max() / J_ref.max() * 100:5.2f}% "
            f"(band {band12 / J_ref.max() * 100:5.2f}%) "
            f"{'OK' if d12.max() <= band12 else 'FAIL'} | "
            f"{wall:5.1f}s"
        )

    print(f"omega = {omega}: 1D-DO ref J_top = {J_ref.max():.4f}")
    print(f"  SC 42 ordinates: {sc['n_iter']} iters, "
          f"max|dJ|/J_top = {np.abs(J_sc - J_ref).max() / J_ref.max() * 100:.2f}%  "
          f"({t_sc:.1f}s)")
    report("MC lambda (fresh)  ", col(r_li["J"]), col(r_li["sigma_J"]), t_li,
           J_ref, r_ex)
    report(f"MC lambda (crn {r_crn['n_iter']:>2d} it)  ", col(r_crn["J"]),
           col(r_crn["sigma_J"]), t_crn, J_ref, r_ex)
    report(f"MC lambda (Ng {r_ng['n_iter']:>2d} it)  ", col(r_ng["J"]),
           col(r_ng["sigma_J"]), t_ng, J_ref, r_ex)
    report("MC explicit        ", col(r_ex["J"]), col(r_ex["sigma_J"]), t_ex,
           J_ref, r_crn)

    # ---- invariant 4: global closure, per unit area ------------------------
    for tag, r in (("lambda crn", r_crn), ("explicit", r_ex)):
        J = col(r["J"])
        Fz = col(r["F"])[:, 2]
        absorbed = 4 * np.pi * KAPPA * (1 - omega) * J.sum() * HZ
        Ftop = 1.5 * Fz[-1] - 0.5 * Fz[-2]
        Fbot = 1.5 * Fz[0] - 0.5 * Fz[1]
        net = Fbot - Ftop
        print(f"  closure {tag:8s}: 4 pi kappa_a int J dz = {absorbed:.5f}, "
              f"Fz(bottom)-Fz(top) = {net:.5f}, "
              f"rel err = {(net - absorbed) / absorbed * 100:+.2f}%")
    Fz_sc = col(sc["F"])[:, 2]
    absorbed_sc = 4 * np.pi * KAPPA * (1 - omega) * J_sc.sum() * HZ
    net_sc = (1.5 * Fz_sc[0] - 0.5 * Fz_sc[1]) - (1.5 * Fz_sc[-1] - 0.5 * Fz_sc[-2])
    print(f"  closure SC      : rel err = {(net_sc - absorbed_sc) / absorbed_sc * 100:+.2f}%")
    print()

# %% plot: J(tau from top) with 2 sigma MC bands
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, omega in zip(axes, OMEGAS):
    r = results[omega]
    ax.plot(tau_from_top, r["J_ref"], "k-", lw=1.6, label="1D DO, 64 mu (reference)")
    ax.plot(tau_from_top, r["J_ts"], "--", color="C2", lw=1.2, label="two-stream")
    ax.plot(tau_from_top, r["J_sc"], "x", ms=4, color="C3", label="SC, 42 ordinates")
    J, s = col(r["r_crn"]["J"]), col(r["r_crn"]["sigma_J"])
    ax.plot(tau_from_top, J, "o", ms=4, mfc="none", color="C0",
            label=f"MC lambda (crn), {r['r_crn']['n_iter']} it")
    ax.fill_between(tau_from_top, J - 2 * s, J + 2 * s, color="C0", alpha=0.15)
    Jx, sx = col(r["r_ex"]["J"]), col(r["r_ex"]["sigma_J"])
    ax.plot(tau_from_top, Jx, "s", ms=4, mfc="none", color="C1",
            label="MC explicit")
    ax.fill_between(tau_from_top, Jx - 2 * sx, Jx + 2 * sx, color="C1", alpha=0.15)
    ax.set_xlabel("tau from top")
    ax.set_ylabel("J")
    ax.set_title(f"g = 0, omega = {omega}, tau = 5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("MC phase M1: stage-1 J on the 1D slab "
             "(bands = 2 sigma history-batch)")
fig.tight_layout()
fig.savefig("plot/mc_forward_slab.png", dpi=150)
print("  -> plot/mc_forward_slab.png")
