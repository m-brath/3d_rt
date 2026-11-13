# -*- coding: utf-8 -*-
# %%
"""
MC phase M0 validation (plan/mc-plan.md section 7): Woodcock tracker,
null-collision event decision, HG inverse-CDF sampler, DDA walkers.

  1. flight distribution + event-decision fractions (sampler units)
  2. HG samples reproduce the hg_matrix quadrature moments to quadrature
     accuracy (companion of validation anchor 2)
  3. kappa_s = 0, B = 0: MC beam transmittance == exp(-tau) along every
     ray (anchor 3), on unigrid AND octree, in cell AND global mode
  4. cell mode (zero nulls) vs global mode (real nulls) agree within
     combined 2 sigma
  5. periodic wrap + ground entry: MC DDA walkers give the same segment
     counts as march_lattice / march_octree

Run:  & C:/Users/fredd/miniforge3/envs/pyarts/python.exe scripts/mc_phase0_check.py
(needs the env's Library/bin on PATH for BLAS, see AGENTS.md)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.directions import hg, icosahedric_grid
from rt3d.geometry import cuboid, march_lattice, unigrid
from rt3d.mc.sources import hg_cos_theta
from rt3d.mc.tracking import (
    EV_ABSORB,
    event_decision,
    flight_distance,
    path_integral_from_segments,
    sample_collisions,
    walk_lattice,
    walk_octree,
)
from rt3d.octree import march_octree, octree_grid

N_REFINE = 1  # development grid

t_start = time.time()
dg = icosahedric_grid(n_refine=N_REFINE)

# %% 1. flight distribution and event decision
rng = np.random.default_rng(1)
s = flight_distance(rng.random(400_000), 1.7)
print("flight distance: mean %.5f (exp %.5f)" % (s.mean(), 1.0 / 1.7))
kappa_max, kappa_s, kappa_e = 2.0, 0.5, 1.0
ev = event_decision(rng.random(400_000), kappa_max,
                    np.full(400_000, kappa_s), np.full(400_000, kappa_e))
frac = np.bincount(ev, minlength=3) / ev.size
print("event fractions (scatter/absorb/null): %s  expected [0.25 0.25 0.5 ]" % frac)

# %% 2. HG CDF vs hg_matrix moments (anchor 2 companion)
def _legendre(n, x):
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    p0, p1 = np.ones_like(x), x
    for m in range(1, n):
        p0, p1 = p1, ((2 * m + 1) * x * p1 - m * p0) / (m + 1)
    return p1


print("\nHG moments E[P_n(cos t)]: MC vs analytic g^n vs 42-ordinate quadrature")
rng = np.random.default_rng(2)
omega0 = dg.directions[0]
cos_q = np.clip(dg.directions @ omega0, -1.0, 1.0)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ig, g in enumerate((0.0, 0.3, 0.6)):
    c = hg_cos_theta(rng.random(200_000), g)
    row = []
    for n in range(4):
        mc = np.mean(_legendre(n, c))
        quad = np.sum(dg.weights * hg(cos_q, g) * _legendre(n, cos_q))
        row.append(f"n={n}: MC {mc:+.4f}  g^n {g**n:+.4f}  quad {quad:+.4f}")
    print(f"  g = {g}:")
    for r in row:
        print("    " + r)
    ax = axes[ig]
    ax.hist(c, bins=40, density=True, alpha=0.6, label="MC samples")
    cc = np.linspace(-1, 1, 201)
    # HG pdf of the scattering-angle cosine (azimuth-integrated): 2*pi*Phi
    ax.plot(cc, 2.0 * np.pi * hg(cc, g), "k", label="2$\\pi\\Phi_{HG}$")
    ax.set_title(f"g = {g}")
    ax.set_xlabel("cos $\\Theta$")
    if ig == 0:
        ax.set_ylabel("pdf")
        ax.legend()
fig.suptitle(f"MC phase M0: HG inverse-CDF samples, {N_REFINE}-refined grid tallies")
fig.tight_layout()
fig.savefig("plot/mc_phase0_hg_cdf.png", dpi=150)
print("figure -> plot/mc_phase0_hg_cdf.png")

# %% 3. transmittance anchor on both grids and in both majorant modes
def transmittance(grid, box, kappa_e, n_photons, seed, octree=False, ordstride=4):
    """
    Beam photons seeded uniformly on the top face, budget = full column
    path Lz/|Omega_z| with periodic x/y wrap.  Analytic tau per photon
    from the exact DDA walk; MC survival vs sum(exp(-tau)), binomial z.
    """
    rng = np.random.default_rng(seed)
    z0, z1 = grid.box[2]
    out = {}
    for mode in ("cell", "global"):
        obs = exp = var = 0.0
        for di in dg.down_idx[::ordstride]:
            omega = dg.directions[di]
            starts = np.column_stack(
                [rng.uniform(box.x0, box.x1, n_photons),
                 rng.uniform(box.y0, box.y1, n_photons),
                 np.full(n_photons, z1 - 1e-9)]
            )
            budget = np.full(n_photons, (z1 - z0) / abs(omega[2]))
            if octree:
                sr, sc, sl = walk_octree(starts, omega, grid, budget, periodic_box=box)
            else:
                sr, sc, sl, *_ = walk_lattice(starts, omega, grid, periodic_box=box)
            p = np.exp(-path_integral_from_segments(sr, sc, sl, kappa_e, n_photons))
            res = sample_collisions(
                starts, omega, budget, grid, kappa_e, np.zeros_like(kappa_e),
                mode=mode, periodic_box=box, seed=seed + di,
            )
            assert np.all(res["coll_event"][~res["survived"]] == EV_ABSORB)
            obs += np.count_nonzero(res["survived"])
            exp += p.sum()
            var += np.sum(p * (1.0 - p))
            n_null = res["n_null"].sum()
            out.setdefault("_null", 0)
            out["_null"] = out["_null"] + n_null
        z_sigma = (obs - exp) / np.sqrt(var)
        out[mode] = z_sigma
    return out


x = np.linspace(0.5, 5.5, 6)
y = np.linspace(0.5, 3.5, 4)
z = np.linspace(0.5, 7.5, 8)
grid = unigrid(x, y, z)
box = cuboid(*grid.box[0], *grid.box[1], *grid.box[2])
rng = np.random.default_rng(11)
kappa_e = rng.uniform(0.0, 1.5, grid.n_cells)
kappa_e[rng.uniform(size=grid.n_cells) < 0.35] = 0.0  # vacuum -> global nulls fire

case = gaussian_cloud()
oct = octree_grid(case["box"], h0=2.5)
oct.refine(case["kappa_e"], tau_max=0.5, eps_grad=0.3, max_depth=3)
oct.balance_21(max_depth=3)
oct.finalize()
oct.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"])

print("\nTransmittance anchor 3 (kappa_s = 0): survival z-score vs exp(-tau)")
t0 = time.time()
res_u = transmittance(grid, box, kappa_e, n_photons=3000, seed=101)
res_o = transmittance(oct, case["box"], oct.kappa_e, n_photons=2000, seed=202,
                      octree=True)
print(f"  unigrid  6^x4^y8^z random kappa: cell {res_u['cell']:+.2f} sigma, "
      f"global {res_u['global']:+.2f} sigma (nulls {int(res_u['_null'])})")
print(f"  octree   gaussian cloud:         cell {res_o['cell']:+.2f} sigma, "
      f"global {res_o['global']:+.2f} sigma (nulls {int(res_o['_null'])})")
print(f"  ({time.time() - t0:.1f} s)")

# %% 4. cell mode vs global mode (kappa_s > 0, combined 2 sigma)
omega = dg.directions[dg.down_idx[np.argmin(dg.directions[dg.down_idx, 2])]]
K = 8000
rng = np.random.default_rng(17)
starts = np.column_stack([rng.uniform(box.x0, box.x1, K),
                          rng.uniform(box.y0, box.y1, K),
                          np.full(K, box.z1 - 1e-9)])
budget = np.full(K, (box.z1 - box.z0) / abs(omega[2]))
kappa_s = 0.8 * kappa_e
sr, sc, sl, *_ = walk_lattice(starts, omega, grid, periodic_box=box)
p_surv = np.exp(-path_integral_from_segments(sr, sc, sl, kappa_e, K))
rc = sample_collisions(starts, omega, budget, grid, kappa_e, kappa_s,
                       mode="cell", periodic_box=box, seed=21)
rg = sample_collisions(starts, omega, budget, grid, kappa_e, kappa_s,
                       mode="global", periodic_box=box, seed=22)
s_c = np.count_nonzero(rc["survived"])
s_g = np.count_nonzero(rg["survived"])
sig = np.sqrt(np.sum(p_surv * (1.0 - p_surv)))
print(f"\ncell vs global (kappa_s = 0.8 kappa_e): survivors cell {s_c}, global {s_g}, "
      f"analytic {p_surv.sum():.1f}")
print(f"  |cell - global| = {abs(s_c - s_g)} vs 3 sigma_combined = {3 * np.sqrt(2) * sig:.1f}")
print(f"  null collisions consumed (global mode): {int(rg['n_null'].sum())}")

# %% 5. DDA walker segment counts identical to the SC marches
print("\nDDA walker vs SC march (segment counts / arrays must be identical):")
rng = np.random.default_rng(7)
starts = rng.uniform([b[0] for b in grid.box], [b[1] for b in grid.box], size=(300, 3))
n_mismatch = 0
for di in dg.down_idx[::3]:
    omega = dg.directions[di]
    msteps, mt, max_, msd = march_lattice(starts, omega, grid, periodic_box=box)
    mr = np.concatenate([s[0] for s in msteps])
    mc = np.concatenate([s[1] for s in msteps])
    ml = np.concatenate([s[2] for s in msteps])
    wr, wc, wl, wt, wax, wsd = walk_lattice(starts, omega, grid, periodic_box=box)
    ok = (np.array_equal(wr, mr) and np.array_equal(wc, mc)
          and np.allclose(wl, ml, rtol=0, atol=1e-14)
          and np.allclose(wt, mt, rtol=0, atol=1e-12)
          and np.array_equal(wax, max_) and np.array_equal(wsd, msd))
    n_mismatch += not ok
print(f"  march_lattice vs walk_lattice: {len(dg.down_idx[::3])} ordinates, "
      f"{n_mismatch} mismatches")

n_mismatch = 0
z_leaf = oct.centers[:, 2]
z0, z1 = oct.box[2]
for di in list(dg.down_idx[::4]) + list(dg.up_idx[::4]):
    omega = dg.directions[di]
    budget = (z1 - z_leaf) / (-omega[2]) if omega[2] < 0 else (z_leaf - z0) / omega[2]
    mr, ml, mlen = march_octree(oct.centers, -omega, oct, budget, periodic_box=case["box"])
    wr, wl, wlen = walk_octree(oct.centers, -omega, oct, budget, periodic_box=case["box"])
    ok = (np.array_equal(wr, mr) and np.array_equal(wl, ml) and np.array_equal(wlen, mlen))
    n_mismatch += not ok
print(f"  march_octree vs walk_octree:   {len(dg.down_idx[::4]) + len(dg.up_idx[::4])} "
      f"ordinates, {n_mismatch} mismatches")

print(f"\nMC phase M0 check done in {time.time() - t_start:.1f} s")
