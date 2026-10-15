# -*- coding: utf-8 -*-
"""
Transport-scheme comparison — chain vs. SC vs. octree, with/without delta-M
(plan/compare-transport-plan.md).

Six comparison runs on the MANUAL section 11 worked example (Gaussian cloud +
Lambertian ground + solar beam, Ng accelerated, 42 ordinates):

    unigrid 24^3, transport="chain"   delta_M = 0 / 1
    unigrid 24^3, transport="sc"      delta_M = 0 / 1
    octree (h0 = 1.25, max_depth 3), exact marches   delta_M = 0 / 1

plus five accuracy-ranking runs (plan/transport-ranking-plan.md) that answer
"which run is best": a fine-octree REFERENCE solution (tau_max = 0.5,
max_depth 5, h_min = 0.078 m, delta_M = 0) plus grid-refinement runs
(unigrid chain/sc at 48^3, octree depth 4, and the reference octree with
delta_M = 1).  Every run is ranked by its error against the reference, and
the refinement sequence shows those errors shrink toward zero (discretization
error, not physics).

The two runs on the fine reference octree additionally use icosahedral
refinement 3 = 642 ordinates instead of the 42-ordinate development grid, so
the reference is finer in ANGLE as well as in space; the ranking of the
42-ordinate runs therefore also measures their angular (ordinate-resolution)
error, while the comparisons among the 42-ordinate runs still cancel it.

Diagnostics per run: mean intensity J, angular variance of the intensity
(1/4pi int (I - J)^2 dOmega), and the upward radiance at TOA (F^+(x,y) maps,
per-ordinate domain-mean radiance).  All fields are sampled onto one common
64^3 lattice (finest octree cell) so every pair of runs is directly
comparable.  Interactive 3D scenes are exported as self-contained vtk.js
HTML (pyvista trame export) with an injected control panel: sliders for the
x/y/z slice-plane positions and for the isosurface level (precomputed
snapshots toggled client-side), scalar and visibility toggles, orbit/zoom in
the browser, no server needed to view.

Run from the repo root:
    macOS/Linux:  ~/miniforge3/envs/pyarts/bin/python scripts/compare_transport.py
    Windows:      $env:Path = "$env:USERPROFILE\\miniforge3\\envs\\pyarts\\Library\\bin;" + $env:Path
                  & "$env:USERPROFILE\\miniforge3\\envs\\pyarts\\python.exe" scripts/compare_transport.py

Units as in the manual: lengths m, kappa in m^-1, radiance normalised to the
incident beam flux F0 = 1.
"""

# %% imports and configuration
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.directions import icosahedric_grid
from rt3d.diagnostics import angular_variance
from rt3d.geometry import unigrid, ground_surface
from rt3d.octree import octree_grid
from rt3d.transport import boundary_conditions
from rt3d.solver import lambda_iteration

OUT = str(Path(__file__).resolve().parent.parent / "plot" / "compare_transport")
os.makedirs(OUT, exist_ok=True)

LBL_X, LBL_Y, LBL_Z = "x  [m]", "y  [m]", "z  [m]"

# MANUAL section 11 setup, unchanged
case = gaussian_cloud(omega_cap=0.9)
box = case["box"]                              # x, y in [-5, 5], z in [0, 10] m

# direction grids keyed by icosahedral refinement: the comparison runs use
# the 42-ordinate development grid; the two runs on the fine reference
# octree (octree_ref_dM0/dM1) use refinement 3 = 642 ordinates
DGS = {1: icosahedric_grid(n_refine=1), 3: icosahedric_grid(n_refine=3)}
for _nr, _dg in DGS.items():
    print(f"n_refine={_nr}: N_dir = {_dg.n_dir},  "
          f"sum(w) - 4pi = {_dg.weights.sum() - 4*np.pi:.3e}")

mu_sun = np.cos(np.radians(30.0))
beam = {"omega_sun": np.array([np.sqrt(1.0 - mu_sun**2), 0.0, -mu_sun]),
        "F0": 1.0}

bc = boundary_conditions(top_I=1.0,
                         ground=ground_surface(albedo=0.3, emissivity=0.0, B_g=0.0))

N_ITER, TOL = 60, 1e-6

# %% grids and input fields (identical to the manual example)
def make_unigrid(n):
    xs = np.linspace(box.x0, box.x1, n + 1)[:-1] + (box.Lx / n) / 2
    zs = np.linspace(box.z0, box.z1, n + 1)[:-1] + (box.Lz / n) / 2
    return unigrid(xs, xs, zs)


def make_octree(tau_max, max_depth):
    o = octree_grid(box, h0=1.25)
    o.refine(case["kappa_e"], tau_max=tau_max, eps_grad=1.0, max_depth=max_depth)
    o.balance_21(max_depth=max_depth)
    o.finalize()
    o.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"],
                    g=case["g"], T=case["T"])
    print(f"octree tau_max={tau_max:g}, max_depth={max_depth}: "
          f"n_leaf={o.n_leaf}, max_level={o.max_level}, "
          f"h_min={2.0 * o.half.min():g} m")
    return o


GRIDS = {
    "u24":  make_unigrid(24),
    "u48":  make_unigrid(48),
    "oct3": make_octree(2.0, 3),   # manual/octree baseline grid
    "oct4": make_octree(1.0, 4),   # octree refinement step
    "oct5": make_octree(0.5, 5),   # reference grid (refinement saturates here)
}
IS_OCT = {k: k.startswith("oct") for k in GRIDS}

FIELDS = {}
for gname, g in GRIDS.items():
    c = g.centers
    if IS_OCT[gname]:
        FIELDS[gname] = (g.kappa_e, g.kappa_s, g.g, np.zeros(g.n_leaf))
    else:
        FIELDS[gname] = (case["kappa_e"](*c.T), case["kappa_s"](*c.T),
                         case["g"](*c.T), np.zeros(g.n_cells))

# %% the runs
# (key, grid, transport, delta_M, direction-grid icosahedral refinement)
RUNS = [
    ("uni_chain_dM0",   "u24",  "chain", 0, 1),
    ("uni_chain_dM1",   "u24",  "chain", 1, 1),
    ("uni_sc_dM0",      "u24",  "sc",    0, 1),
    ("uni_sc_dM1",      "u24",  "sc",    1, 1),
    ("octree_dM0",      "oct3", "chain", 0, 1),
    ("octree_dM1",      "oct3", "chain", 1, 1),
    ("uni_chain48_dM1", "u48",  "chain", 1, 1),   # ranking / convergence runs
    ("uni_sc48_dM1",    "u48",  "sc",    1, 1),
    ("octree_d4_dM1",   "oct4", "chain", 1, 1),
    ("octree_ref_dM1",  "oct5", "chain", 1, 3),   # 642 ordinates
    ("octree_ref_dM0",  "oct5", "chain", 0, 3),   # THE reference, 642 ordinates
]
REF = "octree_ref_dM0"
BASE6 = [k for k, *_ in RUNS[:6]]
RUN_DG = {key: DGS[nref] for key, _, _, _, nref in RUNS}

results, wall = {}, {}
CACHE_DIR = os.path.join(OUT, "_solve_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
for key, gname, transport, dM, nref in RUNS:
    dg = RUN_DG[key]
    cf = os.path.join(CACHE_DIR, f"{key}.npz")
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True)
        cached_ndir = int(z["n_dir"]) if "n_dir" in z.files else 42
        if cached_ndir != dg.n_dir:
            print(f"{key:15s}  cache is stale ({cached_ndir} ordinates != "
                  f"{dg.n_dir}) — re-solving", flush=True)
        else:
            results[key] = {k: z[k] for k in ("I", "J", "F", "S")}
            results[key]["n_iter"] = int(z["n_iter"])
            results[key]["converged"] = bool(z["converged"])
            wall[key] = float(z["wall"])
            print(f"{key:15s}  loaded from cache  wall={wall[key]:7.2f} s", flush=True)
            continue
    k_e, k_s, g_c, B_f = FIELDS[gname]
    t0 = time.perf_counter()
    results[key] = lambda_iteration(
        GRIDS[gname], dg, k_e, k_s, B_f, bc,
        g=g_c, periodic_box=box, beam=beam,
        n_iter=N_ITER, tol=TOL, ng=True, delta_M=dM, transport=transport,
    )
    wall[key] = time.perf_counter() - t0
    r = results[key]
    np.savez_compressed(cf, I=r["I"], J=r["J"], F=r["F"], S=r["S"],
                        n_iter=r["n_iter"], converged=r["converged"],
                        wall=wall[key], n_dir=dg.n_dir)
    print(f"{key:15s}  n_iter={r['n_iter']:3d}  converged={r['converged']}  "
          f"wall={wall[key]:7.2f} s", flush=True)

# %% common lattice: every run sampled onto the baseline-octree-cell grid
# (h = 0.15625 m; exact pointwise sampling for piecewise-constant fields —
# the finer reference cells are sampled, not averaged)
h_min = 2.0 * GRIDS["oct3"].half.min()
nl = (np.array([box.Lx, box.Ly, box.Lz]) / h_min).astype(int)
gx = box.x0 + (np.arange(nl[0]) + 0.5) * h_min
gy = box.y0 + (np.arange(nl[1]) + 0.5) * h_min
gz = box.z0 + (np.arange(nl[2]) + 0.5) * h_min
P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)

# lattice-point -> cell lookup per grid (octree: leaf index; unigrid: (ku, ju, iu))
LAT_IDX = {}
for gname, g in GRIDS.items():
    if IS_OCT[gname]:
        LAT_IDX[gname] = np.asarray(g.locate(P)).reshape(tuple(int(v) for v in nl))
    else:
        i = np.clip(((gx - box.x0) / g.hx).astype(int), 0, g.nx - 1)
        j = np.clip(((gy - box.y0) / g.hy).astype(int), 0, g.ny - 1)
        k = np.clip(((gz - box.z0) / g.hz).astype(int), 0, g.nz - 1)
        # cell arrays are stored (z, y, x): the index tuple must be (KU, JU, IU)
        # with IU varying along lattice axis 0 (x), KU along axis 2 (z)
        LAT_IDX[gname] = np.meshgrid(i, j, k, indexing="ij")[::-1]

RUN_GRID = {key: gname for key, gname, _, _, _ in RUNS}


# angular variance per run (from the swept intensities, each on its own
# direction grid)
for key, *_ in RUNS:
    results[key]["var"] = angular_variance(results[key]["I"], RUN_DG[key])


def lat_field(key, name):
    """Cell field of a run sampled onto the common lattice, shape (nxl, nyl, nzl)."""
    gname = RUN_GRID[key]
    if name == "kappa_e":
        f = FIELDS[gname][0]
    else:
        f = results[key][name]
    if IS_OCT[gname]:
        return np.asarray(f)[LAT_IDX[gname]]
    f4 = np.asarray(f).reshape(GRIDS[gname].nz, GRIDS[gname].ny, GRIDS[gname].nx)
    return f4[LAT_IDX[gname]]


lat = {key: {name: lat_field(key, name) for name in ("J", "var", "kappa_e")}
       for key, *_ in RUNS}

# lattice slice helpers
iy0 = int(np.argmin(np.abs(gy)))              # lattice row nearest y = 0
kz4 = int((4.0 - box.z0) / h_min)             # lattice layer just below z = 4 (cloud core)


def lat_yz(key, name):
    """(nz, nx) slice at y ~ 0 of a lattice field."""
    return lat[key][name][:, iy0, :].T


def lat_xy(key, name, k=kz4):
    """(ny, nx) slice at z ~ gz[k] of a lattice field."""
    return lat[key][name][:, :, k].T


# %% summary table + CSV
def toa_stats(key):
    """Domain-mean upward radiance per upward ordinate + mean F_z at the top layer."""
    r = results[key]
    gname = RUN_GRID[key]
    dg = RUN_DG[key]
    g = GRIDS[gname]
    if IS_OCT[gname]:
        top = g.centers[:, 2] + g.half >= box.z1 - 1e-9
        area = (2.0 * g.half[top]) ** 2
    else:
        top = np.zeros(g.n_cells, dtype=bool)
        top[(g.nz - 1) * g.ny * g.nx:] = True
        area = np.ones(top.sum())
    I_top, F_top = r["I"][top], r["F"][top]
    mean_I = (I_top[:, dg.up_idx] * area[:, None]).sum(axis=0) / area.sum()
    return mean_I, F_top[:, 2].mean()


toa = {key: toa_stats(key) for key, *_ in RUNS}

print("\nrun               transport  delta_M  N_dir  n_iter  conv  wall [s]  "
      "<J>      mean F_z @TOA")
rows = []
for key, gname, transport, dM, nref in RUNS:
    r = results[key]
    meanJ = r["J"].mean()
    rows.append((key, gname, transport, dM, RUN_DG[key].n_dir, r["n_iter"],
                 r["converged"], wall[key], meanJ, toa[key][1]))
    print(f"{key:15s}  {transport:9s}  {dM:7d}  {RUN_DG[key].n_dir:5d}  {r['n_iter']:6d}  "
          f"{str(r['converged']):5s}  {wall[key]:8.2f}  {meanJ:.4f}  {toa[key][1]:+.4f}")

with open(os.path.join(OUT, "summary.csv"), "w") as fh:
    fh.write("run,grid,transport,delta_M,n_dir,n_iter,converged,wall_s,mean_J,mean_Fz_TOA\n")
    for k, gn, t, d, nd, it, cv, w, j, fz in rows:
        fh.write(f"{k},{gn},{t},{d},{nd},{it},{int(cv)},{w:.3f},{j:.6f},{fz:.6f}\n")

# delta-M equivalence check: converged J must not depend on delta_M (same grid)
for base in ("uni_chain", "uni_sc", "octree", "octree_ref"):
    dJ = np.abs(lat[f"{base}_dM1"]["J"] - lat[f"{base}_dM0"]["J"]).max()
    ref = np.abs(lat[f"{base}_dM0"]["J"]).max()
    print(f"delta-M equivalence, {base}: max |dJ| = {dJ:.2e} ({dJ / ref * 100:.2f}% of max|J|)")

# %% comparison pairs (B - A), used for all difference figures
PAIRS = [
    ("uni_chain_dM0", "uni_sc_dM0", "SC - chain, dM off"),
    ("uni_chain_dM1", "uni_sc_dM1", "SC - chain, dM on"),
    ("uni_chain_dM0", "uni_chain_dM1", "dM1 - dM0, unigrid chain"),
    ("uni_sc_dM0", "uni_sc_dM1", "dM1 - dM0, unigrid SC"),
    ("octree_dM0", "octree_dM1", "dM1 - dM0, octree"),
    ("uni_chain_dM0", "octree_dM0", "octree - unigrid chain, dM off"),
]

RUN_TITLES = {
    "uni_chain_dM0": "unigrid chain,  dM off",
    "uni_chain_dM1": "unigrid chain,  dM on",
    "uni_sc_dM0":    "unigrid SC,     dM off",
    "uni_sc_dM1":    "unigrid SC,     dM on",
    "octree_dM0":    "octree chain,   dM off",
    "octree_dM1":    "octree chain,   dM on",
    "uni_chain48_dM1": "unigrid chain 48^3, dM on",
    "uni_sc48_dM1":    "unigrid SC 48^3,    dM on",
    "octree_d4_dM1":   "octree depth 4,     dM on",
    "octree_ref_dM1":  "octree ref grid,  dM on, 642 ord.",
    "octree_ref_dM0":  "REFERENCE (octree ref grid, dM off, 642 ord.)",
}

xe = np.concatenate([gx - h_min / 2, [gx[-1] + h_min / 2]])
ze = np.concatenate([gz - h_min / 2, [gz[-1] + h_min / 2]])
XU, ZU = np.meshgrid(xe, ze)


def diff_stats(a, b, name):
    """3D relative difference of a lattice field: max over the domain."""
    A, B = lat[a][name], lat[b][name]
    d = np.abs(B - A)
    return d, d.max() / max(np.abs(A).max(), 1e-30)


# %% J: slices, band-mean profiles, difference maps
fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), layout="constrained")
vmax = max(lat_yz(k, "J").max() for k in BASE6)
for ax, key in zip(axes.ravel(), BASE6):
    im = ax.pcolormesh(XU, ZU, lat_yz(key, "J"), cmap="viridis",
                       shading="flat", vmin=0, vmax=vmax)
    ax.set_title(RUN_TITLES[key], fontsize=10)
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Z)
fig.suptitle(rf"mean intensity $J\,/F_0$, $y = {gy[iy0]:.3f}$ m slice on the "
             rf"common {nl[0]}$^3$ lattice ($h$ = {h_min:g} m)")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
fig.savefig(os.path.join(OUT, "j_slices.png"), dpi=130)
plt.close(fig)

# band-mean J profiles on the 8 coarse base-cell bands (manual recipe)
nband = int(round(box.Lz / GRIDS["oct3"].h0))
zband = np.linspace(box.z0, box.z1, nband + 1)
zc = 0.5 * (zband[:-1] + zband[1:])
profiles = {}
for key, gname, *_ in RUNS:
    J = results[key]["J"]
    zi = np.clip(((GRIDS[gname].centers[:, 2] - box.z0) / GRIDS["oct3"].h0).astype(int),
                 0, nband - 1)
    profiles[key] = np.array([J[zi == b].mean() for b in range(nband)])

fig, ax = plt.subplots(figsize=(5.6, 4.6))
for key in BASE6:
    style = "--" if key.startswith("octree") else "-"
    style += "." if key.endswith("dM1") else ""
    ax.plot(profiles[key], zc, style, label=RUN_TITLES[key])
ax.plot(profiles[REF], zc, "-k", lw=2.2,
        label=f"reference (fine octree, {RUN_DG[REF].n_dir} ord.)")
ax.set_xlabel(r"band-mean $J\,/F_0$")
ax.set_ylabel(LBL_Z)
ax.set_title("band-mean $J$ profiles (8 base-cell bands)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "j_profiles.png"), dpi=130)
plt.close(fig)
np.savetxt(os.path.join(OUT, "profiles.csv"),
           np.column_stack([zc] + [profiles[k] for k, *_ in RUNS]),
           delimiter=",", fmt="%.6f",
           header="z_band," + ",".join(k for k, *_ in RUNS), comments="")

fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), layout="constrained")
dmax_common = 0.0
diff_maps = []
for a, b, ttl in PAIRS:
    d, rel = diff_stats(a, b, "J")
    diff_maps.append((d, rel))
    dmax_common = max(dmax_common, np.abs(lat_xy(a, "J")).max())
norm_d = Normalize(vmin=-0.2 * dmax_common, vmax=0.2 * dmax_common)
for ax, ((a, b, ttl), (d, rel)) in zip(axes.ravel(), zip(PAIRS, diff_maps)):
    im = ax.pcolormesh(xe, ze, d[:, :, kz4].T, cmap="RdBu_r", shading="flat",
                       norm=norm_d)
    ax.set_title(f"{ttl}\nmax rel. diff {rel * 100:.1f}%", fontsize=9)
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Z)
fig.suptitle(rf"$J$ difference maps at $z = {gz[kz4]:.2f}$ m "
             rf"(relative to max $|J|$ of the reference run)")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01,
             label=r"$\Delta J\,/F_0$")
fig.savefig(os.path.join(OUT, "j_diff.png"), dpi=130)
plt.close(fig)

# %% angular variance: slices + difference maps
fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), layout="constrained")
vmax = max(lat_yz(k, "var").max() for k in BASE6)
for ax, key in zip(axes.ravel(), BASE6):
    im = ax.pcolormesh(XU, ZU, lat_yz(key, "var"), cmap="inferno",
                       shading="flat", vmin=0, vmax=vmax)
    ax.set_title(RUN_TITLES[key], fontsize=10)
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Z)
fig.suptitle(r"angular variance $\sigma^2 = \frac{1}{4\pi}\int (I-J)^2\,d\Omega$ "
             rf"[/$F_0^2$ sr$^2$], $y = {gy[iy0]:.3f}$ m slice")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
fig.savefig(os.path.join(OUT, "var_slices.png"), dpi=130)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), layout="constrained")
vmax_d = max(diff_stats(a, b, "var")[0][:, :, kz4].max() for a, b, _ in PAIRS)
norm_dv = Normalize(vmin=-0.2 * vmax_d, vmax=0.2 * vmax_d)
for ax, (a, b, ttl) in zip(axes.ravel(), PAIRS):
    d, rel = diff_stats(a, b, "var")
    im = ax.pcolormesh(xe, ze, d[:, :, kz4].T, cmap="RdBu_r", shading="flat",
                       norm=norm_dv)
    ax.set_title(f"{ttl}\nmax rel. diff {rel * 100:.1f}%", fontsize=9)
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Z)
fig.suptitle(r"$\sigma^2$ difference maps at "
             rf"$z = {gz[kz4]:.2f}$ m (relative to max $\sigma^2$ of the reference)")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
fig.savefig(os.path.join(OUT, "var_diff.png"), dpi=130)
plt.close(fig)

# %% upward radiance at TOA: F^+ maps, difference maps, mu spectra
def fup_map(key):
    """F^+(x, y) at the top of the domain on the common lattice (nxl, nyl)."""
    r = results[key]
    gname = RUN_GRID[key]
    dg = RUN_DG[key]
    g = GRIDS[gname]
    w_up = dg.weights[dg.up_idx] * dg.directions[dg.up_idx, 2]
    if IS_OCT[gname]:
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        Pt = np.column_stack([GX.ravel(), GY.ravel(),
                              np.full(GX.size, box.z1 - h_min / 2)])
        leaf = g.locate(Pt)
        F_up = r["I"][leaf][:, dg.up_idx] @ w_up
        return F_up.reshape(nl[0], nl[1])
    F_up_top = r["I"][(g.nz - 1) * g.ny * g.nx:][:, dg.up_idx] @ w_up
    F_up_top = F_up_top.reshape(g.ny, g.nx)
    i = np.clip(((gx - box.x0) / g.hx).astype(int), 0, g.nx - 1)
    j = np.clip(((gy - box.y0) / g.hy).astype(int), 0, g.ny - 1)
    return F_up_top[np.ix_(j, i)]


fup = {key: fup_map(key) for key, *_ in RUNS}
ye = np.concatenate([gy - h_min / 2, [gy[-1] + h_min / 2]])
X2, Y2 = np.meshgrid(xe, ye)

fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), layout="constrained")
vmax = max(fup[k].max() for k in BASE6)
for ax, key in zip(axes.ravel(), BASE6):
    im = ax.pcolormesh(X2, Y2, fup[key].T, cmap="magma", shading="flat",
                       vmin=0, vmax=vmax)
    ax.set_title(RUN_TITLES[key], fontsize=10)
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Y)
fig.suptitle(r"upward flux $F^{+} = \int_{\mu > 0} I\,\mu\,d\Omega$ at TOA "
             rf"[$F_0$], top-layer values on the common lattice")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
fig.savefig(os.path.join(OUT, "toa_fup_maps.png"), dpi=130)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), layout="constrained")
ref_max = max(fup[a].max() for a, _, _ in PAIRS)
norm_d = Normalize(vmin=-0.2 * ref_max, vmax=0.2 * ref_max)
for ax, (a, b, ttl) in zip(axes.ravel(), PAIRS):
    d = fup[b] - fup[a]
    rel = np.abs(d).max() / max(fup[a].max(), 1e-30)
    im = ax.pcolormesh(X2, Y2, d.T, cmap="RdBu_r", shading="flat", norm=norm_d)
    ax.set_title(f"{ttl}\nmax rel. diff {rel * 100:.1f}%", fontsize=9)
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Y)
fig.suptitle(r"$F^{+}$ difference maps at TOA (relative to max $F^{+}$ of the reference)")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
fig.savefig(os.path.join(OUT, "toa_fup_diff.png"), dpi=130)
plt.close(fig)

def mu_spec(key):
    """(mu sorted, domain-mean upward radiance) for a run on its own grid."""
    dg = RUN_DG[key]
    mu = dg.directions[dg.up_idx, 2]
    order = np.argsort(mu)
    return mu[order], toa[key][0][order]


fig, ax = plt.subplots(figsize=(6.2, 4.6))
for key in BASE6:
    style = "--" if key.startswith("octree") else "-"
    style += "." if key.endswith("dM1") else ""
    mu_k, I_k = mu_spec(key)
    ax.plot(mu_k, I_k, style, label=RUN_TITLES[key])
mu_r, I_r = mu_spec(REF)
ax.plot(mu_r, I_r, "-k", lw=1.6, alpha=0.85,
        label=f"reference (fine octree, {RUN_DG[REF].n_dir} ord.)")
ax.set_xlabel(r"$\mu = \cos\vartheta$")
ax.set_ylabel(r"domain-mean upward $I\,/F_0$ [sr$^{-1}$]")
ax.set_title("TOA upward radiance vs ordinate direction")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "toa_mu_spectra.png"), dpi=130)
plt.close(fig)

# per-ordinate spectra are only mutually comparable within one direction
# grid -> one CSV per direction grid used in the study
for nref, keys in ((1, [k for k, *_, nr in RUNS if nr == 1]),
                   (3, [k for k, *_, nr in RUNS if nr == 3])):
    if not keys:
        continue
    mu_g = DGS[nref].directions[DGS[nref].up_idx, 2]
    fname = ("toa_mean_radiance.csv" if DGS[nref].n_dir == 42
             else f"toa_mean_radiance_{DGS[nref].n_dir}.csv")
    np.savetxt(os.path.join(OUT, fname),
               np.column_stack([mu_g] + [toa[k][0] for k in keys]),
               delimiter=",", fmt="%.6f",
               header="mu," + ",".join(keys), comments="")

# %% quantitative pair statistics (B - A), quoted in docs/compare-transport.md
print("\npair (B - A)                     field   max rel diff   mean rel diff")
pair_rows = []
for a, b, ttl in PAIRS:
    for name in ("J", "var"):
        d, rel = diff_stats(a, b, name)
        mean_rel = (d / max(np.abs(lat[a][name]).max(), 1e-30)).mean()
        pair_rows.append((ttl, name, rel, mean_rel))
        print(f"{ttl:31s} {name:5s}  {rel * 100:9.1f}%   {mean_rel * 100:11.2f}%")
    d = np.abs(fup[b] - fup[a])
    rel = d.max() / max(fup[a].max(), 1e-30)
    mean_rel = (d / max(fup[a].max(), 1e-30)).mean()
    pair_rows.append((ttl, "F_plus_TOA", rel, mean_rel))
    print(f"{ttl:31s} F+    {rel * 100:9.1f}%   {mean_rel * 100:11.2f}%")

with open(os.path.join(OUT, "pair_stats.csv"), "w") as fh:
    fh.write("pair,field,max_rel_diff,mean_rel_diff\n")
    for ttl, name, rel, mean_rel in pair_rows:
        fh.write(f'"{ttl}",{name},{rel:.6f},{mean_rel:.6f}\n')

# %% accuracy ranking vs the reference run (plan/transport-ranking-plan.md)
# note: the reference uses 642 ordinates, so the errors of the 42-ordinate
# runs below include their angular (ordinate-resolution) error as well
print(f"\naccuracy vs reference '{REF}' ({RUN_DG[REF].n_dir} ordinates)")
print("run               grid  transport  dM  N_dir  n_iter  wall [s]   "
      "J err max/mean      var err max/mean      F+ err max/mean      d<J>     dFz_TOA")
rank_rows = []
for key, gname, transport, dM, nref in RUNS:
    if key == REF:
        continue
    errs = {}
    for name in ("J", "var"):
        A, B = lat[REF][name], lat[key][name]
        d = np.abs(B - A)
        ref_max = max(np.abs(A).max(), 1e-30)
        errs[name] = (d.max() / ref_max, (d / ref_max).mean())
    A, B = fup[REF], fup[key]
    d = np.abs(B - A)
    ref_max = max(A.max(), 1e-30)
    errs["F_plus_TOA"] = (d.max() / ref_max, (d / ref_max).mean())
    dJm = results[key]["J"].mean() - results[REF]["J"].mean()
    dFz = toa[key][1] - toa[REF][1]
    rank_rows.append((key, gname, transport, dM, RUN_DG[key].n_dir,
                      results[key]["n_iter"], wall[key], errs, dJm, dFz))
    print(f"{key:15s} {gname:5s}  {transport:9s} {dM:2d}  {RUN_DG[key].n_dir:5d}  "
          f"{results[key]['n_iter']:6d}  "
          f"{wall[key]:8.2f}  {errs['J'][0] * 100:6.1f}% /{errs['J'][1] * 100:5.2f}%   "
          f"{errs['var'][0] * 100:6.1f}% /{errs['var'][1] * 100:5.2f}%   "
          f"{errs['F_plus_TOA'][0] * 100:6.1f}% /{errs['F_plus_TOA'][1] * 100:5.2f}%   "
          f"{dJm:+.4f}  {dFz:+.4f}")

with open(os.path.join(OUT, "ranking.csv"), "w") as fh:
    fh.write("run,grid,transport,delta_M,n_dir,n_iter,wall_s,"
             "J_err_max,J_err_mean,var_err_max,var_err_mean,"
             "Fplus_err_max,Fplus_err_mean,d_mean_J,d_mean_Fz_TOA\n")
    for key, gn, t, dM, nd, it, w, e, dJm, dFz in rank_rows:
        fh.write(f"{key},{gn},{t},{dM},{nd},{it},{w:.3f},"
                 f"{e['J'][0]:.6f},{e['J'][1]:.6f},{e['var'][0]:.6f},{e['var'][1]:.6f},"
                 f"{e['F_plus_TOA'][0]:.6f},{e['F_plus_TOA'][1]:.6f},"
                 f"{dJm:.6f},{dFz:.6f}\n")

# --- error maps |J - J_ref| at the cloud plane, one panel per non-reference run
fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.4), layout="constrained")
err_max = max(np.abs(lat[k]["J"] - lat[REF]["J"]).max() for k, *_ in RUNS if k != REF)
norm_e = Normalize(vmin=0, vmax=err_max)
for ax, key in zip(axes.ravel(), [k for k, *_ in RUNS if k != REF]):
    d = np.abs(lat[key]["J"] - lat[REF]["J"])
    rel = d.max() / max(np.abs(lat[REF]["J"]).max(), 1e-30)
    im = ax.pcolormesh(xe, ze, d[:, :, kz4].T, cmap="magma", shading="flat",
                       norm=norm_e)
    ax.set_title(f"{RUN_TITLES[key]}\nmax rel. err {rel * 100:.1f}%", fontsize=8)
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Z)
fig.suptitle(rf"$|J - J_{{\rm ref}}|$ at $z = {gz[kz4]:.2f}$ m, "
             rf"reference = {REF} (color scale identical in all panels)")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.008,
             label=r"$|{\Delta}J|\,/F_0$")
fig.savefig(os.path.join(OUT, "j_err_maps.png"), dpi=130)
plt.close(fig)

# --- grid-convergence of the band-mean J profiles toward the reference
CONV_GROUPS = [
    ("chain dM1", ["uni_chain_dM1", "uni_chain48_dM1"], "C0"),
    ("SC dM1",    ["uni_sc_dM1", "uni_sc48_dM1"],       "0.55"),
    ("octree dM1", ["octree_dM1", "octree_d4_dM1", "octree_ref_dM1"], "0.35"),
]
fig, ax = plt.subplots(figsize=(6.4, 5.0))
for ttl, keys, col in CONV_GROUPS:
    for kk, key in enumerate(keys):
        ax.plot(profiles[key], zc, "--", color=col, lw=1.0 + 0.8 * kk, label=key)
ax.plot(profiles[REF], zc, "-k", lw=2.4, label=f"reference ({REF})")
ax.set_xlabel(r"band-mean $J\,/F_0$")
ax.set_ylabel(LBL_Z)
ax.set_title("grid convergence of band-mean $J$ toward the reference")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "j_convergence.png"), dpi=130)
plt.close(fig)

# --- error bar chart (J, sigma^2, F+ max error vs reference)
fig, ax = plt.subplots(figsize=(7.2, 6.0))
keys_ord = [k for k, *_ in RUNS if k != REF]
ypos = np.arange(len(keys_ord))[::-1]
for off, name, col in ((-0.26, "J", "tab:blue"),
                       (0.0, "var", "tab:orange"),
                       (0.26, "F_plus_TOA", "tab:green")):
    vals = [max(r[7][name][0], 1e-5) for r in rank_rows]
    ax.barh(ypos + off, vals, height=0.24, color=col,
            label=f"max rel. err {name}")
ax.set_yticks(ypos)
ax.set_yticklabels(keys_ord, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("max relative error vs reference")
ax.set_title("accuracy of every run against the reference solution")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "err_bars.png"), dpi=130)
plt.close(fig)

# --- TOA mu-spectra convergence toward the reference
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for ttl, keys, col in CONV_GROUPS:
    for kk, key in enumerate(keys):
        mu_k, I_k = mu_spec(key)
        ax.plot(mu_k, I_k, "--", color=col, lw=1.0 + 0.8 * kk, label=key)
mu_r, I_r = mu_spec(REF)
ax.plot(mu_r, I_r, "-k", lw=1.6, alpha=0.85,
        label=f"reference ({REF}, {RUN_DG[REF].n_dir} ord.)")
ax.set_xlabel(r"$\mu = \cos\vartheta$")
ax.set_ylabel(r"domain-mean upward $I\,/F_0$ [sr$^{-1}$]")
ax.set_title("TOA upward radiance: grid convergence toward the reference")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "toa_convergence.png"), dpi=130)
plt.close(fig)

# %% interactive 3D scenes (self-contained vtk.js HTML, offline viewer)
try:
    import pyvista as pv
    from trame_pyvista.jupyter import launch_server
except ImportError:
    pv = None
    print("\nNOTE: pyvista / trame-pyvista not installed in this environment — "
          "skipping the interactive HTML scenes and vol_j.png")

if pv is not None:
    pv.OFF_SCREEN = True

    async def _launch_trame():
        srv = launch_server()
        await srv.ready

    asyncio.run(_launch_trame())


def lattice_image(key):
    """pv.ImageData of the common lattice with point data J, var, kappa_e."""
    img = pv.ImageData(dimensions=tuple(int(v) for v in nl),
                       origin=(gx[0], gy[0], gz[0]), spacing=(h_min,) * 3)
    for name in ("J", "var", "kappa_e"):
        img.point_data[name] = lat[key][name].transpose(2, 1, 0).ravel()
    return img


# --- interactive scene configuration: precomputed slider snapshots.  The
# exported vtk.js scene is static, so slice planes and isosurfaces are baked
# at N_SLICE positions per axis / N_ISO levels; the injected HTML control
# panel (PANEL_JS below) toggles the corresponding actors at runtime.
N_SLICE = 25
N_ISO = 15
SLICE_STRIDE = 2          # slice planes taken every 2nd lattice plane

AXES = ("x", "y", "z")
NRM = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def _plane_idx(n, m, stride):
    """m evenly spaced lattice indices along an n-point axis, on the stride
    grid, kept one stride away from the boundary planes (those slice empty)."""
    coarse = np.arange(stride, int(n) - stride, stride)
    return coarse[np.linspace(0, coarse.size - 1, m).round().astype(int)]


POS = {"x": gx[_plane_idx(nl[0], N_SLICE, SLICE_STRIDE)],
       "y": gy[_plane_idx(nl[1], N_SLICE, SLICE_STRIDE)],
       "z": gz[_plane_idx(nl[2], N_SLICE, SLICE_STRIDE)]}
DEF = {ax: int(np.argmin(np.abs(POS[ax] - t)))
       for ax, t in (("x", 0.0), ("y", 0.0), ("z", 4.0))}


def coarse_image(fields):
    """pv.ImageData of the common lattice subsampled by SLICE_STRIDE (point data)."""
    img = pv.ImageData(
        dimensions=tuple(int((n - 1) // SLICE_STRIDE + 1) for n in nl),
        origin=(gx[0], gy[0], gz[0]), spacing=(h_min * SLICE_STRIDE,) * 3)
    for name, f in fields.items():
        img.point_data[name] = f[::SLICE_STRIDE, ::SLICE_STRIDE,
                                 ::SLICE_STRIDE].transpose(2, 1, 0).ravel()
    return img


class _ActorOrder:
    """Bookkeeping of the vtk.js actor order produced by successive add_mesh
    calls: each mesh adds one actor, an enabled scalar bar one more right
    after it (verified against the exported scene)."""

    def __init__(self):
        self.n = 0

    def add(self, pl, mesh, show_bar=False, **kw):
        i = self.n
        pl.add_mesh(mesh, show_scalar_bar=show_bar, **kw)
        self.n += 1 + int(show_bar)
        return i


def _add_slices(pl, ao, imgc, sc, cmap, clim, slices, bars):
    """N_SLICE slice planes per axis for one scalar; pyvista shares one scalar
    bar per lookup table, so request it only once, at the default x position."""
    for ax in AXES:
        idxs = []
        for j, p in enumerate(POS[ax]):
            bar = (j == DEF[ax] and sc not in bars)
            i = ao.add(pl,
                       imgc.slice(normal=NRM[ax],
                                  origin=tuple(p if a == ax else 0.0
                                             for a in AXES)),
                       show_bar=bar, name=f"sl_{sc}_{ax}_{j}",
                       scalars=sc, cmap=cmap, clim=clim)
            if bar:
                bars[sc] = i + 1
            idxs.append(i)
        slices[sc][ax] = {"pos": [float(v) for v in POS[ax]], "idx": idxs}


EXPORT_TAIL = "OfflineLocalView.load(container, { base64Str });"

PANEL_JS = r"""
window.__rt3d_ui = (function () {
  const M = __MANIFEST__;
  function init(tries) {
    const rw = (window.global && window.global.renderWindow) || window.renderWindow;
    let ren = null, n = 0;
    if (rw && typeof rw.getRenderers === "function")
      rw.getRenderers().forEach(r => {
        if (r.getActors().length > n) { n = r.getActors().length; ren = r; }
      });
    if (!ren) {
      if (tries-- > 0) setTimeout(() => init(tries), 200);
      else console.error("rt3d panel: vtk.js scene not found");
      return;
    }
    const A = ren.getActors();
    if (ren.setBackground) ren.setBackground([0.07, 0.08, 0.11]);
    const st = { iso: M.iso.def, x: M.def.x, y: M.def.y, z: M.def.z,
                 sc: M.scalars[0], isoOn: true, allIso: false,
                 slOn: true, boxOn: true };
    function apply() {
      M.iso.idx.forEach((a, j) =>
        A[a].setVisibility(st.isoOn && (st.allIso || j === st.iso)));
      M.scalars.forEach(sc => {
        M.axes.forEach(ax => M.slices[sc][ax].idx.forEach((a, j) =>
          A[a].setVisibility(st.slOn && sc === st.sc && j === st[ax])));
        if (M.bars[sc] !== undefined)
          A[M.bars[sc]].setVisibility(st.slOn && sc === st.sc);
      });
      A[M.box].setVisibility(st.boxOn);
      rw.render();
    }
    const p = document.createElement("div");
    p.style.cssText = "position:fixed;left:12px;top:12px;z-index:999;width:250px;" +
      "background:rgba(15,17,23,.88);color:#dde3ee;padding:12px 14px;" +
      "border-radius:10px;font:12px/1.5 system-ui,sans-serif;" +
      "box-shadow:0 4px 18px rgba(0,0,0,.5);user-select:none";
    function lab(t) {
      const d = document.createElement("div");
      d.textContent = t; p.appendChild(d); return d;
    }
    function slider(max, val, text, key) {
      const l = lab(text(val));
      const s = document.createElement("input");
      s.type = "range"; s.min = 0; s.max = max; s.step = 1; s.value = val;
      s.style.cssText = "width:100%;margin:2px 0 8px";
      s.addEventListener("input", () => {
        st[key] = +s.value; l.textContent = text(+s.value); apply();
      });
      p.appendChild(s);
    }
    function check(txt, key) {
      const w = document.createElement("label");
      w.style.cssText = "display:block;margin:2px 0";
      const c = document.createElement("input");
      c.type = "checkbox"; c.checked = st[key];
      c.addEventListener("change", () => { st[key] = c.checked; apply(); });
      w.appendChild(c); w.appendChild(document.createTextNode(" " + txt));
      p.appendChild(w);
    }
    lab(M.title).style.fontWeight = "600";
    p.style.marginBottom = "8px";
    if (M.iso.idx.length)
      slider(M.iso.idx.length - 1, st.iso,
             j => M.iso.label + " = " + (+M.iso.levels[j]).toFixed(M.iso.dec) +
                  (M.iso.unit ? " " + M.iso.unit : ""), "iso");
    M.axes.forEach(ax =>
      slider(M.slices[M.scalars[0]][ax].pos.length - 1, st[ax],
             j => ax + " = " + (+M.slices[M.scalars[0]][ax].pos[j]).toFixed(2) +
                  " m", ax));
    if (M.scalars.length > 1) {
      const w = document.createElement("div");
      w.style.cssText = "margin:4px 0";
      M.scalars.forEach(sc => {
        const r = document.createElement("label");
        r.style.cssText = "margin-right:10px";
        const i = document.createElement("input");
        i.type = "radio"; i.name = "rt3d-sc"; i.checked = sc === st.sc;
        i.addEventListener("change", () => { st.sc = sc; apply(); });
        r.appendChild(i); r.appendChild(document.createTextNode(" " + sc));
        w.appendChild(r);
      });
      p.appendChild(w);
    }
    check("slice planes", "slOn");
    check("isosurfaces", "isoOn");
    check("all iso levels", "allIso");
    check("domain box", "boxOn");
    lab("drag: orbit \u00b7 wheel: zoom").style.cssText =
      "margin-top:8px;opacity:.6";
    document.body.appendChild(p);
    apply();
  }
  init(150);
})();
"""


def _finish(pl, ao, path, manifest, title):
    # self-consistency: every manifest actor index must exist in the scene
    for grp in manifest["slices"].values():
        for ax in grp.values():
            assert max(ax["idx"]) < ao.n, f"slice idx out of range in {path}"
    assert max(manifest["iso"]["idx"], default=-1) < ao.n
    assert manifest["box"] == ao.n - 1
    pl.reset_camera()
    pl.camera.azimuth, pl.camera.elevation = 40, 18
    pl.camera.zoom(0.92)
    pl.trame.export_html(path)
    pl.close()
    # inject the slider control panel into the exported static scene
    manifest["title"] = title
    with open(path, encoding="utf-8") as fh:
        html = fh.read()
    if EXPORT_TAIL not in html:
        raise RuntimeError(f"unexpected trame export tail in {path}")
    js = PANEL_JS.replace("__MANIFEST__", json.dumps(manifest))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html.replace(EXPORT_TAIL, EXPORT_TAIL + "\n" + js))


def scene_run(key):
    img = lattice_image(key)
    imgc = coarse_image({sc: lat[key][sc] for sc in ("J", "var")})
    ao = _ActorOrder()
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))

    ke_max = float(lat[key]["kappa_e"].max())
    iso_levels = [float(v) for v in ke_max * np.linspace(0.12, 0.92, N_ISO)]
    iso_cmap = matplotlib.colormaps["Blues"]
    iso_idx = [ao.add(pl, img.contour(isosurfaces=[lev], scalars="kappa_e"),
                      name=f"iso_{j}",
                      color=iso_cmap(0.35 + 0.6 * j / (N_ISO - 1)),
                      opacity=0.32, specular=0.6, specular_power=30)
               for j, lev in enumerate(iso_levels)]
    slices, bars = {sc: {} for sc in ("J", "var")}, {}
    for sc, cm in (("J", "viridis"), ("var", "inferno")):
        _add_slices(pl, ao, imgc, sc, cm,
                    (0.0, float(lat[key][sc].max())), slices, bars)
    box_i = ao.add(pl, pv.Box(bounds=(box.x0, box.x1, box.y0, box.y1,
                                      box.z0, box.z1)),
                   name="domain", style="wireframe", color="white",
                   line_width=1.0)
    _finish(pl, ao, os.path.join(OUT, f"scene_{key}.html"),
            {"scalars": ["J", "var"], "axes": list(AXES),
             "iso": {"label": "kappa_e", "unit": "m^-1", "dec": 2,
                     "levels": iso_levels, "idx": iso_idx, "def": N_ISO // 2},
             "slices": slices, "bars": bars, "box": box_i, "def": DEF},
            RUN_TITLES[key])


def scene_diff(a, b, tag):
    """|J_b - J_a| isosurfaces + slices on the common lattice."""
    d = np.abs(lat[b]["J"] - lat[a]["J"])
    img = pv.ImageData(dimensions=tuple(int(v) for v in nl),
                       origin=(gx[0], gy[0], gz[0]), spacing=(h_min,) * 3)
    img.point_data["dJ"] = d.transpose(2, 1, 0).ravel()
    pos = d[d > 0]
    iso_levels = ([float(v) for v in
                   np.unique(np.quantile(pos, np.linspace(0.5, 0.995, N_ISO)))]
                  if pos.size else [])
    ao = _ActorOrder()
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    iso_idx = [ao.add(pl, img.contour(isosurfaces=[lev], scalars="dJ"),
                      name=f"iso_{j}", cmap="magma", opacity=0.4,
                      specular=0.4, specular_power=20)
               for j, lev in enumerate(iso_levels)]
    slices, bars = {"dJ": {}}, {}
    _add_slices(pl, ao, coarse_image({"dJ": d}), "dJ", "magma",
                (0.0, float(d.max())), slices, bars)
    box_i = ao.add(pl, pv.Box(bounds=(box.x0, box.x1, box.y0, box.y1,
                                      box.z0, box.z1)),
                   name="domain", style="wireframe", color="white",
                   line_width=1.0)
    _finish(pl, ao, os.path.join(OUT, f"scene_diff_{tag}.html"),
            {"scalars": ["dJ"], "axes": list(AXES),
             "iso": {"label": "|dJ|", "unit": "", "dec": 3,
                     "levels": iso_levels, "idx": iso_idx,
                     "def": len(iso_levels) // 2},
             "slices": slices, "bars": bars, "box": box_i, "def": DEF},
            f"|J({b}) - J({a})|")


if pv is not None:
    for key in BASE6 + [REF]:
        scene_run(key)
    scene_diff("uni_chain_dM0", "uni_sc_dM0", "chain_vs_sc_dM0")
    scene_diff("uni_chain_dM0", "octree_dM0", "uni_vs_octree_dM0")
    scene_diff(REF, "uni_chain_dM0", "ref_vs_chain24_dM0")
    scene_diff(REF, "uni_sc_dM0", "ref_vs_sc24_dM0")
    scene_diff(REF, "octree_dM0", "ref_vs_octree3_dM0")
    print(f"HTML scenes written to {OUT}/scene_*.html")

    # %% static volume-rendering overview (manual section 11 style)
    views = []
    for key in BASE6 + [REF]:
        img = lattice_image(key)
        pl = pv.Plotter(off_screen=True, window_size=(560, 480))
        pl.add_volume(img, scalars="J", cmap="inferno",
                      opacity=np.linspace(0.10, 0.0, 24), shade=False,
                      mapper="smart", show_scalar_bar=False)
        pl.add_mesh(pv.Box(bounds=(box.x0, box.x1, box.y0, box.y1, box.z0, box.z1)),
                    style="wireframe", color="black", line_width=1.0)
        pl.reset_camera()
        pl.camera.azimuth, pl.camera.elevation = 40, 18
        pl.camera.zoom(0.92)
        views.append(pl.screenshot(return_img=True))
        pl.close()

    fig, axes = plt.subplots(2, 4, figsize=(16.4, 8.4))
    for ax, img, key in zip(axes.ravel(), views, BASE6 + [REF]):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(RUN_TITLES[key], fontsize=10)
    for ax in axes.ravel()[len(views):]:
        ax.axis("off")
    fig.suptitle(r"$J$ volume rendering on the common lattice "
                 "(opacity increases toward low $J$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vol_j.png"), dpi=120)
    plt.close(fig)

print(f"all outputs in {OUT}/")
