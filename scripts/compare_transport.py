# -*- coding: utf-8 -*-
"""
Transport-scheme comparison — chain vs. SC vs. octree, with/without delta-M
(plan/compare-transport-plan.md).

Six runs on the MANUAL section 11 worked example (Gaussian cloud + Lambertian
ground + solar beam, Ng accelerated, 42 ordinates):

    unigrid 24^3, transport="chain"   delta_M = 0 / 1
    unigrid 24^3, transport="sc"      delta_M = 0 / 1
    octree (h0 = 1.25, max_depth 3), exact marches   delta_M = 0 / 1

Diagnostics per run: mean intensity J, angular variance of the intensity
(1/4pi int (I - J)^2 dOmega), and the upward radiance at TOA (F^+(x,y) maps,
per-ordinate domain-mean radiance).  All fields are sampled onto one common
64^3 lattice (finest octree cell) so every pair of runs is directly
comparable.  Interactive 3D scenes are exported as self-contained vtk.js
HTML (pyvista trame export: orbit / actor toggling / colour maps in the
browser, no server needed to view).

Run from the repo root:
    macOS/Linux:  ~/miniforge3/envs/pyarts/bin/python scripts/compare_transport.py
    Windows:      $env:Path = "$env:USERPROFILE\\miniforge3\\envs\\pyarts\\Library\\bin;" + $env:Path
                  & "$env:USERPROFILE\\miniforge3\\envs\\pyarts\\python.exe" scripts/compare_transport.py

Units as in the manual: lengths m, kappa in m^-1, radiance normalised to the
incident beam flux F0 = 1.
"""

# %% imports and configuration
import asyncio
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

dg = icosahedric_grid(n_refine=1)              # 42 ordinates (development grid)
print(f"N_dir = {dg.n_dir},  sum(w) - 4pi = {dg.weights.sum() - 4*np.pi:.3e}")

mu_sun = np.cos(np.radians(30.0))
beam = {"omega_sun": np.array([np.sqrt(1.0 - mu_sun**2), 0.0, -mu_sun]),
        "F0": 1.0}

bc = boundary_conditions(top_I=1.0,
                         ground=ground_surface(albedo=0.3, emissivity=0.0, B_g=0.0))

N_ITER, TOL = 60, 1e-6

# %% grids and input fields (identical to the manual example)
n = 24
xs = np.linspace(box.x0, box.x1, n + 1)[:-1] + (box.Lx / n) / 2
ugrid = unigrid(xs, xs, np.linspace(box.z0, box.z1, n + 1)[:-1] + (box.Lz / n) / 2)
uc = ugrid.centers
u_kappa_e = case["kappa_e"](*uc.T)
u_kappa_s = case["kappa_s"](*uc.T)
u_g = case["g"](*uc.T)
u_B = np.zeros(ugrid.n_cells)

oct_ = octree_grid(box, h0=1.25)
oct_.refine(case["kappa_e"], tau_max=2.0, eps_grad=1.0, max_depth=3)
oct_.balance_21(max_depth=3)
oct_.finalize()
oct_.sample_fields(kappa_e=case["kappa_e"], kappa_s=case["kappa_s"],
                   g=case["g"], T=case["T"])
print(f"octree: n_leaf={oct_.n_leaf}, max_level={oct_.max_level}")

# %% the six runs
RUNS = [
    ("uni_chain_dM0", "uni", "chain", 0),
    ("uni_chain_dM1", "uni", "chain", 1),
    ("uni_sc_dM0",    "uni", "sc",    0),
    ("uni_sc_dM1",    "uni", "sc",    1),
    ("octree_dM0",    "oct", "chain", 0),
    ("octree_dM1",    "oct", "chain", 1),
]

results, wall = {}, {}
for key, which, transport, dM in RUNS:
    grid = ugrid if which == "uni" else oct_
    k_e = u_kappa_e if which == "uni" else oct_.kappa_e
    k_s = u_kappa_s if which == "uni" else oct_.kappa_s
    g_c = u_g if which == "uni" else oct_.g
    t0 = time.perf_counter()
    results[key] = lambda_iteration(
        grid, dg, k_e, k_s, u_B if which == "uni" else np.zeros(oct_.n_leaf), bc,
        g=g_c, periodic_box=box, beam=beam,
        n_iter=N_ITER, tol=TOL, ng=True, delta_M=dM, transport=transport,
    )
    wall[key] = time.perf_counter() - t0
    r = results[key]
    print(f"{key:14s}  n_iter={r['n_iter']:3d}  converged={r['converged']}  "
          f"wall={wall[key]:7.2f} s")

# %% common lattice: every run sampled onto the finest-octree-cell grid
h_min = 2.0 * oct_.half.min()
nl = (np.array([box.Lx, box.Ly, box.Lz]) / h_min).astype(int)
gx = box.x0 + (np.arange(nl[0]) + 0.5) * h_min
gy = box.y0 + (np.arange(nl[1]) + 0.5) * h_min
gz = box.z0 + (np.arange(nl[2]) + 0.5) * h_min
P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
li_oct = oct_.locate(P)                       # octree leaf per lattice point

# unigrid cell index per lattice coordinate (piecewise-constant, exact)
iu = np.clip(((gx - box.x0) / ugrid.hx).astype(int), 0, ugrid.nx - 1)
ju = np.clip(((gy - box.y0) / ugrid.hy).astype(int), 0, ugrid.ny - 1)
ku = np.clip(((gz - box.z0) / ugrid.hz).astype(int), 0, ugrid.nz - 1)
IU, JU, KU = np.meshgrid(iu, ju, ku, indexing="ij")


# angular variance per run (from the swept intensities)
for key, *_ in RUNS:
    results[key]["var"] = angular_variance(results[key]["I"], dg)

RUN_WHICH = {key: which for key, which, _, _ in RUNS}


def lat_field(key, name):
    """Cell field of a run sampled onto the common lattice, shape (nxl, nyl, nzl)."""
    if name == "kappa_e":
        f = oct_.kappa_e if RUN_WHICH[key] == "oct" else u_kappa_e
    else:
        f = results[key][name]
    if RUN_WHICH[key] == "oct":
        return np.asarray(f)[li_oct].reshape(tuple(int(v) for v in nl))
    f4 = np.asarray(f).reshape(ugrid.nz, ugrid.ny, ugrid.nx)
    return f4[KU, JU, IU]


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
    which = next(w for kk, w, _, _ in RUNS if kk == key)
    if which == "oct":
        top = oct_.centers[:, 2] + oct_.half >= box.z1 - 1e-9
        area = (2.0 * oct_.half[top]) ** 2
        I_top, F_top = r["I"][top], r["F"][top]
    else:
        top = np.zeros(ugrid.n_cells, dtype=bool)
        top[(ugrid.nz - 1) * ugrid.ny * ugrid.nx:] = True
        area = np.ones(top.sum())
        I_top, F_top = r["I"][top], r["F"][top]
    mean_I = (I_top[:, dg.up_idx] * area[:, None]).sum(axis=0) / area.sum()
    return mean_I, F_top[:, 2].mean()


toa = {key: toa_stats(key) for key, *_ in RUNS}

print("\nrun             transport  delta_M  n_iter  conv  wall [s]  "
      "<J>      mean F_z @TOA")
rows = []
for key, which, transport, dM in RUNS:
    r = results[key]
    meanJ = r["J"].mean()
    rows.append((key, transport, dM, r["n_iter"], r["converged"], wall[key],
                 meanJ, toa[key][1]))
    print(f"{key:14s}  {transport:9s}  {dM:7d}  {r['n_iter']:6d}  "
          f"{str(r['converged']):5s}  {wall[key]:8.2f}  {meanJ:.4f}  {toa[key][1]:+.4f}")

with open(os.path.join(OUT, "summary.csv"), "w") as fh:
    fh.write("run,transport,delta_M,n_iter,converged,wall_s,mean_J,mean_Fz_TOA\n")
    for k, t, d, it, cv, w, j, fz in rows:
        fh.write(f"{k},{t},{d},{it},{int(cv)},{w:.3f},{j:.6f},{fz:.6f}\n")

# delta-M equivalence check: converged J must not depend on delta_M (same grid)
for base in ("uni_chain", "uni_sc", "octree"):
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
vmax = max(lat_yz(k, "J").max() for k, *_ in RUNS)
for ax, (key, _, _, _) in zip(axes.ravel(), RUNS):
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
nband = int(round(box.Lz / oct_.h0))
zband = np.linspace(box.z0, box.z1, nband + 1)
zc = 0.5 * (zband[:-1] + zband[1:])
zi_oct = np.clip(((oct_.centers[:, 2] - box.z0) / oct_.h0).astype(int), 0, nband - 1)
kk_u = np.arange(ugrid.n_cells) // (ugrid.ny * ugrid.nx)
profiles = {}
for key, which, _, _ in RUNS:
    J = results[key]["J"]
    if which == "oct":
        prof = np.array([J[zi_oct == b].mean() for b in range(nband)])
    else:
        prof = np.array([J[(kk_u >= int(np.floor((zband[b] - box.z0) / ugrid.hz)))
                           & (kk_u < int(np.floor((zband[b + 1] - box.z0) / ugrid.hz)))].mean()
                         for b in range(nband)])
    profiles[key] = prof

fig, ax = plt.subplots(figsize=(5.6, 4.6))
for key, *_ in RUNS:
    style = "--" if key.startswith("octree") else "-"
    style += "." if key.endswith("dM1") else ""
    ax.plot(profiles[key], zc, style, label=RUN_TITLES[key])
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
vmax = max(lat_yz(k, "var").max() for k, *_ in RUNS)
for ax, (key, _, _, _) in zip(axes.ravel(), RUNS):
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
    which = next(w for kk, w, _, _ in RUNS if kk == key)
    w_up = dg.weights[dg.up_idx] * dg.directions[dg.up_idx, 2]
    if which == "oct":
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        Pt = np.column_stack([GX.ravel(), GY.ravel(),
                              np.full(GX.size, box.z1 - h_min / 2)])
        leaf = oct_.locate(Pt)
        F_up = r["I"][leaf][:, dg.up_idx] @ w_up
        return F_up.reshape(nl[0], nl[1])
    F_up_top = r["I"][(ugrid.nz - 1) * ugrid.ny * ugrid.nx:][:, dg.up_idx] @ w_up
    F_up_top = F_up_top.reshape(ugrid.ny, ugrid.nx)
    return F_up_top[np.ix_(ju, iu)]


fup = {key: fup_map(key) for key, *_ in RUNS}
ye = np.concatenate([gy - h_min / 2, [gy[-1] + h_min / 2]])
X2, Y2 = np.meshgrid(xe, ye)

fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), layout="constrained")
vmax = max(f.max() for f in fup.values())
for ax, (key, _, _, _) in zip(axes.ravel(), RUNS):
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

fig, ax = plt.subplots(figsize=(6.2, 4.6))
mu_up = dg.directions[dg.up_idx, 2]
order = np.argsort(mu_up)
for key, *_ in RUNS:
    style = "--" if key.startswith("octree") else "-"
    style += "." if key.endswith("dM1") else ""
    ax.plot(mu_up[order], toa[key][0][order], style, label=RUN_TITLES[key])
ax.set_xlabel(r"$\mu = \cos\vartheta$")
ax.set_ylabel(r"domain-mean upward $I\,/F_0$ [sr$^{-1}$]")
ax.set_title("TOA upward radiance vs ordinate direction")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "toa_mu_spectra.png"), dpi=130)
plt.close(fig)

np.savetxt(os.path.join(OUT, "toa_mean_radiance.csv"),
           np.column_stack([mu_up] + [toa[k][0] for k, *_ in RUNS]),
           delimiter=",", fmt="%.6f",
           header="mu," + ",".join(k for k, *_ in RUNS), comments="")

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

# %% interactive 3D scenes (self-contained vtk.js HTML, offline viewer)
import pyvista as pv

pv.OFF_SCREEN = True
from trame_pyvista.jupyter import launch_server


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


ISO_LEVELS = [0.5, 1.0, 2.0, 3.0]
ISO_COLS = [(0.75, 0.85, 1.0), (0.55, 0.7, 1.0), (0.35, 0.5, 0.95), (0.2, 0.3, 0.85)]
SLICES = [("x", (0.0, 0.0, 0.0), "x = 0"),
          ("y", (0.0, 0.0, 0.0), "y = 0"),
          ("z", (0.0, 0.0, 4.0), "z = 4")]


def scene_run(key):
    img = lattice_image(key)
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    for lev, col in zip(ISO_LEVELS, ISO_COLS):
        pl.add_mesh(img.contour(isosurfaces=[lev], scalars="kappa_e"),
                    name=f"kappa_e = {lev:g}", color=col, opacity=0.3,
                    specular=0.5, specular_power=25)
    for sc, cm in (("J", "viridis"), ("var", "inferno")):
        for nrm, org, tag in SLICES:
            pl.add_mesh(img.slice(normal=nrm, origin=org), name=f"{sc} @ {tag}",
                        scalars=sc, cmap=cm, show_scalar_bar=(tag == "y = 0"))
    pl.add_mesh(pv.Box(bounds=(box.x0, box.x1, box.y0, box.y1, box.z0, box.z1)),
                name="domain", style="wireframe", color="black", line_width=1.0)
    pl.reset_camera()
    pl.camera.azimuth, pl.camera.elevation = 40, 18
    pl.camera.zoom(0.92)
    pl.trame.export_html(os.path.join(OUT, f"scene_{key}.html"))
    pl.close()


def scene_diff(a, b, tag):
    """|J_b - J_a| isosurfaces + slices on the common lattice."""
    d = np.abs(lat[b]["J"] - lat[a]["J"])
    img = pv.ImageData(dimensions=tuple(int(v) for v in nl),
                       origin=(gx[0], gy[0], gz[0]), spacing=(h_min,) * 3)
    img.point_data["dJ"] = d.transpose(2, 1, 0).ravel()
    pos = d[d > 0]
    levels = np.unique(np.quantile(pos, [0.80, 0.90, 0.99])) if pos.size else []
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    for lev in levels:
        pl.add_mesh(img.contour(isosurfaces=[lev], scalars="dJ"),
                    name=f"|dJ| = {lev:.3f}", cmap="magma", opacity=0.4)
    for nrm, org, t in SLICES:
        pl.add_mesh(img.slice(normal=nrm, origin=org), name=f"|dJ| @ {t}",
                    scalars="dJ", cmap="magma", show_scalar_bar=(t == "y = 0"))
    pl.add_mesh(pv.Box(bounds=(box.x0, box.x1, box.y0, box.y1, box.z0, box.z1)),
                name="domain", style="wireframe", color="black", line_width=1.0)
    pl.reset_camera()
    pl.camera.azimuth, pl.camera.elevation = 40, 18
    pl.camera.zoom(0.92)
    pl.trame.export_html(os.path.join(OUT, f"scene_diff_{tag}.html"))
    pl.close()


for key, *_ in RUNS:
    scene_run(key)
scene_diff("uni_chain_dM0", "uni_sc_dM0", "chain_vs_sc_dM0")
scene_diff("uni_chain_dM0", "octree_dM0", "uni_vs_octree_dM0")
print(f"HTML scenes written to {OUT}/scene_*.html")

# %% static volume-rendering overview (manual section 11 style)
views = []
for key, *_ in RUNS:
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

fig, axes = plt.subplots(2, 3, figsize=(12.4, 10.6))
for ax, img, (key, _, _, _) in zip(axes.ravel(), views, RUNS):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(RUN_TITLES[key], fontsize=10)
fig.suptitle(r"$J$ volume rendering on the common lattice "
             "(opacity increases toward low $J$)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "vol_j.png"), dpi=120)
plt.close(fig)

print(f"all outputs in {OUT}/")
