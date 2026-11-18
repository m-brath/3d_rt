# -*- coding: utf-8 -*-
"""
Report figures for the project report (report/main.tex).

Generates **vector PDF** figures with one shared matplotlib style into
``report/figures/`` and LaTeX table fragments into ``report/tables/``.

Everything is reused from existing artifacts wherever possible:

* ``plot/solver_matrix/_solve_cache/oct-REF_chain.npz`` — the fine-octree
  reference field (42 ordinates) of the solver-matrix case,
* ``plot/compare_transport/_solve_cache/*.npz`` — the transport-comparison
  runs (octree / unigrid chain / unigrid sc, delta-M 0/1, incl. the 642-ordinate
  reference) and the TOA radiance CSVs,
* ``plot/solver_matrix/{matrix,ranking,convergence,mc_stats}.csv`` — all
  scalar report numbers,

and only the few field figures need fresh solves (u24/chain, u24/sc, oct-d3,
slab runs, MC spot checks, ~30 s total); those are disk-cached under
``report/figures/_cache/`` so re-runs are instant.

Run from the repo root:
    ~/miniforge3/envs/pyarts/bin/python scripts/report_figures.py
"""

# %% imports and shared style
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FIG = "report/figures"
TAB = "report/tables"
CACHE = os.path.join(FIG, "_cache")
for d in (FIG, TAB, CACHE):
    os.makedirs(d, exist_ok=True)

MATRIX_CSV = "plot/solver_matrix/matrix.csv"
RANKING_CSV = "plot/solver_matrix/ranking.csv"
CONV_CSV = "plot/solver_matrix/convergence.csv"
MCSTATS_CSV = "plot/solver_matrix/mc_stats.csv"
CT_CACHE = "plot/compare_transport/_solve_cache"
CT_FIG = "plot/compare_transport"

# --- shared style: serif + Computer Modern math, sized for the report text ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 7.0,
    "legend.framealpha": 0.9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # editable text in the PDF
})


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def cached_npz(key, fn):
    """fn() -> dict of arrays, disk-cached under CACHE."""
    p = os.path.join(CACHE, key + ".npz")
    if os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    res = fn()
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in res.items()})
    return res


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# =============================================================================
# helper: vector slice renderer — draws every leaf/cell crossing the yz plane
# at x0 as a rectangle colored by a per-cell scalar (octree-accurate, no
# pointillistic rastering of coarse leaves)
# =============================================================================

def poly_yz(ax, grid, field, cmap, vmin=None, vmax=None, x0=0.0, halfw=0.4):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    c = grid.centers
    if hasattr(grid, "half"):
        hy = hz = hx = grid.half
    else:
        hy = np.full(c.shape[0], grid.hy / 2.0)
        hz = np.full(c.shape[0], grid.hz / 2.0)
        hx = np.full(c.shape[0], grid.hx / 2.0)
    sel = np.abs(c[:, 0] - x0) < hx + halfw
    # draw far-from-slice leaves first, so leaves nearest x0 paint on top
    order = np.argsort(-np.abs(c[sel, 0] - x0))
    C, HY, HZ = c[sel][order], hy[sel][order], hz[sel][order]
    V = np.asarray(field)[sel][order]
    good = np.isfinite(V)
    C, HY, HZ, V = C[good], HY[good], HZ[good], V[good]
    lo = float(np.min(V)) if vmin is None else vmin
    hi = float(np.max(V)) if vmax is None else vmax
    m = ScalarMappable(cmap=cmap, norm=Normalize(lo, hi))
    m.set_array([])
    faces = np.empty((V.size, 4, 2))
    faces[:, :, 0] = np.stack([C[:, 1] - HY, C[:, 1] + HY, C[:, 1] + HY,
                               C[:, 1] - HY], axis=1)
    faces[:, :, 1] = np.stack([C[:, 2] - HZ, C[:, 2] - HZ, C[:, 2] + HZ,
                               C[:, 2] + HZ], axis=1)
    ax.add_collection(PolyCollection(faces, facecolors=m.to_rgba(V),
                                     edgecolors="none"))
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 10)
    return m


# =============================================================================
# Fig 1 — icosahedral direction grids with solid-angle weights
# =============================================================================

def fig_directions():
    from rt3d.directions import icosahedric_grid

    def sphere_panel(ax, dg, title):
        V, T = dg.directions, dg.triangles
        polys = Poly3DCollection(V[T], facecolor="0.90", edgecolor="0.55",
                                 linewidths=0.3, alpha=0.35)
        ax.add_collection3d(polys)
        ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=dg.weights, cmap="viridis",
                   s=18, zorder=5, depthshade=False)
        lim = 1.05
        ax.set_box_aspect((1, 1, 1))
        for a, fn in ((ax.set_xlim, None), (ax.set_ylim, None), (ax.set_zlim, None)):
            a(-lim, lim)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=-62)
        ax.set_title(title, y=0.02)

    fig = plt.figure(figsize=(6.3, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.02)
    for k, nref in enumerate((0, 1)):
        dg = icosahedric_grid(nref)
        ax = fig.add_subplot(gs[0, k], projection="3d")
        sphere_panel(ax, dg, f"$n_{{\\mathrm{{refine}}}}={nref}$: {dg.n_dir} ordinates")
    dg1 = icosahedric_grid(1)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(dg1.weights, bins=8, color="tab:blue", edgecolor="k", linewidth=0.4)
    ax.axhline(4 * np.pi / dg1.n_dir, color="crimson", ls="--", lw=1.0,
               label=r"equal-area $4\pi/N$")
    ax.set_xlabel(r"solid-angle weight $w_i$ [sr]")
    ax.set_ylabel("count")
    ax.set_title(r"weights, $n_{\mathrm{refine}}=1$ ($\sum w_i=4\pi$)")
    ax.legend(loc="upper right")
    save(fig, "fig_directions.pdf")


# =============================================================================
# Fig 2 — octree of the Gaussian-cloud case: fields and leaf structure
# =============================================================================

def fig_octree():
    from matrix_cases import cloud_case, cloud_octree

    case = cloud_case()
    oct3 = cloud_octree(2.0, 3)

    # field panels on a fine yz raster at x = 0
    ny = nz = 240
    ys = np.linspace(-5, 5, ny)
    zs = np.linspace(0, 10, nz)
    Y, Z = np.meshgrid(ys, zs)
    ke = np.asarray(case["kappa_e"](np.zeros_like(Y), Y, Z))
    g_field = np.asarray(case["g"](np.zeros_like(Y), Y, Z))
    omega_s = np.where(ke > 0, np.asarray(case["kappa_s"](np.zeros_like(Y), Y, Z)) / np.maximum(ke, 1e-30), 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.25))
    ext = [-5, 5, 0, 10]
    im0 = axes[0].imshow(ke, origin="lower", extent=ext, cmap="magma",
                         vmin=0, vmax=ke.max())
    axes[0].set_title(r"$\kappa_e(x{=}0,y,z)$ [m$^{-1}$]")
    im1 = axes[1].imshow(omega_s, origin="lower", extent=ext,
                         cmap="viridis", vmin=0, vmax=0.9)
    axes[1].set_title(r"single-scattering albedo $\omega_s$")
    for ax in axes[:2]:
        ax.set_aspect("equal")
    cb = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    cb = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)

    ax = axes[2]
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    cmap = plt.get_cmap("tab10")
    # leaves crossing the x = 0 plane, drawn in (y, z) to match the field
    # panels; the base lattice face sits exactly on x = 0, so the straddle
    # test must include equality (|c_x| <= h) or no leaf is ever selected
    C, H, L = oct3.centers, oct3.half, oct3.level
    sel = np.abs(C[:, 0]) <= H + 1e-12
    order = np.argsort(-H[sel])  # coarse first, fine paint on top
    C, H, L = C[sel][order], H[sel][order], L[sel][order]
    faces = np.empty((C.shape[0], 4, 2))
    faces[:, :, 0] = np.stack([C[:, 1] - H, C[:, 1] + H, C[:, 1] + H,
                               C[:, 1] - H], axis=1)
    faces[:, :, 1] = np.stack([C[:, 2] - H, C[:, 2] - H, C[:, 2] + H,
                               C[:, 2] + H], axis=1)
    ax.add_collection(PolyCollection(faces,
                                     facecolors=[cmap(int(l)) for l in L],
                                     edgecolors="0.25", linewidths=0.3,
                                     alpha=0.75))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(lev),
                             edgecolor="0.25", lw=0.4)
               for lev in (0, 1, 2, 3)]
    ax.legend(handles, [f"level {lev}" for lev in (0, 1, 2, 3)],
              loc="upper right", fontsize=6.5, framealpha=0.95,
              borderpad=0.4, handlelength=1.1, handletextpad=0.5)
    ax.set_title(f"octree $\\tau_{{\\max}}$=2, depth 3 ({oct3.n_leaf} leaves)")
    ax.set_xlabel("y [m]")
    save(fig, "fig_octree.pdf")


# =============================================================================
# Fig 3 — lattice stencil selection (unigrid chain)
# =============================================================================

def fig_stencil():
    from rt3d.characteristics import stencil_for_direction

    def arrow(ax, x, y, color, lw=2.0, z=4):
        ax.annotate("", xy=(x, y), xytext=(0, 0), zorder=z,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=18))

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
    ax = axes[0]
    h = np.array([1.0, 1.0, 1.0])
    omega = np.array([0.526, 0.316, 0.789])
    omega = omega / np.linalg.norm(omega)
    s, vhat = stencil_for_direction(omega, h, max_offset=3)
    R = 3
    cands = np.array([t for t in np.ndindex(*([2 * R + 1] * 3))
                      if any(c != R for c in t)]) - R
    cands = cands[cands[:, 2] != 0]
    cos = (cands @ omega) / np.linalg.norm(cands * h, axis=1)
    ax.scatter(cands[:, 0], cands[:, 2], c=cos, cmap="viridis", vmin=-1,
               vmax=1, s=22, zorder=2)
    arrow(ax, 3 * omega[0], 3 * omega[2], "crimson")
    arrow(ax, s[0], s[2], "tab:blue")
    ax.scatter([s[0]], [s[2]], facecolors="none", edgecolors="tab:blue",
               s=200, lw=1.6, zorder=5)
    ax.plot([0], [0], "k*", ms=11, zorder=5)
    ax.annotate(r"$\Omega$", (3 * omega[0], 3 * omega[2]),
                xytext=(3 * omega[0] - 0.15, 3 * omega[2] + 0.35),
                color="crimson", fontsize=9)
    ax.annotate(rf"chosen $\mathbf{{s}}=({s[0]},{s[2]})$", (s[0], s[2]),
                xytext=(s[0] - 3.4, s[2] - 0.55), color="tab:blue", fontsize=8)
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("z [cells]")
    ax.set_title("candidates colored by $\\cos\\angle(\\mathbf{sh},\\Omega)$")

    ax = axes[1]
    h2 = np.array([1.0, 1.0, 0.25])
    omega2 = np.array([1.0, 0.0, 3.0])
    omega2 = omega2 / np.linalg.norm(omega2)
    s2, vhat2 = stencil_for_direction(omega2, h2)
    GX, GZ = np.meshgrid(np.arange(0, 5), np.arange(1, 15))
    pts = np.column_stack([GX.ravel() * h2[0], GZ.ravel() * h2[2]])
    cos2 = (pts @ omega2[[0, 2]]) / np.linalg.norm(pts, axis=1)
    ax.scatter(pts[:, 0], pts[:, 1], c=cos2, cmap="viridis", vmin=-1, vmax=1,
               s=22, zorder=2)
    arrow(ax, s2[0] * h2[0], s2[2] * h2[2], "tab:blue", lw=3.2)
    arrow(ax, omega2[0] * 3.3, omega2[2] * 3.3, "crimson", lw=1.2, z=5)
    arrow(ax, 1 * h2[0], 3 * h2[2], "0.5", lw=1.2, z=3)
    ax.scatter([s2[0] * h2[0]], [s2[2] * h2[2]], facecolors="none",
               edgecolors="tab:blue", s=200, lw=1.6, zorder=6)
    ax.plot([0], [0], "k*", ms=11, zorder=5)
    ax.annotate(r"$\Omega\propto(1,3)$", (1.05, 3.0), xytext=(1.3, 2.9),
                color="crimson", fontsize=8)
    ax.annotate(r"chosen $\mathbf{s}=(1,12)$", (1.0, 3.0), xytext=(1.4, 2.0),
                color="tab:blue", fontsize=8)
    ax.annotate(r"$\mathbf{s}=(1,3)$: rejected", (1.0, 0.75), xytext=(1.3, 0.25),
                color="0.35", fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("anisotropic $h_z=h_x/4$: radius scales with $h_{\\max}/h_{\\min}$")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_stencil.pdf")


# =============================================================================
# Fig 4 — sub-segment splitting of one unigrid characteristic
# =============================================================================

def fig_subsegments():
    from rt3d.characteristics import subsegments_uniform

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    h = np.array([1.0, 1.0, 1.0])
    s = np.array([2, 0, 3])
    rel, lengths = subsegments_uniform(s, h)
    start = -s
    cells = np.vstack([start, start + rel])
    for idx, (ci, cj, ck) in enumerate(cells):
        ax.add_patch(plt.Rectangle((ci - 0.5, ck - 0.5), 1, 1,
                                   facecolor="0.92" if idx == 0 else "none",
                                   edgecolor="0.6", lw=0.7, zorder=1))
    for m, (ri, rj, rk) in enumerate(rel):
        ax.add_patch(plt.Rectangle((start[0] + ri - 0.5, start[2] + rk - 0.5),
                                   1, 1, facecolor=plt.cm.viridis(m / len(rel)),
                                   edgecolor="0.3", lw=0.7, alpha=0.55, zorder=2))
    p0 = np.array([start[0], start[2]], float)
    p1 = np.array([0.0, 0.0])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", lw=1.6, zorder=4)
    frac = np.concatenate([[0], np.cumsum(lengths) / lengths.sum()])
    pts = p0[None, :] + frac[:, None] * (p1 - p0)[None, :]
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=4, mfc="white", mec="k", zorder=5)
    ax.plot(*p0, "ks", ms=6, zorder=6)
    ax.plot(*p1, "k^", ms=7, zorder=6)
    ax.annotate("$u$ center", p0, xytext=(p0[0] - 0.4, p0[1] - 0.62), fontsize=7.5)
    ax.annotate("$c$ center", p1, xytext=(0.15, -0.55), fontsize=7.5)
    for m in range(len(rel)):
        mid = 0.5 * (pts[m] + pts[m + 1])
        ax.annotate(f"{m}", mid, xytext=(mid[0] - 0.42, mid[1] + 0.16),
                    fontsize=6.5, color="0.15")
    ax.set_xlim(-3.3, 1.1)
    ax.set_ylim(-4.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("z [cells]")
    ax.set_title(rf"$u_c\to c_c$ sub-segments, $\mathbf{{s}}=({s[0]},{s[1]},{s[2]})$, $M$={len(rel)}")
    save(fig, "fig_subsegments.pdf")


# =============================================================================
# Fig 5 — exact backward march through the octree
# =============================================================================

def fig_octree_march():
    from rt3d.geometry import cuboid
    from rt3d.octree import march_octree, octree_grid

    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    box = cuboid(0, 4, 0, 4, 0, 4)
    oct = octree_grid(box, h0=1.0)

    def base_id(i, j, k):
        return k * 16 + j * 4 + i

    oct.split(np.array([base_id(1, 1, 1), base_id(2, 2, 2), base_id(2, 1, 2)]))
    oct.balance_21(max_depth=6)
    oct.finalize()

    omega = np.array([0.55, 0.25, -0.80])
    omega = omega / np.linalg.norm(omega)
    target = np.array([3.3, 2.0, 1.3])
    leaf0 = int(np.argmin(np.linalg.norm(oct.centers - target, axis=1)))
    origin = oct.centers[leaf0]
    budget = (box.z1 - origin[2]) / abs(omega[2])
    sr, sl, sLen = march_octree(origin[None, :], -omega, oct, np.array([budget]))

    for n in range(oct.n_leaf):
        c, h = oct.centers[n], oct.half[n]
        if abs(c[1] - 2.0) <= h:
            ax.add_patch(plt.Rectangle((c[0] - h, c[2] - h), 2 * h, 2 * h,
                                       facecolor="none", edgecolor="0.55",
                                       lw=0.7, zorder=1))
    M = sLen.size
    p = origin.copy()
    cmap = plt.cm.plasma
    for m in range(M):
        q = p + sLen[m] * (-omega)
        ax.plot([p[0], q[0]], [p[2], q[2]], color=cmap(m / max(M - 1, 1)),
                lw=2.2, zorder=3, solid_capstyle="round")
        mid = 0.5 * (p + q)
        ax.text(mid[0], mid[2], str(m), fontsize=6, color="0.15", zorder=4,
                ha="center", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
        p = q
    ax.plot(*origin[[0, 2]], "k*", ms=11, zorder=5)
    ax.annotate("leaf center", origin[[0, 2]],
                xytext=(origin[0] - 1.15, origin[2] - 0.15), fontsize=7)
    ax.plot(*p[[0, 2]], "kv", ms=8, zorder=5)
    ax.annotate("entry", p[[0, 2]], xytext=(p[0] - 1.35, p[2] - 0.32), fontsize=7)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("backward march along $-\\Omega$ (segments in order)")
    save(fig, "fig_octree_march.pdf")


# =============================================================================
# Fig 6 — Beer-Lambert invariant + F11 (grazing-ordinate error of unigrid)
# =============================================================================

def fig_beer_f11():
    from rt3d.characteristics import build_chain
    from rt3d.directions import icosahedric_grid
    from rt3d.geometry import cuboid, unigrid
    from rt3d.octree import octree_grid
    from rt3d.short_characteristics import build_sc_chain, sweep_direction_sc
    from rt3d.transport import (boundary_conditions, precompute_octree_march,
                                sweep_direction, sweep_direction_octree_exact)

    nz = 40
    z = np.linspace(0.5, 10.0 - 0.5, nz)
    grid = unigrid(np.array([0.0]), np.array([0.0]), z)
    box = cuboid(-0.5, 0.5, -0.5, 0.5, 0.0, 10.0)
    kappa = 0.7
    kappa_e = np.full(grid.n_cells, kappa)
    z_top, z_bot = grid.box[2][1], grid.box[2][0]
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    S = np.zeros(grid.n_cells)
    dg = icosahedric_grid(1)

    # octree slab (exact march): homogeneous refinement
    oct = octree_grid(box, h0=1.0)
    oct.refine(lambda x, y, zz: np.full_like(np.asarray(x, float), kappa),
               tau_max=0.35, eps_grad=1.0, max_depth=5)
    oct.balance_21(max_depth=5)
    oct.finalize()
    ke_o = np.full(oct.n_leaf, kappa)

    def exact(centers, vz):
        entry_I = np.where(vz < 0, bc.top_I, bc.bottom_I)
        z_entry = np.where(vz < 0, z_top, z_bot)
        path = np.abs(z_entry - centers[:, 2]) / abs(vz)
        return entry_I * np.exp(-kappa * path), path

    err_chain, err_sc, err_oct, taus = [], [], [], []
    sample = []
    for d in range(dg.n_dir):
        om = dg.directions[d]
        chain = build_chain(grid, om, periodic_box=box)
        vz = chain["vhat"][2]
        I_exact, path = exact(grid.centers, vz)
        I1 = sweep_direction(grid, chain, kappa_e, S, bc)
        err_chain.append(np.max(np.abs(I1 - I_exact)))
        I2 = sweep_direction_sc(grid, build_sc_chain(grid, om, periodic_box=box),
                                kappa_e, S, bc)
        err_sc.append(np.max(np.abs(I2 - I_exact)))
        march = precompute_octree_march(oct, om, periodic_box=box)
        I3 = sweep_direction_octree_exact(oct, march, ke_o, S, bc)
        Ie_o, _ = exact(oct.centers, om[2])
        err_oct.append(np.max(np.abs(I3 - Ie_o)))
        taus.append(kappa * path.max())
        if d in (2, 12, 24, 35):
            sample.append((om, I1, vz))
    err_chain, err_sc, err_oct = map(np.array, (err_chain, err_sc, err_oct))
    taus = np.array(taus)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    ax = axes[0]
    for om, I, vz in sample:
        ax.plot(I, grid.centers[:, 2], ".", ms=3.2, label=f"chain, $\\mu$={vz:+.2f}")
    ax.plot(np.exp(-kappa * (z_top - grid.centers[:, 2])), grid.centers[:, 2],
            "k-", lw=1.2, label=r"exact $e^{-\kappa_e z'}$")
    ax.set_xlabel(r"$I$ (entry $I=1$)")
    ax.set_ylabel("z [m]")
    ax.set_title("pure absorber, $\\kappa_e=0.7$ m$^{-1}$")
    ax.legend(loc="center right")

    ax = axes[1]
    ax.semilogy(taus, np.maximum(err_chain, 1e-16), "o", ms=4,
                label="chain: exact to fp precision")
    ax.semilogy(taus, np.maximum(err_oct, 1e-16), "s", ms=3.5, mfc="none",
                label="octree march: exact")
    ax.semilogy(taus, np.maximum(err_sc, 1e-16), "^", ms=4, color="crimson",
                label="unigrid SC stencil (F11)")
    ax.set_xlabel(r"optical depth along ray $\kappa_e\,\ell_{\max}$")
    ax.set_ylabel(r"$\max_I|I-I_{\rm exact}|$")
    ax.set_title("per-ordinate max deviation, all 42 ordinates")
    ax.legend(loc="center right", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_beer_f11.pdf")


# =============================================================================
# Fig 7 — 1D slab against the 1D DO reference (phase-2 validation)
# =============================================================================

def fig_slab():
    from rt3d.cases.slab_1d import d1o_J, slab_grid, two_stream_J
    from rt3d.directions import icosahedric_grid
    from rt3d.solver import lambda_iteration
    from rt3d.transport import boundary_conditions

    KAPPA, TAU, NZ = 1.0, 5.0, 50
    grid, box, HZ = slab_grid(TAU, KAPPA, NZ)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    dg = icosahedric_grid(1)
    tau_from_top = TAU - KAPPA * grid.z

    def col(field):
        return np.asarray(field).reshape(grid.nz, grid.nx, grid.ny)[:, 1, 1]

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.35))
    for ax, omega in zip(axes, (0.5, 0.9)):
        kappa_e = np.full(grid.n_cells, KAPPA)
        kappa_s = omega * kappa_e
        B = np.zeros(grid.n_cells)
        J_ref = d1o_J(KAPPA, omega, HZ, NZ)
        J_ts = two_stream_J(KAPPA, omega, grid.box[2][0], grid.box[2][1], grid.z)
        res = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, g=0.0,
                               periodic_box=box, n_iter=2000, tol=1e-10)
        J = col(res["J"])
        ax.plot(tau_from_top, J_ref, "k-", lw=1.4, label="1D DO, 64 $\\mu$")
        ax.plot(tau_from_top, J_ts, "--", color="C2", lw=1.0, label="two-stream")
        ax.plot(tau_from_top, J, "o", ms=3.6, mfc="none", color="C0",
                label="rt3d SC, 42 ord.")
        ax.set_xlabel(r"$\tau$ from top")
        ax.set_ylabel("$J$")
        ax.set_title(rf"isotropic ($g=0$), $\omega_s$={omega}, $\tau$={TAU:g}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$J$")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_slab.pdf")


# =============================================================================
# Fig 8 — Ng acceleration: iteration histories
# =============================================================================

def fig_ng():
    from rt3d.cases.slab_1d import slab_grid
    from rt3d.directions import icosahedric_grid
    from rt3d.solver import lambda_iteration
    from rt3d.transport import boundary_conditions

    KAPPA, TAU, NZ = 1.0, 5.0, 50
    grid, box, HZ = slab_grid(TAU, KAPPA, NZ)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    dg = icosahedric_grid(1)
    kappa_e = np.full(grid.n_cells, KAPPA)
    omega = 0.99
    kappa_s = omega * kappa_e
    B = np.zeros(grid.n_cells)

    def hist(ng):
        def run():
            r = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, g=0.0,
                                 periodic_box=box, n_iter=3000, tol=1e-10,
                                 ng=ng)
            return {"h": np.asarray(r["history"]), "n": np.asarray(r["n_iter"])}
        return cached_npz(f"ng_hist_{'ng' if ng else 'li'}", run)

    li, ng = hist(False), hist(True)
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    ax.semilogy(np.arange(1, li["h"].size + 1), li["h"], label=f"lambda iter. ({int(li['n'])} iters)")
    ax.semilogy(np.arange(1, ng["h"].size + 1), ng["h"], label=f"Ng accelerated ({int(ng['n'])} iters)")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"rel.\ source change $\| S^{(k)}-S^{(k-1)}\|_\infty$")
    ax.set_title(rf"slab $\tau$=5, $\omega_s$={omega}, $g$=0")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig_ng.pdf")


# =============================================================================
# Fig 9 — matrix case: J maps and relative error maps vs the REF solve
# =============================================================================

def fig_jmaps():
    from matrix_cases import cloud_octree, cloud_unigrid, direction_grid
    from matrix_stats import sample_at
    from rt3d.solver import lambda_iteration

    from matrix_cases import cloud_beam, cloud_case, cloud_fields

    BOX = cloud_case()["box"]
    DG = direction_grid(1)

    def bc_ground():
        from rt3d.geometry import ground_surface
        from rt3d.transport import boundary_conditions
        return boundary_conditions(
            top_I=1.0,
            ground=ground_surface(albedo=0.3, emissivity=1.0, B_g=1.0))

    with np.load(os.path.join("plot/solver_matrix/_solve_cache",
                              "oct-REF_chain.npz"), allow_pickle=False) as z:
        refJ = z["J"]
    ref_grid = cloud_octree(0.5, 5)

    def sc_solve(grid, **kw):
        kappa_e, kappa_s, g, B = cloud_fields(grid)
        return lambda_iteration(grid, DG, kappa_e, kappa_s, B, bc_ground(),
                                g=g, periodic_box=BOX, beam=cloud_beam(),
                                n_iter=400, tol=1e-8, **kw)

    def solve(vid):
        grid = (cloud_unigrid(24) if vid == "u24/chain" else
                cloud_unigrid(24) if vid == "u24/sc" else cloud_octree(2.0, 3))
        kw = dict(transport="sc") if vid == "u24/sc" else {}
        return {"J": sc_solve(grid, **kw)["J"]}

    vids = ["u24/chain", "u24/sc", "oct-d3"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.0),
                             constrained_layout=True)
    m0 = poly_yz(axes[0], ref_grid, refJ, "viridis")
    axes[0].set_title("REF $J$ (50 380 leaves)")
    cb = plt.colorbar(m0, ax=axes[0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    for ax, vid in zip(axes[1:], vids):
        grid = (cloud_unigrid(24) if vid.startswith("u")
                else cloud_octree(2.0, 3))
        J = solve(vid)["J"]
        Jref = sample_at(grid, ref_grid, refJ)
        err = (J - Jref) / np.maximum(np.abs(Jref), 1e-3)
        m = poly_yz(ax, grid, err, "coolwarm", vmin=-0.15, vmax=0.15)
        ax.set_title(vid + "  rel. $J$ err")
        cb = plt.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5)
    for ax in axes:
        ax.set_xlabel("y [m]")
    axes[0].set_ylabel("z [m]")
    save(fig, "fig_jmaps.pdf")


# =============================================================================
# Fig 10 — ranking bar chart (rms J error vs REF, from ranking.csv)
# =============================================================================

def fig_ranking():
    rows = read_csv(RANKING_CSV)
    rows = [r for r in rows if r["id"] not in
            ("oct-REF/chain", "u24/chain/jit-false", "u24/chain/njobs1")]
    ids = [r["id"].replace("/chain", "") for r in rows]
    vals = [float(r["J_rms_err"]) for r in rows]
    wall = [float(r["wall_s"]) for r in rows]
    order = np.argsort(vals)
    ids = [ids[i] for i in order]
    vals = [vals[i] for i in order]
    wall = [wall[i] for i in order]
    colors = ["tab:red" if vid.startswith("u") else "tab:blue"
              for vid in ids]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ypos = np.arange(len(ids))
    ax.barh(ypos, vals, color=colors, height=0.72, alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(4.1e-2, color="crimson", lw=0.9, ls=":")
    ax.text(4.4e-2, 0.15, "F11 plateau\n$\\approx4\\%$ rms", fontsize=6.5,
            color="crimson")
    for y, v, wl in zip(ypos, vals, wall):
        ax.text(max(v, 1e-9) * 1.15, y, f"{wl:.1f} s", va="center", fontsize=6,
                color="0.35")
    ax.set_yticks(ypos, ids, fontsize=6.8)
    ax.set_xlabel("rms $J$ error vs REF")
    ax.set_title("variant ranking (red: unigrid, blue: octree)")
    save(fig, "fig_ranking.pdf")


# =============================================================================
# Fig 11 — refinement ladders + MC convergence (from convergence.csv)
# =============================================================================

def fig_ladders():
    rows = read_csv(CONV_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.15))

    ax = axes[0]
    cauch = [(12 * 2 ** i, float(r["err"]))
             for i, r in enumerate(r for r in rows
                                   if r["kind"] == "Cauchy pair")]
    slope = next(float(r["err"]) for r in rows
                 if r["kind"] == "fitted slope (Cauchy)")
    xs, ys = zip(*cauch)
    ax.loglog(xs, ys, "o-", label=f"Cauchy pairs (slope {slope:.2f})")
    e0 = ys[0]
    ax.loglog(xs, [e0 * x / 12 for x in (12, 48)], "--", color="0.4",
              label="first order")
    ax.set_xticks([12, 24])
    ax.set_xticklabels(["12", "24"])
    ax.minorticks_off()
    ax.set_xlabel("unigrid resolution $n$")
    ax.set_ylabel("pairwise rms $J$")
    ax.set_title("unigrid chain ladder")
    ax.legend(loc="lower right", fontsize=6.5)

    ax = axes[1]
    tau_rows = sorted((float(r["rung_a"].replace("tau", "")), float(r["err"]))
                      for r in rows if r["kind"] == "tau rung vs REF")
    xs, ys = zip(*tau_rows)
    ax.loglog(xs, ys, "s-", label="octree rungs vs REF")
    ax.loglog(xs, [ys[0] * (x / xs[0]) ** 0.6 for x in xs], "--", color="0.4",
              label="fitted slope 0.6")
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel(r"$\tau_{\max}$ rung (depth 5 fixed)")
    ax.set_ylabel("rms $J$ error vs REF")
    ax.set_title("octree ladder (F10: saturates)")
    ax.legend(loc="upper left", fontsize=6.5)

    ax = axes[2]
    mc_rows = sorted((float(r["rung_a"].split("=")[1]), float(r["err"]))
                     for r in rows if r["kind"] == "power seed-std rel err")
    slope_mc = next(float(r["err"]) for r in rows
                    if r["kind"] == "fitted slope (power err vs N)")
    xs, ys = zip(*mc_rows)
    ax.loglog(xs, ys, "D-", label=f"seed-split power err (slope {slope_mc:.2f})")
    ax.loglog(xs, [ys[0] * (x / xs[0]) ** -0.5 for x in xs], "--", color="0.4",
              label="$N^{-1/2}$")
    ax.set_xticks([2e4, 8e4, 3.2e5])
    ax.set_xticklabels([r"$2{\cdot}10^4$", r"$8{\cdot}10^4$", r"$3.2{\cdot}10^5$"])
    ax.minorticks_off()
    ax.set_xlabel("histories per population $N$")
    ax.set_ylabel("rel. power error")
    ax.set_title("MC convergence (A1 explicit)")
    ax.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout(w_pad=0.9)
    save(fig, "fig_ladders.pdf")


# =============================================================================
# Fig 12 — energy-closure residuals (invariant 4, from matrix.csv)
# =============================================================================

def fig_closure():
    rows = read_csv(MATRIX_CSV)
    pts = [(int(r["n_cells"]), float(r["closure_rel"]), r["id"])
           for r in rows
           if r["family"] == "SC" and r["closure_rel"] not in ("", None)
           and "jit" not in r["id"] and "njobs" not in r["id"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    offs = {"oct-REF/chain": (6, 7), "oct-REF/ng-off": (2, 8),
            "oct-REF/dM2": (2, -14), "oct-d5(tau1)/chain": (4, -13),
            "oct-d3(tau2)/chain": (6, -14),
            "u12/chain": (-6, 7), "u24/chain": (4, -13),
            "u48/chain": (-18, 8), "u24/sc": (-16, 7)}
    for x, y, vid in pts:
        ax.plot(x, max(abs(y), 1e-6), "o" if y > 0 else "s", ms=5,
                color="tab:blue" if y > 0 else "tab:red")
        ax.annotate(vid.replace("/chain", ""), (x, max(abs(y), 1e-6)),
                    fontsize=6,
                    xytext=offs.get(vid, (3, 3)), textcoords="offset points")
    ax.axhline(1e-6, color="0.5", lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells")
    ax.set_ylabel(r"residual $|\oint F\cdot dA-\int 4\pi\kappa_a B\,dV|$"
                  " (abs.)")
    ax.set_title(r"invariant 4 (closure): blue $\geq 0$, red $<0$" "\n"
                 "(F7: face extrapolation dominated)")
    save(fig, "fig_closure.pdf")


# =============================================================================
# Fig 13 — MC vs SC absorbed power with batch sigma (re-solves, cached)
# =============================================================================

def fig_mcpower():
    from matrix_cases import (cloud_beam, cloud_case, cloud_fields,
                              cloud_octree, direction_grid)
    from rt3d.diagnostics import cell_volumes
    from rt3d.mc.forward import _beam_j_analytic, j_mc
    from rt3d.solver import lambda_iteration

    BOX = cloud_case()["box"]
    oct2 = cloud_octree(2.0, 2)
    ke2, ks2, g2, B2 = cloud_fields(oct2)
    vol2 = cell_volumes(oct2)
    w2 = 4 * np.pi * (ke2 - ks2) * vol2
    beam = cloud_beam()

    def bc_ground():
        from rt3d.geometry import ground_surface
        from rt3d.transport import boundary_conditions
        return boundary_conditions(
            top_I=1.0,
            ground=ground_surface(albedo=0.3, emissivity=1.0, B_g=1.0))

    bc = bc_ground()
    res2 = lambda_iteration(oct2, direction_grid(1), ke2, ks2, B2, bc, g=g2,
                            periodic_box=BOX, beam=beam, n_iter=400, tol=1e-8)
    J_sc2 = res2["J"] + _beam_j_analytic(oct2, beam["omega_sun"], ke2,
                                         beam["F0"], periodic_box=BOX)
    P_sc = float(np.sum(w2 * J_sc2))

    def mc_run(tag, **kw):
        def run():
            r = j_mc(oct2, ke2, ks2, B2, bc, g=g2, beam=beam, periodic_box=BOX,
                     stage1="explicit", n_histories=0,
                     n_boundary_histories=120_000, n_beam_histories=120_000,
                     n_ground_histories=120_000, n_batches=300, seed=101,
                     jit=True, **kw)
            J = r["J_total"]
            return {"P": np.asarray(float(np.sum(w2 * J))),
                    "sP": np.asarray(float(np.sqrt(np.sum((w2 * r["sigma_J_total"]) ** 2)))),
                    "Jmean": np.asarray(float(np.mean(J)))}
        return cached_npz("mc_power_" + tag, run)

    loc = mc_run("local")
    glob = mc_run("global", mode="global")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    labels = ["SC oct2\n(reference)", "A1 explicit\n(local max)",
              "A1 explicit\n(global max)", "A1 lambda\n(see text)"]
    vals = [P_sc, float(loc["P"]), float(glob["P"]), 230.53]
    errs = [0.0, float(loc["sP"]), float(glob["sP"]), np.nan]
    xpos = np.arange(4)
    ax.bar(xpos, [v - P_sc for v in vals], color=["0.6", "C0", "C1", "C3"],
           yerr=errs, capsize=3, alpha=0.85)
    ax.set_xticks(xpos, labels, fontsize=6.5)
    ax.axhline(-0.0215 * P_sc, color="crimson", ls=":", lw=1.0)
    ax.text(2.35, -0.0215 * P_sc, "T3 §4 all-sky band\n$-2.15\\%$ vs SC",
            fontsize=6, color="crimson", ha="right")
    ax.set_ylabel("absorbed power $-$ SC (same grid)")
    ax.set_title("MC stage-1 vs SC; bars = batch $\\sigma$ (F9 caveat)")
    save(fig, "fig_mcpower.pdf")


# =============================================================================
# Fig 14 — F3 noise story: masked vs unmasked next-flight z (re-solve, cached)
# =============================================================================

def fig_zhist():
    from matrix_cases import (cloud_beam, cloud_case, cloud_fields,
                              cloud_octree, direction_grid)
    from matrix_stats import (next_flight_mask, ray_optical_depth_octree,
                              z_scores, z_stats)
    from rt3d.mc.ground import ground_beam_irradiance, ground_radiance
    from rt3d.mc.forward import radiance_from_source
    from rt3d.solver import lambda_iteration
    from rt3d.transport import (beam_optical_depth_octree, bottom_mask,
                                boundary_conditions, downward_irradiance)

    DG = direction_grid(1)
    BOX = cloud_case()["box"]
    oct2 = cloud_octree(2.0, 2)
    ke2, ks2, g2, B2 = cloud_fields(oct2)
    beam = cloud_beam()

    def bc_ground():
        from rt3d.geometry import ground_surface
        return boundary_conditions(
            top_I=1.0,
            ground=ground_surface(albedo=0.3, emissivity=1.0, B_g=1.0))

    bc = bc_ground()
    NF = 100

    def run():
        res2 = lambda_iteration(oct2, DG, ke2, ks2, B2, bc, g=g2,
                                periodic_box=BOX, beam=beam, n_iter=400,
                                tol=1e-8)
        mb2 = bottom_mask(oct2)
        F_down2 = downward_irradiance(res2["I"][:, DG.down_idx], DG)[mb2]
        tau_b2 = beam_optical_depth_octree(oct2, beam["omega_sun"], ke2,
                                           periodic_box=BOX)
        Fb2 = ground_beam_irradiance(oct2, beam, ke2, tau_b2, periodic_box=BOX)
        I_g2 = ground_radiance(oct2, bc.ground, F_down2, beam_irr=Fb2)
        det = radiance_from_source(oct2, DG, res2["kappa_e_delta"], res2["S"],
                                   bc, periodic_box=BOX,
                                   estimator="deterministic", ground_I=I_g2)
        nf = radiance_from_source(oct2, DG, estimator="next_flight",
                                  kappa_e=res2["kappa_e_delta"],
                                  S_total=res2["S"], bc=bc, periodic_box=BOX,
                                  ground_I=I_g2, n_histories=NF,
                                  n_batches=25, seed=11, jit=True)
        tau_ray = ray_optical_depth_octree(oct2, DG, res2["kappa_e_delta"], BOX)
        keep = next_flight_mask(nf["sigma_I"], nf["I"], tau_ray,
                                n_histories=NF, min_scoring=0.16 * NF)
        zn = z_scores(nf["I"], nf["sigma_I"], det["I"])
        return {"zn": zn, "keep": keep}

    d = cached_npz("zhist_a2", run)
    zn, keep = d["zn"].ravel(), d["keep"].ravel()
    st_m = z_stats(d["zn"][d["keep"]])
    st_u = z_stats(zn[np.isfinite(zn)])

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2))
    n_m, _, _ = axes[0].hist(np.clip(d["zn"][d["keep"]], -8, 8), bins=80,
                             color="tab:blue", edgecolor="k", linewidth=0.2)
    axes[0].set_title(f"masked (F3 noise mask): rms $z$={st_m['rms']:.2f}, "
                      r"$\max|z|$" + f"={st_m['max_abs']:.1f}")
    axes[0].set_xlabel("$z$")
    n_u, _, _ = axes[1].hist(np.clip(zn[np.isfinite(zn)], -8, 8), bins=80,
                             color="tab:red", edgecolor="k", linewidth=0.2)
    axes[1].set_title(f"unmasked: rms $z$={st_u['rms']:.1f}, "
                      r"$\max|z|$" + f"={st_u['max_abs']:.0f}")
    axes[1].set_xlabel("$z$")
    axes[1].annotate("central spike $=$ never-scoring cells ($\\sigma\\to0$);\n"
                     r"outliers beyond $\pm8$ clipped",
                     xy=(7.9, n_u.max() * 0.85), fontsize=6, ha="right",
                     color="0.25")
    for ax in axes:
        ax.set_ylabel("count")
    fig.suptitle(f"A2 next-flight per-ordinate $z$ ({NF} flights, "
                 "source = SC converged)", fontsize=8.5, y=1.04)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_zhist.pdf")
    print(f"  z stats masked {st_m['rms']:.3f} / unmasked {st_u['rms']:.3f}")


# =============================================================================
# Fig 15 — MC stage-1 slab vs 1D reference (small re-runs, cached)
# =============================================================================

def fig_mc_slab():
    from rt3d.cases.slab_1d import d1o_J, slab_grid, two_stream_J
    from rt3d.directions import icosahedric_grid
    from rt3d.mc.forward import j_mc
    from rt3d.solver import lambda_iteration
    from rt3d.transport import boundary_conditions

    KAPPA, TAU, NZ = 1.0, 5.0, 50
    N_MC, N_BATCH = 80_000, 200
    grid, box, HZ = slab_grid(TAU, KAPPA, NZ)
    bc = boundary_conditions(top_I=1.0, bottom_I=0.0)
    dg = icosahedric_grid(1)
    tau_from_top = TAU - KAPPA * grid.z

    def col(field):
        a = np.asarray(field)
        return a.reshape(grid.nz, grid.nx, grid.ny)[:, 1, 1]

    def run(omega):
        kappa_e = np.full(grid.n_cells, KAPPA)
        kappa_s = omega * kappa_e
        B = np.zeros(grid.n_cells)
        sc = lambda_iteration(grid, dg, kappa_e, kappa_s, B, bc, g=0.0,
                              periodic_box=box, n_iter=2000, tol=1e-10)
        r = j_mc(grid, kappa_e, kappa_s, B, bc, seed=103, stage1="explicit",
                 g=0.0, periodic_box=box, n_histories=N_MC,
                 n_boundary_histories=N_MC, n_batches=N_BATCH)
        return {"J_sc": col(sc["J"]), "J": col(r["J"]), "sJ": col(r["sigma_J"])}

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.35))
    for ax, omega in zip(axes, (0.5, 0.9)):
        kappa_e = np.full(grid.n_cells, KAPPA)
        kappa_s = omega * kappa_e
        J_ref = d1o_J(KAPPA, omega, HZ, NZ)
        J_ts = two_stream_J(KAPPA, omega, grid.box[2][0], grid.box[2][1], grid.z)
        d = cached_npz(f"mc_slab_{omega}", lambda: run(omega))
        ax.plot(tau_from_top, J_ref, "k-", lw=1.4, label="1D DO, 64 $\\mu$")
        ax.plot(tau_from_top, J_ts, "--", color="C2", lw=1.0, label="two-stream")
        ax.plot(tau_from_top, d["J_sc"], "x", ms=3.6, color="C3",
                label="SC, 42 ord.")
        J, s = d["J"], d["sJ"]
        ax.plot(tau_from_top, J, "o", ms=3.6, mfc="none", color="C0",
                label="MC explicit")
        ax.fill_between(tau_from_top, J - 2 * s, J + 2 * s, color="C0",
                        alpha=0.15)
        ax.set_xlabel(r"$\tau$ from top")
        ax.set_title(rf"$g=0$, $\omega_s$={omega}, $\tau$={TAU:g}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$J$")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_mc_slab.pdf")


# =============================================================================
# Fig 16 — delta-M effect on the octree case (compare_transport cache)
# =============================================================================

def fig_deltaM():
    from matrix_cases import cloud_octree
    from matrix_stats import sample_at

    oct3 = cloud_octree(2.0, 3)
    oct5 = cloud_octree(0.5, 5)
    with np.load(os.path.join(CT_CACHE, "octree_dM0.npz")) as z0:
        J0 = z0["J"]
    with np.load(os.path.join(CT_CACHE, "octree_dM1.npz")) as z1:
        J1 = z1["J"]
    with np.load(os.path.join(CT_CACHE, "octree_ref_dM0.npz")) as zr:
        Jr = zr["J"]
    Jr3 = sample_at(oct3, oct5, Jr)
    dJ = J1 - J0

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.2))
    md = poly_yz(axes[0], oct3, dJ, "coolwarm", vmin=-0.02, vmax=0.02,
                 halfw=0.625)
    axes[0].set_title(r"$J^{\delta M1}-J^{\delta M0}$ (same grid)")
    plt.colorbar(md, ax=axes[0], fraction=0.046, pad=0.02)

    zc = oct3.centers[:, 2]
    hb = oct3.half
    band = np.clip(((zc - 0.0) / 1.25).astype(int), 0, 7)
    prof = []
    for J in (J0, J1, Jr3):
        p = np.zeros(8)
        cnt = np.zeros(8)
        w = (2 * hb) ** 2
        np.add.at(p, band, J * w)
        np.add.at(cnt, band, w)
        prof.append(p / np.maximum(cnt, 1e-30))
    zb = np.arange(8) * 1.25 + 0.625
    axes[1].plot(zb, prof[2], "k-", lw=1.4, label="REF (642 ord.)")
    axes[1].plot(zb, prof[0], "o", ms=3.5, color="C0", label=r"$\delta$-M 0")
    axes[1].plot(zb, prof[1], "s", ms=3.5, mfc="none", color="C1",
                 label=r"$\delta$-M 1")
    axes[1].set_xlabel("z [m] (1.25 m bands)")
    axes[1].set_ylabel("band-mean $J$")
    axes[1].set_title(r"band-mean profiles: $\delta$-M 0 vs 1 vs REF")
    axes[1].legend(loc="lower right")
    for ax in axes:
        ax.set_xlabel("y [m]" if ax is axes[0] else "z [m]")
    axes[0].set_ylabel("z [m]")
    fig.tight_layout(w_pad=0.9)
    save(fig, "fig_deltaM.pdf")


# =============================================================================
# Fig 17 — TOA upward radiance spectra (transport comparison, cached CSVs)
# =============================================================================

def fig_toa():
    rows = read_csv(os.path.join(CT_FIG, "toa_mean_radiance.csv"))
    rows642 = read_csv(os.path.join(CT_FIG, "toa_mean_radiance_642.csv"))
    mu = np.array([float(r["mu"]) for r in rows])
    mu642 = np.array([float(r["mu"]) for r in rows642])
    I642 = np.array([float(r["octree_ref_dM0"]) for r in rows642])
    o642 = np.argsort(mu642)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(mu642[o642], I642[o642], color="0.35", lw=0.8, zorder=2,
            label="REF octree, 642 ord.")
    for colr, mk, colr_ in (("uni_chain_dM0", "o", "tab:blue"),
                            ("uni_sc_dM0", "^", "darkorange"),
                            ("octree_dM0", "s", "tab:green")):
        I = np.array([float(r[colr]) for r in rows])
        label = {"uni_chain_dM0": "unigrid chain", "uni_sc_dM0": "unigrid SC",
                 "octree_dM0": "octree"}[colr] + " (42 ord.)"
        ax.scatter(mu, I, marker=mk, s=16, color=colr_, zorder=3,
                   label=label)
    ax.set_xlabel(r"$\mu=\Omega_z$ (outgoing)")
    ax.set_ylabel(r"domain-mean upward $I$ at TOA")
    ax.set_title("TOA spectrum, all-sky cloud case ($\\delta$-M 0)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    save(fig, "fig_toa.pdf")


# =============================================================================
# LaTeX table fragments
# =============================================================================

def tables():
    # matrix.csv -> compact variant table
    rows = read_csv(MATRIX_CSV)
    with open(os.path.join(TAB, "matrix_table.tex"), "w") as fh:
        fh.write(r"""\begin{tabular}{llrrrrrr}
\toprule
variant & grid & cells & iters & wall [s] & $\overline{J}$ &
closure & rms $J$ err vs REF \\
\midrule
""")
        for r in rows:
            vid = r["id"].replace("_", r"\_")
            cells = f"{int(r['n_cells']):,}"
            it = r["n_iter"] if r["n_iter"] else "--"
            wall = r["wall_s"] if r["wall_s"] else "--"
            closure = r["closure_rel"]
            if closure.startswith("n/a"):
                closure = "--"
            err = r["J_err_rms"] if r["J_err_rms"] else "--"
            if err == "0":
                err = "0 (ref)"
            fh.write(f"{vid} & {r['grid'].replace('_', chr(92)+'_')} & {cells} & {it} & {wall} & "
                     f"{r['mean_J']} & {closure} & {err} \\\\\n")
        fh.write(r"""\bottomrule
\end{tabular}
""")

    # mc_stats.csv -> z-statistics table
    rows = read_csv(MCSTATS_CSV)
    with open(os.path.join(TAB, "mcstats_table.tex"), "w") as fh:
        fh.write(r"""\begin{tabular}{llrrrrr}
\toprule
quantity & mask & $n$ & mean $z$ & rms $z$ &
$\{|z|>2\}$ & $\{|z|>3\}$ \\
\midrule
""")
        for r in rows:
            q = r["quantity"].replace("_", r"\_").replace("(", " (")
            fh.write(f"{q} & {r['mask']} & {r['n']} & {r['mean']} & "
                     f"{r['rms']} & {r['tail2']} & {r['tail3']} \\\\\n")
        fh.write(r"""\bottomrule
\end{tabular}
""")

    # capability matrix
    CAP = [
        ("thermal emission $B$", "yes", "yes", "yes", "yes", "yes", "yes"),
        ("solar beam", "yes", "yes", "yes", "yes", "yes", "yes"),
        ("Lambertian ground", "yes", "yes", "yes", "yes (M6)",
         "yes (ground\\_I)", "yes (M6)"),
        ("variable-height ground", r"n/a$^\dagger$", r"n/a$^\dagger$",
         r"n/a$^\dagger$", r"n/a$^\dagger$", r"n/a$^\dagger$", r"n/a$^\dagger$"),
        ("angular quadrature", "42--642", "42--642", "42--642", "none",
         "binning", "exact HG"),
        ("energy closure (invariant 4)", "yes", "yes", "yes (octree)",
         r"n/a ($J$ only)", "yes (stage-2 $I$)", r"n/a (detector)"),
    ]
    with open(os.path.join(TAB, "capability_table.tex"), "w") as fh:
        fh.write(r"""\begin{tabular}{@{}llcccccc@{}}
\toprule
feature & \multicolumn{3}{c}{SC family} & \multicolumn{3}{c}{MC family} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
 & chain & unigrid SC & octree & A1 stage-1 & A2 stage-2 & adjoint B \\
\midrule
""")
        for row in CAP:
            fh.write(" & ".join(row) + r" \\" + "\n")
        fh.write(r"""\bottomrule
\end{tabular}
""")


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    t0 = time.time()
    print("report figures (vector PDF) -> report/figures/")
    fig_directions()
    fig_octree()
    fig_stencil()
    fig_subsegments()
    fig_octree_march()
    fig_beer_f11()
    fig_slab()
    fig_ng()
    fig_jmaps()
    fig_ranking()
    fig_ladders()
    fig_closure()
    fig_mcpower()
    fig_zhist()
    fig_mc_slab()
    fig_deltaM()
    fig_toa()
    tables()
    print(f"done in {time.time() - t0:.0f} s -> {FIG}/, {TAB}/")
