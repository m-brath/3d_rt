# -*- coding: utf-8 -*-
r"""
Schematic figures for docs/sc-transport.md (short-characteristics transport
on the unigrid and the octree).  Pure matplotlib, no solver runs - the
geometry helpers from rt3d are used so the figures show the *actual*
stencils, sub-segments and octree marches the solver uses.

Run from the repo root:
  Windows:      $env:Path = "$env:USERPROFILE\miniforge3\envs\pyarts\Library\bin;" + $env:Path
                & "$env:USERPROFILE\miniforge3\envs\pyarts\python.exe" scripts/sc_transport_figures.py
  macOS/Linux:  ~/miniforge3/envs/pyarts/bin/python scripts/sc_transport_figures.py
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from rt3d.characteristics import stencil_for_direction, subsegments_uniform
from rt3d.geometry import cuboid
from rt3d.octree import march_octree, octree_grid

OUT = "plot"

# %% ===========================================================================
# Fig 1: lattice stencil selection (isotropic + anisotropic panel)
# %% ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

# --- panel A: isotropic lattice, one icosahedral-looking ordinate
ax = axes[0]
h = np.array([1.0, 1.0, 1.0])
omega = np.array([0.526, 0.316, 0.789])
omega = omega / np.linalg.norm(omega)
s, vhat = stencil_for_direction(omega, h, max_offset=3)

R = 3
cands = np.array(
    [t for t in np.ndindex(*([2 * R + 1] * 3)) if any(c != R for c in t)]
) - R
cands = cands[(cands[:, 2] != 0)]
cos = (cands @ omega) / np.linalg.norm(cands * h, axis=1)
sc = ax.scatter(
    cands[:, 0], cands[:, 2], c=cos, cmap="viridis", vmin=-1, vmax=1, s=28, zorder=2
)


def arrow(ax, x, y, color, lw=2.0, z=4):
    ax.annotate(
        "", xy=(x, y), xytext=(0, 0), zorder=z,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=22),
    )


arrow(ax, 3 * omega[0], 3 * omega[2], "crimson")
ax.annotate(
    r"$\Omega$",
    (3 * omega[0], 3 * omega[2]),
    xytext=(3 * omega[0] - 1.1, 3 * omega[2] + 0.25),
    color="crimson",
    fontsize=12,
)
arrow(ax, s[0], s[2], "tab:blue")
ax.scatter(
    [s[0]], [s[2]], facecolors="none", edgecolors="tab:blue", s=260, lw=2, zorder=5
)
ax.annotate(
    f"chosen $s\\cdot h$ = ({s[0]}, {s[2]})",
    (s[0], s[2]),
    xytext=(s[0] - 2.8, s[2] - 0.7),
    color="tab:blue",
    fontsize=12,
)
ax.plot([0], [0], "k*", ms=14, zorder=5)
ax.set_title("isotropic lattice: candidates colored by $\\cos\\angle(s\\cdot h,\\Omega)$")
ax.set_xlabel("x [cells]")
ax.set_ylabel("z [cells]")

# --- panel B: anisotropic lattice hz = hx/4, ray with dz/dx = 3
ax = axes[1]
h2 = np.array([1.0, 1.0, 0.25])
omega2 = np.array([1.0, 0.0, 3.0])
omega2 = omega2 / np.linalg.norm(omega2)
s2, vhat2 = stencil_for_direction(omega2, h2)  # automatic radius

sx = np.arange(0, 5)
sz = np.arange(1, 15)
GX, GZ = np.meshgrid(sx, sz)
pts = np.column_stack([GX.ravel() * h2[0], GZ.ravel() * h2[2]])
v = np.column_stack([GX.ravel() * h2[0], GZ.ravel() * h2[2]])
cos2 = (v[:, 0] * omega2[0] + v[:, 1] * omega2[2]) / np.linalg.norm(v, axis=1)
ax.scatter(pts[:, 0], pts[:, 1], c=cos2, cmap="viridis", vmin=-1, vmax=1, s=26, zorder=2)
# chosen stencil s2=(1,12): physical direction (1, 12*0.25) = (1, 3) - exactly
# along Omega, so the two arrows coincide (blue under, crimson on top)
arrow(ax, s2[0] * h2[0], s2[2] * h2[2], "tab:blue", lw=4.0)
arrow(ax, omega2[0] * 3.3, omega2[2] * 3.3, "crimson", lw=1.5, z=5)
ax.scatter(
    [s2[0] * h2[0]], [s2[2] * h2[2]],
    facecolors="none", edgecolors="tab:blue", s=260, lw=2, zorder=6,
)
ax.annotate(
    r"$\Omega \propto (1,3)$  =  chosen $s=(1,12)$ direction",
    (1.0, 3.0),
    xytext=(1.3, 2.6),
    color="crimson",
    fontsize=11,
)
# a naive short stencil that would be badly misaligned
arrow(ax, 1 * h2[0], 3 * h2[2], "0.5", lw=1.5, z=3)
ax.annotate(
    "$s=(1,3)$: direction\n$(1, 0.75)$ — rejected",
    (1.0, 0.75),
    xytext=(1.3, 1.1),
    color="0.35",
    fontsize=10,
)
ax.plot([0], [0], "k*", ms=14, zorder=5)
ax.set_title("anisotropic $h_z=h_x/4$: radius must scale with $\\max(h)/\\min(h)$")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")

fig.tight_layout()
fig.savefig(f"{OUT}/sc_stencil_lattice.png", dpi=150)
plt.close(fig)

# %% ===========================================================================
# Fig 2: sub-segment splitting of one characteristic (unigrid)
# %% ===========================================================================
fig, ax = plt.subplots(figsize=(7.2, 6.4))
h = np.array([1.0, 1.0, 1.0])
s = np.array([2, 0, 3])
rel, lengths = subsegments_uniform(s, h)

start = -s
# traversed cells: start + rel (parent -> target), plus the start cell itself
cells = np.vstack([start, start + rel])
for idx, (ci, cj, ck) in enumerate(cells):
    ax.add_patch(
        plt.Rectangle(
            (ci - 0.5, ck - 0.5), 1, 1,
            facecolor="0.92" if idx == 0 else "none",
            edgecolor="0.6", lw=1.0, zorder=1,
        )
    )
for m, (ri, rj, rk) in enumerate(rel):
    ax.add_patch(
        plt.Rectangle(
            (start[0] + ri - 0.5, start[2] + rk - 0.5), 1, 1,
            facecolor=plt.cm.viridis(m / len(rel)),
            edgecolor="0.3", lw=1.0, alpha=0.55, zorder=2,
        )
    )

p0 = np.array([start[0], start[2]], float)
p1 = np.array([0.0, 0.0])
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", lw=2, zorder=4)

# segment boundary points along the path (cumulative lengths / |s*h| fraction)
frac = np.concatenate([[0], np.cumsum(lengths) / lengths.sum()])
pts = p0[None, :] + frac[:, None] * (p1 - p0)[None, :]
ax.plot(pts[:, 0], pts[:, 1], "o", ms=6, mfc="white", mec="k", zorder=5)
ax.plot(*p0, "ks", ms=9, zorder=6)
ax.plot(*p1, "k^", ms=10, zorder=6)
ax.annotate("$u$ center (cell $-s$)", p0, xytext=(p0[0] - 1.35, p0[1] - 0.55), fontsize=11)
ax.annotate("$c$ center (target)", p1, xytext=(0.25, -0.45), fontsize=11)

for m in range(len(rel)):
    mid = 0.5 * (pts[m] + pts[m + 1])
    ax.annotate(
        f"{m}: $\\ell$={lengths[m]:.3f}",
        mid,
        xytext=(mid[0] - 1.55, mid[1] + 0.28),
        fontsize=9,
        color="0.2",
    )
ax.set_xlim(-3.4, 1.2)
ax.set_ylim(-4.2, 1.4)
ax.set_aspect("equal")
ax.set_title(
    f"sub-segments of $u_{{center}}\\to c_{{center}}$,  $s=({s[0]},{s[1]},{s[2]})$,"
    f"  $M$={len(rel)}"
)
ax.set_xlabel("x [cells]")
ax.set_ylabel("z [cells]")
fig.tight_layout()
fig.savefig(f"{OUT}/sc_subsegments.png", dpi=150)
plt.close(fig)

# %% ===========================================================================
# Fig 3: topological BFS waves (non-periodic depth = min over stencil axes)
# %% ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 6.2))
s = np.array([2, 1, 1])
nx, ny, k_slice = 12, 12, 3
depth = np.minimum(np.minimum(np.arange(nx)[:, None], np.arange(ny)[None, :]), k_slice)
ax.imshow(
    depth.T,
    origin="lower",
    cmap="tab10",
    vmin=0,
    extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
)
for i in range(nx):
    for j in range(ny):
        ax.text(i, j, str(depth[i, j]), ha="center", va="center", fontsize=8)
# in-plane projection of the parent connection c -> c - s
for i in range(s[0], nx):
    for j in range(s[1], ny):
        if depth[i, j] == k_slice and i >= nx - 3 and j >= ny - 3:
            ax.annotate(
                "",
                xy=(i - s[0], j - s[1]),
                xytext=(i, j),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.6),
                zorder=5,
            )
ax.set_xlabel("i")
ax.set_ylabel("j")
ax.set_title(
    "wave index = chain depth for $s=(2,1,1)$, slice $k=3$ (non-periodic):\n"
    "depth$=\\min(i,j,k)$; red arrows: parent $c-s$ (in-plane projection)"
)
fig.tight_layout()
fig.savefig(f"{OUT}/sc_bfs_waves.png", dpi=150)
plt.close(fig)

# %% ===========================================================================
# Fig 4: exact octree march — one backward ray through mixed-level leaves
# %% ===========================================================================
fig, ax = plt.subplots(figsize=(7.6, 6.6))
box = cuboid(0, 4, 0, 4, 0, 4)
oct = octree_grid(box, h0=1.0)


def base_id(i, j, k):
    return k * 16 + j * 4 + i


oct.split(np.array([base_id(1, 1, 1), base_id(2, 2, 2), base_id(2, 1, 2)]))
oct.balance_21(max_depth=6)
oct.finalize()

y_slice = 2.0
omega = np.array([0.55, 0.25, -0.80])
omega = omega / np.linalg.norm(omega)

# start from the leaf center closest to (3.3, 2.0, 1.3): the backward ray
# then traverses the refined region around (2.2, 2.2) on its way to the top
target = np.array([3.3, 2.0, 1.3])
leaf0 = int(np.argmin(np.linalg.norm(oct.centers - target, axis=1)))
origin = oct.centers[leaf0]
# backward march along -omega up to the top face (entry budget for omega_z < 0)
budget = (box.z1 - origin[2]) / abs(omega[2])
sr, sl, sLen = march_octree(origin[None, :], -omega, oct, np.array([budget]))

# draw leaves cut by the y = y_slice plane
for n in range(oct.n_leaf):
    c, h = oct.centers[n], oct.half[n]
    if abs(c[1] - y_slice) <= h:
        ax.add_patch(
            plt.Rectangle(
                (c[0] - h, c[2] - h), 2 * h, 2 * h,
                facecolor="none", edgecolor="0.55", lw=1.0, zorder=1,
            )
        )

# draw the ray segments, colored by traversal order
M = sLen.size
p = origin.copy()
cmap = plt.cm.plasma
for m in range(M):
    q = p + sLen[m] * (-omega)
    ax.plot(
        [p[0], q[0]], [p[2], q[2]],
        color=cmap(m / max(M - 1, 1)), lw=3.0, zorder=3,
        solid_capstyle="round",
    )
    mid = 0.5 * (p + q)
    ax.text(mid[0], mid[2], str(m), fontsize=8, color="0.15", zorder=4,
            ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    p = q
ax.plot(*origin[[0, 2]], "k*", ms=15, zorder=5)
ax.annotate("leaf center (start)", origin[[0, 2]], xytext=(origin[0] + 0.15, origin[2] + 0.25))
ax.plot(*p[[0, 2]], "kv", ms=11, zorder=5)
ax.annotate("domain entry", p[[0, 2]], xytext=(p[0] - 1.5, p[2] - 0.35))
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_aspect("equal")
ax.set_title(
    "one exact backward march from a leaf center along $-\\Omega$\n"
    "through the octree (x–z slice at y=2, segments in traversal order)"
)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()
fig.savefig(f"{OUT}/sc_octree_march.png", dpi=150)
plt.close(fig)

# %% ===========================================================================
# Fig 5: ray-contiguous block layout and _block_prefixes cumsum trick
# %% ===========================================================================
fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.axis("off")

# toy example: 3 rays with 4, 2, 3 segments
lengths = np.array([0.7, 1.3, 0.4, 0.9, 1.1, 0.6, 0.5, 1.4, 0.8])
counts = np.array([4, 2, 3])
offsets = np.concatenate([[0], np.cumsum(counts)])
tau = lengths.copy()  # pretend kappa_e = 1
ray_colors = ["tab:blue", "tab:orange", "tab:green"]

x0, w, gap = 1.9, 0.95, 0.12
y_seg = 3.4
for m in range(lengths.size):
    ray = int(np.searchsorted(offsets, m, side="right") - 1)
    x = x0 + m * (w + gap)
    bw = 0.6 * w * lengths[m]  # box width proportional to segment length
    ax.add_patch(
        plt.Rectangle((x, y_seg), bw, 0.7,
                      facecolor=ray_colors[ray], alpha=0.75, ec="k")
    )
    ax.text(x + 0.5 * bw, y_seg + 0.35, f"{lengths[m]:.1f}",
            ha="center", va="center", fontsize=8)
    ax.text(x + 0.5 * w, y_seg - 0.25, f"seg {m}",
            ha="center", va="top", fontsize=8, color="0.3")

# ray blocks
for r in range(counts.size):
    xa = x0 + offsets[r] * (w + gap)
    xb = x0 + (offsets[r + 1] * (w + gap) - gap)
    ax.add_patch(
        plt.Rectangle((xa - 0.06, y_seg - 0.06), xb - xa + 0.12, 0.82,
                      fill=False, ec=ray_colors[r], lw=2)
    )
    ax.text(0.5 * (xa + xb), y_seg + 1.05, f"ray {r} (leaf {r})",
            ha="center", color=ray_colors[r], fontsize=11)
    ax.text(0.5 * (xa + xb), y_seg - 0.55, f"offsets {offsets[r]}..{offsets[r+1]}",
            ha="center", fontsize=9, color="0.35")

# global cumsum row
cs = np.cumsum(tau)
y_cs = 1.9
for m in range(lengths.size):
    x = x0 + m * (w + gap)
    ax.text(x + 0.5 * w, y_cs, f"{cs[m]:.2f}",
            ha="center", fontsize=9, color="0.1")
ax.text(0.05, y_cs, "$C=\\mathrm{cumsum}(\\tau_{seg})$", fontsize=11, va="center")

# per-block corrected prefixes: prev = C[offset-1] for offset > 0, else 0
y_pre = 1.0
prev = np.zeros(counts.size)
for r in range(counts.size):
    if offsets[r] > 0:
        prev[r] = cs[offsets[r] - 1]
prev_rep = np.repeat(prev, counts)
pre_before = cs - tau - prev_rep
pre_thru = cs - prev_rep
for m in range(lengths.size):
    x = x0 + m * (w + gap)
    ax.text(x + 0.5 * w, y_pre, f"{pre_before[m]:.2f}",
            ha="center", fontsize=9, color="crimson")
ax.text(0.05, y_pre, "prefix_before", fontsize=11, va="center", color="crimson")

ax.annotate("", xy=(x0 - 0.3, y_seg - 0.1), xytext=(x0 - 0.3, y_pre + 0.5),
            arrowprops=dict(arrowstyle="->", color="0.3"))
ax.text(
    x0 + 0.5 * lengths.size * (w + gap), 0.55,
    "$\\tau_{tot}$ per leaf: $\\mathtt{np.add.at}$ over ray_of    "
    "attenuation of a segment to the leaf center: $e^{-\\text{prefix\\_before}}$",
    fontsize=10, color="0.25", ha="center", va="center",
)
ax.set_xlim(-0.1, x0 + lengths.size * (w + gap) + 0.3)
ax.set_ylim(0.2, 4.8)
ax.set_title(
    "ray-contiguous segment blocks (center $\\to$ entry order) and the "
    "single-cumsum block-prefix correction"
)
fig.tight_layout()
fig.savefig(f"{OUT}/sc_block_prefixes.png", dpi=150)
plt.close(fig)

print("wrote sc_stencil_lattice.png, sc_subsegments.png, sc_bfs_waves.png, "
      "sc_octree_march.png, sc_block_prefixes.png to plot/")
