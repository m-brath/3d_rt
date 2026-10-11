# -*- coding: utf-8 -*-
"""
Phase 2 validation: scattering + plain lambda iteration on a 1D slab.

Case: plane-parallel slab (periodic x,y), isotropic scattering g = 0,
no thermal emission (B = 0), isotropic incident intensity I = 1 at the top,
vacuum at the bottom. omega = 0.5 and 0.9, tau_total = 5.

References:
  1. Independent 1D discrete-ordinates solver on the SAME z grid but with
     64 Gauss-Legendre mu levels (angular error ~ negligible): the strong
     reference for the 42-ordinate icosahedral result.
  2. Two-stream approximation (Eddington closure, Marshak BCs), analytic:
     context curve only.

The expected deviation of the 42-ordinate solution from reference 1 is the
coarse-SN angular error the plan documents (section 9); it must be a few
percent, not a structural offset.
"""

# %% imports and case setup
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import roots_legendre

from rt3d.directions import icosahedric_grid
from rt3d.geometry import cuboid, unigrid
from rt3d.solver import lambda_iteration
from rt3d.transport import boundary_conditions

KAPPA = 1.0  # extinction, uniform
TAU = 5.0  # slab optical thickness
NZ = 50  # z cells (hz = 0.25 -> cubic cells, hx = hy = hz)
HZ = TAU / KAPPA / NZ

# cubic cells: horizontal footprint 4 x 4 cells of size HZ, periodic
hx = HZ
x = (np.arange(4) + 0.5) * hx
y = (np.arange(4) + 0.5) * hx
z = (np.arange(NZ) + 0.5) * HZ
grid = unigrid(x, y, z)
box = cuboid(0.0, 4 * hx, 0.0, 4 * hx, grid.box[2][0], grid.box[2][1])

z_prof = z  # (NZ,) cell-center profile
tau_coord = KAPPA * z_prof  # optical depth coordinate from the bottom


# %% reference 1: 1D discrete ordinates, 64 Gauss-Legendre mu levels
def slab_d1o(nz, kappa, omega, hz, mu_levels=64, n_iter=4000, tol=1e-12):
    """
    Plane-parallel slab, z upward, cells k = 0 (bottom) .. nz-1 (top).
    Isotropic incident I = 1 at the top face, vacuum at the bottom.
    Isotropic scattering, B = 0. Plain lambda iteration.
    Returns J(k).
    """
    mu, w = roots_legendre(mu_levels)  # on (-1,1); use positive half
    mu, w = mu[mu > 0], w[mu > 0]  # J = 1/2 sum w (I_up + I_dn), sum w = 1

    S = np.zeros(nz)
    h_half = hz / 2.0
    for _ in range(n_iter):
        J = np.zeros(nz)
        for m, (mu_m, w_m) in enumerate(zip(mu, w)):
            tau_h = kappa * h_half / mu_m  # half-cell optical depth
            # downward: enters at top face with I = 1
            I = np.zeros(nz)
            t_prev = np.exp(-tau_h)  # top face -> center of cell nz-1
            I[nz - 1] = 1.0 * t_prev + S[nz - 1] * (1 - t_prev)
            for k in range(nz - 2, -1, -1):
                # center k+1 -> center k: half in cell k+1, half in cell k
                e = np.exp(-2 * tau_h)
                I[k] = (
                    I[k + 1] * e
                    + S[k + 1] * (1 - np.exp(-tau_h)) * np.exp(-tau_h)
                    + S[k] * (1 - np.exp(-tau_h))
                )
            J += 0.5 * w_m * I
            # upward: enters at bottom face with I = 0
            I = np.zeros(nz)
            t_prev = np.exp(-tau_h)
            I[0] = S[0] * (1 - t_prev)
            for k in range(1, nz):
                e = np.exp(-2 * tau_h)
                I[k] = (
                    I[k - 1] * e
                    + S[k - 1] * (1 - np.exp(-tau_h)) * np.exp(-tau_h)
                    + S[k] * (1 - np.exp(-tau_h))
                )
            J += 0.5 * w_m * I
        S_new = omega * J
        if np.max(np.abs(S_new - S)) < tol:
            S = S_new
            break
        S = S_new
    return J  # mean intensity from the last (converged) sweep


# %% reference 2: two-stream (Eddington closure, Marshak BCs), analytic
def slab_two_stream(z0, z1, kappa, omega):
    """J(z) for the slab z in [z0, z1] (faces): dH/dz = kappa (1-omega) J,
    dJ/dz = 3 kappa H; Marshak BCs I_down(top) = 1, I_up(bottom) = 0 with
    I+- = J +- H. Evaluated at cell centers zc."""
    lam = kappa * np.sqrt(3.0 * (1.0 - omega))
    gam = np.sqrt((1.0 - omega) / 3.0)
    # H = -(1/3kappa) dJ/dz = -gam (A e^{lam z} - C e^{-lam z});
    # bottom face z0: I+ = 0 -> J + H = 0; top face z1: I- = 1 -> J - H = 1
    M = np.array(
        [
            [(1 - gam) * np.exp(lam * z0), (1 + gam) * np.exp(-lam * z0)],
            [(1 + gam) * np.exp(lam * z1), -(1 - gam) * np.exp(-lam * z1)],
        ]
    )
    b = np.array([0.0, 1.0])
    A, C = np.linalg.solve(M, b)
    return A * np.exp(lam * z_prof) + C * np.exp(-lam * z_prof)


# %% run 3D solver for omega = 0.5 and 0.9, compare
dg = icosahedric_grid(n_refine=1)
bc = boundary_conditions(top_I=1.0, bottom_I=0.0)

results = {}
for omega in (0.5, 0.9):
    kappa_e = np.full(grid.n_cells, KAPPA)
    kappa_s = omega * kappa_e
    B = np.zeros(grid.n_cells)
    res = lambda_iteration(
        grid, dg, kappa_e, kappa_s, B, bc, g=0.0,
        periodic_box=box, n_iter=2000, tol=1e-10,
    )
    J3d = res["J"].reshape(grid.nz, grid.nx, grid.ny)[:, 1, 1]  # any column
    J_ref = slab_d1o(NZ, KAPPA, omega, HZ)
    J_ts = slab_two_stream(grid.box[2][0], grid.box[2][1], KAPPA, omega)
    # normalized errors (J spans ~3 orders of magnitude through tau=5;
    # relative error is only meaningful where J_ref is not tiny)
    scale = J_ref.max()
    rel = np.abs(J3d - J_ref).max() / scale
    rel_loc = (np.abs(J3d - J_ref) / J_ref)[J_ref > 1e-2 * scale].max()
    rel_ts = np.abs(J_ts - J_ref).max() / scale
    results[omega] = (J3d, J_ref, J_ts, res)
    print(
        f"omega={omega}: converged in {res['n_iter']} iters | "
        f"3D(42 ord) vs 1D-DO(64 mu): max|dJ|/J_top = {rel:.3e}, "
        f"local rel err (J>1e-2 J_top) = {rel_loc:.3e} | "
        f"two-stream vs 1D-DO: {rel_ts:.3e}"
    )
    # energy conservation: dFz/dz = -4 pi kappa_a J (B = 0); compare the
    # flux difference (midpoint quantity) with midpoint heating
    Fz = res["F"][:, 2].reshape(grid.nz, grid.nx, grid.ny)[:, 1, 1]
    dFz = np.diff(Fz) / HZ
    heating = 4 * np.pi * KAPPA * (1 - omega) * 0.5 * (J3d[1:] + J3d[:-1])
    err = np.abs(dFz + heating).max() / heating.max()
    print(f"           div F vs heating: max|err|/max(heating) = {err:.3e}")

# %% plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, omega in zip(axes, (0.5, 0.9)):
    J3d, J_ref, J_ts, _ = results[omega]
    tau_from_top = TAU - tau_coord
    ax.plot(tau_from_top, J_ref, "k-", lw=1.6, label="1D DO, 64 mu (reference)")
    ax.plot(tau_from_top, J3d, "o", ms=4, mfc="none", color="C0",
            label="rt3d, 42 ordinates")
    ax.plot(tau_from_top, J_ts, "--", color="C2", lw=1.2, label="two-stream")
    ax.set_xlabel("tau from top")
    ax.set_ylabel("J")
    ax.set_title(f"isotropic scattering, g=0, omega={omega}, tau=5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("phase 2: 1D slab, scattering + plain lambda iteration")
fig.tight_layout()
fig.savefig("plot/phase2_slab.png", dpi=150)
print("  -> plot/phase2_slab.png")
