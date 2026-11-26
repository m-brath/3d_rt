# -*- coding: utf-8 -*-
"""
Shared cases, grids and boundary setups for the solver-matrix tests
(plan/solver-test-matrix-plan.md).

The matrix tests all run on the same two physical setups so that every solver
variant is differenced against the *same* fields, boundary conditions and
direction grid:

``gaussian_cloud`` case  (the MANUAL section 11 / scripts/compare_transport.py
    case): Gaussian cloud, omega capped so plain lambda iteration stays
    tractable, per-cell g, 42 icosahedral ordinates, top inflow, 30 deg beam.
``slab`` case  (rt3d.cases.slab_1d): 1D slab, periodic x/y, useful where an
    analytic 1D reference exists.

Two boundary setups are offered, because the solver families differ today
(plan section 1, finding F1):

``common_bc``   top/bottom face inflows only -- supported by every solver.
``allsky_bc``   top inflow + Lambertian, emissive ground -- the full all-sky
                case; SC-only until MC phase M6 lands.

Grids are cached (building the octree is cheap but repeated in dozens of
parametrizations).  The cached objects are **read-only** for tests: never
mutate a grid returned by these factories.
"""

import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rt3d.cases.gaussian_cloud import gaussian_cloud
from rt3d.cases.slab_1d import slab_grid, slab_profile
from rt3d.directions import icosahedric_grid
from rt3d.geometry import ground_surface, unigrid
from rt3d.octree import octree_grid
from rt3d.transport import boundary_conditions

# --- cloud case constants (match scripts/compare_transport.py) --------------
OMEGA_CAP = 0.9          # keeps plain lambda iteration tractable
OCTREE_H0 = 1.25         # base cell size of the octree build
SUN_ZENITH_DEG = 30.0    # beam direction of the MANUAL case
GROUND_ALBEDO = 0.3


# =============================================================================
# cases
# =============================================================================


def cloud_case():
    """The Gaussian-cloud case dict (box + callable input fields)."""
    return gaussian_cloud(omega_cap=OMEGA_CAP)


@functools.lru_cache(maxsize=None)
def _beam_cached():
    mu = np.cos(np.radians(SUN_ZENITH_DEG))
    return {
        "omega_sun": np.array([np.sqrt(1.0 - mu * mu), 0.0, -mu]),
        "F0": 1.0,
    }


def cloud_beam():
    """Solar beam of the MANUAL case (read-only copy)."""
    b = _beam_cached()
    return {"omega_sun": b["omega_sun"].copy(), "F0": b["F0"]}


@functools.lru_cache(maxsize=None)
def direction_grid(n_refine=1):
    """Icosahedral direction grid, cached per refinement level."""
    return icosahedric_grid(n_refine=n_refine)


# =============================================================================
# grids
# =============================================================================


@functools.lru_cache(maxsize=None)
def cloud_unigrid(n=24):
    """Uniform ``n x n x n`` grid over the cloud box (cells at centers)."""
    box = cloud_case()["box"]
    xs = np.linspace(box.x0, box.x1, n + 1)[:-1] + (box.Lx / n) / 2
    zs = np.linspace(box.z0, box.z1, n + 1)[:-1] + (box.Lz / n) / 2
    return unigrid(xs, xs, zs)


@functools.lru_cache(maxsize=None)
def cloud_octree(tau_max=2.0, depth=3, h0=OCTREE_H0):
    """
    Refined octree over the cloud box, fields sampled from the case.

    ``tau_max``/``depth`` default to the ``oct3`` rung of
    scripts/compare_transport.py (25 264 leaves at depth 3); ``depth=5,
    tau_max=0.5`` is the refinement-saturated reference rung.  ``h0`` is the
    **base cell size**: refinement only adds leaves where the cloud is, so
    the boundary faces keep base-size leaves and ``h0`` -- not ``depth`` --
    sets the accuracy of anything evaluated on a face (the energy closure of
    rt3d/diagnostics.py, plan section 8 finding F7).
    """
    case = cloud_case()
    o = octree_grid(case["box"], h0=h0)
    o.refine(case["kappa_e"], tau_max=tau_max, eps_grad=1.0, max_depth=depth)
    o.balance_21(max_depth=depth)
    o.finalize()
    o.sample_fields(
        kappa_e=case["kappa_e"],
        kappa_s=case["kappa_s"],
        g=case["g"],
        T=case["T"],
    )
    return o


@functools.lru_cache(maxsize=None)
def slab_setup(tau=5.0, kappa_e=1.0, nz=40, nxy=4):
    """1D-slab unigrid (periodic x/y) plus its box, cell size and profile."""
    grid, box, hz = slab_grid(tau=tau, kappa_e=kappa_e, nz=nz, nxy=nxy)
    return grid, box, hz


# =============================================================================
# fields
# =============================================================================


def cloud_fields(grid, beam_scattering=False):
    """
    ``(kappa_e, kappa_s, g, B)`` arrays of the cloud case on `grid`.

    ``B`` is zero (pure scattering) like the MANUAL/compare case; the beam is
    passed separately in the solver call.
    """
    case = cloud_case()
    c = grid.centers
    if isinstance(grid, octree_grid):
        return grid.kappa_e, grid.kappa_s, grid.g, np.zeros(grid.n_leaf)
    return (
        case["kappa_e"](*c.T),
        case["kappa_s"](*c.T),
        case["g"](*c.T),
        np.zeros(grid.n_cells),
    )


# =============================================================================
# boundary conditions
# =============================================================================


def common_bc(top_I=1.0, bottom_I=0.5, side_I=0.0):
    """Face inflows only -- the setup every solver family supports today."""
    return boundary_conditions(top_I=top_I, bottom_I=bottom_I, side_I=side_I)


def allsky_bc(top_I=1.0, albedo=GROUND_ALBEDO, emissivity=0.0, B_g=0.0):
    """Top inflow + Lambertian/emissive ground (the full all-sky case)."""
    return boundary_conditions(
        top_I=top_I,
        ground=ground_surface(albedo=albedo, emissivity=emissivity, B_g=B_g),
    )


def ground_case():
    """Case + box + resolved direction grid + beam of the all-sky setup."""
    return cloud_case(), direction_grid(1), cloud_beam()


# =============================================================================
# converged deterministic solves (memoized)
# =============================================================================

_SOLVES = {}


def solve_cloud(grid, bc="common", top_I=1.0, bottom_I=0.5, albedo=0.0,
                emissivity=1.0, B_g=0.0, beam=True, periodic=True,
                n_iter=80, tol=1e-6, **solver_kw):
    """
    Converged ``lambda_iteration`` solve of the cloud case, memoized.

    ``bc`` is ``"common"`` (face inflows ``top_I``/``bottom_I``) or
    ``"ground"`` (``top_I`` plus a flat Lambertian/emissive ground).  All
    remaining keyword arguments go to the solver.  Results are shared
    between tests, so **never mutate a returned dict or its arrays**.
    """
    from rt3d.solver import lambda_iteration

    key = (id(grid), bc, top_I, bottom_I, albedo, emissivity, B_g, beam,
           periodic, n_iter, tol, tuple(sorted(solver_kw.items())))
    if key not in _SOLVES:
        kappa_e, kappa_s, g, B = cloud_fields(grid)
        if bc == "common":
            bcs = boundary_conditions(top_I=top_I, bottom_I=bottom_I)
        elif bc == "ground":
            bcs = boundary_conditions(
                top_I=top_I,
                ground=ground_surface(albedo=albedo, emissivity=emissivity,
                                      B_g=B_g),
            )
        else:
            raise ValueError(f"unknown bc kind {bc!r}")
        _SOLVES[key] = lambda_iteration(
            grid, direction_grid(1), kappa_e, kappa_s, B, bcs,
            g=g, periodic_box=cloud_case()["box"] if periodic else None,
            beam=cloud_beam() if beam else None,
            n_iter=n_iter, tol=tol, **solver_kw,
        )
    return _SOLVES[key]


# =============================================================================
# solver-variant descriptions (ids used in parametrizations and reports)
# =============================================================================

# deterministic solver variants that must all reproduce the same converged
# field (within their documented discretization ranges)
SC_VARIANTS = {
    # key: kwargs for lambda_iteration
    "chain-jit": {"transport": "chain", "jit": None},
    "chain-numpy": {"transport": "chain", "jit": False},
    "sc-jit": {"transport": "sc", "jit": None},
    "sc-numpy": {"transport": "sc", "jit": False},
    "chain-jobs4": {"transport": "chain", "jit": None, "n_jobs": 4},
}
