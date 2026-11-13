# -*- coding: utf-8 -*-
"""
rt3d.mc — Monte Carlo solver family (plan/mc-plan.md).

Woodcock null-collision transport on the same grids, direction set and
boundary conditions as the short-characteristics solver, so the two solver
families can be differenced cell-by-cell:

    tracking.py   shared flight sampler + null-collision event decision,
                  unigrid and octree DDA traversal kernels
    sources.py    HG inverse-CDF scatter sampling (thermal/fixed-source
                  birth sampling arrives with stage 1, M1)
    forward.py    solver A (stage-1 J, stage-2 radiance)      — M1 / M2
    backward.py   solver B (adjoint detector MC)              — M3

Phase M0 delivers the tracker (both grids, mode="cell" and mode="global"
with real null collisions), the event decision and the HG sampler.
"""

from rt3d.mc.tracking import (
    EV_ABSORB,
    EV_EXIT,
    EV_NULL,
    EV_SCATTER,
    event_decision,
    flight_distance,
    path_integral_from_segments,
    sample_collisions,
    walk_lattice,
    walk_octree,
)
from rt3d.mc.sources import hg_cos_theta, hg_directions
