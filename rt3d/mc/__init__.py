# -*- coding: utf-8 -*-
"""
rt3d.mc — Monte Carlo solver family (plan/mc-plan.md).

Woodcock null-collision transport on the same grids, direction set and
boundary conditions as the short-characteristics solver, so the two solver
families can be differenced cell-by-cell:

    tracking.py     shared flight sampler + null-collision event decision,
                    unigrid and octree DDA traversal kernels, analog history
                    chains with HG resampling (stage-1 explicit transport)
    sources.py      HG inverse-CDF scatter sampling, isotropic / face-inflow /
                    beam direction and position samplers, birth-point sampling
                    from inhomogeneous source rates (null source events)
    estimators.py   track-length (exponential-path) J and F tallies with
                    history-batch error bars
    forward.py      solver A stage 1: j_mc(stage1="lambda" | "explicit")   (M1)
                    solver A stage 2: radiance from the fixed source        (M2)
    backward.py     solver B (adjoint detector MC)                          (M3)

Phase M0 delivered the tracker (both grids, mode="cell" and mode="global"
with real null collisions), the event decision and the HG sampler; M1 adds the
stage-1 mean-intensity solvers.
"""

from rt3d.mc.backward import all_sky_cone_half, backward_all_sky, backward_detector_mc
from rt3d.mc.estimators import cell_volumes, heating_rate, n_cells_of, track_tally
from rt3d.mc.forward import j_mc, radiance_from_source, stage2_source
from rt3d.mc.sources import (
    beam_seed,
    face_area,
    face_inflow_directions,
    face_points,
    hg_cos_theta,
    hg_directions,
    hg_scatter_directions,
    isotropic_directions,
    sample_births,
)
from rt3d.mc.tracking import (
    EV_ABSORB,
    EV_EXIT,
    EV_NULL,
    EV_SCATTER,
    event_decision,
    flight_budget,
    flight_distance,
    flight_segments,
    path_integral_from_segments,
    sample_collisions,
    track_histories,
    walk_lattice,
    walk_octree,
)
