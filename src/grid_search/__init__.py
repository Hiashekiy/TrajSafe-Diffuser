"""PDG-Focal A*: progress-aware diffusion-guided Focal A* on an occupancy grid.

The package is independent of the diffusion network: it consumes

    * an occupancy bitmap ``[H, W]`` (project canonical, row 0 <-> scene_y = -1),
    * ``start`` / ``goal`` in PROJECT SCENE coordinates (the ``[-1, 1]`` crop),
    * a reference polyline (the intermediate clean state ``x0`` of the reverse
      diffusion, decoded to scene coordinates),

and returns a safe grid path plus search statistics.  Nothing here changes the
diffusion model, the corridor, the ALM step or any downstream module.

Quick start::

    import numpy as np
    from src.grid_search import (GridSpec, build_safe_grid, build_neighbor_table,
                                 prepare_reference_path, build_progress_guidance,
                                 baselines)

    spec = GridSpec.from_resolution(256, 256, 256, scene_to_meter=80.0)
    safe = build_safe_grid(occupancy, spec, robot_radius_m=0.8,
                           safety_margin_m=0.3, already_inflated_cells=4.0)
    nbr = build_neighbor_table(safe.free, spec.cell_m)
    ref = prepare_reference_path(x0_curve_world, spec, start_world, goal_world)
    field = build_progress_guidance(ref, spec, J=3)

    s = int(spec.cell_to_index(spec.world_to_cell(start_world)))
    g = int(spec.cell_to_index(spec.world_to_cell(goal_world)))
    res = baselines.pdg_focal_astar(safe, s, g, field, w=1.2)   # Ours
    res.path        # [n] cell indices
    res.cost_m      # REAL path length (metres)
    res.stats       # expanded_nodes / search_time_ms / ...

See ``src/grid_search/README.md`` for the full API and the experiment commands.
"""

from .grid_map import (GridSpec, OccupancyGrid, SafeGrid, build_safe_grid,
                       clearance_field, inflate_obstacles)
from .grid_neighbors import (MOVES, NeighborTable, build_neighbor_table,
                             corner_cutting_violations)
from .diffusion_path import ReferencePath, prepare_reference_path, resample_polyline
from .progress_guidance import GuidanceField, build_progress_guidance
from .focal_astar import SearchConfig, SearchResult, focal_search, octile_heuristic
from .baselines import (astar, dijkstra, focal_astar, guided_astar,
                        pdg_focal_astar, run_method, METHODS)
from .path_validation import (path_cost_m, turn_count, validate_path,
                              world_collision_rate)
from .benchmark import (Case, PreparedCase, aggregate, evaluate_case,
                        prepare_case, run_suite, write_csv)
from . import hard_scenes, path_validation, scenes, visualization

__all__ = [
    # grid / safety model
    "GridSpec", "OccupancyGrid", "SafeGrid", "build_safe_grid",
    "clearance_field", "inflate_obstacles",
    "MOVES", "NeighborTable", "build_neighbor_table",
    "corner_cutting_violations",
    # reference trajectory + guidance field
    "ReferencePath", "prepare_reference_path", "resample_polyline",
    "GuidanceField", "build_progress_guidance",
    # search
    "SearchConfig", "SearchResult", "focal_search", "octile_heuristic",
    "dijkstra", "astar", "focal_astar", "pdg_focal_astar", "guided_astar",
    "run_method", "METHODS",
    # validation / benchmark / scenes
    "validate_path", "path_cost_m", "turn_count", "world_collision_rate",
    "Case", "PreparedCase", "prepare_case", "evaluate_case", "run_suite",
    "aggregate", "write_csv",
    "scenes", "hard_scenes", "visualization", "path_validation",
]
