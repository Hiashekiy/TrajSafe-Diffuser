"""field-guided-search: map + guidance path -> obstacle-aware field -> guided grid search.

Minimal core, six modules, one required dependency (numpy; scipy optional):

    from field_guided_search import GuidanceField

    gf = GuidanceField(occupancy)        # (H, W), 0 = free
    field = gf.build(trajectory)         # (N, 2) as [row, col]; may cross obstacles
    res = gf.search(start, goal)         # field-only; alpha=0.5 fuses the goal distance
    res["path"], res["length"], res["expanded"], res["status"]

Convenience functions: :func:`field_from_trajectory`, :func:`guided_search`.
"""

from __future__ import annotations

__version__ = "1.1.0"

from .api import GuidanceField, field_from_trajectory, guided_search
from .clearance import (
    RISK_MODES,
    ClearanceRiskMap,
    clearance_distance,
    clearance_risk,
    validate_penalty_map,
)
from .field import (
    DEFAULT_LAM,
    WEIGHTS,
    build_field,
    propagate,
    propagate_heap,
    propagate_scipy,
    weight_at_progress,
)
from .grid import Grid
from .search import (
    FIELD_CLIP,
    astar,
    normalise_distance,
    normalise_field,
    path_is_valid,
)
from .search import guided_search as guided_search_grid
from .sources import resample_polyline, trajectory_sources

__all__ = [
    "__version__",
    # high level
    "GuidanceField", "field_from_trajectory", "guided_search",
    # building blocks
    "Grid", "resample_polyline", "trajectory_sources",
    "build_field", "propagate", "propagate_scipy", "propagate_heap",
    "weight_at_progress", "WEIGHTS", "DEFAULT_LAM",
    # search
    "astar", "guided_search_grid", "path_is_valid",
    "normalise_field", "normalise_distance", "FIELD_CLIP",
    # clearance-aware search
    "ClearanceRiskMap", "clearance_distance", "clearance_risk", "validate_penalty_map",
    "RISK_MODES",
]
