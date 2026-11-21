"""Public API: map + guidance path -> guidance field -> guided search.

    from field_guided_search import GuidanceField

    gfield = GuidanceField(occupancy)          # occupancy: (H, W) with 0 = free
    field  = gfield.build(trajectory)          # trajectory: (N, 2) as [row, col]
    result = gfield.search(start, goal)        # or search(..., alpha=0.5) for fusion
    result["path"], result["expanded"], result["length"], result["status"]

``GuidanceField`` is a thin state holder; the same work is available as two functions:
:func:`field_from_trajectory` and :func:`guided_search`.
"""

from __future__ import annotations

import numpy as np

from .clearance import ClearanceRiskMap
from .field import DEFAULT_LAM, WEIGHTS, build_field, propagate
from .grid import Grid
from .search import (
    DISTANCE_MODES,
    FIELD_CLIP,
    astar,
    guided_search as _guided_search,
    normalise_field,
    path_is_valid,
)
from .sources import trajectory_sources


class GuidanceField:
    """Turns one occupancy map into a reusable field-guided-search builder + searcher."""

    def __init__(self, occupancy, lam: float = DEFAULT_LAM, weight: str = "linear",
                 num_samples: int = 64, source_method: str = "drop",
                 projection_radius: int = 3, method: str = "auto", clip=None,
                 distance: str = "manhattan"):
        self.grid = Grid(occupancy)
        if weight not in WEIGHTS:
            raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
        if source_method not in ("project", "drop"):
            raise ValueError("source_method must be 'project' or 'drop'")
        if distance not in DISTANCE_MODES:
            raise ValueError(f"distance must be one of {DISTANCE_MODES}, got {distance!r}")
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError(f"lam must be finite and > 0, got {lam!r}")
        if int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples!r}")
        if int(projection_radius) < 0:
            raise ValueError(f"projection_radius must be >= 0, got {projection_radius!r}")
        self.lam = float(lam)
        self.weight = weight
        self.num_samples = int(num_samples)
        self.source_method = source_method
        self.projection_radius = int(projection_radius)
        self.method = method
        self.distance = distance
        self.clip = clip if clip is not None else FIELD_CLIP.get(weight)
        self.field: np.ndarray | None = None
        self.sources: dict | None = None
        self._clearance: ClearanceRiskMap | None = None

    # ------------------------------------------------------------------ #
    def build(self, trajectory, num_samples: int | None = None) -> np.ndarray:
        """Build and store the field from a guidance trajectory that may cross obstacles."""
        sources = trajectory_sources(
            self.grid, trajectory,
            num_samples=self.num_samples if num_samples is None else int(num_samples),
            method=self.source_method, radius=self.projection_radius,
        )
        if len(sources["positions_rc"]) == 0:
            raise RuntimeError(
                "no legal source left after handling collisions "
                f"(method={self.source_method!r}, {sources['num_out_of_bounds']} sample(s) "
                "outside the map, "
                f"{sources['num_dropped']} inside obstacles); the trajectory never touches "
                "free space -- pass a trajectory that follows the map, or use "
                "source_method='project'")
        # one source is legal: it yields the degenerate field F(x) = w_1 - lam*d(x, s_1)
        field = build_field(self.grid, sources["positions_rc"], sources["progress"],
                            self.lam, self.weight, self.method)
        self.sources = sources
        self.field = field
        return field

    # ------------------------------------------------------------------ #
    def search(self, start, goal, alpha: float = 0.0, gamma: float = 1.0,
               adaptive: bool = False, alpha_far: float = 0.25, alpha_near: float = 0.75,
               distance: str | None = None, cutoff: int = 0,
               clearance_beta: float = 0.0, clearance_mode: str = "quadratic_hinge",
               clearance_threshold: float = 3.0, clearance_sigma: float = 2.0,
               clearance_tau: float = 0.7,
               clearance_penalty_map: np.ndarray | None = None) -> dict:
        """Field-guided best-first search (needs :meth:`build` first unless alpha == 1).

        ``distance`` overrides the model chosen in the constructor for this call only.
        ``clearance_beta > 0`` adds the obstacle-clearance risk to the node priority; the
        distance transform is cached on the instance, so repeated searches on this map pay for
        it once.
        """
        if distance is not None and distance not in DISTANCE_MODES:
            raise ValueError(f"distance must be one of {DISTANCE_MODES}, got {distance!r}")
        distance = self.distance if distance is None else distance
        if self.field is None and not (alpha >= 1.0 and not adaptive):
            raise RuntimeError("build() the guidance field first, or use astar()")
        for name, cell in (("start", start), ("goal", goal)):
            if not (0 <= int(cell[0]) < self.grid.height and 0 <= int(cell[1]) < self.grid.width):
                raise ValueError(f"{name} {tuple(cell)} is outside the {self.grid.height}x"
                                 f"{self.grid.width} map")
            if not self.grid.free[int(cell[0]), int(cell[1])]:
                raise ValueError(f"{name} {tuple(cell)} is an obstacle")
        return _guided_search(self.grid, start, goal, self.field, alpha=alpha, gamma=gamma,
                              adaptive=adaptive, alpha_far=alpha_far, alpha_near=alpha_near,
                              clip=self.clip, weight=self.weight, distance=distance,
                              cutoff=cutoff, clearance_beta=clearance_beta,
                              clearance_mode=clearance_mode,
                              clearance_threshold=clearance_threshold,
                              clearance_sigma=clearance_sigma, clearance_tau=clearance_tau,
                              clearance_penalty_map=clearance_penalty_map,
                              clearance_cache=self.clearance())

    def clearance(self) -> ClearanceRiskMap:
        """The clearance cache bound to this map (``D`` is computed on first use)."""
        if self._clearance is None:
            self._clearance = ClearanceRiskMap(self.grid)
        return self._clearance

    def astar(self, start, goal, cutoff: int = 0) -> dict:
        """Shortest-path baseline for this map."""
        return astar(self.grid, start, goal, cutoff)

    def is_valid(self, path, start, goal) -> dict:
        return path_is_valid(self.grid, path, start, goal)

    def normalised(self) -> dict:
        """The two scalar maps the search actually uses (handy for plotting or reuse)."""
        if self.field is None:
            raise RuntimeError("build() the guidance field first")
        return {"F_hat": normalise_field(self.field, self.clip, self.weight),
                "field": self.field}


# --------------------------------------------------------------------------- #
# functional shortcuts
# --------------------------------------------------------------------------- #
def field_from_trajectory(occupancy, trajectory, num_samples: int = 64, **kwargs) -> np.ndarray:
    """One call: map + guidance path -> field ``(H, W)`` (NaN on obstacles)."""
    gfield = GuidanceField(occupancy, **_field_kwargs(kwargs))
    return gfield.build(trajectory, num_samples=num_samples)


def guided_search(occupancy, start, goal, trajectory=None, field=None, **kwargs) -> dict:
    """One call: map + (guidance path or prebuilt field) -> search result."""
    grid = Grid(occupancy)
    if field is None:
        if trajectory is None:
            raise ValueError("pass either `trajectory` or `field`")
        gfield = GuidanceField(occupancy, **_field_kwargs(kwargs))
        field = gfield.build(trajectory, num_samples=kwargs.get("num_samples", 64))
        kwargs.setdefault("weight", gfield.weight)
        kwargs.setdefault("clip", gfield.clip)
    return _guided_search(grid, start, goal, field, **_search_kwargs(kwargs))


def _field_kwargs(kwargs) -> dict:
    keys = ("lam", "weight", "source_method", "projection_radius", "method", "distance")
    return {k: kwargs[k] for k in keys if k in kwargs}


def _search_kwargs(kwargs) -> dict:
    keys = ("alpha", "gamma", "adaptive", "alpha_far", "alpha_near", "clip", "cutoff",
            "distance", "clearance_beta", "clearance_mode", "clearance_threshold",
            "clearance_sigma", "clearance_tau", "clearance_penalty_map")
    out = {k: kwargs[k] for k in keys if k in kwargs}
    if "clip" not in out:
        out["clip"] = kwargs.get("field_clip")
    return out
