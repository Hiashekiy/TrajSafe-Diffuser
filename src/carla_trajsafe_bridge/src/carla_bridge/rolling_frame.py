"""World-aligned 160 m frame used by the rolling CARLA planner.

Unlike the legacy 80 m frame this frame never rotates with the ego vehicle.
That matches ``carla_full_160_256``, whose scene coordinates are an affine
mapping of CARLA world X/Y and whose canonical image has +world Y downward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCENE_SIZE_M = 160.0
SCENE_SCALE_M = SCENE_SIZE_M / 2.0
DEFAULT_RES = 256


@dataclass(frozen=True)
class WorldSceneFrame:
    """Axis-aligned square world window mapped onto scene ``[-1, 1]^2``."""

    center_world: np.ndarray
    size_m: float = SCENE_SIZE_M
    resolution: int = DEFAULT_RES

    def __post_init__(self):
        center = np.asarray(self.center_world, dtype=np.float64).reshape(2)
        object.__setattr__(self, "center_world", center)
        if float(self.size_m) <= 0:
            raise ValueError("size_m must be positive")
        if int(self.resolution) < 2:
            raise ValueError("resolution must be >= 2")

    @property
    def scene_scale_m(self) -> float:
        return float(self.size_m) * 0.5

    @property
    def bounds_world(self) -> np.ndarray:
        half = self.scene_scale_m
        return np.array([self.center_world[0] - half,
                         self.center_world[1] - half,
                         self.center_world[0] + half,
                         self.center_world[1] + half], dtype=np.float64)

    @classmethod
    def around_polyline(cls, points_world, size_m: float = SCENE_SIZE_M,
                        resolution: int = DEFAULT_RES,
                        margin_m: float = 10.0) -> "WorldSceneFrame":
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 2)
        if len(points) == 0:
            raise ValueError("points_world must not be empty")
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        usable = float(size_m) - 2.0 * float(margin_m)
        if np.any(hi - lo > usable + 1e-6):
            raise ValueError("route span %s m does not fit %.1f m window with %.1f m margin"
                             % (np.round(hi - lo, 3).tolist(), size_m, margin_m))
        return cls((lo + hi) * 0.5, size_m=float(size_m),
                   resolution=int(resolution))

    def scene_from_world(self, points_world) -> np.ndarray:
        pts = np.asarray(points_world, dtype=np.float64)
        return (pts - self.center_world) / self.scene_scale_m

    def world_from_scene(self, points_scene) -> np.ndarray:
        pts = np.asarray(points_scene, dtype=np.float64)
        return self.center_world + pts * self.scene_scale_m

    # Compatibility names used by the legacy adapter.
    to_scene = scene_from_world

    def scene_to_grid(self, points_scene) -> np.ndarray:
        pts = np.asarray(points_scene, dtype=np.float64)
        return (pts + 1.0) * 0.5 * (int(self.resolution) - 1)

    def grid_from_world(self, points_world) -> np.ndarray:
        return self.scene_to_grid(self.scene_from_world(points_world))

    def contains_world(self, points_world, margin_m: float = 0.0) -> bool:
        scene = np.abs(self.scene_from_world(points_world))
        limit = 1.0 - float(margin_m) / self.scene_scale_m
        return bool(np.all(scene <= limit + 1e-9))

    def describe(self) -> dict:
        return {"center_world": self.center_world.tolist(),
                "bounds_world": self.bounds_world.tolist(),
                "size_m": float(self.size_m),
                "resolution": int(self.resolution),
                "scene_scale_m": self.scene_scale_m}

