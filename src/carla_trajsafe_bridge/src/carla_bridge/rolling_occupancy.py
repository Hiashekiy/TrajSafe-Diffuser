"""Build canonical 160 m occupancy grids directly from a CARLA map."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np

from .rolling_frame import WorldSceneFrame


@dataclass
class LaneSegment:
    start: np.ndarray
    end: np.ndarray
    width_m: float


class CarlaMapRasterizer:
    """Cache CARLA driving-lane segments, then rasterize cheap rolling crops."""

    def __init__(self, segments: Iterable[LaneSegment]):
        self.segments = list(segments)

    @classmethod
    def from_carla_map(cls, carla_map, spacing_m: float = 1.5):
        segments: List[LaneSegment] = []
        waypoints = carla_map.generate_waypoints(float(spacing_m))
        for waypoint in waypoints:
            try:
                next_points = waypoint.next(float(spacing_m))
            except Exception:
                next_points = []
            p0 = np.array([waypoint.transform.location.x,
                           waypoint.transform.location.y], dtype=np.float64)
            for nxt in next_points:
                p1 = np.array([nxt.transform.location.x,
                               nxt.transform.location.y], dtype=np.float64)
                segments.append(LaneSegment(
                    start=p0, end=p1,
                    width_m=max(float(waypoint.lane_width),
                                float(nxt.lane_width))))
        if not segments:
            raise RuntimeError("CARLA map yielded no driving-lane segments")
        return cls(segments)

    @staticmethod
    def _pixels(frame: WorldSceneFrame, points_world) -> np.ndarray:
        # Canonical model images increase in the same direction as scene Y.
        return np.rint(frame.grid_from_world(points_world)).astype(np.int32)

    def rasterize(self, frame: WorldSceneFrame,
                  obstacle_polygons_world: Optional[Iterable[np.ndarray]] = None,
                  obstacle_erosion_cells: int = 8) -> np.ndarray:
        res = int(frame.resolution)
        free = np.zeros((res, res), dtype=np.uint8)
        bounds = frame.bounds_world
        pixel_m = float(frame.size_m) / float(res)
        for segment in self.segments:
            lo = np.minimum(segment.start, segment.end) - segment.width_m
            hi = np.maximum(segment.start, segment.end) + segment.width_m
            if hi[0] < bounds[0] or lo[0] > bounds[2] \
                    or hi[1] < bounds[1] or lo[1] > bounds[3]:
                continue
            p = self._pixels(frame, np.stack((segment.start, segment.end)))
            thickness = max(1, int(math.ceil(segment.width_m / pixel_m)))
            cv2.line(free, tuple(p[0]), tuple(p[1]), 1, thickness,
                     lineType=cv2.LINE_8)

        obstacle = (free == 0).astype(np.uint8)
        k = int(obstacle_erosion_cells)
        if k > 0:
            source = obstacle.copy()
            obstacle = cv2.erode(
                obstacle, np.ones((3, 3), dtype=np.uint8), iterations=k,
                borderType=cv2.BORDER_CONSTANT, borderValue=1)
            # Match the protected-border k=8 training cache.
            obstacle[:k, :] |= source[:k, :]
            obstacle[-k:, :] |= source[-k:, :]
            obstacle[:, :k] |= source[:, :k]
            obstacle[:, -k:] |= source[:, -k:]

        for polygon in obstacle_polygons_world or []:
            poly = self._pixels(frame, np.asarray(polygon, dtype=np.float64))
            if len(poly) >= 3:
                cv2.fillPoly(obstacle, [poly.reshape(-1, 1, 2)], 1)
        return np.ascontiguousarray(obstacle.astype(np.uint8))


def vehicle_obstacle_polygons(world, exclude_actor_id: Optional[int] = None,
                              margin_m: float = 0.3) -> List[np.ndarray]:
    """Snapshot all other vehicle footprints as world-XY polygons."""
    polygons: List[np.ndarray] = []
    for actor in world.get_actors().filter("vehicle.*"):
        if exclude_actor_id is not None and int(actor.id) == int(exclude_actor_id):
            continue
        transform = actor.get_transform()
        extent = actor.bounding_box.extent
        yaw = math.radians(float(transform.rotation.yaw))
        along = np.array([math.cos(yaw), math.sin(yaw)])
        right = np.array([-math.sin(yaw), math.cos(yaw)])
        center = np.array([transform.location.x, transform.location.y])
        half_l = float(extent.x) + float(margin_m)
        half_w = float(extent.y) + float(margin_m)
        polygons.append(np.stack([
            center + along * half_l + right * half_w,
            center + along * half_l - right * half_w,
            center - along * half_l - right * half_w,
            center - along * half_l + right * half_w,
        ]))
    return polygons
