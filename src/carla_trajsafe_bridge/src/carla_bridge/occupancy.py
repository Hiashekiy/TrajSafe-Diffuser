"""Occupancy sources for the CARLA bridge.

Two sources are supported and both return the SAME canonical 256 x 256 image
the checkpoint was trained on (0 = free, 1 = obstacle):

dataset   data/carla_processed/<split>/occupancy.npy -- the frozen snapshot the
          dashboard uses.  It is already canonical (no flip needed).
live      rasterised from the CARLA Town driving-lane quad cache
          (data/carla_v1/maps/<Town>/driving_lane_quads.npy) with the recorded
          ego basis of the anchor frame, then flipped ONCE by
          frame.raw_occupancy_to_canonical().

Measured agreement between the two for test_0056 is 99.33 percent of cells;
the difference is the dataset cleaning margin, not a geometry error.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

from .frame import DEFAULT_RES, LocalFrame, raw_occupancy_to_canonical

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
CARLA_TOOLS = os.path.join(WORKSPACE_ROOT, "tools", "carla")

__all__ = [
    "REPO_ROOT", "WORKSPACE_ROOT", "CARLA_TOOLS", "default_dataset_root",
    "load_town_quads", "load_dataset_occupancy", "build_canonical_occupancy",
    "occupancy_agreement", "scene_points_are_free", "free_mask",
    "add_box_obstacles", "box_corners_local", "inflate_obstacles",
    "body_free_rate", "body_penetration",
]


def _import_map_cache():
    """Import the dataset rasteriser (tools/carla/map_cache.py) lazily."""
    if CARLA_TOOLS not in sys.path:
        sys.path.insert(0, CARLA_TOOLS)
    import map_cache

    return map_cache


def default_dataset_root() -> str:
    return os.path.join(WORKSPACE_ROOT, "data", "carla_v1")


def load_town_quads(dataset_root: str, town: str) -> np.ndarray:
    """Driving-lane world quads [Q,4,2] of one Town (cached by the pipeline)."""
    path = os.path.join(str(dataset_root), "maps", str(town), "driving_lane_quads.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "missing driving-lane quad cache %s; run tools/carla/collect_dataset.py "
            "or tools/carla/export_carla_map.py for Town %s" % (path, town))
    quads = np.load(path)
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        raise ValueError("unexpected quad shape %r" % (quads.shape,))
    return quads


def load_dataset_occupancy(processed_root: str, split: str, index: int) -> np.ndarray:
    """Frozen canonical occupancy of one dataset index (no flip applied)."""
    path = os.path.join(str(processed_root), str(split), "occupancy.npy")
    if not os.path.exists(path):
        raise FileNotFoundError("missing processed occupancy %s" % path)
    arr = np.load(path, mmap_mode="r")
    occ = np.array(arr[int(index)], dtype=np.uint8)
    if occ.shape != (DEFAULT_RES, DEFAULT_RES):
        raise ValueError("unexpected occupancy shape %r" % (occ.shape,))
    return occ


def build_canonical_occupancy(quads: np.ndarray, frame: LocalFrame,
                              res: int = DEFAULT_RES) -> np.ndarray:
    """Rasterise world lane quads in the anchor frame and flip exactly once."""
    map_cache = _import_map_cache()
    raw = map_cache.rasterize_local_occupancy(
        np.asarray(quads, dtype=np.float64),
        frame.anchor_xy,
        frame.forward_xy,
        frame.right_xy,
        int(res),
    )
    return raw_occupancy_to_canonical(raw).astype(np.uint8)


def occupancy_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch %r vs %r" % (a.shape, b.shape))
    return float((a == b).mean())


def free_mask(occupancy: np.ndarray, xy_scene: np.ndarray) -> np.ndarray:
    """Per-point free test of scene XY [N,2] on the canonical image."""
    pts = LocalFrame.scene_to_grid(np.asarray(xy_scene, dtype=np.float64),
                                   np.asarray(occupancy).shape[0])
    rows = np.rint(pts[..., 1]).astype(np.int64)
    cols = np.rint(pts[..., 0]).astype(np.int64)
    res = np.asarray(occupancy).shape[0]
    inside = (rows >= 0) & (rows < res) & (cols >= 0) & (cols < res)
    out = np.zeros(rows.shape, dtype=bool)
    out[inside] = np.asarray(occupancy)[rows[inside], cols[inside]] == 0
    return out


def scene_points_are_free(occupancy: np.ndarray, xy_scene: np.ndarray) -> bool:
    return bool(free_mask(occupancy, xy_scene).all())

def inflate_obstacles(occupancy: np.ndarray, radius_m: float,
                      res: int = None) -> np.ndarray:
    """Erode the drivable area by radius_m (Minkowski inflation of obstacles).

    The planner treats the trajectory as a POINT, so a point-in-free-space plan
    can still put a 1.99 m wide car over the kerb.  Inflating every obstacle by
    the ego half-width plus a margin turns the occupancy into configuration
    space: a centreline that is free there keeps the whole vehicle body inside.
    """
    import cv2

    occ = np.asarray(occupancy)
    res = int(occ.shape[0]) if res is None else int(res)
    cells = int(round(float(radius_m) * res / 80.0))
    if cells <= 0:
        return np.array(occ, dtype=np.uint8, copy=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * cells + 1, 2 * cells + 1))
    out = cv2.dilate((occ > 0).astype(np.uint8), kernel)
    return out.astype(np.uint8)


def body_corner_points(xy_local: np.ndarray, heading: np.ndarray,
                       half_length_m: float, half_width_m: float) -> np.ndarray:
    """Four body corners of the ego at every pose, in local metres [N*4, 2]."""
    xy_local = np.asarray(xy_local, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    along = np.stack((cos_h, sin_h), axis=1)
    across = np.stack((-sin_h, cos_h), axis=1)
    out = []
    for sign_l, sign_w in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        out.append(xy_local + sign_l * half_length_m * along
                   + sign_w * half_width_m * across)
    return np.concatenate(out, axis=0)


def body_free_rate(occupancy: np.ndarray, xy_local: np.ndarray, heading: np.ndarray,
                   half_length_m: float, half_width_m: float):
    """(free rate, worst penetration in metres) of the ego footprint."""
    import cv2

    corners = body_corner_points(xy_local, heading, half_length_m, half_width_m)
    scene = LocalFrame.local_to_scene(corners)
    free = free_mask(occupancy, scene)
    inside = cv2.distanceTransform((np.asarray(occupancy) != 0).astype(np.uint8),
                                   cv2.DIST_L2, 5) / 3.1875
    grid = LocalFrame.scene_to_grid(scene, np.asarray(occupancy).shape[0])
    rows = np.clip(np.rint(grid[:, 1]).astype(int), 0, inside.shape[0] - 1)
    cols = np.clip(np.rint(grid[:, 0]).astype(int), 0, inside.shape[1] - 1)
    penetration = np.where(free, 0.0, inside[rows, cols])
    return float(free.mean()), float(penetration.max())

def box_corners_local(x_local: float, y_local: float, yaw_world_deg: float,
                      length_m: float, width_m: float,
                      frame: LocalFrame, margin_m: float = 0.0) -> np.ndarray:
    """Four corners of a yawed box, in ego-local metres.

    A world yaw theta maps to the local heading (theta - anchor_yaw) because the
    local frame is the anchor own orthonormal basis.  The box is inflated by
    margin_m on every side: that is the Minkowski expansion the integration
    plan asks for, and it covers the controller cross-track error.
    """
    half_l = 0.5 * float(length_m) + float(margin_m)
    half_w = 0.5 * float(width_m) + float(margin_m)
    local_heading = math.radians(float(yaw_world_deg) - float(frame.yaw_deg))
    cos_h, sin_h = math.cos(local_heading), math.sin(local_heading)
    along = np.array([cos_h, sin_h])
    across = np.array([-sin_h, cos_h])
    center = np.array([float(x_local), float(y_local)])
    corners = []
    for sign_l, sign_w in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        corners.append(center + sign_l * half_l * along + sign_w * half_w * across)
    return np.asarray(corners, dtype=np.float64)


def add_box_obstacles(occupancy: np.ndarray, frame: LocalFrame,
                      boxes, margin_m: float = 0.35) -> np.ndarray:
    """Burn yawed boxes into a COPY of the canonical occupancy.

    Each box is a dict with x_local / y_local / yaw_world_deg / length_m /
    width_m, so a scenario is described in the same metric frame the plan is
    reasoned about in.  Pixels are written through LocalFrame.scene_to_grid,
    which is the very mapping the planner uses, so an obstacle can never end
    up half a cell away from where the sampler believes it is.
    """
    import cv2

    out = np.array(occupancy, dtype=np.uint8, copy=True)
    res = int(out.shape[0])
    for box in boxes:
        corners_local = box_corners_local(box["x_local"], box["y_local"],
                                          box["yaw_world_deg"], box["length_m"],
                                          box["width_m"], frame, margin_m)
        corners_scene = LocalFrame.local_to_scene(corners_local)
        pixels = LocalFrame.scene_to_grid(corners_scene, res)
        polygon = np.rint(pixels).astype(np.int32)
        cv2.fillPoly(out, [polygon], color=1)
    return out
