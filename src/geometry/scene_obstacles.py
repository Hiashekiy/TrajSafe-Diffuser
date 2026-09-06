"""Obstacle boundary points for scene maps (used by data precompute + stage-2).

Pure geometry/numpy helpers moved out of the deleted Phase-3 AL loss module so
that scripts/data/08_precompute_scene_obstacles.py keeps working.
"""
import os

import numpy as np

from src.geometry.iris_solver import extract_obstacle_constraints

MAZE_NAMES = ["umaze", "medium", "large"]

# obstacle boundary point cache, keyed by (maze_id, dilation, boundary_jitter).
_OBSTACLE_POINT_CACHE: dict = {}


def scene_obstacle_points(occ, extent=(-1.0, 1.0, -1.0, 1.0),
                          dilation=1, boundary_jitter=1, cache_key=None,
                          cache_path=None):
    """Obstacle boundary points of a full scene occupancy map in scene coords.

    Results are cached by cache_key (e.g. maze_id) in memory; if cache_path is
    given and the .npy already exists it is loaded directly from the dataset.
    """
    if cache_key is not None:
        key = (cache_key, int(dilation), int(boundary_jitter))
        if key in _OBSTACLE_POINT_CACHE:
            return _OBSTACLE_POINT_CACHE[key]

    if cache_path is not None and os.path.exists(cache_path):
        out = np.asarray(np.load(cache_path), dtype=float).reshape(-1, 2)
    else:
        occ = np.asarray(occ)
        H, W = occ.shape
        x0, x1, y0, y1 = extent
        _, pts = extract_obstacle_constraints(
            np.asarray(occ, dtype=np.uint8),
            dilation_iters=dilation, boundary_jitter=boundary_jitter)
        if len(pts) == 0:
            out = np.empty((0, 2), dtype=float)
        else:
            dx = (float(x1) - float(x0)) / W
            dy = (float(y1) - float(y0)) / H
            gx = pts[:, 0].astype(float)
            gy = pts[:, 1].astype(float)
            sx = x0 + (gx + 0.5) * dx
            sy = y0 + (gy + 0.5) * dy
            out = np.column_stack([sx, sy]).astype(float)
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, out)

    if cache_key is not None:
        _OBSTACLE_POINT_CACHE[(cache_key, int(dilation), int(boundary_jitter))] = out
    return out
