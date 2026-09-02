"""Offline IRIS wrapper: generate world-frame ellipse labels with the real IRIS
MVIE solver (not the Neural-IRIS network)."""

import numpy as np

from .maze_occupancy import crop_local_patch
from .d4rl_coordinates import MUJOCO_MARGIN
from .iris_solver import (
    extract_obstacle_constraints, solve_iris_offline, is_ellipse_safe,
    is_trivial_ellipse, is_anchor_inside_ellipse, P_to_ellipse_params,
)
from .ellipse_utils import patch_Q_to_world
from .convex_region import generate_convex_region, halfspaces_to_vertices


def _cached_key(pos, cache_resolution):
    return (round(float(pos[0]) / cache_resolution), round(float(pos[1]) / cache_resolution))


class OfflineIrisWrapper:
    def __init__(self, global_res=20.0, cache_resolution=0.05, local_res=20.0,
                 patch_size=128, sdf_fn=None, obstacle_dilation=1, boundary_jitter=1):
        self.global_res = global_res
        self.cache_resolution = cache_resolution
        self.local_res = local_res
        self.patch_size = patch_size
        self.sdf_fn = sdf_fn
        self.obstacle_dilation = obstacle_dilation
        self.boundary_jitter = boundary_jitter
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def infer_position(self, pos, global_occ):
        """Run offline IRIS at one obs-frame position; returns dict or None."""
        pos = np.asarray(pos, dtype=float).reshape(2)
        key = _cached_key(pos, self.cache_resolution)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1

        patch, patch_to_world, world_to_patch, _ = crop_local_patch(
            global_occ, pos, global_res=self.global_res, local_res=self.local_res,
            patch_size=self.patch_size)
        obstacle_mask, obs_points = extract_obstacle_constraints(
            patch, dilation_iters=self.obstacle_dilation,
            boundary_jitter=self.boundary_jitter)

        P, c = solve_iris_offline(obs_points, bounds=(0, self.patch_size, 0, self.patch_size),
                                  seed=(self.patch_size / 2.0, self.patch_size / 2.0),
                                  max_iters=15, K_bins=32)
        if P is None or c is None:
            out = None
        else:
            anchor = np.array([self.patch_size / 2.0, self.patch_size / 2.0], dtype=float)
            safe = is_ellipse_safe(P, c, obstacle_mask)
            trivial = is_trivial_ellipse(P, c, self.patch_size)
            contains = is_anchor_inside_ellipse(P, c, anchor)
            valid_sdf = self.sdf_fn(pos) if self.sdf_fn is not None else True
            valid = bool(safe and (not trivial) and contains and valid_sdf)
            center_pix, Q_pix, params_pix = P_to_ellipse_params(P, c)
            center_world = np.asarray(patch_to_world(center_pix[0], center_pix[1]), dtype=float)
            Q_world = patch_Q_to_world(Q_pix, self.local_res)
            r1 = params_pix[2] / self.local_res
            r2 = params_pix[3] / self.local_res
            theta = params_pix[4]
            out = {
                "center": center_world,
                "Q": Q_world,
                "params": np.array([center_world[0], center_world[1], r1, r2, theta]),
                "valid": valid,
                "A": None,
                "b": None,
            }
        self.cache[key] = out
        return out

    def infer_positions(self, positions, global_occ):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        out_center = np.zeros((len(positions), 2), dtype=float)
        out_Q = np.zeros((len(positions), 2, 2), dtype=float)
        out_params = np.zeros((len(positions), 5), dtype=float)
        out_valid = np.zeros(len(positions), dtype=bool)
        for i, p in enumerate(positions):
            r = self.infer_position(p, global_occ)
            if r is None:
                continue
            out_center[i] = r["center"]
            out_Q[i] = r["Q"]
            out_params[i] = r["params"]
            out_valid[i] = r["valid"]
        return out_center, out_Q, out_params, out_valid


def infer_convex_region_from_scene_occupancy(
    occ, center_scene, r1, r2, theta,
    extent=(-1.0, 1.0, -1.0, 1.0),
    window_half=0.25,
    safety_margin=0.0,
    include_boundary=True,
    dilation=1,
    boundary_jitter=1,
):
    """Generate convex region (A, b, vertices) from a scene ellipse + occupancy.

    This works directly on the scene-normalized occupancy map (maze maps at
    data/processed_scene/maps/{maze}.npy, 256x256 over scene [-1,1]^2) and the
    scene-coordinate ellipse produced by the model, so it can be used for the
    scene-frame visualizers without converting to the world/obs frame.

    Parameters
    ----------
    occ : [H,W] occupancy array (1 = wall, 0 = free) over scene extent.
    center_scene : (cx, cy) ellipse centre in scene coords.
    r1, r2, theta : ellipse semi-axes (scene units) and orientation (radians).
    extent : (x0, x1, y0, y1) scene-coordinate extent (default [-1,1]^2).
    window_half : half-size of the local crop around the ellipse in scene units.
    safety_margin : inward offset for the halfspaces (scene units).

    Returns
    -------
    (A, b, vertices) with A x <= b in scene coords; vertices may be None.
    """
    occ = np.asarray(occ)
    H, W = occ.shape
    x0, x1, y0, y1 = extent
    cx, cy = float(center_scene[0]), float(center_scene[1])
    dx = (float(x1) - float(x0)) / W
    dy = (float(y1) - float(y0)) / H

    pxC = (cx - x0) / (x1 - x0) * W - 0.5
    pyC = (cy - y0) / (y1 - y0) * H - 0.5
    half_x = int(np.ceil(window_half / dx))
    half_y = int(np.ceil(window_half / dy))
    gx0 = max(0, int(np.round(pxC)) - half_x)
    gx1 = min(W, int(np.round(pxC)) + half_x + 1)
    gy0 = max(0, int(np.round(pyC)) - half_y)
    gy1 = min(H, int(np.round(pyC)) + half_y + 1)
    if gx1 <= gx0 or gy1 <= gy0:
        return None, None, None

    sub = occ[gy0:gy1, gx0:gx1]
    _, obs_points_pix = extract_obstacle_constraints(
        sub, dilation_iters=dilation, boundary_jitter=boundary_jitter)
    if len(obs_points_pix) == 0:
        obs_scene = np.empty((0, 2), dtype=float)
    else:
        gx = obs_points_pix[:, 0] + gx0
        gy = obs_points_pix[:, 1] + gy0
        sx = x0 + (gx + 0.5) * dx
        sy = y0 + (gy + 0.5) * dy
        obs_scene = np.column_stack([sx, sy]).astype(float)

    theta = float(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=float)
    P_scene = R @ np.diag([float(r1), float(r2)]) @ R.T
    c_scene = np.array([cx, cy], dtype=float)

    bx0 = x0 + gx0 * dx
    bx1 = x0 + gx1 * dx
    by0 = y0 + gy0 * dy
    by1 = y0 + gy1 * dy
    A, b = generate_convex_region(
        P_scene, c_scene, obs_scene,
        bounds=(bx0, bx1, by0, by1),
        safety_margin=safety_margin,
        include_boundary=include_boundary,
    )
    vertices = halfspaces_to_vertices(A, b, c_scene)
    return A, b, vertices
