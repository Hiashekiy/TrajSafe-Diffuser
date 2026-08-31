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
