"""Canonical Maze2D coordinate conventions.

This module is the single source of truth for coordinate frames so that the
geometry helpers (d4rl_geometry, maze_occupancy) and the
data-preparation scripts all agree.

Frames
------
* "obs frame" (a.k.a. qpos frame): what D4RL stores in observations[:, :2].
  The data-driven occupancy grid is built in this frame.
* "world frame": particle body frame.  The MuJoCo particle has a fixed body
  offset pos=[1.2, 1.2, 0], so
        p_world = p_obs + [1.2, 1.2].
* A wall tile at maze (row, col) has a MuJoCo box centred at world
  [row + 1, col + 1] with half-extent 0.5, i.e. obs-frame centre
  [row - 0.2, col - 0.2].
"""

import numpy as np

PARTICLE_BODY_OFFSET = np.array([1.2, 1.2], dtype=np.float64)
WALL_HALF_SIZE = 0.5
PARTICLE_RADIUS = 0.1
MUJOCO_MARGIN = 0.002


def maze_rows(maze_name):
    """Non-empty lines of the maze layout string."""
    from .maze2d_env import MAZES
    return [l for l in MAZES[maze_name].split("\n") if l != ""]


def get_wall_centers_qpos(maze_name):
    """[M,2] wall-centre positions in the observation/qpos frame."""
    rows = maze_rows(maze_name)
    centers = []
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "#":
                wall_world = np.array([r + 1.0, c + 1.0], dtype=np.float64)
                centers.append(wall_world - PARTICLE_BODY_OFFSET)
    return np.asarray(centers, dtype=np.float64)


def box_sdf(points, centers, half_size=WALL_HALF_SIZE):
    """AABB SDF: positive=outside. points [N,2], centers [M,2]."""
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    q = np.abs(points[:, None, :] - centers[None, :, :]) - half_size
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
    return outside + inside


def distance_to_wall(points, wall_centers):
    sdf = box_sdf(points, wall_centers)
    return sdf.min(axis=1)


def particle_clearance(points, wall_centers):
    return distance_to_wall(points, wall_centers) - PARTICLE_RADIUS


