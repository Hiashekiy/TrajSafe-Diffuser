"""Principled D4RL Maze2D geometry (observation = slide-joint qpos).

This module now re-exports the canonical implementation from
`src/geometry/d4rl_coordinates.py` so all code shares one coordinate frame.
"""

import numpy as np

from .d4rl_coordinates import (
    MUJOCO_MARGIN,
    PARTICLE_RADIUS,
    distance_to_wall,
    get_wall_centers_qpos,
)


def build_occupancy_grid(maze_name, extent=(0, 4, 0, 4), global_res=20.0,
                         inflate_particle=True):
    """Build grid occupancy consistent with maze_occupancy.crop_local_patch.

    Uses a uniform pixel resolution (px per world unit).  Grid pixel centre for
    world (x,y): px = x*gres, py = y*gres (origin 0).  Returns (occ 1=wall,
    distance_field, global_res).  occ shape is (ny, nx).
    """
    x0, x1, y0, y1 = extent
    ny = int(round((y1 - y0) * global_res))
    nx = int(round((x1 - x0) * global_res))
    X = x0 + (np.arange(nx) + 0.5) / global_res
    Y = y0 + (np.arange(ny) + 0.5) / global_res
    XX, YY = np.meshgrid(X, Y, indexing="xy")  # XX:(ny,nx)=x, YY:(ny,nx)=y
    points = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    dist = distance_to_wall(points, get_wall_centers_qpos(maze_name))
    if inflate_particle:
        dist = dist - PARTICLE_RADIUS
    fields = dist.reshape(ny, nx)          # [y, x]
    occupancy = (fields <= MUJOCO_MARGIN).astype(np.uint8)
    return occupancy, fields, global_res
