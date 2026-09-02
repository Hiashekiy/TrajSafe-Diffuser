"""Precompute scene obstacle boundary points and store them in the dataset.

For each scene maze (umaze / medium / large) this extracts the obstacle boundary
points once and saves them as:

    data/processed_scene/maps/{maze}_obstacle_points.npy

AL / convex-region construction can then load these by maze_id instead of
re-running the dilation / boundary extraction at runtime.

Usage:
    python scripts/data/08_precompute_scene_obstacles.py
"""

import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.losses.al_loss import _scene_obstacle_points, _MAZE_NAMES


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/config.yaml"), encoding="utf-8"))
    scfg = cfg.get("segment_safety", {})
    extent = (-1.0, 1.0, -1.0, 1.0)
    dilation = int(scfg.get("dilation", 1))
    boundary_jitter = int(scfg.get("boundary_jitter", 1))
    maps_dir = scfg.get("maps_dir", os.path.join(ROOT, "data/processed_scene/maps"))

    if not os.path.isabs(maps_dir):
        maps_dir = os.path.join(ROOT, maps_dir)
    os.makedirs(maps_dir, exist_ok=True)

    for maze in _MAZE_NAMES:
        map_path = os.path.join(maps_dir, f"{maze}.npy")
        if not os.path.exists(map_path):
            print(f"[08] missing map {map_path}")
            continue
        occ = np.load(map_path)
        pts = _scene_obstacle_points(occ, extent=extent, dilation=dilation,
                                     boundary_jitter=boundary_jitter,
                                     cache_key=None)
        out = os.path.join(maps_dir, f"{maze}_obstacle_points.npy")
        np.save(out, pts)
        print(f"[08] saved {out}  ({len(pts)} obstacle points)")


if __name__ == "__main__":
    main()
