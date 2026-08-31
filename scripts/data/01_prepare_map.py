"""01_prepare_map.py: rebuild the D4RL Maze2D occupancy map offline and validate it.

Outputs to processed_dir/map/:
    occupancy.npy, obstacle_mask.npy, sdf.npy, map_xy.npy, meta.json
Also writes outputs/data_checks/map_validation.json and FAILS if the raw
observation collision rate is > 0.01.
"""

import json
import os
import sys

import numpy as np
import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.d4rl_coordinates import (
    MUJOCO_MARGIN, get_wall_centers_qpos, particle_clearance,
)
from src.geometry.d4rl_geometry import build_occupancy_grid


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    geom = cfg["geometry"]
    maze = data["maze"]
    extent = tuple(geom["extent"])
    global_res = float(geom["global_res"])
    processed = data["processed_dir"]

    occ, sdf, _ = build_occupancy_grid(maze, extent=extent, global_res=global_res,
                                       inflate_particle=geom.get("inflate_particle", True))
    ny, nx = occ.shape
    x0, x1, y0, y1 = extent
    X = x0 + (np.arange(nx) + 0.5) / global_res
    Y = y0 + (np.arange(ny) + 0.5) / global_res
    XX, YY = np.meshgrid(X, Y, indexing="xy")
    map_xy = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=-1)

    out_dir = os.path.join(processed, "map")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "occupancy.npy"), occ)
    np.save(os.path.join(out_dir, "obstacle_mask.npy"), occ)
    np.save(os.path.join(out_dir, "sdf.npy"), sdf)
    np.save(os.path.join(out_dir, "map_xy.npy"), map_xy)

    meta = {
        "maze": maze, "extent": extent, "global_res": global_res,
        "shape": [int(ny), int(nx)], "inflate_particle": geom.get("inflate_particle", True),
        "coord_module": geom.get("coord_module", "src/geometry/d4rl_coordinates.py"),
        "note": "evaluated against reconstructed D4RL geometry",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ---- Gate: raw observation collision sanity check ----
    wall_centers = get_wall_centers_qpos(maze)
    with h5py.File(data["raw_hdf5"], "r") as f:
        obs = f["observations"][:: max(1, len(f["observations"]) // 20000)][:, :2].astype(np.float64)
    clear = particle_clearance(obs, wall_centers)
    collision_rate = float(np.mean(clear <= MUJOCO_MARGIN))
    validation = {
        "observation_collision_rate": collision_rate,
        "mean_clearance": float(np.mean(clear)),
        "clearance_p05": float(np.percentile(clear, 5)),
        "samples": int(len(obs)),
        "gate": "pass" if collision_rate <= 0.01 else "FAIL",
    }
    os.makedirs("outputs/data_checks", exist_ok=True)
    with open("outputs/data_checks/map_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"[01] map saved to {out_dir}  shape={occ.shape}")
    print(f"[01] validation: {json.dumps(validation, ensure_ascii=False)}")
    if collision_rate > 0.01:
        print("[01] GATE FAILED: observation collision rate > 0.01")
        sys.exit(1)


if __name__ == "__main__":
    main()
