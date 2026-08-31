"""04_generate_ellipse_labels.py: offline IRIS MVIE ellipse labels for every waypoint.

For each maze in cfg["mazes"], reads trajectories.npy (normalized) and writes
ellipse_params.npy [N,H,5], ellipse_Q.npy [N,H,2,2], ellipse_valid.npy [N,H]
in world (obs) coordinates.

Gate: labels are valid only when SDF(waypoint) > 0.

Only the offline IRIS MVIE solver is used -- the Neural-IRIS network is NOT used.
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.d4rl_geometry import build_occupancy_grid
from src.geometry.offline_iris_wrapper import OfflineIrisWrapper
from src.datasets.normalization import load_normalization


def process_maze(spec, iris):
    name = spec["name"]
    processed = spec["processed_dir"]
    extent = tuple(spec["extent"])
    global_res = float(spec["global_res"])
    local_res = float(spec.get("local_res", 20.0))
    patch_size = int(iris["patch_size"])
    cache_res = float(iris["cache_resolution"])
    inflate = spec.get("inflate_particle", True)

    occ, sdf, _ = build_occupancy_grid(name, extent=extent,
                                       global_res=global_res,
                                       inflate_particle=inflate)

    norm, _ = load_normalization(os.path.join(processed, "normalization.json"))
    mins = np.asarray(norm.mins[2:4], dtype=np.float64)
    maxs = np.asarray(norm.maxs[2:4], dtype=np.float64)
    eps = norm.eps

    def norm_to_world_pos(p):
        return (p + 1.0) / 2.0 * (maxs - mins + eps) + mins

    # sdf_fn for validity: waypoint must be inside free space
    def sdf_fn(p):
        px = int(round((p[0] - extent[0]) * global_res))
        py = int(round((p[1] - extent[2]) * global_res))
        if px < 0 or px >= sdf.shape[1] or py < 0 or py >= sdf.shape[0]:
            return False
        return bool(sdf[py, px] > 0.0)

    wrapper = OfflineIrisWrapper(global_res=global_res,
                                 cache_resolution=cache_res,
                                 local_res=local_res,
                                 patch_size=patch_size,
                                 sdf_fn=sdf_fn,
                                 obstacle_dilation=1,
                                 boundary_jitter=1)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(processed, split)
        traj = np.load(os.path.join(split_dir, "trajectories.npy"))
        N, H, _ = traj.shape
        eparams = np.zeros((N, H, 5), dtype=np.float32)
        eQ = np.zeros((N, H, 2, 2), dtype=np.float32)
        evalid = np.zeros((N, H), dtype=bool)
        t0 = time.time()
        for i in range(N):
            pos_norm = traj[i, :, 2:4]  # [H,2] normalized
            pos_world = norm_to_world_pos(pos_norm)
            center, Q, params, valid = wrapper.infer_positions(pos_world, occ)
            eparams[i] = params.astype(np.float32)
            eQ[i] = Q.astype(np.float32)
            evalid[i] = valid
            if (i + 1) % 200 == 0:
                print(f"[04] {name}/{split} {i+1}/{N} elapsed={time.time()-t0:.1f}s "
                      f"cache hits={wrapper.hits} misses={wrapper.misses}", flush=True)
        np.save(os.path.join(split_dir, "ellipse_params.npy"), eparams)
        np.save(os.path.join(split_dir, "ellipse_Q.npy"), eQ)
        np.save(os.path.join(split_dir, "ellipse_valid.npy"), evalid)
        valid_rate = float(evalid.mean())
        print(f"[04] {name}/{split} saved N={N} valid_rate={valid_rate:.3f} "
              f"elapsed={time.time()-t0:.1f}s hits={wrapper.hits} misses={wrapper.misses}", flush=True)

    with open(os.path.join(processed, "ellipse_labels.json"), "w", encoding="utf-8") as f:
        json.dump({"maze": name, "cache_hits": wrapper.hits,
                   "cache_misses": wrapper.misses, "cache_resolution": cache_res,
                   "local_res": local_res, "patch_size": patch_size}, f, indent=2)
    print(f"[04] finished {name} -> {processed}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--maze", default=None, help="only regenerate this maze name; all by default")
    args = ap.parse_args()
    cfg = load_config(args.config)
    iris = cfg["iris"]
    specs = cfg["mazes"]
    if args.maze:
        specs = [s for s in specs if s["name"] == args.maze]
        if not specs:
            raise SystemExit(f"unknown maze: {args.maze}")
    for spec in specs:
        process_maze(spec, iris)


if __name__ == "__main__":
    main()
