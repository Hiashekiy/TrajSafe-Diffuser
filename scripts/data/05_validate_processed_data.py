"""05_validate_processed_data.py: sanity check of all processed arrays."""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.d4rl_geometry import build_occupancy_grid
from src.datasets.normalization import load_normalization, round_trip_error


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    geom = cfg["geometry"]
    extent = tuple(geom["extent"])
    global_res = float(geom["global_res"])
    processed = data["processed_dir"]

    norm, _ = load_normalization(os.path.join(processed, "normalization.json"))
    occ, sdf, _ = build_occupancy_grid(data["maze"], extent=extent, global_res=global_res,
                                       inflate_particle=geom.get("inflate_particle", True))

    report = {}
    for split in ["train", "val", "test"]:
        d = os.path.join(processed, split)
        traj = np.load(os.path.join(d, "trajectories.npy"))
        cond = np.load(os.path.join(d, "conditions.npy"))
        eparams = np.load(os.path.join(d, "ellipse_params.npy"))
        eQ = np.load(os.path.join(d, "ellipse_Q.npy"))
        ev = np.load(os.path.join(d, "ellipse_valid.npy"))
        r = {}
        r["shape"] = list(traj.shape)
        r["nan_inf"] = bool(np.any(~np.isfinite(traj)))
        # round-trip error on a sample (reconstruct normalized values)
        sample = traj[:2000]
        rebuilt = norm.normalize(norm.unnormalize(sample))
        r["round_trip_max"] = float(np.max(np.abs(rebuilt - sample)))
        # continuity in world positions
        mins = np.asarray(norm.mins[2:4], dtype=np.float64)
        maxs = np.asarray(norm.maxs[2:4], dtype=np.float64)
        eps = norm.eps
        pos = (traj[:, :, 2:4] + 1.0) / 2.0 * (maxs - mins + eps) + mins
        jumps = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
        r["max_jump"] = float(jumps.max()) if jumps.size else 0.0
        r["start_goal_dist_mean"] = float(np.mean(np.linalg.norm(pos[:, 0] - pos[:, -1], axis=-1)))
        r["ellipse_valid_rate"] = float(np.mean(ev))
        # fraction of valid centers in free space
        valid_idx = np.where(ev.reshape(-1))[0]
        if valid_idx.size:
            centers = eparams.reshape(-1, 5)[valid_idx][:, :2]
            px = np.clip(np.round((centers[:, 0] - extent[0]) * global_res).astype(int), 0, sdf.shape[1] - 1)
            py = np.clip(np.round((centers[:, 1] - extent[2]) * global_res).astype(int), 0, sdf.shape[0] - 1)
            r["valid_center_free_rate"] = float(np.mean(sdf[py, px] > 0))
        else:
            r["valid_center_free_rate"] = 0.0
        report[split] = r
        print(f"[05] {split}: {json.dumps(r, ensure_ascii=False)}")

    out = os.path.join("outputs", "processed_dataset_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
