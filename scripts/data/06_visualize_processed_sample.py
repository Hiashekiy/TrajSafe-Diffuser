"""06_visualize_processed_sample.py: visualize map + waypoints + ellipses."""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.d4rl_geometry import build_occupancy_grid
from src.datasets.normalization import load_normalization
from src.utils.visualization import plot_map_traj_ellipses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="outputs/visualize_processed_sample.png")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    geom = cfg["geometry"]
    ext = tuple(geom["extent"])
    res = float(geom["global_res"])
    processed = data["processed_dir"]
    occ, sdf, _ = build_occupancy_grid(data["maze"], extent=ext, global_res=res,
                                       inflate_particle=geom.get("inflate_particle", True))
    norm, _ = load_normalization(os.path.join(processed, "normalization.json"))
    split_dir = os.path.join(processed, args.split)
    traj = np.load(os.path.join(split_dir, "trajectories.npy"))[args.index]
    eparams = np.load(os.path.join(split_dir, "ellipse_params.npy"))[args.index]
    mins = np.asarray(norm.mins[2:4], dtype=np.float64)
    maxs = np.asarray(norm.maxs[2:4], dtype=np.float64)
    pos = (traj[:, 2:4] + 1.0) / 2.0 * (maxs - mins + norm.eps) + mins
    plot_map_traj_ellipses(occ, ext, pos, eparams, args.out)
    print(f"[06] saved {args.out}")


if __name__ == "__main__":
    main()
