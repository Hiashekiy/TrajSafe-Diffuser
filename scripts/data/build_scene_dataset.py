"""Build the scene-normalized dataset for the zero-sum bridge planner.

Reads the per-maze processed data (data/processed/maze2d_*) and writes a new
dataset at data/processed_scene/ where:
  - each maze map is 256x256 over the scene frame [-1,1]^2,
  - trajectories are positions only [H,2] in scene coords,
  - conditions are (start, goal) in scene coords [2,2],
  - ellipses are kept for arriving waypoints only (E_1..E_{H-1}).

Usage:
  python scripts/data/build_scene_dataset.py --config configs/config.yaml
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import load_config
from src.datasets.normalization import LimitsNormalizer
from src.geometry.scene_frame import (SceneFrame, build_scene_occupancy,
                                      build_scene_sdf)

DEST = "data/processed_scene"
RES = 256
SPLITS = ["train", "val", "test"]
MAZE_INDEX = {}


def load_norm(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    n = LimitsNormalizer.from_dict(d["state"])
    return n


def unnorm_positions(cond_norm, norm):
    c = np.asarray(cond_norm, dtype=np.float64).reshape(-1, 2)
    mins = norm.mins[2:4]; maxs = norm.maxs[2:4]; eps = norm.eps
    return ((c + 1.0) / 2.0 * (maxs - mins + eps) + mins).reshape(cond_norm.shape)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    os.makedirs(DEST, exist_ok=True)
    os.makedirs(os.path.join(DEST, "maps"), exist_ok=True)

    splits_accum = {s: [] for s in SPLITS}
    for mi, spec in enumerate(cfg["mazes"]):
        name = spec["name"]
        src = spec["processed_dir"]
        extent = tuple(spec["extent"])
        frame = SceneFrame(extent)

        # ---- maps ----
        occ_orig = np.load(os.path.join(src, "map", "occupancy.npy"))
        gres = float(spec["global_res"])
        meta = None
        sc_occ = build_scene_occupancy(frame, occ_orig, extent, gres, RES)
        sc_sdf = build_scene_sdf(sc_occ)
        np.save(os.path.join(DEST, "maps", f"{name}.npy"), sc_occ.astype(np.float32))
        np.save(os.path.join(DEST, "maps", f"{name}_sdf.npy"), sc_sdf.astype(np.float32))

        norm = load_norm(os.path.join(src, "normalization.json"))
        for split in SPLITS:
            sd = os.path.join(src, split)
            traj_norm = np.load(os.path.join(sd, "trajectories.npy"))   # [n,H,6]
            cond_norm = np.load(os.path.join(sd, "conditions.npy"))     # [n,2,2]
            ep = np.load(os.path.join(sd, "ellipse_params.npy"))        # [n,H,4] world
            eq = np.load(os.path.join(sd, "ellipse_Q.npy"))             # [n,H,2,2] world
            ev = np.load(os.path.join(sd, "ellipse_valid.npy"))         # [n,H]

            world6 = norm.unnormalize(np.asarray(traj_norm, dtype=np.float64))  # [n,H,6]
            pos_world = world6[..., 2:4]                                # [n,H,2] world
            pos_scene = frame.world_to_scene_np(pos_world)              # [n,H,2] scene

            cond_world = unnorm_positions(cond_norm, norm)              # [n,2,2] world
            cond_scene = frame.world_to_scene_np(cond_world)            # [n,2,2] scene

            # ellipse: drop E0 (start), keep E1..E_{H-1}; center/radii to scene.
            # Source layout is [cx, cy, r1, r2, theta]; theta is NOT scaled/rotated.
            ep_scene = ep[:, 1:].copy().astype(np.float64)              # [n,H-1,5]
            ep_scene[..., 0:2] = frame.world_to_scene_np(ep[:, 1:, 0:2])
            ep_scene[..., 2:4] = ep[:, 1:, 2:4] * frame.scale()
            ep_scene[..., 4] = ep[:, 1:, 4]                             # theta unchanged
            eq_scene = eq[:, 1:] / (frame.scale() ** 2)                 # [n,H-1,2,2]
            ev_scene = ev[:, 1:].astype(np.bool_)

            splits_accum[split].append({
                "pos": pos_scene.astype(np.float32),
                "cond": cond_scene.astype(np.float32),
                "ep": ep_scene.astype(np.float32),
                "eq": eq_scene.astype(np.float32),
                "ev": ev_scene,
                "mid": np.full(len(pos_scene), mi, dtype=np.int64),
            })
            print(f"[{name}/{split}] n={len(pos_scene)} s={frame.scale():.4f}")

    for split in SPLITS:
        pos = np.concatenate([d["pos"] for d in splits_accum[split]], 0)
        cond = np.concatenate([d["cond"] for d in splits_accum[split]], 0)
        ep = np.concatenate([d["ep"] for d in splits_accum[split]], 0)
        eq = np.concatenate([d["eq"] for d in splits_accum[split]], 0)
        ev = np.concatenate([d["ev"] for d in splits_accum[split]], 0)
        mid = np.concatenate([d["mid"] for d in splits_accum[split]], 0)
        out = os.path.join(DEST, split)
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "positions.npy"), pos)
        np.save(os.path.join(out, "conditions.npy"), cond)
        np.save(os.path.join(out, "ellipse_params.npy"), ep)
        np.save(os.path.join(out, "ellipse_Q.npy"), eq)
        np.save(os.path.join(out, "ellipse_valid.npy"), ev)
        np.save(os.path.join(out, "maze_id.npy"), mid)
        print(f"[{split}] n={len(pos)} sizes={pos.shape} ep={ep.shape}")

    print("DONE scene dataset at", DEST)


if __name__ == "__main__":
    main()
