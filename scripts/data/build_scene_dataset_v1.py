"""build_scene_dataset_v1.py: build the V1 joint-diffusion scene dataset.

V1 (docs/联合扩散.md) needs, per sample, H=128 waypoints P (scene coords) and an
ellipse label E_k for EVERY waypoint, including the start (E0):

    positions  [N,H,2]   scene waypoints, p_0 = start ... p_{H-1} = goal
    conditions [N,2,2]   scene (start, goal)
    ellipses6  [N,H,6]   e = [cx-px, cy-py, log a, log b, cos 2t, sin 2t]
    maze_id    [N]       0/1/2 -> umaze/medium/large

Policy (user decision):
  * The offline-IRIS "valid" gating (SDF(waypoint)>0 / safe / contains /
    trivial) is REMOVED: every stored solver result is used as a real label.
    (The solver parameters were always stored; the flag only suppressed them.)
  * Defensive fallback only when a row is degenerate / non-finite / zero
    (solver produced no solution): force a small centred ellipse
    e = [0,0,log r0, log r0, 1, 0], r0 = 0.02 scene units, so all H anchors
    always carry a true value.  No ellipse_valid file is produced.

Maps are copied from data/processed_scene/maps (same scene-frame 256x256 maps).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.datasets.normalization import LimitsNormalizer
from src.geometry.scene_frame import SceneFrame

DEST = "data/processed_scene_v1"
SPLITS = ["train", "val", "test"]
FALLBACK_R0 = 0.02          # scene units, ~2.5 map cells at res 256
E6_FALLBACK = np.array([0.0, 0.0, float(np.log(FALLBACK_R0)),
                        float(np.log(FALLBACK_R0)), 1.0, 0.0], dtype=np.float32)


def load_norm(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return LimitsNormalizer.from_dict(d["state"])


def to_e6(p_scene, ep_scene):
    """p_scene [H,2] (scene), ep_scene [H,5] (cx,cy,a,b,theta scene) -> [H,6]."""
    d = ep_scene[:, 0:2] - p_scene                    # [H,2] delta center
    r1 = ep_scene[:, 2]
    r2 = ep_scene[:, 3]
    th = ep_scene[:, 4]
    bad = (~np.isfinite(ep_scene).all(axis=1)) | (r1 <= 0) | (r2 <= 0)
    e6 = np.stack([d[:, 0], d[:, 1],
                   np.log(np.maximum(r1, 1e-8)),
                   np.log(np.maximum(r2, 1e-8)),
                   np.cos(2.0 * th), np.sin(2.0 * th)], axis=-1).astype(np.float32)
    e6[bad] = E6_FALLBACK
    return e6, int(bad.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--dest", default=DEST)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dest = args.dest

    src_maps = os.path.join("data", "processed_scene", "maps")
    os.makedirs(os.path.join(dest, "maps"), exist_ok=True)
    for f in os.listdir(src_maps):
        if f.endswith(".npy"):
            shutil.copy2(os.path.join(src_maps, f), os.path.join(dest, "maps", f))

    accum = {s: [] for s in SPLITS}
    stats = {}
    for mi, spec in enumerate(cfg["mazes"]):
        name = spec["name"]
        src = spec["processed_dir"]
        extent = tuple(spec["extent"])
        frame = SceneFrame(extent)
        norm = load_norm(os.path.join(src, "normalization.json"))
        mins = norm.mins[2:4]
        maxs = norm.maxs[2:4]
        eps = norm.eps

        def unnorm_positions(c0):
            orig_shape = np.asarray(c0).shape
            c = np.asarray(c0, dtype=np.float64).reshape(-1, 2)
            return ((c + 1.0) / 2.0 * (maxs - mins + eps) + mins).reshape(orig_shape)

        m_stats = {}
        for split in SPLITS:
            sd = os.path.join(src, split)
            traj_norm = np.load(os.path.join(sd, "trajectories.npy"))     # [n,H,6]
            cond_norm = np.load(os.path.join(sd, "conditions.npy"))       # [n,2,2]
            ep_world = np.load(os.path.join(sd, "ellipse_params.npy"))    # [n,H,5]
            ev_flag = np.load(os.path.join(sd, "ellipse_valid.npy"))      # [n,H] (ignored)

            world6 = norm.unnormalize(np.asarray(traj_norm, dtype=np.float64))
            pos_world = world6[..., 2:4].astype(np.float32)
            cond_world = unnorm_positions(cond_norm)
            pos_scene = frame.world_to_scene_np(pos_world)                # [n,H,2]
            cond_scene = frame.world_to_scene_np(cond_world)              # [n,2,2]

            ep_scene = ep_world.copy().astype(np.float64)
            ep_scene[..., 0:2] = frame.world_to_scene_np(ep_world[..., 0:2].reshape(-1, 2)).reshape(-1, ep_world.shape[1], 2)
            ep_scene[..., 2:4] = ep_world[..., 2:4] * frame.scale()
            ep_scene[..., 4] = ep_world[..., 4]                            # theta unchanged
            ep_scene = ep_scene.astype(np.float32)

            n, H, _ = pos_scene.shape
            e6 = np.zeros((n, H, 6), dtype=np.float32)
            n_fb = 0
            for i in range(n):
                e6[i], fb = to_e6(pos_scene[i], ep_scene[i])
                n_fb += fb
            accum[split].append({
                "pos": pos_scene,
                "cond": cond_scene.astype(np.float32),
                "e6": e6,
                "mid": np.full(n, mi, dtype=np.int64),
            })
            m_stats[split] = {"n": n, "legacy_invalid_rate":
                              float(1.0 - ev_flag.mean()), "fallback": n_fb}
            print(f"[{name}/{split}] n={n} legacy_invalid_rate="
                  f"{1.0 - ev_flag.mean():.3f} fallback_rows={n_fb}")
        stats[name] = m_stats

    for split in SPLITS:
        pos = np.concatenate([d["pos"] for d in accum[split]], 0)
        cond = np.concatenate([d["cond"] for d in accum[split]], 0)
        e6 = np.concatenate([d["e6"] for d in accum[split]], 0)
        mid = np.concatenate([d["mid"] for d in accum[split]], 0)
        out = os.path.join(dest, split)
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "positions.npy"), pos)
        np.save(os.path.join(out, "conditions.npy"), cond)
        np.save(os.path.join(out, "ellipses6.npy"), e6)
        np.save(os.path.join(out, "maze_id.npy"), mid)
        print(f"[{split}] n={len(pos)} shapes pos={pos.shape} e6={e6.shape}")

    with open(os.path.join(dest, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"mazes": cfg["mazes"], "scene_units": "[-1,1]^2",
                   "fallback_r0_scene": FALLBACK_R0, "per_maze": stats}, f,
                  indent=2, default=str)
    print("DONE V1 scene dataset at", dest)


if __name__ == "__main__":
    main()
