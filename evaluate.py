"""Evaluate the V1 joint trajectory-ellipse diffusion model on test conditions.

For each maze (umaze/medium/large) it samples --n trajectories from the test
conditions, then reports:
  * endpoint error            (hard condition should give ~0)
  * P collision fraction      SDF(P) <= 0   + clearance p05 (scene units)
  * ellipse sanity            finite/positive radii, centre inside free space
                               (SDF(c) > 0)

Usage:
  python evaluate.py --config configs/config_v1.yaml \
      --ckpt outputs/ckpt_v1/best.pt --n 40 --out outputs/evaluate_report_v1.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import load_checkpoint
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler_v1 import sample_joint
from src.models.joint import JointPlanner
from src.datasets.joint_dataset import JointDataset
from src.geometry.scene_frame import sample_sdf_scene

MAZE_NAMES = ["umaze", "medium", "large"]


def pick_conditions(ds, maze, n, rng):
    sel = np.where(ds.mid == MAZE_NAMES.index(maze))[0]
    if len(sel) <= n:
        return sel
    edges = np.linspace(0.0, 1.0, n + 2)[1:-1]
    idx = np.unique(np.round(edges * (len(sel) - 1)).astype(int))
    return sel[idx]


def e6_to_ellipse5(p, e6):
    c = p + e6[..., :2]
    a = np.exp(np.clip(e6[..., 2], -20, 20))
    b = np.exp(np.clip(e6[..., 3], -20, 20))
    th = 0.5 * np.arctan2(e6[..., 5], e6[..., 4])
    return np.stack([c[..., 0], c[..., 1], a, b, th], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v1.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt_v1/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=40, help="samples per maze")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="outputs/evaluate_report_v1.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = cfg["data"]["base"]
    model = JointPlanner(cfg["model"]).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                                "squaredcos_cap_v2")).to(device)

    ds = JointDataset(os.path.join(base, args.split))
    rng = np.random.default_rng(args.seed)
    report = {"device": device, "split": args.split, "seed": args.seed,
              "checkpoint": args.ckpt, "mazes": {}}
    t_all = time.time()
    for mi, maze in enumerate(MAZE_NAMES):
        sel = pick_conditions(ds, maze, args.n, rng)
        cond = torch.as_tensor(ds.cond[sel], dtype=torch.float32).to(device)
        map_t = ds.maps[mi].to(device).expand(len(sel), -1, -1, -1).contiguous()
        sdf_t = ds.sdfs[mi].to(device).expand(len(sel), -1, -1, -1).contiguous()

        t0 = time.time()
        P, E6 = sample_joint(model, schedule, cond, map_t, device,
                             steps=args.steps, seed=args.seed + mi * 101)
        dt = time.time() - t0
        P = P.cpu().numpy()
        E6 = E6.cpu().numpy()

        # endpoint error
        end_err = float(np.max(np.abs(P[:, 0] - ds.cond[sel, 0])) + 0)
        goal_err = float(np.max(np.abs(P[:, -1] - ds.cond[sel, 1])))
        # P collision / clearance via bilinear scene SDF
        with torch.no_grad():
            sdf_vals = sample_sdf_scene(sdf_t, torch.as_tensor(P, dtype=torch.float32).to(device))
        sdf_vals = sdf_vals.cpu().numpy()
        coll = float((sdf_vals <= 0.0).mean())
        clear_p05 = float(np.percentile(sdf_vals, 5))

        # ellipse sanity
        E5 = e6_to_ellipse5(P, E6)
        a, b = E5[..., 2], E5[..., 3]
        sane = float((np.isfinite(a) & (a > 0) & (a <= 2.0) &
                      np.isfinite(b) & (b > 0) & (b <= 2.0)).mean())
        c = E5[..., :2]
        with torch.no_grad():
            sdf_c = sample_sdf_scene(sdf_t, torch.as_tensor(c, dtype=torch.float32).to(device))
        sdf_c = sdf_c.cpu().numpy()
        center_free = float((sdf_c > 0.0).mean())

        m = {"n": int(len(sel)),
             "time_s": round(dt, 2),
             "endpoint_err": round(end_err, 6),
             "goal_err": round(goal_err, 6),
             "collision_frac": round(coll, 4),
             "clearance_p05": round(clear_p05, 4),
             "ellipse_sane_frac": round(sane, 4),
             "ellipse_center_free_frac": round(center_free, 4)}
        report["mazes"][maze] = m
        print(f"[{maze}] n={m['n']} dt={dt:.1f}s end_err={m['endpoint_err']:.2e} "
              f"coll={m['collision_frac']:.4f} clear_p05={m['clearance_p05']:.3f} "
              f"e_sane={m['ellipse_sane_frac']:.3f} e_center_free={m['ellipse_center_free_frac']:.3f}",
              flush=True)

    report["total_time_s"] = round(time.time() - t_all, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("saved:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
