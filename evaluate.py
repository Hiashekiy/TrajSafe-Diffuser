"""Evaluate the trained mixed-scene planner on test conditions.

Reads the mixed [0,8]^2 dataset (data/processed/mixed), samples with the
checkpoint (default outputs/ckpt/best.pt), and reports collision /
clearance metrics computed on the shared SDF field.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.datasets.mixed_dataset import EXTENT
from src.datasets.data_io import load_maze, sdf_metrics
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pt")
    ap.add_argument("--maze", default="umaze", choices=["umaze", "medium", "large"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed; omit for a different random case set / sampling noise each run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    case_rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() and cfg["env"].get("device", "cuda") == "cuda" else "cpu"

    norm, occ, sdf, conds = load_maze(args.maze, split="test", n=args.n, rng=case_rng)
    cond_t = torch.as_tensor(conds, dtype=torch.float32).to(device)

    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"], beta_end=cfg["diffusion"]["beta_end"]).to(device)
    geom = dict(cfg["geometry"]); geom["extent"] = list(EXTENT)
    model = Planner(cfg["model"], geom, norm).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)

    map_t = torch.as_tensor(occ, dtype=torch.float32).to(device)[None, None]
    x0, _ = sample(model, map_t, schedule, cond_t, args.n, device=device, steps=args.steps)

    mins = np.asarray(norm.mins, dtype=np.float64)
    maxs = np.asarray(norm.maxs, dtype=np.float64)
    eps = norm.eps
    world_state = (x0.cpu().numpy() + 1.0) / 2.0 * (maxs - mins + eps) + mins
    world = world_state[:, :, 2:4]  # [N,H,2] positions in [0,8]^2

    all_metrics = [sdf_metrics(world[i], sdf, device=device, extent=EXTENT) for i in range(args.n)]
    agg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    agg["n"] = args.n

    out = os.path.join("outputs", f"evaluate_report_{args.maze}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"maze": args.maze, "seed": args.seed, "metrics": agg,
                  "note": "SDF on mixed [0,8]^2 grid"}, f, indent=2)
    np.save(f"outputs/evaluated_world_state_{args.maze}.npy", world.astype(np.float32))
    print(f"[evaluate] maze={args.maze} seed={args.seed} {json.dumps(agg, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
