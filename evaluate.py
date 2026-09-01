"""Evaluate the zero-sum bridge planner on scene-normalized test conditions."""
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
from src.datasets.scene_dataset import load_maze_scene
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.geometry.scene_frame import sample_sdf_scene


def scene_metrics(pos_scene, sdf, device="cuda"):
    """Collision / clearance on the dense scene path (no world conversion)."""
    from src.utils.metrics import interp_trajectory
    sdf_t = torch.as_tensor(sdf, dtype=torch.float32).to(device)[None, None]
    dense = interp_trajectory(pos_scene, interp_steps=8)
    pts = torch.as_tensor(dense, dtype=torch.float32).to(device)[None]
    d = sample_sdf_scene(sdf_t, pts).cpu().numpy()[0]
    return {
        "collision_rate": float(np.mean(d <= 0.0)),
        "mean_clearance": float(np.mean(d)),
        "clearance_p05": float(np.percentile(d, 5)),
        "n_collisions": int(np.sum(d <= 0.0)),
        "n_dense_points": int(len(dense)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pt")
    ap.add_argument("--maze", default="umaze", choices=["umaze", "medium", "large"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    case_rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() and cfg["env"].get("device", "cuda") == "cuda" else "cpu"

    extent = next(m["extent"] for m in cfg["mazes"] if m["name"] == args.maze)
    frame, occ, sdf, conds = load_maze_scene(args.maze, split="test", n=args.n,
                                             rng=case_rng, extent=extent)
    cond_t = torch.as_tensor(conds, dtype=torch.float32).to(device)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"],
                             beta_end=cfg["diffusion"]["beta_end"]).to(device)
    model = Planner(cfg["model"], cfg["geometry"], None).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)

    map_t = torch.as_tensor(occ, dtype=torch.float32).to(device)[None, None]
    pos_scene, _ = sample(model, map_t, schedule, cond_t, args.n, device=device, steps=args.steps)
    pos_np = pos_scene.cpu().numpy()

    metrics = [scene_metrics(pos_np[i], sdf, device=device) for i in range(args.n)]
    agg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    agg["n"] = args.n
    out = os.path.join("outputs", f"evaluate_report_{args.maze}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"maze": args.maze, "seed": args.seed, "metrics": agg,
                   "note": "scene frame [-1,1]^2"}, f, indent=2)
    world = frame.scene_to_world_np(pos_np)
    np.save(f"outputs/evaluated_world_state_{args.maze}.npy", world.astype(np.float32))
    print(f"[evaluate] maze={args.maze} seed={args.seed} {json.dumps(agg, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
