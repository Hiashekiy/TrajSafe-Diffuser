"""Sample on test conditions and visualize map + trajectories + ellipses.

Reads the mixed [0,8]^2 dataset, loads the checkpoint, and plots n
samples for one maze with SDF-based collision/clearance metrics.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.datasets.mixed_dataset import EXTENT
from src.datasets.data_io import load_maze, unnorm_positions, sdf_metrics
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import draw_traj, set_map_limits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pt")
    ap.add_argument("--maze", default="umaze", choices=["umaze", "medium", "large"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", default="outputs/test_sample_visual.png")
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
    ext = EXTENT

    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"], beta_end=cfg["diffusion"]["beta_end"]).to(device)
    geom = dict(cfg["geometry"]); geom["extent"] = list(EXTENT)
    model = Planner(cfg["model"], geom, norm).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)

    map_t = torch.as_tensor(occ, dtype=torch.float32).to(device)[None, None]
    x0, _ = sample(model, map_t, schedule, cond_t, args.n, device=device, steps=args.steps)

    with torch.no_grad():
        t0 = torch.zeros(x0.shape[0], device=device, dtype=torch.long)
        pred = model(x0, t0, map_t, cond=cond_t)
    c_norm = pred["ellipse_center"].cpu().numpy()
    r1 = pred["ellipse_radii"][..., 0].cpu().numpy()
    r2 = pred["ellipse_radii"][..., 1].cpu().numpy()
    theta = pred["ellipse_theta"].cpu().numpy()

    traj_world = unnorm_positions(x0.cpu().numpy()[..., 2:4], norm)   # [N,H,2]
    c_world = unnorm_positions(c_norm, norm)
    metrics = [sdf_metrics(traj_world[i], sdf, device=device, extent=EXTENT) for i in range(args.n)]

    ncol = 2; nrow = max(1, (args.n + 1) // 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6 * nrow))
    axes = np.array(axes).reshape(-1)
    for i in range(args.n):
        ax = axes[i]
        ax.imshow(occ, origin="lower", extent=(ext[0], ext[1], ext[2], ext[3]),
                  cmap="gray_r", alpha=0.8)
        pos = traj_world[i]
        vel = x0.cpu().numpy()[i, :, 4:6]   # normalized velocity, used as direction arrows
        draw_traj(ax, pos, velocities=vel, marker_every=0, arrow_every=0)
        for j in range(0, len(pos), 8):
            cx, cy = c_world[i][j]
            if not np.isfinite(r1[i][j] + r2[i][j]) or r1[i][j] <= 0:
                continue
            e = Ellipse((cx, cy), 2 * r1[i][j], 2 * r2[i][j], angle=np.degrees(theta[i][j]),
                        fill=False, edgecolor="tab:red", lw=1.0, alpha=0.7)
            ax.add_patch(e)
        m = metrics[i]
        ax.set_title(f"{args.maze} #{i}  coll={m['collision_rate']:.3f} clear={m['mean_clearance']:.3f}")
        ax.set_aspect("equal"); set_map_limits(ax, EXTENT); ax.legend(loc="upper right", fontsize=6)
    for k in range(args.n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Maze2D sampling ({args.maze}, [0,8]^2)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print("saved", args.out)

    report = {"maze": args.maze, "n": args.n, "steps": args.steps, "seed": args.seed,
              "metrics": metrics, "note": "SDF on mixed [0,8]^2 grid"}
    rep_path = os.path.join("outputs", f"test_sample_report_{args.maze}.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for i, m in enumerate(metrics):
        print(f"{args.maze} #{i}: {m}")
    print("report", rep_path)


if __name__ == "__main__":
    main()
