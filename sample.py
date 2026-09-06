"""Sample with the V1 joint trajectory-ellipse diffusion model.

docs/联合扩散.md #28: P_T,E_T ~ N(0,I); both denoised in lockstep with the shared
schedule; after every reverse step the trajectory endpoints are overwritten with
the exact start/goal (hard inpainting condition).

Outputs an npz (scene P [n,H,2], E6 [n,H,6], ellipse5 [n,H,5]) and a PNG grid
(map + trajectory + ellipses) into outputs/.

Usage:
  python sample.py --config configs/config_v1.yaml --ckpt outputs/ckpt_v1/best.pt \
      --maze umaze --n 8 --out outputs/v1_sample_umaze
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import load_checkpoint
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler_v1 import sample_joint
from src.models.joint import JointPlanner
from src.datasets.joint_dataset import JointDataset

MAZE_NAMES = ["umaze", "medium", "large"]


def e6_to_ellipse5(p, e6):
    """scene P [H,2] + e6 [H,6] -> [H,5] (cx,cy,a,b,theta)."""
    c = p + e6[..., :2]
    a = np.exp(np.clip(e6[..., 2], -20, 20))
    b = np.exp(np.clip(e6[..., 3], -20, 20))
    th = 0.5 * np.arctan2(e6[..., 5], e6[..., 4])
    return np.stack([c[..., 0], c[..., 1], a, b, th], axis=-1)


def pick_conditions(ds, maze, n, rng):
    sel = np.where(ds.mid == MAZE_NAMES.index(maze))[0]
    if len(sel) <= n:
        return sel
    edges = np.linspace(0.0, 1.0, n + 2)[1:-1]
    idx = np.unique(np.round(edges * (len(sel) - 1)).astype(int))
    return sel[idx]


def plot_results(maze, occ, cond, P, E5, out_png, ncols=4):
    occ = np.asarray(occ)
    n = len(P)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r")
        p = P[i]
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:blue", lw=1.3, zorder=3)
        ax.scatter(*cond[i, 0], marker="o", s=30, color="lime", zorder=5)
        ax.scatter(*cond[i, 1], marker="*", s=110, color="red", zorder=5)
        for k in range(0, p.shape[0]):
            cx, cy, a, b, th = E5[i, k]
            if not np.isfinite(a + b) or a <= 0 or b <= 0:
                continue
            e = Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(th),
                        fill=False, edgecolor="tab:red", lw=0.6, alpha=0.7,
                        zorder=4)
            ax.add_patch(e)
        ax.set_title(f"{maze} #{i}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"V1 joint diffusion samples — {maze} (scene [-1,1]^2)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v1.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint (default <config.train.ckpt_dir>/best.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--maze", default="umaze", choices=MAZE_NAMES)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/v1_sample")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = cfg["data"]["base"]

    model = JointPlanner(cfg["model"]).to(device)
    ckpt = args.ckpt or os.path.join(cfg["train"]["ckpt_dir"], "best.pt")
    load_checkpoint(ckpt, model, map_location=device)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                                "squaredcos_cap_v2")).to(device)

    split_dir = os.path.join(base, args.split)
    ds = JointDataset(split_dir)
    rng = np.random.default_rng(args.seed)
    sel = pick_conditions(ds, args.maze, args.n, rng)
    print(f"[sample] maze={args.maze} n={len(sel)} ids={sel.tolist()} device={device}")

    cond = torch.as_tensor(ds.cond[sel], dtype=torch.float32).to(device)
    mi = MAZE_NAMES.index(args.maze)
    occ_map = ds.maps[mi].to(device)                # [1,1,256,256]
    map_t = occ_map.expand(len(sel), -1, -1, -1).contiguous()

    P, E6 = sample_joint(model, schedule, cond, map_t, device,
                         steps=args.steps, seed=args.seed)
    P = P.cpu().numpy()
    E6 = E6.cpu().numpy()
    E5 = e6_to_ellipse5(P, E6)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out + ".npz", P=P, E6=E6, E5=E5,
                        cond=ds.cond[sel], ids=np.asarray(sel),
                        maze=np.asarray(args.maze))
    occ_full = np.load(os.path.join(base, "maps", f"{args.maze}.npy"))
    plot_results(args.maze, occ_full, ds.cond[sel], P, E5, args.out + ".png")
    print("saved:", os.path.abspath(args.out + ".npz"),
          os.path.abspath(args.out + ".png"))


if __name__ == "__main__":
    main()
