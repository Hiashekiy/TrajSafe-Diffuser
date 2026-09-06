"""Plot an overview of the scene dataset (map + GT trajectory + GT ellipses).

Reads data/processed_scene/<split> (the dataset train.py currently trains on):
  positions.npy       [N,H,2]    scene coords ([-1,1]^2)
  conditions.npy      [N,2,2]    scene (start, goal)
  ellipse_params.npy  [N,H-1,5]  scene (cx, cy, r1, r2, theta)
  ellipse_valid.npy   [N,H-1]    bool
  maze_id.npy         [N]        0/1/2 -> umaze/medium/large
and renders one panel per sample: map (256x256 scene occupancy) under
a time-coloured GT trajectory with the GT ellipses overlaid.

Usage:
  python scripts/plot/plot_scene_dataset_overview.py
  python scripts/plot/plot_scene_dataset_overview.py --split val --per-maze 3 \
      --out outputs/dataset_scene_overview_val.png
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from src.utils.visualization import draw_traj

BASE = os.path.join("data", "processed_scene")
MAZE_NAMES = ["umaze", "medium", "large"]
MAPS_DIR = os.path.join(BASE, "maps")
RES = 256  # scene map grid size -> extent [-1,1]^2


def pick_sample_ids(mids, per_maze, maze_id):
    """Deterministic, spread-out sample indices of one maze in its split."""
    sel = np.where(mids == maze_id)[0]
    if len(sel) <= per_maze:
        return list(sel)
    edges = np.linspace(0.0, 1.0, per_maze + 2)[1:-1]
    idx = np.unique(np.round(edges * (len(sel) - 1)).astype(int))
    return [int(sel[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--per-maze", type=int, default=4,
                    help="samples per maze (default 4)")
    ap.add_argument("--ellipse-stride", type=int, default=1,
                    help="draw every Nth ellipse anchor (default 1 = all)")
    ap.add_argument("--ellipse-lw", type=float, default=0.6)
    ap.add_argument("--ellipse-alpha", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0, help="unused; sampling is spread-based")
    ap.add_argument("--out", default="outputs/dataset_scene_overview.png")
    args = ap.parse_args()

    split_dir = os.path.join(BASE, args.split)
    pos = np.load(os.path.join(split_dir, "positions.npy"))          # [N,H,2]
    cond = np.load(os.path.join(split_dir, "conditions.npy"))        # [N,2,2]
    ep = np.load(os.path.join(split_dir, "ellipse_params.npy"))      # [N,H-1,5]
    ev = np.load(os.path.join(split_dir, "ellipse_valid.npy"))       # [N,H-1]
    mids = np.load(os.path.join(split_dir, "maze_id.npy"))           # [N]
    print(f"[data] {args.split}: {len(pos)} samples, H={pos.shape[1]} "
          f"({np.bincount(mids, minlength=3).tolist()} per umaze/medium/large)")

    maps = {m: np.load(os.path.join(MAPS_DIR, f"{m}.npy")) for m in MAZE_NAMES}

    nrow, ncol = len(MAZE_NAMES), args.per_maze
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.1 * ncol, 3.1 * nrow + 0.6))
    axes = np.asarray(axes).reshape(nrow, ncol)
    extent = (-1.0, 1.0, -1.0, 1.0)

    row_meta = []          # (maze, n_samples)
    sampled_ids = {}       # maze -> [global sample ids]
    for r, maze in enumerate(MAZE_NAMES):
        ids = pick_sample_ids(mids, args.per_maze, r)
        sampled_ids[maze] = ids
        row_meta.append((maze, len(ids)))
        occ = maps[maze]
        for c, sid in enumerate(ids):
            ax = axes[r, c]
            ax.imshow(occ, origin="lower", extent=extent, cmap="gray_r",
                      vmin=0, vmax=1)
            ax.set_xlim(*extent[:2]); ax.set_ylim(*extent[2:])
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

            # GT trajectory, time-coloured (start -> goal)
            draw_traj(ax, pos[sid], marker_every=0, arrow_every=0, lw=1.4)

            # GT ellipses (anchor k sits at trajectory point k+1)
            stride = max(1, int(args.ellipse_stride))
            anchors = np.arange(0, ep.shape[1], stride)
            for k in anchors:
                if not ev[sid, k]:
                    continue
                cx, cy, r1, r2, th = ep[sid, k]
                if not np.isfinite(r1 + r2) or r1 <= 0 or r2 <= 0:
                    continue
                e = Ellipse((cx, cy), width=2 * r1, height=2 * r2,
                            angle=np.degrees(th), fill=False,
                            edgecolor="tab:red", lw=args.ellipse_lw,
                            alpha=args.ellipse_alpha, zorder=4)
                ax.add_patch(e)
            ax.set_title(f"{maze} #{sid}", fontsize=10)

    # row labels + counts
    for r, (maze, n) in enumerate(row_meta):
        axes[r, 0].set_ylabel(maze, rotation=0, fontsize=11,
                              labelpad=22, va="center")

    handles = [
        Line2D([], [], color="tab:blue", lw=1.6, label="GT trajectory (t=0→1)"),
        Line2D([], [], color="tab:red", lw=0, marker="o", markeredgecolor="tab:red",
               markerfacecolor="none", label=f"GT ellipses (every {max(1,int(args.ellipse_stride))}th)"),
        Line2D([], [], color="lime", marker="*", ls="", markersize=11, label="start"),
        Line2D([], [], color="red", marker="*", ls="", markersize=11, label="goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Scene dataset overview — {args.split} split "
                 f"(map + GT trajectory + GT ellipses, scene coords [-1,1]²)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.98))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=135)
    plt.close(fig)
    print("saved:", os.path.abspath(args.out))
    print("sampled ids per maze:", sampled_ids)


if __name__ == "__main__":
    main()
