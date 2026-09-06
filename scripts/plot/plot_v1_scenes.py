"""Plot V1 samples across the three scene mazes (map + trajectory + ellipses).

Reads the npz files produced by sample.py (scene P [n,128,2], E5 [n,128,5]
cx,cy,a,b,theta, cond [n,2,2]) and the scene maps, then renders one grid
row per maze: occupancy map + sampled trajectory + sampled ellipses.

Usage:
  python scripts/plot/plot_v1_scenes.py --n 6 --out outputs/v1_scenes_overview.png
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

MAZE_NAMES = ["umaze", "medium", "large"]
MAPS_DIR = os.path.join("data", "processed_scene_v1", "maps")
RES = 256


def nearest_sdf(sdf_grid, p):
    """Nearest-cell scene-unit SDF at points p [H,2] -> [H]."""
    cell = np.clip(np.round((p + 1.0) / 2.0 * (RES - 1)).astype(int), 0, RES - 1)
    return sdf_grid[cell[:, 1], cell[:, 0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="samples per maze (must match sample.py --n)")
    ap.add_argument("--out", default="outputs/v1_scenes_overview.png")
    args = ap.parse_args()

    ncol = args.n
    fig, axes = plt.subplots(len(MAZE_NAMES), ncol,
                             figsize=(3.3 * ncol, 3.3 * len(MAZE_NAMES)))
    axes = np.asarray(axes).reshape(len(MAZE_NAMES), ncol)
    stats = {}
    for r, maze in enumerate(MAZE_NAMES):
        d = np.load(os.path.join("outputs", f"v1_sample_{maze}.npz"))
        P = d["P"]
        E5 = d["E5"]
        cond = d["cond"]
        occ = np.load(os.path.join(MAPS_DIR, f"{maze}.npy"))
        sdf = np.load(os.path.join(MAPS_DIR, f"{maze}_sdf.npy"))
        coll_frac = float((nearest_sdf(sdf, P.reshape(-1, 2)) <= 0.0).mean())
        stats[maze] = {"n": int(P.shape[0]), "collision_frac": round(coll_frac, 4)}

        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r")
            p = P[c]
            ax.plot(p[:, 0], p[:, 1], "-", color="tab:blue", lw=1.3, zorder=3)
            ax.scatter(*cond[c, 0], marker="o", s=28, color="lime", zorder=6)
            ax.scatter(*cond[c, 1], marker="*", s=100, color="red", zorder=6)
            for k in range(p.shape[0]):
                cx, cy, a, b, th = E5[c, k]
                if not np.isfinite(a + b) or a <= 0 or b <= 0:
                    continue
                e = Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(th),
                            fill=False, edgecolor="tab:red", lw=0.55, alpha=0.8,
                            zorder=4)
                ax.add_patch(e)
            per = float((nearest_sdf(sdf, p) <= 0.0).mean())
            ax.set_title(f"{maze} #{c}  coll={per:.2f}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(maze, rotation=0, fontsize=11,
                              labelpad=20, va="center")

    handles = [
        Line2D([], [], color="tab:blue", lw=1.6, label="sampled trajectory"),
        Line2D([], [], color="tab:red", lw=0, marker="o", markersize=6,
               markerfacecolor="none", markeredgecolor="tab:red",
               label="sampled ellipses (all 128)"),
        Line2D([], [], color="lime", marker="o", ls="", markersize=8, label="start"),
        Line2D([], [], color="red", marker="*", ls="", markersize=11, label="goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("V1 joint diffusion test samples — umaze / medium / large "
                 "(scene [-1,1]²)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.985))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=135)
    plt.close(fig)
    print("saved:", os.path.abspath(args.out))
    print("per-maze collision fraction:", stats)


if __name__ == "__main__":
    main()
