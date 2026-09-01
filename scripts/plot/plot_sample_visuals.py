"""Visualize scene-dataset samples: map + trajectory + GT ellipses (scene coords)."""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from src.utils.visualization import draw_traj, set_map_limits

BASE = "data/processed_scene"
SPLIT = "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze", default="umaze", choices=["umaze", "medium", "large"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--idxs", type=str, default=None)
    args = ap.parse_args()

    mid = np.load(os.path.join(BASE, SPLIT, "maze_id.npy"))
    mi = {"umaze": 0, "medium": 1, "large": 2}[args.maze]
    sel = np.where(mid == mi)[0]
    if args.idxs is not None:
        idxs = [int(x.strip()) for x in args.idxs.split(",") if x.strip()]
    else:
        idxs = np.linspace(0, len(sel) - 1, max(1, min(args.n, len(sel)))).astype(int).tolist()

    occ = np.load(os.path.join(BASE, "maps", f"{args.maze}.npy"))
    pos = np.load(os.path.join(BASE, SPLIT, "positions.npy"))
    ep = np.load(os.path.join(BASE, SPLIT, "ellipse_params.npy"))
    ev = np.load(os.path.join(BASE, SPLIT, "ellipse_valid.npy"))
    ncol = 2; nrow = max(1, (len(idxs) + 1) // 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, k in zip(axes.ravel(), idxs):
        i = sel[k]
        ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r", alpha=0.8)
        draw_traj(ax, pos[i], marker_every=0, arrow_every=0)
        for j in range(0, len(ep[i]), 12):
            if not ev[i][j]:
                continue
            cx, cy, r1, r2, th = ep[i][j]
            ax.add_patch(Ellipse((cx, cy), 2 * r1, 2 * r2, angle=np.degrees(th),
                                 fill=False, edgecolor="tab:red", lw=1.2, alpha=0.8))
        ax.set_title(f"{args.maze} sample #{k}  valid={int(ev[i].sum())}/{len(ev[i])}")
        ax.set_aspect("equal"); set_map_limits(ax, (-1, 1, -1, 1)); ax.legend(loc="upper right", fontsize=7)
    for k in range(len(idxs), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Scene dataset samples: {args.maze}", fontsize=13)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/sample_visual_grid_scene_{args.maze}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
