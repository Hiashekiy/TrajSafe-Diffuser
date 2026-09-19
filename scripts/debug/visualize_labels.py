"""Visualise offline ellipse labels for manual inspection.

For each sampled OD pair it draws:

  * occupancy map + all valid candidate Skeletons (grey),
    the nDTW-best candidate Gamma* (blue) and the GT trajectory (green);
  * GT ellipse centres c_i^* = Gamma*(s_i^*) and the GT ellipses;
  * progress_gt(s) with the ShapeValid mask;
  * the GT soft ellipse mask for one valid waypoint.

    python scripts/debug/visualize_labels.py \
        --config configs/config.yaml \
        --split test --num 8 --out outputs/label_samples
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.ellipse_shape import shape4_to_abtheta
from src.datasets.skeleton_dataset import SkeletonDataset, MAZE_NAMES


def to_px(points, res):
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * res - 0.5


def ellipse_xy(center, a, b, theta, n=96):
    ang = np.linspace(0.0, 2.0 * np.pi, int(n))
    ct, st = np.cos(theta), np.sin(theta)
    ex, ey = a * np.cos(ang), b * np.sin(ang)
    return np.stack([ct * ex - st * ey + center[0],
                     st * ex + ct * ey + center[1]], axis=-1)


def draw_ellipse(ax, center, a, b, theta, res, color="#d62728", lw=0.6,
                 alpha=0.8):
    pts = ellipse_xy(center, a, b, theta)
    px = to_px(pts, res)
    ax.plot(px[:, 0], px[:, 1], color=color, lw=lw, alpha=alpha)


def plot_one(ds, idx, out_png, stride=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    item = ds[idx]
    maze = int(item["maze_id"])
    occ = ds.maps[maze][0, 0].numpy()
    res = occ.shape[0]
    cond = item["cond"].numpy()
    pos = item["pos"].numpy()
    geom = item["candidate_geometry"].numpy()
    glen = item["candidate_geometry_lengths"].numpy()
    mask = item["candidate_mask"].numpy()
    best = int(item["topology_best"])
    # The ellipse centre target is the GT trajectory waypoint itself.
    center = pos
    shape4 = item["ellipse_shape4_gt"].numpy()
    valid = item["shape_valid"].numpy()
    gt_mask = item["ellipse_mask"].numpy()
    a, b, theta = shape4_to_abtheta(torch.as_tensor(shape4))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.5), dpi=120)
    ax = axes[0, 0]
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    for m in np.nonzero(mask)[0]:
        n = int(glen[m])
        if n > 1:
            p = to_px(geom[m, :n], res)
            ax.plot(p[:, 0], p[:, 1], color="#9e9e9e", lw=0.8, alpha=0.8,
                    zorder=2)
    if int(glen[best]) > 0:
        p = to_px(geom[best, :int(glen[best])], res)
        ax.plot(p[:, 0], p[:, 1], color="#1f77b4", lw=2.0, label="Gamma*")
    tp = to_px(pos, res)
    ax.plot(tp[:, 0], tp[:, 1], color="#2ca02c", lw=1.6, label="GT traj")
    cp = to_px(center, res)
    ax.scatter(cp[:, 0], cp[:, 1], s=2.0, c="#d62728", label="c* = Gamma*(s*)")
    ax.set_title("sample %d (%s): candidates + Gamma* + GT traj" %
                 (idx, MAZE_NAMES[maze]), fontsize=10)
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[0, 1]
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    for k in range(0, len(center), max(1, stride)):
        draw_ellipse(ax, center[k], float(a[k]), float(b[k]), float(theta[k]),
                     res, color="#d62728", lw=0.6, alpha=0.75)
    ax.scatter(cp[:, 0], cp[:, 1], s=2.0, c="#1f77b4")
    bad = np.nonzero(~valid)[0]
    if len(bad):
        ax.scatter(cp[bad, 0], cp[bad, 1], s=8.0, facecolors="none",
                   edgecolors="#ff9800", label="ShapeValid=False")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title("GT ellipses (a >= b > 0), invalid = orange", fontsize=10)

    ax = axes[1, 0]
    ax.step(np.arange(len(valid)), valid.astype(np.float32), where="mid",
            color="#2ca02c", lw=1.2)
    if len(bad):
        ax.scatter(bad, np.ones_like(bad, dtype=np.float32), s=18,
                   facecolors="none", edgecolors="#ff9800",
                   label="ShapeValid=False")
        ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("waypoint index")
    ax.set_ylabel("shape_valid")
    ax.set_title("ShapeValid per waypoint (no progress_gt dependency)",
                 fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    valid_idx = np.nonzero(valid)[0]
    k = int(valid_idx[len(valid_idx) // 2]) if len(valid_idx) else 0
    ax.imshow(gt_mask[k], origin="lower", cmap="Reds", alpha=0.55,
              interpolation="nearest", extent=(-0.5, res - 0.5, -0.5, res - 0.5))
    draw_ellipse(ax, center[k], float(a[k]), float(b[k]), float(theta[k]),
                 res, color="#111111", lw=1.2)
    cx, cy = cp[k]
    pad = max(10.0, 0.14 * res)
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_title("GT soft ellipse mask @ waypoint %d (valid=%s, zoom)" %
                 (k, bool(valid[k])), fontsize=10)

    fig.suptitle("Large label check  sample %d  valid=%.3f  best=%d" %
                 (idx, float(valid.mean()), best), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    np.savez_compressed(
        os.path.splitext(out_png)[0] + ".npz",
        center_gt=center, shape4_gt=shape4,
        shape_valid=valid, best=best, cond=cond, pos=pos)
    return {
        "idx": idx, "maze": MAZE_NAMES[maze], "best": best,
        "n_valid_candidates": int(mask.sum()),
        "shape_valid_fraction": float(valid.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="outputs/label_samples")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scenes_root = cfg["data"].get("scenes", "data/scenes")
    skeleton_root = cfg["data"].get("skeleton", "data/skeleton")
    mazes = cfg["data"].get("mazes", ["large"])
    ds = SkeletonDataset(
        args.split, scenes_root, skeleton_root,
        geometry_points=int((cfg.get("topology") or {}).get(
            "candidate_geometry_points", 1280)),
        ellipse_mask_res=int(cfg["data"].get("ellipse_mask_res", 64)),
        ellipse_mask_tau=float(cfg["data"].get("ellipse_mask_tau", 10.0)),
        mazes=mazes)
    os.makedirs(args.out, exist_ok=True)
    print("[viz] split=%s mazes=%s samples=%d/%d" %
          (args.split, mazes, args.num, len(ds)), flush=True)
    summary = []
    for k in range(args.num):
        if args.offset + k >= len(ds):
            break
        idx = args.offset + k
        out_png = os.path.join(args.out, "label_%s_%04d.png" % (args.split, idx))
        info = plot_one(ds, idx, out_png, stride=args.stride)
        summary.append(info)
        print("  %s" % info, flush=True)
    print("[viz] wrote %d PNGs to %s" % (len(summary), args.out), flush=True)


if __name__ == "__main__":
    main()
