"""plot_dataset_samples.py - visual check of the processed CARLA caches.

Three figures, all built straight from the on-disk arrays (no model, no
sampler involved):

    1. samples_grid.png   3 splits x 4 samples.  Occupancy (0 = free = white,
                          1 = obstacle = black) + GT curve + start/goal +
                          the w=8 border ring that --border-mode protect
                          restores, + per-sample free-space ratio.
    2. border_moat.png    the SAME scene in three caches side by side:
                          source (untouched), 160k8 (legacy erosion, the moat),
                          160k8p (border-protected), plus the obstacle diff.
    3. free_ratio.png     per-sample free-ratio distribution of all three
                          caches, with the 2/3 target line.

Usage:

    python scripts/plot/plot_dataset_samples.py
    python scripts/plot/plot_dataset_samples.py --ids 3 17 42 88 --out outputs/preview
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

CACHES = {
    "k8p": "data/carla_processed_160k8p",   # border-protected  (current)
    "k8":  "data/carla_processed_160k8",    # legacy erosion    (moat)
    "k0":  "data/carla_processed_160",      # untouched source
}

SPLIT_COLOR = {"train": "#00897b", "val": "#5e35b1", "test": "#e65100"}
GT_COLOR = "#00e5ff"
RING_COLOR = "#ff6d00"

CELL_M = 160.0 / 256.0      # scene is [-1, 1]^2 -> 160 m across, 256 cells
RES = 256


def to_px(points, res: int = RES):
    """scene [-1,1] -> pixel coords (inverse of SkeletonGraph.pixel_to_scene).

    curve_gt / control_gt / conditions / candidate_xy / alm_cell_* are all in
    SCENE units; the occupancy array is indexed [row=y, col=x] in pixel units,
    so everything has to go through this before it can be overlaid.
    """
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * res - 0.5


def show_occ(ax, occ):
    """Occupancy with the project's canonical orientation (row 0 at the bottom)."""
    ax.imshow(occ, cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest",
              origin="lower", extent=(-0.5, RES - 0.5, -0.5, RES - 0.5))


def load(root: str, split: str, name: str):
    return np.load(os.path.join(ROOT, root, split, f"{name}.npy"), mmap_mode="r")


def free_ratio(root: str, split: str, stride: int = 4) -> np.ndarray:
    """Per-sample free ratio (1 - obstacle mean), subsampled along N."""
    occ = load(root, split, "occupancy")
    out = np.empty(occ.shape[0], dtype=np.float32)
    for i in range(0, occ.shape[0], 512):
        blk = np.asarray(occ[i:i + 512], dtype=np.float32)
        out[i:i + 512] = 1.0 - blk.reshape(blk.shape[0], -1).mean(axis=1)
    return out


def draw_ring(ax, w: int = 8, res: int = 256):
    """Outline the outer w-cell band that must stay blocked."""
    ax.add_patch(Rectangle((-0.5, -0.5), res, res, fill=False,
                           ec=RING_COLOR, lw=1.4, ls=(0, (5, 3)), alpha=0.95))
    ax.add_patch(Rectangle((w - 0.5, w - 0.5), res - 2 * w, res - 2 * w,
                           fill=False, ec=RING_COLOR, lw=1.0, ls=":", alpha=0.75))


def scene_panel(ax, root: str, split: str, i: int, ring: bool = True,
                title: str | None = None, small: bool = False):
    occ = np.asarray(load(root, split, "occupancy")[i])
    curve = to_px(load(root, split, "curve_gt")[i])
    cond = to_px(load(root, split, "conditions")[i])   # [start, goal]

    show_occ(ax, occ)
    if ring:
        draw_ring(ax)
    ax.plot(curve[:, 0], curve[:, 1], color=GT_COLOR, lw=1.9, solid_capstyle="round",
            zorder=4)
    ax.plot(*cond[0], "o", ms=6, mfc="#2e7d32", mec="white", mew=1.0, zorder=5)
    ax.plot(*cond[1], "s", ms=6, mfc="#c62828", mec="white", mew=1.0, zorder=5)

    free = 1.0 - float(occ.mean())
    if title is None:
        title = f"{split} #{i}   free {free:.2f}"
    else:
        title = f"{title}\nfree {free:.2f}"
    ax.set_title(title, fontsize=7 if small else 9, color=SPLIT_COLOR[split],
                 pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bdbdbd"); s.set_linewidth(0.6)


# --------------------------------------------------------------------------
def fig_grid(out: str, ids: list[int], splits=("train", "val", "test"), root="k8p"):
    fig, axes = plt.subplots(len(splits), len(ids),
                             figsize=(2.1 * len(ids), 2.2 * len(splits) + 0.7))
    axes = np.atleast_2d(axes)
    for r, split in enumerate(splits):
        for c, i in enumerate(ids):
            scene_panel(axes[r, c], CACHES[root], split, i)
    fig.suptitle("carla_processed_160k8p - occupancy (white = free, black = obstacle)\n"
                 "cyan = GT curve    green/red = start/goal    "
                 "orange = protected w=8 border ring",
                 fontsize=10, y=0.985)
    fig.tight_layout(rect=(0, 0.005, 1, 0.90), h_pad=2.0, w_pad=0.5)
    p = os.path.join(out, "samples_grid.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig_moat(out: str, split: str, idx: int):
    """One scene, three caches: shows the moat and that it is gone."""
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.8))
    order = ["k0", "k8", "k8p"]
    label = {"k0": "source  carla_processed_160\n(untouched, no erosion)",
             "k8": "carla_processed_160k8\n(legacy k=8 -> MOAT)",
             "k8p": "carla_processed_160k8p\n(k=8, border protected)"}
    for c, key in enumerate(order):
        ax = axes[0, c]
        scene_panel(ax, CACHES[key], split, idx, ring=(key != "k0"),
                    title=label[key])
        ax.set_title(label[key], fontsize=8.5, color="#212121", pad=4)

    src = np.asarray(load(CACHES["k0"], split, "occupancy")[idx]).astype(np.int16)
    for c, key in enumerate(order[1:], start=0):
        ax = axes[1, c]
        cur = np.asarray(load(CACHES[key], split, "occupancy")[idx]).astype(np.int16)
        diff = cur - src                      # +1 = obstacle invented, -1 = removed
        ax.imshow(diff, cmap=plt.get_cmap("coolwarm"), vmin=-1, vmax=1,
                  interpolation="nearest", origin="lower",
                  extent=(-0.5, RES - 0.5, -0.5, RES - 0.5))
        cur_curve = to_px(load(CACHES[key], split, "curve_gt")[idx])
        ax.plot(cur_curve[:, 0], cur_curve[:, 1], color="k", lw=1.0, alpha=0.55,
                zorder=4)
        lost = int((diff == -1).sum()); added = int((diff == 1).sum())
        w = 8
        inner = np.zeros_like(diff, dtype=bool); inner[w:-w, w:-w] = True
        lost_ring = int(((diff == -1) & ~inner).sum())
        ax.set_title(f"diff vs source   blue = wall removed (now free)\n"
                     f"removed {lost}  ({lost_ring} in outer w=8 ring)   added {added}",
                     fontsize=8, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.95,
                    "Read the middle column:\n\n"
                    "legacy k=8 erosion treats everything outside\n"
                    "the crop as FREE (scipy border_value=0), so the\n"
                    "outer 8 cells are eaten from the outside too.\n"
                    "Result: a free ring all around the scene -> the\n"
                    "outer w=1/2/4/8 obstacle rate drops to 0.000 in\n"
                    "EVERY sample, a path can loop around the map,\n"
                    "and the reported free ratio is inflated by ~10 pt.\n\n"
                    "Right column (k8p): the same k=8 widening inside,\n"
                    "but the outer ring is restored from the source, so\n"
                    "the border can never become emptier than it was.\n"
                    "No obstacle is invented either -> GT stays feasible.",
                    fontsize=8.5, va="top", family="DejaVu Sans")
    fig.suptitle(f"border fix - {split} #{idx}   (1 cell = {CELL_M:.3f} m, scene = 160 m)",
                 fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "border_moat.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def _chebyshev_center(A, b, guess):
    from scipy.optimize import linprog
    n = A.shape[0]
    norms = np.linalg.norm(A, axis=1)
    # maximize r  s.t.  A x + r||a_i|| <= b_i
    c = np.array([0.0, 0.0, -1.0])
    Aub = np.hstack([A, norms[:, None]])
    res = linprog(c, A_ub=Aub, b_ub=b, bounds=[(None, None), (None, None), (0, None)],
                  method="highs")
    if res.success and res.x[2] > 1e-4:
        return res.x[:2]
    return guess


def fig_detail(out: str, split: str, idx: int, root: str = "k8p"):
    """What the trainer actually sees: candidates, GT, and the ALM corridor."""
    from src.geometry.convex_region import halfspaces_to_vertices

    rootp = CACHES[root]
    occ = np.asarray(load(rootp, split, "occupancy")[idx])
    curve = to_px(load(rootp, split, "curve_gt")[idx])
    cand = to_px(load(rootp, split, "candidate_xy")[idx])
    cmask = np.asarray(load(rootp, split, "candidate_mask")[idx])
    best = int(np.asarray(load(rootp, split, "topology_best")[idx]))
    A = np.asarray(load(rootp, split, "alm_cell_a")[idx])
    b = np.asarray(load(rootp, split, "alm_cell_b")[idx])
    cv = np.asarray(load(rootp, split, "alm_cell_valid")[idx])
    alm_ok = bool(np.asarray(load(rootp, split, "alm_valid")[idx]))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.9))
    for ax in axes:
        show_occ(ax, occ)

    ax = axes[0]
    for k in range(cand.shape[0]):
        if not cmask[k]:
            continue
        is_best = (k == best)
        ax.plot(cand[k, :, 0], cand[k, :, 1],
                color="#0288d1" if is_best else "#9e9e9e",
                lw=2.0 if is_best else 1.0, alpha=1.0 if is_best else 0.75,
                label="best topology" if is_best else None, zorder=4)
    ax.plot(curve[:, 0], curve[:, 1], color=GT_COLOR, lw=2.0, label="GT curve", zorder=6)
    ax.plot([], [], color=RING_COLOR, ls="--", lw=1.2, label="w=8 protected ring")
    draw_ring(ax)
    ax.set_title(f"candidates   {int(cmask.sum())}/4 valid   topology_best={best}",
                 fontsize=9.5, pad=4)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    ax = axes[1]
    polys = 0
    curve_s = np.asarray(load(rootp, split, "curve_gt")[idx])   # scene units
    for i in range(128):
        if not cv[i]:
            continue
        # halfspaces live in scene units -> solve there, draw in pixels
        v = halfspaces_to_vertices(A[i], b[i], interior_point=curve_s[i])
        if v is None:
            v = halfspaces_to_vertices(
                A[i], b[i], interior_point=_chebyshev_center(A[i], b[i], curve_s[i]))
        if v is None:
            continue
        polys += 1
        v = to_px(v)
        ax.fill(v[:, 0], v[:, 1], facecolor=plt.get_cmap("viridis")(i / 127.0),
                edgecolor="#263238", lw=0.4, alpha=0.55, zorder=3)
    ax.plot(curve[:, 0], curve[:, 1], color=GT_COLOR, lw=2.0, zorder=6)
    for k in range(cand.shape[0]):
        if cmask[k]:
            ax.plot(cand[k, :, 0], cand[k, :, 1], color="#37474f", lw=0.6,
                    alpha=0.45, zorder=4)
    ax.plot([], [], color="#2e7d32", lw=6, alpha=0.35, label="ALM convex cells")
    ax.set_title(f"ALM safety corridor   alm_valid={alm_ok}   "
                 f"cells {int(cv.sum())}/128 drawn {polys}", fontsize=9.5, pad=4)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bdbdbd")
    fig.suptitle(f"carla_processed_160k8p - {split} #{idx}   "
                 f"(white = free, black = obstacle, 1 cell = {CELL_M:.3f} m)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(out, f"sample_detail_{split}{idx}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig_free(out: str, splits=("train", "val", "test")):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    bins = np.linspace(0, 1, 51)
    stats = {}
    for ax, split in zip(axes, splits):
        for key in ("k0", "k8", "k8p"):
            v = free_ratio(CACHES[key], split)
            stats.setdefault(key, {})[split] = (float(v.mean()), float(v.min()))
            ax.hist(v, bins=bins, histtype="step", lw=1.6,
                    label=f"{key}  mean {v.mean():.3f}", alpha=0.95)
        ax.axvline(2 / 3, color="k", ls="--", lw=1.2)
        ax.text(2 / 3, ax.get_ylim()[1] * 0.96, " target 2/3", fontsize=7.5,
                va="top", ha="left")
        ax.set_title(f"{split}", color=SPLIT_COLOR[split])
        ax.set_xlabel("per-sample free-space ratio"); ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("samples")
    fig.suptitle("free-space ratio - the 2/3 target costs ~30 more cells (19 m) "
                 "of road network per scene", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out, "free_ratio.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[3, 17, 42, 88])
    ap.add_argument("--split", default="test", help="split for the border figure")
    ap.add_argument("--idx", type=int, default=None,
                    help="sample for the border figure (default: --ids[1])")
    ap.add_argument("--out", default="outputs/preview_dataset")
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    ids = a.ids[:4]
    fig_grid(out, ids)
    fig_moat(out, a.split, a.idx if a.idx is not None else ids[1])
    fig_detail(out, a.split, a.idx if a.idx is not None else ids[1])
    st = fig_free(out)
    print("\nfree-ratio means:", {k: {s: round(v[0], 4) for s, v in d.items()}
                                  for k, d in st.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
