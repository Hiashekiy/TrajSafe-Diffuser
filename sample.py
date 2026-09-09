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
from matplotlib.patches import Ellipse, Polygon

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import load_checkpoint
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler_v1 import sample_joint
from src.models.joint import JointPlanner
from src.datasets.joint_dataset import JointDataset
from src.geometry.convex_corridor import EllipseRegionBuilder
from src.geometry.ellipse_utils import physical_ellipse_center
from src.geometry.convex_region import halfspaces_to_vertices

MAZE_NAMES = ["umaze", "medium", "large"]


def e6_to_ellipse5(p, e6, absolute=False):
    """scene P [H,2] + e6 [H,6] -> [H,5] (cx,cy,a,b,theta)."""
    c = physical_ellipse_center(p, e6, absolute)
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


def build_region_polygons(P, E6, map_tensor, alm_cfg, stride=8, absolute=False):
    """Build sparse polygon overlays from the final predicted ellipses."""
    device = map_tensor.device
    p_t = torch.as_tensor(P, dtype=torch.float32, device=device)
    e_t = torch.as_tensor(E6, dtype=torch.float32, device=device)
    with torch.no_grad():
        A, b, mask, valid = EllipseRegionBuilder(
            map_tensor, {**alm_cfg, "center_absolute": absolute})(p_t, e_t)
    A, b = A.cpu().numpy(), b.cpu().numpy()
    mask, valid = mask.cpu().numpy(), valid.cpu().numpy()
    centers = physical_ellipse_center(P, E6, absolute)
    overlays = [[] for _ in range(len(P))]
    for i in range(len(P)):
        for k in range(0, P.shape[1], max(1, int(stride))):
            if not valid[i, k]:
                continue
            keep = mask[i, k]
            vertices = halfspaces_to_vertices(
                A[i, k, keep], b[i, k, keep], centers[i, k])
            if vertices is not None:
                overlays[i].append((k, vertices))
    return overlays


def _draw_regions(ax, regions):
    if not regions:
        return
    for _, vertices in regions:
        ax.add_patch(Polygon(
            vertices, closed=True, facecolor="tab:cyan", edgecolor="cyan",
            linewidth=0.8, alpha=0.14, zorder=2,
        ))


def plot_results(maze, occ, cond, P, E5, out_png, ncols=4,
                 draw_ellipses=True, regions=None):
    occ = np.asarray(occ)
    n = len(P)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r")
        _draw_regions(ax, None if regions is None else regions[i])
        p = P[i]
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:blue", lw=1.3, zorder=5)
        ax.scatter(*cond[i, 0], marker="o", s=30, color="lime", zorder=6)
        ax.scatter(*cond[i, 1], marker="*", s=110, color="red", zorder=6)
        if draw_ellipses:
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


def plot_comparison(maze, occ, cond, p_base, e5_base, p_alm, e5_alm,
                    out_png, pairs_per_row=2, draw_ellipses=True,
                    regions_base=None, regions_alm=None):
    """Plot matched baseline/ALM samples side by side.

    Each pair uses the same condition and initial Gaussian noise.  Baseline
    trajectories are blue and ALM trajectories are green; ellipses remain red.
    """
    occ = np.asarray(occ)
    n = len(p_base)
    nrows = int(np.ceil(n / pairs_per_row))
    ncols = pairs_per_row * 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 3.4 * nrows))
    axes = np.asarray(axes).reshape(nrows, ncols)

    def draw(ax, p, e5, index, label, color, shift=None, reference=None,
             regions=None):
        ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r")
        _draw_regions(ax, regions)
        if draw_ellipses:
            for cx, cy, a, b, th in e5:
                if not np.isfinite(a + b) or a <= 0 or b <= 0:
                    continue
                ax.add_patch(Ellipse(
                    (cx, cy), 2 * a, 2 * b, angle=np.degrees(th),
                    fill=False, edgecolor="tab:red", lw=0.5, alpha=0.55,
                    zorder=3,
                ))
        if reference is not None:
            ax.plot(reference[:, 0], reference[:, 1], "--", color="tab:blue",
                    lw=1.2, alpha=0.75, zorder=4, label="Baseline")
        ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=1.8, zorder=5,
                label="ALM" if reference is not None else None)
        ax.scatter(*cond[index, 0], marker="o", s=30, color="lime", zorder=6)
        ax.scatter(*cond[index, 1], marker="*", s=110, color="red", zorder=6)
        suffix = "" if shift is None else f"  mean Δ={shift:.4f}"
        ax.set_title(f"{maze} #{index} — {label}{suffix}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if reference is not None and index == 0:
            ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

    for i in range(n):
        row = i // pairs_per_row
        pair = i % pairs_per_row
        displacement = float(np.linalg.norm(p_alm[i] - p_base[i], axis=-1).mean())
        draw(axes[row, 2 * pair], p_base[i], e5_base[i], i,
             "Baseline", "tab:blue",
             regions=None if regions_base is None else regions_base[i])
        draw(axes[row, 2 * pair + 1], p_alm[i], e5_alm[i], i,
             "+ ALM", "tab:green", displacement, reference=p_base[i],
             regions=None if regions_alm is None else regions_alm[i])

    for flat_index in range(n * 2, nrows * ncols):
        axes.reshape(-1)[flat_index].axis("off")
    fig.suptitle(
        f"Matched V1 sampling comparison — {maze} (same conditions and noise)",
        fontsize=13,
    )
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
    ap.add_argument("--compare-alm", action="store_true",
                    help="draw matched baseline vs ALM samples using the same noise")
    ap.add_argument("--no-ellipses", action="store_true",
                    help="hide predicted ellipses in PNG plots")
    ap.add_argument("--draw-convex-regions", action="store_true",
                    help="overlay per-ellipse convex safe-region polygons")
    ap.add_argument("--convex-stride", type=int, default=8,
                    help="draw one convex region every N waypoints (default 8)")
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
    center_absolute = cfg.get("data", {}).get("ellipse_center_mode", "offset") == "absolute"

    if args.compare_alm:
        if not bool(cfg.get("alm", {}).get("enabled", False)):
            raise ValueError("--compare-alm requires an enabled 'alm' config section")
        p_base, e6_base = sample_joint(
            model, schedule, cond, map_t, device,
            steps=args.steps, seed=args.seed, alm_config=None,
            center_absolute=center_absolute,
        )
        P, E6 = sample_joint(
            model, schedule, cond, map_t, device,
            steps=args.steps, seed=args.seed, alm_config=cfg.get("alm"),
            center_absolute=center_absolute,
        )
        p_base = p_base.cpu().numpy()
        e6_base = e6_base.cpu().numpy()
    else:
        P, E6 = sample_joint(model, schedule, cond, map_t, device,
                             steps=args.steps, seed=args.seed,
                             alm_config=cfg.get("alm"), center_absolute=center_absolute)
    P = P.cpu().numpy()
    E6 = E6.cpu().numpy()
    E5 = e6_to_ellipse5(P, E6, center_absolute)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    occ_full = np.load(os.path.join(base, "maps", f"{args.maze}.npy"))
    if args.compare_alm:
        e5_base = e6_to_ellipse5(p_base, e6_base, center_absolute)
        regions_base = regions_alm = None
        if args.draw_convex_regions:
            regions_base = build_region_polygons(
                p_base, e6_base, map_t, cfg.get("alm", {}), args.convex_stride,
                absolute=center_absolute)
            regions_alm = build_region_polygons(
                P, E6, map_t, cfg.get("alm", {}), args.convex_stride,
                absolute=center_absolute)
        np.savez_compressed(
            args.out + ".npz",
            P_baseline=p_base, E6_baseline=e6_base, E5_baseline=e5_base,
            P_alm=P, E6_alm=E6, E5_alm=E5,
            cond=ds.cond[sel], ids=np.asarray(sel), maze=np.asarray(args.maze),
        )
        plot_comparison(args.maze, occ_full, ds.cond[sel],
                        p_base, e5_base, P, E5, args.out + ".png",
                        draw_ellipses=not args.no_ellipses,
                        regions_base=regions_base, regions_alm=regions_alm)
    else:
        np.savez_compressed(args.out + ".npz", P=P, E6=E6, E5=E5,
                            cond=ds.cond[sel], ids=np.asarray(sel),
                            maze=np.asarray(args.maze))
        regions = (build_region_polygons(
            P, E6, map_t, cfg.get("alm", {}), args.convex_stride,
            absolute=center_absolute)
            if args.draw_convex_regions else None)
        plot_results(args.maze, occ_full, ds.cond[sel], P, E5,
                     args.out + ".png", draw_ellipses=not args.no_ellipses,
                     regions=regions)
    print("saved:", os.path.abspath(args.out + ".npz"),
          os.path.abspath(args.out + ".png"))


if __name__ == "__main__":
    main()
