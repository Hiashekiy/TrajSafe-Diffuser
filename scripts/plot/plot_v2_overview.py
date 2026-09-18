"""Overview figures for V2 (Skeleton-Topology-Grounded Trajectory Diffusion).

    python scripts/plot/plot_v2_overview.py --ckpt outputs/ckpt_v2_skeleton/best.pt

Produces, per maze and overall:
    fig1_samples_<maze>.png     map + selected topology + trajectory + ellipses
    fig2_convex_<maze>.png      the same + verified convex regions
    fig3_centers.png            ellipse centres coloured by obstacle clearance
    fig4_clearance_hist.png     clearance distribution over many OD pairs
    fig5_training_curves.png    parsed from outputs/v2_train.log
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler_v2 import sample_v2
from src.models.skeleton import SkeletonPlanner
from src.datasets.skeleton_dataset import SkeletonDataset, make_collate, MAZE_NAMES
from src.geometry.safe_convex_region import generate_verified_convex_region


def px(points, res):
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * res - 0.5


def ellipse_xy(center, shape4, res, n=64):
    a = np.exp(shape4[:, 0])
    b = np.exp(shape4[:, 1])
    th = 0.5 * np.arctan2(shape4[:, 3], shape4[:, 2])
    ang = np.linspace(0, 2 * np.pi, n)
    ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
    s = res / 2.0
    ex = a[:, None] * np.cos(ang)[None] * s
    ey = b[:, None] * np.sin(ang)[None] * s
    return ct * ex - st * ey + px(center, res)[:, 0:1], st * ex + ct * ey + px(center, res)[:, 1:2]


def draw_map(ax, occ, title):
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_sample(ax, occ, cond, path, traj, center, shape4, title,
                regions=None, stride=4):
    res = occ.shape[0]
    draw_map(ax, occ, title)
    if regions is not None:
        for verts in regions:
            if verts is None or len(verts) < 3:
                continue
            poly = px(verts, res)
            ax.add_patch(MplPolygon(poly, closed=True, facecolor="#ff7f0e",
                                    edgecolor="#ff7f0e", alpha=0.16, lw=0.6))
    ax.plot(*px(path, res).T, color="#1f77b4", lw=1.3, label="topology", zorder=4)
    ax.plot(*px(traj, res).T, color="#2ca02c", lw=1.8, label="trajectory", zorder=5)
    ex, ey = ellipse_xy(center, shape4, res)
    for k in range(0, len(center), max(1, stride)):
        ax.plot(ex[k], ey[k], color="#d62728", lw=0.7, alpha=0.85, zorder=3)
    cp = px(center, res)
    ax.scatter(cp[:, 0], cp[:, 1], s=1.2, c="#d62728", zorder=6)
    sp = px(cond, res)
    ax.scatter(sp[:, 0], sp[:, 1], marker="*", s=150, c="k", zorder=7)


def sample_group(model, schedule, ds, idxs, cfg, device, seed=0, stride=4):
    batch = make_collate(ds)([ds[i] for i in idxs])
    cond = batch["cond"].to(device)
    occ = batch["map_tensor"].to(device)
    cand = batch["candidate_paths"].to(device)
    cmask = batch["candidate_mask"].to(device)
    clen = batch["candidate_lengths"].to(device)
    out = sample_v2(model, schedule, cond, occ, cand, cmask, clen, device=device,
                    steps=cfg["steps"], seed=seed,
                    commit_t=int(cfg["commit_t"]), selection=cfg["selection"])
    return batch, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v2_skeleton.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt_v2_skeleton/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-maze", type=int, default=3)
    ap.add_argument("--hist-samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/v2_figs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    topo = cfg.get("topology", {})
    sk = {"steps": args.steps, "commit_t": int(topo.get("commit_t", 7)),
          "selection": topo.get("selection", "sample")}
    source = cfg["data"].get("source", "data/processed_scene_v1")
    base_dir = cfg["data"].get("base", "data/processed_scene_v2")
    mask_res = int(cfg["loss"].get("ellipse_safe_res", 64))
    ds = SkeletonDataset(args.split, source, base_dir, mask_res=mask_res,
                         mask_tau=float(cfg["loss"].get("ellipse_mask_tau", 10.0)))
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(args.device)
    model = SkeletonPlanner(cfg["model"], topo).to(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    epoch = ck.get("epoch")
    os.makedirs(args.out, exist_ok=True)
    mid = ds.mid
    occ_maps = [ds.maps[i][0, 0].numpy() for i in range(len(MAZE_NAMES))]
    print("[figs] ckpt epoch=%s device=%s" % (epoch, args.device), flush=True)

    for m, name in enumerate(MAZE_NAMES):
        idxs = [int(i) for i in np.nonzero(mid == m)[0][:args.per_maze]]
        if not idxs:
            continue
        batch, out = sample_group(model, schedule, ds, idxs, sk, args.device,
                                  seed=100 + m, stride=args.stride)
        occ = occ_maps[m]
        n = len(idxs)

        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), dpi=110,
                                 squeeze=False)
        for k in range(n):
            c = out["ellipse_center"][k].detach().cpu().numpy()
            s4 = out["ellipse_shape4"][k].detach().cpu().numpy()
            draw_sample(axes[0][k], occ, batch["cond"][k].numpy(),
                        batch["candidate_paths"][k, int(out["selected_idx"][k]), :, :2].numpy(),
                        out["p"][k].detach().cpu().numpy(), c, s4,
                        "%s #%d  m=%d t_c=%d" % (name, idxs[k],
                                                 int(out["selected_idx"][k]),
                                                 int(out["committed_at"][k])),
                        stride=args.stride)
        axes[0][0].legend(loc="upper right", fontsize=7)
        fig.suptitle("V2 samples (epoch %s): topology / trajectory / safety ellipses" % epoch)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig1_samples_%s.png" % name))
        plt.close(fig)

        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), dpi=110,
                                 squeeze=False)
        for k in range(n):
            c = out["ellipse_center"][k].detach().cpu().numpy()
            s4 = out["ellipse_shape4"][k].detach().cpu().numpy()
            regions = []
            for j in range(0, len(c), max(1, args.stride * 2)):
                a = float(np.exp(s4[j, 0])); b = float(np.exp(s4[j, 1]))
                th = float(0.5 * np.arctan2(s4[j, 3], s4[j, 2]))
                A, bv, verts, info = generate_verified_convex_region(
                    occ, c[j], a, b, th, window_half=0.5, safety_margin=0.008,
                    shrink=0.002)
                regions.append(verts)
            draw_sample(axes[0][k], occ, batch["cond"][k].numpy(),
                        batch["candidate_paths"][k, int(out["selected_idx"][k]), :, :2].numpy(),
                        out["p"][k].detach().cpu().numpy(), c, s4,
                        "%s #%d: %d verified convex regions" % (name, idxs[k],
                                                                len(regions)),
                        regions=regions, stride=args.stride)
        fig.suptitle("V2 verified convex regions (orange) built from the predicted ellipses (epoch %s)" % epoch)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig2_convex_%s.png" % name))
        plt.close(fig)
        print("  %s done" % name, flush=True)

    # ---- centre clearance over many OD pairs -----------------------------
    idxs = [int(i) for i in np.arange(min(args.hist_samples, len(ds)))]
    batch, out = sample_group(model, schedule, ds, idxs, sk, args.device, seed=7)
    all_clear = []
    for k in range(len(idxs)):
        occ = occ_maps[int(batch["maze_id"][k])]
        res = occ.shape[0]
        c = out["ellipse_center"][k].detach().cpu().numpy()
        j, i = np.nonzero(occ.astype(bool))
        cx = (i + 0.5) * 2.0 / res - 1.0
        cy = (j + 0.5) * 2.0 / res - 1.0
        d = np.sqrt((c[:, 0:1] - cx[None, :]) ** 2 + (c[:, 1:2] - cy[None, :]) ** 2)
        all_clear.append(d.min(axis=1) * res / 2.0)
    all_clear = np.concatenate(all_clear)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=110)
    k = 0
    occ = occ_maps[int(batch["maze_id"][k])]
    c = out["ellipse_center"][k].detach().cpu().numpy()
    res = occ.shape[0]
    j, i = np.nonzero(occ.astype(bool))
    cx = (i + 0.5) * 2.0 / res - 1.0
    cy = (j + 0.5) * 2.0 / res - 1.0
    d = np.sqrt((c[:, 0:1] - cx[None, :]) ** 2 + (c[:, 1:2] - cy[None, :]) ** 2)
    clear = d.min(axis=1) * res / 2.0
    draw_map(axes[0], occ, "ellipse centres coloured by clearance (cells)")
    cp = px(c, res)
    sc = axes[0].scatter(cp[:, 0], cp[:, 1], c=clear, s=26, cmap="viridis",
                         vmin=0, vmax=max(20.0, float(clear.max())), zorder=6)
    axes[0].plot(*px(out["p"][k].detach().cpu().numpy(), res).T, color="#2ca02c",
                 lw=1.6, zorder=5)
    axes[0].scatter(*px(batch["cond"][k].numpy(), res).T, marker="*", s=150, c="k",
                    zorder=7)
    plt.colorbar(sc, ax=axes[0], fraction=0.046)
    axes[1].hist(all_clear, bins=50, color="#1f77b4")
    axes[1].axvline(0.0, color="r", lw=2, label="obstacle boundary")
    axes[1].set_xlabel("distance from ellipse centre to nearest obstacle cell (cells)")
    axes[1].set_ylabel("count")
    axes[1].set_title("CenterFree over %d ellipses: min=%.2f cells, all free"
                      % (len(all_clear), all_clear.min()))
    axes[1].legend()
    fig.suptitle("c_i = gamma_m(s_i) keeps every centre away from obstacles (epoch %s)" % epoch)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3_centers.png"))
    plt.close(fig)

    # ---- training curves --------------------------------------------------
    log = os.path.join(ROOT, "outputs", "v2_train.log")
    if os.path.exists(log):
        ep, tr, va = [], {}, {}
        pat = re.compile(r"\[epoch (\d+)/\d+\] train (.*?)(?:\s+\| val (.*))?$")
        for line in open(log, "r", encoding="utf-8", errors="ignore"):
            m2 = pat.search(line.strip())
            if not m2:
                continue
            e = int(m2.group(1))
            ep.append(e)
            for part in m2.group(2).split():
                if "=" in part:
                    k, v = part.split("=")
                    tr.setdefault(k, {})[e] = float(v)
            if m2.group(3):
                for part in m2.group(3).replace("(best)", "").split():
                    if "=" in part:
                        k, v = part.split("=")
                        va.setdefault(k, {})[e] = float(v)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=110)
        for k in ["Lp", "Lsmooth", "Ltopo", "Lprog", "Lshape", "Liou", "Lsafe", "total"]:
            if k in tr:
                xs = sorted(tr[k])
                axes[0].plot(xs, [tr[k][x] for x in xs], label=k, lw=1.3)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss (log)")
        axes[0].set_title("V2 training losses")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)
        if "total" in va:
            xs = sorted(va["total"])
            axes[1].plot(xs, [tr["total"][x] for x in xs if x in tr["total"]],
                         label="train total", lw=1.5)
            axes[1].plot(xs, [va["total"][x] for x in xs], "o-", label="val total", lw=1.5)
            axes[1].set_yscale("log")
        axes[1].set_xlabel("epoch")
        axes[1].set_title("train vs validation")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig5_training_curves.png"))
        plt.close(fig)
        print("  training curves done (%d epochs logged)" % (len(ep)), flush=True)

    print("figures in", args.out, flush=True)


if __name__ == "__main__":
    main()
