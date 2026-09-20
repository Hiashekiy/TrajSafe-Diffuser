"""inspect_sample.py - look at ONE sample in detail with the trained model.

    python scripts/inspect_sample.py --split test --index 42
    python scripts/inspect_sample.py --split test --index -1 --random 3

Panels:
  1. occupancy + ALL candidates + GT curve/controls + predicted curve/controls
  2. the 128 fixed Skeleton centres + the predicted safety ellipses
  3. topology pi (bar) + a text box with the quantitative metrics
  4. the DDIM reverse process: the decoded curve q_t -> p_t at several steps

Metrics printed and shown in the figure:
  curve RMSE (m), control RMSE (m), goal error (m),
  point-collision %, any-collision flag, selected vs nDTW-best candidate.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.trajsafe import TrajSafePlanner
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,
                                               make_collate)

METERS = 40.0


def to_px(points, res):
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * res - 0.5


def point_hits(points, occ):
    res = occ.shape[0]
    px = np.rint((points[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
    py = np.rint((points[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
    ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
    hit = np.ones(len(points), dtype=bool)
    hit[ok] = occ[py[ok], px[ok]].astype(bool)
    return hit


def draw_ellipses(ax, center, a, b, theta, res, stride=3, color="#d62728",
                  lw=0.6, alpha=0.8):
    ang = np.linspace(0, 2 * np.pi, 40)
    c = to_px(center, res)
    for k in range(0, len(center), max(1, stride)):
        ct, st = np.cos(theta[k]), np.sin(theta[k])
        ex = a[k] * np.cos(ang) * res / 2.0
        ey = b[k] * np.sin(ang) * res / 2.0
        ax.plot(ct * ex - st * ey + c[k, 0], st * ex + ct * ey + c[k, 1],
                color=color, lw=lw, alpha=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="outputs/bspline_carla/ckpt/best_task.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", type=int, default=0,
                    help="-1 = pick `--random` random samples")
    ap.add_argument("--random", type=int, default=3,
                    help="how many random samples when --index -1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None,
                    help="DDIM steps (default: the full schedule)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/bspline_carla/inspect")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    processed_root = cfg["data"].get("processed_root", "data/carla_processed")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))

    ds = CarlaSplineDataset(args.split, processed_root,
                            geometry_points=geo_points, require_labels=False)
    if args.index >= 0:
        idxs = [args.index]
    else:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(ds), size=max(1, args.random),
                          replace=False).tolist()

    model = TrajSafePlanner(cfg["model"], cfg.get("ellipse_label"),
                            cfg.get("bspline")).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    os.makedirs(args.out, exist_ok=True)

    for n, i in enumerate(idxs):
        batch = make_collate(ds)([ds[int(i)]])
        with torch.no_grad():
            out = sample(model, schedule, batch["cond"].to(device),
                         batch["occupancy"].to(device),
                         batch["candidate_xy"].to(device),
                         batch["candidate_mask"].to(device),
                         batch["candidate_geometry"].to(device),
                         batch["candidate_geometry_lengths"].to(device),
                         device=device, steps=args.steps, seed=args.seed + n,
                         return_trace=True)
        res = int(batch["occupancy"].shape[-1])
        occ = batch["occupancy"][0, 0].numpy()
        cond = batch["cond"][0].numpy()
        gt = batch["pos"][0].numpy()
        gtc = batch["control_gt"][0].numpy()
        pred = out["p"][0].cpu().numpy()
        pc = out["control"][0].cpu().numpy()
        sel = int(out["selected_idx"][0])
        best = int(batch["topology_best"][0])
        pi = out["topology_pi"][0].cpu().numpy()
        center = out["ellipse_center"][0].cpu().numpy()
        a = out["ellipse_a"][0].cpu().numpy()
        b = out["ellipse_b"][0].cpu().numpy()
        th = out["ellipse_theta"][0].cpu().numpy()
        mask = batch["candidate_mask"][0].numpy()

        hit = point_hits(pred, occ)
        rmse = float(np.linalg.norm(pred - gt, axis=1).mean()) * METERS
        crmse = float(np.linalg.norm(pc[1:-1] - gtc[1:-1], axis=1).mean()) * METERS
        goal_err = float(np.linalg.norm(pred[-1] - cond[1])) * METERS
        coll = 100.0 * float(hit.mean())
        print("[inspect] %s #%d  curve_rmse=%.3f m  ctrl_rmse=%.3f m  "
              "goal_err=%.4f m  point_collision=%.2f%%  any_collision=%s  "
              "selected=%d%s  pi=%s"
              % (args.split, int(i), rmse, crmse, goal_err, coll,
                 bool(hit.any()), sel,
                 "" if sel == best else " (nDTW-best=%d)" % best,
                 np.round(pi, 3).tolist()), flush=True)

        fig = plt.figure(figsize=(17.5, 10.5), dpi=110)
        gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])

        # --- panel 1: curves + candidate skeletons -------------------------
        ax1.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
        for m in range(len(mask)):
            if mask[m]:
                ax1.plot(*to_px(batch["candidate_xy"][0, m].numpy(), res).T,
                         color="#1f77b4", lw=1.0, alpha=0.9 if m == sel else 0.35)
        ax1.plot(*to_px(gtc, res).T, color="#2ca02c", lw=0.8, marker="o",
                 ms=2.2, alpha=0.65, label="GT controls")
        ax1.plot(*to_px(gt, res).T, color="#2ca02c", lw=2.2, label="GT curve")
        ax1.plot(*to_px(pc, res).T, color="#d62728", lw=0.8, marker="o",
                 ms=2.2, alpha=0.65, label="pred controls")
        ax1.plot(*to_px(pred, res).T, color="#d62728", lw=1.9, ls="--",
                 label="pred curve")
        ax1.scatter(*to_px(cond, res).T, marker="*", s=170, c="k", zorder=6,
                    label="start / goal")
        ax1.set_title("curve + control polygons (blue = candidates)", fontsize=9)
        ax1.legend(loc="upper right", fontsize=6.5)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # --- panel 2: ellipses ---------------------------------------------
        ax2.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
        n_sel = max(2, int(batch["candidate_geometry_lengths"][0, sel]))
        ax2.plot(*to_px(batch["candidate_geometry"][0, sel, :n_sel].numpy(),
                        res).T, color="#1f77b4", lw=1.1, label="selected skeleton")
        draw_ellipses(ax2, center, a, b, th, res)
        cp = to_px(center, res)
        ax2.scatter(cp[:, 0], cp[:, 1], s=1.6, c="#d62728")
        ax2.plot(*to_px(pred, res).T, color="#d62728", lw=1.4, ls="--")
        ax2.scatter(*to_px(cond, res).T, marker="*", s=170, c="k", zorder=6)
        ax2.set_title("128 fixed Skeleton centres + predicted ellipses",
                      fontsize=9)
        ax2.legend(loc="upper right", fontsize=6.5)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # --- panel 3: topology + metrics -----------------------------------
        ax3.bar(np.arange(len(pi)), pi, color=["#d62728" if m == sel else "#9ecae1"
                                               for m in range(len(pi))])
        for m in range(len(pi)):
            if mask[m]:
                ax3.text(m, pi[m] + 0.02, "m*" if m == best else "",
                         ha="center", fontsize=8, color="#2ca02c")
        ax3.set_xticks(np.arange(len(pi)))
        ax3.set_xlabel("candidate index")
        ax3.set_ylabel("pi (masked softmax)")
        ax3.set_title("topology distribution (red = selected, m* = nDTW best)",
                      fontsize=9)
        ax3.set_ylim(0.0, max(1.05, float(pi.max()) * 2.3))
        txt = ("sample #%d (%s)\n"
               "curve RMSE      : %.3f m\n"
               "control RMSE    : %.3f m\n"
               "goal error      : %.4f m\n"
               "point collision : %.2f %%\n"
               "any collision   : %s\n"
               "selected / best : %d / %d\n"
               "pi              : %s"
               % (int(i), args.split, rmse, crmse, goal_err, coll,
                  "YES" if hit.any() else "no", sel, best,
                  np.round(pi, 3).tolist()))
        ax3.text(0.03, 0.97, txt, transform=ax3.transAxes, fontsize=8.5,
                 family="monospace", va="top",
                 bbox=dict(facecolor="white", edgecolor="#bbbbbb", alpha=0.9))

        # --- panel 4: DDIM reverse trace -----------------------------------
        trace = out["trace"]
        steps = list(range(0, len(trace), max(1, len(trace) // 6)))[:6]
        for j, s_i in enumerate(steps):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(occ, origin="lower", cmap="gray_r",
                      interpolation="nearest")
            st = trace[s_i]
            ax.plot(*to_px(st["p"][0].numpy(), res).T, color="#ff7f0e",
                    lw=1.0, alpha=0.8, label="q_t decoded")
            ax.plot(*to_px(st["final"][0].numpy(), res).T, color="#d62728",
                    lw=1.4, label="x0_hat")
            ax.plot(*to_px(gt, res).T, color="#2ca02c", lw=1.1, alpha=0.8)
            ax.set_title("t=%d -> %d" % (st["t"], st["s"]), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.legend(loc="upper right", fontsize=6)
        fig.suptitle("inspect %s #%d  (green = GT, red = predicted, "
                     "orange = noisy state decoded)" % (args.split, int(i)),
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(args.out, "inspect_%s_%d.png" % (args.split, int(i)))
        fig.savefig(path)
        plt.close(fig)
        print("[inspect] saved", path, flush=True)


if __name__ == "__main__":
    main()
