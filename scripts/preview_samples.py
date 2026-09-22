"""preview_samples.py - qualitative preview of a whole pool of samples.

Runs the ACTUAL DDIM sampler (argmax(pi) routing) on a pool of samples of one
split, computes per-sample metrics, then draws two comparison grids:

  * <name>_curves.png    occupancy + GT curve + predicted curve + GT/pred
                         control polygons + all candidate skeletons
  * <name>_ellipses.png  occupancy + 128 fixed Skeleton centres + predicted
                         ellipses (+ predicted curve)

The samples are sorted by curve RMSE and picked evenly across the quality range,
so the grid shows the best, the typical and the worst cases.

    python scripts/preview_samples.py --config configs/config.yaml \
        --ckpt outputs/bspline_carla/ckpt/best_task.pt \
        --split test --pool 128 --num 8 --out outputs/bspline_carla/preview_test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.checkpoint import load_model
from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.trajsafe import TrajSafePlanner
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,
                                               make_collate)

# Metres per scene unit; DATA, set from data.scene_to_meter in main().
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


def draw_ellipses(ax, center, a, b, theta, res, stride=4, color="#d62728",
                  lw=0.6, alpha=0.75):
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
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/bspline_carla/preview")
    ap.add_argument("--no-alm", action="store_true",
                    help="raw network only; by default the alm/corridor sections "
                         "of the config are used, exactly like evaluate.py")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global METERS
    cfg = load_config(args.config)
    METERS = float((cfg.get("data") or {}).get("scene_to_meter", 40.0))
    # sample() leaves ALM OFF unless alm_config is passed, so the preview has to
    # forward the config's alm / corridor sections explicitly.  --no-alm pins it
    # to ablation A (raw diffusion) for a like-for-like raw comparison.
    alm_cfg = dict(cfg.get("alm") or {})
    corridor_cfg = dict(cfg.get("corridor") or {})
    if args.no_alm:
        from src.diffusion.sampler import ablation_configs
        alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg, "A")
    print("[preview] ALM enabled=%s mode=%s corridor=%s"
          % (bool(alm_cfg.get("enabled")), alm_cfg.get("mode"),
             bool(corridor_cfg.get("enabled"))), flush=True)
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    processed_root = cfg["data"].get("processed_root", "data/carla_processed")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))

    ds = CarlaSplineDataset(args.split, processed_root,
                            geometry_points=geo_points, limit=args.pool,
                            indices=list(range(args.offset,
                                               args.offset + args.pool)),
                            require_labels=False)
    # arch='auto': pre-refactor checkpoints replay the legacy curve-token chain
    model, ckpt, _ = load_model(cfg, args.ckpt, arch="auto", device=device)
    model.eval()
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)

    n = len(ds)
    rows = {"p": [], "control": [], "sel": [], "pi": [], "center": [],
            "a": [], "b": [], "theta": [], "gt": [], "cond": [], "occ": [],
            "cand_xy": [], "geo": [], "glen": [], "cand_mask": [], "best": []}
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            idxs = list(range(start, min(start + args.batch_size, n)))
            batch = make_collate(ds)([ds[i] for i in idxs])
            out = sample(model, schedule, batch["cond"].to(device),
                         batch["occupancy"].to(device),
                         batch["candidate_xy"].to(device),
                         batch["candidate_mask"].to(device),
                         batch["candidate_geometry"].to(device),
                         batch["candidate_geometry_lengths"].to(device),
                         device=device, steps=args.steps, seed=args.seed,
                         return_trace=False, alm_config=alm_cfg,
                         corridor_config=corridor_cfg)
            rows["p"].append(out["p"].cpu().numpy())
            rows["control"].append(out["control"].cpu().numpy())
            rows["sel"].append(out["selected_idx"].cpu().numpy())
            rows["pi"].append(out["topology_pi"].cpu().numpy())
            rows["center"].append(out["ellipse_center"].cpu().numpy())
            rows["a"].append(out["ellipse_a"].cpu().numpy())
            rows["b"].append(out["ellipse_b"].cpu().numpy())
            rows["theta"].append(out["ellipse_theta"].cpu().numpy())
            rows["gt"].append(batch["pos"].numpy())
            rows["cond"].append(batch["cond"].numpy())
            rows["occ"].append(batch["occupancy"].numpy())
            rows["cand_xy"].append(batch["candidate_xy"].numpy())
            rows["geo"].append(batch["candidate_geometry"].numpy())
            rows["glen"].append(batch["candidate_geometry_lengths"].numpy())
            rows["cand_mask"].append(batch["candidate_mask"].numpy())
            rows["best"].append(batch["topology_best"].numpy())
            print("[preview] %d/%d" % (min(start + args.batch_size, n), n),
                  flush=True)
    data = {k: np.concatenate(v, axis=0) for k, v in rows.items()}

    rmse, pf, anyc, topo_ok = [], [], [], []
    for k in range(n):
        occ = data["occ"][k, 0]
        hit = point_hits(data["p"][k], occ)
        rmse.append(float(np.linalg.norm(data["p"][k] - data["gt"][k],
                                         axis=1).mean()) * METERS)
        pf.append(float(hit.mean()))
        anyc.append(float(hit.any()))
        topo_ok.append(int(data["sel"][k]) == int(data["best"][k]))
    rmse = np.asarray(rmse)
    print("[preview] pool=%d  curve_rmse_m mean=%.3f p50=%.3f p95=%.3f max=%.3f"
          % (n, rmse.mean(), np.percentile(rmse, 50), np.percentile(rmse, 95),
             rmse.max()), flush=True)
    print("[preview] point-collision mean=%.5f  any-collision rate=%.4f "
          "sel_best_rate=%.4f" % (np.mean(pf), np.mean(anyc),
                                  np.mean(topo_ok)), flush=True)

    order = np.argsort(rmse)
    picks = list(order[np.linspace(0, n - 1, args.num).astype(int)])
    os.makedirs(args.out, exist_ok=True)
    res = data["occ"].shape[-1]

    def _grid(name, ellipse_panel):
        cols = min(4, len(picks))
        rows_n = int(np.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(3.9 * cols, 3.9 * rows_n),
                                 dpi=110, squeeze=False)
        for ax_i, k in enumerate(picks):
            ax = axes[ax_i // cols][ax_i % cols]
            ax.imshow(data["occ"][k, 0], origin="lower", cmap="gray_r",
                      interpolation="nearest")
            cond = data["cond"][k]
            mx = data["cand_mask"][k]
            for m in range(len(mx)):
                if not mx[m]:
                    continue
                g = data["cand_xy"][k, m]
                ax.plot(*to_px(g, res).T, color="#1f77b4", lw=0.9,
                        alpha=0.55 if ellipse_panel else 0.9)
            ax.plot(*to_px(data["gt"][k], res).T, color="#2ca02c", lw=1.8,
                    label="GT curve")
            ax.plot(*to_px(data["p"][k], res).T, color="#d62728", lw=1.5,
                    ls="--", label="pred curve")
            if not ellipse_panel:
                ax.plot(*to_px(data["control"][k], res).T, color="#d62728",
                        lw=0.7, marker="o", ms=1.6, alpha=0.7,
                        label="pred controls")
            ax.scatter(*to_px(cond, res).T, marker="*", s=140, c="k", zorder=6)
            if ellipse_panel:
                draw_ellipses(ax, data["center"][k], data["a"][k],
                              data["b"][k], data["theta"][k], res)
                cp = to_px(data["center"][k], res)
                ax.scatter(cp[:, 0], cp[:, 1], s=1.4, c="#d62728")
            sel = int(data["sel"][k])
            best = int(data["best"][k])
            ax.set_title("#%d  rmse=%.2fm  coll=%.0f%%  m=%d%s"
                         % (k + args.offset, rmse[k], 100.0 * pf[k], sel,
                            "" if sel == best else " (best=%d)" % best),
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for j in range(len(picks), rows_n * cols):
            axes[j // cols][j % cols].axis("off")
        axes[0][0].legend(loc="upper right", fontsize=6)
        fig.suptitle("%s: %s pool=%d, sorted best -> worst "
                     "(green=GT, red=predicted, blue=candidate skeletons)"
                     % (name, args.split, n), fontsize=10)
        fig.tight_layout()
        path = os.path.join(args.out, "%s.png" % name)
        fig.savefig(path)
        plt.close(fig)
        print("[preview] saved", path, flush=True)
        return path

    p1 = _grid("preview_%s_curves" % args.split, False)
    p2 = _grid("preview_%s_ellipses" % args.split, True)

    with open(os.path.join(args.out, "preview_%s.json" % args.split), "w",
              encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "split": args.split, "pool": n,
                   "steps": args.steps, "seed": args.seed,
                   "curve_rmse_m": {"mean": float(rmse.mean()),
                                    "p50": float(np.percentile(rmse, 50)),
                                    "p95": float(np.percentile(rmse, 95)),
                                    "max": float(rmse.max())},
                   "point_collision_mean": float(np.mean(pf)),
                   "any_collision_rate": float(np.mean(anyc)),
                   "sel_best_rate": float(np.mean(topo_ok)),
                   "picked": [int(k + args.offset) for k in picks],
                   "picked_rmse_m": [float(rmse[k]) for k in picks],
                   "figures": [p1, p2]}, f, indent=2)


if __name__ == "__main__":
    main()
