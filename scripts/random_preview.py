"""random_preview.py - draw N RANDOM samples from a split and render them.

One command, random sample every run (the seed is printed so a run can be
reproduced).  For each sample it draws the full 160 m scene next to a
native-resolution zoom around the path, with:

    green   GT curve
    red     predicted curve
    ORANGE  circle  -> predicted point ON AN OBSTACLE        (real collision)
    MAGENTA square  -> predicted point OUTSIDE the mapped window [-1,1]
    CYAN    dashed  -> predicted safety ellipses
    orange  line    -> ALM safety corridor cell boundaries (when ALM is on)

Examples
--------
    # 8 random samples from val (default), best checkpoint, ALM on
    python scripts/random_preview.py --split val

    # 12 random samples from test
    python scripts/random_preview.py --split test --num 12

    # reproducible draw
    python scripts/random_preview.py --split train --num 8 --seed 1234

    # raw network, no ALM post-processing
    python scripts/random_preview.py --split val --no-alm

    # a different model / cache
    python scripts/random_preview.py --split val --ckpt outputs/.../epoch_200.pt
    python scripts/random_preview.py --split val --processed data/carla_processed_160
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate
from src.diffusion.sampler import sample
from src.diffusion.schedule import NoiseSchedule
from src.utils.checkpoint import load_model
from src.utils.config import load_config

RES = 256
CELL = 2.0 / RES
ZOOM_MARGIN = 26
PAD = 0.10


def classify(points, occ):
    """(on_obstacle, outside_window) - the two failures are NEVER merged."""
    px = np.rint((points[:, 0] + 1.0) / 2.0 * RES - 0.5).astype(int)
    py = np.rint((points[:, 1] + 1.0) / 2.0 * RES - 0.5).astype(int)
    inside = (px >= 0) & (px < RES) & (py >= 0) & (py < RES)
    on_obs = np.zeros(len(points), dtype=bool)
    on_obs[inside] = occ[py[inside], px[inside]] > 0
    return on_obs, ~inside


def clearance_m(points, occ, meters):
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(occ == 0)
    px = np.clip(np.rint((points[:, 0] + 1.0) / 2.0 * RES - 0.5).astype(int), 0, RES - 1)
    py = np.clip(np.rint((points[:, 1] + 1.0) / 2.0 * RES - 0.5).astype(int), 0, RES - 1)
    return dt[py, px] * CELL * meters


def draw_ellipses(ax, centers, a, b, theta, stride=6, lw=0.9, alpha=0.6):
    ang = np.linspace(0.0, 2.0 * np.pi, 48)
    for k in range(0, len(centers), max(1, stride)):
        co, si = float(np.cos(theta[k])), float(np.sin(theta[k]))
        R = np.array([[co, -si], [si, co]])
        e = np.stack([a[k] * np.cos(ang), b[k] * np.sin(ang)], 1) @ R.T \
            + centers[k]
        ax.plot(e[:, 0], e[:, 1], "-", color="#17becf", lw=lw, alpha=alpha,
                zorder=3)


def draw_corridor(ax, corridor, color="#ff9f40", lw=1.0, alpha=0.9):
    """Outline the ALM safety-corridor cells that actually closed."""
    if not corridor:
        return
    for cell in corridor.get("cells", []):
        poly = cell.get("polygon") or []
        if len(poly) < 3 or not cell.get("valid", True):
            continue
        p = np.asarray(poly, dtype=np.float64)
        ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]], "-",
                color=color, lw=lw, alpha=alpha, zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_160.yaml")
    ap.add_argument("--ckpt", default="outputs/bspline_carla_160/ckpt/best_task.pt")
    ap.add_argument("--processed", default=None,
                    help="processed cache root; default = data.processed_root")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="which split to draw the random samples FROM")
    ap.add_argument("--num", type=int, default=8, help="how many samples to draw")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: random every run (the used seed is printed)")
    ap.add_argument("--no-alm", action="store_true",
                    help="raw network; default uses the config alm/corridor sections")
    ap.add_argument("--no-ellipses", action="store_true")
    ap.add_argument("--no-corridor", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=105)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(seed)

    cfg = load_config(args.config)
    meters = float((cfg.get("data") or {}).get("scene_to_meter", 80.0))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = args.processed or cfg["data"].get("processed_root", "data/carla_processed")
    geo = int((cfg.get("topology") or {}).get("candidate_geometry_points", 1280))

    alm_cfg = dict(cfg.get("alm") or {})
    corridor_cfg = dict(cfg.get("corridor") or {})
    if args.no_alm:
        from src.diffusion.sampler import ablation_configs
        alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg, "A")

    # ---- draw the random indices ------------------------------------------
    probe = CarlaSplineDataset(args.split, root, geometry_points=geo,
                               require_labels=False)
    n_total = len(probe)
    num = min(int(args.num), n_total)
    picks = np.sort(rng.choice(n_total, size=num, replace=False)).tolist()
    print("split      : %s   (total %d samples in %s)" % (args.split, n_total, root))
    print("seed       : %d   -> indices %s" % (seed, picks))
    print("checkpoint : %s" % args.ckpt)
    print("ALM        : %s   corridor=%s  ellipses=%s"
          % ("ON" if alm_cfg.get("enabled") else "OFF",
             bool(corridor_cfg.get("enabled")), not args.no_ellipses))

    ds = CarlaSplineDataset(args.split, root, geometry_points=geo,
                            indices=picks, require_labels=False)
    model, ckpt, _ = load_model(cfg, args.ckpt, arch="auto", device=device)
    model.eval()
    print("epoch      : %s" % (ckpt.get("epoch") if isinstance(ckpt, dict) else "?"))
    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get(
                              "beta_schedule", "squaredcos_cap_v2")).to(device)

    P, GT, OCC, SEL, BEST, COND = [], [], [], [], [], []
    EC, EA, EB, ET = [], [], [], []
    COR = []
    with torch.no_grad():
        for s in range(0, num, 16):
            idxs = list(range(s, min(s + 16, num)))
            b = make_collate(ds)([ds[i] for i in idxs])
            out = sample(model, sched, b["cond"].to(device),
                         b["occupancy"].to(device),
                         b["candidate_xy"].to(device),
                         b["candidate_mask"].to(device),
                         b["candidate_geometry"].to(device),
                         b["candidate_geometry_lengths"].to(device),
                         device=device, steps=args.steps, seed=0,
                         return_trace=False, alm_config=alm_cfg,
                         corridor_config=corridor_cfg)
            P.append(out["p"].cpu().numpy())
            GT.append(b["pos"].numpy())
            OCC.append(b["occupancy"].numpy()[:, 0])
            SEL.append(out["selected_idx"].cpu().numpy())
            BEST.append(b["topology_best"].numpy())
            COND.append(b["cond"].numpy())
            EC.append(out["ellipse_center"].cpu().numpy())
            EA.append(out["ellipse_a"].cpu().numpy())
            EB.append(out["ellipse_b"].cpu().numpy())
            ET.append(out["ellipse_theta"].cpu().numpy())
            # sample() exposes the activated corridors under "corridors"
            # (already .to_dict()-ed); missing entirely when ALM is off.
            COR.extend(out.get("corridors") or [None] * len(idxs))
    P = np.concatenate(P); GT = np.concatenate(GT); OCC = np.concatenate(OCC)
    SEL = np.concatenate(SEL); BEST = np.concatenate(BEST); COND = np.concatenate(COND)
    EC = np.concatenate(EC); EA = np.concatenate(EA)
    EB = np.concatenate(EB); ET = np.concatenate(ET)
    if len(COR) != num:
        COR = [None] * num

    rmse, orate, urate = [], [], []
    OBS, OUT = [], []
    for k in range(num):
        rmse.append(float(np.linalg.norm(P[k] - GT[k], axis=1).mean()) * meters)
        o, u = classify(P[k], OCC[k])
        OBS.append(o); OUT.append(u)
        orate.append(float(o.mean())); urate.append(float(u.mean()))
    rmse = np.asarray(rmse); orate = np.asarray(orate); urate = np.asarray(urate)
    # curve roughness: mean |p[i+1] - 2 p[i] + p[i-1]| in metres.  This is
    # the number that exposes the ALM's per-step projection kinks; it is
    # independent of how far the path is from the GT.
    rough = np.asarray([
        float(np.linalg.norm(P[k][2:] - 2.0 * P[k][1:-1] + P[k][:-2],
                             axis=-1).mean()) * meters for k in range(num)])
    gt_rough = np.asarray([
        float(np.linalg.norm(GT[k][2:] - 2.0 * GT[k][1:-1] + GT[k][:-2],
                             axis=-1).mean()) * meters for k in range(num)])

    # ---- draw --------------------------------------------------------------
    rows = int(np.ceil(num / 2.0))
    fig, axes = plt.subplots(rows, 4, figsize=(4.6 * 4, 4.6 * rows),
                             squeeze=False)
    for k in range(num):
        r, half = divmod(k, 2)
        axF, axZ = axes[r][half * 2], axes[r][half * 2 + 1]
        occ, gt, pr = OCC[k], GT[k], P[k]
        o_mask, u_mask = OBS[k], OUT[k]

        axF.set_facecolor("#e6e6e6")
        axF.imshow(occ, cmap="gray_r", origin="lower", extent=[-1, 1, -1, 1],
                   interpolation="nearest", vmin=0, vmax=1, zorder=1)
        if not args.no_corridor:
            draw_corridor(axF, COR[k])
        if not args.no_ellipses:
            draw_ellipses(axF, EC[k], EA[k], EB[k], ET[k], stride=8, lw=0.6)
        axF.plot(gt[:, 0], gt[:, 1], "-", color="#2ca02c", lw=1.3, zorder=5)
        axF.plot(pr[:, 0], pr[:, 1], "-", color="#d62728", lw=1.1, zorder=5)
        if o_mask.any():
            axF.plot(pr[o_mask, 0], pr[o_mask, 1], "o", color="#ff7f0e", ms=2.6,
                     mec="none", zorder=6)
        if u_mask.any():
            axF.plot(pr[u_mask, 0], pr[u_mask, 1], "s", color="#c2185b", ms=2.6,
                     mec="none", zorder=6)
        axF.plot([COND[k, 0, 0]], [COND[k, 0, 1]], "k*", ms=8, zorder=7)
        axF.plot([COND[k, 1, 0]], [COND[k, 1, 1]], "k*", ms=8, zorder=7)
        axF.set_xlim(-1 - PAD, 1 + PAD); axF.set_ylim(-1 - PAD, 1 + PAD)
        axF.set_title("[%d] idx %d  rmse %.2f m  obst %.0f%%  out %.0f%%  "
                      "m=%d/best=%d  FULL 160 m"
                      % (k + 1, picks[k], rmse[k], 100 * orate[k],
                         100 * urate[k], SEL[k], BEST[k]), fontsize=9)
        axF.set_xticks([]); axF.set_yticks([])

        pts = np.vstack([gt, pr])
        px = (pts[:, 0] + 1.0) / 2.0 * RES
        py = (pts[:, 1] + 1.0) / 2.0 * RES
        x0 = int(max(0, np.floor(px.min()) - ZOOM_MARGIN))
        x1 = int(min(RES, np.ceil(px.max()) + ZOOM_MARGIN))
        y0 = int(max(0, np.floor(py.min()) - ZOOM_MARGIN))
        y1 = int(min(RES, np.ceil(py.max()) + ZOOM_MARGIN))
        ext = [-1 + x0 * CELL, -1 + x1 * CELL, -1 + y0 * CELL, -1 + y1 * CELL]
        axZ.set_facecolor("#e6e6e6")
        axZ.imshow(occ[y0:y1, x0:x1], cmap="gray_r", origin="lower", extent=ext,
                   interpolation="nearest", vmin=0, vmax=1, zorder=1)
        if not args.no_corridor:
            draw_corridor(axZ, COR[k])
        if not args.no_ellipses:
            draw_ellipses(axZ, EC[k], EA[k], EB[k], ET[k], stride=8)
        axZ.plot(gt[:, 0], gt[:, 1], "-", color="#2ca02c", lw=1.8, zorder=5)
        axZ.plot(pr[:, 0], pr[:, 1], "-", color="#d62728", lw=1.6, zorder=5)
        if o_mask.any():
            axZ.plot(pr[o_mask, 0], pr[o_mask, 1], "o", color="#ff7f0e", ms=3.6,
                     mec="none", zorder=6)
        if u_mask.any():
            axZ.plot(pr[u_mask, 0], pr[u_mask, 1], "s", color="#c2185b", ms=3.6,
                     mec="none", zorder=6)
        cl = clearance_m(pr, occ, meters); clg = clearance_m(gt, occ, meters)
        axZ.set_xlim(ext[0] - PAD, ext[1] + PAD)
        axZ.set_ylim(ext[2] - PAD, ext[3] + PAD)
        axZ.set_title("[%d] ZOOM %.2f m/cell  pred clear %.2f m / GT %.2f m"
                      % (k + 1, CELL * meters, np.median(cl), np.median(clg)),
                      fontsize=9)
        axZ.set_xticks([]); axZ.set_yticks([])
    for k in range(num, rows * 2):
        r, half = divmod(k, 2)
        axes[r][half * 2].axis("off"); axes[r][half * 2 + 1].axis("off")

    fig.suptitle("RANDOM %d from %s | %s | epoch %s | ALM=%s | seed=%d | "
                 "mean rmse %.2f m | points on-obstacle %.1f%%, out-of-window %.1f%%"
                 % (num, args.split, os.path.basename(args.ckpt),
                    ckpt.get("epoch") if isinstance(ckpt, dict) else "?",
                    "ON" if alm_cfg.get("enabled") else "OFF", seed,
                    rmse.mean(), 100 * orate.mean(), 100 * urate.mean()),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    out_dir = args.out or os.path.join(
        os.path.dirname(args.ckpt), "..", "random",
        "%s_%s" % (args.split, time.strftime("%Y%m%d_%H%M%S")))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "random_%s_seed%d.png" % (args.split, seed))
    fig.savefig(dest, dpi=args.dpi)
    print()
    print("mean rmse_m        : %.2f" % rmse.mean())
    print("points ON OBSTACLE : %.2f%%" % (100 * orate.mean()))
    print("points OUT OF WINDOW: %.2f%%" % (100 * urate.mean()))
    print("curve roughness (2nd diff): pred %.4f m   GT %.4f m"
          % (rough.mean(), gt_rough.mean()))
    print("samples w/ any failure: %.1f%%"
          % (100 * ((orate > 0) | (urate > 0)).mean()))
    print("saved              : %s" % dest)
    with open(os.path.join(out_dir, "random_%s_seed%d.json" % (args.split, seed)),
              "w", encoding="utf-8") as fh:
        json.dump({"split": args.split, "ckpt": args.ckpt,
                   "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
                   "seed": int(seed), "indices": picks,
                   "alm_enabled": bool(alm_cfg.get("enabled")),
                   "rmse_m": [float(x) for x in rmse],
                   "on_obstacle_rate": [float(x) for x in orate],
                   "curve_roughness_m": float(rough.mean()),
                   "gt_curve_roughness_m": float(gt_rough.mean()),
                   "out_of_window_rate": [float(x) for x in urate],
                   "figure": dest}, fh, indent=2)


if __name__ == "__main__":
    main()
