"""Zoomed, collision-annotated preview for one split.

The standard preview grid draws a 256x256 occupancy into ~150 px, i.e. ~0.6 px
per cell.  A lane here is only ~8-10 cells (5-6 m) wide, so a 1-2 m trajectory
error is SUB-PIXEL and invisible.  This script therefore draws, for each picked
sample, the full scene next to a NATIVE-RESOLUTION zoom around the path.

Two DIFFERENT failure modes are marked separately, because collapsing them into
one "collision" number is misleading:

    ORANGE  circle   point lands on an OBSTACLE cell           (real collision)
    MAGENTA square   point leaves the mapped window [-1,1]^2   (out of bounds;
                     the 160 m window simply does not cover it)

The predicted safety ellipses are drawn dashed cyan.

    python scripts/preview_zoom.py --config configs/config_160.yaml \
        --ckpt outputs/bspline_carla_160/ckpt/best_task.pt --split val \
        --pool 128 --num 4 --out outputs/bspline_carla_160/zoom_val
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
from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate

RES = 256
CELL = 2.0 / RES          # scene units per cell
ZOOM_MARGIN = 26          # cells of context around the path bbox
PAD = 0.10                # scene-unit margin drawn OUTSIDE the mapped window


def classify(points, occ):
    """-> (on_obstacle, outside_window) boolean masks, never merged."""
    px = np.rint((points[:, 0] + 1.0) / 2.0 * RES - 0.5).astype(int)
    py = np.rint((points[:, 1] + 1.0) / 2.0 * RES - 0.5).astype(int)
    inside = (px >= 0) & (px < RES) & (py >= 0) & (py < RES)
    on_obs = np.zeros(len(points), dtype=bool)
    on_obs[inside] = occ[py[inside], px[inside]] > 0
    return on_obs, ~inside


def clearance_scene_m(points, occ, meters):
    """Distance from each point to the nearest obstacle cell, in metres."""
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(occ == 0)
    px = np.clip(np.rint((points[:, 0] + 1.0) / 2.0 * RES - 0.5).astype(int), 0, RES - 1)
    py = np.clip(np.rint((points[:, 1] + 1.0) / 2.0 * RES - 0.5).astype(int), 0, RES - 1)
    return dt[py, px] * CELL * meters


def draw_ellipses(ax, centers, a, b, theta, stride=4, color="#17becf",
                  lw=0.6, alpha=0.55, zorder=3):
    ang = np.linspace(0.0, 2.0 * np.pi, 48)
    for k in range(0, len(centers), max(1, stride)):
        co, si = float(np.cos(theta[k])), float(np.sin(theta[k]))
        R = np.array([[co, -si], [si, co]])
        e = np.stack([a[k] * np.cos(ang), b[k] * np.sin(ang)], 1) @ R.T \
            + centers[k]
        ax.plot(e[:, 0], e[:, 1], "-", color=color, lw=lw, alpha=alpha,
                zorder=zorder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_160.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-alm", action="store_true",
                    help="raw network only; by default the alm/corridor sections "
                         "of the config are used, exactly like evaluate.py")
    ap.add_argument("--no-ellipses", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    meters = float((cfg.get("data") or {}).get("scene_to_meter", 80.0))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = cfg["data"].get("processed_root", "data/carla_processed")
    geo = int((cfg.get("topology") or {}).get("candidate_geometry_points", 1280))

    # sample() leaves ALM OFF unless alm_config is passed (see preview_samples).
    alm_cfg = dict(cfg.get("alm") or {})
    corridor_cfg = dict(cfg.get("corridor") or {})
    if args.no_alm:
        from src.diffusion.sampler import ablation_configs
        alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg, "A")
    print("[zoom] ALM enabled=%s mode=%s corridor=%s ellipses=%s"
          % (bool(alm_cfg.get("enabled")), alm_cfg.get("mode"),
             bool(corridor_cfg.get("enabled")), not args.no_ellipses), flush=True)

    ds = CarlaSplineDataset(args.split, root, geometry_points=geo,
                            limit=args.pool, indices=list(range(args.pool)),
                            require_labels=False)
    model, _, _ = load_model(cfg, args.ckpt, arch="auto", device=device)
    model.eval()
    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get(
                              "beta_schedule", "squaredcos_cap_v2")).to(device)

    n = len(ds)
    P, GT, OCC, SEL, BEST, COND = [], [], [], [], [], []
    EC, EA, EB, ET = [], [], [], []
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            idxs = list(range(s, min(s + args.batch_size, n)))
            b = make_collate(ds)([ds[i] for i in idxs])
            out = sample(model, sched, b["cond"].to(device),
                         b["occupancy"].to(device),
                         b["candidate_xy"].to(device),
                         b["candidate_mask"].to(device),
                         b["candidate_geometry"].to(device),
                         b["candidate_geometry_lengths"].to(device),
                         device=device, steps=args.steps, seed=args.seed,
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
            print("[zoom] %d/%d" % (min(s + args.batch_size, n), n), flush=True)
    P = np.concatenate(P); GT = np.concatenate(GT); OCC = np.concatenate(OCC)
    SEL = np.concatenate(SEL); BEST = np.concatenate(BEST); COND = np.concatenate(COND)
    EC = np.concatenate(EC); EA = np.concatenate(EA)
    EB = np.concatenate(EB); ET = np.concatenate(ET)

    rmse, obs_rate, out_rate = [], [], []
    OBS, OUT = [], []
    for k in range(n):
        rmse.append(float(np.linalg.norm(P[k] - GT[k], axis=1).mean()) * meters)
        o, u = classify(P[k], OCC[k])
        OBS.append(o); OUT.append(u)
        obs_rate.append(float(o.mean()))
        out_rate.append(float(u.mean()))
    rmse = np.asarray(rmse); obs_rate = np.asarray(obs_rate)
    out_rate = np.asarray(out_rate)

    print("[zoom] pool=%d  rmse mean=%.2f m  |  points ON OBSTACLE=%.2f%%  "
          "points OUT OF WINDOW=%.2f%%  (any-failure %.1f%%)"
          % (n, rmse.mean(), 100 * obs_rate.mean(), 100 * out_rate.mean(),
             100 * ((obs_rate > 0) | (out_rate > 0)).mean()), flush=True)

    order = np.argsort(rmse)
    qs = np.linspace(0.03, 0.97, args.num)
    picks = [int(order[min(n - 1, int(q * (n - 1)))]) for q in qs]

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(len(picks), 2, figsize=(11.5, 5.0 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for r, k in enumerate(picks):
        occ = OCC[k]
        gt, pr = GT[k], P[k]
        o_mask, u_mask = OBS[k], OUT[k]
        axL = axes[r, 0]
        axL.set_facecolor("#e6e6e6")           # everything outside the window
        axL.imshow(occ, cmap="gray_r", origin="lower", extent=[-1, 1, -1, 1],
                   interpolation="nearest", vmin=0, vmax=1, zorder=1)
        axL.plot(gt[:, 0], gt[:, 1], "-", color="#2ca02c", lw=1.4, zorder=4)
        axL.plot(pr[:, 0], pr[:, 1], "-", color="#d62728", lw=1.2, zorder=4)
        if not args.no_ellipses:
            draw_ellipses(axL, EC[k], EA[k], EB[k], ET[k], stride=6)
        if o_mask.any():
            axL.plot(pr[o_mask, 0], pr[o_mask, 1], "o", color="#ff7f0e", ms=3.2,
                     mec="none", zorder=6)
        if u_mask.any():
            axL.plot(pr[u_mask, 0], pr[u_mask, 1], "s", color="#c2185b", ms=3.2,
                     mec="none", zorder=6)
        axL.plot([COND[k, 0, 0]], [COND[k, 0, 1]], "k*", ms=9, zorder=7)
        axL.plot([COND[k, 1, 0]], [COND[k, 1, 1]], "k*", ms=9, zorder=7)
        axL.set_xlim(-1 - PAD, 1 + PAD); axL.set_ylim(-1 - PAD, 1 + PAD)
        axL.set_title("sample #%d  rmse=%.2f m  on-obstacle=%.0f%%  "
                      "out-of-window=%.0f%%  m=%d (best=%d)   [full 160 m scene]"
                      % (k, rmse[k], 100 * obs_rate[k], 100 * out_rate[k],
                         SEL[k], BEST[k]), fontsize=9)
        axL.set_xticks([]); axL.set_yticks([])

        pts = np.vstack([gt, pr])
        px = (pts[:, 0] + 1.0) / 2.0 * RES
        py = (pts[:, 1] + 1.0) / 2.0 * RES
        x0 = int(max(0, np.floor(px.min()) - ZOOM_MARGIN))
        x1 = int(min(RES, np.ceil(px.max()) + ZOOM_MARGIN))
        y0 = int(max(0, np.floor(py.min()) - ZOOM_MARGIN))
        y1 = int(min(RES, np.ceil(py.max()) + ZOOM_MARGIN))
        ext = [-1 + x0 * CELL, -1 + x1 * CELL, -1 + y0 * CELL, -1 + y1 * CELL]
        axR = axes[r, 1]
        axR.set_facecolor("#e6e6e6")
        axR.imshow(occ[y0:y1, x0:x1], cmap="gray_r", origin="lower", extent=ext,
                   interpolation="nearest", vmin=0, vmax=1, zorder=1)
        axR.plot(gt[:, 0], gt[:, 1], "-", color="#2ca02c", lw=2.0, zorder=4)
        axR.plot(pr[:, 0], pr[:, 1], "-", color="#d62728", lw=1.8, zorder=4)
        if not args.no_ellipses:
            draw_ellipses(axR, EC[k], EA[k], EB[k], ET[k], stride=6, lw=0.9)
        if o_mask.any():
            axR.plot(pr[o_mask, 0], pr[o_mask, 1], "o", color="#ff7f0e", ms=4.5,
                     mec="none", zorder=6)
        if u_mask.any():
            axR.plot(pr[u_mask, 0], pr[u_mask, 1], "s", color="#c2185b", ms=4.5,
                     mec="none", zorder=6)
        cl = clearance_scene_m(pr, occ, meters)
        clg = clearance_scene_m(gt, occ, meters)
        axR.set_xlim(ext[0] - PAD, ext[1] + PAD)
        axR.set_ylim(ext[2] - PAD, ext[3] + PAD)
        axR.set_title("ZOOMED %.2f m/cell | pred clearance p50=%.2f m  "
                      "GT clearance p50=%.2f m | orange=ON OBSTACLE  "
                      "magenta=OUT OF WINDOW  cyan=predicted ellipses"
                      % (CELL * meters, np.median(cl), np.median(clg)), fontsize=9)
        axR.set_xticks([]); axR.set_yticks([])

    fig.suptitle("%s | %s | pool=%d  rmse mean=%.2f m | ALM=%s | "
                 "on-obstacle %.1f%% of points, out-of-window %.1f%%"
                 % (args.ckpt, args.split, n, rmse.mean(),
                    "ON" if alm_cfg.get("enabled") else "OFF",
                    100 * obs_rate.mean(), 100 * out_rate.mean()), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    dest = os.path.join(args.out, "zoom_%s.png" % args.split)
    fig.savefig(dest, dpi=115)
    print("saved", dest)
    print("picked:", picks)
    print("rmse_m      :", [round(rmse[k], 2) for k in picks])
    print("on-obstacle%:", [round(100 * obs_rate[k], 1) for k in picks])
    print("out-window% :", [round(100 * out_rate[k], 1) for k in picks])
    with open(os.path.join(args.out, "zoom_%s.json" % args.split), "w",
              encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "split": args.split, "pool": n,
                   "alm_enabled": bool(alm_cfg.get("enabled")),
                   "rmse_m_mean": float(rmse.mean()),
                   "on_obstacle_point_rate": float(obs_rate.mean()),
                   "out_of_window_point_rate": float(out_rate.mean()),
                   "any_failure_rate": float(((obs_rate > 0)
                                              | (out_rate > 0)).mean()),
                   "picked": picks,
                   "picked_rmse_m": [float(rmse[k]) for k in picks],
                   "picked_on_obstacle": [float(obs_rate[k]) for k in picks],
                   "picked_out_window": [float(out_rate[k]) for k in picks]},
                  f, indent=2)


if __name__ == "__main__":
    main()
