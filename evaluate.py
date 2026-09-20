"""Evaluate the control-space (32 B-spline controls) TrajSafe-Diffuser.

Metrics:
  curve       collision rate, goal distance, smoothness, cross-seed diversity,
              curve RMSE vs the GT curve (m), B-spline control RMSE (m)
  centre      free-space rate and minimum clearance (c_i = Gamma_m(i/127))
  ellipse     boundary+interior collision rate, mean area, free fraction
  topology    selected-vs-GT nDTW, Recall@M, entropy, cross-seed diversity,
              per-step selection jitter, argmax(pi)==m* rate
  progress    monotonicity violations (must be 0; the progress is a fixed
              buffer so this only guards against regressions)

    python evaluate.py --config configs/config.yaml \
        --ckpt outputs/bspline_carla/ckpt/best.pt --split test --runs 4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.trajsafe import TrajSafePlanner
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,
                                               make_collate)
from src.geometry.skeleton_paths import normalized_dtw, resample_polyline

SCENE_TO_METER = 40.0


def _collides(points, occ):
    res = occ.shape[0]
    px = np.rint((points[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
    py = np.rint((points[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
    if (px < 0).any() or (px >= res).any() or (py < 0).any() or (py >= res).any():
        return True
    return bool(occ[py, px].astype(bool).any())


def _free(points, occ):
    res = occ.shape[0]
    px = np.rint((points[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
    py = np.rint((points[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
    ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
    out = np.zeros(len(points), dtype=bool)
    out[ok] = ~occ[py[ok], px[ok]].astype(bool)
    return out


def _ellipse_points(center, a, b, theta, n_b=64, n_i=48, rng=None):
    center = np.asarray(center, dtype=np.float64)
    ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
    rng = rng if rng is not None else np.random.default_rng(0)

    def world(ex, ey):
        return np.stack([ct * ex - st * ey + center[:, 0:1],
                         st * ex + ct * ey + center[:, 1:2]], axis=-1)

    ang = np.linspace(0.0, 2 * np.pi, n_b, endpoint=False)[None, :]
    ring = world(a[:, None] * np.cos(ang), b[:, None] * np.sin(ang))
    r = np.sqrt(rng.random((len(center), n_i)))
    th = rng.uniform(0.0, 2 * np.pi, (len(center), n_i))
    inner = world(a[:, None] * r * np.cos(th), b[:, None] * r * np.sin(th))
    return np.concatenate([ring, inner], axis=1)


def _smoothness(p):
    p = np.asarray(p, dtype=float)
    axis = 0 if p.ndim == 2 else 1
    v = np.diff(p, axis=axis)
    acc = np.diff(v, axis=axis)
    step = float(np.linalg.norm(v, axis=-1).mean()) + 1e-9
    return float(np.linalg.norm(acc, axis=-1).mean() / step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-batches", type=int, default=3)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--recall-tau", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    processed_root = cfg["data"].get("processed_root", "data/carla_processed")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))
    batch_size = int(cfg["data"].get("batch_size", 16))

    ds = CarlaSplineDataset(args.split, processed_root,
                            geometry_points=geo_points, require_labels=False)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model = TrajSafePlanner(cfg["model"], cfg.get("ellipse_label"),
                            cfg.get("bspline")).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    agg = {k: [] for k in
           ["traj_collision", "goal_dist_m", "smooth", "center_free",
            "center_min_clearance", "ellipse_collision", "area",
            "curve_rmse_m", "ctrl_rmse_m", "progress_violations",
            "topo_entropy", "selected_ndtw", "recall", "topo_diversity",
            "traj_diversity", "step_jitter", "step_switch_rate",
            "sel_best_rate", "pred_topo_best_rate"]}
    rng = np.random.default_rng(0)

    for bi in range(args.num_batches):
        idxs = list(range(bi * batch_size, min((bi + 1) * batch_size, len(ds))))
        if not idxs:
            break
        batch = make_collate(ds)([ds[i] for i in idxs])
        cond = batch["cond"].to(device)
        occ_t = batch["occupancy"].to(device)
        mask = batch["candidate_mask"].clone()
        if args.max_candidates is not None:
            keep = min(int(args.max_candidates), mask.shape[1])
            mask[:, keep:] = False
        runs = []
        for r in range(max(1, args.runs)):
            runs.append(sample(
                model, schedule, cond, occ_t,
                batch["candidate_xy"].to(device), mask,
                batch["candidate_geometry"].to(device),
                batch["candidate_geometry_lengths"].to(device),
                device=device, steps=args.steps, seed=1000 * bi + r,
                return_trace=True))
        best = batch["topology_best"].numpy()
        occ_all = batch["occupancy"][:, 0].numpy()
        curve_gt = batch["pos"].numpy()
        ctrl_gt = batch["control_gt"].numpy()
        geom = batch["candidate_geometry"].numpy()
        glen = batch["candidate_geometry_lengths"].numpy()
        for b in range(len(idxs)):
            occ_map = occ_all[b]
            p_runs = [run["p"][b].cpu().numpy() for run in runs]
            agg["traj_collision"].append(
                float(np.mean([1.0 if _collides(p, occ_map) else 0.0
                               for p in p_runs])))
            agg["goal_dist_m"].append(float(np.mean(
                [np.linalg.norm(p[-1] - cond[b, 1].cpu().numpy())
                 for p in p_runs])) * SCENE_TO_METER)
            agg["smooth"].append(float(np.mean([_smoothness(p) for p in p_runs])))
            agg["curve_rmse_m"].append(float(np.mean(
                [np.sqrt(((p - curve_gt[b]) ** 2).sum(1).mean())
                 for p in p_runs])) * SCENE_TO_METER)
            agg["ctrl_rmse_m"].append(float(np.mean(
                [np.sqrt(((run["control"][b].cpu().numpy()[1:-1]
                           - ctrl_gt[b][1:-1]) ** 2).sum(1).mean())
                 for run in runs])) * SCENE_TO_METER)
            if len(p_runs) > 1:
                d = [np.linalg.norm(p_runs[i] - p_runs[j])
                     for i in range(len(p_runs)) for j in range(i + 1, len(p_runs))]
                agg["traj_diversity"].append(float(np.mean(d)))

            center = runs[0]["ellipse_center"][b].cpu().numpy()
            a = runs[0]["ellipse_a"][b].cpu().numpy()
            bb = runs[0]["ellipse_b"][b].cpu().numpy()
            th = runs[0]["ellipse_theta"][b].cpu().numpy()
            agg["center_free"].append(float(_free(center, occ_map).mean()))
            j, i = np.nonzero(occ_map.astype(bool))
            if len(j):
                cx = (i + 0.5) * 2.0 / occ_map.shape[0] - 1.0
                cy = (j + 0.5) * 2.0 / occ_map.shape[0] - 1.0
                d = np.sqrt((center[:, 0:1] - cx[None]) ** 2
                            + (center[:, 1:2] - cy[None]) ** 2)
                agg["center_min_clearance"].append(
                    float(d.min() * occ_map.shape[0] / 2.0))
            pts = _ellipse_points(center, a, bb, th, rng=rng)
            agg["ellipse_collision"].append(float(np.mean(
                [1.0 if _collides(p, occ_map) else 0.0 for p in pts])))
            agg["area"].append(float((np.pi * a * bb).mean()))
            s = runs[0]["progress"][b].cpu().numpy()
            agg["progress_violations"].append(float((np.diff(s) < -1e-6).sum()))

            if not bool(mask[b].any()):
                continue
            valid = np.nonzero(mask[b].cpu().numpy())[0]
            pi = runs[0]["topology_pi"][b].cpu().numpy()
            pi = pi / max(pi.sum(), 1e-12)
            nz = pi > 1e-12
            agg["topo_entropy"].append(float(-(pi[nz] * np.log(pi[nz])).sum()))
            gt_poly = curve_gt[b]
            dd = [normalized_dtw(gt_poly, batch["candidate_xy"][b, m].numpy())
                  for m in valid]
            agg["recall"].append(float(min(dd) <= args.recall_tau))
            sel = int(runs[0]["selected_idx"][b])
            sel_poly = (geom[b, sel, :int(glen[b, sel])]
                        if glen[b, sel] > 1 else batch["candidate_xy"][b, sel].numpy())
            agg["selected_ndtw"].append(float(normalized_dtw(
                gt_poly, resample_polyline(sel_poly, 128))))
            agg["sel_best_rate"].append(float(sel == int(best[b])))
            agg["pred_topo_best_rate"].append(
                float(int(pi.argmax()) == int(best[b])))
            sel_all = [int(run["selected_idx"][b]) for run in runs]
            agg["topo_diversity"].append(float(len(set(sel_all)) > 1))
            per_step = [int(step["selected_idx"][b]) for step in runs[0]["trace"]]
            switches = sum(1 for u, v in zip(per_step[:-1], per_step[1:]) if u != v)
            agg["step_jitter"].append(float(switches))
            agg["step_switch_rate"].append(
                float(switches) / max(len(per_step) - 1, 1))
        print("[eval] batch %d/%d" % (bi + 1, args.num_batches), flush=True)

    summary = {}
    for k, v in agg.items():
        if not v:
            summary[k] = None
        elif k == "progress_violations":
            summary[k] = float(np.sum(v))
        else:
            summary[k] = float(np.mean(v))
    summary.update({"M": args.max_candidates or int(
        (cfg.get("topology") or {}).get("num_candidates", 4)),
        "steps": args.steps, "runs": args.runs, "ckpt": args.ckpt,
        "epoch": ckpt.get("epoch"), "split": args.split})
    out = args.out or os.path.join(
        os.path.dirname(args.ckpt),
        "eval_%s_M%s.json" % (args.split, summary["M"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
