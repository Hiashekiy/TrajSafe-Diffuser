"""Evaluate the control-space (32 B-spline controls) TrajSafe-Diffuser.

Metrics:
  curve       collision rate, goal distance, smoothness, cross-seed diversity,
              curve RMSE vs the GT curve (m), B-spline control RMSE (m)
              (the curve is decoded from Q_final only for these metrics)
  centre      free-space rate and minimum clearance (c_i = Gamma_m(i/(Q-1)))
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
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config, num_controls, num_safety_queries
from src.utils.checkpoint import load_model
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,
                                               make_collate)
from src.geometry.skeleton_paths import normalized_dtw, resample_polyline

# Metres per scene unit; DATA, read from data.scene_to_meter in main().
SCENE_TO_METER = 40.0


def set_scene_to_meter(cfg, default: float = 40.0) -> float:
    global SCENE_TO_METER
    try:
        SCENE_TO_METER = float((cfg.get("data") or {}).get(
            "scene_to_meter", default))
    except (TypeError, ValueError):
        SCENE_TO_METER = float(default)
    return SCENE_TO_METER


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
    ap.add_argument("--ablation", default=None, choices=["A", "B", "C", "D"],
                    help="report section 48 ablation preset "
                         "A raw / B final-only / C guided / D no-bridge")
    ap.add_argument("--compare-raw", action="store_true",
                    help="also run one raw (ALM disabled) sample per batch so "
                         "the collision delta is reported")
    ap.add_argument("--arch", default="auto",
                    choices=["auto", "control", "legacy"],
                    help="forward chain: auto = from the checkpoint (legacy "
                         "checkpoints replay the 128-curve-token chain)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("[evaluate] scene_to_meter=%.1f m/unit (reporting only)"
          % set_scene_to_meter(cfg), flush=True)
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    processed_root = cfg["data"].get("processed_root", "data/carla_processed")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))
    batch_size = int(cfg["data"].get("batch_size", 16))
    n_ctrl = num_controls(cfg)
    n_safety = num_safety_queries(cfg)

    ds = CarlaSplineDataset(args.split, processed_root,
                            geometry_points=geo_points, require_labels=False,
                            num_controls=n_ctrl, num_safety_queries=n_safety)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model, ckpt, _ = load_model(cfg, args.ckpt, arch=args.arch, device=device)
    model.eval()
    print("[eval] arch=%s controls=%d"
          % ("control_space" if model.control_space else "legacy_curve",
             model.num_controls), flush=True)

    agg = {k: [] for k in
           ["traj_collision", "goal_dist_m", "smooth", "center_free",
            "center_min_clearance", "ellipse_collision", "area",
            "curve_rmse_m", "ctrl_rmse_m", "progress_violations",
            "topo_entropy", "selected_ndtw", "recall", "topo_diversity",
            "traj_diversity", "step_jitter", "step_switch_rate",
            "sel_best_rate", "pred_topo_best_rate"]}
    # --- corridor / ALM bridge metrics (report section 47) -----------------
    alm_agg = {k: [] for k in
               ["alm_activation_rate", "topology_fallback_rate",
                "activation_step", "activation_attempts",
                "corridor_base_cells", "corridor_bridge_cells",
                "corridor_min_overlap", "corridor_mean_overlap",
                "corridor_region_faces", "constraint_pieces",
                "constraint_active_faces",
                "alm_max_violation_before", "alm_max_violation_after",
                "alm_mean_violation_before", "alm_mean_violation_after",
                "alm_feasible_rate", "alm_curve_correction_m",
                "alm_lambda_max", "alm_inner_steps", "alm_runtime_ms",
                "final_collision", "final_max_constraint_violation",
                "final_corridor_membership_rate", "final_endpoint_error",
                "raw_collision"]}
    rng = np.random.default_rng(0)

    alm_cfg, corridor_cfg = cfg.get("alm") or {}, cfg.get("corridor") or {}
    if args.ablation:
        from src.diffusion.sampler import ablation_configs
        alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg,
                                                 args.ablation)
        print("[eval] ablation %s -> alm.mode=%s enabled=%s bridge=%s"
              % (args.ablation, alm_cfg.get("mode"), alm_cfg.get("enabled"),
                 (corridor_cfg.get("bridge") or {}).get("enabled")),
              flush=True)

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
            started = time.perf_counter()
            runs.append(sample(
                model, schedule, cond, occ_t,
                batch["candidate_xy"].to(device), mask,
                batch["candidate_geometry"].to(device),
                batch["candidate_geometry_lengths"].to(device),
                device=device, steps=args.steps, seed=1000 * bi + r,
                return_trace=True, alm_config=alm_cfg,
                corridor_config=corridor_cfg))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            alm_agg["alm_runtime_ms"].append(
                (time.perf_counter() - started) * 1000.0 / len(idxs))

        raw_runs = []
        if args.compare_raw:
            from src.diffusion.sampler import ablation_configs as _abl
            raw_alm, raw_cor = _abl(alm_cfg, corridor_cfg, "A")
            raw_runs.append(sample(
                model, schedule, cond, occ_t,
                batch["candidate_xy"].to(device), mask,
                batch["candidate_geometry"].to(device),
                batch["candidate_geometry_lengths"].to(device),
                device=device, steps=args.steps, seed=1000 * bi + 0,
                return_trace=False, alm_config=raw_alm,
                corridor_config=raw_cor))
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

            # ---------------- corridor / ALM bridge diagnostics ------------
            run0 = runs[0]
            alm_agg["alm_activation_rate"].append(
                float(bool(run0["guided"][b])))
            alm_agg["topology_fallback_rate"].append(float(
                run0["activation_info"]["topology_fallback"][b]))
            alm_agg["activation_step"].append(
                float(run0["activation_step"][b]) if bool(run0["guided"][b])
                else float("nan"))
            alm_agg["activation_attempts"].append(
                float(run0["activation_info"]["attempts"][b]))
            corridor = run0["corridors"][b]
            if corridor is not None:
                alm_agg["corridor_base_cells"].append(
                    float(corridor["base_cell_count"]))
                alm_agg["corridor_bridge_cells"].append(
                    float(corridor["bridge_cell_count"]))
                ov = corridor["overlap_ratio"]
                alm_agg["corridor_min_overlap"].append(
                    float(min(ov)) if ov else float("nan"))
                alm_agg["corridor_mean_overlap"].append(
                    float(sum(ov) / len(ov)) if ov else float("nan"))
                alm_agg["corridor_region_faces"].append(float(
                    sum(c["face_count"] for c in corridor["cells"])))
            pack = run0["pack_summary"]
            if pack is not None:
                alm_agg["constraint_pieces"].append(
                    float(max(pack["num_pieces"])))
                alm_agg["constraint_active_faces"].append(
                    float(pack["num_active_constraints"]))
            stats = [s["alm_stats"] for s in run0["trace"]
                     if s["alm_stats"] is not None]
            if stats:
                alm_agg["alm_max_violation_before"].append(float(np.mean(
                    [float(s["max_violation_before"][b]) for s in stats])))
                alm_agg["alm_max_violation_after"].append(float(np.mean(
                    [float(s["max_violation_after"][b]) for s in stats])))
                alm_agg["alm_mean_violation_before"].append(float(np.mean(
                    [float(s["mean_positive_violation_before"][b])
                     for s in stats])))
                alm_agg["alm_mean_violation_after"].append(float(np.mean(
                    [float(s["mean_positive_violation_after"][b])
                     for s in stats])))
                alm_agg["alm_feasible_rate"].append(float(np.mean(
                    [float(s["constraint_feasible_rate"][b])
                     for s in stats])))
                alm_agg["alm_curve_correction_m"].append(float(np.mean(
                    [float(s["max_curve_correction_m"][b])
                     for s in stats])))
                alm_agg["alm_lambda_max"].append(float(np.mean(
                    [float(s["lambda_max"][b]) for s in stats])))
                alm_agg["alm_inner_steps"].append(float(np.mean(
                    [float(s["inner_steps_used"][b]) for s in stats])))
            final = run0["final_validation"][b]
            alm_agg["final_collision"].append(float(final["final_collision"]))
            alm_agg["final_endpoint_error"].append(
                float(final["endpoint_error"]))
            if final["final_max_constraint_violation"] is not None:
                alm_agg["final_max_constraint_violation"].append(
                    float(final["final_max_constraint_violation"]))
            if final["final_corridor_membership_rate"] is not None:
                alm_agg["final_corridor_membership_rate"].append(
                    float(final["final_corridor_membership_rate"]))
            raw_run = raw_runs[0] if raw_runs else None
            if raw_run is not None:
                alm_agg["raw_collision"].append(
                    float(raw_run["final_validation"][b]["final_collision"]))
        print("[eval] batch %d/%d" % (bi + 1, args.num_batches), flush=True)

    def _mean(values):
        finite = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(finite)) if finite else None

    summary = {}
    for k, v in agg.items():
        if not v:
            summary[k] = None
        elif k == "progress_violations":
            summary[k] = float(np.sum(v))
        else:
            summary[k] = float(np.mean(v))
    for k, v in alm_agg.items():
        summary[k] = _mean(v)
    summary.update({"M": args.max_candidates or int(
        (cfg.get("topology") or {}).get("num_candidates", 4)),
        "steps": args.steps, "runs": args.runs, "ckpt": args.ckpt,
        "epoch": ckpt.get("epoch"), "split": args.split,
        "ablation": args.ablation,
        "alm_settings": {"enabled": alm_cfg.get("enabled"),
                         "mode": alm_cfg.get("mode"),
                         "warmup_reverse_steps":
                             alm_cfg.get("warmup_reverse_steps"),
                         "max_activation_delay_steps":
                             alm_cfg.get("max_activation_delay_steps"),
                         "bridge_enabled":
                             (corridor_cfg.get("bridge") or {}).get("enabled")}})
    out = args.out or os.path.join(
        os.path.dirname(args.ckpt),
        "eval_%s_M%s%s.json" % (args.split, summary["M"],
                                ("_abl%s" % args.ablation)
                                if args.ablation else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
