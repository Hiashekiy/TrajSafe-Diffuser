"""Evaluate V2 (Skeleton-Topology-Grounded Trajectory Diffusion).

Metrics (docs/V2.md section 38):

  trajectory   collision rate, goal distance, smoothness, diversity
  ellipse      boundary/interior collision rate, area, shape error,
               CenterFree  (MUST be 1.0000 - a lower value means the
                            c_i = gamma_m(s_i) contract was violated)
  topology     entropy, soft-target CE, selected-topology nDTW,
               Candidate Recall@M, cross-seed topology diversity
  progress     MAE, monotonicity violations (must be 0)

    python evaluate_v2.py --config configs/config_v2_skeleton.yaml \
        --ckpt outputs/ckpt_v2_skeleton/best.pt --split test --runs 4
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
from src.diffusion.sampler_v2 import sample_v2
from src.models.skeleton import SkeletonPlanner
from src.datasets.skeleton_dataset import make_loader, MAZE_NAMES
from src.geometry.skeleton_paths import resample_polyline, normalized_dtw
from src.models.skeleton.path_ops import shape4_to_abtheta


def _dense(points, n=384):
    return resample_polyline(points, n)


def _collides(points, occ):
    res = occ.shape[0]
    px = np.rint((points[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
    py = np.rint((points[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
    if (px < 0).any() or (px >= res).any() or (py < 0).any() or (py >= res).any():
        return True
    return bool(occ[py, px].astype(bool).any())


def _ellipse_points(center, shape4, n_b=64, n_i=48, rng=None):
    """Sample boundary + interior points of K ellipses -> [K, n_b + n_i, 2]."""
    center = np.asarray(center, dtype=np.float64)
    shape4 = np.asarray(shape4, dtype=np.float64)
    a = np.exp(shape4[:, 0])
    b = np.exp(shape4[:, 1])
    theta = 0.5 * np.arctan2(shape4[:, 3], shape4[:, 2])
    ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
    rng = rng if rng is not None else np.random.default_rng(0)

    def to_world(ex, ey):
        wx = ct * ex - st * ey + center[:, 0:1]
        wy = st * ex + ct * ey + center[:, 1:2]
        return np.stack([wx, wy], axis=-1)

    ang = np.linspace(0.0, 2.0 * np.pi, n_b, endpoint=False)[None, :]
    boundary = to_world(a[:, None] * np.cos(ang), b[:, None] * np.sin(ang))
    r = np.sqrt(rng.random((len(center), n_i)))
    th = rng.uniform(0.0, 2.0 * np.pi, (len(center), n_i))
    interior = to_world(a[:, None] * r * np.cos(th), b[:, None] * r * np.sin(th))
    return np.concatenate([boundary, interior], axis=1)


def _smoothness(p):
    """Mean |acceleration| normalized by the mean step length (lower = smoother).

    p is ONE trajectory [H, 2], so the difference is taken along the waypoint
    axis (0).  Taking it along axis 1 would differentiate across x/y and always
    return zero.
    """
    p = np.asarray(p, dtype=float)
    axis = 0 if p.ndim == 2 else 1
    v = np.diff(p, axis=axis)
    acc = np.diff(v, axis=axis)
    step = float(np.linalg.norm(v, axis=-1).mean()) + 1e-9
    return float(np.linalg.norm(acc, axis=-1).mean() / step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v2_skeleton.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-batches", type=int, default=4)
    ap.add_argument("--runs", type=int, default=4, help="seeds per OD")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="M ablation: keep only the first M candidate slots")
    ap.add_argument("--recall-tau", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    topo_cfg = cfg.get("topology", {})
    selection = args.selection or topo_cfg.get("selection", "sample")
    commit_t = int(topo_cfg.get("commit_t", 7))
    source = cfg["data"].get("source", "data/processed_scene_v1")
    base_dir = cfg["data"].get("base", "data/processed_scene_v2")

    loader, ds = make_loader(args.split, source, base_dir,
                             batch_size=int(cfg["data"].get("batch_size", 32)),
                             shuffle=False, num_workers=0,
                             mask_res=int(cfg["loss"].get("ellipse_safe_res", 64)),
                             mask_tau=float(cfg["loss"].get("ellipse_mask_tau", 10.0)))
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model = SkeletonPlanner(cfg["model"], topo_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    occ_all = [ds.maps[i][0, 0].numpy() for i in range(len(MAZE_NAMES))]

    agg = {k: [] for k in
           ["traj_collision", "goal_dist", "smooth", "ellipse_collision",
            "center_free", "area", "shape_mse", "progress_mae",
            "progress_violations", "topo_entropy", "topo_ce",
            "selected_ndtw", "best_ndtw", "recall", "topo_diversity",
            "traj_diversity", "committed_at"]}
    rng = np.random.default_rng(0)

    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break
        cond = batch["cond"].to(device)
        occ = batch["map_tensor"].to(device)
        cand = batch["candidate_paths"].to(device)
        cand_mask = batch["candidate_mask"].to(device).clone()
        cand_len = batch["candidate_lengths"].to(device)
        if args.max_candidates is not None:
            keep = min(int(args.max_candidates), cand_mask.shape[1])
            cand_mask[:, keep:] = False
        q = batch["topology_target"].numpy()
        prog_gt = batch["progress_gt"].numpy()
        mid = batch["maze_id"].numpy()
        best = batch["topology_best"].numpy()
        cand_np = cand.cpu().numpy()
        cand_len_np = cand_len.cpu().numpy()

        runs = []
        for r in range(max(1, args.runs)):
            runs.append(sample_v2(
                model, schedule, cond, occ, cand, cand_mask, cand_len,
                device=device, steps=args.steps,
                seed=1000 * bi + r, commit_t=commit_t, selection=selection))

        for b in range(cond.shape[0]):
            maze = int(mid[b])
            occ_map = occ_all[maze]
            p_runs = [run["p"][b].cpu().numpy() for run in runs]
            center_runs = [run["ellipse_center"][b].cpu().numpy() for run in runs]
            shape_runs = [run["ellipse_shape4"][b].cpu().numpy() for run in runs]

            # --- trajectory ---
            coll = [1.0 if _collides(_dense(p), occ_map) else 0.0 for p in p_runs]
            agg["traj_collision"].append(float(np.mean(coll)))
            goal = cond[b, 1].cpu().numpy()
            agg["goal_dist"].append(float(np.mean(
                [np.linalg.norm(p[-1] - goal) for p in p_runs])))
            agg["smooth"].append(float(np.mean([_smoothness(p) for p in p_runs])))
            if len(p_runs) > 1:
                d = [np.linalg.norm(p_runs[i] - p_runs[j])
                     for i in range(len(p_runs)) for j in range(i + 1, len(p_runs))]
                agg["traj_diversity"].append(float(np.mean(d)))

            # --- ellipse ---
            ec, cf = [], []
            for c, s4 in zip(center_runs, shape_runs):
                pts = _ellipse_points(c, s4, rng=rng)
                ec.append(float(np.mean([1.0 if _collides(p, occ_map) else 0.0
                                         for p in pts])))
                res = occ_map.shape[0]
                px = np.rint((c[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
                py = np.rint((c[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
                ok = ((px >= 0) & (px < res) & (py >= 0) & (py < res))
                free = np.zeros(len(c), dtype=bool)
                free[ok] = ~occ_map[py[ok], px[ok]].astype(bool)
                cf.append(float(free.mean()))
            agg["ellipse_collision"].append(float(np.mean(ec)))
            agg["center_free"].append(float(np.mean(cf)))
            agg["area"].append(float(np.mean(
                [np.pi * np.exp(s4[:, 0] + s4[:, 1]).mean() for s4 in shape_runs])))

            if not batch["has_candidate"][b]:
                continue
            b4 = int(best[b])
            # shape supervision target at the nearest skeleton pixel
            tgt, ok = ds.shape_at(maze, center_runs[0])
            if ok.any():
                agg["shape_mse"].append(float(
                    ((shape_runs[0][ok] - tgt[ok]) ** 2).mean()))

            # --- progress ---
            pref = runs[0]["progress"][b].cpu().numpy()
            agg["progress_mae"].append(float(np.abs(pref - prog_gt[b]).mean()))
            agg["progress_violations"].append(
                float((np.diff(pref) < -1e-6).sum()))

            # --- topology ---
            pi = runs[0]["topology_pi"][b].cpu().numpy()
            pi = pi / max(pi.sum(), 1e-12)
            nz = pi > 1e-12
            agg["topo_entropy"].append(float(-(pi[nz] * np.log(pi[nz])).sum()))
            qb = q[b]
            if qb.sum() > 0:
                qb = qb / qb.sum()
                agg["topo_ce"].append(float(
                    -(qb[qb > 0] * np.log(np.maximum(pi[qb > 0], 1e-12))).sum()))
            gt = cand_np[b, b4, :, :2]
            d = [normalized_dtw(gt, cand_np[b, m, :, :2]) for m in
                 np.nonzero(cand_mask[b].cpu().numpy())[0]]
            if d:
                agg["best_ndtw"].append(float(min(d)))
                agg["recall"].append(float(min(d) <= args.recall_tau))
                sel = int(runs[0]["selected_idx"][b])
                agg["selected_ndtw"].append(
                    float(normalized_dtw(gt, cand_np[b, sel, :, :2])))
            sel_all = [int(run["selected_idx"][b]) for run in runs]
            agg["topo_diversity"].append(float(len(set(sel_all)) > 1))
            agg["committed_at"].append(float(runs[0]["committed_at"][b]))
        print("[eval] batch %d/%d" % (bi + 1, args.num_batches), flush=True)

    summary = {}
    for k, v in agg.items():
        if not v:
            summary[k] = None
        elif k in ("progress_violations",):
            summary[k] = float(np.sum(v))
        else:
            summary[k] = float(np.mean(v))
    summary["M"] = args.max_candidates or int(topo_cfg.get("num_candidates", 4))
    summary["selection"] = selection
    summary["steps"] = args.steps
    summary["runs"] = args.runs
    summary["ckpt"] = args.ckpt
    out = args.out or os.path.join(
        os.path.dirname(args.ckpt),
        "eval_%s_M%s_%s.json" % (args.split, summary["M"], selection))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
