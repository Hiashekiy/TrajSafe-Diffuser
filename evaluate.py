"""Evaluate the V1 joint trajectory-ellipse diffusion model on test conditions.

For each maze (umaze/medium/large) it samples --n trajectories from the test
conditions, then reports:

  * endpoint error            (hard condition should give ~0)
  * P collision fraction      SDF(P) <= 0   + clearance p05 (scene units)
  * ellipse sanity            finite/positive radii, centre inside free space
  * ellipse vs GT errors      |dC|, |log a|, |log b|, circular theta error
                              (GT from data/processed_scene_v1 ellipses6)
  * ellipse boundary collision (perimeter points inside wall, sampled every 4th anchor)
  * trajectory smoothness     mean ||p_{k+1} - 2p_k + p_{k-1}||_2

Usage:
  python evaluate.py --config configs/config_v1.yaml \
      --ckpt outputs/ckpt_v1/best.pt --n 40 --out outputs/evaluate_report_v1.json
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

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import load_checkpoint
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler_v1 import sample_joint
from src.models.joint import JointPlanner
from src.datasets.joint_dataset import JointDataset
from src.geometry.scene_frame import sample_sdf_scene

MAZE_NAMES = ["umaze", "medium", "large"]
RES = 256


def pick_conditions(ds, maze, n, rng):
    sel = np.where(ds.mid == MAZE_NAMES.index(maze))[0]
    if len(sel) <= n:
        return sel
    edges = np.linspace(0.0, 1.0, n + 2)[1:-1]
    idx = np.unique(np.round(edges * (len(sel) - 1)).astype(int))
    return sel[idx]


def e6_to_ellipse5(p, e6):
    c = p + e6[..., :2]
    a = np.exp(np.clip(e6[..., 2], -20, 20))
    b = np.exp(np.clip(e6[..., 3], -20, 20))
    th = 0.5 * np.arctan2(e6[..., 5], e6[..., 4])
    return np.stack([c[..., 0], c[..., 1], a, b, th], axis=-1)


def angle_from_e6(e6):
    return 0.5 * np.arctan2(e6[..., 5], e6[..., 4])       # in [-pi/2, pi/2]


def circular_diff_pi(a, b):
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


def ellipse_boundary_collision(E5, sdf_grid, stride=4, n_per=24, b_min=2e-3):
    """Fraction of ellipse perimeter sample points inside walls (scene units).

    Only every `stride`-th anchor is sampled to keep the check cheap.
    """
    N = len(E5)
    bad = 0.0
    tot = 0.0
    al = np.linspace(0, 2 * np.pi, n_per, endpoint=False)
    for i in range(N):
        e = E5[i]                                           # [128,5]
        sel = np.arange(0, len(e), stride)
        c = e[sel, :2]
        a = e[sel, 2]
        b = e[sel, 3]
        th = e[sel, 4]
        ok = np.isfinite(a) & (a > 0) & np.isfinite(b) & (b > b_min)
        c, a, b, th = c[ok], a[ok], b[ok], th[ok]
        if len(a) == 0:
            continue
        u = np.stack([np.cos(th), np.sin(th)], axis=-1)     # [K,2]
        v = np.stack([-np.sin(th), np.cos(th)], axis=-1)
        ca = np.cos(al)[None, :, None]                      # [1,24,1]
        sa = np.sin(al)[None, :, None]
        pts = (c[:, None, :] + a[:, None, None] * (ca * u[:, None, :]) +
               b[:, None, None] * (sa * v[:, None, :])).reshape(-1, 2)
        cell = np.clip(np.round((pts + 1.0) / 2.0 * (RES - 1)).astype(int), 0, RES - 1)
        bad += float((sdf_grid[cell[:, 1], cell[:, 0]] <= 0.0).sum())
        tot += len(pts)
    return bad / max(tot, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v1.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint (default <config.train.ckpt_dir>/best.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=40, help="samples per maze")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="outputs/evaluate_report_v1.json")
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

    ds = JointDataset(os.path.join(base, args.split))
    rng = np.random.default_rng(args.seed)
    report = {"device": device, "split": args.split, "seed": args.seed,
              "checkpoint": ckpt,
              "global_mem_res": cfg["model"].get("global_mem_res",
                                                 cfg["model"].get("mem_res", 16)),
              "mazes": {}}
    t_all = time.time()
    for mi, maze in enumerate(MAZE_NAMES):
        sel = pick_conditions(ds, maze, args.n, rng)
        cond = torch.as_tensor(ds.cond[sel], dtype=torch.float32).to(device)
        map_t = ds.maps[mi].to(device).expand(len(sel), -1, -1, -1).contiguous()
        sdf_t = ds.sdfs[mi].to(device).expand(len(sel), -1, -1, -1).contiguous()

        t0 = time.time()
        alm_enabled = bool(cfg.get("alm", {}).get("enabled", False))
        sampled = sample_joint(
            model, schedule, cond, map_t, device,
            steps=args.steps, seed=args.seed + mi * 101,
            alm_config=cfg.get("alm"), return_alm_stats=alm_enabled,
        )
        if alm_enabled:
            P, E6, alm_stats = sampled
            if cfg.get("alm", {}).get("log_per_step", False):
                for step_stats in alm_stats.get("per_step", []):
                    print(
                        f"[{maze}][ALM t={step_stats['t']}] "
                        f"valid={step_stats['corridor_valid_rate']:.3f} "
                        f"collision={step_stats['physical_collision_rate_before']:.3f}->"
                        f"{step_stats['physical_collision_rate_after']:.3f} "
                        f"covered={step_stats['collision_covered_rate']:.3f} "
                        f"invalid={step_stats['collision_but_invalid_rate']:.3f} "
                        f"inside={step_stats['collision_inside_region_rate']:.3f} "
                        f"new={step_stats['new_physical_collision_rate']:.3f} "
                        f"active={step_stats['raw_max_positive_rate']:.3f} "
                        f"lambda={step_stats['lambda_mean']:.4f}/"
                        f"{step_stats['lambda_max']:.4f} "
                        f"corr={step_stats['mean_correction']:.5f}/"
                        f"{step_stats['max_correction']:.5f}",
                        flush=True,
                    )
        else:
            P, E6 = sampled
            alm_stats = {}
        dt = time.time() - t0
        P = P.cpu().numpy()
        E6 = E6.cpu().numpy()

        # endpoint error
        end_err = float(max(np.abs(P[:, 0] - ds.cond[sel, 0]).max(),
                            np.abs(P[:, -1] - ds.cond[sel, 1]).max()))
        # P collision / clearance via bilinear scene SDF
        with torch.no_grad():
            sdf_vals = sample_sdf_scene(sdf_t, torch.as_tensor(P, dtype=torch.float32).to(device))
        sdf_vals = sdf_vals.cpu().numpy()
        coll = float((sdf_vals <= 0.0).mean())
        clear_p05 = float(np.percentile(sdf_vals, 5))

        # ellipse sanity
        E5 = e6_to_ellipse5(P, E6)
        a, b = E5[..., 2], E5[..., 3]
        sane = float((np.isfinite(a) & (a > 0) & (a <= 2.0) &
                      np.isfinite(b) & (b > 0) & (b <= 2.0)).mean())
        c = E5[..., :2]
        with torch.no_grad():
            sdf_c = sample_sdf_scene(sdf_t, torch.as_tensor(c, dtype=torch.float32).to(device))
        center_free = float((sdf_c.cpu().numpy() > 0.0).mean())

        # ---- ablation metrics vs GT ellipse labels ----
        e6_gt = ds.e6[sel].astype(np.float32)
        pred_la = E6[..., 2]
        pred_lb = E6[..., 3]
        pred_th = angle_from_e6(E6)
        gt_la = e6_gt[..., 2]
        gt_lb = e6_gt[..., 3]
        gt_th = angle_from_e6(e6_gt)
        center_err = float(np.linalg.norm(E6[..., :2] - e6_gt[..., :2], axis=-1).mean())
        la_err = float(np.abs(pred_la - gt_la).mean())
        lb_err = float(np.abs(pred_lb - gt_lb).mean())
        th_err = float(circular_diff_pi(pred_th, gt_th).mean())

        sdf_grid = np.load(os.path.join(base, "maps", f"{maze}_sdf.npy"))
        e_coll = ellipse_boundary_collision(E5, sdf_grid)

        # trajectory smoothness: mean ||p_{k+1} - 2 p_k + p_{k-1}||_2
        acc = P[:, 2:] - 2.0 * P[:, 1:-1] + P[:, :-2]
        smooth = float(np.linalg.norm(acc, axis=-1).mean())

        m = {"n": int(len(sel)),
             "time_s": round(dt, 2),
             "endpoint_err": round(end_err, 6),
             "collision_frac": round(coll, 4),
             "clearance_p05": round(clear_p05, 4),
             "ellipse_sane_frac": round(sane, 4),
             "ellipse_center_free_frac": round(center_free, 4),
             "ellipse_center_err": round(center_err, 4),
             "ellipse_loga_err": round(la_err, 4),
             "ellipse_logb_err": round(lb_err, 4),
             "ellipse_theta_err": round(th_err, 4),
             "ellipse_boundary_coll": round(e_coll, 4),
             "traj_smoothness": round(smooth, 4)}
        if alm_stats:
            m["alm"] = {key: round(value, 6) if isinstance(value, float) else value
                        for key, value in alm_stats.items()}
        report["mazes"][maze] = m
        print(f"[{maze}] n={m['n']} dt={dt:.1f}s end_err={m['endpoint_err']:.2e} "
              f"coll={m['collision_frac']:.4f} clear_p05={m['clearance_p05']:.3f} "
              f"e_sane={m['ellipse_sane_frac']:.3f} e_free={m['ellipse_center_free_frac']:.3f} "
              f"| e_center_err={m['ellipse_center_err']:.4f} "
              f"e_la_err={m['ellipse_loga_err']:.4f} e_lb_err={m['ellipse_logb_err']:.4f} "
              f"e_th_err={m['ellipse_theta_err']:.4f} e_bnd_coll={m['ellipse_boundary_coll']:.4f} "
              f"smooth={m['traj_smoothness']:.4f}", flush=True)

    report["total_time_s"] = round(time.time() - t_all, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("saved:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
