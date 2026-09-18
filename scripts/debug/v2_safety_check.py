"""V2 safety verification: ellipse centres + convex regions against the map.

Answers two questions with an INDEPENDENT check (it never trusts the
constructor of the region):

  1. CenterFree - does every predicted ellipse centre c_i = gamma_m(s_i) stay in
     free space?  V2 makes this structural, so the rate must be exactly 1.0000.
  2. Convex regions - build the Neural-IRIS style convex polytope
     { x : A x <= b } from each predicted ellipse and then test EVERY occupancy
     cell whose centre lies inside the polytope.  A single obstacle cell inside
     means the region is not safe, whatever the constructor assumed.

    python scripts/debug/v2_safety_check.py \
        --ckpt outputs/ckpt_v2_skeleton/best.pt --num 6 --device cpu

Outputs: outputs/v2_safety/center_convex_<tag>.json and .png
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler_v2 import sample_v2
from src.models.skeleton import SkeletonPlanner
from src.datasets.skeleton_dataset import make_loader, MAZE_NAMES
from src.geometry.offline_iris_wrapper import infer_convex_region_from_scene_occupancy
from src.geometry.safe_convex_region import generate_verified_convex_region


def to_pixels(points, size):
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * size - 0.5


def scene_to_cell(points, res):
    """Scene coords -> integer cell indices (i = column, j = row)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    i = np.floor((p[:, 0] + 1.0) / 2.0 * res).astype(int)
    j = np.floor((p[:, 1] + 1.0) / 2.0 * res).astype(int)
    return i, j


def cell_is_free(occ, i, j):
    res = occ.shape[0]
    ok = (i >= 0) & (i < res) & (j >= 0) & (j < res)
    out = np.zeros(len(i), dtype=bool)
    out[ok] = ~occ[j[ok], i[ok]].astype(bool)
    return out


def distance_to_obstacle_cells(points, occ, extent=(-1.0, 1.0, -1.0, 1.0)):
    """Exact distance (in cells) from each point to the nearest obstacle CELL AREA.

    A value of 0 means the point touches (or is inside) an obstacle cell; a
    positive value means it is strictly inside free space.
    """
    res = occ.shape[0]
    x0, x1, y0, y1 = extent
    hx = (x1 - x0) / (2.0 * res)
    hy = (y1 - y0) / (2.0 * res)
    j, i = np.nonzero(occ.astype(bool))
    if len(i) == 0:
        return np.full(len(points), np.inf)
    cx = x0 + (i + 0.5) * (x1 - x0) / res
    cy = y0 + (j + 0.5) * (y1 - y0) / res
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    dx = np.maximum(np.abs(pts[:, 0:1] - cx[None, :]) - hx, 0.0)
    dy = np.maximum(np.abs(pts[:, 1:2] - cy[None, :]) - hy, 0.0)
    return np.sqrt(dx ** 2 + dy ** 2).min(axis=1) * res / 2.0


def penetration_into_obstacle_cells(points, occ, extent=(-1.0, 1.0, -1.0, 1.0)):
    """Depth (in cells) by which each point lies INSIDE an obstacle cell area.

    0 means the point is outside or exactly on the cell boundary.
    """
    res = occ.shape[0]
    x0, x1, y0, y1 = extent
    hx = (x1 - x0) / (2.0 * res)
    hy = (y1 - y0) / (2.0 * res)
    j, i = np.nonzero(occ.astype(bool))
    if len(i) == 0:
        return np.zeros(len(points))
    cx = x0 + (i + 0.5) * (x1 - x0) / res
    cy = y0 + (j + 0.5) * (y1 - y0) / res
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    ex = np.abs(pts[:, 0:1] - cx[None, :]) - hx
    ey = np.abs(pts[:, 1:2] - cy[None, :]) - hy
    # inside a cell iff both are negative; depth = -max(ex, ey)
    depth = -np.maximum(ex, ey)
    return np.maximum(depth.max(axis=1), 0.0) * res / 2.0


def ellipse_points(center, shape4, n_b=96, n_i=64, rng=None):
    a = np.exp(shape4[:, 0])
    b = np.exp(shape4[:, 1])
    th = 0.5 * np.arctan2(shape4[:, 3], shape4[:, 2])
    ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
    rng = rng if rng is not None else np.random.default_rng(0)

    def world(ex, ey):
        return np.stack([ct * ex - st * ey + center[:, 0:1],
                         st * ex + ct * ey + center[:, 1:2]], axis=-1)

    ang = np.linspace(0.0, 2 * np.pi, n_b, endpoint=False)[None, :]
    ring = world(a[:, None] * np.cos(ang), b[:, None] * np.sin(ang))
    r = np.sqrt(rng.random((len(center), n_i)))
    t = rng.uniform(0.0, 2 * np.pi, (len(center), n_i))
    inner = world(a[:, None] * r * np.cos(t), b[:, None] * r * np.sin(t))
    return np.concatenate([ring, inner], axis=1)


def check_convex_region(A, b, occ, extent=(-1.0, 1.0, -1.0, 1.0)):
    """Independent test: every occupancy cell inside {A x <= b} must be free."""
    if A is None or len(A) == 0:
        return {"ok": False, "reason": "no_halfspaces", "n_cells": 0,
                "n_blocked": 0}
    res = occ.shape[0]
    # bounding box of the polytope: cheapest valid box is the map itself, but we
    # restrict to the region where the map raster is dense enough
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for row, rhs in zip(A, b):
        n = np.linalg.norm(row)
        if n < 1e-9:
            continue
        lo = np.minimum(lo, (rhs / n) * np.sign(row) - 1e9 * 0)
        hi = np.maximum(hi, (rhs / n) * np.sign(row) + 1e9 * 0)
    # simpler and exact enough: scan the whole map (256x256 = 65536 points)
    xs = (np.arange(res) + 0.5) * 2.0 / res - 1.0
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)          # [P,2]
    inside = (pts @ A.T <= b[None, :] + 1e-12).all(axis=1)
    n_cells = int(inside.sum())
    if n_cells == 0:
        return {"ok": False, "reason": "empty_region", "n_cells": 0,
                "n_blocked": 0}
    ij = np.nonzero(inside)[0]
    j = ij // res
    i = ij % res
    blocked = occ[j, i].astype(bool)
    return {"ok": bool(not blocked.any()), "reason": "ok" if not blocked.any()
            else "obstacle_inside", "n_cells": n_cells,
            "n_blocked": int(blocked.sum())}


def plot_check(occ, sample, out_png, title, stride=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = occ.shape[0]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    ax.plot(*to_pixels(sample["path"], res).T, color="#1f77b4", lw=1.4,
            label="selected topology")
    ax.plot(*to_pixels(sample["traj"], res).T, color="#2ca02c", lw=1.8,
            label="trajectory")

    center = sample["center"]
    shape4 = sample["shape4"]
    a = np.exp(shape4[:, 0])
    b = np.exp(shape4[:, 1])
    th = 0.5 * np.arctan2(shape4[:, 3], shape4[:, 2])
    scale = res / 2.0
    ang = np.linspace(0, 2 * np.pi, 64)
    cm = to_pixels(center, res)
    for k in range(0, len(center), max(1, int(stride))):
        ct, st = np.cos(th[k]), np.sin(th[k])
        ex, ey = a[k] * np.cos(ang) * scale, b[k] * np.sin(ang) * scale
        ax.plot(ct * ex - st * ey + cm[k, 0], st * ex + ct * ey + cm[k, 1],
                color="#d62728", lw=0.8, alpha=0.9)
        verts = sample["vertices"][k]
        if verts is not None and len(verts) >= 3:
            poly = np.vstack([to_pixels(verts, res), to_pixels(verts, res)[:1]])
            ax.fill(poly[:, 0], poly[:, 1], color="#ff7f0e", alpha=0.12,
                    linewidth=0)
            ax.plot(poly[:, 0], poly[:, 1], color="#ff7f0e", lw=0.6, alpha=0.8)
    ax.scatter(cm[:, 0], cm[:, 1], s=1.5, c="#d62728", zorder=6)
    ax.scatter(*to_pixels(sample["cond"], res).T, marker="*", s=140, c="k",
               zorder=7)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v2_skeleton.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window-half", type=float, default=0.25)
    ap.add_argument("--safety-margin", type=float, default=0.0)
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="sub-cell inward erosion applied to the region")
    ap.add_argument("--stride", type=int, default=4,
                    help="convex regions are built for every Stride-th ellipse")
    ap.add_argument("--out", default="outputs/v2_safety")
    args = ap.parse_args()

    cfg = load_config(args.config)
    topo_cfg = cfg.get("topology", {})
    source = cfg["data"].get("source", "data/processed_scene_v1")
    base_dir = cfg["data"].get("base", "data/processed_scene_v2")
    mask_res = int(cfg["loss"].get("ellipse_safe_res", 64))

    from src.datasets.skeleton_dataset import SkeletonDataset, make_collate
    ds = SkeletonDataset(args.split, source, base_dir, mask_res=mask_res,
                         mask_tau=float(cfg["loss"].get("ellipse_mask_tau", 10.0)))
    idxs = list(range(args.offset, min(args.offset + args.num, len(ds))))
    if not idxs:
        raise SystemExit("offset %d is beyond the %s split" % (args.offset, args.split))
    batch = make_collate(ds)([ds[i] for i in idxs])
    args.num = len(idxs)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(args.device)
    model = SkeletonPlanner(cfg["model"], topo_cfg).to(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    print("[safety] ckpt=%s epoch=%s device=%s"
          % (args.ckpt, ckpt.get("epoch"), args.device), flush=True)

    sl = slice(0, args.num)
    cond = batch["cond"][sl].to(args.device)
    occ = batch["map_tensor"][sl].to(args.device)
    cand = batch["candidate_paths"][sl].to(args.device)
    cand_mask = batch["candidate_mask"][sl].to(args.device)
    cand_len = batch["candidate_lengths"][sl].to(args.device)
    mid = batch["maze_id"][sl].numpy()

    t0 = time.time()
    out = sample_v2(model, schedule, cond, occ, cand, cand_mask, cand_len,
                    device=args.device, steps=args.steps, seed=args.seed,
                    commit_t=int(topo_cfg.get("commit_t", 7)),
                    selection=topo_cfg.get("selection", "sample"))
    print("[safety] sampling took %.1fs" % (time.time() - t0), flush=True)

    occ_all = [ds.maps[i][0, 0].numpy() for i in range(len(MAZE_NAMES))]
    rng = np.random.default_rng(0)
    K = model.horizon
    stats = {
        "n_samples": int(cond.shape[0]),
        "n_ellipses": 0,
        "center_free": 0,
        "center_min_clearance_cells": None,
        "ellipse_fully_free": 0,
        "convex_regions": 0,
        "convex_safe": 0,
        "convex_blocked_cells_total": 0,
        "convex_empty_or_failed": 0,
        "vertex_outside_free": 0,
        "ellipse_outside_region": 0,
        "region_margin_cells_min": None,
        "convex_needing_repair": 0,
        "blocked_cells_before_total": 0,
        "repair_rounds_total": 0,
        "convex_safe_raw": 0,
        "vertex_touching_obstacle_cell": 0,
        "vertex_min_clearance_cells": None,
        "vertex_max_penetration_cells": None,
    }
    clearances = []
    region_margins = []
    vertex_clearances = []
    edt_cache = {}
    per_sample = []

    for b in range(cond.shape[0]):
        occ_map = occ_all[int(mid[b])]
        res = occ_map.shape[0]
        center = out["ellipse_center"][b].detach().cpu().numpy()
        shape4 = out["ellipse_shape4"][b].detach().cpu().numpy()
        ci, cj = scene_to_cell(center, res)
        free_c = cell_is_free(occ_map, ci, cj)
        stats["center_free"] += int(free_c.sum())
        stats["n_ellipses"] += len(ci)

        # clearance of every centre to the nearest obstacle cell (in cells)
        obs_j, obs_i = np.nonzero(occ_map.astype(bool))
        obs_xy = np.stack([(obs_i + 0.5) * 2.0 / res - 1.0,
                           (obs_j + 0.5) * 2.0 / res - 1.0], axis=-1)
        d = np.linalg.norm(center[:, None, :] - obs_xy[None, :, :], axis=-1)
        clearances.append(d.min(axis=1) * res / 2.0)      # in cells

        pts = ellipse_points(center, shape4, rng=rng)
        ok_e = []
        for p in pts:
            i, j = scene_to_cell(p, res)
            ok_e.append(bool(cell_is_free(occ_map, i, j).all()))
        stats["ellipse_fully_free"] += int(sum(ok_e))

        if int(mid[b]) not in edt_cache:
            from scipy import ndimage
            edt_cache[int(mid[b])] = ndimage.distance_transform_edt(
                ~occ_map.astype(bool))
        edt = edt_cache[int(mid[b])]

        ring = ellipse_points(center, shape4, n_b=64, n_i=1, rng=rng)[:, :64, :]
        verts_list = [None] * K
        for k in range(0, K, max(1, int(args.stride))):
            a = float(np.exp(shape4[k, 0]))
            bmin = float(np.exp(shape4[k, 1]))
            th = float(0.5 * np.arctan2(shape4[k, 3], shape4[k, 2]))
            A, bvec, verts, info = generate_verified_convex_region(
                occ_map, center[k], a, bmin, th,
                window_half=args.window_half, safety_margin=args.safety_margin,
                dilation=1, boundary_jitter=1, shrink=args.shrink)
            if A is None or len(A) == 0:
                stats["convex_empty_or_failed"] += 1
                continue
            stats["convex_regions"] += 1
            stats["repair_rounds_total"] += int(info["rounds"])
            if int(info["blocked_before"]) > 0:
                stats["convex_needing_repair"] += 1
                stats["blocked_cells_before_total"] += int(info["blocked_before"])
            else:
                stats["convex_safe_raw"] += 1
            # independent re-check over the whole map (never trusts the builder)
            chk = check_convex_region(A, bvec, occ_map)
            if chk["ok"]:
                stats["convex_safe"] += 1
            else:
                stats["convex_blocked_cells_total"] += chk["n_blocked"]
            # the ellipse is supposed to be the certified inner set of the region
            if bool((ring[k] @ A.T > bvec[None, :] + 1e-9).any()):
                stats["ellipse_outside_region"] += 1
            # how much clearance does the whole region keep from any obstacle
            xs = (np.arange(res) + 0.5) * 2.0 / res - 1.0
            gx, gy = np.meshgrid(xs, xs, indexing="xy")
            pts_all = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
            ins = (pts_all @ A.T <= bvec[None, :] + 1e-12).all(axis=1)
            if ins.any():
                jj = np.nonzero(ins)[0] // res
                ii = np.nonzero(ins)[0] % res
                region_margins.append(float(edt[jj, ii].min()))
            if verts is not None:
                vd = distance_to_obstacle_cells(verts, occ_map)
                vertex_clearances.append(vd)
                vi, vj = scene_to_cell(verts, res)
                if not cell_is_free(occ_map, vi, vj).all():
                    stats["vertex_outside_free"] += 1
                if float(vd.min()) < 1e-9:
                    stats["vertex_touching_obstacle_cell"] += 1
                pen = penetration_into_obstacle_cells(verts, occ_map)
                stats["vertex_max_penetration_cells"] = max(
                    stats.get("vertex_max_penetration_cells") or 0.0,
                    float(pen.max()))
            verts_list[k] = verts

        sample = {"center": center, "shape4": shape4,
                  "traj": out["p"][b].detach().cpu().numpy(),
                  "path": cand[b, int(out["selected_idx"][b]), :, :2].cpu().numpy(),
                  "cond": cond[b].detach().cpu().numpy(), "vertices": verts_list}
        per_sample.append({"center_free": float(free_c.mean()),
                           "ellipse_free": float(np.mean(ok_e))})

        if b < 3:
            os.makedirs(args.out, exist_ok=True)
            plot_check(occ_map, sample,
                       os.path.join(args.out, "center_convex_%s_%d.png"
                                    % (args.split, args.offset + b)),
                       "V2 safety check: ellipses (red) + convex regions (orange)")

    clearances = np.concatenate(clearances)
    if region_margins:
        stats["region_margin_cells_min"] = float(np.min(region_margins))
        stats["region_margin_cells_mean"] = float(np.mean(region_margins))
    if vertex_clearances:
        vc = np.concatenate(vertex_clearances)
        stats["vertex_min_clearance_cells"] = float(vc.min())
        stats["vertex_clearance_mean_cells"] = float(vc.mean())
    stats["center_min_clearance_cells"] = float(clearances.min())
    stats["center_clearance_mean_cells"] = float(clearances.mean())
    stats["center_free_rate"] = stats["center_free"] / max(stats["n_ellipses"], 1)
    stats["ellipse_free_rate"] = stats["ellipse_fully_free"] / max(stats["n_ellipses"], 1)
    stats["convex_safe_rate"] = stats["convex_safe"] / max(stats["convex_regions"], 1)
    stats["convex_safe_rate_before_repair"] = (
        stats["convex_safe_raw"] / max(stats["convex_regions"], 1))
    stats["ckpt"] = args.ckpt
    stats["epoch"] = ckpt.get("epoch")
    stats["split"] = args.split
    stats["seed"] = args.seed

    os.makedirs(args.out, exist_ok=True)
    tag = "%s_%d_%d" % (args.split, args.offset, args.seed)
    with open(os.path.join(args.out, "center_convex_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump({"summary": stats, "per_sample": per_sample}, f, indent=2)
    print(json.dumps(stats, indent=2))
    print("saved", os.path.join(args.out, "center_convex_%s.json" % tag))


if __name__ == "__main__":
    main()
