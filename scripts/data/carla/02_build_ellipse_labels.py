"""02_build_ellipse_labels.py - CARLA fixed-centre safe ellipse labels.

For every cleaned sample this script:

  1. takes m* = ``topology_best`` (from 01) and decodes its DENSE safe polyline
     from the stored cell indices: ``scene = (px + 0.5) * (2/256) - 1`` with the
     first/last point forced to ``start`` / ``goal``;
  2. fixes the progress to ``s_i = i / 127`` and decodes the 128 ellipse centres
     by ARC LENGTH interpolation on that dense chain
     (``skeleton_paths.interpolate_path``) - never on the 128-point resample;
  3. computes, on the SAME canonical occupancy, the largest safe axis-aligned
     rotated ellipse at every centre with exactly the algorithm of
     ``scripts/data/03_build_ellipse_labels.py`` (36 orientations, local-radius
     cap, DDA ray clearance, full boundary+interior safety check, binary search).

Outputs per split (consumed by ``src/datasets/carla_spline_dataset.py``):

    ellipse_shape4_gt.npy  [N, 128, 4] f32  [log a, log b, cos 2t, sin 2t]
    shape_valid.npy        [N, 128]    bool

There is NO progress label and NO ellipse-centre label.
Per-sample caching in ``<split>/_cache/ell_<i:06d>.npz`` makes the run
resumable and isolates single-sample failures.

    python scripts/data/carla/02_build_ellipse_labels.py --processed data/carla_processed --workers 12
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import interpolate_path

SPLITS = ["train", "val", "test"]
CELL = 2.0 / 256.0
HORIZON = 128
_W = {}


# ---------------------------------------------------------------------------
# occupancy safety + ray clearance + ellipse safety (same math as 03)
# ---------------------------------------------------------------------------


def occupancy_safe(occ: np.ndarray, dilation_cells: int) -> np.ndarray:
    obstacle = np.asarray(occ) > 0.5
    if int(dilation_cells) > 0:
        obstacle = binary_dilation(obstacle,
                                   structure=np.ones((3, 3), dtype=bool),
                                   iterations=int(dilation_cells))
    return obstacle


def ray_clearance(safe, center_px, direction, max_dist_px: float) -> float:
    h, w = safe.shape
    x0, y0 = float(center_px[0]), float(center_px[1])
    ix, iy = int(np.floor(x0)), int(np.floor(y0))
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return 0.0
    if safe[iy, ix]:
        return 0.0
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    inf = float("inf")
    if abs(dx) < 1e-12:
        t_max_x, t_delta_x = inf, inf
    else:
        next_x = ix + (1 if dx > 0 else 0)
        t_max_x = abs((next_x - x0) / dx)
        t_delta_x = 1.0 / abs(dx)
    if abs(dy) < 1e-12:
        t_max_y, t_delta_y = inf, inf
    else:
        next_y = iy + (1 if dy > 0 else 0)
        t_max_y = abs((next_y - y0) / dy)
        t_delta_y = 1.0 / abs(dy)
    max_steps = int(np.ceil(float(max_dist_px))) + 8
    t = 0.0
    for _ in range(max_steps):
        if t_max_x < t_max_y:
            ix += step_x
            t = t_max_x
            t_max_x += t_delta_x
        else:
            iy += step_y
            t = t_max_y
            t_max_y += t_delta_y
        if t > float(max_dist_px):
            return float(max_dist_px)
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return t
        if safe[iy, ix]:
            return t
    return float(max_dist_px)


def clearance_scene(safe, graph, center_scene, direction,
                    max_dist_scene: float) -> float:
    center_px = graph.scene_to_pixel(np.asarray(center_scene, dtype=np.float64))
    return ray_clearance(safe, center_px, direction,
                         float(max_dist_scene) / graph.cell) * graph.cell


def ellipse_sample_points(center, a, b, theta, boundary=128,
                          interior_rings=4, interior_angles=32):
    center = np.asarray(center, dtype=np.float64).reshape(2)
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    pts = []
    ang = np.linspace(0.0, 2.0 * np.pi, int(boundary), endpoint=False)
    ex, ey = float(a) * np.cos(ang), float(b) * np.sin(ang)
    pts.append(np.stack([ct * ex - st * ey + center[0],
                         st * ex + ct * ey + center[1]], axis=1))
    if interior_rings > 0 and interior_angles > 0:
        for r in np.linspace(0.25, 0.75, int(interior_rings)):
            ang_i = np.linspace(0.0, 2.0 * np.pi, int(interior_angles),
                                endpoint=False)
            ex, ey = r * float(a) * np.cos(ang_i), r * float(b) * np.sin(ang_i)
            pts.append(np.stack([ct * ex - st * ey + center[0],
                                 st * ex + ct * ey + center[1]], axis=1))
    pts.append(center[None, :])
    return np.concatenate(pts, axis=0)


def ellipse_is_safe(safe, graph, center, a, b, theta, boundary=128,
                    interior_rings=4, interior_angles=32) -> bool:
    pts = ellipse_sample_points(center, a, b, theta, boundary,
                                interior_rings, interior_angles)
    ix = np.floor((pts[:, 0] + 1.0) / graph.cell).astype(np.int64)
    iy = np.floor((pts[:, 1] + 1.0) / graph.cell).astype(np.int64)
    h, w = safe.shape
    if (ix < 0).any() or (ix >= w).any() or (iy < 0).any() or (iy >= h).any():
        return False
    return not bool(safe[iy, ix].any())


def shape_at_center(safe, graph, params, center_scene):
    """Largest safe ellipse at one CONTINUOUS scene centre -> shape4 or None."""
    center = np.asarray(center_scene, dtype=np.float64).reshape(2)
    local_radius = params["local_radius"]
    min_axis = params["min_semi_axis"]
    boundary = params["boundary"]
    interior_rings = params["interior_rings"]
    interior_angles = params["interior_angles"]
    binary_iters = params["binary_iters"]
    best = None
    for theta in params["thetas"]:
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-np.sin(theta), np.cos(theta)])
        d_p = clearance_scene(safe, graph, center, u, local_radius)
        d_m = clearance_scene(safe, graph, center, -u, local_radius)
        d_pp = clearance_scene(safe, graph, center, v, local_radius)
        d_mm = clearance_scene(safe, graph, center, -v, local_radius)
        a0 = min(d_p, d_m, float(local_radius))
        b0 = min(d_pp, d_mm, float(local_radius))
        if a0 < float(min_axis) or b0 < float(min_axis):
            continue
        if ellipse_is_safe(safe, graph, center, a0, b0, theta, boundary,
                           interior_rings, interior_angles):
            lam = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(int(binary_iters)):
                mid = 0.5 * (lo + hi)
                if ellipse_is_safe(safe, graph, center, a0 * mid, b0 * mid,
                                   theta, boundary, interior_rings,
                                   interior_angles):
                    lo = mid
                else:
                    hi = mid
            lam = lo
        area = (a0 * lam) * (b0 * lam)
        if best is None or area > best[0]:
            best = (area, a0 * lam, b0 * lam, theta)
    if best is None or best[0] <= 0.0:
        return None
    _, a, b, theta = best
    if a < b:
        a, b = b, a
        theta = theta + 0.5 * np.pi
    if a < float(min_axis) or b < float(min_axis):
        return None
    theta = float(theta % np.pi)
    return np.array([np.log(a), np.log(b), np.cos(2 * theta),
                     np.sin(2 * theta)], dtype=np.float32)


# ---------------------------------------------------------------------------
# multiprocessing
# ---------------------------------------------------------------------------


def _cache_path(processed, split, i):
    return os.path.join(processed, split, "_cache", "ell_%06d.npz" % int(i))


def _init_worker(processed, split, params):
    d = os.path.join(processed, split)
    _W["processed"] = processed
    _W["split"] = split
    _W["occ"] = np.load(os.path.join(d, "occupancy.npy"), mmap_mode="r")
    _W["cond"] = np.load(os.path.join(d, "conditions.npy"), mmap_mode="r")
    _W["mask"] = np.load(os.path.join(d, "candidate_mask.npy"))
    _W["offsets"] = np.load(os.path.join(d, "candidate_geometry_offsets.npy"))
    _W["glen"] = np.load(os.path.join(d, "candidate_geometry_lengths.npy"))
    _W["geom"] = np.load(os.path.join(d, "candidate_geometry.npy"),
                         mmap_mode="r")
    _W["best"] = np.load(os.path.join(d, "topology_best.npy"))
    _W["params"] = dict(params)
    _W["skel"] = dict(params.get("skel_cfg", {}))
    os.makedirs(os.path.join(d, "_cache"), exist_ok=True)


def _gamma_scene(i, m):
    lo, hi = int(_W["offsets"][i, m]), int(_W["offsets"][i, m + 1])
    if hi <= lo:
        return None
    px = np.asarray(_W["geom"][lo:hi], dtype=np.float64)
    return (px + 0.5) * CELL - 1.0


def _work(i):
    import numpy as np

    params = _W["params"]
    try:
        occ = np.asarray(_W["occ"][i])
        cond = np.asarray(_W["cond"][i], dtype=np.float64).reshape(2, 2)
        mask = np.asarray(_W["mask"][i]).astype(bool)
        valid = np.nonzero(mask)[0]
        s4 = np.zeros((HORIZON, 4), np.float32)
        ok = np.zeros(HORIZON, bool)
        n_ok = 0
        if len(valid):
            m = int(_W["best"][i])
            if not (0 <= m < len(mask) and mask[m]):
                m = int(valid[0])
            poly = _gamma_scene(i, m)
            if poly is not None and len(poly) >= 2:
                poly = poly.copy()
                poly[0] = cond[0]
                poly[-1] = cond[1]
                graph = build_skeleton_graph(
                    occ,
                    safety_dilation_cells=int(
                        _W["skel"].get("safety_dilation_cells", 1)),
                    thinning_backend=str(
                        _W["skel"].get("thinning_backend", "auto")),
                    pure_cycle_aux_nodes=int(
                        _W["skel"].get("pure_cycle_aux_nodes", 2)))
                safe = occupancy_safe(occ, int(params["dilation"]))
                s = np.linspace(0.0, 1.0, HORIZON)
                centers = interpolate_path(poly, s)
                for k in range(HORIZON):
                    shape = shape_at_center(safe, graph, params, centers[k])
                    if shape is not None:
                        s4[k] = shape
                        ok[k] = True
                        n_ok += 1
        np.savez_compressed(_cache_path(_W["processed"], _W["split"], i),
                            idx=i, shape4=s4, valid=ok)
        return {"idx": i, "ok": True, "valid": n_ok,
                "has_candidate": bool(len(valid))}
    except Exception as exc:                                   # noqa: BLE001
        s4 = np.zeros((HORIZON, 4), np.float32)
        ok = np.zeros(HORIZON, bool)
        try:
            np.savez_compressed(_cache_path(_W["processed"], _W["split"], i),
                                idx=i, shape4=s4, valid=ok)
        except Exception:
            pass
        return {"idx": i, "ok": False, "error": repr(exc), "valid": 0,
                "has_candidate": False}


def assemble(processed, split, n):
    d = os.path.join(processed, split)
    s4 = np.zeros((n, HORIZON, 4), np.float32)
    valid = np.zeros((n, HORIZON), bool)
    missing = []
    for i in range(n):
        p = _cache_path(processed, split, i)
        if not os.path.exists(p):
            missing.append(i)
            continue
        with np.load(p, allow_pickle=False) as z:
            s4[i] = np.asarray(z["shape4"], np.float32)
            valid[i] = np.asarray(z["valid"]).astype(bool)
    np.save(os.path.join(d, "ellipse_shape4_gt.npy"), s4)
    np.save(os.path.join(d, "shape_valid.npy"), valid)
    for stale in ("progress_gt.npy", "ellipse_center_gt.npy"):
        sp = os.path.join(d, stale)
        if os.path.exists(sp):
            os.remove(sp)
    return s4, valid, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--num-orientations", type=int, default=None)
    ap.add_argument("--local-radius", type=float, default=None)
    ap.add_argument("--dilation", type=int, default=None)
    ap.add_argument("--boundary", type=int, default=None)
    ap.add_argument("--interior-rings", type=int, default=None)
    ap.add_argument("--interior-angles", type=int, default=None)
    ap.add_argument("--binary-iters", type=int, default=None)
    ap.add_argument("--min-semi-axis", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    lab = cfg.get("ellipse_label") or {}
    processed = args.processed or cfg["data"].get("processed_root",
                                                  "data/carla_processed")
    processed = os.path.abspath(processed)
    num_orientations = int(args.num_orientations
                           or lab.get("num_orientations", 36))
    params = {
        "thetas": np.arange(num_orientations) * np.pi / float(num_orientations),
        "local_radius": float(args.local_radius
                              if args.local_radius is not None
                              else lab.get("local_radius", 0.25)),
        "dilation": int(args.dilation if args.dilation is not None
                        else lab.get("dilation", 1)),
        "boundary": int(args.boundary or lab.get("boundary_points", 48)),
        "interior_rings": int(args.interior_rings
                              or lab.get("interior_rings", 2)),
        "interior_angles": int(args.interior_angles
                               or lab.get("interior_angles", 12)),
        "binary_iters": int(args.binary_iters or lab.get("binary_iters", 8)),
        "min_semi_axis": float(args.min_semi_axis
                               if args.min_semi_axis is not None
                               else lab.get("min_semi_axis", 1e-3)),
        "skel_cfg": cfg.get("skeleton") or {},
    }
    report = {"processed": processed, "params": {k: v for k, v in params.items()
                                                 if k != "thetas"},
              "num_orientations": num_orientations, "per_split": {}}
    t_all = time.time()

    for split in args.splits:
        d = os.path.join(processed, split)
        n_full = int(len(np.load(os.path.join(d, "conditions.npy"))))
        n = n_full if args.limit is None else min(int(args.limit), n_full)
        if args.force and os.path.isdir(os.path.join(d, "_cache")):
            for f in os.listdir(os.path.join(d, "_cache")):
                if f.startswith("ell_"):
                    os.remove(os.path.join(d, "_cache", f))
        os.makedirs(os.path.join(d, "_cache"), exist_ok=True)
        cache_dir = os.path.join(d, "_cache")
        todo = [i for i in range(n)
                if not (args.resume and os.path.exists(
                    os.path.join(cache_dir, "ell_%06d.npz" % i)))]
        print("[%s] n=%d cached=%d todo=%d" % (split, n, n - len(todo),
                                               len(todo)), flush=True)
        t0 = time.time()
        results = []
        if todo:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=max(1, int(args.workers)),
                          initializer=_init_worker,
                          initargs=(processed, split, params)) as pool:
                for k, r in enumerate(pool.imap_unordered(_work, todo,
                                                          chunksize=2)):
                    results.append(r)
                    if (k + 1) % 100 == 0:
                        el = time.time() - t0
                        print("   %d/%d %.0fs (%.0f ms/sample)"
                              % (k + 1, len(todo), el,
                                 el / (k + 1) * 1000.0), flush=True)
        # always assemble the FULL split so a partial (resumable) run still
        # produces arrays the dataset can load
        s4, valid, missing = assemble(processed, split, n_full)
        failed = [r for r in results if not r.get("ok")]
        stats = {
            "n": int(n_full),
            "processed": int(n),
            "valid_entries": int(valid.sum()),
            "valid_fraction": float(valid.sum() / max(n * HORIZON, 1)),
            "samples_with_any_valid": int((valid.any(axis=1)).sum()),
            "samples_with_no_valid": int((~valid.any(axis=1)).sum()),
            "missing_cache": len(missing),
            "failed_samples": [r["idx"] for r in failed][:200],
            "failure_reasons": {str(r.get("error"))[:120]: 1 for r in failed},
            "seconds": float(time.time() - t0),
        }
        report["per_split"][split] = stats
        print("[%s] done n=%d valid=%.3f no_valid_samples=%d %.0fs"
              % (split, n, stats["valid_fraction"],
                 stats["samples_with_no_valid"], stats["seconds"]), flush=True)

    report["valid_fraction"] = float(np.mean(
        [v["valid_fraction"] for v in report["per_split"].values()])) \
        if report["per_split"] else 0.0
    report["seconds"] = float(time.time() - t_all)
    with open(os.path.join(processed, "preprocess_labels_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DONE", processed, "valid_fraction=%.4f"
          % report["valid_fraction"], flush=True)


if __name__ == "__main__":
    main()
