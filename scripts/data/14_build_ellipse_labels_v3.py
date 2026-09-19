"""14_build_ellipse_labels_v3.py - report section 20 label generation.

Builds, from the V3 compact candidate cache and the occupancy maps:

  <base>/skeletons/<maze>_shape4.npy       [H,W,4]  ShapeTable(q_k)
  <base>/skeletons/<maze>_shape_valid.npy  [H,W]    Valid(q_k)

  <base>/<split>/ellipse_center_gt.npy     [N,H,2]  c_i^* = Gamma*(s_i^*)
  <base>/<split>/ellipse_shape4_gt.npy     [N,H,4]  nearest ShapeTable lookup
  <base>/<split>/shape_valid.npy           [N,H]    label validity
  <base>/<split>/progress_gt.npy           [N,H]    monotone GT progress

The ShapeTable is built exactly as described in section 20.3:

  1. dilate the occupancy by ``occupancy_dilation`` cells -> O_safe;
  2. for each Skeleton dense point, enumerate N_theta orientations in [0, pi);
  3. run a grid ray traversal in the major (+/-) and minor (+/-) directions;
  4. build the initial axes with the local-radius cap;
  5. check the complete ellipse (boundary + interior samples) against O_safe;
  6. binary-search the largest safe uniform scale;
  7. keep the orientation with the largest safe area and re-order the axes so
     that a >= b, rotating theta by pi/2 when necessary.

A waypoint label is then ``ShapeTable(q_k)`` for the nearest dense point q_k of
the selected Skeleton Curve Gamma*; no interpolation of the angle is done.

    python scripts/data/14_build_ellipse_labels_v3.py \
        --config configs/config_v3_skeleton.yaml
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.skeleton_graph import load_graph_npz

MAZE_NAMES = ["umaze", "medium", "large"]
CELL = 2.0 / 256.0


# ---------------------------------------------------------------------------
# occupancy safety
# ---------------------------------------------------------------------------


def occupancy_safe(occ: np.ndarray, dilation_cells: int) -> np.ndarray:
    """1 = unsafe (dilated obstacle) in the scene-normalized map frame."""
    obstacle = np.asarray(occ) > 0.5
    if int(dilation_cells) > 0:
        obstacle = binary_dilation(
            obstacle, structure=np.ones((3, 3), dtype=bool),
            iterations=int(dilation_cells))
    return obstacle


def ray_clearance(safe: np.ndarray, center_px, direction,
                  max_dist_px: float) -> float:
    """DDA grid traversal: distance (pixels) to the first unsafe cell.

    The returned distance is measured to the entry face of the first unsafe
    cell, or ``max_dist_px`` when no unsafe cell is found inside the budget.
    """
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
    max_dist_px = float(max_dist_scene) / graph.cell
    return ray_clearance(safe, center_px, direction, max_dist_px) * graph.cell


# ---------------------------------------------------------------------------
# complete ellipse safety check
# ---------------------------------------------------------------------------


def ellipse_sample_points(center, a, b, theta, boundary=128,
                          interior_rings=4, interior_angles=32):
    """Deterministic boundary + interior coverage of one ellipse."""
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


# ---------------------------------------------------------------------------
# ShapeTable
# ---------------------------------------------------------------------------


_WORKER_SAFE = None
_WORKER_GRAPH = None
_WORKER_PARAMS = None


def _init_shape_worker(safe, graph, params):
    global _WORKER_SAFE, _WORKER_GRAPH, _WORKER_PARAMS
    _WORKER_SAFE = safe
    _WORKER_GRAPH = graph
    _WORKER_PARAMS = params


def _shape_for_point_with(safe, graph, params, yx):
    """One ShapeTable entry: returns (y, x, shape4 or None)."""
    y, x = int(yx[0]), int(yx[1])
    if safe[y, x]:
        return y, x, None
    center = graph.pixel_to_scene(np.array([x, y], dtype=np.float64))
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
        return y, x, None
    _, a, b, theta = best
    if a < b:
        a, b = b, a
        theta = theta + 0.5 * np.pi
    if a < float(min_axis) or b < float(min_axis):
        return y, x, None
    theta = float(theta % np.pi)
    return y, x, np.array([np.log(a), np.log(b), np.cos(2 * theta),
                           np.sin(2 * theta)], dtype=np.float32)


def _shape_for_point(yx):
    return _shape_for_point_with(_WORKER_SAFE, _WORKER_GRAPH, _WORKER_PARAMS, yx)


def build_shape_table(occ, graph, points_yx, num_orientations=36,
                      local_radius=0.25, dilation_cells=1, boundary=128,
                      interior_rings=4, interior_angles=32, binary_iters=12,
                      min_semi_axis=1e-3, workers=1):
    h, w = occ.shape
    safe = occupancy_safe(occ, dilation_cells)
    shape4 = np.zeros((h, w, 4), dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)

    points_yx = np.asarray(points_yx, dtype=np.int64).reshape(-1, 2)
    order = np.lexsort((points_yx[:, 1], points_yx[:, 0]))
    points_yx = points_yx[order]
    thetas = np.arange(int(num_orientations)) * np.pi / float(num_orientations)
    params = {
        "thetas": thetas,
        "local_radius": float(local_radius),
        "boundary": int(boundary),
        "interior_rings": int(interior_rings),
        "interior_angles": int(interior_angles),
        "binary_iters": int(binary_iters),
        "min_semi_axis": float(min_semi_axis),
    }
    t0 = time.time()
    n_done = 0

    def _consume(y, x, s4):
        nonlocal n_done
        if s4 is not None:
            shape4[y, x] = s4
            valid[y, x] = True
        n_done += 1
        if n_done % 2000 == 0:
            el = time.time() - t0
            print("    %d/%d  %.0fs (%.1f ms/point)"
                  % (n_done, len(points_yx), el,
                     el / max(n_done, 1) * 1000.0), flush=True)

    workers = int(workers or 1)
    points_list = [tuple(int(v) for v in p) for p in points_yx.tolist()]
    if workers > 1:
        ctx = mp.get_context("spawn")
        chunksize = max(1, len(points_list) // (workers * 8))
        with ctx.Pool(processes=workers, initializer=_init_shape_worker,
                      initargs=(safe, graph, params)) as pool:
            for y, x, s4 in pool.imap_unordered(_shape_for_point, points_list,
                                                chunksize=chunksize):
                _consume(y, x, s4)
    else:
        for yx in points_list:
            y, x, s4 = _shape_for_point_with(safe, graph, params, yx)
            _consume(y, x, s4)
    return shape4, valid


# ---------------------------------------------------------------------------
# waypoint labels
# ---------------------------------------------------------------------------


def build_shape_kdtree(shape4_lut, valid_lut):
    """KD-tree over the valid ShapeTable points (scene coordinates).

    The GT trajectory waypoint is used directly as the ellipse centre; its
    shape label is the nearest valid safe-ShapeTable entry.  No GT skeleton
    projection and no progress label are involved.
    """
    ys, xs = np.nonzero(valid_lut)
    if len(ys) == 0:
        return None, None, None
    pts = np.stack([(xs + 0.5) * CELL - 1.0,
                    (ys + 0.5) * CELL - 1.0], axis=1)
    return cKDTree(pts), ys, xs


def collect_dense_points(base, source, splits, geometry_points):
    """Union of all dense Skeleton-Curve cell centres used by the split cache."""
    points = {i: set() for i in range(len(MAZE_NAMES))}
    for split in splits:
        split_dir = os.path.join(base, split)
        if not os.path.exists(os.path.join(split_dir, "candidate_geometry.npy")):
            continue
        geometry = np.load(os.path.join(split_dir, "candidate_geometry.npy"),
                           mmap_mode="r")
        offsets = np.load(os.path.join(split_dir,
                                       "candidate_geometry_offsets.npy"))
        glens = np.load(os.path.join(split_dir, "candidate_geometry_lengths.npy"))
        cmask = np.load(os.path.join(split_dir, "candidate_mask.npy"))
        mid = np.load(os.path.join(source, split, "maze_id.npy"))
        for i in range(len(glens)):
            maze = int(mid[i])
            for m in range(glens.shape[1]):
                if not bool(cmask[i, m]):
                    continue
                n = int(glens[i, m])
                if n <= 0:
                    continue
                lo, hi = int(offsets[i, m]), int(offsets[i, m + 1])
                px = np.asarray(geometry[lo:hi])
                points[maze].update(tuple(v) for v in px.tolist())
        del geometry, offsets, glens, cmask, mid
        print("[points] %s: %s" % (split, {MAZE_NAMES[k]: len(v)
                                           for k, v in points.items()}),
              flush=True)
    return points


def build_split(split, source_dir, v3_dir, graphs, shape_luts, valid_luts,
                geometry_points, maze_names=None, limit=None,
                max_distance_cells=3.0):
    """Write the per-waypoint shape label.

    GT trajectory waypoint p_i^GT is used directly as the ellipse centre; the
    shape label is the nearest valid safe ShapeTable entry.  There is no GT
    skeleton projection, no s*, no ellipse_center_gt and no progress_gt.
    """
    pos = np.load(os.path.join(source_dir, split, "positions.npy"))
    mid = np.load(os.path.join(source_dir, split, "maze_id.npy"))
    split_dir = os.path.join(v3_dir, split)
    cmask = np.load(os.path.join(split_dir, "candidate_mask.npy"))

    n = len(pos) if limit is None else min(int(limit), len(pos))
    H = int(pos.shape[1])
    shape4_gt = np.zeros((n, H, 4), dtype=np.float32)
    shape_valid = np.zeros((n, H), dtype=bool)
    n_empty = 0
    n_shape = 0
    t0 = time.time()
    selected = None
    if maze_names:
        selected = {MAZE_NAMES.index(m) if isinstance(m, str) else int(m)
                    for m in maze_names}
    trees = {}
    for maze in range(len(MAZE_NAMES)):
        trees[maze] = build_shape_kdtree(shape_luts[maze], valid_luts[maze])
    max_dist = float(max_distance_cells) * CELL

    for i in range(n):
        maze = int(mid[i])
        if selected is not None and maze not in selected:
            continue
        if not bool(cmask[i].any()):
            n_empty += 1
            continue
        tree, ys, xs = trees[maze]
        if tree is None:
            n_empty += 1
            continue
        dist, nn = tree.query(np.asarray(pos[i], dtype=np.float64), k=1)
        ok = np.asarray(dist) <= max_dist
        if bool(ok.any()):
            kk = np.asarray(nn)[ok]
            shape4_gt[i, ok] = shape_luts[maze][ys[kk], xs[kk]]
            shape_valid[i, ok] = True
            n_shape += int(ok.sum())
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print("  %d/%d %.0fs (%.1f ms/sample)"
                  % (i + 1, n, el, el / (i + 1) * 1000.0), flush=True)

    np.save(os.path.join(split_dir, "ellipse_shape4_gt.npy"), shape4_gt)
    np.save(os.path.join(split_dir, "shape_valid.npy"), shape_valid)
    # Remove the projection-chain artifacts if an older run left them behind.
    for stale in ("ellipse_center_gt.npy", "progress_gt.npy"):
        path = os.path.join(split_dir, stale)
        if os.path.exists(path):
            os.remove(path)
    stats = {
        "n": int(n),
        "empty_or_invalid_candidate": int(n_empty),
        "shape_valid_entries": int(n_shape),
        "shape_valid_fraction": float(n_shape / max(n * H, 1)),
        "seconds": float(time.time() - t0),
    }
    print("[%s] n=%d empty=%d shape_valid=%.3f in %.0fs"
          % (split, n, n_empty, stats["shape_valid_fraction"],
             stats["seconds"]), flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v3_skeleton.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-orientations", type=int, default=36)
    ap.add_argument("--local-radius", type=float, default=0.25)
    ap.add_argument("--dilation", type=int, default=1)
    ap.add_argument("--boundary", type=int, default=128)
    ap.add_argument("--interior-rings", type=int, default=4)
    ap.add_argument("--interior-angles", type=int, default=32)
    ap.add_argument("--binary-iters", type=int, default=12)
    ap.add_argument("--min-semi-axis", type=float, default=1e-3)
    ap.add_argument("--max-shape-distance-cells", type=float, default=3.0)
    ap.add_argument("--mazes", nargs="*", default=MAZE_NAMES)
    ap.add_argument("--point-splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=0,
                    help="ShapeTable worker processes (0 = single process)")
    ap.add_argument("--skip-shape-table", action="store_true")
    ap.add_argument("--only-shape-table", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = args.source or cfg["data"].get("source", "data/processed_scene_v1")
    base = args.base or cfg["data"].get("base", "data/processed_scene_v3")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))
    skel_dir = os.path.join(base, "skeletons")
    graphs = [load_graph_npz(os.path.join(skel_dir, name + ".npz"))
              for name in MAZE_NAMES]

    shape_luts, valid_luts = [], []
    report = {"source": source, "base": base,
              "shape_table": {"num_orientations": args.num_orientations,
                              "local_radius": args.local_radius,
                              "dilation": args.dilation,
                              "boundary": args.boundary,
                              "interior_rings": args.interior_rings,
                              "interior_angles": args.interior_angles,
                              "binary_iters": args.binary_iters,
                              "min_semi_axis": args.min_semi_axis},
              "mazes": {}, "splits": {}}

    dense_points = {}
    if not args.skip_shape_table:
        print("[points] collecting the dense-point union ...", flush=True)
        dense_points = collect_dense_points(base, source, args.point_splits,
                                            geo_points)

    for i, name in enumerate(MAZE_NAMES):
        s4_path = os.path.join(skel_dir, name + "_shape4.npy")
        v_path = os.path.join(skel_dir, name + "_shape_valid.npy")
        build_this = name in args.mazes
        if (not build_this) or (args.skip_shape_table
                                and os.path.exists(s4_path)
                                and os.path.exists(v_path)):
            if not (os.path.exists(s4_path) and os.path.exists(v_path)):
                raise FileNotFoundError(
                    "%s ShapeTable is required for label generation but was "
                    "not built" % name)
            s4 = np.load(s4_path)
            valid = np.load(v_path).astype(bool)
        else:
            occ = np.load(os.path.join(source, "maps", name + ".npy"))
            points = set()
            for x, y in dense_points.get(i, set()):
                points.add((int(y), int(x)))
            ys, xs = np.nonzero(graphs[i].skeleton)
            points.update(zip(ys.tolist(), xs.tolist()))
            pts = np.asarray(sorted(points), dtype=np.int64)
            if len(pts) == 0:
                raise RuntimeError("no dense/skeleton points for %s" % name)
            print("[%s] building ShapeTable over %d points ..."
                  % (name, len(pts)), flush=True)
            s4, valid = build_shape_table(
                occ, graphs[i], pts, num_orientations=args.num_orientations,
                local_radius=args.local_radius, dilation_cells=args.dilation,
                boundary=args.boundary, interior_rings=args.interior_rings,
                interior_angles=args.interior_angles,
                binary_iters=args.binary_iters,
                min_semi_axis=args.min_semi_axis, workers=args.workers)
            np.save(s4_path, s4)
            np.save(v_path, valid)
            print("[%s] ShapeTable valid=%d/%d"
                  % (name, int(valid.sum()), len(pts)), flush=True)
        shape_luts.append(s4)
        valid_luts.append(valid)
        report["mazes"][name] = {
            "table_points": int(valid.size),
            "valid": int(valid.sum()),
            "valid_fraction": float(valid.sum() / max(valid.size, 1)),
        }

    if not args.only_shape_table:
        for split in args.splits:
            report["splits"][split] = build_split(
                split, source, base, graphs, shape_luts, valid_luts,
                geo_points, maze_names=args.mazes, limit=args.limit,
                max_distance_cells=args.max_shape_distance_cells)

    with open(os.path.join(base, "ellipse_labels_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DONE", base)


if __name__ == "__main__":
    main()
