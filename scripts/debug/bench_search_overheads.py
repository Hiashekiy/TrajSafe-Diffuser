"""Microbenchmarks for the fixed per-sample costs of the step-1 search branch.

Uses a REAL occupancy crop from the processed dataset (not noise) so the
clearance/centering/neighbour costs are the ones the pipeline actually pays.

    python scripts/debug/bench_search_overheads.py
"""

from __future__ import annotations

import importlib
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from scipy.ndimage import distance_transform_edt      # noqa: E402

gm = importlib.import_module("src.grid_search.grid_map")
gn = importlib.import_module("src.grid_search.grid_neighbors")
pc = importlib.import_module("src.grid_search.path_centering")
lg = importlib.import_module("src.grid_search.lazy_guidance")
dp = importlib.import_module("src.grid_search.diffusion_path")


def timeit(fn, n=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def load_one_map():
    import torch
    from src.datasets.carla_spline_dataset import make_loader
    from src.utils.config import load_config, num_controls, num_safety_queries
    cfg = load_config("configs/config_v5_c32q128.yaml")
    loader, _ = make_loader("test", cfg["data"]["processed_root"], batch_size=1,
                            shuffle=False, num_workers=0, limit=1,
                            require_labels=False, num_controls=num_controls(cfg),
                            num_safety_queries=num_safety_queries(cfg))
    b = next(iter(loader))
    occ = b["occupancy"][0, 0].numpy().astype(np.uint8)
    cond = b["cond"][0].numpy().astype(np.float64)
    pos = b.get("pos")
    gt = pos[0].numpy().astype(np.float64) if pos is not None else None
    return occ, cond, gt


def main():
    occ, cond, gt = load_one_map()
    H, W = occ.shape
    spec = gm.GridSpec(H, W, 2.0 * 80.0 / max(H, W), 80.0)
    raw_free = occ == 0
    safe = gm.build_safe_grid(occ, spec, robot_radius_m=0.8, safety_margin_m=0.3)
    free = safe.free
    print("grid %dx%d  raw free %.1f%%  safe free %.1f%%  cell %.3f m"
          % (H, W, 100 * raw_free.mean(), 100 * free.mean(), spec.cell_m))

    print("\n--- EDT work per sample (scipy, 256x256) ---")
    print("  distance_transform_edt(~free)                       %6.2f ms"
          % timeit(lambda: distance_transform_edt(~raw_free)))
    print("  distance_transform_edt(~free, return_indices=True)  %6.2f ms  <- center_path, UNUSED"
          % timeit(lambda: distance_transform_edt(~raw_free, return_indices=True)))
    print("  clearance_field(raw_free)  [inflate_obstacles]      %6.2f ms"
          % timeit(lambda: gm.clearance_field(raw_free, spec.cell_m)))
    print("  clearance_field(safe_free) [clearance_m]            %6.2f ms"
          % timeit(lambda: gm.clearance_field(free, spec.cell_m)))
    print("  build_safe_grid()  (2 x clearance_field inside)     %6.2f ms"
          % timeit(lambda: gm.build_safe_grid(occ, spec), 10))

    print("\n--- neighbour table (per sample) ---")
    print("  build_neighbor_table()  current                     %6.2f ms"
          % timeit(lambda: gn.build_neighbor_table(free, spec.cell_m), 10))
    ref_tab = gn.build_neighbor_table(free, spec.cell_m)

    def build_slice(free_mask, cell_m):
        f = np.ascontiguousarray(free_mask, dtype=bool)
        h, w = f.shape
        pad = np.zeros((h + 2, w + 2), dtype=bool)
        pad[1:-1, 1:-1] = f
        rr = np.arange(h, dtype=np.int32)[:, None]
        cc = np.arange(w, dtype=np.int32)[None, :]
        idx = np.empty((h, w, 8), dtype=np.int32)
        for k, (dr, dc) in enumerate(gn.MOVES):
            ok = pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
            if gn.DIAGONAL[k]:
                ok = ok & pad[1 + dr:1 + dr + h, 1:-1] & pad[1:-1, 1 + dc:1 + dc + w]
            np.copyto(idx[..., k], (rr + dr) * w + (cc + dc), where=ok)
            idx[..., k][~ok] = -1
        cost = np.array([np.sqrt(2.0) if d else 1.0 for d in gn.DIAGONAL]) * cell_m
        return gn.NeighborTable(idx=idx.reshape(h * w, 8), cost=cost,
                                free=f.reshape(-1), width=int(w))

    lean = build_slice(free, spec.cell_m)
    print("  build_slice() 2D-slicing                           %6.2f ms   identical=%s"
          % (timeit(lambda: build_slice(free, spec.cell_m), 10),
             np.array_equal(lean.idx, ref_tab.idx)))

    # scalar per-expansion neighbour lookup (what a lean A* would do)
    flat_free = free.reshape(-1)
    h, w = free.shape

    def scalar_nbr(i):
        r, c = divmod(i, w)
        out = []
        for dr, dc in gn.MOVES:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and flat_free[rr * w + cc]:
                out.append(rr * w + cc)
        return out

    t0 = time.perf_counter()
    for i in range(2000):
        scalar_nbr(1000 + i)
    print("  scalar 8-neighbour lookup, per expansion           %6.2f us"
          % ((time.perf_counter() - t0) / 2000 * 1e6))

    print("\n--- centering (per sample) ---")
    if gt is not None:
        path = gt.copy()
        path[0], path[-1] = cond[0], cond[1]
    else:
        path = np.stack([np.linspace(-0.8, 0.8, 128),
                         np.linspace(-0.5, 0.6, 128)], -1)
    print("  center_path(iters=40)  full                        %6.2f ms"
          % timeit(lambda: pc.center_path(path, raw_free, spec.cell_m, iters=40), 10))
    print("  center_path(iters=10)                              %6.2f ms"
          % timeit(lambda: pc.center_path(path, raw_free, spec.cell_m, iters=10), 10))
    print("  _bilinear (1 call)                                 %6.3f ms" %
          timeit(lambda: pc._bilinear(raw_free.astype(float), path), 300))
    print("  clearance_field(raw_free) inside center_path       %6.2f ms"
          % timeit(lambda: gm.clearance_field(raw_free, spec.cell_m)))
    # vectorised bilinear for all 128 points at once vs 122 tiny calls
    print("  => 3 x 40 = 120 tiny _bilinear calls dominate")
    print("  np.gradient(clear)                                 %6.2f ms"
          % timeit(lambda: np.gradient(np.random.rand(H, W)), 10))

    print("\n--- LazyGuidance.score (one call) ---")
    if gt is not None:
        pts = spec.world_to_cell(gt)
    else:
        pts = np.stack([np.linspace(10, 240, 128), np.linspace(10, 240, 128)], -1)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)]) * spec.cell_m
    ref = dp.ReferencePath(pts, cum, float(cum[-1]), True)
    print("  reference: %d segments, %.1f m" % (ref.n_segments, ref.length_m))
    rng = np.random.default_rng(0)
    for band in (1, 64, 264, 736, 3000):
        idx = rng.choice(H * W, band, replace=False)
        g = np.full(band, 50.0)
        fresh = lg.LazyGuidance(ref, spec, J=3)
        t0 = time.perf_counter()
        fresh.score(idx, g)                     # cold: fills the per-cell cache
        t_cold = (time.perf_counter() - t0) * 1000
        t_warm = timeit(lambda: fresh.score(idx, g), 30)
        print("  band=%6d   cold %8.1f us   warm %7.1f us" % (band, t_cold * 1000,
                                                              t_warm * 1000))


if __name__ == "__main__":
    main()
