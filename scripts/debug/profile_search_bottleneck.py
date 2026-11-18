"""Where does the v5 path search actually spend its time?

Runs the real sampler with ``steps=1`` (so ONLY the step-1 PDG-Focal A* branch
executes) and attributes wall-clock time to every stage with monkey-patched
timers, then prints an aggregate table + a cProfile breakdown of the search.

    python scripts/debug/profile_search_bottleneck.py --limit 40
    python scripts/debug/profile_search_bottleneck.py --limit 420 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

T = defaultdict(float)          # accumulated seconds per stage
C = defaultdict(int)            # call counts
S = defaultdict(float)          # accumulated "amounts" (cells, nodes, ...)
FOCAL = []                      # FOCAL band size per expansion (one sample)

from src.datasets.carla_spline_dataset import make_loader          # noqa: E402
from src.diffusion.schedule import NoiseSchedule                     # noqa: E402
from src.models.factory import build_model                           # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402


def timed(mod, name, key, extra=None):
    orig = getattr(mod, name)
    if not callable(orig):
        return orig

    def fn(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            T[key] += dt
            C[key] += 1
            if extra is not None:
                extra(a, k, dt)
    fn.__wrapped__ = orig
    setattr(mod, name, fn)
    return orig


def install_probes(reset_focal=False):
    """Patch every stage of the step-1 search path with a timer."""
    import importlib
    import scipy.ndimage as ndi
    diffusion_path = importlib.import_module("src.grid_search.diffusion_path")
    focal_astar = importlib.import_module("src.grid_search.focal_astar")
    grid_map = importlib.import_module("src.grid_search.grid_map")
    grid_neighbors = importlib.import_module("src.grid_search.grid_neighbors")
    path_centering = importlib.import_module("src.grid_search.path_centering")
    lazy_guidance = importlib.import_module("src.grid_search.lazy_guidance")

    timed(ndi, "distance_transform_edt", "scipy_edt")
    timed(grid_map, "clearance_field", "clearance_field")
    timed(grid_map, "inflate_obstacles", "inflate_obstacles")
    timed(grid_map, "build_safe_grid", "build_safe_grid")
    timed(grid_neighbors, "build_neighbor_table", "build_neighbor_table")
    timed(diffusion_path, "prepare_reference_path", "prepare_reference")

    # LazyGuidance: whole-object + the inner 65536xJ cache fill
    orig_score = lazy_guidance.LazyGuidance.score
    orig_values = lazy_guidance.LazyGuidance._values

    def score(self, node_idx, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig_score(self, node_idx, *a, **k)
        finally:
            dt = time.perf_counter() - t0
            idx = np.asarray(node_idx).reshape(-1)
            T["guidance_score"] += dt
            C["guidance_score"] += 1
            S["guidance_query_cells"] += len(idx)

    def values(self, idx):
        t0 = time.perf_counter()
        try:
            return orig_values(self, idx)
        finally:
            dt = time.perf_counter() - t0
            n = len(np.asarray(idx).reshape(-1))
            T["guidance_values"] += dt
            C["guidance_values"] += 1
            S["guidance_values_cells"] += n
            S["guidance_segment_tests"] += n * max(int(self.n_segments), 1)

    lazy_guidance.LazyGuidance.score = score
    lazy_guidance.LazyGuidance._values = values

    # FOCAL band size / expansion count, from inside the loop
    if reset_focal:
        FOCAL.clear()

    def tiebreak(cand_idx, *a, **k):
        FOCAL.append(len(cand_idx))
        return orig_tiebreak(cand_idx, *a, **k)

    orig_tiebreak = focal_astar._tiebreak_pick
    focal_astar._tiebreak_pick = tiebreak

    timed(focal_astar, "focal_search", "focal_search")
    timed(path_centering, "center_path", "center_path_total")
    timed(path_centering, "_bilinear", "center_bilinear")
    timed(path_centering, "clearance_field", "center_clearance_edt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q128.yaml")
    ap.add_argument("--ckpt", default="outputs/models/v5_c32q128/ckpt/final_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--profile", type=int, default=3,
                    help="how many samples to also cProfile (0 = none)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--only-ids", default=None,
                    help="comma separated sample ids to profile (others skipped)")
    ap.add_argument("--legacy", action="store_true",
                    help="force the pre-optimisation search (focal_w 1.2, octile h)")
    ap.add_argument("--focal-w", type=float, default=None, help="override focal_w")
    ap.add_argument("--heuristic", default=None, help="override search heuristic")
    args = ap.parse_args()

    from src.diffusion.sampler_v5 import sample_v5

    cfg = load_config(args.config)
    if args.legacy:
        cfg.setdefault("grid_search", {})["focal_w"] = 1.2
        cfg["grid_search"]["heuristic"] = "octile"
    if args.focal_w is not None:
        cfg.setdefault("grid_search", {})["focal_w"] = float(args.focal_w)
    if args.heuristic is not None:
        cfg.setdefault("grid_search", {})["heuristic"] = args.heuristic
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader, ds = make_loader(
        args.split, cfg["data"]["processed_root"], batch_size=args.batch,
        shuffle=False, num_workers=0, limit=args.limit, require_labels=False,
        num_controls=num_controls(cfg), num_safety_queries=num_safety_queries(cfg))
    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                             "squaredcos_cap_v2")).to(device)
    model = build_model(cfg).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()

    install_probes()

    from src.grid_search.grid_map import GridSpec        # noqa: F401

    rows = []
    profiled = 0
    pending = []
    only = (set(int(x) for x in args.only_ids.split(","))
            if args.only_ids else None)
    for bi, batch in enumerate(loader):
        if only is not None and int(batch["sample_id"][0]) not in only:
            continue
        if bi < args.warmup:
            sample_v5(model, sched, batch, cfg, device=device, steps=1, seed=bi)
            continue
        T.clear()
        C.clear()
        S.clear()
        FOCAL.clear()
        if profiled < args.profile:
            import cProfile
            import pstats
            pr = cProfile.Profile()
            pr.enable()
            out = sample_v5(model, sched, batch, cfg, device=device, steps=1, seed=bi)
            pr.disable()
            store = io_string()
            pstats.Stats(pr, stream=store).sort_stats("tottime").print_stats(22)
            pending.append((int(batch["sample_id"][0]),
                            int((out["trace"][0]["search"] or [{}])[0].get("expanded", 0)),
                            store.getvalue()))
            profiled += 1
        else:
            out = sample_v5(model, sched, batch, cfg, device=device, steps=1, seed=bi)
        snap = dict(T)
        counts = dict(C)
        amounts = dict(S)
        st = (out["trace"][0]["search"] or [{}])[0]
        rows.append({
            "sample": int(batch["sample_id"][0]),
            "status": st.get("status"),
            "fallback": st.get("fallback"),
            "guide_valid": st.get("guide_valid"),
            "expanded": st.get("expanded"),
            "search_ms": st.get("search_ms"),
            "grid_ms": out["grid_ms"],
            "total_ms": out["total_ms"],
            "focal_mean": float(np.mean(FOCAL)) if FOCAL else 0.0,
            "focal_max": int(np.max(FOCAL)) if FOCAL else 0,
            "focal_iters": len(FOCAL),
            "focal_sum": float(np.sum(FOCAL)) if FOCAL else 0.0,
            **{"t_" + k: v * 1000.0 for k, v in snap.items()},
            **{"c_" + k: v for k, v in counts.items()},
            **{"s_" + k: v for k, v in amounts.items()},
        })
        print("[prof] %d  exp=%s  search=%.1f ms  focal_mean=%.0f  guide_valid=%s"
              % (rows[-1]["sample"], rows[-1]["expanded"], rows[-1]["search_ms"] or 0.0,
                 rows[-1]["focal_mean"], rows[-1]["guide_valid"]), flush=True)

    # ------------------------------------------------------------------ report
    def col(k, default=0.0):
        return np.asarray([r.get(k, default) if r.get(k) is not None else default
                           for r in rows], dtype=np.float64)

    n = len(rows)
    keys = [c for c in rows[0] if c.startswith("t_") and not c.startswith("t_c_")]
    print("\n=== per-sample wall clock, mean over %d samples ===" % n)
    search_ms = np.maximum(col("search_ms"), 1e-9)
    for k in sorted(keys, key=lambda x: -col(x, 0.0).mean()):
        v = col(k)
        print("  %-28s mean %8.2f ms   median %7.2f   p90 %8.2f   max %8.2f   "
              "share_of_search %5.1f%%"
              % (k, v.mean(), np.median(v), np.percentile(v, 90), v.max(),
                 100.0 * v.mean() / search_ms.mean()))

    print("\n=== amounts (mean per sample) ===")
    for k in sorted([c for c in rows[0] if c.startswith("s_") or c.startswith("c_")
                     or c in ("focal_mean", "focal_max", "focal_iters", "focal_sum",
                              "expanded", "grid_ms", "total_ms", "search_ms")]):
        v = col(k)
        if k.startswith("c_"):
            print("  %-28s mean %10.1f" % (k, v.mean()))
        elif k.startswith("s_guidance_segment_tests"):
            print("  %-28s mean %10.0f" % (k, v.mean()))
        else:
            print("  %-28s mean %10.2f   median %9.2f   max %10.2f"
                  % (k, v.mean(), np.median(v), v.max()))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n[csv] ->", args.csv)

    # correlation between expansions and search time
    e, t = col("expanded"), col("search_ms")
    print("\nexpanded vs search_ms: median expansions %.0f, median search %.1f ms, "
          "%.2f ms per 100 expansions"
          % (np.median(e), np.median(t),
             (t[-1] / e[-1] * 100) if e[-1] else float("nan")))
    print("fixed part (expanded<200): median %.1f ms"
          % (np.median(t[e < 200]) if (e < 200).any() else float("nan")))
    print("heavy part (expanded>2000): median %.1f ms, n=%d"
          % (np.median(t[e > 2000]) if (e > 2000).any() else float("nan"),
             int((e > 2000).sum())))

    for sid, exp, text in pending:
        print("\n=== cProfile sample %d (expanded=%d) ===" % (sid, exp))
        print(text[:6000])


def io_string():
    import io
    return io.StringIO()


if __name__ == "__main__":
    main()
