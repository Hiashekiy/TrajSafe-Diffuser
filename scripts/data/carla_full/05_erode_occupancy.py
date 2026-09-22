#!/usr/bin/env python
"""05_erode_occupancy.py - widen the free space by eroding the obstacles.

The 160 m windows are ~81 % non-road, and the free space is a thin network of
4-8 m channels, so a learned trajectory has very little room for error.  This
script writes a NEW processed root whose occupancy has the obstacles eroded,
which widens every channel by k cells on each side (1 cell = 0.625 m here).

    binary_erosion(obstacle, structure=ones(3,3), iterations=k)

Iterating a 3x3 structuring element k times is EQUIVALENT to one erosion with a
(2k+1)x(2k+1) square: repeating a small kernel is not gentler than a big one.
And because the obstacles are SOLID BLOCKS tens of metres thick (median
inscribed radius 8-59 m, 2-14 components per scene), reaching a 2/3 free ratio
would require eating ~30 cells (~19 m) into every boundary, which erases the
whole road network.  Use -k/--k for a sane widening and --target-free only if
you really want the network gone.

Only the BASE arrays are copied; candidate / ellipse / ALM labels depend on the
occupancy and must be regenerated afterwards:

    python scripts/data/carla_full/05_erode_occupancy.py \
        --source data/carla_processed_160 --out data/carla_processed_160k4 --k 4
    python scripts/data/carla/01_build_candidates.py     --processed <out> --config <cfg>
    python scripts/data/carla/02_build_ellipse_labels.py --processed <out> --config <cfg>
    python scripts/data/carla_full/04_build_alm_constraints.py --processed <out> --config <cfg>
    python scripts/data/carla/03_validate_processed.py   --processed <out> --config <cfg>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.ndimage import binary_erosion

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

SPLITS = ("train", "val", "test")
#: files that do NOT depend on the occupancy and can be copied verbatim
BASE_FILES = ("conditions.npy", "control_gt.npy", "curve_gt.npy",
              "episode_id.npy", "sample_id.npy")
STRUCT = np.ones((3, 3), dtype=bool)


def resolve(p) -> str:
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def erode_to_target(obstacle: np.ndarray, target: float, k_max: int = 80):
    """Smallest k with free ratio >= target (k_max if unreachable)."""
    cur = obstacle
    if 1.0 - cur.mean() >= target:
        return cur, 0
    for k in range(1, k_max + 1):
        cur = binary_erosion(cur, structure=STRUCT)
        if 1.0 - cur.mean() >= target:
            return cur, k
    return cur, k_max


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/carla_processed_160")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=None,
                    help="fixed number of 3x3 erosion iterations")
    ap.add_argument("--target-free", type=float, default=None,
                    help="per-scene adaptive erosion until the free ratio "
                         "reaches this value (e.g. 0.6667)")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--copy-base", action="store_true", default=True)
    args = ap.parse_args()

    if args.k is None and args.target_free is None:
        args.k = 4
    src, out = resolve(args.source), resolve(args.out)
    if os.path.abspath(src) == os.path.abspath(out):
        print("[error] --out must differ from --source")
        return 2
    print("[erode] source=%s" % src)
    print("[erode] out   =%s" % out)
    print("[erode] mode  =%s"
          % (("k=%d (3x3 iterations)" % args.k) if args.k is not None
             else "adaptive to free>=%.3f" % args.target_free))

    os.makedirs(out, exist_ok=True)
    report = {"source": src, "out": out, "k": args.k,
              "target_free": args.target_free, "per_split": {}}
    t0 = time.time()
    for split in args.splits:
        d_in, d_out = os.path.join(src, split), os.path.join(out, split)
        if not os.path.isdir(d_in):
            continue
        os.makedirs(d_out, exist_ok=True)
        occ_path = os.path.join(d_in, "occupancy.npy")
        occ = np.load(occ_path, mmap_mode="r")
        n = len(occ) if args.limit is None else min(int(args.limit), len(occ))
        new = np.zeros((len(occ), occ.shape[1], occ.shape[2]), np.uint8)
        ks, fr_before, fr_after = [], [], []
        for i in range(n):
            ob = np.asarray(occ[i]) > 0
            fr_before.append(1.0 - ob.mean())
            if args.k is not None:
                nb, k = binary_erosion(ob, structure=STRUCT,
                                       iterations=int(args.k)), int(args.k)
            else:
                nb, k = erode_to_target(ob, float(args.target_free))
            ks.append(k)
            fr_after.append(1.0 - nb.mean())
            # nb is True where an OBSTACLE survived the erosion, and the
            # occupancy convention is 0 = free / 1 = obstacle, so it is written
            # straight through (inverting it here silently breaks every
            # downstream stage: skeleton, candidates, ellipse labels, ALM).
            new[i] = nb.astype(np.uint8)
        for i in range(n, len(occ)):          # untouched tail keeps its data
            new[i] = np.asarray(occ[i])
        np.save(os.path.join(d_out, "occupancy.npy"), new)
        if args.copy_base:
            for f in BASE_FILES:
                p = os.path.join(d_in, f)
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(d_out, f))
        man = os.path.join(src, "clean_manifest.jsonl")
        if os.path.exists(man):
            shutil.copy2(man, os.path.join(out, "clean_manifest.jsonl"))
        st = {"n": int(n),
              "k_p50": int(np.median(ks)), "k_p90": int(np.percentile(ks, 90)),
              "k_max": int(np.max(ks)),
              "free_before_mean": float(np.mean(fr_before)),
              "free_after_mean": float(np.mean(fr_after)),
              "free_after_min": float(np.min(fr_after)),
              "cells_widened_m": float(np.median(ks) * 0.625 * 2)}
        report["per_split"][split] = st
        print("[%s] n=%d  free %.3f -> %.3f  k p50=%d p90=%d  (channel +%.2f m)"
              % (split, n, st["free_before_mean"], st["free_after_mean"],
                 st["k_p50"], st["k_p90"], st["cells_widened_m"]), flush=True)
    report["seconds"] = float(time.time() - t0)
    with open(os.path.join(out, "erode_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DONE %s" % out)
    print("next: 01_build_candidates -> 02_build_ellipse_labels -> "
          "04_build_alm_constraints -> 03_validate_processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
