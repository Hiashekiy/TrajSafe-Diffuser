"""Centre the cached PDG-Focal A* paths (train/val/test) before training.

Why: inference now centres the searched path (`grid_search.path_centering`) and the
safety ellipses/corridor are built on THAT path.  The cached `pdg_path_points.npy`
was generated before centering existed, so a model trained on it would see a
different path distribution than it gets at inference.

This script rewrites `pdg_path_points.npy` in place after saving the originals as
`pdg_path_points_raw.npy`.  It does NOT touch validity/status arrays, and it does
not change start/goal (centring pins the endpoints by construction).

    python scripts/data/new_framework/14_center_pdg_paths.py \
        --root data/processed/160m-k4p-c32q64 [--splits train val test] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(__file__)))))
sys.path.insert(0, ROOT)

from src.grid_search.path_centering import center_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--cell_m", type=float, default=0.625)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--step_cells", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report = {"root": args.root, "params": {"iters": args.iters,
                                            "step_cells": args.step_cells,
                                            "smooth": args.smooth,
                                            "cell_m": args.cell_m},
              "splits": {}}
    for split in args.splits:
        sp = os.path.join(args.root, split)
        p_path = os.path.join(sp, "pdg_path_points.npy")
        p_occ = os.path.join(sp, "occupancy.npy")
        if not (os.path.exists(p_path) and os.path.exists(p_occ)):
            print("[skip] %s (missing files)" % split)
            continue
        paths = np.load(p_path)
        occ = np.load(p_occ, mmap_mode="r")
        n = len(paths) if args.limit is None else min(args.limit, len(paths))
        out = paths.copy()
        before, after = [], []
        for i in range(n):
            free = np.asarray(occ[i])
            free = (free[0] if free.ndim == 3 else free) < 0.5
            c, st = center_path(paths[i], free, args.cell_m, iters=args.iters,
                                step_cells=args.step_cells, smooth=args.smooth)
            out[i] = c
            before.append(st["clearance_before"])
            after.append(st["clearance_after"])
            if (i + 1) % 500 == 0:
                print("   [%s] %d/%d" % (split, i + 1, n), flush=True)
        drift = float(np.abs(out[:n, [0, -1]] - paths[:n, [0, -1]]).max())
        stats = {"n": int(n), "clearance_before_mean": float(np.mean(before)),
                 "clearance_after_mean": float(np.mean(after)),
                 "endpoint_drift": drift,
                 "clearance_before_min": float(np.min(before)),
                 "clearance_after_min": float(np.min(after))}
        report["splits"][split] = stats
        print("[center] %s n=%d  clearance %.2f -> %.2f m (min %.2f -> %.2f)  "
              "endpoint drift %.1e" % (split, n, stats["clearance_before_mean"],
                                       stats["clearance_after_mean"],
                                       stats["clearance_before_min"],
                                       stats["clearance_after_min"], drift))
        if args.dry_run:
            continue
        raw = os.path.join(sp, "pdg_path_points_raw.npy")
        if not os.path.exists(raw):
            shutil.copy2(p_path, raw)
        np.save(p_path, out.astype(np.float32))
        print("[write] %s (backup: %s)" % (p_path, os.path.basename(raw)))

    with open(os.path.join(args.root, "center_paths_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("[report] center_paths_report.json")


if __name__ == "__main__":
    main()
