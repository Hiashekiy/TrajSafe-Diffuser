"""V3 candidate quality check: can the candidates even fit the demonstration?

topology_best always picks SOME candidate, so m* alone says nothing about
whether L_align / L_topo can converge.  This tool reports, per split:

    best_nDTW      distribution of min_m nDTW(P0, S_m)   <- the floor of L_align
    recall@tau     fraction of samples whose BEST candidate is within tau
    best_rate      does the stored topology_best agree with the argmin here
    valid/multi    how many candidates exist (and how often more than one)

It decodes the 128-point metric polylines out of the stored candidate features
(channels 0:2), so it never re-runs the (slow) route generator.

    python scripts/debug/v3_candidate_recall.py --split test --samples 1000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.geometry.skeleton_paths import normalized_dtw, resample_polyline
from src.utils.config import load_config

TAUS = (0.02, 0.05, 0.10, 0.20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v3_skeleton.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = cfg["data"].get("source", "data/processed_scene_v1")
    base = cfg["data"].get("base", "data/processed_scene_v3")
    n_pts = int((cfg.get("topology") or {}).get("candidate_points", 128))

    split_dir = os.path.join(base, args.split)
    src_dir = os.path.join(source, args.split)
    pos = np.load(os.path.join(src_dir, "positions.npy"))
    feats = np.load(os.path.join(split_dir, "candidate_features.npy"),
                    mmap_mode="r")
    mask = np.load(os.path.join(split_dir, "candidate_mask.npy"))
    best = np.load(os.path.join(split_dir, "topology_best.npy"))

    n = len(pos)
    step = max(1, n // max(1, args.samples))
    idxs = list(range(0, n, step))
    best_d, oracle, agree, n_valid, multi, empty = [], [], 0, [], 0, 0
    t0 = time.time()
    for c, i in enumerate(idxs):
        valid = np.nonzero(mask[i])[0]
        if valid.size == 0:
            empty += 1
            continue
        gt = resample_polyline(pos[i], n_pts)
        d = np.array([normalized_dtw(gt, np.asarray(feats[i, m, :, :2]))
                      for m in valid])
        k = int(np.argmin(d))
        best_d.append(float(d[k]))
        oracle.append(int(valid[k]) == int(best[i]))
        agree += int(int(valid[k]) == int(best[i]))
        n_valid.append(int(valid.size))
        multi += int(valid.size > 1)
        if (c + 1) % 200 == 0:
            print("  %d/%d %.0fs" % (c + 1, len(idxs), time.time() - t0),
                  flush=True)

    d = np.asarray(best_d)
    report = {
        "split": args.split, "n_total": int(n), "n_checked": len(idxs),
        "n_scored": int(d.size), "empty_od": int(empty),
        "mean_valid": float(np.mean(n_valid)) if n_valid else 0.0,
        "multi_candidate_frac": float(multi / max(len(n_valid), 1)),
        "best_ndtw_mean": float(d.mean()) if d.size else None,
        "best_ndtw_median": float(np.median(d)) if d.size else None,
        "best_ndtw_p10": float(np.percentile(d, 10)) if d.size else None,
        "best_ndtw_p90": float(np.percentile(d, 90)) if d.size else None,
        "best_ndtw_max": float(d.max()) if d.size else None,
        "topology_best_agrees": float(agree / max(len(oracle), 1)),
        "recall": {("tau_%.2f" % t): float((d <= t).mean()) for t in TAUS},
        "seconds": float(time.time() - t0),
    }
    txt = json.dumps(report, indent=2)
    print(txt)
    out = args.out or os.path.join(base, "candidate_recall_%s.json" % args.split)
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("saved", out)


if __name__ == "__main__":
    main()
