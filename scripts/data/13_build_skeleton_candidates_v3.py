"""13_build_skeleton_candidates_v3.py - V3 candidate preprocessing.

Runs the FIXED candidate generator (parallel branches preserved, multi-anchor
super source/sink, no unsafe fallback, dense safe geometry) for every OD pair
and stores, per split:

    candidate_features.npy         [N, M, L, 5]  float32  legacy diagnostics
    candidate_xy.npy               [N, M, L, 2]  float32  S_m network input
    candidate_mask.npy             [N, M]        bool
    candidate_lengths.npy          [N, M]        float32  scene arc length
    candidate_geometry.npy         [P, 2]        int16    dense cell chain (flat)
    candidate_geometry_offsets.npy [N, M+1]      int64    flat slice per candidate
    candidate_geometry_lengths.npy [N, M]        int32    points per candidate
    topology_best.npy              [N]           int64    argmin_m nDTW(P0, S_m)
    candidate_branch_counts.npy    [N, M]        int32    diagnostics

The dense geometry is stored as integer cell indices (lossless, half the size of
float32 scene coordinates).  It stays SEPARATE from the 128-point network
feature path: gamma_m(s) only ever runs on the dense chain.

    python scripts/data/13_build_skeleton_candidates_v3.py --config configs/config_v3_skeleton.yaml
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.skeleton_graph import load_graph_npz
from src.geometry.skeleton_paths import (CandidateConfig, generate_candidates,
                                         normalized_dtw)

MAZE_NAMES = ["umaze", "medium", "large"]


def load_graphs(skeleton_dir):
    graphs = []
    for name in MAZE_NAMES:
        path = os.path.join(skeleton_dir, name + ".npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "missing skeleton cache %s; run scripts/data/10_build_skeletons.py "
                "--out <base>/skeletons first" % path)
        graphs.append(load_graph_npz(path))
    return graphs


def build_split(split, source_dir, out_dir, graphs, cfg, limit=None):
    pos = np.load(os.path.join(source_dir, split, "positions.npy"))
    cond = np.load(os.path.join(source_dir, split, "conditions.npy"))
    mid = np.load(os.path.join(source_dir, split, "maze_id.npy"))
    n = len(pos) if limit is None else min(int(limit), len(pos))
    M, L = cfg.num_candidates, cfg.candidate_points

    features = np.zeros((n, M, L, 5), dtype=np.float32)
    xy = np.zeros((n, M, L, 2), dtype=np.float32)
    mask = np.zeros((n, M), dtype=bool)
    lengths = np.zeros((n, M), dtype=np.float32)
    branch_counts = np.zeros((n, M), dtype=np.int32)
    offsets = np.zeros((n, M + 1), dtype=np.int64)
    geom_lengths = np.zeros((n, M), dtype=np.int32)
    best = np.zeros(n, dtype=np.int64)
    flat = []
    cursor = 0
    cache = {}
    empty = 0
    t0 = time.time()
    for i in range(n):
        maze = int(mid[i])
        key = (maze,) + tuple(np.round(cond[i].reshape(-1), 5).tolist())
        cands = cache.get(key)
        if cands is None:
            cands = generate_candidates(graphs[maze], cond[i, 0], cond[i, 1], cfg)
            cache[key] = cands
        offsets[i, 0] = cursor
        for m in range(cands.num_slots):
            if m < M and cands.mask[m]:
                features[i, m] = cands.paths[m]
                xy[i, m] = cands.paths[m][:, :2]
                mask[i, m] = True
                lengths[i, m] = cands.lengths[m]
                branch_counts[i, m] = len(cands.branch_ids[m])
                px = np.rint(graphs[maze].scene_to_pixel(
                    cands.geometry[m])).astype(np.int16)
                flat.append(px)
                geom_lengths[i, m] = len(px)
                cursor += len(px)
            if m < M:
                offsets[i, m + 1] = cursor
        if cands.num_valid == 0:
            empty += 1
            best[i] = 0
        else:
            d = [normalized_dtw(pos[i], cands.metric_polyline(m))
                 for m in cands.valid_index()]
            best[i] = int(cands.valid_index()[int(np.argmin(d))])
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print("  %d/%d %.0fs (%.1f ms/sample)"
                  % (i + 1, n, el, el / (i + 1) * 1000), flush=True)

    geom = (np.concatenate(flat, axis=0) if flat
            else np.zeros((0, 2), dtype=np.int16))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "candidate_features.npy"), features)
    np.save(os.path.join(out_dir, "candidate_xy.npy"), xy)
    np.save(os.path.join(out_dir, "candidate_mask.npy"), mask)
    np.save(os.path.join(out_dir, "candidate_lengths.npy"), lengths)
    np.save(os.path.join(out_dir, "candidate_geometry.npy"), geom)
    np.save(os.path.join(out_dir, "candidate_geometry_offsets.npy"), offsets)
    np.save(os.path.join(out_dir, "candidate_geometry_lengths.npy"), geom_lengths)
    np.save(os.path.join(out_dir, "topology_best.npy"), best)
    np.save(os.path.join(out_dir, "candidate_branch_counts.npy"), branch_counts)

    valid = mask.sum(axis=1)
    # (branch_counts == 0) alone also counts the INVALID padding slots, which
    # inflate the "no branch" count by up to M-1 per sample.  Only valid
    # candidates with an empty branch set are real (and they are legitimate:
    # start and goal can sit next to the same skeleton node).
    no_branch = int((mask & (branch_counts == 0)).sum())
    stats = {
        "n": int(n), "empty_od": int(empty),
        "mean_valid": float(valid.mean()),
        "valid_hist": [int((valid == k).sum()) for k in range(M + 1)],
        "geometry_points_total": int(geom.shape[0]),
        "geometry_points_max": int(geom_lengths.max()) if n else 0,
        "geometry_mean_per_valid": float(geom_lengths.sum() / max(int(mask.sum()), 1)),
        "candidates_without_branch": no_branch,
        "candidates_without_branch_frac": float(
            no_branch / max(int(mask.sum()), 1)),
        "candidates_without_branch_incl_invalid": int((branch_counts == 0).sum()),
        "branch_hist": [int((branch_counts[mask] == k).sum())
                        for k in range(int(branch_counts.max()) + 1)]
        if int(mask.sum()) else [],
        "cache_entries": len(cache),
        "seconds": float(time.time() - t0),
    }
    print("[%s] n=%d empty=%d mean_valid=%.2f geom_pts=%d max_len=%d no_branch=%d"
          % (split, n, empty, stats["mean_valid"], geom.shape[0],
             stats["geometry_points_max"], no_branch), flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v3_skeleton.yaml")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = cfg["data"].get("source", "data/processed_scene_v1")
    base = cfg["data"].get("base", "data/processed_scene_v3")
    cand_cfg = CandidateConfig.from_dict(cfg.get("topology"), strict=False)
    graphs = load_graphs(os.path.join(base, "skeletons"))
    print("[graphs] " + ", ".join(
        "%s: %d nodes / %d branches" % (MAZE_NAMES[i], len(g.nodes), len(g.branches))
        for i, g in enumerate(graphs)), flush=True)

    report = {"source": source, "base": base,
              "candidate_config": cand_cfg.__dict__, "splits": {}}
    for split in args.splits:
        report["splits"][split] = build_split(
            split, source, os.path.join(base, split), graphs, cand_cfg,
            limit=args.limit)
    with open(os.path.join(base, "candidates_report_v3.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DONE", base)


if __name__ == "__main__":
    main()
