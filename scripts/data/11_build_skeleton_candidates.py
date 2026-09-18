"""11_build_skeleton_candidates.py - V2 step 2: offline candidate topologies.

For every OD pair of every split it runs the *same* candidate generator that
inference uses (src.geometry.skeleton_paths.generate_candidates) and stores the
fixed-size padded candidate set plus the two ground-truth targets:

    candidate_paths.npy        [N, M, L, 5]  float32  [x, y, tx, ty, u]
    candidate_mask.npy         [N, M]        bool
    candidate_lengths.npy      [N, M]        float32  scene arc length
    candidate_branch_ids.npy   [N, M, B]     int32    (-1 padded)
    candidate_branch_counts    [N, M]        int32
    topology_target.npy        [N, M]        float32  soft nDTW target q_m
    topology_best.npy          [N]           int64    argmin_m nDTW (teacher)
    progress_gt.npy            [N, K]        float32  monotone arc length of GT

    python scripts/data/11_build_skeleton_candidates.py --config configs/config_v2_skeleton.yaml
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
from src.geometry.skeleton_paths import (
    CandidateConfig, generate_candidates, progress_target, soft_topology_target,
)

MAZE_NAMES = ["umaze", "medium", "large"]


def load_graphs(skeleton_dir):
    graphs = []
    for name in MAZE_NAMES:
        path = os.path.join(skeleton_dir, name + ".npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "missing skeleton cache %s; run scripts/data/10_build_skeletons.py"
                % path
            )
        graphs.append(load_graph_npz(path))
    return graphs


def build_split(split_dir, source_dir, graphs, cfg, limit=None):
    pos = np.load(os.path.join(source_dir, split_dir, "positions.npy"))
    cond = np.load(os.path.join(source_dir, split_dir, "conditions.npy"))
    mid = np.load(os.path.join(source_dir, split_dir, "maze_id.npy"))
    n = len(pos) if limit is None else min(int(limit), len(pos))
    M, L, K = cfg.num_candidates, cfg.candidate_points, pos.shape[1]

    paths = np.zeros((n, M, L, 5), dtype=np.float32)
    mask = np.zeros((n, M), dtype=bool)
    lengths = np.zeros((n, M), dtype=np.float32)
    max_branches = max(1, max((len(g.branches) for g in graphs), default=1))
    branch_ids = np.full((n, M, max_branches), -1, dtype=np.int32)
    branch_counts = np.zeros((n, M), dtype=np.int32)
    topo = np.zeros((n, M), dtype=np.float32)
    best = np.zeros(n, dtype=np.int64)
    progress = np.zeros((n, K), dtype=np.float32)

    cache = {}
    n_empty = 0
    n_valid_hist = np.zeros(M + 1, dtype=np.int64)
    t0 = time.time()
    for i in range(n):
        mi = int(mid[i])
        key = (mi, round(float(cond[i, 0, 0]), 5), round(float(cond[i, 0, 1]), 5),
               round(float(cond[i, 1, 0]), 5), round(float(cond[i, 1, 1]), 5))
        cands = cache.get(key)
        if cands is None:
            cands = generate_candidates(graphs[mi], cond[i, 0], cond[i, 1], cfg)
            cache[key] = cands
        if cands.num_valid == 0:
            n_empty += 1
            best[i] = 0
            continue
        paths[i] = cands.paths
        mask[i] = cands.mask
        lengths[i] = cands.lengths
        for m in cands.valid_index():
            ids = cands.branch_ids[m][:max_branches]
            branch_ids[i, m, :len(ids)] = ids
            branch_counts[i, m] = len(ids)
        q = soft_topology_target(pos[i], cands, tau=cfg.tau_gt)
        topo[i] = q.astype(np.float32)
        b = int(np.argmax(q))
        best[i] = b
        progress[i] = progress_target(pos[i], cands.coords[b], K).astype(np.float32)
        n_valid_hist[cands.num_valid] += 1
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1fs (%.2f ms/sample)" % (i + 1, n, el,
                                                      el / (i + 1) * 1000), flush=True)

    out = os.path.join(source_dir, "..", "processed_scene_v2", split_dir)
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "candidate_paths.npy"), paths)
    np.save(os.path.join(out, "candidate_mask.npy"), mask)
    np.save(os.path.join(out, "candidate_lengths.npy"), lengths)
    np.save(os.path.join(out, "candidate_branch_ids.npy"), branch_ids)
    np.save(os.path.join(out, "candidate_branch_counts.npy"), branch_counts)
    np.save(os.path.join(out, "topology_target.npy"), topo)
    np.save(os.path.join(out, "topology_best.npy"), best)
    np.save(os.path.join(out, "progress_gt.npy"), progress)

    valid = mask.sum(axis=1)
    ent = -(np.where(topo > 0, topo * np.log(np.maximum(topo, 1e-12)), 0.0)).sum(axis=1)
    mono = np.diff(progress, axis=1)
    stats = {
        "n": int(n),
        "empty_od": int(n_empty),
        "candidates_per_od_hist": n_valid_hist.tolist(),
        "mean_valid": float(valid.mean()),
        "topology_entropy_mean": float(ent.mean()),
        "topology_best_share_mean": float(topo.max(axis=1).mean()),
        "progress_monotonic_violations": int((mono < -1e-6).sum()),
        "progress_start_nonzero": int((np.abs(progress[:, 0]) > 1e-6).sum()),
        "progress_end_not_one": int((np.abs(progress[:, -1] - 1.0) > 1e-6).sum()),
        "cache_entries": len(cache),
        "out": out,
    }
    print("[%s] n=%d empty=%d mean_valid=%.2f entropy=%.3f best_share=%.3f"
          % (split_dir, n, n_empty, stats["mean_valid"],
             stats["topology_entropy_mean"], stats["topology_best_share_mean"]))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v2_skeleton.yaml")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="debug: cap samples")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = cfg["data"].get("source", "data/processed_scene_v1")
    base = cfg["data"].get("base", "data/processed_scene_v2")
    cand_cfg = CandidateConfig.from_dict(cfg.get("topology"), strict=False)
    graphs = load_graphs(os.path.join(base, "skeletons"))
    print("[graphs] " + ", ".join(
        "%s: %d nodes / %d branches" % (MAZE_NAMES[i], len(g.nodes), len(g.branches))
        for i, g in enumerate(graphs)), flush=True)

    report = {"source": source, "base": base,
              "candidate_config": cand_cfg.__dict__, "splits": {}}
    for split in args.splits:
        report["splits"][split] = build_split(split, source, graphs, cand_cfg,
                                              limit=args.limit)
    with open(os.path.join(base, "candidates_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DONE", base)


if __name__ == "__main__":
    main()
