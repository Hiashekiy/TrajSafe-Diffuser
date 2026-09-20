"""01_build_candidates.py - CARLA per-sample candidate cache.

For every cleaned sample (canonical occupancy, conditions, curve_gt written by
``00_clean_dataset.py``) this script runs the FIXED candidate generator on the
sample's own local occupancy map and writes the per-split arrays consumed by
``src/datasets/carla_spline_dataset.py``:

    candidate_xy.npy               [N, M, 128, 2] f32   S_m network input
    candidate_mask.npy             [N, M]         bool
    candidate_lengths.npy          [N, M]         f32
    candidate_geometry.npy         [P, 2]         i16   dense safe cell chain
    candidate_geometry_offsets.npy [N, M+1]       i64
    candidate_geometry_lengths.npy [N, M]         i32
    topology_best.npy              [N]            i64   argmin_m nDTW(curve_gt, S_m)
    candidate_branch_counts.npy    [N, M]         i32   diagnostics

Per-sample results are cached in ``<split>/_cache/cand_<i:06d>.npz`` so the run
is resumable and a single bad sample can never abort the whole preprocessing.

    python scripts/data/carla/01_build_candidates.py --processed data/carla_processed
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import (CandidateConfig, generate_candidates,
                                         normalized_dtw)

SPLITS = ["train", "val", "test"]
_W = {}


def _cache_dir(processed, split):
    return os.path.join(processed, split, "_cache")


def _cache_path(processed, split, i):
    return os.path.join(_cache_dir(processed, split), "cand_%06d.npz" % int(i))


def _init_worker(processed, split, cand_cfg, skel_cfg):
    _W["processed"] = processed
    _W["split"] = split
    d = os.path.join(processed, split)
    _W["occ"] = np.load(os.path.join(d, "occupancy.npy"), mmap_mode="r")
    _W["cond"] = np.load(os.path.join(d, "conditions.npy"), mmap_mode="r")
    _W["curve"] = np.load(os.path.join(d, "curve_gt.npy"), mmap_mode="r")
    _W["cand_cfg"] = CandidateConfig.from_dict(cand_cfg, strict=False)
    _W["skel_cfg"] = dict(skel_cfg or {})
    os.makedirs(_cache_dir(processed, split), exist_ok=True)


def _empty_payload(i, error):
    cfg = _W["cand_cfg"]
    M, L = int(cfg.num_candidates), int(cfg.candidate_points)
    return dict(idx=i, candidate_xy=np.zeros((M, L, 2), np.float32),
                mask=np.zeros(M, bool), lengths=np.zeros(M, np.float32),
                geom=np.zeros((0, 2), np.int16),
                geom_offsets=np.zeros(M + 1, np.int64),
                geom_lengths=np.zeros(M, np.int32), best=0, n_nodes=0,
                n_branches=0, n_valid=0, error=str(error))


def _work(i):
    import numpy as np

    cfg = _W["cand_cfg"]
    skel = _W["skel_cfg"]
    M, L = int(cfg.num_candidates), int(cfg.candidate_points)
    try:
        occ = np.asarray(_W["occ"][i])
        cond = np.asarray(_W["cond"][i], dtype=np.float64).reshape(2, 2)
        curve = np.asarray(_W["curve"][i], dtype=np.float64)
        graph = build_skeleton_graph(
            occ,
            safety_dilation_cells=int(skel.get("safety_dilation_cells", 1)),
            thinning_backend=str(skel.get("thinning_backend", "auto")),
            pure_cycle_aux_nodes=int(skel.get("pure_cycle_aux_nodes", 2)))
        cands = generate_candidates(graph, cond[0], cond[1], cfg)

        xy = np.zeros((M, L, 2), np.float32)
        mask = np.zeros(M, bool)
        lengths = np.zeros(M, np.float32)
        glen = np.zeros(M, np.int32)
        offsets = np.zeros(M + 1, np.int64)
        flat = []
        cursor = 0
        for m in range(min(M, cands.num_slots)):
            if bool(cands.mask[m]):
                xy[m] = np.asarray(cands.paths[m][:, :2], dtype=np.float32)
                mask[m] = True
                lengths[m] = float(cands.lengths[m])
                px = np.rint(graph.scene_to_pixel(
                    np.asarray(cands.geometry[m], dtype=np.float64))).astype(np.int16)
                flat.append(px.reshape(-1, 2))
                glen[m] = len(px)
                cursor += len(px)
            offsets[m + 1] = cursor
        if flat:
            geom = np.concatenate(flat, axis=0)
        else:
            geom = np.zeros((0, 2), np.int16)
        best = 0
        valid = cands.valid_index() if cands.num_valid else np.zeros(0, int)
        if len(valid):
            d = [normalized_dtw(curve, cands.metric_polyline(int(m))) for m in valid]
            best = int(valid[int(np.argmin(d))])
        payload = dict(idx=i, candidate_xy=xy, mask=mask, lengths=lengths,
                       geom=geom, geom_offsets=offsets, geom_lengths=glen,
                       best=best, n_nodes=len(graph.nodes),
                       n_branches=len(graph.branches), n_valid=int(cands.num_valid),
                       error="")
        np.savez_compressed(_cache_path(_W["processed"], _W["split"], i), **payload)
        return {"idx": i, "ok": True, "n_valid": int(cands.num_valid),
                "n_nodes": len(graph.nodes), "n_branches": len(graph.branches),
                "best_ndtw": float(min(d)) if len(valid) else float("nan")}
    except Exception as exc:                                   # noqa: BLE001
        payload = _empty_payload(i, repr(exc))
        try:
            np.savez_compressed(_cache_path(_W["processed"], _W["split"], i),
                                **payload)
        except Exception:
            pass
        return {"idx": i, "ok": False, "error": repr(exc)}


def _load_cache(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def assemble(processed, split, n, M, L, G, indices=None):
    d = os.path.join(processed, split)
    xy = np.zeros((n, M, L, 2), np.float32)
    mask = np.zeros((n, M), bool)
    lengths = np.zeros((n, M), np.float32)
    glen = np.zeros((n, M), np.int32)
    offsets = np.zeros((n, M + 1), np.int64)
    best = np.zeros(n, np.int64)
    branch = np.zeros((n, M), np.int32)
    flat = []
    cursor = 0
    missing = []
    idxs = range(n) if indices is None else indices
    for i in idxs:
        p = _cache_path(processed, split, i)
        offsets[i, 0] = cursor
        if not os.path.exists(p):
            missing.append(int(i))
            for m in range(M):
                offsets[i, m + 1] = cursor
            continue
        c = _load_cache(p)
        m_mask = np.asarray(c["mask"]).astype(bool)
        mask[i] = m_mask[:M]
        lengths[i] = np.asarray(c["lengths"], np.float32)[:M]
        xy[i] = np.asarray(c["candidate_xy"], np.float32)[:M]
        glen[i] = np.asarray(c["geom_lengths"], np.int32)[:M]
        best[i] = int(np.asarray(c["best"]).reshape(-1)[0])
        geom = np.asarray(c["geom"], np.int16).reshape(-1, 2)
        # per-candidate slices: glen[i][m] points of candidate m, in slot order
        off = np.zeros(M + 1, np.int64)
        cur = cursor
        for m in range(M):
            off[m] = cur
            cur += int(glen[i, m])
        off[M] = cur
        offsets[i] = off
        if int(glen[i].sum()) != len(geom):
            missing.append(int(i))
            glen[i] = 0
            off[:] = cursor
            offsets[i] = off
            continue
        if len(geom):
            flat.append(geom)
            cursor = cur
    geom_all = (np.concatenate(flat, axis=0) if flat
                else np.zeros((0, 2), np.int16))
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "candidate_xy.npy"), xy)
    np.save(os.path.join(d, "candidate_mask.npy"), mask)
    np.save(os.path.join(d, "candidate_lengths.npy"), lengths)
    np.save(os.path.join(d, "candidate_geometry.npy"), geom_all)
    np.save(os.path.join(d, "candidate_geometry_offsets.npy"), offsets)
    np.save(os.path.join(d, "candidate_geometry_lengths.npy"), glen)
    np.save(os.path.join(d, "topology_best.npy"), best)
    np.save(os.path.join(d, "candidate_branch_counts.npy"), branch)
    return {"mask": mask, "glen": glen, "missing": missing,
            "geometry_points": int(geom_all.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed = args.processed or cfg["data"].get("processed_root",
                                                  "data/carla_processed")
    processed = os.path.abspath(processed)
    cand_cfg = cfg.get("topology") or {}
    skel_cfg = cfg.get("skeleton") or {}
    M = int(cand_cfg.get("num_candidates", 4))
    L = int(cand_cfg.get("candidate_points", 128))
    G = int(cand_cfg.get("candidate_geometry_points", 1280))
    report = {"processed": processed, "config": args.config,
              "candidate_config": dict(cand_cfg), "per_split": {}}
    t_all = time.time()

    for split in args.splits:
        d = os.path.join(processed, split)
        cond = np.load(os.path.join(d, "conditions.npy"))
        n_full = int(len(cond))
        n = n_full if args.limit is None else min(int(args.limit), n_full)
        if args.force and os.path.isdir(_cache_dir(processed, split)):
            shutil.rmtree(_cache_dir(processed, split))
        os.makedirs(_cache_dir(processed, split), exist_ok=True)
        todo = [i for i in range(n)
                if not (args.resume and os.path.exists(
                    _cache_path(processed, split, i)))]
        print("[%s] n=%d cached=%d todo=%d" % (split, n, n - len(todo),
                                               len(todo)), flush=True)
        t0 = time.time()
        results = []
        if todo:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=max(1, int(args.workers)),
                          initializer=_init_worker,
                          initargs=(processed, split, cand_cfg, skel_cfg)) as pool:
                for k, r in enumerate(pool.imap_unordered(_work, todo,
                                                          chunksize=4)):
                    results.append(r)
                    if (k + 1) % 200 == 0:
                        el = time.time() - t0
                        print("   %d/%d %.0fs (%.0f ms/sample)"
                              % (k + 1, len(todo), el,
                                 el / (k + 1) * 1000.0), flush=True)
        # always assemble the FULL split so a partial (resumable) run still
        # produces arrays the dataset can load
        asm = assemble(processed, split, n_full, M, L, G)
        n_valid = asm["mask"].sum(axis=1)
        empty = int((n_valid == 0).sum())
        failed = [r for r in results if not r.get("ok")]
        stats = {
            "n": int(n_full),
            "processed": int(n),
            "empty_candidate_count": empty,
            "empty_rate": float(empty / max(n, 1)),
            "mean_valid_candidates": float(n_valid.mean()),
            "valid_hist": [int((n_valid == k).sum()) for k in range(M + 1)],
            "geometry_points": asm["geometry_points"],
            "geometry_points_max": int(asm["glen"].max()) if n else 0,
            "missing_cache": len(asm["missing"]),
            "failed_samples": [r["idx"] for r in failed][:200],
            "failure_reasons": {str(r.get("error"))[:120]: 1 for r in failed},
            "seconds": float(time.time() - t0),
        }
        report["per_split"][split] = stats
        print("[%s] done n=%d empty=%d (%.2f%%) mean_valid=%.2f %.0fs"
              % (split, n, empty, 100.0 * empty / max(n, 1),
                 stats["mean_valid_candidates"], stats["seconds"]), flush=True)

    report["empty_candidate_count"] = int(sum(
        v["empty_candidate_count"] for v in report["per_split"].values()))
    report["total_samples"] = int(sum(v["n"] for v in report["per_split"].values()))
    report["empty_rate"] = float(report["empty_candidate_count"]
                                 / max(report["total_samples"], 1))
    report["seconds"] = float(time.time() - t_all)
    with open(os.path.join(processed, "preprocess_candidates_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DONE", processed, "empty_rate=%.4f" % report["empty_rate"], flush=True)


if __name__ == "__main__":
    main()
