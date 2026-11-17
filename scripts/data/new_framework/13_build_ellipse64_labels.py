#!/usr/bin/env python
"""13_build_ellipse64_labels.py - 64-query ellipse shape labels along the PDG path.

The new framework queries 64 ellipses whose centres are sampled by NORMALISED ARC
LENGTH along the active path (``spd_i = i / (Q-1)``, plan §5.1), where the path is
``pdg_path_points.npy`` from ``12_build_pdg_paths.py``.  The SHAPE criterion is
exactly the one of ``scripts/data/carla/02_build_ellipse_labels.py``: that module
is imported and its functions are used verbatim (no copy-paste):

    safe   = 02.occupancy_safe(occ, dilation_cells)      # obstacles + k cells
    shape  = 02.shape_at_center(safe, graph, params, centre)
             (36 orientations, local-radius clearance cap, DDA ray clearance,
              boundary+interior safety check, binary search on the homothety,
              largest area wins)

Only the CENTRES differ from the legacy labels (path arc length instead of the
selected Skeleton), which is why this is a new array instead of a re-run of 02.

Output (new root, per split) -- REPLACES the legacy 128-query labels:
    ellipse_shape4_gt.npy  [N,64,4] float32  [log a, log b, cos 2t, sin 2t]
    shape_valid.npy        [N,64]   bool

Resumable: per-sample cache ``<split>/_cache/ell64_%06d.npz``.

    python scripts/data/new_framework/13_build_ellipse64_labels.py --limit 8
    python scripts/data/new_framework/13_build_ellipse64_labels.py --splits test
    python scripts/data/new_framework/13_build_ellipse64_labels.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.utils.config import load_config  # noqa: E402

DEFAULT_ROOT = "data/processed/160m-k4p-c32q64"
DEFAULT_CONFIG = "configs/config_160k4p_c48_oneshot.yaml"
ELLIPSE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "data", "carla",
                              "02_build_ellipse_labels.py")
SPLITS = ("train", "val", "test")
NUM_QUERIES = 64
NOTE = ("Q=64 ellipse centres are uniform in NORMALISED ARC LENGTH along the "
        "PDG-Focal A* path (pdg_path_points.npy); the shape criterion is "
        "imported verbatim from scripts/data/carla/02_build_ellipse_labels.py. "
        "The occupancy is the k4p map whose obstacles were ERODED by 4 cells, so "
        "the labelled ellipses are optimistic w.r.t. the original obstacle "
        "boundary (see NEW_LABELS_REPORT.md).")
W = {}


def _resolve(path_like) -> str:
    p = os.path.expanduser(str(path_like))
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def load_ellipse_module():
    """Import 02_build_ellipse_labels.py (digit-leading filename -> spec load)."""
    spec = importlib.util.spec_from_file_location("ellipse_labels_02",
                                                  ELLIPSE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ellipse_params(cfg, num_orientations=None):
    """Same parameter block as ``02_build_ellipse_labels.main()``."""
    lab = cfg.get("ellipse_label") or {}
    n_orient = int(num_orientations or lab.get("num_orientations", 36))
    return {
        "thetas": np.arange(n_orient) * np.pi / float(n_orient),
        "local_radius": float(lab.get("local_radius", 0.25)),
        "dilation": int(lab.get("dilation", 1)),
        "boundary": int(lab.get("boundary_points", 48)),
        "interior_rings": int(lab.get("interior_rings", 2)),
        "interior_angles": int(lab.get("interior_angles", 12)),
        "binary_iters": int(lab.get("binary_iters", 8)),
        "min_semi_axis": float(lab.get("min_semi_axis", 1e-3)),
    }, n_orient


def centres_from_path(path: np.ndarray, q: int) -> np.ndarray:
    """``q`` centres uniform in normalised arc length (``spd_i = i/(q-1)``)."""
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 1:
        return np.repeat(pts, int(q), axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-12:
        return np.repeat(pts[:1], int(q), axis=0)
    s = np.linspace(0.0, cum[-1], int(q))
    return np.stack([np.interp(s, cum, pts[:, k]) for k in range(2)], axis=-1)


def _cache_path(processed, split, i):
    return os.path.join(processed, split, "_cache", "ell64_%06d.npz" % int(i))


def _init_worker(processed, split, params, q, skel_cfg):
    d = os.path.join(processed, split)
    W["processed"], W["split"] = processed, split
    W["occ"] = np.load(os.path.join(d, "occupancy.npy"), mmap_mode="r")
    W["path"] = np.load(os.path.join(d, "pdg_path_points.npy"), mmap_mode="r")
    W["path_valid"] = np.load(os.path.join(d, "pdg_path_valid.npy"))
    W["cond"] = np.load(os.path.join(d, "conditions.npy"), mmap_mode="r")
    W["params"] = params
    W["q"] = int(q)
    W["mod"] = load_ellipse_module()
    W["skel"] = dict(skel_cfg or {})
    os.makedirs(os.path.join(d, "_cache"), exist_ok=True)


def _work(i):
    mod = W["mod"]
    params, q = W["params"], W["q"]
    s4 = np.zeros((q, 4), np.float32)
    ok = np.zeros(q, bool)
    info = {"idx": int(i), "ok": True, "valid": 0, "path_valid": False}
    try:
        occ = np.asarray(W["occ"][i])
        path = np.asarray(W["path"][i], np.float64)
        info["path_valid"] = bool(W["path_valid"][i])
        cond = np.asarray(W["cond"][i], np.float64).reshape(2, 2)
        if not np.isfinite(path).all():
            raise ValueError("non-finite pdg path")
        path = path.copy()
        path[0], path[-1] = cond[0], cond[1]
        centres = centres_from_path(path, q)
        graph = mod.build_skeleton_graph(
            occ,
            safety_dilation_cells=int(W["skel"].get("safety_dilation_cells", 1)),
            thinning_backend=str(W["skel"].get("thinning_backend", "auto")),
            pure_cycle_aux_nodes=int(W["skel"].get("pure_cycle_aux_nodes", 2)))
        safe = mod.occupancy_safe(occ, int(params["dilation"]))
        for k in range(q):
            shape = mod.shape_at_center(safe, graph, params, centres[k])
            if shape is not None:
                s4[k] = shape
                ok[k] = True
                info["valid"] += 1
    except Exception as exc:                                     # noqa: BLE001
        info["ok"] = False
        info["error"] = repr(exc)
    try:
        np.savez_compressed(_cache_path(W["processed"], W["split"], i),
                            idx=i, shape4=s4, valid=ok,
                            path_valid=info["path_valid"])
    except Exception:                                            # noqa: BLE001
        pass
    return info


def _run_todo(root, split, params, q, skel_cfg, todo, workers, t0):
    """Process ``todo`` in a spawn pool, or sequentially if pipes are blocked.

    ``multiprocessing`` on Windows needs named pipes; a confined environment can
    deny them (``PermissionError: [WinError 5]``).  In that case this falls back
    to an in-process loop, and ``--shard K/N`` gives file-based parallelism
    instead (each shard is an independent process writing its own caches).
    """
    results = []
    if int(workers) > 1:
        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(workers), initializer=_init_worker,
                          initargs=(root, split, params, q, skel_cfg)) as pool:
                for k, r in enumerate(pool.imap_unordered(_work, todo,
                                                          chunksize=2)):
                    results.append(r)
                    if (k + 1) % 100 == 0:
                        el = time.time() - t0
                        print("   [%s] %d/%d %.0fs (%.0f ms/sample)"
                              % (split, k + 1, len(todo), el,
                                 el / (k + 1) * 1000.0), flush=True)
            return results
        except (PermissionError, OSError) as exc:
            print("[ell64] WARNING: multiprocessing unavailable (%s)\n"
                  "         -> sequential in-process loop; use --shard K/N to "
                  "parallelise with independent processes" % exc, flush=True)
    _init_worker(root, split, params, q, skel_cfg)
    for k, i in enumerate(todo):
        results.append(_work(i))
        if (k + 1) % 25 == 0:
            el = time.time() - t0
            print("   [%s] %d/%d %.0fs (%.0f ms/sample)"
                  % (split, k + 1, len(todo), el, el / (k + 1) * 1000.0),
                  flush=True)
    return results


def assemble(processed, split, n, q, limit=None):
    """Write [n,Q,4]/[n,Q]; ``--limit`` goes to a sidecar, never over the real file."""
    d = os.path.join(processed, split)
    out_dir = d
    if limit is not None:
        out_dir = os.path.join(d, "_selfcheck_partial")
        os.makedirs(out_dir, exist_ok=True)
    s4 = np.zeros((n, q, 4), np.float32)
    valid = np.zeros((n, q), bool)
    missing = []
    for i in range(n):
        p = _cache_path(processed, split, i)
        if not os.path.exists(p):
            missing.append(i)
            continue
        with np.load(p, allow_pickle=False) as z:
            s4[i] = np.asarray(z["shape4"], np.float32)
            valid[i] = np.asarray(z["valid"]).astype(bool)
    if limit is None and (s4.shape[0] != n or missing):
        raise SystemExit("[error] %s: refusing to write the official arrays "
                         "(%d rows, %d samples without a cache); compute every "
                         "sample first" % (split, s4.shape[0], len(missing)))
    np.save(os.path.join(out_dir, "ellipse_shape4_gt.npy"), s4)
    np.save(os.path.join(out_dir, "shape_valid.npy"), valid)
    return s4, valid, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--splits", nargs="*", default=["test", "val", "train"])
    ap.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--num-orientations", type=int, default=None)
    ap.add_argument("--shard", nargs=2, type=int, default=None,
                    metavar=("K", "N"),
                    help="compute only indices with i %% N == K (file-based "
                         "parallelism); caches only, no assembly")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip the per-sample work, just assemble + report")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scene_to_meter = float((cfg.get("data") or {}).get("scene_to_meter", 80.0))
    root = _resolve(args.root)
    q = int(args.num_queries)
    params, n_orient = ellipse_params(cfg, args.num_orientations)
    skel_cfg = cfg.get("skeleton") or {}
    report = {"root": root, "config": args.config, "num_queries": q,
              "num_orientations": n_orient,
              "params": {k: v for k, v in params.items() if k != "thetas"},
              "scene_to_meter": scene_to_meter, "note": NOTE,
              "ellipse_criterion_source": ELLIPSE_SCRIPT,
              "limit": args.limit, "per_split": {}}
    # keep the stats of splits finished by an earlier invocation (sharded runs
    # assemble one split at a time)
    prev_report = os.path.join(root, "ellipse64_report.json")
    if os.path.exists(prev_report) and not args.limit:
        try:
            with open(prev_report, encoding="utf-8") as fh:
                prev = json.load(fh)
            report["per_split"].update(prev.get("per_split") or {})
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    print("[ell64] root=%s Q=%d orientations=%d local_radius=%.3f dilation=%d "
          "workers=%d limit=%s"
          % (root, q, n_orient, params["local_radius"], params["dilation"],
             args.workers, args.limit), flush=True)

    t_all = time.time()
    for split in args.splits:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            print("[ell64] %-5s MISSING" % split)
            continue
        n_full = int(len(np.load(os.path.join(d, "conditions.npy"),
                                 mmap_mode="r")))
        pathv = np.load(os.path.join(d, "pdg_path_valid.npy"))
        if pathv.shape[0] != n_full:
            if args.limit is None:
                print("[ell64] %-5s pdg_path_valid has %d rows, expected %d "
                      "(run 12 first)" % (split, pathv.shape[0], n_full))
                continue
            print("[ell64] %-5s WARNING: pdg arrays are a %d-row partial "
                  "(--limit self-check) -> processing those rows only"
                  % (split, pathv.shape[0]), flush=True)
            n_full = int(pathv.shape[0])
        n = n_full if args.limit is None else min(int(args.limit), n_full)
        cache = os.path.join(d, "_cache")
        os.makedirs(cache, exist_ok=True)
        if args.force:
            for f in os.listdir(cache):
                if f.startswith("ell64_"):
                    os.remove(os.path.join(cache, f))
        todo = [i for i in range(n)
                if not os.path.exists(_cache_path(root, split, i))]
        if args.shard is not None:
            k_shard, n_shard = int(args.shard[0]), int(args.shard[1])
            todo = [i for i in todo if i % n_shard == k_shard]
        print("[ell64] %-5s n=%d cached=%d todo=%d%s"
              % (split, n, n - len(todo), len(todo),
                 "" if args.shard is None else "  (shard %s)" % (args.shard,)),
              flush=True)
        t0 = time.time()
        results = []
        if todo and not args.assemble_only:
            results = _run_todo(root, split, params, q, skel_cfg, todo,
                                args.workers, t0)
        if args.shard is not None and not args.assemble_only:
            print("[ell64] %-5s shard %s wrote %d caches in %.0fs"
                  % (split, args.shard, len(todo), time.time() - t0), flush=True)
            continue
        s4, valid, missing = assemble(root, split, n_full, q, args.limit)
        if args.limit is not None:
            print("[ell64] %-5s --limit %d: official arrays NOT touched; wrote "
                  "self-check copies to %s/_selfcheck_partial"
                  % (split, args.limit, split), flush=True)
        axes = np.exp(s4[:n][valid[:n]][:, :2]) * scene_to_meter \
            if valid[:n].any() else None
        failed = [r for r in results if not r.get("ok")]
        pv = pathv[:n].astype(bool)
        stats = {
            "n": int(n_full), "processed": int(n),
            "valid_entries": int(valid[:n].sum()),
            "valid_fraction": float(valid[:n].sum() / max(n * q, 1)),
            "samples_with_any_valid": int(valid[:n].any(axis=1).sum()),
            "samples_with_no_valid": int((~valid[:n].any(axis=1)).sum()),
            "valid_frac_of_pdg_valid": (
                float(valid[:n][pv].sum() / max(int(pv.sum()) * q, 1))),
            "semi_axis_m_mean": (None if axes is None
                                 else [float(axes[:, 0].mean()),
                                       float(axes[:, 1].mean())]),
            "semi_axis_m_max": (None if axes is None
                                else [float(axes[:, 0].max()),
                                      float(axes[:, 1].max())]),
            "log_axis_range": (None if not valid.any() else
                               [[float(s4[..., 0].min()), float(s4[..., 0].max())],
                                [float(s4[..., 1].min()), float(s4[..., 1].max())]]),
            "missing_cache": len(missing),
            "failed_samples": [r["idx"] for r in failed][:50],
            "failure_reasons": {str(r.get("error"))[:120]: 1 for r in failed},
            "seconds": float(time.time() - t0),
        }
        report["per_split"][split if args.limit is None
                            else "%s__limit%d" % (split, args.limit)] = stats
        print("[ell64] %-5s done n=%d valid_frac=%.3f (of pdg-valid %.3f) "
              "no_valid_samples=%d semi_axes_m=%s %.0fs"
              % (split, n_full, stats["valid_fraction"],
                 stats["valid_frac_of_pdg_valid"], stats["samples_with_no_valid"],
                 "n/a" if axes is None else "[%.2f, %.2f]"
                 % (axes[:, 0].mean(), axes[:, 1].mean()),
                 stats["seconds"]), flush=True)
        with open(os.path.join(root, "ellipse64_report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=float)

    report["seconds"] = float(time.time() - t_all)
    with open(os.path.join(root, "ellipse64_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    print("\nDONE ellipse64 -> %s  (report: ellipse64_report.json)" % root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
