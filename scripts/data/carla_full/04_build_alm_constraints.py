#!/usr/bin/env python
"""04_build_alm_constraints.py - offline ALM/corridor constraint pack.

The inference-time ALM (src/diffusion/bspline_alm.py + sampler.py) projects the
predicted curve into a SAFETY CORRIDOR: 128 convex cells, one per fixed
Skeleton progress point, built from the ellipse head output.  That machinery is
numpy/torch glue and costs ~240 ms per sample, so it can never run inside the
training loop.

This script therefore PRE-COMPUTES, per sample, the corridor built from the GT
route (m* = topology_best, plus the GT ellipse labels) and stores it as a
half-space pack.  A point p is inside cell i iff A_i @ p <= b_i for every face.
With

    violation(p) = min_i max_f (A_if . p - b_if)

the training loss L_alm = mean(relu(violation(decoded_curve))) is exact (no
rasterisation error) and differentiable w.r.t. the control polygon.

Outputs per split:
    alm_cell_a.npy     [N, C, F, 2] f32   padded face normals (0 where padded)
    alm_cell_b.npy     [N, C, F]    f32   padded face offsets (+inf where padded)
    alm_cell_valid.npy [N, C]       bool  that cell produced a convex polyhedron
    alm_valid.npy      [N]          bool  the whole corridor closed
    alm_anchor_s.npy   [N, C]       f32   progress of each cell (diagnostics)

F is the global max face count per cell, measured over the dataset.  Invalid
corridors are stored with alm_valid = False so the loss can mask them out.

    python scripts/data/carla_full/04_build_alm_constraints.py --workers 12
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.geometry.safety_corridor import build_safety_corridor  # noqa: E402
from src.geometry.skeleton_paths import interpolate_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402

SPLITS = ["train", "val", "test"]
_W = {}


def _cache_dir(processed, split):
    return os.path.join(processed, split, "_cache")


def _cache_path(processed, split, i):
    return os.path.join(_cache_dir(processed, split), "alm_%06d.npz" % int(i))


def _init_worker(processed, split, cfg, bcfg):
    _W["processed"] = processed
    _W["split"] = split
    d = os.path.join(processed, split)
    _W["occ"] = np.load(os.path.join(d, "occupancy.npy"), mmap_mode="r")
    _W["cond"] = np.load(os.path.join(d, "conditions.npy"), mmap_mode="r")
    _W["geom"] = np.load(os.path.join(d, "candidate_geometry.npy"), mmap_mode="r")
    _W["offsets"] = np.load(os.path.join(d, "candidate_geometry_offsets.npy"))
    _W["best"] = np.load(os.path.join(d, "topology_best.npy"))
    _W["mask"] = np.load(os.path.join(d, "candidate_mask.npy"))
    _W["shape4"] = np.load(os.path.join(d, "ellipse_shape4_gt.npy"), mmap_mode="r")
    _W["cfg"] = cfg
    _W["bcfg"] = bcfg
    os.makedirs(_cache_dir(processed, split), exist_ok=True)


def _empty(horizon, reason):
    return dict(a=np.zeros((0, 2), np.float32), b=np.zeros((0,), np.float32),
                cell_offsets=np.zeros(horizon + 1, np.int64),
                cell_valid=np.zeros(horizon, bool),
                anchor_s=np.linspace(0.0, 1.0, horizon).astype(np.float32),
                valid=False, reason=reason)


def _work(i):
    import numpy as np  # noqa: F811
    import torch

    horizon = int(_W["cfg"]["horizon"])
    try:
        # np.array (not asarray): torch warns on non-writable memmap inputs
        occ = np.array(_W["occ"][i])
        cond = np.asarray(_W["cond"][i], np.float64).reshape(2, 2)
        best = int(_W["best"][i])
        mask = np.asarray(_W["mask"][i]).astype(bool)
        if not (0 <= best < len(mask) and mask[best]):
            payload = _empty(horizon, "no_valid_candidate")
            payload["idx"] = i
            np.savez_compressed(_cache_path(_W["processed"], _W["split"], i), **payload)
            return {"idx": i, "ok": True, "valid": False, "reason": "no_valid_candidate"}

        lo, hi = int(_W["offsets"][i, best]), int(_W["offsets"][i, best + 1])
        px = np.asarray(_W["geom"][lo:hi], np.float64)
        cell = 2.0 / 256.0
        poly = (px + 0.5) * cell - 1.0
        if len(poly) >= 2:
            poly = poly.copy()
            poly[0] = cond[0]
            poly[-1] = cond[1]
        centers = interpolate_path(poly, np.linspace(0.0, 1.0, horizon))
        shape4 = np.asarray(_W["shape4"][i], np.float32)

        from src.diffusion.sampler import _region_builder
        builder = _region_builder(
            torch.as_tensor(occ[None, None], dtype=torch.float32), dict(_W["bcfg"]))
        corridor = build_safety_corridor(
            builder,
            torch.as_tensor(centers, dtype=torch.float32),
            torch.as_tensor(shape4, dtype=torch.float32),
            torch.linspace(0.0, 1.0, horizon),
            gamma=torch.as_tensor(poly, dtype=torch.float32),
            gamma_lengths=max(len(poly), 1),
            config=dict(_W["cfg"]["corridor"]))

        if not corridor.valid:
            reason = str(corridor.failure_reason or "corridor_invalid")
            payload = _empty(horizon, reason)
            payload["idx"] = i
            np.savez_compressed(_cache_path(_W["processed"], _W["split"], i), **payload)
            return {"idx": i, "ok": True, "valid": False, "reason": reason}

        A_list, b_list, cell_valid, anchor_s = [], [], [], []
        offs = np.zeros(horizon + 1, np.int64)
        cur = 0
        for k, c in enumerate(corridor.cells):
            A_k = np.asarray(c.A, np.float32).reshape(-1, 2)
            b_k = np.asarray(c.b, np.float32).reshape(-1)
            A_list.append(A_k)
            b_list.append(b_k)
            cell_valid.append(bool(c.valid))
            anchor_s.append(float(c.anchor_s))
            cur += len(A_k)
            offs[k + 1] = cur
        payload = dict(
            idx=i,
            a=(np.concatenate(A_list, 0) if A_list else np.zeros((0, 2), np.float32)),
            b=(np.concatenate(b_list, 0) if b_list else np.zeros((0,), np.float32)),
            cell_offsets=offs, cell_valid=np.asarray(cell_valid, bool),
            anchor_s=np.asarray(anchor_s, np.float32), valid=True, reason="")
        np.savez_compressed(_cache_path(_W["processed"], _W["split"], i), **payload)
        return {"idx": i, "ok": True, "valid": True,
                "cells": len(corridor.cells), "faces": int(cur)}
    except Exception as exc:                                     # noqa: BLE001
        reason = "exception:%s" % repr(exc)[:120]
        payload = _empty(horizon, reason)
        payload["idx"] = i
        try:
            np.savez_compressed(_cache_path(_W["processed"], _W["split"], i), **payload)
        except Exception:
            pass
        return {"idx": i, "ok": False, "valid": False, "reason": reason}


def assemble(processed, split, n, cells_per_sample, fmax):
    d = os.path.join(processed, split)
    A = np.zeros((n, cells_per_sample, fmax, 2), np.float32)
    B = np.full((n, cells_per_sample, fmax), np.inf, np.float32)
    cv = np.zeros((n, cells_per_sample), bool)
    valid = np.zeros(n, bool)
    anchors = np.linspace(0.0, 1.0, cells_per_sample).astype(np.float32)
    missing = 0
    for i in range(n):
        p = _cache_path(processed, split, i)
        if not os.path.exists(p):
            missing += 1
            continue
        with np.load(p, allow_pickle=False) as z:
            v = bool(np.asarray(z["valid"]).reshape(-1)[0])
            valid[i] = v
            if not v:
                continue
            a = np.asarray(z["a"], np.float32).reshape(-1, 2)
            b = np.asarray(z["b"], np.float32).reshape(-1)
            offs = np.asarray(z["cell_offsets"], np.int64).reshape(-1)
            cellv = np.asarray(z["cell_valid"]).astype(bool).reshape(-1)
            k = min(len(cellv), cells_per_sample)
            cv[i, :k] = cellv[:k]
            for c in range(k):
                lo, hi = int(offs[c]), int(offs[c + 1])
                m = min(hi - lo, fmax)
                if m <= 0:
                    continue
                A[i, c, :m] = a[lo:lo + m]
                B[i, c, :m] = b[lo:lo + m]
    np.save(os.path.join(d, "alm_cell_a.npy"), A)
    np.save(os.path.join(d, "alm_cell_b.npy"), B)
    np.save(os.path.join(d, "alm_cell_valid.npy"), cv)
    np.save(os.path.join(d, "alm_valid.npy"), valid)
    np.save(os.path.join(d, "alm_anchor_s.npy"),
            np.broadcast_to(anchors[None], (n, cells_per_sample)).copy())
    return {"missing": missing, "valid": int(valid.sum()),
            "bytes": int(A.nbytes + B.nbytes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=None)
    ap.add_argument("--config", default="configs/config_160.yaml")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed = os.path.abspath(args.processed or cfg["data"]["processed_root"])
    horizon = int(cfg["model"]["num_safety_queries"])
    alm_cfg = dict(cfg.get("alm") or {})
    corridor_cfg = dict(cfg.get("corridor") or {})
    from src.diffusion.sampler import _DEFAULT_REGION_KEYS
    bcfg = {k: alm_cfg[k] for k in _DEFAULT_REGION_KEYS if k in alm_cfg}
    bcfg.update(corridor_cfg.get("region") or {})
    wcfg = {"horizon": horizon, "corridor": corridor_cfg, "alm": alm_cfg}

    print("[alm] processed=%s horizon=%d workers=%d" % (processed, horizon, args.workers))
    report = {"processed": processed, "horizon": horizon,
              "corridor_cfg": corridor_cfg, "per_split": {}}
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
                if not os.path.exists(_cache_path(processed, split, i))]
        print("[%s] n=%d cached=%d todo=%d"
              % (split, n, n - len(todo), len(todo)), flush=True)
        t0 = time.time()
        res = []
        if todo:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=max(1, int(args.workers)),
                          initializer=_init_worker,
                          initargs=(processed, split, wcfg, bcfg)) as pool:
                for k, r in enumerate(pool.imap_unordered(_work, todo, chunksize=2)):
                    res.append(r)
                    if (k + 1) % 500 == 0:
                        el = time.time() - t0
                        print("   %d/%d %.0fs (%.0f ms/sample)"
                              % (k + 1, len(todo), el, el / (k + 1) * 1000.0),
                              flush=True)
        fmax = 1
        for i in range(n):
            p = _cache_path(processed, split, i)
            if not os.path.exists(p):
                continue
            with np.load(p, allow_pickle=False) as z:
                if not bool(np.asarray(z["valid"]).reshape(-1)[0]):
                    continue
                offs = np.asarray(z["cell_offsets"], np.int64).reshape(-1)
                if len(offs) > 1:
                    fmax = max(fmax, int(np.diff(offs).max()))
        asm = assemble(processed, split, n_full, horizon, fmax)
        stats = {"n": int(n_full), "processed": int(n), "fmax": int(fmax),
                 "valid": int(asm["valid"]),
                 "valid_rate": float(asm["valid"] / max(n_full, 1)),
                 "bytes": int(asm["bytes"]), "seconds": float(time.time() - t0),
                 "reasons": {}}
        for r in res:
            if not r.get("valid"):
                key = str(r.get("reason") or "unknown").split(":")[0]
                stats["reasons"][key] = stats["reasons"].get(key, 0) + 1
        report["per_split"][split] = stats
        print("[%s] done valid=%d/%d (%.1f%%) Fmax=%d %.1f MB %.0fs"
              % (split, stats["valid"], n_full, 100 * stats["valid_rate"], fmax,
                 asm["bytes"] / 1e6, stats["seconds"]), flush=True)

    report["seconds"] = float(time.time() - t_all)
    with open(os.path.join(processed, "alm_constraints_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("DONE", processed)


if __name__ == "__main__":
    main()
