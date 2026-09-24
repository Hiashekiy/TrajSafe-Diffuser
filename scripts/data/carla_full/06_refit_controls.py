#!/usr/bin/env python
"""06_refit_controls.py - rebuild the OFFLINE control labels for a new C.

``control_gt.npy`` is the only array in a processed cache that depends on the
number of B-spline control points C:

    conditions.npy        [N,2,2]        occupancy-independent, C-independent
    curve_gt.npy          [N,H,2]        trajectory_128, C-independent
    occupancy.npy         [N,256,256]    C-independent
    candidate_* / ellipse_shape4_gt / shape_valid / topology_best
                                         derived from occupancy + conditions
    alm_cell_* / alm_anchor_s            one cell per model.num_safety_queries
                                         corridor station (Q), NOT per control
    control_gt.npy        [N,C,2]        <-- DEPENDS ON C

so changing ``bspline.num_controls`` does NOT require re-running the expensive
parts of the pipeline (candidates / ellipse labels / ALM constraints).  It only
requires this script, which re-fits the endpoint-constrained least-squares
projection of ``curve_gt`` onto C controls - exactly the fit
``00_build_processed.py`` performs (``--fit``), with the same codec and the same
uniform parameterisation the network decodes with:

    ctrl = numpy_fit_curve_to_controls(knots, C, degree, curve_gt, start, goal, H)

Everything else is copied byte-for-byte, so the new cache differs from the old
one in control_gt.npy ONLY (that is what makes a C-sweep a single-variable
experiment).

Verified equivalence: re-fitting the shipped C=32 cache from its own
``curve_gt.npy`` reproduces the stored ``control_gt.npy`` to max |dq| ~ 3e-8
(float32 storage noise).

    python scripts/data/carla_full/06_refit_controls.py \
        --source data/carla_processed_160k4p \
        --out    data/carla_processed_160k4p_c48 \
        --config configs/config_160k4p_c48_oneshot.yaml

then re-validate the contract:

    python scripts/data/carla/03_validate_processed.py \
        --processed data/carla_processed_160k4p_c48 \
        --config configs/config_160k4p_c48_oneshot.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.geometry.bspline import (  # noqa: E402
    BSplineCodec,
    default_knots,
    numpy_fit_curve_to_controls,
)
from src.utils.config import curve_points, load_config, num_controls  # noqa: E402

SPLITS: Tuple[str, ...] = ("train", "val", "test")

#: the ONLY array whose shape is a function of C
REFIT_FILES: Tuple[str, ...] = ("control_gt.npy",)
#: always copied verbatim (C-independent contract)
COPY_FILES: Tuple[str, ...] = (
    "conditions.npy", "curve_gt.npy", "occupancy.npy",
    "episode_id.npy", "sample_id.npy",
    "candidate_xy.npy", "candidate_mask.npy", "candidate_lengths.npy",
    "candidate_branch_counts.npy", "candidate_geometry.npy",
    "candidate_geometry_offsets.npy", "candidate_geometry_lengths.npy",
    "topology_best.npy", "ellipse_shape4_gt.npy", "shape_valid.npy",
    "alm_cell_a.npy", "alm_cell_b.npy", "alm_cell_valid.npy",
    "alm_valid.npy", "alm_anchor_s.npy",
)
#: provenance / report files worth carrying over (missing ones are skipped)
OPTIONAL_FILES: Tuple[str, ...] = (
    "erode_report.json", "alm_constraints_report.json",
    "preprocess_candidates_report.json", "preprocess_labels_report.json",
    "preprocess_report.json", "build_processed_report.json",
)
#: root-level files of the processed root (NOT per split).  The validator reads
#: clean_manifest.jsonl and rewrites preprocess_report.json here.
ROOT_FILES: Tuple[str, ...] = (
    "clean_manifest.jsonl", "erode_report.json",
    "alm_constraints_report.json", "preprocess_candidates_report.json",
    "preprocess_labels_report.json", "preprocess_report.json",
)
#: deliberately NOT copied: it records the C of the ORIGINAL build (and the fit
#: stats of that C), which would be a lie in a re-fitted root.
ROOT_FILES_SKIPPED: Tuple[str, ...] = ("build_processed_report.json",)
DECODE_CHUNK = 512
#: metres per scene unit for the reported errors; overwritten from
#: data.scene_to_meter (80.0 for the 160 m crop) in main()
DECODE_SCENE_TO_METER = 80.0


def _resolve(path_like) -> str:
    p = os.path.expanduser(str(path_like))
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _solve_knots(bs_cfg: dict, c: int, degree: int) -> np.ndarray:
    """Same resolution rule as 00_build_processed.py (``auto`` -> clamped uniform)."""
    raw = bs_cfg.get("knots", "auto")
    if raw is None or str(raw).strip().lower() in ("", "auto", "none", "null"):
        return default_knots(c, degree)
    path = _resolve(raw)
    if not os.path.exists(path):
        print("[knots] %s not found; using default_knots(%d,%d)"
              % (path, c, degree), flush=True)
        return default_knots(c, degree)
    knots = np.load(path).astype(np.float64).reshape(-1)
    if knots.shape[0] != c + degree + 1:
        print("[knots] %s has %d entries, expected %d; using default_knots"
              % (path, knots.shape[0], c + degree + 1), flush=True)
        return default_knots(c, degree)
    return knots


def _decode_rmse(codec: BSplineCodec, q: np.ndarray, curve: np.ndarray
                 ) -> Dict[str, float]:
    """Per-sample decode error of controls ``q`` against the dense curve."""
    import torch

    errs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, q.shape[0], DECODE_CHUNK):
            rec = codec.decode_controls(
                torch.as_tensor(q[i:i + DECODE_CHUNK], dtype=torch.float32))
            e = (rec.numpy() - curve[i:i + DECODE_CHUNK]).astype(np.float64)
            errs.append(np.linalg.norm(e, axis=-1))
    e = np.concatenate(errs, axis=0)                     # [N,H]
    rmse = np.sqrt((e ** 2).mean(axis=1))                # per-sample, scene units
    return dict(rmse_mean=float(rmse.mean()), rmse_max=float(rmse.max()),
                pt_max=float(e.max()))


def _copy_split(src_dir: str, dst_dir: str) -> Tuple[List[str], List[str]]:
    """Copy everything except the refit arrays; _cache (ALM resume cache) too."""
    os.makedirs(dst_dir, exist_ok=True)
    copied: List[str] = []
    missing: List[str] = []
    names = set(COPY_FILES) | set(OPTIONAL_FILES)
    names |= {f for f in os.listdir(src_dir) if f.endswith((".json", ".jsonl"))}
    for name in sorted(names - set(REFIT_FILES)):
        s = os.path.join(src_dir, name)
        if not os.path.isfile(s):
            missing.append(name)
            continue
        shutil.copy2(s, os.path.join(dst_dir, name))
        copied.append(name)
    src_cache = os.path.join(src_dir, "_cache")
    if os.path.isdir(src_cache):
        dst_cache = os.path.join(dst_dir, "_cache")
        os.makedirs(dst_cache, exist_ok=True)
        n = 0
        for name in os.listdir(src_cache):
            s = os.path.join(src_cache, name)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(dst_cache, name))
                n += 1
        copied.append("_cache/ (%d files)" % n)
    return copied, missing


def _copy_root(src: str, dst: str) -> Tuple[List[str], List[str]]:
    """Copy the root-level contract files (manifest + C-independent reports)."""
    copied: List[str] = []
    missing: List[str] = []
    for name in ROOT_FILES:
        s = os.path.join(src, name)
        if not os.path.isfile(s):
            missing.append(name)
            continue
        shutil.copy2(s, os.path.join(dst, name))
        copied.append(name)
    return copied, missing


def refit_split(src_dir: str, dst_dir: str, c: int, h: int, degree: int,
                knots: np.ndarray, c_src: int) -> Dict[str, object]:
    t0 = time.time()
    copied, missing = _copy_split(src_dir, dst_dir)

    curve = np.load(os.path.join(src_dir, "curve_gt.npy"))
    cond = np.load(os.path.join(src_dir, "conditions.npy"))
    ctrl_src = np.load(os.path.join(src_dir, "control_gt.npy"))
    n = int(curve.shape[0])
    if curve.shape[1] != h:
        raise SystemExit("[error] %s: curve_gt has H=%d, config says H=%d"
                         % (src_dir, curve.shape[1], h))
    if ctrl_src.shape[1] != c_src:
        raise SystemExit("[error] %s: control_gt has C=%d, expected C_src=%d"
                         % (src_dir, ctrl_src.shape[1], c_src))

    ctrl = np.zeros((n, c, 2), np.float32)
    start = cond[:, 0, :].astype(np.float64)
    goal = cond[:, 1, :].astype(np.float64)
    t_fit = time.time()
    for i in range(n):
        ctrl[i] = numpy_fit_curve_to_controls(
            knots, c, degree, curve[i].astype(np.float64), start[i], goal[i], h
        ).astype(np.float32)
    fit_s = time.time() - t_fit

    np.save(os.path.join(dst_dir, "control_gt.npy"), ctrl)

    codec_new = BSplineCodec(degree=degree, num_controls=c, curve_points=h,
                             knots=knots)
    m_new = _decode_rmse(codec_new, ctrl, curve)
    m_src = None
    if c_src != c:
        # the SOURCE controls were built with their own knot vector; with
        # knots="auto" in every shipped config that is default_knots(C_src)
        codec_src = BSplineCodec(degree=degree, num_controls=c_src,
                                 curve_points=h,
                                 knots=default_knots(c_src, degree))
        m_src = _decode_rmse(codec_src, ctrl_src, curve)

    # the endpoint constraint is structural: Q_0 = start, Q_{C-1} = goal
    end_err = float(max(np.abs(ctrl[:, 0, :] - start).max(),
                        np.abs(ctrl[:, -1, :] - goal).max()))
    sc = float(DECODE_SCENE_TO_METER)

    meta = dict(
        n=n, C=int(c), H=int(h), degree=int(degree),
        C_source=int(c_src),
        endpoint_err_scene=end_err,
        decode_rmse_scene=m_new["rmse_mean"], decode_rmse_m=m_new["rmse_mean"] * sc,
        decode_rmse_max_scene=m_new["rmse_max"],
        decode_point_max_m=m_new["pt_max"] * sc,
        source_decode_rmse_m=(None if m_src is None else m_src["rmse_mean"] * sc),
        source_decode_rmse_max_m=(None if m_src is None else m_src["rmse_max"] * sc),
        fit_seconds=fit_s, copied=len(copied), missing=sorted(missing),
        seconds=time.time() - t0,
    )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="existing processed root (provides curve_gt + all "
                         "C-independent arrays)")
    ap.add_argument("--out", required=True, help="new processed root to write")
    ap.add_argument("--config", required=True,
                    help="config that carries the TARGET bspline.num_controls")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into an existing --out directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = num_controls(cfg)
    h = curve_points(cfg)
    bs_cfg = cfg.get("bspline") or {}
    degree = int(bs_cfg.get("degree", 3))
    knots = _solve_knots(bs_cfg, c, degree)
    global DECODE_SCENE_TO_METER
    DECODE_SCENE_TO_METER = float((cfg.get("data") or {}).get("scene_to_meter", 80.0))

    src = _resolve(args.source)
    out = _resolve(args.out)
    if not os.path.isdir(src):
        print("[error] source not found: %s" % src)
        return 2
    if os.path.abspath(src) == os.path.abspath(out):
        print("[error] --source and --out are the same directory; this script "
              "never rewrites a cache in place (the C=32 run must stay loadable)")
        return 2
    if os.path.isdir(out) and os.listdir(out) and not args.overwrite:
        print("[error] %s exists and is not empty (pass --overwrite)" % out)
        return 2

    ctrl0 = np.load(os.path.join(src, SPLITS[0], "control_gt.npy"), mmap_mode="r")
    c_src = int(ctrl0.shape[1])
    print("[refit] source   : %s  (C=%d)" % (src, c_src))
    print("[refit] out      : %s  (C=%d)" % (out, c))
    print("[refit] C=%d -> %d   H=%d  degree=%d  knot_sum=%.4f  scene_to_meter=%.1f"
          % (c_src, c, h, degree, float(np.asarray(knots).sum()),
             DECODE_SCENE_TO_METER))
    if c_src == c:
        print("[warn] source already has C=%d; this is a pure re-fit" % c)

    report = dict(config=args.config, source=src, out=out,
                  C_source=c_src, C=int(c), H=int(h), degree=int(degree),
                  knots=[float(v) for v in np.asarray(knots).reshape(-1)],
                  root_files_skipped=list(ROOT_FILES_SKIPPED),
                  per_split={})
    os.makedirs(out, exist_ok=True)
    root_copied, root_missing = _copy_root(src, out)
    print("[refit] root files: %s%s"
          % (", ".join(root_copied) or "(none)",
             "" if not root_missing else "   MISSING: " + ", ".join(root_missing)))
    report["root_files"] = root_copied
    out_ok = True
    for split in SPLITS:
        s_dir = os.path.join(src, split)
        if not os.path.isdir(s_dir):
            print("[refit] %-5s : MISSING" % split)
            continue
        meta = refit_split(s_dir, os.path.join(out, split), c, h, degree,
                           knots, c_src)
        report["per_split"][split] = meta
        if meta["endpoint_err_scene"] > 1e-6:
            out_ok = False
        print("[refit] %-5s : n=%d  decode_rmse C=%d: %.4f m (max %.4f)  ->  "
              "C=%d: %.4f m (max %.4f)  endpoint_err=%.1e  %.1fs"
              % (split, meta["n"], c_src,
                 meta["source_decode_rmse_m"] or float("nan"),
                 meta["source_decode_rmse_max_m"] or float("nan"),
                 c, meta["decode_rmse_m"], meta["decode_rmse_max_scene"]
                 * DECODE_SCENE_TO_METER, meta["endpoint_err_scene"],
                 meta["seconds"]), flush=True)

    report["endpoint_ok"] = bool(out_ok)
    with open(os.path.join(out, "refit_controls_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\nDONE %s   (report: refit_controls_report.json)" % out)
    print("next: python scripts/data/carla/03_validate_processed.py "
          "--processed %s --config %s" % (args.out, args.config))
    return 0 if out_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
