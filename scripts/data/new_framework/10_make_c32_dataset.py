#!/usr/bin/env python
"""10_make_c32_dataset.py - new processed root with C=32 control labels.

The new framework (see ``TrajSafe-Diffuser 新框架整体改造实施方案.md`` §2.1/§9.1)
uses 32 B-spline control points and 64 ellipse queries.  ``control_gt.npy`` is
the ONLY array of the shipped k4p cache whose shape depends on C, so the new
root ``data/processed/160m-k4p-c32q64`` is produced exactly the way
``scripts/data/carla_full/06_refit_controls.py`` produced the C=48 root:

    control_gt.npy  [N,32,2]   endpoint-constrained LS projection of
                              ``curve_gt.npy`` onto 32 controls
                              (``src.geometry.bspline.numpy_fit_curve_to_controls``)
    everything else            copied byte-for-byte from the source split

``06_refit_controls.py`` is imported and its functions are reused
(``_solve_knots`` / ``_copy_split`` / ``_decode_rmse`` / ``_copy_root``); the only
reason this script exists instead of calling 06 directly is the ``--limit`` hook
and the report.  NOTE: ``ellipse_shape4_gt.npy`` / ``shape_valid.npy`` are still
the LEGACY 128-query arrays here -- ``13_build_ellipse64_labels.py`` replaces
them with the new ``[N,64,4]`` / ``[N,64]`` labels.

    python scripts/data/new_framework/10_make_c32_dataset.py --limit 8
    python scripts/data/new_framework/10_make_c32_dataset.py --overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.geometry.bspline import (  # noqa: E402
    BSplineCodec, default_knots, numpy_fit_curve_to_controls)
from src.utils.config import curve_points, load_config  # noqa: E402

SPLITS = ("train", "val", "test")
DEFAULT_SOURCE = "data/processed/160m-k4p-c48"
DEFAULT_OUT = "data/processed/160m-k4p-c32q64"
DEFAULT_CONFIG = "configs/config_160k4p_c48_oneshot.yaml"
REFIT_SCRIPT = os.path.join(REPO_ROOT, "scripts", "data", "carla_full",
                            "06_refit_controls.py")
#: arrays whose byte identity is verified against the source split
CHECK_FILES = ("conditions.npy", "curve_gt.npy", "occupancy.npy",
               "candidate_xy.npy", "candidate_mask.npy",
               "candidate_geometry.npy", "topology_best.npy")


def _resolve(path_like) -> str:
    p = os.path.expanduser(str(path_like))
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _load_refit_module():
    """Import 06_refit_controls.py (file name starts with a digit -> spec load)."""
    spec = importlib.util.spec_from_file_location("refit_controls_06", REFIT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _refit_split(mod, src_dir, dst_dir, c, h, degree, knots, c_src, limit,
                 scene_to_meter):
    """06's ``refit_split`` + a ``--limit`` hook (same math, same copy path)."""
    t0 = time.time()
    copied, missing = mod._copy_split(src_dir, dst_dir)

    curve = np.load(os.path.join(src_dir, "curve_gt.npy")).astype(np.float64)
    cond = np.load(os.path.join(src_dir, "conditions.npy")).astype(np.float64)
    ctrl_src = np.load(os.path.join(src_dir, "control_gt.npy"))
    n_full = int(curve.shape[0])
    n = n_full if limit is None else min(int(limit), n_full)
    if curve.shape[1] != h:
        raise SystemExit("[error] %s: curve_gt has H=%d, config says H=%d"
                         % (src_dir, curve.shape[1], h))
    if ctrl_src.shape[1] != c_src:
        raise SystemExit("[error] %s: control_gt has C=%d, expected C_src=%d"
                         % (src_dir, ctrl_src.shape[1], c_src))

    ctrl = np.zeros((n, c, 2), np.float32)
    start, goal = cond[:n, 0, :], cond[:n, 1, :]
    t_fit = time.time()
    for i in range(n):
        ctrl[i] = numpy_fit_curve_to_controls(
            knots, c, degree, curve[i], start[i], goal[i], h).astype(np.float32)
    fit_s = time.time() - t_fit
    np.save(os.path.join(dst_dir, "control_gt.npy"), ctrl)

    codec_new = BSplineCodec(degree=degree, num_controls=c, curve_points=h,
                             knots=knots)
    m_new = mod._decode_rmse(codec_new, ctrl, curve[:n])
    m_src = None
    if c_src != c:
        codec_src = BSplineCodec(degree=degree, num_controls=c_src,
                                  curve_points=h,
                                  knots=default_knots(c_src, degree))
        m_src = mod._decode_rmse(codec_src, ctrl_src[:n], curve[:n])
    end_err = float(max(np.abs(ctrl[:, 0, :] - start).max(),
                        np.abs(ctrl[:, -1, :] - goal).max()))
    sc = float(scene_to_meter)
    bytes_ok = {}
    for name in CHECK_FILES:
        s, d = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if os.path.isfile(s) and os.path.isfile(d):
            bytes_ok[name] = _md5(s) == _md5(d)
    meta = dict(
        n=int(n), n_full=int(n_full), C=int(c), H=int(h), degree=int(degree),
        C_source=int(c_src), limited=bool(limit is not None),
        endpoint_err_scene=end_err,
        decode_rmse_scene=m_new["rmse_mean"],
        decode_rmse_m=m_new["rmse_mean"] * sc,
        decode_rmse_max_m=m_new["rmse_max"] * sc,
        decode_point_max_m=m_new["pt_max"] * sc,
        source_decode_rmse_m=(None if m_src is None else m_src["rmse_mean"] * sc),
        source_decode_rmse_max_m=(None if m_src is None else m_src["rmse_max"] * sc),
        control_range=[[float(ctrl.min()), float(ctrl.max())]],
        byte_identical=bytes_ok, copied=len(copied), missing=sorted(missing),
        fit_seconds=fit_s, seconds=time.time() - t0)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="config that carries H / degree / knots / scene_to_meter")
    ap.add_argument("--num-controls", type=int, default=32,
                    help="target C (the new framework uses 32)")
    ap.add_argument("--limit", type=int, default=None,
                    help="self-check: fit only the first N samples per split")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = int(args.num_controls)
    h = curve_points(cfg)
    bs = cfg.get("bspline") or {}
    degree = int(bs.get("degree", 3))
    scene_to_meter = float((cfg.get("data") or {}).get("scene_to_meter", 80.0))
    src, out = _resolve(args.source), _resolve(args.out)
    if not os.path.isdir(src):
        print("[error] source not found: %s" % src)
        return 2
    if os.path.abspath(src) == os.path.abspath(out):
        print("[error] --source == --out; this script never rewrites a cache in place")
        return 2
    if os.path.isdir(out) and os.listdir(out) and not args.overwrite:
        print("[error] %s exists and is not empty (pass --overwrite)" % out)
        return 2

    mod = _load_refit_module()
    mod.DECODE_SCENE_TO_METER = scene_to_meter
    knots = mod._solve_knots(bs, c, degree)
    c_src = int(np.load(os.path.join(src, SPLITS[0], "control_gt.npy"),
                        mmap_mode="r").shape[1])
    print("[c32] source : %s (C=%d)" % (src, c_src))
    print("[c32] out    : %s (C=%d)" % (out, c))
    print("[c32] H=%d degree=%d knots=%d scene_to_meter=%.1f limit=%s"
          % (h, degree, len(np.asarray(knots).reshape(-1)), scene_to_meter,
             args.limit))

    report = dict(config=args.config, source=src, out=out, C=int(c),
                  C_source=c_src, H=int(h), degree=int(degree),
                  knots=[float(v) for v in np.asarray(knots).reshape(-1)],
                  limit=args.limit, per_split={})
    os.makedirs(out, exist_ok=True)
    root_copied, root_missing = mod._copy_root(src, out)
    report["root_files"] = root_copied
    print("[c32] root files: %s%s"
          % (", ".join(root_copied) or "(none)",
             "" if not root_missing else "  MISSING: " + ", ".join(root_missing)))

    ok = True
    for split in SPLITS:
        s_dir = os.path.join(src, split)
        if not os.path.isdir(s_dir):
            print("[c32] %-5s : MISSING" % split)
            continue
        meta = _refit_split(mod, s_dir, os.path.join(out, split), c, h, degree,
                            knots, c_src, args.limit, scene_to_meter)
        report["per_split"][split] = meta
        ok = ok and meta["endpoint_err_scene"] <= 1e-6
        print("[c32] %-5s : n=%d/%d  decode RMSE C=%d %.3f m (max %.3f) -> "
              "C=%d %.3f m (max %.3f)  endpoint_err=%.1e  bytes_ok=%s  %.1fs"
              % (split, meta["n"], meta["n_full"], c_src,
                 meta["source_decode_rmse_m"] or float("nan"),
                 meta["source_decode_rmse_max_m"] or float("nan"), c,
                 meta["decode_rmse_m"], meta["decode_rmse_max_m"],
                 meta["endpoint_err_scene"], all(meta["byte_identical"].values()),
                 meta["seconds"]), flush=True)
    report["endpoint_ok"] = bool(ok)
    report["note"] = ("control_gt is the only array refit; ellipse_shape4_gt / "
                      "shape_valid are still the legacy 128-query labels and are "
                      "replaced by 13_build_ellipse64_labels.py")
    with open(os.path.join(out, "c32_controls_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\nDONE %s  (report: c32_controls_report.json)" % out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
