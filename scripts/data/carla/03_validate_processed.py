"""03_validate_processed.py - full offline validator for the CARLA processed cache.

Checks the contract of ``docs/CARLA_BSPLINE_PIPELINE_SPEC.md`` before training:

  * all arrays exist with the expected shape/dtype;
  * no NaN/Inf; occupancy in {0,1};
  * control_gt endpoints == start/goal and decode(control_gt) == curve_gt;
  * candidate mask / offset / length consistency;
  * topology_best is a valid index whenever the sample has a candidate;
  * the fixed ellipse centres Gamma_m(i/127) are finite, in range, monotone;
  * the ellipse labels are valid where flagged and a>=b>0;
  * no progress / ellipse-centre label file exists;
  * no episode crosses splits;
  * clean_manifest.jsonl sample order matches sample_id.npy.

Exit code 0 when there is no ERROR (warnings are allowed with --allow-warnings).

    python scripts/data/carla/03_validate_processed.py --processed data/carla_processed
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import interpolate_path
from src.geometry.bspline import BSplineCodec

SPLITS = ["train", "val", "test"]
CELL = 2.0 / 256.0
METERS = 40.0
FILES = {
    "conditions": ((None, 2, 2), np.float32),
    "control_gt": ((None, 32, 2), np.float32),
    "curve_gt": ((None, 128, 2), np.float32),
    "occupancy": ((None, 256, 256), np.uint8),
    "episode_id": ((None,), np.int64),
    "sample_id": ((None,), np.int64),
    "candidate_xy": ((None, None, 128, 2), np.float32),
    "candidate_mask": ((None, None), np.bool_),
    "candidate_lengths": ((None, None), np.float32),
    "candidate_geometry": ((None, 2), np.int16),
    "candidate_geometry_offsets": ((None, None), np.int64),
    "candidate_geometry_lengths": ((None, None), np.int32),
    "topology_best": ((None,), np.int64),
    "ellipse_shape4_gt": ((None, 128, 4), np.float32),
    "shape_valid": ((None, 128), np.bool_),
}


class Checker:
    def __init__(self, limit=None):
        self.errors = []
        self.warnings = []
        self.limit = limit

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def validate_split(chk, processed, split, knots_path, geometry_points):
    d = os.path.join(processed, split)
    info = {"split": split}
    if not os.path.isdir(d):
        chk.error("%s: missing split directory %s" % (split, d))
        return info
    arrays = {}
    for name, (shape, dtype) in FILES.items():
        p = os.path.join(d, name + ".npy")
        if not os.path.exists(p):
            chk.error("%s: missing %s" % (split, p))
            continue
        arr = np.load(p, mmap_mode="r")
        arrays[name] = arr
        if arr.dtype != dtype:
            chk.error("%s: %s dtype %s != %s" % (split, name, arr.dtype, dtype))
        exp = tuple(shape)
        got = tuple(arr.shape)
        for k, e in enumerate(exp):
            if e is not None and k < len(got) and got[k] != e:
                chk.error("%s: %s shape %s incompatible with %s"
                          % (split, name, got, exp))
                break
    if "conditions" not in arrays:
        return info
    n = int(arrays["conditions"].shape[0])
    if chk.limit is not None:
        n = min(n, int(chk.limit))
    info["n"] = n
    M = int(arrays["candidate_mask"].shape[1])
    H = 128

    for name in ("conditions", "control_gt", "curve_gt", "candidate_xy",
                 "ellipse_shape4_gt"):
        if name in arrays and not np.isfinite(
                np.asarray(arrays[name][:n], dtype=np.float32)).all():
            chk.error("%s: %s contains NaN/Inf" % (split, name))
    occ = arrays["occupancy"]
    if occ.size and not np.isin(np.unique(np.asarray(occ[:min(n, 200)])),
                                [0, 1]).all():
        chk.error("%s: occupancy has values outside {0,1}" % split)

    cond = np.asarray(arrays["conditions"][:n], dtype=np.float64)
    q = np.asarray(arrays["control_gt"][:n], dtype=np.float64)
    curve = np.asarray(arrays["curve_gt"][:n], dtype=np.float64)
    e0 = np.abs(q[:, 0] - cond[:, 0]).max(initial=0.0)
    e1 = np.abs(q[:, -1] - cond[:, 1]).max(initial=0.0)
    if max(e0, e1) > 1e-5:
        chk.error("%s: control endpoints != start/goal (%.3g / %.3g)"
                  % (split, e0, e1))
    codec = BSplineCodec(degree=3, num_controls=int(q.shape[1]),
                         curve_points=int(curve.shape[1]), knots_path=knots_path)
    import torch
    with torch.no_grad():
        rec = codec.decode_controls(
            torch.as_tensor(q, dtype=torch.float32)).numpy()
    err = np.linalg.norm(rec - curve, axis=2)
    info["control_fit_rmse_m"] = float(np.sqrt((err ** 2).mean())) * METERS
    info["control_fit_rmse_p95_m"] = float(
        np.percentile(np.sqrt((err ** 2).mean(1)), 95)) * METERS
    info["control_fit_max_m"] = float(err.max()) * METERS
    info["endpoint_err"] = float(max(e0, e1))

    mask = np.asarray(arrays["candidate_mask"][:n]).astype(bool)
    glen = np.asarray(arrays["candidate_geometry_lengths"][:n]).astype(np.int64)
    offs = np.asarray(arrays["candidate_geometry_offsets"][:n]).astype(np.int64)
    best = np.asarray(arrays["topology_best"][:n]).astype(np.int64)
    geom = arrays["candidate_geometry"]
    if (np.diff(offs, axis=1) < 0).any():
        chk.error("%s: candidate_geometry_offsets are not monotone" % split)
    gv = glen[mask]
    if gv.size and int(gv.min()) < 2:
        chk.error("%s: %d valid candidates have < 2 dense points"
                  % (split, int((gv < 2).sum())))
    if glen[~mask].max(initial=0) != 0:
        chk.warn("%s: invalid candidates carry non-zero geometry length" % split)
    if int(glen.sum()) != int(geom.shape[0]):
        chk.error("%s: sum(geometry_lengths)=%d != geometry rows=%d"
                  % (split, glen.sum(), geom.shape[0]))
    if int(glen.max(initial=0)) > geometry_points:
        chk.error("%s: dense geometry %d > padding budget %d"
                  % (split, glen.max(), geometry_points))
    has = mask.any(axis=1)
    bad_best = has & ((best < 0) | (best >= M) | ~mask[np.arange(n), best])
    if bad_best.any():
        chk.error("%s: topology_best invalid for %d samples"
                  % (split, int(bad_best.sum())))
    info["empty_candidate_count"] = int((~has).sum())
    info["empty_rate"] = float((~has).sum() / max(n, 1))

    s4 = np.asarray(arrays["ellipse_shape4_gt"][:n], dtype=np.float64)
    valid = np.asarray(arrays["shape_valid"][:n]).astype(bool)
    a = np.exp(s4[..., 0])
    b = np.exp(s4[..., 1])
    dirn = s4[..., 2] ** 2 + s4[..., 3] ** 2
    if valid.any() and not (a[valid] >= b[valid] - 1e-6).all():
        chk.error("%s: ellipse labels violate a >= b" % split)
    if valid.any() and not np.allclose(dirn[valid], 1.0, atol=1e-4):
        chk.error("%s: ellipse direction is not a unit vector" % split)
    if not np.isfinite(s4).all():
        chk.error("%s: ellipse_shape4_gt contains NaN/Inf" % split)
    bad_valid = valid & ~has[:, None]
    if bad_valid.any():
        chk.error("%s: %d shape labels on samples without candidates"
                  % (split, int(bad_valid.any(axis=1).sum())))
    info["shape_valid_fraction"] = float(valid.mean())
    info["samples_with_no_valid_shape"] = int((~valid.any(axis=1)).sum())
    info["a_median"] = float(np.median(a[valid])) if valid.any() else None

    # fixed-centre spot check (first up-to-64 samples with a candidate)
    idxs = np.nonzero(has)[0][:64]
    if len(idxs):
        boxes = []
        for i in idxs:
            m = int(best[i])
            lo, hi = int(offs[i, m]), int(offs[i, m + 1])
            if hi - lo < 2:
                continue
            poly = (np.asarray(geom[lo:hi], dtype=np.float64) + 0.5) * CELL - 1.0
            poly[0] = cond[i, 0]
            poly[-1] = cond[i, 1]
            c = interpolate_path(poly, np.linspace(0.0, 1.0, H))
            boxes.append((np.isfinite(c).all(), c.min(), c.max()))
        fin = all(b[0] for b in boxes)
        lo = min(b[1] for b in boxes)
        hi = max(b[2] for b in boxes)
        if not fin:
            chk.error("%s: fixed centre decode produced NaN" % split)
        if lo < -1.05 or hi > 1.05:
            chk.error("%s: fixed centres out of range [%.3f, %.3f]"
                      % (split, lo, hi))
        info["fixed_center_range"] = [float(lo), float(hi)]
        info["fixed_center_checked"] = len(boxes)

    for stale in ("progress_gt.npy", "ellipse_center_gt.npy"):
        if os.path.exists(os.path.join(d, stale)):
            chk.error("%s: stale label file %s must not exist" % (split, stale))
    return info


def validate_manifest(chk, processed):
    path = os.path.join(processed, "clean_manifest.jsonl")
    if not os.path.exists(path):
        chk.error("missing clean_manifest.jsonl")
        return {}
    per_split = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            per_split.setdefault(rec["split"], []).append(rec)
    out = {}
    for split, recs in per_split.items():
        d = os.path.join(processed, split)
        p = os.path.join(d, "sample_id.npy")
        if not os.path.exists(p):
            continue
        ids = np.load(p)
        mismatch = 0
        for rec in recs:
            i = int(rec.get("index_in_split", -1))
            if 0 <= i < len(ids) and int(rec["sample_id"]) != int(ids[i]):
                mismatch += 1
        if mismatch:
            chk.error("manifest/%s: %d sample_id mismatches" % (split, mismatch))
        out[split] = len(recs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-warnings", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed = os.path.abspath(args.processed or cfg["data"].get(
        "processed_root", "data/carla_processed"))
    knots_path = (cfg.get("bspline") or {}).get("knots")
    geometry_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))
    chk = Checker(limit=args.limit)
    report = {"processed": processed, "config": args.config, "splits": {}}
    for split in args.splits:
        report["splits"][split] = validate_split(chk, processed, split,
                                                 knots_path, geometry_points)

    # episode leakage
    eps = {}
    for split in args.splits:
        p = os.path.join(processed, split, "episode_id.npy")
        if os.path.exists(p):
            eps[split] = set(np.unique(np.load(p)).tolist())
            report["splits"][split]["unique_episodes"] = len(eps[split])
    for a in args.splits:
        for b in args.splits:
            if a < b and a in eps and b in eps:
                inter = eps[a] & eps[b]
                if inter:
                    chk.error("episode leakage between %s and %s: %d episodes"
                              % (a, b, len(inter)))
    report["manifest_per_split"] = validate_manifest(chk, processed)
    report["errors"] = chk.errors
    report["warnings"] = chk.warnings
    report["valid"] = len(chk.errors) == 0

    print("=" * 74)
    print("PROCESSED CACHE VALIDATION")
    for split, info in report["splits"].items():
        print("[%s] %s" % (split, json.dumps(info, default=str)))
    print("[manifest] %s" % report["manifest_per_split"])
    print("[errors] %d" % len(chk.errors))
    for e in chk.errors:
        print("   ERROR  ", e)
    print("[warnings] %d" % len(chk.warnings))
    for w in chk.warnings:
        print("   WARN   ", w)
    print("VALID =", report["valid"])
    print("=" * 74)
    with open(os.path.join(processed, "preprocess_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if chk.errors:
        return 1
    if chk.warnings and not args.allow_warnings:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
