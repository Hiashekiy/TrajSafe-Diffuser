#!/usr/bin/env python
"""11_build_heading_labels.py - endpoint heading labels for the new framework.

The new framework conditions on the heading at BOTH endpoints (root-cause of the
"endpoint orientation" condition, plan §8/§9.1).  Vehicle yaw is NOT available in
this snapshot: the 160 m k4p cache was built from ``trajectory_128`` only, with no
pose / yaw channel, so the heading stored here is a PROXY:

    start heading = unit tangent fitted to the FIRST ``--window`` curve points,
                    oriented along the travel direction
    goal  heading = unit tangent fitted to the LAST ``--window`` curve points,
                    oriented along the travel direction

(the tangent of a degree-3 clamped B-spline / the polyline chord of a short
window; a least-squares principal direction of the window, sign-aligned with the
window's own forward difference, so a hairpin cannot flip it.)

``heading_valid`` is False when the tangent is not defined:
  * any non-finite coordinate in the curve,
  * the polyline is shorter than ``--min-curve-len-m`` (degenerate curve),
  * the window's principal spread is below ``--min-spread-m`` (all points equal).

Output (new root, per split):
    heading.npy        [N,2,2] float32  [start_dir, goal_dir], unit vectors, 0 if invalid
    heading_valid.npy  [N,2]   bool

    python scripts/data/new_framework/11_build_heading_labels.py --limit 8
    python scripts/data/new_framework/11_build_heading_labels.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402

SPLITS = ("train", "val", "test")
DEFAULT_ROOT = "data/processed/160m-k4p-c32q64"
DEFAULT_CONFIG = "configs/config_160k4p_c48_oneshot.yaml"
NOTE = ("proxy heading from GT curve tangent, not vehicle yaw "
        "(the 160 m snapshot carries no pose/yaw channel)")


def _resolve(path_like) -> str:
    p = os.path.expanduser(str(path_like))
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _principal_direction(pts: np.ndarray) -> np.ndarray:
    """Unit principal direction of a short window (SVD of the centred points)."""
    c = pts - pts.mean(axis=0, keepdims=True)
    try:
        _, sv, vt = np.linalg.svd(c, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(2)
    if sv.size == 0 or sv[0] <= 0.0:
        return np.zeros(2)
    return np.asarray(vt[0], dtype=np.float64)


def _endpoint_heading(pts: np.ndarray, window: int, at_start: bool,
                      min_spread: float):
    """``(direction, valid, reason)`` for one endpoint of one curve."""
    n = len(pts)
    k = int(min(max(int(window), 2), n))
    win = pts[:k] if at_start else pts[n - k:]
    if not np.isfinite(win).all():
        return np.zeros(2), False, "non_finite"
    # the window's own travel direction (start window -> inward, end -> outward)
    forward = win[-1] - win[0]
    spread = float(np.linalg.norm(win - win.mean(axis=0, keepdims=True),
                                  axis=1).max())
    if spread < float(min_spread):
        return np.zeros(2), False, "degenerate_window"
    d = _principal_direction(win)
    nd = float(np.linalg.norm(d))
    nf = float(np.linalg.norm(forward))
    if nd <= 1e-12:
        return np.zeros(2), False, "degenerate_window"
    d = d / nd
    if nf <= 1e-12:
        # the window does not move: the sign is undecidable
        return np.zeros(2), False, "ambiguous_sign"
    if float(np.dot(d, forward)) < 0.0:
        d = -d
    return d, True, ""


def build_split(root: str, split: str, window: int, min_curve_len_m: float,
                min_spread_m: float, scene_to_meter: float, limit=None):
    d = os.path.join(root, split)
    curve = np.load(os.path.join(d, "curve_gt.npy"))
    n_full = int(curve.shape[0])
    n = n_full if limit is None else min(int(limit), n_full)
    heading = np.zeros((n, 2, 2), np.float32)
    valid = np.zeros((n, 2), bool)
    reasons = {"start": {}, "end": {}}
    chord_angle = np.full((n, 2), np.nan)
    min_spread = float(min_spread_m) / float(scene_to_meter)
    min_len = float(min_curve_len_m) / float(scene_to_meter)
    for i in range(n):
        pts = np.asarray(curve[i], dtype=np.float64)
        finite = np.isfinite(pts).all(axis=1)
        length = float(np.linalg.norm(np.diff(pts[finite], axis=0), axis=1).sum()) \
            if finite.sum() >= 2 else 0.0
        if finite.sum() < 4 or length < min_len:
            why = "non_finite" if finite.sum() < 4 else "short_curve"
            for side in ("start", "end"):
                reasons[side][why] = reasons[side].get(why, 0) + 1
            continue
        pts = pts[finite]
        chord = pts[-1] - pts[0]
        for j, at_start in enumerate((True, False)):
            if at_start and not finite[0]:
                reasons["start"]["non_finite_endpoint"] = \
                    reasons["start"].get("non_finite_endpoint", 0) + 1
                continue
            if (not at_start) and not finite[-1]:
                reasons["end"]["non_finite_endpoint"] = \
                    reasons["end"].get("non_finite_endpoint", 0) + 1
                continue
            dirn, ok, why = _endpoint_heading(pts, window, at_start, min_spread)
            if not ok:
                side = "start" if at_start else "end"
                reasons[side][why] = reasons[side].get(why, 0) + 1
                continue
            heading[i, j] = dirn
            valid[i, j] = True
            nc = float(np.linalg.norm(chord))
            if nc > 1e-9:
                chord_angle[i, j] = np.degrees(np.arccos(np.clip(
                    float(np.dot(dirn, chord)) / nc, -1.0, 1.0)))
    np.save(os.path.join(d, "heading.npy"), heading)
    np.save(os.path.join(d, "heading_valid.npy"), valid)
    both = valid.all(axis=1)
    ang = chord_angle[valid]
    stats = dict(
        n=int(n), n_full=int(n_full), window=int(window),
        valid_start=int(valid[:, 0].sum()), valid_end=int(valid[:, 1].sum()),
        valid_both=int(both.sum()),
        valid_frac_start=float(valid[:, 0].mean()),
        valid_frac_end=float(valid[:, 1].mean()),
        valid_frac_both=float(both.mean()),
        invalid_reasons=reasons,
        unit_norm_max_err=float(np.abs(np.linalg.norm(
            heading[valid], axis=-1) - 1.0).max()) if valid.any() else 0.0,
        heading_range=[[float(heading[..., 0].min()), float(heading[..., 0].max())],
                       [float(heading[..., 1].min()), float(heading[..., 1].max())]],
        chord_angle_deg=(dict(mean=float(np.mean(ang)), median=float(np.median(ang)),
                              p95=float(np.percentile(ang, 95)),
                              max=float(np.max(ang))) if ang.size else None),
        min_curve_len_m=float(min_curve_len_m),
        min_spread_m=float(min_spread_m),
    )
    return heading, valid, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--window", type=int, default=4,
                    help="number of curve points used for the tangent fit")
    ap.add_argument("--min-curve-len-m", type=float, default=1.0)
    ap.add_argument("--min-spread-m", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}
    scene_to_meter = float((cfg.get("data") or {}).get("scene_to_meter", 80.0))
    root = _resolve(args.root)
    if not os.path.isdir(root):
        print("[error] root not found: %s (run 10_make_c32_dataset.py first)" % root)
        return 2
    report = dict(root=root, config=args.config, window=int(args.window),
                  scene_to_meter=scene_to_meter, limit=args.limit,
                  note=NOTE, per_split={})
    print("[heading] root=%s  window=%d  scene_to_meter=%.1f  limit=%s"
          % (root, args.window, scene_to_meter, args.limit))
    for split in args.splits:
        if not os.path.isdir(os.path.join(root, split)):
            print("[heading] %-5s MISSING" % split)
            continue
        heading, valid, stats = build_split(
            root, split, args.window, args.min_curve_len_m, args.min_spread_m,
            scene_to_meter, args.limit)
        report["per_split"][split] = stats
        ca = stats["chord_angle_deg"]
        print("[heading] %-5s n=%d  valid start/end/both=%.3f/%.3f/%.3f  "
              "range x=[%.2f,%.2f] y=[%.2f,%.2f]  |chord angle| mean=%s"
              % (split, stats["n"], stats["valid_frac_start"],
                 stats["valid_frac_end"], stats["valid_frac_both"],
                 *stats["heading_range"][0], *stats["heading_range"][1],
                 "n/a" if ca is None else "%.1f deg" % ca["mean"]), flush=True)
        if stats["invalid_reasons"]["start"] or stats["invalid_reasons"]["end"]:
            print("           invalid reasons: start=%s end=%s"
                  % (stats["invalid_reasons"]["start"],
                     stats["invalid_reasons"]["end"]), flush=True)
    out = os.path.join(root, "heading_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\nDONE heading -> %s  (report: %s)" % (root, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
