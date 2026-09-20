#!/usr/bin/env python
"""Clean the raw CARLA v1 dataset into fixed-length per-split arrays.

This script implements the *raw -> processed* stage of
``docs/CARLA_BSPLINE_PIPELINE_SPEC.md``.

It reads ``data/carla_v1/samples.jsonl`` (and ``episodes.jsonl``) and, for every
record, validates the corresponding ``.npz`` sample.  Accepted samples are
projected onto the fixed 32-control cubic B-spline codec
(``src.geometry.bspline.numpy_fit_curve_to_controls``) and written as
per-split ``.npy`` arrays under ``data/carla_processed``.

Rules that are easy to get wrong and are therefore stated explicitly:

* the raw CARLA occupancy row 0 corresponds to ``y_local = +40 m`` while the
  repository geometry convention is row 0 -> ``scene_y = -1``.  Consequently
  the *canonical* occupancy used everywhere downstream is
  ``np.flipud(occupancy).copy()`` and that is the only occupancy written out;
* one scene unit is 40 m (the crop is 80 m wide and the scene span is 2.0);
* a single broken sample must never abort the run: every failure is recorded as
  a reason and the sample is skipped;
* ``data/carla_v1`` is strictly read-only.

The script is intentionally dependency-free apart from numpy + PyYAML (optional)
and ``src.geometry.bspline``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# repository import path (the script lives in scripts/data/carla/)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.geometry.bspline import (  # noqa: E402  (import after sys.path tweak)
    bspline_basis_matrix,
    default_knots,
    default_knots_path,
    numpy_fit_curve_to_controls,
)

# ---------------------------------------------------------------------------
# constants fixed by the contract
# ---------------------------------------------------------------------------

SPLITS: Tuple[str, ...] = ("train", "val", "test")
# DEFAULTS only: ``configure_codec`` overrides them from the config / CLI so the
# number of B-spline control points is a config knob (bspline.num_controls), not
# a constant baked into the data pipeline.
NUM_CONTROLS = 32
DEGREE = 3
CURVE_POINTS = 128
OCC_RES = 256

# 1 scene unit == 40 m (crop is 80 m, scene span is 2.0)
METERS_PER_SCENE_UNIT = 40.0
SCENE_LIMIT = 1.05
MIN_GOAL_DISTANCE = 0.05
MAX_FIT_RMSE_M_DEFAULT = 0.25
MAX_GOAL_GAP_M_DEFAULT = 5.0
MAX_REPORT_SAMPLES_DEFAULT = 300
PROGRESS_EVERY = 500

EXPECTED_KNOTS_LEN = NUM_CONTROLS + DEGREE + 1


def configure_codec(num_controls: Optional[int] = None,
                    curve_points: Optional[int] = None) -> None:
    """Apply the configured control count / curve density to this module.

    The offline control labels are data, so changing ``bspline.num_controls``
    in the config means this script has to be re-run with the same value; the
    knot vector is regenerated (clamped uniform) when the stored one does not
    match, so no knot file has to be hand-edited.
    """
    global NUM_CONTROLS, CURVE_POINTS, EXPECTED_KNOTS_LEN
    if num_controls is not None:
        NUM_CONTROLS = int(num_controls)
    if curve_points is not None:
        CURVE_POINTS = int(curve_points)
    if NUM_CONTROLS <= DEGREE:
        raise SystemExit("[error] bspline.num_controls=%d must be > degree=%d"
                         % (NUM_CONTROLS, DEGREE))
    EXPECTED_KNOTS_LEN = NUM_CONTROLS + DEGREE + 1


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


class SkipSample(Exception):
    """Raised when a record must be skipped; carries a stable reason string."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _clean_message(exc: BaseException) -> str:
    msg = str(exc).replace("\r", " ").replace("\n", " ").strip()
    if len(msg) > 400:
        msg = msg[:397] + "..."
    return msg


def _exception_reason(exc: BaseException) -> str:
    return "exception:%s:%s" % (type(exc).__name__, _clean_message(exc))


def _resolve(path_like: Any) -> Path:
    """Resolve a config/CLI path: absolute stays, relative is repo-root based."""
    p = Path(str(path_like)).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _reason_key(reason: str) -> str:
    """Normalise a skip reason into a histogram bucket."""
    if reason.startswith("exception:"):
        return "exception"
    return reason.split(":", 1)[0] if ":" in reason else reason


def _as_shape(shape) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape)


def _shape_str(shape) -> str:
    try:
        return "x".join(str(int(x)) for x in _as_shape(shape))
    except Exception:
        return str(shape)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print("[config] not found: %s (using CLI defaults)" % path, flush=True)
        return {}
    try:
        import yaml  # local import so the script still runs without PyYAML
    except Exception as exc:  # pragma: no cover - environment guard
        print("[config] PyYAML unavailable (%s); using CLI defaults"
              % _clean_message(exc), flush=True)
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    except Exception as exc:
        print("[config] failed to parse %s (%s); using CLI defaults"
              % (path, _clean_message(exc)), flush=True)
        return {}
    if not isinstance(cfg, dict):
        print("[config] %s is not a mapping; using CLI defaults" % path,
              flush=True)
        return {}
    return cfg


def _cfg_get(cfg: Dict[str, Any], dotted: str) -> Optional[Any]:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def resolve_paths(args: argparse.Namespace, cfg: Dict[str, Any]
                  ) -> Tuple[Path, Path, Path, str]:
    """Return (root, out, knots_path, knots_source)."""
    root_raw = args.root
    if root_raw is None:
        root_raw = _cfg_get(cfg, "data.root") or "data/carla_v1"
    out_raw = args.out
    if out_raw is None:
        out_raw = _cfg_get(cfg, "data.processed_root") or "data/carla_processed"

    root = _resolve(root_raw)
    out = _resolve(out_raw)

    # Never write inside the read-only raw dataset.
    try:
        root_res = root.resolve()
        out_res = out.resolve()
        if out_res == root_res or root_res in out_res.parents:
            raise SystemExit(
                "[error] refusing to write into the raw dataset: out=%s root=%s"
                % (out, root))
    except OSError:
        pass

    knots_cfg = _cfg_get(cfg, "bspline.knots")
    knots_source = "config:bspline.knots"
    if knots_cfg:
        knots_path = _resolve(knots_cfg)
    else:
        candidates = [root / "bspline_knots.npy", Path(default_knots_path())]
        knots_path = candidates[0]
        knots_source = "root:bspline_knots.npy"
        if not knots_path.exists() and candidates[1].exists():
            knots_path = candidates[1]
            knots_source = "default:src.geometry.bspline.default_knots_path"
    return root, out, knots_path, knots_source


# ---------------------------------------------------------------------------
# record validation + per-sample computation
# ---------------------------------------------------------------------------


def process_record(record: Dict[str, Any], root: Path, knots: np.ndarray,
                   basis: np.ndarray, max_fit_rmse_m: float,
                   max_goal_gap_m: float = MAX_GOAL_GAP_M_DEFAULT
                   ) -> Dict[str, Any]:
    """Validate one record and compute its cleaned tensors + stats.

    Returns a dict with at least ``accepted`` and ``reason``.  Accepted samples
    also carry ``data`` (arrays) and the fit statistics.  Rejected samples can
    still carry the computed fit statistics (used for ``control_refit_error``).
    """
    # ---- step 1: required metadata -------------------------------------
    for key in ("file", "split", "episode_id"):
        if key not in record or record[key] is None:
            raise SkipSample("missing_field:%s" % key)
    split = record["split"]
    if not isinstance(split, str) or split not in SPLITS:
        raise SkipSample("invalid_split:%s" % split)

    raw_file = record["file"]
    if not isinstance(raw_file, str) or not raw_file.strip():
        raise SkipSample("invalid_file")
    raw_file = raw_file.strip()

    if "sample_id" not in record or record["sample_id"] is None:
        raise SkipSample("missing_field:sample_id")
    try:
        sample_id = int(record["sample_id"])
    except Exception:
        raise SkipSample("invalid_sample_id")
    try:
        episode_id = int(record["episode_id"])
    except Exception:
        raise SkipSample("invalid_episode_id")

    town = str(record.get("town") or "")

    # ---- step 2: path exists + parent dir == split ----------------------
    sample_path = Path(raw_file)
    if not sample_path.is_absolute():
        sample_path = root / sample_path
    if not sample_path.exists():
        raise SkipSample("npz_missing:%s" % raw_file)
    if sample_path.parent.name != split:
        raise SkipSample("parent_dir_mismatch:%s!=%s"
                         % (sample_path.parent.name, split))

    # ---- step 3: np.load succeeds + required keys ----------------------
    arrays: Dict[str, np.ndarray] = {}
    with np.load(sample_path, allow_pickle=False) as npz:
        keys = set(npz.files)
        for key in ("occupancy", "trajectory_128", "start", "goal"):
            if key not in keys:
                raise SkipSample("missing_key:%s" % key)
        for key in npz.files:
            arrays[key] = np.asarray(npz[key])

    occupancy = arrays["occupancy"]
    trajectory = arrays["trajectory_128"]
    start = arrays["start"]
    goal = arrays["goal"]
    controls_file = arrays.get("bspline_controls")

    # ---- step 4: shapes -------------------------------------------------
    if occupancy.shape != (OCC_RES, OCC_RES):
        raise SkipSample("bad_shape:occupancy:%s" % _shape_str(occupancy.shape))
    if trajectory.shape != (CURVE_POINTS, 2):
        raise SkipSample("bad_shape:trajectory_128:%s"
                         % _shape_str(trajectory.shape))
    if start.shape != (2,):
        raise SkipSample("bad_shape:start:%s" % _shape_str(start.shape))
    if goal.shape != (2,):
        raise SkipSample("bad_shape:goal:%s" % _shape_str(goal.shape))
    if controls_file is not None and controls_file.shape != (NUM_CONTROLS, 2):
        raise SkipSample("bad_shape:bspline_controls:%s"
                         % _shape_str(controls_file.shape))

    # occupancy values must be exactly {0, 1}
    if not np.all((occupancy == 0) | (occupancy == 1)):
        raise SkipSample("bad_values:occupancy")

    # ---- step 5: finite floats + ranges ---------------------------------
    for key, array in arrays.items():
        if np.issubdtype(array.dtype, np.floating):
            if not np.all(np.isfinite(array)):
                raise SkipSample("nonfinite:%s" % key)

    if np.any(np.abs(start) > SCENE_LIMIT) or np.any(np.abs(goal) > SCENE_LIMIT):
        raise SkipSample("start_goal_out_of_range")
    if float(np.linalg.norm(goal.astype(np.float64)
                            - start.astype(np.float64))) <= MIN_GOAL_DISTANCE:
        raise SkipSample("goal_start_too_close")

    # ---- step 6: embedded episode_id matches the record -----------------
    if "episode_id" not in arrays:
        raise SkipSample("missing_key:episode_id")
    stored_episode = np.asarray(arrays["episode_id"]).reshape(-1)
    if stored_episode.size == 0:
        raise SkipSample("bad_shape:episode_id:empty")
    if int(stored_episode[0]) != episode_id:
        raise SkipSample("episode_id_mismatch:%d!=%d"
                         % (int(stored_episode[0]), episode_id))

    # ---- accepted: canonical occupancy + endpoint-constrained refit -----
    occupancy_canonical = np.flipud(occupancy).astype(np.uint8, copy=True)
    curve_gt = np.asarray(trajectory, dtype=np.float32).copy()
    start_f = np.asarray(start, dtype=np.float32).reshape(2).copy()
    goal_f = np.asarray(goal, dtype=np.float32).reshape(2).copy()

    control_gt = numpy_fit_curve_to_controls(
        knots, NUM_CONTROLS, DEGREE, curve_gt, start_f, goal_f, CURVE_POINTS)
    control_gt = np.asarray(control_gt, dtype=np.float32).copy()
    if control_gt.shape != (NUM_CONTROLS, 2):
        raise SkipSample("bad_shape:control_gt:%s" % _shape_str(control_gt.shape))

    reconstructed = basis @ control_gt.astype(np.float64)
    diff = reconstructed - curve_gt.astype(np.float64)
    point_sq = np.sum(diff * diff, axis=1)
    fit_rmse_scene = float(np.sqrt(float(np.mean(point_sq))))
    fit_max_error_scene = float(np.max(np.sqrt(point_sq)))
    fit_rmse_meter = fit_rmse_scene * METERS_PER_SCENE_UNIT
    fit_max_error_meter = fit_max_error_scene * METERS_PER_SCENE_UNIT
    goal_to_executed_endpoint_meter = float(
        np.linalg.norm(goal_f.astype(np.float64)
                       - curve_gt[-1].astype(np.float64))
    ) * METERS_PER_SCENE_UNIT

    stats = {
        "fit_rmse_scene": fit_rmse_scene,
        "fit_rmse_meter": fit_rmse_meter,
        "fit_max_error_meter": fit_max_error_meter,
        "fit_max_error_scene": fit_max_error_scene,
        "goal_to_executed_endpoint_meter": goal_to_executed_endpoint_meter,
    }

    # A sample is rejected only when the endpoint-constrained refit is really
    # broken.  The pointwise maximum is normally exactly the route goal vs the
    # executed trajectory end (the goal is up to 50 m ahead on the global route),
    # which is a property of the data, not a broken fit: only an extreme gap
    # (default 5 m) or a large RMS refit error (default 0.25 m) is rejected.
    if (fit_rmse_meter > max_fit_rmse_m
            or goal_to_executed_endpoint_meter > max_goal_gap_m):
        return {
            "accepted": False,
            "reason": "control_refit_error",
            "stats": stats,
            "meta": {
                "sample_id": sample_id,
                "episode_id": episode_id,
                "town": town,
                "split": split,
                "file": raw_file,
            },
        }

    return {
        "accepted": True,
        "reason": None,
        "stats": stats,
        "meta": {
            "sample_id": sample_id,
            "episode_id": episode_id,
            "town": town,
            "split": split,
            "file": raw_file,
        },
        "data": {
            "conditions": np.stack([start_f, goal_f], axis=0).astype(np.float32),
            "control_gt": control_gt,
            "curve_gt": curve_gt,
            "occupancy": occupancy_canonical,
            "episode_id": np.int64(episode_id),
            "sample_id": np.int64(sample_id),
        },
    }


def _metric_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None,
                "max": None, "min": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# episodes (informational cross-check)
# ---------------------------------------------------------------------------


def load_episodes(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Optional[str]]:
    episodes: Dict[int, Dict[str, Any]] = {}
    error: Optional[str] = None
    if not path.exists():
        return episodes, "episodes.jsonl not found"
    try:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    episodes[int(rec["episode_id"])] = rec
                except Exception as exc:
                    if error is None:
                        error = "line %d: %s" % (lineno, _clean_message(exc))
    except Exception as exc:
        error = _clean_message(exc)
    return episodes, error


# ---------------------------------------------------------------------------
# array assembly / output
# ---------------------------------------------------------------------------


def _stack(list_of_arrays: Sequence[np.ndarray], trailing_shape: Tuple[int, ...],
           dtype: np.dtype) -> np.ndarray:
    if not list_of_arrays:
        return np.zeros((0,) + trailing_shape, dtype=dtype)
    return np.stack(list_of_arrays).astype(dtype, copy=False)


def write_split_arrays(out: Path, split: str, samples: Sequence[Dict[str, Any]]
                       ) -> Dict[str, Tuple[int, ...]]:
    split_dir = out / split
    split_dir.mkdir(parents=True, exist_ok=True)

    conditions = _stack([s["conditions"] for s in samples], (2, 2), np.float32)
    control_gt = _stack([s["control_gt"] for s in samples],
                        (NUM_CONTROLS, 2), np.float32)
    curve_gt = _stack([s["curve_gt"] for s in samples],
                      (CURVE_POINTS, 2), np.float32)
    occupancy = _stack([s["occupancy"] for s in samples],
                       (OCC_RES, OCC_RES), np.uint8)
    episode_id = np.asarray([s["episode_id"] for s in samples], dtype=np.int64)
    sample_id = np.asarray([s["sample_id"] for s in samples], dtype=np.int64)
    if episode_id.ndim == 0:
        episode_id = episode_id.reshape(1)
    if sample_id.ndim == 0:
        sample_id = sample_id.reshape(1)

    n = len(samples)
    shapes = {
        "conditions": (n, 2, 2),
        "control_gt": (n, NUM_CONTROLS, 2),
        "curve_gt": (n, CURVE_POINTS, 2),
        "occupancy": (n, OCC_RES, OCC_RES),
        "episode_id": (n,),
        "sample_id": (n,),
    }

    for name, array in (
        ("conditions", conditions),
        ("control_gt", control_gt),
        ("curve_gt", curve_gt),
        ("occupancy", occupancy),
        ("episode_id", episode_id),
        ("sample_id", sample_id),
    ):
        expected = shapes[name]
        if array.shape != expected:
            raise RuntimeError("internal shape error for %s/%s: %s != %s"
                               % (split, name, array.shape, expected))
        np.save(split_dir / (name + ".npy"), array)
    return shapes


def write_manifest(out: Path, split_entries: "OrderedDict[str, List[Dict[str, Any]]]"
                   ) -> int:
    total = 0
    manifest_path = out / "clean_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for split in SPLITS:
            for index, entry in enumerate(split_entries.get(split, [])):
                row = {
                    "sample_id": int(entry["sample_id"]),
                    "episode_id": int(entry["episode_id"]),
                    "town": entry["town"],
                    "split": split,
                    "file": entry["file"],
                    "index_in_split": int(index),
                    "fit_rmse_meter": float(entry["fit_rmse_meter"]),
                    "fit_max_error_meter": float(entry["fit_max_error_meter"]),
                    "goal_to_executed_endpoint_meter": float(
                        entry["goal_to_executed_endpoint_meter"]),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    return total


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean CARLA v1 raw samples into fixed per-split arrays.")
    parser.add_argument("--root", default=None,
                        help="raw dataset root (default: data/carla_v1 or "
                             "config data.root)")
    parser.add_argument("--out", default=None,
                        help="processed output root (default: "
                             "data/carla_processed or config data.processed_root)")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="YAML config; bspline.num_controls, "
                             "bspline.curve_points, bspline.knots, data.root and "
                             "data.processed_root are read")
    parser.add_argument("--num-controls", type=int, default=None,
                        help="override bspline.num_controls (B-spline control "
                             "points; must match the trained model)")
    parser.add_argument("--curve-points", type=int, default=None,
                        help="override bspline.curve_points (decoded density)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N records (debugging aid)")
    parser.add_argument("--max-fit-rmse-m", type=float,
                        default=MAX_FIT_RMSE_M_DEFAULT,
                        help="reject a sample when the RMS refit error exceeds "
                             "this many meters (default: 0.25)")
    parser.add_argument("--max-goal-gap-m", type=float,
                        default=MAX_GOAL_GAP_M_DEFAULT,
                        help="reject a sample when the route goal is further "
                             "than this from the executed trajectory end "
                             "(default: 5.0)")
    parser.add_argument("--max-report-samples", type=int,
                        default=MAX_REPORT_SAMPLES_DEFAULT,
                        help="cap on the skipped-sample list in the report "
                             "(default: 300)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg_path = _resolve(args.config)
    cfg = load_config(cfg_path)

    # ---- C (control points) and the curve density come from the config ----
    c_cfg = _cfg_get(cfg, "bspline.num_controls")
    h_cfg = _cfg_get(cfg, "bspline.curve_points")
    configure_codec(args.num_controls if args.num_controls is not None else c_cfg,
                    args.curve_points if args.curve_points is not None
                    else h_cfg)

    root, out, knots_path, knots_source = resolve_paths(args, cfg)

    print("=" * 78, flush=True)
    print("CARLA v1 cleaning", flush=True)
    print("  repo root    : %s" % REPO_ROOT, flush=True)
    print("  root (read)  : %s" % root, flush=True)
    print("  out (write)  : %s" % out, flush=True)
    print("  config       : %s" % cfg_path, flush=True)
    print("  num_controls : %d (degree %d, curve_points %d)"
          % (NUM_CONTROLS, DEGREE, CURVE_POINTS), flush=True)
    print("  knots        : %s (%s)" % (knots_path, knots_source), flush=True)
    print("=" * 78, flush=True)

    if not root.exists():
        print("[error] raw root does not exist: %s" % root, flush=True)
        return 2

    knots = None
    if knots_path.exists():
        knots = np.asarray(np.load(knots_path), dtype=np.float64).reshape(-1)
    if knots is None or knots.shape[0] != EXPECTED_KNOTS_LEN:
        knots = default_knots(NUM_CONTROLS, DEGREE)
        print("[warn] knots vector regenerated for num_controls=%d "
              "(clamped uniform, len=%d)" % (NUM_CONTROLS, knots.shape[0]),
              flush=True)

    params = np.linspace(0.0, 1.0, CURVE_POINTS)
    basis = bspline_basis_matrix(knots, NUM_CONTROLS, DEGREE, params)

    samples_path = root / "samples.jsonl"
    if not samples_path.exists():
        print("[error] samples.jsonl not found under %s" % root, flush=True)
        return 2

    episodes, episodes_error = load_episodes(root / "episodes.jsonl")
    print("[episodes] loaded %d episodes%s"
          % (len(episodes),
             "" if episodes_error is None else " (warning: %s)" % episodes_error),
          flush=True)

    # ---- scan + clean ---------------------------------------------------
    split_samples: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict(
        (split, []) for split in SPLITS)
    split_entries: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict(
        (split, []) for split in SPLITS)
    skipped: List[Dict[str, Any]] = []
    reason_hist: Counter = Counter()
    reason_detail_hist: Counter = Counter()
    scanned = 0
    n_existing = 0
    n_missing = 0
    n_records_skipped = 0
    accepted_total = 0
    ep_splits: Dict[int, set] = defaultdict(set)
    all_metric_lists = {
        "fit_rmse_meter": [],
        "fit_max_error_meter": [],
        "goal_to_executed_endpoint_meter": [],
    }
    refit_rejected_metrics = {
        "fit_rmse_meter": [],
        "fit_max_error_meter": [],
        "goal_to_executed_endpoint_meter": [],
    }

    print("[scan] reading %s" % samples_path, flush=True)
    try:
        handle = samples_path.open("r", encoding="utf-8")
    except Exception as exc:
        print("[error] cannot open samples.jsonl: %s" % _clean_message(exc),
              flush=True)
        return 2

    with handle:
        for lineno, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            if args.limit is not None and scanned >= args.limit:
                break
            scanned += 1

            meta: Dict[str, Any] = {
                "line": lineno,
                "sample_id": None,
                "episode_id": None,
                "town": None,
                "split": None,
                "file": None,
            }
            try:
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise SkipSample("invalid_record:not_a_json_object")
                for key in ("sample_id", "episode_id", "town", "split", "file"):
                    if key in record and record[key] is not None:
                        meta[key] = record[key]

                # existing count is measured independently of later failures
                raw_file = record.get("file")
                if isinstance(raw_file, str) and raw_file.strip():
                    probe = Path(raw_file.strip())
                    if not probe.is_absolute():
                        probe = root / probe
                    if probe.exists():
                        n_existing += 1
                    else:
                        n_missing += 1

                result = process_record(record, root, knots, basis,
                                        args.max_fit_rmse_m,
                                        args.max_goal_gap_m)
            except SkipSample as exc:
                reason = exc.reason
                if reason.startswith("npz_missing"):
                    pass  # already counted above
                result = {"accepted": False, "reason": reason, "stats": None}
            except Exception as exc:  # never let a single bad sample abort
                result = {"accepted": False, "reason": _exception_reason(exc),
                          "stats": None}

            if result["accepted"]:
                accepted_total += 1
                info = result["meta"]
                split = info["split"]
                distance_meta = result["stats"]
                data = result["data"]
                split_samples[split].append(data)
                split_entries[split].append({
                    "sample_id": info["sample_id"],
                    "episode_id": info["episode_id"],
                    "town": info["town"],
                    "split": split,
                    "file": info["file"],
                    "fit_rmse_meter": distance_meta["fit_rmse_meter"],
                    "fit_max_error_meter": distance_meta["fit_max_error_meter"],
                    "goal_to_executed_endpoint_meter":
                        distance_meta["goal_to_executed_endpoint_meter"],
                })
                ep_splits[int(info["episode_id"])].add(split)
                for key in all_metric_lists:
                    all_metric_lists[key].append(float(distance_meta[key]))
            else:
                n_records_skipped += 1
                reason = str(result.get("reason") or "unknown")
                reason_hist[_reason_key(reason)] += 1
                reason_detail_hist[reason] += 1
                stats = result.get("stats") or {}
                if reason == "control_refit_error" and stats:
                    for key in refit_rejected_metrics:
                        refit_rejected_metrics[key].append(float(stats[key]))
                skip_row = dict(meta)
                skip_row["reason"] = reason
                if stats:
                    skip_row["fit_rmse_meter"] = float(stats["fit_rmse_meter"])
                    skip_row["fit_max_error_meter"] = float(
                        stats["fit_max_error_meter"])
                    skip_row["goal_to_executed_endpoint_meter"] = float(
                        stats["goal_to_executed_endpoint_meter"])
                skipped.append(skip_row)

            if scanned % PROGRESS_EVERY == 0:
                print("[scan] %d records | accepted=%d existing_npz=%d "
                      "missing_npz=%d skipped=%d"
                      % (scanned, accepted_total, n_existing, n_missing,
                         n_records_skipped), flush=True)

    print("[scan] done: %d records | accepted=%d existing_npz=%d missing_npz=%d "
          "skipped=%d" % (scanned, accepted_total, n_existing, n_missing,
                          n_records_skipped), flush=True)

    # ---- assemble + write arrays ---------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    written_shapes: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for split in SPLITS:
        written_shapes[split] = write_split_arrays(out, split,
                                                   split_samples[split])
        print("[write] %-5s N=%d %s" % (split, len(split_samples[split]),
                                        written_shapes[split]), flush=True)

    manifest_entries = write_manifest(out, split_entries)

    per_split_counts = {split: len(split_entries[split]) for split in SPLITS}
    unique_episodes = {
        split: len({int(e["episode_id"]) for e in split_entries[split]})
        for split in SPLITS
    }
    all_split_episodes = {
        split: sorted({int(e["episode_id"]) for e in split_entries[split]})
        for split in SPLITS
    }

    leakage = sorted(ep for ep, splits in ep_splits.items() if len(splits) > 1)
    leakage_detail = [
        {"episode_id": int(ep), "splits": sorted(ep_splits[ep])}
        for ep in leakage
    ]

    # episode-json cross-check (informational only, never invalidates a sample)
    episode_conflicts = []
    for entry in (e for split in SPLITS for e in split_entries[split]):
        ep = int(entry["episode_id"])
        ep_rec = episodes.get(ep)
        if ep_rec is not None and ep_rec.get("split") not in (None, entry["split"]):
            episode_conflicts.append({
                "episode_id": ep,
                "episodes_jsonl_split": ep_rec.get("split"),
                "sample_split": entry["split"],
            })
            if len(episode_conflicts) >= 50:
                break

    report = {
        "root": str(root),
        "out": str(out),
        "config": str(cfg_path),
        "knots_path": str(knots_path),
        "knots_source": knots_source,
        "samples_jsonl": str(samples_path),
        "episodes_jsonl": str(root / "episodes.jsonl"),
        "episodes_jsonl_loaded": len(episodes),
        "episodes_jsonl_warning": episodes_error,
        "scanned_records": int(scanned),
        "manifest_entries": int(manifest_entries),
        "existing_npz": int(n_existing),
        "missing_npz": int(n_missing),
        "valid_samples": int(accepted_total),
        "invalid_count": int(n_records_skipped),
        "reason_histogram": dict(sorted(reason_hist.items(),
                                        key=lambda kv: (-kv[1], kv[0]))),
        "reason_detail_histogram": dict(
            sorted(reason_detail_hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        "per_split_counts": per_split_counts,
        "unique_episodes_per_split": unique_episodes,
        "episode_ids_per_split": all_split_episodes,
        "fit_rmse_meter": _metric_stats(all_metric_lists["fit_rmse_meter"]),
        "fit_max_error_meter": _metric_stats(
            all_metric_lists["fit_max_error_meter"]),
        "goal_to_executed_endpoint_meter": _metric_stats(
            all_metric_lists["goal_to_executed_endpoint_meter"]),
        "control_refit_rejected": {
            key: _metric_stats(values)
            for key, values in refit_rejected_metrics.items()
        },
        "split_leakage": leakage_detail,
        "split_leakage_episode_ids": [int(ep) for ep in leakage],
        "episode_split_conflicts": episode_conflicts,
        "skipped_total": int(n_records_skipped),
        "skipped_samples": skipped[:max(0, int(args.max_report_samples))],
        "array_shapes": {split: {name: list(shape) for name, shape
                                 in written_shapes[split].items()}
                         for split in SPLITS},
    }

    report_path = out / "cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # ---- compact stdout summary -----------------------------------------
    def _fmt(stats: Dict[str, Any]) -> str:
        if not stats or stats.get("count", 0) == 0:
            return "n=0"
        return ("n=%d p50=%.6f p95=%.6f max=%.6f"
                % (stats["count"], stats["p50"], stats["p95"], stats["max"]))

    print("", flush=True)
    print("=" * 78, flush=True)
    print("CLEANING SUMMARY", flush=True)
    print("-" * 78, flush=True)
    print("root                       : %s" % root, flush=True)
    print("out                        : %s" % out, flush=True)
    print("scanned_records            : %d" % scanned, flush=True)
    print("manifest_entries           : %d" % manifest_entries, flush=True)
    print("valid_samples              : %d" % accepted_total, flush=True)
    print("invalid_count              : %d" % n_records_skipped, flush=True)
    print("existing_npz               : %d" % n_existing, flush=True)
    print("missing_npz                : %d" % n_missing, flush=True)
    print("per-split counts           : train=%d val=%d test=%d"
          % (per_split_counts["train"], per_split_counts["val"],
             per_split_counts["test"]), flush=True)
    print("unique episodes per split  : train=%d val=%d test=%d"
          % (unique_episodes["train"], unique_episodes["val"],
             unique_episodes["test"]), flush=True)
    print("refit RMSE (m)             : %s"
          % _fmt(report["fit_rmse_meter"]), flush=True)
    print("refit max error (m)        : %s"
          % _fmt(report["fit_max_error_meter"]), flush=True)
    print("goal vs executed end (m)   : %s"
          % _fmt(report["goal_to_executed_endpoint_meter"]), flush=True)
    if refit_rejected_metrics["fit_max_error_meter"]:
        print("control_refit rejects (m)  : %s"
              % _fmt(report["control_refit_rejected"]["fit_max_error_meter"]),
              flush=True)
    print("split leakage              : %s"
          % ("none" if not leakage else "episodes=%s" % leakage), flush=True)
    print("reason histogram           : %s" % report["reason_histogram"],
          flush=True)
    if episode_conflicts:
        print("episode split conflicts    : %d (first: %s)"
              % (len(episode_conflicts), episode_conflicts[0]), flush=True)
    if skipped:
        show = skipped[:min(5, len(skipped))]
        print("first skipped samples      :", flush=True)
        for row in show:
            print("  line=%s split=%s sample_id=%s file=%s reason=%s"
                  % (row.get("line"), row.get("split"), row.get("sample_id"),
                     row.get("file"), row.get("reason")), flush=True)
    print("report                     : %s" % report_path, flush=True)
    print("manifest                   : %s" % (out / "clean_manifest.jsonl"),
          flush=True)
    print("=" * 78, flush=True)

    if accepted_total < 4000:
        print("[warning] valid_samples=%d is below 4000; investigate."
              % accepted_total, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
