#!/usr/bin/env python
"""00_build_processed.py - carla_full_160_256 -> processed cache contract.

Bridges data/carla_full_160_256 (scene/task layout, see its DATASET_REPORT.md)
to the FIXED per-split snapshot consumed by src/datasets/carla_spline_dataset.py
(see docs/CARLA_BSPLINE_PIPELINE_SPEC.md).

This script ONLY writes the "raw -> processed" part of that contract; the
candidate / topology / ellipse labels are built afterwards by the unchanged
scripts/data/carla/01_build_candidates.py and
scripts/data/carla/02_build_ellipse_labels.py.

Scale note (IMPORTANT): the 160 m windows map onto scene [-1, 1], i.e.
1 scene unit = 80 m, not the 40 m of the older carla_v1 crop.  Every geometric
constant in this repository is expressed in SCENE units on a fixed 256^2 grid,
so the NUMERIC geometry (skeleton, candidates, ellipse cells) is unchanged; only
data.scene_to_meter (a pure reporting / log constant) differs.  That is NOT a
reason to rescale the config knobs.

Split policy: the generator used an 80 m window stride with 160 m windows, so
neighbouring scenes overlap by 50 %.  A random scene split would leak, and
inside a town the overlap graph is a single connected component, so whole TOWNS
are held out (see DEFAULT_SPLIT_BY_TOWN).

Outputs per split (N = split size, C = bspline.num_controls,
H = bspline.curve_points):

    conditions.npy   [N,2,2]        f32   [start, goal]         (scene)
    control_gt.npy   [N,C,2]        f32   endpoint-constrained controls
    curve_gt.npy     [N,H,2]        f32   trajectory_128        (scene)
    occupancy.npy    [N,256,256]    u8    CANONICAL (flipud) occupancy
    episode_id.npy   [N]            i64   scene_id
    sample_id.npy    [N]            i64   task_id

plus clean_manifest.jsonl and build_processed_report.json.

The control labels are RE-FIT here from trajectory_128 with the very same
endpoint-constrained least-squares projection the network decodes with, which
keeps the repository invariant decode(control_gt) ~= curve_gt exactly.  The
dataset's own bspline_controls are only used as a cross-check.

    python scripts/data/carla_full/00_build_processed.py --source data/carla_full_160_256 --out data/carla_processed_160
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.geometry.bspline import (  # noqa: E402
    BSplineCodec,
    default_knots,
    numpy_fit_curve_to_controls,
)
from src.utils.config import curve_points, load_config, num_controls  # noqa: E402

SPLITS: Tuple[str, ...] = ("train", "val", "test")
OCC_RES = 256
DEGREE = 3

#: Town -> split.  Whole towns are held out; adjacent windows inside a town
#: overlap by 50 %, so scenes are never split within a town.
DEFAULT_SPLIT_BY_TOWN: Dict[str, str] = {
    "Town04": "train",
    "Town03": "train",
    "Town05": "val",
    "Town01": "test",
    "Town02": "test",
    "Town10HD": "test",
}

PROGRESS_EVERY = 250


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve(path_like) -> Path:
    p = Path(str(path_like)).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _norm_town(town: str) -> str:
    """'Town10HD_Opt' / 'Town10HD' -> 'Town10HD' (case/whitespace tolerant)."""
    t = str(town).strip()
    if t.lower().endswith("_opt"):
        t = t[: -len("_opt")]
    return t


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _scene_to_pixel(xy: np.ndarray) -> np.ndarray:
    """Scene [-1,1] -> canonical occupancy cell index on the 256^2 grid."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return np.floor((xy + 1.0) * (OCC_RES / 2.0)).astype(np.int64)


def _cell_free(occ: np.ndarray, xy: np.ndarray) -> bool:
    """True when every scene point lands on a free (0) canonical cell."""
    idx = _scene_to_pixel(xy)
    if (idx < 0).any() or (idx >= OCC_RES).any():
        return False
    return not bool((occ[idx[:, 1], idx[:, 0]] != 0).any())


def _solve_knots(bs_cfg: dict, c: int, degree: int) -> np.ndarray:
    raw = bs_cfg.get("knots", "auto")
    if raw is None or str(raw).strip().lower() in ("", "auto", "none", "null"):
        return default_knots(c, degree)
    path = _resolve(raw)
    if not path.exists():
        print("[knots] %s not found; using default_knots(%d,%d)"
              % (path, c, degree), flush=True)
        return default_knots(c, degree)
    knots = np.load(path).astype(np.float64).reshape(-1)
    want = c + degree + 1
    if knots.shape[0] != want:
        print("[knots] %s has %d entries, expected %d; using default_knots"
              % (path, knots.shape[0], want), flush=True)
        return default_knots(c, degree)
    return knots


# ---------------------------------------------------------------------------
# module-level source binding (workers-free, single process is fast enough)
# ---------------------------------------------------------------------------

_SOURCE: Path = Path(".")
_CODEC_CACHE: Dict[Tuple[float, int, int, int], BSplineCodec] = {}


def _resolve_from_source(rel: str) -> Path:
    return _SOURCE / str(rel)


def _scene_npz(scene_id: int) -> Path:
    return _SOURCE / "scenes" / ("scene_%04d.npz" % int(scene_id))


def _codec(knots: np.ndarray, c: int, h: int, degree: int) -> BSplineCodec:
    key = (float(knots.sum()), int(c), int(h), int(degree))
    if key not in _CODEC_CACHE:
        _CODEC_CACHE[key] = BSplineCodec(degree=degree, num_controls=c,
                                         curve_points=h, knots=knots)
    return _CODEC_CACHE[key]


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------


def build_split(rows: List[dict], scenes: Dict[int, dict], c: int, h: int,
                degree: int, knots: np.ndarray, fit: bool, split: str = "",
                limit: Optional[int] = None):
    import torch

    n = len(rows) if limit is None else min(int(limit), len(rows))
    cond = np.zeros((n, 2, 2), np.float32)
    ctrl = np.zeros((n, c, 2), np.float32)
    curve = np.zeros((n, h, 2), np.float32)
    occ = np.zeros((n, OCC_RES, OCC_RES), np.uint8)
    ep = np.zeros(n, np.int64)
    sid = np.zeros(n, np.int64)

    occ_cache: Dict[int, np.ndarray] = {}
    manifest: List[dict] = []
    fit_rmse: List[float] = []
    fit_max: List[float] = []
    ctrl_diff: List[float] = []
    n_cond_bad = 0
    n_traj_bad = 0
    n_zero = 0
    codec = _codec(knots, c, h, degree)
    t0 = time.time()

    for i, row in enumerate(rows[:n]):
        scene_id = int(row["scene_id"])
        with np.load(_resolve_from_source(row["file"])) as z:
            start = np.asarray(z["start"], np.float64).reshape(2)
            goal = np.asarray(z["goal"], np.float64).reshape(2)
            traj = np.asarray(z["trajectory_128"], np.float64).reshape(-1, 2)
            ds_ctrl = np.asarray(z["bspline_controls"], np.float64).reshape(-1, 2)

        if scene_id not in occ_cache:
            with np.load(_scene_npz(scene_id)) as z:
                # raw row 0 <-> scene_y = +1; canonical is row 0 <-> scene_y = -1
                occ_cache[scene_id] = np.flipud(z["occupancy_train"]).copy()
        occ_i = occ_cache[scene_id]

        cond[i] = np.stack([start, goal]).astype(np.float32)
        curve[i] = traj.astype(np.float32)
        occ[i] = occ_i
        ep[i] = scene_id
        sid[i] = int(row.get("task_id", i))

        if fit:
            q = numpy_fit_curve_to_controls(knots, c, degree, traj, start, goal, h)
        else:
            q = ds_ctrl.copy()
        ctrl[i] = q.astype(np.float32)

        if float(np.linalg.norm(goal - start)) <= 0.0:
            n_zero += 1
        if not _cell_free(occ_i, cond[i]):
            n_cond_bad += 1
        if not _cell_free(occ_i, traj):
            n_traj_bad += 1

        with torch.no_grad():
            rec = codec.decode_controls(
                torch.as_tensor(q[None], dtype=torch.float32))[0].numpy()
        err = np.linalg.norm(rec.astype(np.float64) - traj, axis=1)
        fit_rmse.append(float(np.sqrt((err ** 2).mean())))
        fit_max.append(float(err.max()))
        ctrl_diff.append(float(np.abs(q - ds_ctrl).max()))

        manifest.append(dict(
            sample_id=int(sid[i]), episode_id=int(scene_id),
            split=str(split),
            scene_id=int(scene_id), task_id=int(sid[i]),
            town=str(scenes[scene_id].get("town", "")),
            tier=str(scenes[scene_id].get("tier", "")),
            file=str(row["file"]), index_in_split=i,
            fit_rmse_scene=fit_rmse[-1], fit_max_scene=fit_max[-1],
        ))
        if (i + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            print("      %d/%d  %.0fs (%.1f ms/sample)"
                  % (i + 1, n, el, el / (i + 1) * 1000.0), flush=True)

    meta = dict(
        n=int(n),
        cond_not_free=int(n_cond_bad),
        traj_not_free=int(n_traj_bad),
        zero_length=int(n_zero),
        fit_rmse_scene=float(np.mean(fit_rmse)) if fit_rmse else None,
        fit_max_scene=float(np.max(fit_max)) if fit_max else None,
        ctrl_max_diff_vs_dataset=float(np.max(ctrl_diff)) if ctrl_diff else None,
        seconds=float(time.time() - t0),
    )
    arrs = dict(cond=cond, ctrl=ctrl, curve=curve, occ=occ, ep=ep, sid=sid)
    return arrs, meta, manifest


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    global _SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/carla_full_160_256")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="configs/config_160.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-fit", dest="fit", action="store_false", default=True,
                    help="use the dataset bspline_controls instead of re-fitting")
    args = ap.parse_args()

    cfg = load_config(args.config)
    C = num_controls(cfg)
    H = curve_points(cfg)
    bs_cfg = cfg.get("bspline") or {}
    degree = int(bs_cfg.get("degree", DEGREE))
    knots = _solve_knots(bs_cfg, C, degree)
    data_cfg = cfg.get("data") or {}
    split_by_town = dict(DEFAULT_SPLIT_BY_TOWN)
    split_by_town.update(data_cfg.get("split_by_town") or {})
    meters = float(data_cfg.get("scene_to_meter", 80.0))

    src = _resolve(args.source)
    _SOURCE = src
    out = _resolve(args.out or data_cfg.get("processed_root",
                                            "data/carla_processed_160"))
    if not src.is_dir():
        print("[error] source not found: %s" % src)
        return 2

    print("[build] source   : %s" % src)
    print("[build] out      : %s" % out)
    print("[build] controls : C=%d  curve_points=%d  degree=%d  refit=%s"
          % (C, H, degree, args.fit))
    print("[build] scene_to_meter = %.1f (reporting only)" % meters)

    scene_rows = _read_jsonl(src / "scenes.jsonl")
    scenes = {int(s["scene_id"]): s for s in scene_rows}
    tasks = _read_jsonl(src / "tasks.jsonl")

    unknown = sorted({_norm_town(t.get("town", "")) for t in tasks}
                     - set(split_by_town))
    if unknown:
        print("[error] no split assigned for town(s): %s" % ", ".join(unknown))
        print("        add them to data.split_by_town in %s" % args.config)
        return 2

    by_split: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    for t in tasks:
        by_split[split_by_town[_norm_town(t.get("town", ""))]].append(t)

    out.mkdir(parents=True, exist_ok=True)
    report = dict(source=str(src), out=str(out), config=args.config,
                  num_controls=int(C), curve_points=int(H), degree=int(degree),
                  refit_controls=bool(args.fit), scene_to_meter=meters,
                  split_by_town=split_by_town, per_split={},
                  limit=args.limit)
    manifest_all: List[dict] = []

    for split in SPLITS:
        rows = by_split[split]
        print("\n[%s] scenes=%d tasks=%d"
              % (split, len({r["scene_id"] for r in rows}), len(rows)), flush=True)
        arrs, meta, man = build_split(rows, scenes, C, H, degree, knots,
                                      args.fit, split=split, limit=args.limit)
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "conditions.npy", arrs["cond"])
        np.save(d / "control_gt.npy", arrs["ctrl"])
        np.save(d / "curve_gt.npy", arrs["curve"])
        np.save(d / "occupancy.npy", arrs["occ"])
        np.save(d / "episode_id.npy", arrs["ep"])
        np.save(d / "sample_id.npy", arrs["sid"])
        meta["towns"] = dict(Counter(m["town"] for m in man))
        meta["scenes"] = len({m["scene_id"] for m in man})
        report["per_split"][split] = meta
        manifest_all.extend(man)
        print("[%s] n=%d scenes=%d cond_not_free=%d traj_not_free=%d "
              "fit_rmse(max)=%.4f m fit_max=%.4f m %.0fs"
              % (split, meta["n"], meta["scenes"], meta["cond_not_free"],
                 meta["traj_not_free"],
                 (meta["fit_rmse_scene"] or 0.0) * meters,
                 (meta["fit_max_scene"] or 0.0) * meters, meta["seconds"]),
              flush=True)

    with (out / "clean_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for rec in manifest_all:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    report["total"] = int(sum(v["n"] for v in report["per_split"].values()))
    report["cond_not_free"] = int(sum(v["cond_not_free"]
                                      for v in report["per_split"].values()))
    report["traj_not_free"] = int(sum(v["traj_not_free"]
                                      for v in report["per_split"].values()))
    with (out / "build_processed_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\nDONE %s  total=%d  cond_not_free=%d  traj_not_free=%d"
          % (out, report["total"], report["cond_not_free"],
             report["traj_not_free"]))
    print("next: python scripts/data/carla/01_build_candidates.py --processed %s"
          % out)
    return 0 if (report["cond_not_free"] == 0 and report["traj_not_free"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
