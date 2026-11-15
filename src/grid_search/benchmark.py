"""Case preparation, benchmark harness and aggregation.

One "case" is one planning problem: an occupancy bitmap, a start/goal in scene
coordinates and the reference polyline taken from the intermediate clean state
of the reverse diffusion.  ``prepare_case`` does everything that is shared by
all methods (safe grid, neighbour table, guidance field) and times each block
separately, so the report can split the search time from the guidance build
time and from the end-to-end cost.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field

import numpy as np

from .baselines import METHODS, run_method
from .diffusion_path import prepare_reference_path
from .grid_map import GridSpec, build_safe_grid
from .grid_neighbors import build_neighbor_table
from .path_validation import validate_path, world_collision_rate
from .progress_guidance import build_progress_guidance

__all__ = ["Case", "PreparedCase", "prepare_case", "evaluate_case",
           "run_suite", "aggregate", "write_csv", "DEFAULT_CONFIGS"]

DEFAULT_CONFIGS = (
    # (label, method, w, alpha_d, alpha_p)
    ("dijkstra", "dijkstra", 1.0, 1.0, 0.0),
    ("astar", "astar", 1.0, 1.0, 0.0),
    ("focal_clearance", "focal_clearance", 1.2, 1.0, 0.0),
    ("focal_index", "focal_index", 1.2, 1.0, 0.0),
    ("pdg_w1.0", "pdg", 1.0, 1.0, 1.0),
    ("pdg_w1.2", "pdg", 1.2, 1.0, 1.0),
    ("pdg_g1", "pdg", 1.2, 1.0, 0.0),
    ("pdg_g2", "pdg", 1.2, 1.0, 1.0),
)


@dataclass
class Case:
    case_id: str
    occupancy: np.ndarray
    spec: GridSpec
    start_world: np.ndarray
    goal_world: np.ndarray
    ref_world: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class PreparedCase:
    case: Case
    safe: object
    neighbors: object
    ref: object
    guidance: object
    start_idx: int
    goal_idx: int
    timings: dict
    notes: list


def prepare_case(case: Case, robot_radius_m: float = 0.8,
                 safety_margin_m: float = 0.3, already_inflated_cells: float = 0.0,
                 J: int = 3, ref_kwargs: dict = None) -> PreparedCase:
    t0 = time.perf_counter()
    safe = build_safe_grid(case.occupancy, case.spec,
                           robot_radius_m=robot_radius_m,
                           safety_margin_m=safety_margin_m,
                           already_inflated_cells=already_inflated_cells)
    neighbors = build_neighbor_table(safe.free, case.spec.cell_m)
    t_pre = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = prepare_reference_path(case.ref_world, case.spec,
                                 start_world=case.start_world,
                                 goal_world=case.goal_world,
                                 **(ref_kwargs or {}))
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    guidance = build_progress_guidance(ref, case.spec, J=J)
    t_gui = time.perf_counter() - t0

    start_idx = int(case.spec.cell_to_index(
        case.spec.world_to_cell(case.start_world)))
    goal_idx = int(case.spec.cell_to_index(
        case.spec.world_to_cell(case.goal_world)))
    timings = {"preprocess_ms": t_pre * 1000.0, "reference_ms": t_ref * 1000.0,
               "guidance_ms": t_gui * 1000.0}
    notes = list(safe.notes) + ["reference valid=%s (%s)"
                                % (ref.valid, ref.reason or "ok")]
    return PreparedCase(case, safe, neighbors, ref, guidance, start_idx,
                        goal_idx, timings, notes)


def evaluate_case(prep: PreparedCase, label: str, method: str, w: float,
                  alpha_d: float, alpha_p: float, sigma_m: float = None,
                  L_ref_m: float = None, lam: float = 0.0,
                  secondary: str = "index",
                  snapshots=(1, 10, 50, 100, 500)) -> dict:
    case = prep.case
    sigma = float(sigma_m if sigma_m is not None else 3.0 * case.spec.cell_m)
    L_ref = float(L_ref_m if L_ref_m is not None else
                  max(prep.ref.length_m, 1e-9))
    result = run_method(method, prep.safe, prep.start_idx, prep.goal_idx,
                        neighbors=prep.neighbors, guidance=prep.guidance, w=w,
                        alpha_d=alpha_d, alpha_p=alpha_p, sigma_m=sigma,
                        L_ref_m=L_ref, lam=lam, secondary=secondary,
                        snapshots=snapshots)
    t0 = time.perf_counter()
    check = validate_path(prep.safe, result.path, prep.start_idx, prep.goal_idx,
                          result.cost_m) if result.path is not None else {}
    t_val = (time.perf_counter() - t0) * 1000.0

    ref_world_rate = None
    if prep.ref.valid:
        dense = case.spec.cell_to_index(prep.ref.sample(256))
        dense = np.clip(dense, 0, case.spec.n_cells - 1)
        ref_world_rate = world_collision_rate(case.occupancy, case.spec, dense)
    rec = {
        "case_id": case.case_id,
        "label": label,
        "method": method,
        "w": float(w),
        "alpha_d": float(alpha_d),
        "alpha_p": float(alpha_p),
        "sigma_m": sigma,
        "L_ref_m": L_ref,
        "status": result.status,
        "success": int(result.status == "SUCCESS"),
        "expanded_nodes": result.stats.get("expanded_nodes", 0),
        "generated_nodes": result.stats.get("generated_nodes", 0),
        "reopened_nodes": result.stats.get("reopened_nodes", 0),
        "peak_open_size": result.stats.get("peak_open_size", 0),
        "search_time_ms": result.stats.get("search_time_ms", 0.0),
        "guidance_build_time_ms": prep.timings["guidance_ms"],
        "preprocess_time_ms": prep.timings["preprocess_ms"],
        "reference_time_ms": prep.timings["reference_ms"],
        "validation_time_ms": t_val,
        "path_cost_m": result.cost_m if result.status == "SUCCESS" else None,
        "search_cost_m": result.stats.get("search_cost_m"),
        "path_valid": int(bool(check.get("valid", False))),
        "path_cells": check.get("n_cells"),
        "turn_count": check.get("turn_count"),
        "clearance_min_m": check.get("clearance_min_m"),
        "reference_valid": int(bool(prep.ref.valid)),
        "reference_length_m": float(prep.ref.length_m or 0.0),
        "reference_collision_rate": ref_world_rate,
        "guidance_valid": int(bool(getattr(prep.guidance, "valid", False))),
        "guidance_coverage": float(getattr(prep.guidance, "coverage", 0.0)),
        "guidance_cover_frac": float(getattr(prep.guidance, "cover_frac", 0.0)),
        "guidance_segments": int(getattr(prep.guidance, "n_segments", 0)),
        "guidance_used": int(bool(result.guidance_used)),
        "lambda": float(lam),
        "guide_penalty_m": result.stats.get("guide_penalty_m"),
        "path_true_m": result.stats.get("path_true_m"),
    }
    rec.update({k: v for k, v in case.meta.items() if k not in rec})
    return {**rec, "_result": result, "_check": check}


def run_suite(cases, configs=DEFAULT_CONFIGS, robot_radius_m: float = 0.8,
              safety_margin_m: float = 0.3, already_inflated_cells: float = 0.0,
              J: int = 3, sigma_m: float = None, L_ref_m: float = None,
              progress=None) -> list:
    records = []
    for i, case in enumerate(cases):
        prep = prepare_case(case, robot_radius_m=robot_radius_m,
                            safety_margin_m=safety_margin_m,
                            already_inflated_cells=already_inflated_cells, J=J)
        for label, method, w, ad, ap in configs:
            rec = evaluate_case(prep, label, method, w, ad, ap,
                                sigma_m=sigma_m, L_ref_m=L_ref_m)
            records.append(rec)
        if progress is not None and (i + 1) % 25 == 0:
            progress(i + 1, len(cases))
    return records


def _stats(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if len(v) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "p95": None}
    return {"n": int(len(v)), "mean": float(v.mean()),
            "median": float(np.median(v)),
            "std": float(v.std(ddof=0)),
            "p95": float(np.percentile(v, 95))}


def aggregate(records, group_keys=("label",)):
    groups = {}
    for r in records:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = {k: v for k, v in zip(group_keys, key)}
        row["n_cases"] = len(rs)
        row["success_rate"] = float(np.mean([r["success"] for r in rs]))
        row["path_valid_rate"] = float(np.mean([r["path_valid"] for r in rs]))
        row["expanded_mean"] = _stats([r["expanded_nodes"] for r in rs])["mean"]
        row["expanded_median"] = _stats([r["expanded_nodes"] for r in rs])["median"]
        row["expanded_std"] = _stats([r["expanded_nodes"] for r in rs])["std"]
        t = _stats([r["search_time_ms"] for r in rs])
        row["search_ms_median"] = t["median"]
        row["search_ms_mean"] = t["mean"]
        row["search_ms_p95"] = t["p95"]
        row["guidance_ms_median"] = _stats(
            [r["guidance_build_time_ms"] for r in rs])["median"]
        cost = _stats([r["path_cost_m"] for r in rs if r["path_cost_m"]])
        row["cost_mean"] = cost["mean"]
        row["cost_median"] = cost["median"]
        row["cost_std"] = cost["std"]
        row["turn_mean"] = _stats([r["turn_count"] for r in rs if r["turn_count"]
                                   is not None])["mean"]
        row["clearance_min_mean"] = _stats(
            [r["clearance_min_m"] for r in rs if r["clearance_min_m"]
             is not None])["mean"]
        row["reopened_mean"] = _stats([r["reopened_nodes"] for r in rs])["mean"]
        row["peak_open_mean"] = _stats([r["peak_open_size"] for r in rs])["mean"]
        rows.append(row)
    return rows


def write_csv(records, path, drop_private=True):
    rows = []
    for r in records:
        row = {k: v for k, v in r.items()
               if not (drop_private and k.startswith("_"))}
        rows.append(row)
    if not rows:
        return 0
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    return len(rows)


