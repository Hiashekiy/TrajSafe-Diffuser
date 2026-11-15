"""Independent path verification (design sections 1.2, 5.5, 8.2).

The search result is NEVER trusted on its own: every edge is re-checked against
the safe grid, the cost is recomputed from scratch, and the geometric properties
reported in the experiment (length, turns, clearance) are derived here.
"""

from __future__ import annotations

import numpy as np

from .grid_map import SafeGrid
from .grid_neighbors import corner_cutting_violations

__all__ = ["validate_path", "path_cost_m", "turn_count", "world_collision_rate"]


def path_cost_m(safe: SafeGrid, path) -> float:
    spec = safe.spec
    rc = spec.index_to_rc(np.asarray(path, dtype=np.int64)).astype(np.float64)
    if len(rc) < 2:
        return 0.0
    step = np.linalg.norm(np.diff(rc, axis=0), axis=1)
    return float(step.sum() * spec.cell_m)


def turn_count(safe: SafeGrid, path) -> int:
    rc = safe.spec.index_to_rc(np.asarray(path, dtype=np.int64))
    if len(rc) < 3:
        return 0
    d = np.diff(rc, axis=0)
    return int((np.abs(np.diff(d, axis=0)).sum(axis=1) > 0).sum())


def validate_path(safe: SafeGrid, path, start_idx: int = None,
                  goal_idx: int = None, claimed_cost: float = None) -> dict:
    """Re-check a path from scratch.  Returns a dict with ``valid`` + details."""
    out = {"valid": False, "errors": []}
    if path is None or len(path) == 0:
        out["errors"].append("empty_path")
        return out
    path = np.asarray(path, dtype=np.int64)
    spec = safe.spec
    rc = spec.index_to_rc(path)
    h, w = spec.height, spec.width

    if (rc[:, 0] < 0).any() or (rc[:, 0] >= h).any() \
            or (rc[:, 1] < 0).any() or (rc[:, 1] >= w).any():
        out["errors"].append("out_of_bounds")
        return out
    blocked = int((~safe.free[rc[:, 0], rc[:, 1]]).sum())
    if blocked:
        out["errors"].append("blocked_cells:%d" % blocked)

    bad_edges = 0
    diagonal_edges = 0
    for a, b in zip(rc[:-1], rc[1:]):
        dr, dc = int(b[0] - a[0]), int(b[1] - a[1])
        if max(abs(dr), abs(dc)) != 1:
            bad_edges += 1
            continue
        if dr != 0 and dc != 0:
            diagonal_edges += 1
            if not safe.free[a[0] + dr, a[1]] or not safe.free[a[0], a[1] + dc]:
                bad_edges += 1
    if bad_edges:
        out["errors"].append("illegal_edges:%d" % bad_edges)
    if corner_cutting_violations(safe.free, rc):
        out["errors"].append("corner_cutting")

    if start_idx is not None and int(path[0]) != int(start_idx):
        out["errors"].append("start_mismatch")
    if goal_idx is not None and int(path[-1]) != int(goal_idx):
        out["errors"].append("goal_mismatch")

    cost = path_cost_m(safe, path)
    if claimed_cost is not None and abs(cost - float(claimed_cost)) > 1e-6:
        out["errors"].append("cost_mismatch:%.6f!=%.6f" % (cost, claimed_cost))

    clear = safe.clearance_at_index(path)
    out.update({
        "valid": not out["errors"],
        "n_cells": int(len(path)),
        "diagonal_edges": int(diagonal_edges),
        "path_cost_m": float(cost),
        "path_length_m": float(np.linalg.norm(
            np.diff(spec.index_to_rc(path).astype(np.float64), axis=0),
            axis=1).sum() * spec.cell_m) if len(path) > 1 else 0.0,
        "turn_count": turn_count(safe, path),
        "clearance_min_m": float(clear.min()) if len(path) else 0.0,
    })
    return out


def world_collision_rate(occupancy, spec, path) -> float:
    """Collision rate of the path vertices on the RAW (un-inflated) map.

    This mirrors ``evaluate.py::_collides`` so the number is comparable with the
    project's own trajectory collision metric.
    """
    occ = np.asarray(occupancy)
    rc = spec.index_to_rc(np.asarray(path, dtype=np.int64))
    ok = ((rc[:, 0] >= 0) & (rc[:, 0] < spec.height)
          & (rc[:, 1] >= 0) & (rc[:, 1] < spec.width))
    hit = np.ones(len(rc), dtype=bool)
    hit[ok] = occ[rc[ok, 0], rc[ok, 1]].astype(bool)
    return float(hit.mean())
