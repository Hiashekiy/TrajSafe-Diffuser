"""Reference-trajectory preprocessing (design section 3).

The reference is the INTERMEDIATE CLEAN STATE ``x0`` of the reverse diffusion,
decoded to a polyline in project scene coordinates.  It is a soft guide only:
it may cross obstacles, double back or self-intersect, and nothing here ever
rejects a path because of that -- only *degenerate* references (NaN, empty,
mostly outside the crop, zero length) are marked invalid so the search can fall
back to an unguided Focal A*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ReferencePath", "prepare_reference_path", "resample_polyline"]


@dataclass
class ReferencePath:
    points: np.ndarray            # [n, 2] continuous cell coords (col, row)
    cum_m: np.ndarray             # [n] arc length in metres
    length_m: float
    valid: bool
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)

    # -- derived ------------------------------------------------------------
    @property
    def p0(self) -> np.ndarray:
        return self.points[:-1]

    @property
    def p1(self) -> np.ndarray:
        return self.points[1:]

    @property
    def s0(self) -> np.ndarray:
        return self.cum_m[:-1]

    @property
    def s1(self) -> np.ndarray:
        return self.cum_m[1:]

    @property
    def n_segments(self) -> int:
        return max(0, len(self.points) - 1)

    def sample(self, n: int) -> np.ndarray:
        """``n`` points uniformly spaced by arc length (cell coords)."""
        if self.n_segments == 0:
            return np.repeat(self.points[:1], int(n), axis=0)
        s = np.linspace(0.0, self.length_m, int(n))
        return np.stack([np.interp(s, self.cum_m, self.points[:, k])
                         for k in range(2)], axis=-1)


def _drop_duplicates(points: np.ndarray, min_dist: float) -> np.ndarray:
    if len(points) <= 2 or min_dist <= 0:
        return points
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) >= min_dist:
            keep.append(i)
    if keep[-1] != len(points) - 1:
        keep.append(len(points) - 1)
    return points[keep]


def resample_polyline(points: np.ndarray, cum: np.ndarray,
                      step: float) -> np.ndarray:
    """Uniform arc-length resampling; endpoints are preserved exactly."""
    length = float(cum[-1])
    if length <= 0.0:
        return points[:1].copy()
    n = max(2, int(np.ceil(length / max(step, 1e-9))) + 1)
    s = np.linspace(0.0, length, n)
    out = np.stack([np.interp(s, cum, points[:, k]) for k in range(2)], axis=-1)
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def _merge_collinear(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Drop interior points that continue the previous direction."""
    if len(points) <= 2:
        return points
    cos_tol = float(np.cos(np.deg2rad(angle_deg)))
    keep = [0]
    for i in range(1, len(points) - 1):
        a = points[i] - points[keep[-1]]
        b = points[i + 1] - points[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na <= 1e-12 or nb <= 1e-12:
            continue
        if float(np.dot(a, b) / (na * nb)) >= cos_tol:
            continue
        keep.append(i)
    keep.append(len(points) - 1)
    return points[keep]


def prepare_reference_path(points_world, spec, start_world=None, goal_world=None,
                           resample_step_m=None, dedup_m=None, min_length_m=None,
                           max_oob_fraction: float = 0.5,
                           merge_angle_deg: float = 5.0,
                           max_segments: int = 512) -> ReferencePath:
    """Convert the decoded ``x0`` polyline to cell coordinates and clean it up.

    The returned object is always usable: ``valid=False`` means the caller must
    run the unguided Focal A* instead (never a planning failure).
    """
    diag = {}
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return ReferencePath(np.zeros((2, 2)), np.zeros(2), 0.0, False,
                             "empty_or_shapeless", diag)
    diag["n_raw"] = int(len(pts))
    finite = np.isfinite(pts).all(axis=1)
    diag["n_nonfinite"] = int((~finite).sum())
    if not finite.all():
        if finite.sum() < 2:
            return ReferencePath(np.zeros((2, 2)), np.zeros(2), 0.0, False,
                                 "non_finite", diag)
        pts = pts[finite]

    cell = spec.world_to_cell(pts)

    # --- validity checks run on the MODEL output, before any correction ----
    # (endpoint correction would otherwise hide a degenerate / out-of-bounds
    # prediction behind two perfectly valid endpoints.)
    lo = np.array([-0.5, -0.5])
    hi = np.array([spec.width - 0.5, spec.height - 0.5])
    oob = ((cell < lo) | (cell > hi)).any(axis=1)
    diag["oob_fraction"] = float(oob.mean())
    if diag["oob_fraction"] > float(max_oob_fraction):
        return ReferencePath(cell[:2].copy(), np.zeros(2), 0.0, False,
                             "out_of_bounds", diag)
    cell = np.clip(cell, lo, hi)

    dedup_cells = float((dedup_m if dedup_m is not None
                         else 0.1 * spec.cell_m) / spec.cell_m)
    cell = _drop_duplicates(cell, dedup_cells)
    if len(cell) < 2:
        return ReferencePath(np.repeat(cell[:1], 2, axis=0), np.zeros(2), 0.0,
                             False, "degenerate_after_dedup", diag)

    seg = np.linalg.norm(np.diff(cell, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)]) * spec.cell_m
    diag["length_raw_m"] = float(cum[-1])
    if cum[-1] <= max(1e-6, min_length_m or 2.0 * spec.cell_m):
        return ReferencePath(cell, cum, float(cum[-1]), False,
                             "degenerate_length", diag)

    # --- endpoint correction (never rescale the whole polyline) ------------
    # the model's own endpoint error is recorded first, then p0/pN are pinned
    # to the planning start/goal.
    if start_world is not None and goal_world is not None:
        s_cell = spec.world_to_cell(np.asarray(start_world, dtype=np.float64))
        g_cell = spec.world_to_cell(np.asarray(goal_world, dtype=np.float64))
        diag["endpoint_error_m"] = float(
            (np.linalg.norm(cell[0] - s_cell)
             + np.linalg.norm(cell[-1] - g_cell)) * spec.cell_m)
    if start_world is not None:
        cell[0] = spec.world_to_cell(np.asarray(start_world, dtype=np.float64))
    if goal_world is not None:
        cell[-1] = spec.world_to_cell(np.asarray(goal_world, dtype=np.float64))
    cell = np.clip(cell, lo, hi)

    step = float(resample_step_m if resample_step_m is not None
                 else 2.0 * spec.cell_m)
    pts_cell = cell
    for _ in range(4):
        seg = np.linalg.norm(np.diff(pts_cell, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)]) * spec.cell_m
        pts_cell = resample_polyline(pts_cell, cum, step)
        pts_cell = _merge_collinear(pts_cell, merge_angle_deg)
        if len(pts_cell) - 1 <= int(max_segments):
            break
        step *= float(np.ceil((len(pts_cell) - 1) / float(max_segments)))
    seg = np.linalg.norm(np.diff(pts_cell, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)]) * spec.cell_m
    diag.update({"resample_step_m": step,
                 "n_after": int(len(pts_cell)),
                 "n_segments": int(len(pts_cell) - 1),
                 "length_resampled_m": float(cum[-1])})
    return ReferencePath(pts_cell, cum, float(cum[-1]), True, "", diag)
