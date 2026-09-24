"""Arc-length path post-processing and the longitudinal speed profile.

The model emits 128 scene-space samples that are NOT uniformly spaced in arc
length, so they are resampled before any control.  Everything downstream
(Pure Pursuit, the speed profile, the stop condition) then works on one
uniform arc-length table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = ["PathProfile", "resample_by_arc_length", "speed_profile"]


def resample_by_arc_length(xy: np.ndarray, spacing_m: float = 0.20):
    """Resample a polyline [N,2] to a uniform arc length.  Returns [M,2], [M]."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("polyline must be [N,2] with N >= 2, got %r" % (pts.shape,))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-9))
    pts = pts[keep]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if total <= 0.0:
        raise ValueError("degenerate polyline with zero length")
    n = max(int(np.floor(total / float(spacing_m))) + 1, 2)
    s_new = np.linspace(0.0, total, n)
    out = np.stack((np.interp(s_new, s, pts[:, 0]),
                    np.interp(s_new, s, pts[:, 1])), axis=1)
    return out, s_new


def _headings(xy: np.ndarray) -> np.ndarray:
    d = np.gradient(np.asarray(xy, dtype=np.float64), axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    return np.unwrap(ang)


def _curvature(heading: np.ndarray, s: np.ndarray) -> np.ndarray:
    ds = np.gradient(np.asarray(s, dtype=np.float64))
    ds[ds < 1e-9] = 1e-9
    kappa = np.gradient(np.unwrap(np.asarray(heading, dtype=np.float64))) / ds
    return np.abs(kappa)


def speed_profile(curvature: np.ndarray, s: np.ndarray, cfg: Dict) -> np.ndarray:
    """Curvature + braking limit, then a backward pass so braking is feasible."""
    curv = np.asarray(curvature, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    v_max = float(cfg.get("max_speed_mps", 3.0))
    v_min = float(cfg.get("min_speed_mps", 0.8))
    a_lat = float(cfg.get("a_lat_max_mps2", 1.5))
    a_brake = float(cfg.get("a_brake_mps2", 2.0))
    stop_radius = float(cfg.get("stop_radius_m", 0.6))

    remaining = float(s[-1]) - s
    v_curve = np.sqrt(a_lat / (curv + 1e-6))
    # 0.5 m of margin absorbs the controller's reaction lag before the goal
    v_goal = np.sqrt(2.0 * a_brake * np.clip(remaining - 0.5, 0.0, None))
    v = np.minimum(np.minimum(v_curve, v_goal), v_max)
    floor = np.where(remaining > stop_radius + 1.0, v_min, 0.0)
    v = np.maximum(v, floor)
    # backward feasibility pass: v[i]^2 <= v[i+1]^2 + 2 a ds
    for i in range(len(v) - 2, -1, -1):
        ds = float(s[i + 1] - s[i])
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_brake * ds))
    v[0] = min(v[0], v_max)
    return v


@dataclass
class PathProfile:
    """Uniform arc-length path with heading, curvature and reference speed."""

    xy: np.ndarray
    s: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    v_ref: np.ndarray

    @classmethod
    def from_world_curve(cls, xy_world: np.ndarray, controller_cfg: Dict) -> "PathProfile":
        spacing = float(controller_cfg.get("path_spacing_m", 0.20))
        xy, s = resample_by_arc_length(xy_world, spacing)
        heading = _headings(xy)
        curvature = _curvature(heading, s)
        v_ref = speed_profile(curvature, s, controller_cfg)
        return cls(xy=xy, s=s, heading=heading, curvature=curvature, v_ref=v_ref)

    # ---------------------------------------------------------------- query
    @property
    def total_length(self) -> float:
        return float(self.s[-1])

    def __len__(self) -> int:
        return int(len(self.s))

    def nearest_index(self, xy_world, from_index: int = 0) -> int:
        p = np.asarray(xy_world, dtype=np.float64).reshape(2)
        lo = max(int(from_index) - 2, 0)
        d = np.linalg.norm(self.xy[lo:] - p[None, :], axis=1)
        return int(lo + int(np.argmin(d)))

    def project(self, xy_world, from_index: int = 0):
        """Nearest segment projection.  Returns (index, s, lateral_error)."""
        p = np.asarray(xy_world, dtype=np.float64).reshape(2)
        i = self.nearest_index(p, from_index)
        j = min(i + 1, len(self.xy) - 1)
        k = max(i - 1, 0)
        best = None
        for a, b in ((i, j), (k, i)):
            pa = self.xy[a]
            pb = self.xy[b]
            seg = pb - pa
            denom = float(seg @ seg)
            if denom < 1e-12:
                continue
            t = float(np.clip((p - pa) @ seg / denom, 0.0, 1.0))
            foot = pa + t * seg
            dist = float(np.linalg.norm(p - foot))
            if best is None or dist < best[0]:
                direction = seg / np.sqrt(denom)
                right = np.array([-direction[1], direction[0]])
                lat = float((p - foot) @ right)
                s = float(self.s[a] + t * (self.s[b] - self.s[a]))
                best = (dist, a, s, lat)
        _, index, s, lat = best
        return index, s, lat

    def sample_at(self, s_query: float):
        """Interpolated (xy, heading, curvature, v_ref) at one arc length."""
        s = float(np.clip(s_query, 0.0, self.total_length))
        xy = np.array([np.interp(s, self.s, self.xy[:, 0]),
                       np.interp(s, self.s, self.xy[:, 1])])
        heading = float(np.interp(s, self.s, self.heading))
        curvature = float(np.interp(s, self.s, self.curvature))
        v_ref = float(np.interp(s, self.s, self.v_ref))
        return xy, heading, curvature, v_ref

    def describe(self) -> Dict[str, float]:
        return {
            "points": int(len(self.s)),
            "length_m": round(self.total_length, 3),
            "spacing_m": round(float(self.s[1] - self.s[0]), 4),
            "v_ref_max": round(float(self.v_ref.max()), 3),
            "v_ref_min_moving": round(float(self.v_ref[:-1].min()), 3),
            "curvature_max": round(float(self.curvature.max()), 5),
        }
