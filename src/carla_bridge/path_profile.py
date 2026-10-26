"""Geometry-only arc-length processing (no alternate path generator)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Path:
    xy: np.ndarray
    s: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    reference_speed: np.ndarray = None

    @classmethod
    def build(cls, xy, spacing=0.2):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
            raise ValueError('INVALID_GEOMETRY')
        xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1)>1e-7]]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        if len(xy) < 3 or s[-1] < 1:
            raise ValueError('DEGENERATE_PATH')
        ss = np.linspace(0, s[-1], max(3, int(np.ceil(s[-1]/spacing))+1))
        pts = np.column_stack([np.interp(ss, s, xy[:, i]) for i in range(2)])
        tangent = np.gradient(pts, axis=0)
        yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        return cls(pts, ss, yaw, np.gradient(yaw, ss))

    def nearest(self, xy):
        return int(np.argmin(np.linalg.norm(self.xy-xy, axis=1)))

    def pose(self, s):
        return np.array([np.interp(s, self.s, self.xy[:, j]) for j in range(2)]), float(np.interp(s, self.s, self.yaw))
