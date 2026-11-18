"""DMA (Marie et al. 2016), equations 3--5 and Algorithm 4.

Continuous segment predicate uses the paper's Lipschitz exclusion steps,
with exact disk/segment interval coverage for equality/stagnation cases.
KDTree evaluates distance to background exactly instead of the conjectured
four-projection shortcut (Algorithm 2). No FDMA rounding or postprocessing.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


class DeltaMedialAxis:
    """Extract a delta-filtered medial axis from a 2D free-space mask.

    ``True`` means free. Distances and ``delta`` are in pixels. The image
    exterior is treated as occupied. ``distance`` and ``projection`` include
    the one-pixel padding; the returned skeleton has the original shape.
    """

    def __init__(self, free):
        free = np.asarray(free, dtype=bool)
        if free.ndim != 2:
            raise ValueError("free must be a 2D binary array")
        self.free = np.pad(free, 1, constant_values=False)
        self.distance, self.projection = distance_transform_edt(
            self.free, return_indices=True
        )
        self.background = np.argwhere(~self.free)
        self.tree = cKDTree(self.background)
        self.cache = {}

    def _coverage(self, a, b, delta):
        """Exact continuous predicate max D(z) > delta, without sampling.

        Each background pixel's closed delta disk covers an interval of the
        segment. A gap in their union is precisely a point with D(z)>delta.
        Only disks within L/2+delta of the segment midpoint can intersect it.
        """
        v = b - a
        length = np.linalg.norm(v)
        u = v / length
        ids = self.tree.query_ball_point((a + b) / 2, length / 2 + delta + 1e-10)
        q = self.background[ids] - a
        t = q @ u
        perp2 = np.maximum(0.0, np.sum(q * q, axis=1) - t * t)
        keep = perp2 <= delta * delta
        t = t[keep]
        half = np.sqrt(np.maximum(0.0, delta * delta - perp2[keep]))
        left, right = t - half, t + half
        end = 0.0
        for i in np.argsort(left):
            if right[i] < 0 or left[i] > length:
                continue
            if left[i] > end + 1e-10:
                return True
            end = max(end, right[i])
            if end >= length - 1e-10:
                return False
        return end < length - 1e-10

    def test_segment(self, a, b, delta):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = np.linalg.norm(b - a)
        if length <= 2 * delta:
            return False
        # Algorithm 3: expand from midpoint by (delta-D(z))/L.
        # Exact nearest-background queries replace Algorithm 2's conjecture.
        lo = hi = 0.5
        for _ in range(80):
            z = a[None, :] + np.array([lo, hi])[:, None] * (b - a)
            d = self.tree.query(z)[0]
            if np.any(d > delta + 1e-10):
                return True
            step = (delta - d) / length
            if np.min(step) < 1e-12:
                break
            lo -= step[0]
            hi += step[1]
            if lo * length < delta and (1 - hi) * length < delta:
                return False
        return self._coverage(a, b, delta)

    def extract(self, delta=3.0):
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive (pixels)")
        out = np.zeros_like(self.free)
        p = self.projection
        for dr, dc in ((1, 0), (0, 1)):
            pairs = self.free[: -dr or None, : -dc or None] & self.free[dr:, dc:]
            rr, cc = np.nonzero(pairs)
            aa = p[:, rr, cc].T
            bb = p[:, rr + dr, cc + dc].T
            keep = np.sum((aa - bb) ** 2, axis=1) > (2 * delta) ** 2
            for r, c, a, b in zip(rr[keep], cc[keep], aa[keep], bb[keep]):
                key = (tuple(sorted((tuple(a), tuple(b)))), float(delta))
                if key not in self.cache:
                    self.cache[key] = self.test_segment(a, b, delta)
                if not self.cache[key]:
                    continue
                m = np.array([r + dr / 2, c + dc / 2])
                da, db = np.sum((m - a) ** 2), np.sum((m - b) ** 2)
                if da >= db:
                    out[r, c] = True
                if db >= da:
                    out[r + dr, c + dc] = True
        return out[1:-1, 1:-1]
