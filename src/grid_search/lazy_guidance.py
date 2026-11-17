"""Lazy guidance: compute ``H_D`` only for the nodes the search actually touches.

``build_progress_guidance`` precomputes the ``J`` nearest path-segment
projections for **every** cell (65536 x 128 segment tests), which profiling puts
at ~27 ms per sample — and the A* then touches only ~600 nodes.

:class:`LazyGuidance` keeps the same ``score()`` contract but computes
``d_j / u_j`` for the queried indices only (cached per index), so the field costs
proportional to the search, not to the grid.  Values are identical to the
precomputed field up to floating-point ties (verified in tests).
"""

from __future__ import annotations

import numpy as np

__all__ = ["LazyGuidance"]


class LazyGuidance:
    """Drop-in replacement for :class:`GuidanceField` (same ``score`` API)."""

    def __init__(self, ref, spec, J: int = 3, cover_m: float | None = None):
        self.cell_m = float(spec.cell_m)
        self.width = int(spec.width)
        self.height = int(spec.height)
        self.J = max(int(J), 1)
        self.valid = bool(getattr(ref, "valid", False)) and \
            int(getattr(ref, "n_segments", 0)) >= 1
        self.length_m = float(getattr(ref, "length_m", 0.0) or 0.0)
        self.reason = getattr(ref, "reason", None)
        self.cover_m = float(cover_m if cover_m is not None else 3.0 * self.cell_m)
        self.build_ms = 0.0
        self.n_segments = 0
        self._p0 = self._dx = self._den = self._s0 = self._s1 = None
        # dense lazy cache (per-cell arrays + a "computed" bitmap): the search
        # asks for large candidate sets, so a dict-of-tuples gather is too slow
        n = self.width * self.height
        self._d = np.zeros((n, self.J), dtype=np.float32)
        self._u = np.zeros((n, self.J), dtype=np.float32)
        self._has = np.zeros(n, dtype=bool)
        self._checked = np.zeros(n, dtype=bool)
        if self.valid:
            p0 = np.asarray(ref.p0, dtype=np.float64)
            p1 = np.asarray(ref.p1, dtype=np.float64)
            dx = p1 - p0
            self._p0 = p0
            self._dx = dx
            self._den = np.maximum((dx * dx).sum(axis=1), 1e-12)
            self._s0 = np.asarray(ref.s0, dtype=np.float64)
            self._s1 = np.asarray(ref.s1, dtype=np.float64)
            self.n_segments = int(len(p0))

    # ------------------------------------------------------------------ core
    def _values(self, idx: np.ndarray):
        """``d[n,J]`` (metres) and ``u[n,J]`` (0..1 progress) for cell indices."""
        idx = np.asarray(idx, dtype=np.int64)
        cols = (idx % self.width).astype(np.float64)
        rows = (idx // self.width).astype(np.float64)
        p0, dx, den = self._p0, self._dx, self._den
        rx = cols[:, None] - p0[None, :, 0]
        ry = rows[:, None] - p0[None, :, 1]
        t = (rx * dx[None, :, 0] + ry * dx[None, :, 1]) / den[None]
        np.clip(t, 0.0, 1.0, out=t)
        qx = t * dx[None, :, 0] - rx
        qy = t * dx[None, :, 1] - ry
        d = np.sqrt(qx * qx + qy * qy) * self.cell_m
        u = (self._s0[None] + t * (self._s1 - self._s0)[None]) / max(self.length_m,
                                                                    1e-9)
        if self.n_segments > self.J:
            sel = np.argpartition(d, self.J - 1, axis=1)[:, :self.J]
            d = np.take_along_axis(d, sel, axis=1)
            u = np.take_along_axis(u, sel, axis=1)
        elif self.n_segments < self.J:
            # fewer segments than J: pad with the last one (edge) so the result
            # always has J columns and the min() is unchanged
            pad = self.J - self.n_segments
            d = np.pad(d, ((0, 0), (0, pad)), mode="edge")
            u = np.pad(u, ((0, 0), (0, pad)), mode="edge")
        return d, u

    # ------------------------------------------------------------- public API
    def score(self, node_idx, g_m, alpha_d: float = 1.0, alpha_p: float = 1.0,
              sigma_m: float = None, L_ref: float = None,
              chunk: int = 1 << 16) -> np.ndarray:
        idx = np.asarray(node_idx, dtype=np.int64).reshape(-1)
        if not self.valid:
            return np.full(idx.shape, np.inf)
        sigma = float(sigma_m if sigma_m is not None else 3.0 * self.cell_m)
        L = float(L_ref if L_ref is not None else max(self.length_m, 1e-9))
        g_arr = np.asarray(g_m, dtype=np.float64)
        if len(idx) and self.n_segments < 1:
            return np.full(idx.shape, np.inf)
        if len(idx):
            missing = idx[~self._checked[idx]]
            if len(missing):
                d_new, u_new = self._values(missing)
                self._d[missing] = d_new
                self._u[missing] = u_new
                self._checked[missing] = True
        d_all = self._d[idx]
        u_all = self._u[idx]
        g = np.full(len(idx), float(g_arr)) if g_arr.ndim == 0 else g_arr
        u_g = np.clip(g / L, 0.0, 1.0)[:, None]
        h = (alpha_d * d_all / sigma) + alpha_p * np.abs(u_all - u_g)
        return h.min(axis=1)
