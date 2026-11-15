"""Global distance-progress guidance field (design section 4).

For every grid cell we keep the ``J`` closest SEGMENT projections of the
reference polyline together with the normalised progress ``u`` of each
projection.  Keeping several candidates is what makes self-intersections,
hairpins and near-parallel legs unambiguous: a cell next to two different legs
of the same reference then owns two candidates with very different ``u``.

The guidance value of a search node is NOT a per-cell constant -- it depends on
the node's own ``g`` (how far the search has already travelled):

    u_g(x) = clip(g(x) / L_ref, 0, 1)
    H_D(x, g) = min_j [ alpha_d * d_j(x) / sigma_d + alpha_p * |u_j(x) - u_g(x)| ]

Only ``(d_j, u_j)`` are precomputed; the ``g`` term is evaluated when a node
enters FOCAL.  No cell is ever excluded from the search: cells far from the
reference keep finite candidates (the nearest segment is always defined).

Implementation note (design 4.1): the projection is EXACT segment projection for
every cell -- no rasterisation shortcut -- evaluated in cell chunks so the
temporaries stay cache-sized.  Its cost is reported as ``build_ms`` and as
``guidance_build_time_ms`` in every experiment row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .diffusion_path import ReferencePath

__all__ = ["GuidanceField", "build_progress_guidance"]

_MAX_SEGMENTS_ONE_PASS = 256
_CELL_CHUNK = 8192


@dataclass
class GuidanceField:
    dist_m: np.ndarray        # [N, J] float32 distance to the j-th candidate
    prog: np.ndarray          # [N, J] float32 normalised arc length u_j
    seg_id: np.ndarray        # [N, J] int32 which segment produced the candidate
    valid: bool
    length_m: float
    build_ms: float
    n_segments: int
    coverage: float           # fraction of cells that own a candidate
    cover_m: float
    cover_frac: float = 0.0   # fraction of cells within ``cover_m``
    cell_m: float = 1.0
    reason: str = ""

    def score(self, node_idx, g_m, alpha_d: float = 1.0, alpha_p: float = 1.0,
              sigma_m: float = None, L_ref: float = None,
              chunk: int = 1 << 16) -> np.ndarray:
        """``H_D`` for a set of OPEN nodes (vectorised)."""
        idx = np.asarray(node_idx, dtype=np.int64)
        sigma = float(sigma_m if sigma_m is not None else 3.0 * self.cell_m)
        L = float(L_ref if L_ref is not None else max(self.length_m, 1e-9))
        out = np.empty(idx.shape, dtype=np.float64)
        g_arr = np.asarray(g_m, dtype=np.float64)
        for start in range(0, len(idx), chunk):
            sl = slice(start, start + chunk)
            d = self.dist_m[idx[sl]].astype(np.float64)
            u = self.prog[idx[sl]].astype(np.float64)
            g = np.full(len(idx[sl]), float(g_arr)) if g_arr.ndim == 0 else g_arr[sl]
            u_g = np.clip(g / L, 0.0, 1.0)[:, None]
            h = (alpha_d * d / sigma) + alpha_p * np.abs(u - u_g)
            out[sl] = h.min(axis=1)
        return out


def _top_j(dist, prog, seg, J):
    keep = min(int(J), dist.shape[1])
    part = np.argpartition(dist, keep - 1, axis=1)[:, :keep]
    order = np.take_along_axis(dist, part, axis=1).argsort(axis=1)
    sel = np.take_along_axis(part, order, axis=1)
    return (np.take_along_axis(dist, sel, axis=1),
            np.take_along_axis(prog, sel, axis=1),
            np.take_along_axis(seg, sel, axis=1))


def build_progress_guidance(ref: ReferencePath, spec, J: int = 3,
                            cover_m: float = None,
                            segment_block: int = _MAX_SEGMENTS_ONE_PASS,
                            cell_chunk: int = _CELL_CHUNK) -> GuidanceField:
    """Precompute the ``J`` nearest segment projections of every grid cell."""
    t0 = time.perf_counter()
    N = spec.n_cells
    cover = float(cover_m if cover_m is not None else 3.0 * spec.cell_m)
    if not ref.valid or ref.n_segments < 1:
        return GuidanceField(np.full((N, 1), np.inf, np.float32),
                             np.full((N, 1), np.nan, np.float32),
                             np.full((N, 1), -1, np.int32), False,
                             float(ref.length_m or 0.0),
                             (time.perf_counter() - t0) * 1000.0, 0, 0.0,
                             cover, cell_m=float(spec.cell_m),
                             reason=ref.reason)

    rows, cols = np.divmod(np.arange(N, dtype=np.int64), spec.width)
    Xx = cols.astype(np.float32)
    Xy = rows.astype(np.float32)
    p0, p1 = ref.p0, ref.p1
    dx = p1 - p0
    denom = np.maximum((dx * dx).sum(axis=1), 1e-12)
    s0, s1 = ref.s0, ref.s1
    L = max(ref.length_m, 1e-9)
    n_seg = len(p0)
    cell_m = float(spec.cell_m)

    dist = np.full((N, 0), np.inf, dtype=np.float32)
    prog = np.full((N, 0), np.nan, dtype=np.float32)
    seg = np.full((N, 0), -1, dtype=np.int32)

    for lo in range(0, n_seg, int(segment_block)):
        hi = min(n_seg, lo + int(segment_block))
        dxb = dx[lo:hi].astype(np.float32)
        p0b = p0[lo:hi].astype(np.float32)
        den = denom[lo:hi].astype(np.float32)
        s0b = s0[lo:hi].astype(np.float32)
        s1b = s1[lo:hi].astype(np.float32)
        c = hi - lo
        dblk = np.empty((N, c), dtype=np.float32)
        ublk = np.empty((N, c), dtype=np.float32)
        for cs in range(0, N, int(cell_chunk)):
            ce = min(N, cs + int(cell_chunk))
            rx = Xx[cs:ce, None] - p0b[None, :, 0]
            ry = Xy[cs:ce, None] - p0b[None, :, 1]
            t = (rx * dxb[None, :, 0] + ry * dxb[None, :, 1]) / den[None]
            np.clip(t, 0.0, 1.0, out=t)
            # vector from the projection to the cell (no [n,c,2] temporary)
            qx = t * dxb[None, :, 0] - rx
            qy = t * dxb[None, :, 1] - ry
            dblk[cs:ce] = np.sqrt(qx * qx + qy * qy) * cell_m
            ublk[cs:ce] = (s0b[None] + t * (s1b - s0b)[None]) / L
        segblk = np.broadcast_to(np.arange(lo, hi, dtype=np.int32), (N, c))
        dist = np.concatenate([dist, dblk], axis=1)
        prog = np.concatenate([prog, ublk], axis=1)
        seg = np.concatenate([seg, segblk], axis=1)
        dist, prog, seg = _top_j(dist, prog, seg, J)

    nearest = dist[:, 0]
    coverage = float(np.isfinite(nearest).mean())
    cover_frac = float((nearest <= cover).mean())
    return GuidanceField(dist_m=dist.astype(np.float32),
                         prog=prog.astype(np.float32),
                         seg_id=np.ascontiguousarray(seg, dtype=np.int32),
                         valid=True, length_m=float(ref.length_m),
                         build_ms=(time.perf_counter() - t0) * 1000.0,
                         n_segments=int(n_seg), coverage=coverage,
                         cover_m=cover, cover_frac=cover_frac,
                         cell_m=cell_m)
