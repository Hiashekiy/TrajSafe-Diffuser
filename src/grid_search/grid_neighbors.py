"""8-connected grid graph: movement costs and the no-corner-cutting rule.

``MOVES`` order is fixed so tie-breaks and statistics are reproducible:

    0:+x  1:-x  2:+y  3:-y  4:(+x,+y)  5:(+x,-y)  6:(-x,+y)  7:(-x,-y)

A diagonal move ``(x, y) -> (x+dx, y+dy)`` is only legal when BOTH side cells
``(x+dx, y)`` and ``(x, y+dy)`` are free.  Checking only the diagonal target
lets a path slip through the diagonal contact point of two obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MOVES", "DIAGONAL", "NeighborTable", "build_neighbor_table",
           "corner_cutting_violations"]

MOVES = ((0, 1), (0, -1), (1, 0), (-1, 0),
         (1, 1), (1, -1), (-1, 1), (-1, -1))
DIAGONAL = np.array([False, False, False, False, True, True, True, True])
SQRT2 = float(np.sqrt(2.0))


@dataclass
class NeighborTable:
    """Flattened 8-neighbour table for a boolean free mask.

    ``idx[i, k]`` is the neighbour cell index of move ``k`` or ``-1`` when the
    move leaves the map / enters a blocked cell / cuts a corner.
    """

    idx: np.ndarray          # [N, 8] int32
    cost: np.ndarray         # [8] float64, already scaled to metres
    free: np.ndarray         # [N] bool, flattened input mask
    width: int

    def n_valid(self) -> np.ndarray:
        return (self.idx >= 0).sum(axis=1)


def build_neighbor_table(safe_free: np.ndarray, cell_m: float = 1.0,
                         allow_corner_cut: bool = False) -> NeighborTable:
    free = np.ascontiguousarray(safe_free, dtype=bool)
    h, w = free.shape
    flat = free.reshape(-1)
    rows, cols = np.divmod(np.arange(h * w, dtype=np.int64), w)
    idx = np.full((h * w, 8), -1, dtype=np.int32)
    for k, (dr, dc) in enumerate(MOVES):
        r = rows + dr
        c = cols + dc
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        nbr = np.full(h * w, -1, dtype=np.int64)
        rr, cc = r[ok], c[ok]
        nbr[ok] = rr * w + cc
        valid = ok.copy()
        valid[ok] = free[rr, cc]
        if DIAGONAL[k] and not allow_corner_cut:
            # both side cells must be free as well
            side_a = np.full(h * w, -1, dtype=np.int64)
            side_b = np.full(h * w, -1, dtype=np.int64)
            ok_a = (rows + dr >= 0) & (rows + dr < h) & (cols >= 0) & (cols < w)
            ok_b = (rows >= 0) & (rows < h) & (cols + dc >= 0) & (cols + dc < w)
            side_a[ok_a] = (rows[ok_a] + dr) * w + cols[ok_a]
            side_b[ok_b] = rows[ok_b] * w + (cols[ok_b] + dc)
            for side in (side_a, side_b):
                has = side >= 0
                side_free = np.zeros(h * w, dtype=bool)
                side_free[has] = flat[side[has]]
                valid &= side_free
        idx[valid, k] = nbr[valid]
    cost = np.array([SQRT2 if d else 1.0 for d in DIAGONAL],
                    dtype=np.float64) * float(cell_m)
    return NeighborTable(idx=idx, cost=cost, free=flat, width=int(w))


def corner_cutting_violations(safe_free: np.ndarray, path_rc) -> int:
    """Count illegal diagonal transitions in a (row, col) path."""
    rc = np.asarray(path_rc, dtype=np.int64)
    if len(rc) < 2:
        return 0
    h, w = safe_free.shape
    bad = 0
    for (r0, c0), (r1, c1) in zip(rc[:-1], rc[1:]):
        dr, dc = int(r1 - r0), int(c1 - c0)
        if abs(dr) != 1 or abs(dc) != 1:
            continue
        a = (r0 + dr, c0)
        b = (r0, c0 + dc)
        if not (0 <= a[0] < h and 0 <= a[1] < w):
            bad += 1
        elif not safe_free[a]:
            bad += 1
        elif not (0 <= b[0] < h and 0 <= b[1] < w):
            bad += 1
        elif not safe_free[b]:
            bad += 1
    return bad
