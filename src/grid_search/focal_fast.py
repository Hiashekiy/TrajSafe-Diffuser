"""Heap-based inner loop for the w = 1 focal search (the config the planner uses).

Why this module exists
----------------------
The vectorised loop in :mod:`grid_search.focal_astar` rebuilds the FOCAL band by
scanning the WHOLE open set every iteration and re-scores every band member with
a ~20-call numpy pipeline.  Measured on the test split that is ~90-150 us per
expansion and it grows with the open set, so it dominates both the median
(221 expansions) and the tail (52 k expansions).

At ``w == 1`` the focal rule degenerates to "expand a node with ``f0 == f_min``",
i.e. a plain priority queue with a tie-break: exactly what ``heapq`` does.  This
module therefore

* keeps OPEN in a binary heap ordered by ``(f0, tie_break, node, g_at_push)``
  with lazy deletion (an entry is stale when the node's ``g`` improved since the
  push, or when the node is already closed);
* evaluates the tie-break **scalarly** from per-cell caches in ``array('f')``
  buffers (a few hundred ns) instead of a per-call numpy pipeline;
* batches cold guidance fills through a small pending queue (>= 32 cells per
  numpy call) so one-cell fills stop paying the numpy call overhead.

It is the same search: nodes leave the queue in ``(f0, key)`` order, the path is
reconstructed from the same parent chain and reported with its true grid cost.
Equivalence with the vectorised loop is checked in ``tests/test_focal_fast.py``.

Only ``w == 1`` and ``guide_weight == 0`` take this path; everything else keeps
using :func:`grid_search.focal_astar.focal_search` unchanged.
"""

from __future__ import annotations

import heapq
import time
from array import array

import numpy as np

from .focal_astar import (SearchConfig, SearchResult, _path_cost, _reconstruct,
                          _stats, octile_heuristic)
from .grid_map import SafeGrid
from .grid_neighbors import NeighborTable, build_neighbor_table

__all__ = ["focal_search_fast", "supports"]

_PENDING_FLUSH = 32


def supports(cfg: SearchConfig) -> bool:
    """Whether ``cfg`` is the degenerate focal case this module implements."""
    if not bool(getattr(cfg, "fast_path", True)):
        return False
    if float(cfg.guide_weight) > 0.0:
        return False
    if abs(float(cfg.w) - 1.0) > 1e-9:
        return False
    return cfg.secondary in ("guidance", "guidance_h", "clearance", "f0_index",
                             "index")


class _GuidanceCache:
    """Per-cell ``(d, u)`` of the J nearest reference segments, scalar access."""

    def __init__(self, field, n: int):
        self.field = field
        J = int(getattr(field, "J", 0) or 0)
        if not J:                                  # GuidanceField: dense [N, J]
            J = int(np.asarray(getattr(field, "dist_m")).shape[1])
        self.J = max(J, 1)
        self.d = [array("f", bytes(4 * n)) for _ in range(self.J)]
        self.u = [array("f", bytes(4 * n)) for _ in range(self.J)]
        self.has = bytearray(n)
        self.pending = []
        self.fills = 0

    def fill(self, cells) -> None:
        """Batch-fill ``cells`` in ONE vectorised guidance pass.

        Supports both field kinds: ``LazyGuidance`` (computes the J nearest
        segments on demand) and a precomputed ``GuidanceField`` (reads its
        dense ``dist_m``/``prog`` tables).
        """
        idx = np.asarray(cells, dtype=np.int64).reshape(-1)
        if hasattr(self.field, "_values"):
            d, u = self.field._values(idx)
        else:
            d, u = self.field.dist_m[idx], self.field.prog[idx]
        d = np.asarray(d, dtype=np.float32).reshape(len(idx), -1)
        u = np.asarray(u, dtype=np.float32).reshape(len(idx), -1)
        if d.shape[1] != self.J:                 # reference shorter than J
            pad = self.J - d.shape[1]
            mode = "edge" if d.shape[1] else "constant"
            d = np.pad(d, ((0, 0), (0, pad)), mode=mode)
            u = np.pad(u, ((0, 0), (0, pad)), mode=mode)
        # a non-finite progress would poison the heap order: neutralise it
        np.nan_to_num(d, copy=False, nan=np.inf, posinf=np.inf)
        np.nan_to_num(u, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        cols = idx.tolist()
        for j in range(self.J):
            dj, uj = d[:, j].tolist(), u[:, j].tolist()
            dst, ust = self.d[j], self.u[j]
            for pos, cell in enumerate(cols):
                dst[cell] = dj[pos]
                ust[cell] = uj[pos]
        for cell in cols:
            self.has[cell] = 1
        self.fills += 1

    def queue(self, cell: int) -> None:
        self.pending.append(cell)
        if len(self.pending) >= _PENDING_FLUSH:
            self.flush()

    def flush(self) -> None:
        if self.pending:
            self.fill(self.pending)
            self.pending = []


def _score_scalar(cache: _GuidanceCache, node: int, g_node: float, inv_L: float,
                  scale_d: float, alpha_p: float) -> float:
    """``min_j alpha_d*d_j/sigma + alpha_p*|u_j - g/L|`` for ONE cell."""
    ug = g_node * inv_L
    if ug < 0.0:
        ug = 0.0
    elif ug > 1.0:
        ug = 1.0
    d, u = cache.d, cache.u
    best = np.inf
    for j in range(cache.J):
        term = scale_d * d[j][node] + alpha_p * abs(u[j][node] - ug)
        if term < best:
            best = term
    return best


def focal_search_fast(safe: SafeGrid, start_idx: int, goal_idx: int,
                      cfg: SearchConfig = None,
                      neighbors: NeighborTable = None) -> SearchResult:
    cfg = cfg or SearchConfig(w=1.0)
    neighbors = neighbors or build_neighbor_table(safe.free, safe.spec.cell_m)
    spec = safe.spec
    n = spec.n_cells
    start_idx, goal_idx = int(start_idx), int(goal_idx)
    secondary = cfg.secondary
    field = cfg.guidance if bool(getattr(cfg.guidance, "valid", False)) else None
    if secondary in ("guidance", "guidance_h") and field is None:
        secondary = cfg.fallback_secondary
    if secondary == "guidance_h" and cfg.h_table is None:
        secondary = "guidance"

    start_free, goal_free = safe.is_free_index(np.array([start_idx, goal_idx]))
    if not bool(start_free):
        return SearchResult("INVALID_START",
                            stats=_stats(0, 0, 0, 0, 0.0, cfg, secondary),
                            guidance_used=False)
    if not bool(goal_free):
        return SearchResult("INVALID_GOAL",
                            stats=_stats(0, 0, 0, 0, 0.0, cfg, secondary),
                            guidance_used=False)
    if start_idx == goal_idx:
        return SearchResult("SUCCESS", path=np.array([start_idx], dtype=np.int32),
                            cost_m=0.0,
                            stats=_stats(0, 0, 0, 0, 0.0, cfg, secondary),
                            expanded=np.zeros(0, np.int32),
                            open_at_end=np.zeros(0, np.int32),
                            closed=np.zeros(0, np.int32), guidance_used=False)

    cell_m = float(spec.cell_m)
    width = int(spec.width)
    if cfg.h_table is not None:
        h = np.ascontiguousarray(np.asarray(cfg.h_table, dtype=np.float64))
        if h.shape != (n,):
            raise ValueError("h_table must have shape (%d,), got %s"
                             % (n, h.shape))
    elif cfg.heuristic == "none":
        h = None
    else:
        goal_rc = spec.index_to_rc(np.array([goal_idx]))[0]
        rr, cc = np.divmod(np.arange(n, dtype=np.int64), width)
        h = octile_heuristic(rr, cc, goal_rc, cell_m)

    g = array("d", [np.inf]) * n
    parent = array("i", [-1]) * n
    closed = bytearray(n)
    in_open = bytearray(n)

    cache = None
    inv_L = scale_d = alpha_p = 0.0
    if secondary in ("guidance", "guidance_h"):
        sigma = float(cfg.sigma_m if cfg.sigma_m is not None else 3.0 * cell_m)
        L_ref = float(cfg.L_ref_m if cfg.L_ref_m is not None
                      else max(getattr(field, "length_m", 1.0), 1e-9))
        inv_L = 1.0 / max(L_ref, 1e-9)
        scale_d = float(cfg.alpha_d) / sigma
        alpha_p = float(cfg.alpha_p)
        cache = _GuidanceCache(field, n)
    clearance_flat = (safe.clearance_m.reshape(-1).astype(np.float64)
                      if secondary == "clearance" else None)

    def key2(node: int, g_node: float) -> float:
        """Tie-break inside the ``f0 == f_min`` group (the FOCAL secondary rule)."""
        if secondary == "guidance":
            if cache.has[node]:
                return _score_scalar(cache, node, g_node, inv_L, scale_d, alpha_p)
            cache.queue(node)
            return np.inf                     # repaired by the re-push on flush
        if secondary == "guidance_h":
            if cache.has[node]:
                return float(h[node])
            cache.queue(node)
            return np.inf
        if secondary == "clearance":
            return -float(clearance_flat[node])
        if secondary == "index":
            return float(node)
        return 0.0

    nbr_idx = neighbors.idx
    move_cost = neighbors.cost.tolist()

    g[start_idx] = 0.0
    in_open[start_idx] = 1
    h0 = float(h[start_idx]) if h is not None else 0.0
    heap = [(h0, key2(start_idx, 0.0), start_idx, 0.0)]
    live_entries = 1
    peak_open = 1
    expanded = generated = reopened = 0
    expansion_order = []
    snap_points = set(int(v) for v in (cfg.snapshots or ()))
    snaps = {}
    t0 = time.perf_counter()

    def _push(node: int, g_node: float):
        nonlocal live_entries
        f0 = g_node + (h[node] if h is not None else 0.0)
        heapq.heappush(heap, (f0, key2(node, g_node), node, g_node))
        live_entries += 1

    def _flush_pending():
        if cache is None or not cache.pending:
            return
        queued = list(dict.fromkeys(cache.pending))
        cache.pending = []
        cache.fill(queued)
        for nb in queued:
            if in_open[nb] and not closed[nb] and np.isfinite(g[nb]):
                _push(nb, g[nb])

    while heap:
        f0, k2, node, g_push = heapq.heappop(heap)
        live_entries -= 1
        if closed[node] or g_push != g[node]:
            continue                          # stale entry (lazy deletion)
        if k2 == np.inf:                      # guidance was cold at push time
            _flush_pending()
            if not closed[node] and not cache.has[node]:
                cache.fill([node])
            if in_open[node] and not closed[node] and np.isfinite(g[node]):
                _push(node, g[node])
            continue
        closed[node] = 1
        in_open[node] = 0
        expanded += 1
        expansion_order.append(node)
        if expanded in snap_points:
            snaps[expanded] = np.asarray(expansion_order, dtype=np.int32)
        if node == goal_idx:
            elapsed = (time.perf_counter() - t0) * 1000.0
            path = _reconstruct(np.asarray(parent, dtype=np.int32), goal_idx)
            real_cost = _path_cost(spec, path)
            stats = _stats(expanded, generated, reopened, peak_open, elapsed,
                           cfg, secondary)
            stats["search_cost_m"] = float(g[goal_idx])
            stats["path_true_m"] = float(real_cost)
            stats["guide_penalty_m"] = 0.0   # w == 1 + no fused lambda here
            stats["path_true_m_relaxed"] = float(g[goal_idx])
            stats["search_backend"] = "heap"
            stats["guidance_fills"] = int(cache.fills if cache else 0)
            return SearchResult("SUCCESS", path=path, cost_m=float(real_cost),
                                stats=stats,
                                expanded=np.asarray(expansion_order,
                                                    dtype=np.int32),
                                open_at_end=_flags(in_open),
                                closed=_flags(closed), snapshots=snaps,
                                guidance_used=cache is not None)

        base = g[node]
        nbs = nbr_idx[node]
        for k in range(8):
            nb = int(nbs[k])
            if nb < 0:
                continue
            tentative = base + move_cost[k]
            if tentative + 1e-12 >= g[nb]:
                continue
            generated += 1
            g[nb] = tentative
            parent[nb] = node
            if closed[nb]:
                closed[nb] = 0
                reopened += 1
            in_open[nb] = 1
            _push(nb, tentative)
        if live_entries > peak_open:
            peak_open = live_entries
        if live_entries > 4 * max(len(heap), 1):     # keep memory flat
            heapq.heapify(heap)
            live_entries = len(heap)

    elapsed = (time.perf_counter() - t0) * 1000.0
    return SearchResult("NO_PATH", path=None, cost_m=np.inf,
                        stats=_stats(expanded, generated, reopened, peak_open,
                                     elapsed, cfg, secondary),
                        expanded=np.asarray(expansion_order, dtype=np.int32),
                        open_at_end=_flags(in_open), closed=_flags(closed),
                        snapshots=snaps, guidance_used=cache is not None)


def _flags(mask: bytearray) -> np.ndarray:
    return np.flatnonzero(np.frombuffer(bytes(mask), dtype=np.uint8))
