"""Guidance-field construction.

    F(x) = max_i [ w(u_i) - lam * d_free(x, s_i) ]

``d_free`` is the 4-connected geodesic distance, so the field never leaks through walls and
obstacle cells stay NaN. Two exact backends:

* ``scipy`` -- one sparse Dijkstra from a virtual super-source (fast; default when available)
* ``heap``  -- multi-source max-Dijkstra in pure Python (fallback, no extra dependency)
"""

from __future__ import annotations

import heapq
import weakref

import numpy as np

from .grid import Grid

try:  # scipy is optional
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as _sparse_dijkstra

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - only without scipy
    _HAVE_SCIPY = False

WEIGHTS = ("uniform", "linear", "quadratic")
DEFAULT_LAM = 0.02
NEG_INF = -np.inf

_CSR_CACHE = weakref.WeakKeyDictionary()


def weight_at_progress(u, kind: str = "linear") -> np.ndarray:
    """w(u): 1 for uniform, u for linear, u^2 for quadratic."""
    u = np.asarray(u, dtype=np.float64)
    if kind == "uniform":
        return np.ones_like(u)
    if kind == "linear":
        return u.copy()
    if kind == "quadratic":
        return u * u
    raise ValueError(f"unknown weight {kind!r}, expected one of {WEIGHTS}")


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def _csr(grid: Grid):
    """Cached unit-weight CSR adjacency matrix of the free cells."""
    graph = _CSR_CACHE.get(grid)
    if graph is not None:
        return graph
    nbr = grid._nbr
    valid = nbr >= 0
    indptr = np.zeros(grid.size + 1, dtype=np.int64)
    np.cumsum(valid.sum(axis=1), out=indptr[1:])
    indices = nbr[valid].astype(np.int32)
    data = np.ones(len(indices), dtype=np.float64)
    graph = csr_matrix((data, indices, indptr), shape=(grid.size, grid.size))
    _CSR_CACHE[grid] = graph
    return graph


def propagate_scipy(grid: Grid, positions, weights, lam: float = DEFAULT_LAM) -> np.ndarray:
    """Exact: minimise ``C(x) = min_i [d + (w_max - w_i)/lam]`` with a virtual super-source."""
    if not _HAVE_SCIPY:  # pragma: no cover
        raise RuntimeError("scipy is not installed; use method='heap'")
    pos = np.asarray(positions, dtype=int).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    flat = np.full(grid.size, np.nan, dtype=np.float64)
    if len(pos) == 0:
        return flat.reshape(grid.height, grid.width)
    graph = _csr(grid)
    w_max = float(w.max())
    ids = pos[:, 0] * grid.width + pos[:, 1]
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    per_cell = np.full(len(unique_ids), NEG_INF, dtype=np.float64)
    np.maximum.at(per_cell, inverse, w)          # several samples may share one cell
    offsets = (w_max - per_cell) / lam + 1.0     # +1 keeps every sparse edge strictly positive
    n = grid.size
    indptr = np.concatenate((graph.indptr.astype(np.int64, copy=False),
                             np.array([graph.nnz + len(unique_ids)], dtype=np.int64)))
    indices = np.concatenate((graph.indices.astype(np.int32, copy=False),
                              unique_ids.astype(np.int32)))
    data = np.concatenate((graph.data, offsets))
    augmented = csr_matrix((data, indices, indptr), shape=(n + 1, n + 1))
    dist = _sparse_dijkstra(augmented, directed=True, indices=n)[:n]
    values = w_max - lam * (dist - 1.0)
    reached = grid.free.ravel() & np.isfinite(dist)
    flat[reached] = values[reached]
    return flat.reshape(grid.height, grid.width)


def propagate_heap(grid: Grid, positions, weights, lam: float = DEFAULT_LAM) -> np.ndarray:
    """Exact fallback: multi-source max-Dijkstra."""
    pos = np.asarray(positions, dtype=int).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    field = np.full(grid.size, NEG_INF, dtype=np.float64)
    seeded = set()
    for (r, c), wi in zip(pos, w):
        v = grid.idx(int(r), int(c))
        if float(wi) > field[v]:
            field[v] = float(wi)
        seeded.add(v)
    # heapify only after every seed reached its final weight: a cell shared by several
    # samples must be expanded with the strongest one, never with the first one seen
    heap = [(-field[v], v) for v in seeded]
    heapq.heapify(heap)
    nbr = grid._nbr
    while heap:
        neg, v = heapq.heappop(heap)
        value = -neg
        if value < field[v]:
            continue
        candidate = value - lam
        for u in nbr[v]:
            if u >= 0 and candidate > field[u]:
                field[u] = candidate
                heapq.heappush(heap, (-candidate, u))
    out = np.where(grid.free.ravel() & (field > NEG_INF), field, np.nan)
    return out.reshape(grid.height, grid.width)


def propagate(grid: Grid, positions, weights, lam: float = DEFAULT_LAM,
              method: str = "auto") -> np.ndarray:
    """Field from explicit sources and weights."""
    if method == "auto":
        method = "scipy" if _HAVE_SCIPY else "heap"
    if method == "scipy":
        return propagate_scipy(grid, positions, weights, lam)
    if method == "heap":
        return propagate_heap(grid, positions, weights, lam)
    raise ValueError(f"unknown method {method!r}, expected auto|scipy|heap")


def build_field(grid: Grid, positions, progress, lam: float = DEFAULT_LAM,
                weight: str = "linear", method: str = "auto") -> np.ndarray:
    """Field for sources given as ``(row, col)`` cells plus their trajectory progress ``u``."""
    return propagate(grid, positions, weight_at_progress(progress, weight), lam, method)
