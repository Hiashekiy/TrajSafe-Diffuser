"""Search: plain A* plus the field-guided best-first family.

The guided priority is a single expression with two knobs:

    J(n) = alpha(n) * d_hat_G(n) ** gamma - (1 - alpha(n)) * F_hat(n)

* ``alpha = 0``            -> pure field follower            (field-only)
* ``alpha = 1, gamma = 1`` -> pure distance greedy on d_hat_G
* ``0 < alpha < 1``        -> linear fusion
* ``gamma != 1``           -> nonlinear fusion (gamma < 1 sharpens near the goal)
* ``adaptive``             -> alpha varies with the remaining distance

``F_hat`` and ``d_hat_G`` are normalised *before* the search, and the priority function only
ever sees those precomputed maps, never the accumulated cost ``g``. Only ``astar`` claims
shortest-path optimality.

Two distance models exist because they are not interchangeable:

* ``distance="manhattan"`` (default) -- the definition the published experiments used
* ``distance="geodesic"``            -- free-space BFS distance, respects walls

Timing is reported as ``preprocess_ms`` (distance model, normalisation, priority map),
``search_ms`` (the queue loop) and ``total_ms``; ``time_ms`` is kept as an alias of
``search_ms`` for callers written before the split.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

from .clearance import ClearanceRiskMap, validate_penalty_map, validate_risk_params
from .grid import Grid

# canonical display/normalisation ranges per weighting
FIELD_CLIP = {
    "uniform": (0.50, 1.00),
    "linear": (-0.10, 1.00),
    "quadratic": (-0.10, 1.00),
}

DISTANCE_MODES = ("manhattan", "geodesic")
EPS = 1e-12


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def normalise_field(field: np.ndarray, clip=None, weight: str = "linear") -> np.ndarray:
    """Map the field into [0, 1] with a fixed clip so one alpha means one thing."""
    lo, hi = clip if clip is not None else FIELD_CLIP.get(weight, (-0.1, 1.0))
    values = np.where(np.isfinite(field), np.clip(field, lo, hi), lo)
    return (values - lo) / (hi - lo)


def normalise_distance(grid: Grid, goal, mode: str = "manhattan") -> tuple[np.ndarray, float]:
    """d_hat_G in [0, 1]: distance to the goal divided by its maximum.

    ``manhattan`` ignores obstacles (the published experiments used this); ``geodesic`` uses
    the free-space BFS distance, so it respects walls. Results obtained with one are not
    evidence for the other.
    """
    if mode not in DISTANCE_MODES:
        raise ValueError(f"distance must be one of {DISTANCE_MODES}, got {mode!r}")
    if mode == "manhattan":
        rows = np.arange(grid.height)[:, None]
        cols = np.arange(grid.width)[None, :]
        dist = (np.abs(rows - goal[0]) + np.abs(cols - goal[1])).astype(np.float64)
        # the maximum is taken over free cells only: obstacle corners are never visited, and
        # using them would shrink every d_hat on the map (this is what the experiment code
        # does, so it is the definition the published alphas belong to)
        d_max = float(dist[grid.free].max()) if grid.free.any() else 0.0
        return dist / (d_max + EPS), d_max
    dist = grid.bfs_distance([grid.idx(*goal)]).astype(np.float64)
    reachable = (dist >= 0) & grid.free
    d_max = float(dist[reachable].max()) if reachable.any() else 0.0
    return np.where(reachable, dist / (d_max + EPS), 1.0), d_max


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pack(grid: Grid, path: list[int]) -> np.ndarray:
    if not path:
        return np.zeros((0, 2), dtype=int)
    rows, cols = np.divmod(np.asarray(path, dtype=int), grid.width)
    return np.stack([rows, cols], axis=1)


def _result(status: str, grid: Grid, path, trace, expanded: int, generated: int,
            length, preprocess_ms: float, t0: float) -> dict:
    search_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "status": status,
        "path": _pack(grid, path),
        "trace": _pack(grid, trace),
        "expanded": int(expanded),
        "generated": int(generated),
        "length": length,
        "search_ms": search_ms,
        "preprocess_ms": float(preprocess_ms),
        "total_ms": float(preprocess_ms) + search_ms,
        "time_ms": search_ms,  # alias kept for callers written before the split
    }


def path_is_valid(grid: Grid, path, start, goal) -> dict:
    """Continuity, obstacle and endpoint check on a returned path."""
    arr = np.asarray(path, dtype=int)
    if arr.size == 0:
        return {"valid": False, "reason": "empty path"}
    if len(arr) == 1:
        cell = (int(arr[0, 0]), int(arr[0, 1]))
        if cell != tuple(start) or cell != tuple(goal):
            return {"valid": False, "reason": "single-cell path is not start == goal"}
        if not grid.free[cell[0], cell[1]]:
            return {"valid": False, "reason": "path crosses an obstacle"}
        return {"valid": True, "length": 0}
    if tuple(arr[0]) != tuple(start) or tuple(arr[-1]) != tuple(goal):
        return {"valid": False, "reason": "endpoints do not match start/goal"}
    if np.any(np.abs(np.diff(arr, axis=0)).sum(axis=1) != 1):
        return {"valid": False, "reason": "path is not 4-connected"}
    if not np.all(grid.free[arr[:, 0], arr[:, 1]]):
        return {"valid": False, "reason": "path crosses an obstacle"}
    return {"valid": True, "length": int(len(arr) - 1)}


def _best_first(grid: Grid, start, goal, priority, cutoff: int = 0,
                preprocess_ms: float = 0.0) -> dict:
    """Generic best-first loop driven by a pure ``priority(r, c)`` callable."""
    t0 = time.perf_counter()
    s, t = grid.idx(*start), grid.idx(*goal)
    if not grid.free.flat[s] or not grid.free.flat[t]:
        return _result("invalid", grid, [], [], 0, 0, None, preprocess_ms, t0)
    parent = {s: -1}
    seen = {s}
    counter = 0
    heap = [(priority(*start), counter, s)]
    trace: list[int] = []
    nbr = grid._nbr
    rows_cache: dict[int, tuple[int, int]] = {}
    while heap:
        _, _, v = heapq.heappop(heap)
        trace.append(v)
        if v == t:
            path = []
            while v != -1:
                path.append(v)
                v = parent[v]
            path.reverse()
            return _result("success", grid, path, trace, len(trace), len(seen),
                           len(path) - 1, preprocess_ms, t0)
        if cutoff and len(trace) >= cutoff:
            return _result("cutoff", grid, [], trace, len(trace), len(seen), None, preprocess_ms, t0)
        for u in nbr[v]:
            u = int(u)
            if u < 0 or u in seen:          # -1 marks "no neighbour" in the table
                continue
            seen.add(u)
            parent[u] = v
            if u not in rows_cache:
                rows_cache[u] = divmod(u, grid.width)
            counter += 1
            heapq.heappush(heap, (priority(*rows_cache[u]), counter, u))
    return _result("unreachable", grid, [], trace, len(trace), len(seen), None, preprocess_ms, t0)


# --------------------------------------------------------------------------- #
# public searches
# --------------------------------------------------------------------------- #
def astar(grid: Grid, start, goal, cutoff: int = 0) -> dict:
    """A* with the Manhattan heuristic; this is the only optimality-claiming search."""
    t0 = time.perf_counter()
    s, t = grid.idx(*start), grid.idx(*goal)
    if not grid.free.flat[s] or not grid.free.flat[t]:
        return _result("invalid", grid, [], [], 0, 0, None, 0.0, t0)
    gr, gc = goal
    g = {s: 0}
    parent = {s: -1}
    counter = 0
    heap = [(abs(start[0] - gr) + abs(start[1] - gc), 0, s)]
    closed = set()
    trace: list[int] = []
    nbr = grid._nbr
    status = "unreachable"
    while heap:
        _, _, v = heapq.heappop(heap)
        if v in closed:
            continue
        closed.add(v)
        trace.append(v)
        if v == t:
            path = []
            while v != -1:
                path.append(v)
                v = parent[v]
            path.reverse()
            return _result("success", grid, path, trace, len(trace), len(g), int(g[t]), 0.0, t0)
        if cutoff and len(trace) >= cutoff:
            status = "cutoff"
            break
        gv = g[v] + 1
        for u in nbr[v]:
            u = int(u)
            if u < 0 or u in closed or g.get(u, 1 << 30) <= gv:   # -1 = no neighbour
                continue
            g[u] = gv
            parent[u] = v
            r, c = divmod(u, grid.width)
            counter += 1
            heapq.heappush(heap, (gv + abs(r - gr) + abs(c - gc), counter, u))
    return _result(status, grid, [], trace, len(trace), len(g), None, 0.0, t0)


def resolve_clearance_risk(grid: Grid, beta, mode, threshold, sigma, tau,
                           penalty_map=None, cache: ClearanceRiskMap | None = None) -> dict:
    """Obtain ``R(D)`` for one search and report what it cost.

    ``penalty_map`` wins (no EDT, no scipy); otherwise the cache computes ``D`` once and every
    later ``(mode, threshold, sigma, tau)`` on the same map is free.  Never called when
    ``beta == 0``, so the old behaviour pays nothing for a feature it does not use.
    """
    if penalty_map is not None:
        return {"risk": validate_penalty_map(penalty_map, grid), "edt_ms": 0.0,
                "risk_ms": 0.0, "source": "provided"}
    if cache is None:
        cache = ClearanceRiskMap(grid)
    elif cache.grid is not grid:
        if cache.grid.free.shape != grid.free.shape or \
                not np.array_equal(cache.grid.free, grid.free):
            raise ValueError("the clearance cache belongs to a different map; build a new "
                             "ClearanceRiskMap / GuidanceField for the new occupancy grid")
    edt_ms, risk_ms = cache.edt_ms, cache.risk_ms
    calls = cache.edt_calls + cache.risk_calls
    risk = cache.risk(mode, threshold, sigma, tau)
    computed = (cache.edt_calls + cache.risk_calls) > calls
    return {"risk": risk, "edt_ms": cache.edt_ms - edt_ms, "risk_ms": cache.risk_ms - risk_ms,
            "source": "computed" if computed else "cache"}


def guided_search(grid: Grid, start, goal, field: np.ndarray | None = None,
                  alpha: float = 0.0, gamma: float = 1.0, adaptive: bool = False,
                  alpha_far: float = 0.25, alpha_near: float = 0.75,
                  clip=None, weight: str = "linear", distance: str = "manhattan",
                  cutoff: int = 0, clearance_beta: float = 0.0,
                  clearance_mode: str = "quadratic_hinge", clearance_threshold: float = 3.0,
                  clearance_sigma: float = 2.0, clearance_tau: float = 0.7,
                  clearance_penalty_map: np.ndarray | None = None,
                  clearance_cache: ClearanceRiskMap | None = None) -> dict:
    """Best-first search on the guidance field, optionally fused with the goal distance.

    ``clearance_beta > 0`` adds ``beta * R(D)`` to the priority map, where ``R`` is one of
    :mod:`field_guided_search.clearance`'s risk functions of the obstacle distance ``D`` in
    cells.  It is a **node priority** term, not an accumulated path cost: the loop still pops
    the smallest ``J`` and still fixes a parent at first discovery, so no optimality is
    claimed and the risk of a whole path is not minimised.  ``clearance_beta == 0`` (the
    default) reproduces the previous search exactly, down to the result keys.
    """
    beta, mode, threshold, sigma, tau = validate_risk_params(
        clearance_beta, clearance_mode, clearance_threshold, clearance_sigma, clearance_tau)
    t_pre = time.perf_counter()
    if field is not None and not np.isfinite(field).any():
        raise ValueError("the guidance field has no finite cell; build it from a trajectory "
                         "that touches free space")
    d_hat, _ = normalise_distance(grid, goal, mode=distance)
    f_hat = (normalise_field(field, clip, weight) if field is not None
             else np.zeros((grid.height, grid.width)))
    if adaptive:
        # far from the goal trust the field more, close to the goal trust the distance more
        alpha_map = alpha_far + (alpha_near - alpha_far) * (1.0 - d_hat)
    else:
        alpha_map = np.full_like(d_hat, float(alpha))
    pow_term = np.power(np.clip(d_hat, 0.0, 1.0), float(gamma))
    priority_map = alpha_map * pow_term - (1.0 - alpha_map) * f_hat
    # clearance is a soft term on the *node priority*; alpha and F_hat are left untouched
    extra: dict = {}
    if beta > 0.0:
        clearance = resolve_clearance_risk(grid, beta, mode, threshold, sigma, tau,
                                           clearance_penalty_map, clearance_cache)
        priority_map = priority_map + beta * clearance["risk"]
        extra = {"clearance_edt_ms": clearance["edt_ms"],
                 "clearance_risk_ms": clearance["risk_ms"],
                 "clearance_source": clearance["source"]}
    preprocess_ms = (time.perf_counter() - t_pre) * 1000.0
    result = _best_first(grid, start, goal, lambda r, c: float(priority_map[r, c]), cutoff,
                         preprocess_ms=preprocess_ms)
    result.update(extra)
    return result
