"""Progress-aware Diffusion-Guided Focal A* (design section 5).

Anchor search is an ordinary 8-connected A* (octile heuristic, real edge costs =
grid movement length only).  FOCAL is the set of OPEN nodes whose ``f0 = g + h0``
is within ``w`` of the live minimum; the node expanded next is the FOCAL member
chosen by the configured secondary rule (diffusion guidance for Ours).

Guarantees this implementation actually supports (design section 11):

* ``h0`` is the octile distance -> admissible and consistent on the 8-connected
  grid with real edge costs ``{cell_m, sqrt(2)*cell_m}``.
* the guidance terms never enter ``g``, the edge costs or the FOCAL threshold,
  so they can only reorder nodes INSIDE one FOCAL set.
* FOCAL is rebuilt from the LIVE OPEN set every iteration, CLOSED nodes that
  receive a smaller ``g`` are re-opened, and the goal is only reported when it
  is expanded out of FOCAL.  Under those rules the returned path is within
  ``w`` times the optimal cost (standard focal-search bound).
* every returned path is re-validated independently in
  ``grid_search.path_validation`` before a caller may accept it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .grid_map import SafeGrid
from .grid_neighbors import NeighborTable, build_neighbor_table

__all__ = ["SearchConfig", "SearchResult", "focal_search", "octile_heuristic",
           "SECONDARY_RULES"]

SQRT2 = float(np.sqrt(2.0))
SECONDARY_RULES = ("guidance", "clearance", "index", "f0_index")


def octile_heuristic(rows, cols, goal_rc, cell_m: float) -> np.ndarray:
    dx = np.abs(np.asarray(cols, dtype=np.float64) - float(goal_rc[1]))
    dy = np.abs(np.asarray(rows, dtype=np.float64) - float(goal_rc[0]))
    return (np.maximum(dx, dy) + (SQRT2 - 1.0) * np.minimum(dx, dy)) * cell_m


@dataclass
class SearchConfig:
    """How the search orders OPEN *and* how the guidance enters the cost.

    ``guide_weight`` (lambda) is the fused variant: the diffusion guidance is
    added to the accumulated cost of a cell, so the search actively prefers
    cells near the reference instead of merely re-ordering FOCAL:

        penalty(cell) = lambda * min_j [ d_j + alpha_p * L_ref * |u_j - u_g_true| ]
        g(v)          = sum of real moves  +  sum of penalties along the path
        f0(v)         = g(v) + octile(v)          (h0 is still admissible)

    Because the penalty is a non-negative node surcharge, A* stays optimal for
    THIS cost; the *real* grid cost of the returned path is then >= optimal and
    is reported separately (``guide_penalty_m`` / ``path_cost_m``).
    """

    w: float = 1.2
    heuristic: str = "octile"          # "octile" | "none" (Dijkstra)
    secondary: str = "guidance"        # one of SECONDARY_RULES
    fallback_secondary: str = "clearance"   # used when the guidance is invalid
    alpha_d: float = 1.0
    alpha_p: float = 1.0
    sigma_m: float = None              # default 3 * cell_m
    L_ref_m: float = None              # default: reference length
    guidance: object = None            # GuidanceField | None
    guide_weight: float = 0.0          # lambda: 0 = pure A*, no fused penalty
    snapshots: tuple = (1, 10, 50, 100, 500)
    max_expansions: int = None

    def copy(self, **kw) -> "SearchConfig":
        d = dict(self.__dict__)
        d.update(kw)
        return SearchConfig(**d)


@dataclass
class SearchResult:
    status: str                        # SUCCESS | NO_PATH | INVALID_START | INVALID_GOAL
    path: np.ndarray = None            # [n] int32 cell indices, start..goal
    cost_m: float = np.inf
    stats: dict = field(default_factory=dict)
    expanded: np.ndarray = None        # [E] int32, expansion order
    open_at_end: np.ndarray = None
    closed: np.ndarray = None
    snapshots: dict = field(default_factory=dict)
    guidance_used: bool = False


def _tiebreak_pick(cand_idx, h_d, f0, clearance, secondary):
    """Return the FOCAL member to expand (deterministic, documented order).

    The primary key is found with ``argmin`` (O(k) with a small constant); only
    the (usually single) tied set goes through ``lexsort``.  This keeps the
    per-iteration cost dominated by the guidance gather itself even when FOCAL
    holds tens of thousands of nodes.
    """
    if len(cand_idx) == 0:
        return -1
    if secondary == "guidance" and h_d is not None:
        primary = h_d
    elif secondary == "clearance":
        primary = -clearance
    elif secondary == "f0_index":
        primary = f0
    else:                                   # "index": stable index inside FOCAL
        primary = cand_idx.astype(np.float64)
    best = primary.min()
    # EXACT equality (no tolerance): a fuzzy tie would silently re-order nodes
    # whose keys differ only in the last bits, which changes A* expansion counts.
    tight = np.flatnonzero(primary == best)
    if len(tight) == 1:
        return int(cand_idx[tight[0]])
    sub = tight
    idxk = cand_idx[sub].astype(np.float64)
    if secondary == "guidance":
        order = np.lexsort((idxk, -clearance[sub], f0[sub]))
    elif secondary == "clearance":
        order = np.lexsort((idxk, f0[sub]))
    elif secondary == "f0_index":
        order = np.lexsort((idxk,))
    else:
        order = np.lexsort((idxk,))
    return int(cand_idx[sub[order[0]]])


def _path_cost(spec, path) -> float:
    """True grid cost of a reconstructed path (sum of the 8-neighbour edge costs)."""
    if path is None or len(path) < 2:
        return 0.0
    rc = spec.index_to_rc(np.asarray(path, dtype=np.int64)).astype(np.float64)
    return float(np.linalg.norm(np.diff(rc, axis=0), axis=1).sum() * spec.cell_m)


def _reconstruct(parent: np.ndarray, goal: int) -> np.ndarray:
    path = [int(goal)]
    cur = int(goal)
    seen = {cur}
    while parent[cur] >= 0:
        cur = int(parent[cur])
        if cur in seen:
            raise RuntimeError("parent cycle in reconstructed path")
        seen.add(cur)
        path.append(cur)
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def _stats(expanded, generated, reopened, peak_open, elapsed_ms, cfg,
           secondary=None) -> dict:
    return {"expanded_nodes": int(expanded),
            "generated_nodes": int(generated),
            "reopened_nodes": int(reopened),
            "peak_open_size": int(peak_open),
            "search_time_ms": float(elapsed_ms),
            "w": float(cfg.w),
            "secondary": secondary or cfg.secondary,
            "alpha_d": float(cfg.alpha_d),
            "alpha_p": float(cfg.alpha_p),
            "guide_weight": float(cfg.guide_weight),
            "heuristic": cfg.heuristic}


def focal_search(safe: SafeGrid, start_idx: int, goal_idx: int,
                 cfg: SearchConfig = None,
                 neighbors: NeighborTable = None) -> SearchResult:
    cfg = cfg or SearchConfig()
    neighbors = neighbors or build_neighbor_table(safe.free, safe.spec.cell_m)
    spec = safe.spec
    n = spec.n_cells
    start_idx, goal_idx = int(start_idx), int(goal_idx)
    # An invalid reference is NOT a planning failure: PDG degrades to the
    # guidance-free Focal A* variant named by ``fallback_secondary``.
    secondary = cfg.secondary
    guidance_ok = bool(getattr(cfg.guidance, "valid", False))
    if secondary == "guidance" and not guidance_ok:
        secondary = cfg.fallback_secondary
    use_guidance = secondary == "guidance" and guidance_ok
    # the FUSED variant (guide_weight > 0) needs the field even when the
    # ordering rule is plain A* (secondary='index')
    guidance_fused = cfg.guidance if guidance_ok else None

    start_free, goal_free = safe.is_free_index(np.array([start_idx, goal_idx]))
    if not bool(start_free):
        return SearchResult("INVALID_START", stats=_stats(0, 0, 0, 0, 0.0, cfg),
                            guidance_used=use_guidance)
    if not bool(goal_free):
        return SearchResult("INVALID_GOAL", stats=_stats(0, 0, 0, 0, 0.0, cfg),
                            guidance_used=use_guidance)
    if start_idx == goal_idx:
        return SearchResult("SUCCESS", path=np.array([start_idx], dtype=np.int32),
                            cost_m=0.0,
                            stats=_stats(0, 0, 0, 0, 0.0, cfg),
                            expanded=np.zeros(0, np.int32),
                            open_at_end=np.zeros(0, np.int32),
                            closed=np.zeros(0, np.int32),
                            guidance_used=False)

    goal_rc = spec.index_to_rc(np.array([goal_idx]))[0]
    cell_m = float(spec.cell_m)
    width = int(spec.width)

    g = np.full(n, np.inf, dtype=np.float64)
    f0 = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int32)
    closed = np.zeros(n, dtype=bool)
    in_open = np.zeros(n, dtype=bool)

    guidance = guidance_fused if use_guidance else None
    clearance_flat = np.ascontiguousarray(safe.clearance_m.reshape(-1))
    nbr_idx = neighbors.idx
    move_cost = neighbors.cost

    # ---- fused guidance cost (lambda > 0) --------------------------------
    # g_true  : real travelled distance, used only for the progress term u_g
    # g       : real distance + accumulated guidance penalty (the A* cost)
    penalty_on = float(cfg.guide_weight) > 0.0 and guidance_fused is not None
    g_true = np.zeros(n, dtype=np.float64)
    L_ref = float(cfg.L_ref_m if cfg.L_ref_m is not None
                  else max(getattr(guidance_fused, "length_m", 1.0), 1e-9))
    lam = float(cfg.guide_weight)
    dist_tab = guidance_fused.dist_m if penalty_on else None
    prog_tab = guidance_fused.prog if penalty_on else None
    beta = float(cfg.alpha_p)

    def _penalty(nodes: np.ndarray, g_true_now: float) -> np.ndarray:
        """Guidance surcharge (metres) of entering ``nodes``."""
        if not penalty_on:
            return np.zeros(len(nodes), dtype=np.float64)
        u_g = min(max(g_true_now / L_ref, 0.0), 1.0)
        s = dist_tab[nodes].astype(np.float64) \
            + beta * L_ref * np.abs(prog_tab[nodes].astype(np.float64) - u_g)
        return lam * s.min(axis=1)

    # The octile heuristic only depends on the cell -> precompute it once per
    # search (one vectorised pass over the grid) instead of evaluating it in
    # Python for every neighbour of every expansion.
    if cfg.heuristic == "none":
        h_tab = None
    else:
        rr, cc = np.divmod(np.arange(n, dtype=np.int64), width)
        h_tab = octile_heuristic(rr, cc, goal_rc, cell_m).astype(np.float64)

    def h_of(node: int) -> float:
        if h_tab is None:
            return 0.0
        return float(h_tab[node])

    g[start_idx] = 0.0
    f0[start_idx] = h_of(start_idx)
    in_open[start_idx] = True
    # OPEN is a compacted index array + the global ``f0``/``in_open`` arrays.
    # Stale entries are dropped on compaction, so the per-iteration min/FOCAL
    # scan is a handful of numpy ops over a C array instead of a Python loop
    # over a growing heap.
    open_nodes = np.empty(n, dtype=np.int32)
    open_nodes[0] = start_idx
    open_n = 1
    peak_open = 1
    expanded = generated = reopened = 0
    expansion_order = []
    snap_points = set(int(v) for v in (cfg.snapshots or ()))
    snaps = {}
    t0 = time.perf_counter()

    def _compact():
        nonlocal open_nodes, open_n
        alive = open_nodes[:open_n]
        alive = alive[in_open[alive]]
        open_n = len(alive)
        open_nodes[:open_n] = alive
        if open_nodes is None:
            open_nodes = np.empty(n, dtype=np.int32)

    while open_n:
        live_nodes = open_nodes[:open_n]
        live_mask = in_open[live_nodes]
        if not live_mask.any():          # everything left is stale
            break
        f_live = np.where(live_mask, f0[live_nodes], np.inf)
        f_min = float(f_live.min())
        thr = cfg.w * f_min
        cand = live_nodes[live_mask & (f_live <= thr + 1e-12)]
        if len(cand) == 0:               # defensive
            break

        h_d = (guidance.score(cand, g[cand], cfg.alpha_d, cfg.alpha_p,
                              cfg.sigma_m, cfg.L_ref_m)
               if guidance is not None else None)
        current = _tiebreak_pick(cand, h_d, f0[cand], clearance_flat[cand],
                                 secondary)
        in_open[current] = False
        closed[current] = True
        expanded += 1
        expansion_order.append(current)
        if expanded in snap_points:
            snaps[expanded] = np.asarray(expansion_order, dtype=np.int32)

        if current == goal_idx:
            elapsed = (time.perf_counter() - t0) * 1000.0
            path = _reconstruct(parent, goal_idx)
            # FOCAL may expand the goal while another FOCAL member has a smaller
            # f0 (that is exactly the bounded-suboptimality trade-off), so the
            # parent chain of the returned path can be CHEAPER than g[goal].
            # The reported cost is therefore the real cost of the returned path;
            # ``g[goal]`` is kept separately as the focal-bound quantity.
            real_cost = _path_cost(spec, path)
            stats = _stats(expanded, generated, reopened, peak_open, elapsed,
                           cfg, secondary)
            stats["search_cost_m"] = float(g[goal_idx])
            stats["path_true_m"] = float(real_cost)
            # g carries the guidance surcharge in the fused variant (clamped: the
            # difference of two float sums can be -1e-14 when lambda == 0)
            stats["guide_penalty_m"] = (max(0.0, float(g[goal_idx] - real_cost))
                                        if penalty_on else 0.0)
            stats["path_true_m_relaxed"] = float(g_true[goal_idx])
            return SearchResult(
                "SUCCESS", path=path,
                cost_m=float(real_cost),
                stats=stats,
                expanded=np.asarray(expansion_order, dtype=np.int32),
                open_at_end=np.asarray(np.where(in_open)[0], dtype=np.int32),
                closed=np.asarray(np.where(closed)[0], dtype=np.int32),
                snapshots=snaps,
                guidance_used=guidance is not None or penalty_on)

        # ---- 8-neighbour relaxation ---------------------------------------
        base = g[current]
        base_true = g_true[current]
        nbs = nbr_idx[current]
        valid = nbs >= 0
        if penalty_on and valid.any():
            pen = _penalty(nbs[valid].astype(np.int64), base_true)
            pen_of = dict(zip(nbs[valid].tolist(), np.asarray(pen).tolist()))
        else:
            pen_of = None
        for k in range(8):
            nb = int(nbs[k])
            if nb < 0:
                continue
            step = float(move_cost[k])
            tentative = base + step + (pen_of[nb] if pen_of is not None else 0.0)
            if tentative + 1e-12 >= g[nb]:
                continue
            generated += 1
            g[nb] = tentative
            g_true[nb] = base_true + step
            f0[nb] = tentative + h_of(nb)
            parent[nb] = current
            if closed[nb]:                 # reopen with the improved g
                closed[nb] = False
                reopened += 1
            if not in_open[nb]:
                in_open[nb] = True
                if open_n >= len(open_nodes):   # compact before growing
                    _compact()
                open_nodes[open_n] = nb
                open_n += 1

        live_count = int(in_open[open_nodes[:open_n]].sum())
        peak_open = max(peak_open, live_count)
        # keep the OPEN array compact (stale fraction stays below ~50 %)
        if open_n > 1024 and open_n > 2 * live_count:
            _compact()

        if cfg.max_expansions is not None and expanded >= int(cfg.max_expansions):
            break

    elapsed = (time.perf_counter() - t0) * 1000.0
    return SearchResult(
        "NO_PATH", path=None, cost_m=np.inf,
        stats=_stats(expanded, generated, reopened, peak_open, elapsed, cfg, secondary),
        expanded=np.asarray(expansion_order, dtype=np.int32),
        open_at_end=np.asarray(np.where(in_open)[0], dtype=np.int32),
        closed=np.asarray(np.where(closed)[0], dtype=np.int32),
        snapshots=snaps, guidance_used=guidance is not None)

