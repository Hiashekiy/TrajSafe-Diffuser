"""Baselines that share the anchor search and the neighbour/collision rules."""

from __future__ import annotations

import numpy as np

from .focal_astar import SearchConfig, SearchResult, focal_search

__all__ = ["dijkstra", "astar", "focal_astar", "pdg_focal_astar", "METHODS",
           "run_method"]


def dijkstra(safe, start_idx, goal_idx, neighbors=None, **kw) -> SearchResult:
    cfg = SearchConfig(w=1.0, heuristic="none", secondary="f0_index",
                       snapshots=kw.pop("snapshots", (1, 10, 50, 100, 500)), **kw)
    return focal_search(safe, start_idx, goal_idx, cfg, neighbors)


def astar(safe, start_idx, goal_idx, neighbors=None, **kw) -> SearchResult:
    cfg = SearchConfig(w=1.0, heuristic="octile", secondary="f0_index",
                       snapshots=kw.pop("snapshots", (1, 10, 50, 100, 500)), **kw)
    return focal_search(safe, start_idx, goal_idx, cfg, neighbors)


def focal_astar(safe, start_idx, goal_idx, w=1.2, secondary="clearance",
                neighbors=None, **kw) -> SearchResult:
    """Plain Focal A*: same anchor, same w, a FIXED guidance-free secondary rule.

    ``secondary='clearance'``  -> larger obstacle clearance first (then f0, index)
    ``secondary='index'``      -> stable cell index inside FOCAL
    ``secondary='f0_index'``   -> smallest f0 (this degenerates to plain A*)
    """
    if secondary == "guidance":
        raise ValueError("plain Focal A* must not use the diffusion guidance")
    cfg = SearchConfig(w=float(w), heuristic="octile", secondary=secondary,
                       snapshots=kw.pop("snapshots", (1, 10, 50, 100, 500)), **kw)
    return focal_search(safe, start_idx, goal_idx, cfg, neighbors)


def pdg_focal_astar(safe, start_idx, goal_idx, guidance, w=1.2, alpha_d=1.0,
                    alpha_p=1.0, sigma_m=None, L_ref_m=None, neighbors=None,
                    snapshots=(1, 10, 50, 100, 500), h_table=None,
                    secondary="guidance", h_tie_tol_cells=0.0) -> SearchResult:
    cfg = SearchConfig(w=float(w), heuristic="octile", secondary=secondary,
                       alpha_d=float(alpha_d), alpha_p=float(alpha_p),
                       sigma_m=sigma_m, L_ref_m=L_ref_m, guidance=guidance,
                       h_table=h_table, snapshots=snapshots,
                       h_tie_tol_cells=float(h_tie_tol_cells))
    return focal_search(safe, start_idx, goal_idx, cfg, neighbors)


def guided_astar(safe, start_idx, goal_idx, guidance, lam=0.1, alpha_d=1.0,
                 alpha_p=1.0, w=1.0, secondary="index", sigma_m=None,
                 L_ref_m=None, neighbors=None,
                 snapshots=(1, 10, 50, 100, 500)) -> SearchResult:
    """A* whose accumulated cost includes the diffusion guidance term.

    ``lam = 0`` reproduces plain A*; larger ``lam`` trades real path length for
    staying near the reference.  The heuristic stays the octile distance, so the
    search is still optimal for the PENALISED cost, while the returned path is
    reported with its REAL length (``cost_m``) and the penalty actually paid
    (``stats['guide_penalty_m']``).
    """
    cfg = SearchConfig(w=float(w), heuristic="octile", secondary=secondary,
                       alpha_d=float(alpha_d), alpha_p=float(alpha_p),
                       sigma_m=sigma_m, L_ref_m=L_ref_m, guidance=guidance,
                       guide_weight=float(lam), snapshots=snapshots)
    return focal_search(safe, start_idx, goal_idx, cfg, neighbors)


METHODS = ("dijkstra", "astar", "focal_clearance", "focal_index", "pdg",
           "guided_astar")


def run_method(name: str, safe, start_idx, goal_idx, neighbors=None,
               guidance=None, w: float = 1.2, alpha_d: float = 1.0,
               alpha_p: float = 1.0, sigma_m=None, L_ref_m=None,
               lam: float = 0.0, secondary: str = "index",
               snapshots=(1, 10, 50, 100, 500)) -> SearchResult:
    """Dispatcher used by the benchmark so every method shares the same inputs."""
    if name == "dijkstra":
        return dijkstra(safe, start_idx, goal_idx, neighbors,
                        snapshots=snapshots)
    if name == "astar":
        return astar(safe, start_idx, goal_idx, neighbors, snapshots=snapshots)
    if name == "focal_clearance":
        return focal_astar(safe, start_idx, goal_idx, w=w,
                           secondary="clearance", neighbors=neighbors,
                           snapshots=snapshots)
    if name == "focal_index":
        return focal_astar(safe, start_idx, goal_idx, w=w, secondary="index",
                           neighbors=neighbors, snapshots=snapshots)
    if name == "pdg":
        return pdg_focal_astar(safe, start_idx, goal_idx, guidance, w=w,
                               alpha_d=alpha_d, alpha_p=alpha_p,
                               sigma_m=sigma_m, L_ref_m=L_ref_m,
                               neighbors=neighbors, snapshots=snapshots)
    if name == "guided_astar":
        return guided_astar(safe, start_idx, goal_idx, guidance, lam=lam,
                            alpha_d=alpha_d, alpha_p=alpha_p, w=w,
                            secondary=secondary, sigma_m=sigma_m,
                            L_ref_m=L_ref_m, neighbors=neighbors,
                            snapshots=snapshots)
    raise ValueError("unknown method %r" % (name,))


def no_guidance_field(n_cells: int):
    """Placeholder used when the reference is invalid: PDG degrades to Focal A*."""
    from .progress_guidance import GuidanceField
    return GuidanceField(dist_m=np.full((n_cells, 1), np.inf, np.float32),
                         prog=np.full((n_cells, 1), np.nan, np.float32),
                         seg_id=np.full((n_cells, 1), -1, np.int32),
                         valid=False, length_m=0.0, build_ms=0.0, n_segments=0,
                         coverage=0.0, cover_m=0.0, cell_m=1.0,
                         reason="invalid_reference")
