"""Verified convex region construction (V2).

generate_convex_region() (the Neural-IRIS style greedy builder) guarantees that
every SAMPLED obstacle boundary point ends up outside the polytope.  That is not
the same as "the polytope contains no obstacle cell": a convex set may still
cross a wall whose boundary samples it has excluded, in particular once the
local window is large enough for the region to reach a wall.

This module adds the missing independent verification and a repair loop:

    verify   - rasterise every occupancy cell whose centre is inside
               { A x <= b } and report the blocked ones;
    repair   - turn each offending cell into an extra halfspace through that
               cell with the normal pointing away from the centre, then verify
               again (the centre stays strictly inside by construction).

The result is a region that is safe at cell resolution, which is what a
downstream planner actually needs.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_EXTENT",
    "region_cells",
    "blocked_cells_in_region",
    "repair_region",
    "generate_verified_convex_region",
]

DEFAULT_EXTENT = (-1.0, 1.0, -1.0, 1.0)


def _cell_axis(indices: np.ndarray, lo: float, hi: float, res: int) -> np.ndarray:
    return (indices + 0.5) * (hi - lo) / res + lo


def region_cells(A: np.ndarray, b: np.ndarray, res: int = 256,
                 vertices: Optional[np.ndarray] = None,
                 extent: Sequence[float] = DEFAULT_EXTENT,
                 tol: float = 1e-12):
    """Indices of the occupancy cells whose centre lies inside {A x <= b}.

    The scan is restricted to the bounding box of the polytope vertices, so it
    is exact (no cell of the polytope is missed) and cheap.
    """
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(A) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    x0, x1, y0, y1 = (float(v) for v in extent)
    if vertices is not None and len(vertices) >= 3:
        v = np.asarray(vertices, dtype=float)
        lo = np.maximum(v.min(axis=0), np.array([x0, y0]))
        hi = np.minimum(v.max(axis=0), np.array([x1, y1]))
    else:
        lo = np.array([x0, y0])
        hi = np.array([x1, y1])
    if not (hi > lo).all():
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    i0 = max(0, int(np.floor((lo[0] - x0) / (x1 - x0) * res)) - 1)
    i1 = min(res, int(np.ceil((hi[0] - x0) / (x1 - x0) * res)) + 2)
    j0 = max(0, int(np.floor((lo[1] - y0) / (y1 - y0) * res)) - 1)
    j1 = min(res, int(np.ceil((hi[1] - y0) / (y1 - y0) * res)) + 2)
    if i1 <= i0 or j1 <= j0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    xs = _cell_axis(np.arange(i0, i1), x0, x1, res)
    ys = _cell_axis(np.arange(j0, j1), y0, y1, res)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    inside = (pts @ A.T <= b[None, :] + tol).all(axis=1)
    idx = np.nonzero(inside)[0]
    n_x = i1 - i0
    ii = i0 + (idx % n_x)
    jj = j0 + (idx // n_x)
    return ii, jj


def blocked_cells_in_region(A, b, occ, res=None, vertices=None,
                            extent=DEFAULT_EXTENT):
    """Cells inside the region that are obstacles.  Empty arrays mean safe."""
    occ = np.asarray(occ)
    res = int(res or occ.shape[0])
    ii, jj = region_cells(A, b, res=res, vertices=vertices, extent=extent)
    if len(ii) == 0:
        return ii, jj
    blocked = occ[jj, ii].astype(bool)
    return ii[blocked], jj[blocked]


def repair_region(A, b, center, blocked_xy, push: float = 0.0):
    """Append one halfspace per offending cell (normal pointing away from c).

    push is an extra inward offset (same units as the coordinates).  It must be
    at least one cell size: a face put exactly THROUGH the offending cell centre
    still leaves the cell touching the region, and the vertex it creates can sit
    at that cell centre.  Pushing a full cell past it removes the whole cell.
    """
    A = [np.asarray(row, dtype=float) for row in np.asarray(A, dtype=float)]
    b = [float(v) for v in np.asarray(b, dtype=float).reshape(-1)]
    center = np.asarray(center, dtype=float).reshape(2)
    added = 0
    for x in np.asarray(blocked_xy, dtype=float).reshape(-1, 2):
        n = x - center
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            continue
        n = n / norm
        A.append(n)
        b.append(float(n @ x) - float(push))
        added += 1
    if not A:
        return np.zeros((0, 2)), np.zeros(0)
    return np.asarray(A, dtype=float), np.asarray(b, dtype=float)


def generate_verified_convex_region(occ, center_scene, a, b_semi, theta,
                                    window_half: float = 0.25,
                                    safety_margin: float = 0.0,
                                    dilation: int = 1, boundary_jitter: int = 1,
                                    extent=DEFAULT_EXTENT,
                                    max_rounds: int = 6, shrink: float = 0.0,
                                    verbose: bool = False):
    """Convex region from an ellipse, verified and repaired until it is safe.

    Returns (A, b, vertices, info) where info contains the number of repair
    rounds, the number of blocked cells before/after, and whether the region is
    safe at cell resolution afterwards.
    """
    from .convex_region import halfspaces_to_vertices
    from .offline_iris_wrapper import infer_convex_region_from_scene_occupancy

    occ = np.asarray(occ)
    res = int(occ.shape[0])
    center = np.asarray(center_scene, dtype=float).reshape(2)
    A, b, vertices = infer_convex_region_from_scene_occupancy(
        occ, center, float(a), float(b_semi), float(theta),
        extent=extent, window_half=window_half, safety_margin=safety_margin,
        include_boundary=True, dilation=dilation, boundary_jitter=boundary_jitter)
    if A is not None and len(A) and float(shrink) > 0.0:
        # uniform inward erosion: every face n.x <= b with |n| = 1 moves inward
        # by 'shrink', which also pulls every vertex strictly inside free space
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float) - float(shrink)
        vertices = halfspaces_to_vertices(A, b, center)
    info = {"rounds": 0, "blocked_before": 0, "blocked_after": 0, "safe": False,
            "vertices": vertices is not None, "shrink": float(shrink)}
    if A is None or len(A) == 0:
        return A, b, vertices, info

    xs_axis = (np.arange(res) + 0.5) * (extent[1] - extent[0]) / res + extent[0]
    ys_axis = (np.arange(res) + 0.5) * (extent[3] - extent[2]) / res + extent[2]

    ii, jj = blocked_cells_in_region(A, b, occ, res=res, vertices=vertices,
                                     extent=extent)
    info["blocked_before"] = int(len(ii))
    for _ in range(int(max_rounds)):
        if len(ii) == 0:
            break
        cells = np.stack([xs_axis[ii], ys_axis[jj]], axis=-1)
        cell = (extent[1] - extent[0]) / float(res)
        A, b = repair_region(A, b, center, cells, push=cell)
        vertices = halfspaces_to_vertices(A, b, center)
        ii, jj = blocked_cells_in_region(A, b, occ, res=res, vertices=vertices,
                                         extent=extent)
        info["rounds"] += 1
        if verbose:
            print("   repair round %d -> %d blocked" % (info["rounds"], len(ii)))
    info["blocked_after"] = int(len(ii))
    info["safe"] = bool(len(ii) == 0 and vertices is not None)
    info["degenerate"] = vertices is None
    info["vertices"] = vertices is not None
    return A, b, vertices, info
