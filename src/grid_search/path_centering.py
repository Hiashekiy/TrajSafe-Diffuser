"""Push a searched grid path towards the middle of the free channel.

Why this exists
---------------
The PDG-Focal A* path is a *grid* path: it follows the shortest route, which
hugs obstacle corners.  The safety ellipses are then centred on that wall-hugging
line, the local convex regions degenerate (`invalid_base_region`) or stop
overlapping (`bridge_gamma_failed`), and the corridor of the plan's section 5.3
cannot be closed.

This module does exactly one thing: given the searched path and the free-space
mask, move its INTERIOR points along the clearance gradient (away from walls),
smooth along the path, keep both endpoints pinned, and never leave free space.
It reuses :func:`src.grid_search.grid_map.clearance_field`, i.e. the same EDT the
project already uses for erosion/inflation, so "clearance" means the same thing
everywhere in the repo.

The path keeps its 128 arc-length samples afterwards, so everything downstream
(ellipse centres, bridge skeleton ``gamma``, path features) is unchanged.
"""

from __future__ import annotations

import numpy as np

from .grid_map import clearance_field

__all__ = ["center_path"]


def _bilinear(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample ``field[H,W]`` at scene coordinates ``xy[K,2]`` in [-1,1]."""
    H, W = field.shape
    px = (xy[:, 0] + 1.0) / 2.0 * (W - 1)
    py = (xy[:, 1] + 1.0) / 2.0 * (H - 1)
    x0 = np.clip(np.floor(px).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, H - 2)
    fx = np.clip(px - x0, 0.0, 1.0)
    fy = np.clip(py - y0, 0.0, 1.0)
    f00 = field[y0, x0]
    f10 = field[y0, x0 + 1]
    f01 = field[y0 + 1, x0]
    f11 = field[y0 + 1, x0 + 1]
    return (f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy)
            + f01 * (1 - fx) * fy + f11 * fx * fy)


def _bilinear3(fields: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Same as :func:`_bilinear` for a stack ``fields[C,H,W]`` -> ``[C,K]``.

    One gather instead of ``C`` calls: at C=3 this turns 3 x ~66 us of numpy
    call overhead per centring iteration into one.  Values are bit-identical to
    calling :func:`_bilinear` per channel (``tests/test_grid_search_hotpath.py``).
    """
    _, H, W = fields.shape
    px = (xy[:, 0] + 1.0) / 2.0 * (W - 1)
    py = (xy[:, 1] + 1.0) / 2.0 * (H - 1)
    x0 = np.clip(np.floor(px).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, H - 2)
    fx = np.clip(px - x0, 0.0, 1.0)
    fy = np.clip(py - y0, 0.0, 1.0)
    f00 = fields[:, y0, x0]
    f10 = fields[:, y0, x0 + 1]
    f01 = fields[:, y0 + 1, x0]
    f11 = fields[:, y0 + 1, x0 + 1]
    return (f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy)
            + f01 * (1 - fx) * fy + f11 * fx * fy)


def center_path(path, free, cell_m: float, *, iters: int = 40,
                step_cells: float = 1.0, smooth: float = 0.25,
                keep_ends: bool = True, clearance=None):
    """Return ``(centered_path, stats)``.

    ``path [K,2]`` scene coordinates, ``free [H,W]`` bool, ``cell_m`` metres per
    cell of that grid.  Only interior points move: ``path[0]``/``path[-1]`` are
    the start/goal and stay exact.

    ``clearance`` optionally supplies ``clearance_field(free, cell_m)`` when the
    caller already has it (``SafeGrid.raw_clearance_m``) -- it is the same array
    the safe-grid inflation computes, so sharing it saves one 256x256 EDT.
    """
    path = np.asarray(path, dtype=np.float64).copy()
    orig_path = path.copy()
    free = np.asarray(free, dtype=bool)
    H, W = free.shape
    K = len(path)
    stats = {"clearance_before": None, "clearance_after": None}
    if K < 3 or not free.any():
        return path, stats

    if clearance is not None and np.shape(clearance) == free.shape:
        clear = np.array(clearance, dtype=np.float64, copy=True)
    else:
        clear = clearance_field(free, cell_m).astype(np.float64)
    clear[~free] = 0.0

    scene_per_cell = 2.0 / W
    step = float(step_cells) * scene_per_cell
    gy, gx = np.gradient(clear)
    # one stacked gather per iteration instead of three ~66 us _bilinear calls
    fields = np.stack([clear, gx, gy])

    before = _bilinear(clear, path)
    stats["clearance_before"] = float(np.mean(before))
    stats["clearance_min_before"] = float(np.min(before))

    for _ in range(int(iters)):
        c, gxv, gyv = _bilinear3(fields, path)
        norm = np.sqrt(gxv ** 2 + gyv ** 2) + 1e-12
        # scale the step by the local clearance so points in wide space settle
        # and points squeezed against a wall move decisively
        mag = step * np.clip(1.0 - c / max(float(np.max(c)), 1e-9), 0.15, 1.0)
        move = np.stack([gxv / norm, gyv / norm], -1) * mag[:, None]
        if keep_ends:
            move[0] = 0.0
            move[-1] = 0.0
        prev = path.copy()
        cand = path + move
        # smooth along the path (1-2-1), endpoints pinned
        sm = cand.copy()
        sm[1:-1] = ((1 - smooth) * cand[1:-1]
                    + smooth * 0.5 * (cand[:-2] + cand[2:]))
        cand = sm
        # stay strictly inside the scene rectangle (a point at x = 1.0001 counts
        # as out-of-bounds, and the verification treats OOB as a collision)
        cand = np.clip(cand, -1.0 + 1e-6, 1.0 - 1e-6)
        # A point that would leave free space REVERTS to where it was: snapping it
        # to the nearest free cell can teleport it across a wall (that made the
        # centred path collide more than the raw grid path).
        px = np.clip(np.rint((cand[:, 0] + 1.0) / 2.0 * W - 0.5).astype(int),
                     0, W - 1)
        py = np.clip(np.rint((cand[:, 1] + 1.0) / 2.0 * H - 0.5).astype(int),
                     0, H - 1)
        bad = ~free[py, px]
        if bad.any():
            cand[bad] = prev[bad]
        path = cand
        if keep_ends:
            path[0], path[-1] = path[0], path[-1]

    # ---- safety gate: never hand back a path that collides more than the input
    def _collides(q):
        px = np.clip(np.rint((q[:, 0] + 1.0) / 2.0 * W - 0.5).astype(int), 0, W - 1)
        py = np.clip(np.rint((q[:, 1] + 1.0) / 2.0 * H - 0.5).astype(int), 0, H - 1)
        return int((~free[py, px]).sum())

    n_in = _collides(np.asarray(orig_path))
    n_out = _collides(path)
    stats["colliding_points_before"] = n_in
    stats["colliding_points_after"] = n_out
    if n_out > n_in:                     # centring must not cost feasibility
        stats["reverted_to_raw"] = True
        after = _bilinear(clear, np.asarray(orig_path))
        stats["clearance_after"] = float(np.mean(after))
        stats["clearance_min_after"] = float(np.min(after))
        return np.asarray(orig_path).copy(), stats

    after = _bilinear(clear, path)
    stats["clearance_after"] = float(np.mean(after))
    stats["clearance_min_after"] = float(np.min(after))
    return path, stats
