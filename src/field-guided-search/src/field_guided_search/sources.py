"""Trajectory -> legal source set.

A demonstrated (or diffused) trajectory is a polyline that may pass straight through
obstacles. Sources must sit on free cells, so every sampled point is either dropped or moved
to the nearest free cell. Each surviving sample keeps its original trajectory parameter ``u``
-- renormalising it would invent progress the trajectory never had.
"""

from __future__ import annotations

import numpy as np


def resample_polyline(points, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Even arc-length resampling of a polyline. Returns (points (K,2), progress (K,))."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"trajectory must be (N, 2) as [row, col], got {pts.shape}")
    if len(pts) == 0:
        raise ValueError("trajectory is empty")
    if len(pts) == 1:
        return pts.copy(), np.zeros(1)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = float(cum[-1])
    if total <= 0.0:
        return pts[:1].copy(), np.zeros(1)
    positions = np.linspace(0.0, total, int(num_samples))
    out = np.empty((len(positions), 2), dtype=np.float64)
    for k, s in enumerate(positions):
        j = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(pts) - 2))
        span = cum[j + 1] - cum[j]
        t = 0.0 if span <= 0 else (s - cum[j]) / span
        out[k] = pts[j] * (1.0 - t) + pts[j + 1] * t
    return out, positions / total


def nearest_free(grid, r: int, c: int, radius: int = 3) -> tuple[int, int] | None:
    """Closest free cell within ``radius`` (square window), or None if there is none."""
    if grid.free[r, c]:
        return int(r), int(c)
    best, best_d = None, None
    for rr in range(max(0, r - radius), min(grid.height - 1, r + radius) + 1):
        for cc in range(max(0, c - radius), min(grid.width - 1, c + radius) + 1):
            if not grid.free[rr, cc]:
                continue
            d = (rr - r) ** 2 + (cc - c) ** 2
            if best_d is None or d < best_d:
                best, best_d = (rr, cc), d
    return best


def trajectory_sources(grid, trajectory, num_samples: int = 64, method: str = "project",
                       radius: int = 3) -> dict:
    """Sample a trajectory into legal sources.

    ``method='project'`` moves offending samples to a nearby free cell (sample count is
    preserved, position is mildly distorted); ``method='drop'`` removes them (positions are
    exact, but a collision interval leaves a hole in progress). Both keep the original ``u``.

    Returns ``{'positions_rc' (K,2), 'progress' (K,), 'num_dropped', 'num_samples'}``.
    """
    if method not in ("project", "drop"):
        raise ValueError(f"method must be 'project' or 'drop', got {method!r}")
    points, progress = resample_polyline(trajectory, num_samples)
    cells = np.rint(points).astype(int)

    kept_pos, kept_u = [], []
    dropped = out_of_bounds = 0
    for (r, c), u in zip(cells, progress):
        r, c = int(r), int(c)
        # a sample outside the map is not "on the nearest cell" -- clipping it there would
        # invent a source the trajectory never visited, so it is skipped instead
        if not (0 <= r < grid.height and 0 <= c < grid.width):
            out_of_bounds += 1
            continue
        if grid.free[r, c]:
            target = (r, c)
        elif method == "drop":
            target = None
        else:
            target = nearest_free(grid, r, c, radius)
        if target is None:
            dropped += 1
            continue
        kept_pos.append(target)
        kept_u.append(float(u))

    return {
        "positions_rc": np.asarray(kept_pos, dtype=int).reshape(-1, 2),
        "progress": np.asarray(kept_u, dtype=np.float64),
        "num_samples": int(num_samples),
        "num_kept": len(kept_pos),
        "num_dropped": int(dropped),
        "num_out_of_bounds": int(out_of_bounds),
    }
