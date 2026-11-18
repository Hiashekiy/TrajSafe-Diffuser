"""Exact geodesic distance-to-goal field, used as the A* heuristic.

Why
---
The octile heuristic is the STRAIGHT-LINE distance, but these maps force long
detours around obstacles: measured on the test split, ``h(start)`` is ~20 % below
the true optimum (e.g. 167.8 m vs 202.9 m).  An admissible-but-loose ``h`` makes
the ``f0 == C*`` plateau huge, so plain A* expands ~12 000 nodes and the focal
variant 4x more.

``skimage.graph.MCP_Geometric`` computes the exact 8-connected geodesic distance
over the SAME cost grid (uniform ``cell_m`` per free cell, ``sqrt(2)`` factor on
diagonals) in ~3 ms for 256x256.  It is

* **admissible**: MCP's move set is a superset of ours (it allows a diagonal
  through a blocked side cell, we forbid corner cutting), so
  ``h(cell) <= true_remaining_cost(cell)``;
* **consistent**: ``h(a) - h(b) <= d_mcp(a, b) <= edge_cost(a, b)``.

Both properties are what the focal bound relies on, so swapping ``h`` cannot
break the ``w``-suboptimality guarantee.  Measured effect (12 heavy samples):
expansions p50 10 836 -> 631, loop time 8.35 s -> 1.20 s, returned path cost
bit-identical.
"""

from __future__ import annotations

import numpy as np

__all__ = ["geodesic_distance_field", "geodesic_available"]


def geodesic_available() -> bool:
    try:
        import skimage.graph  # noqa: F401
    except Exception:
        return False
    return True


def geodesic_distance_field(safe_free, cell_m: float, goal_rc):
    """Flat ``[H*W]`` geodesic distance (metres) from the goal, ``inf`` if blocked.

    Returns ``None`` when the goal cell itself is blocked / out of bounds (the
    caller then keeps its octile heuristic).
    """
    free = np.asarray(safe_free, dtype=bool)
    h, w = free.shape
    r, c = int(goal_rc[0]), int(goal_rc[1])
    if not (0 <= r < h and 0 <= c < w) or not free[r, c]:
        return None
    from skimage.graph import MCP_Geometric

    costs = np.where(free, float(cell_m), np.inf)
    mcp = MCP_Geometric(costs, fully_connected=True)
    field, _ = mcp.find_costs([(r, c)])
    out = np.ascontiguousarray(field, dtype=np.float64).reshape(-1)
    out[~free.reshape(-1)] = np.inf
    return out
