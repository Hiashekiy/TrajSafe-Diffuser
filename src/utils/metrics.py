"""Evaluation metrics helpers (dense-path interpolation)."""

import numpy as np


def interp_trajectory(positions_world, interp_steps=8):
    """Densify a waypoint trajectory by linearly interpolating between
    consecutive waypoints.  Returns an (N,2) dense point set so collision
    checks also cover the *segments* between waypoints, not just the nodes."""
    positions_world = np.asarray(positions_world, dtype=np.float64).reshape(-1, 2)
    T = len(positions_world)
    if T < 2 or interp_steps <= 1:
        return positions_world
    steps = max(2, int(interp_steps))
    dense = []
    for i in range(T - 1):
        for s in range(steps):
            a = positions_world[i]
            b = positions_world[i + 1]
            t = s / steps
            dense.append(a * (1 - t) + b * t)
    dense.append(positions_world[-1])
    return np.asarray(dense, dtype=np.float64)
