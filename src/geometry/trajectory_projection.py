"""Final projection of a sampled trajectory into the local convex safe regions.

Given a sampled clean trajectory and the predicted ellipse per anchor, build the
local convex safe regions A_k x <= b_k using the SAME reliable constructor used by
the --convex overlay (infer_convex_region_from_scene_occupancy), then solve

    min  0.5 ||x - x0||^2 + lambda_smooth * ||acc(x)||^2
    s.t. A_k x_k <= b_k  for each anchor k and its covered points
         x_0 = start, x_H = goal

so the result is inside the (reliable) safe regions while staying close to the
model output and smooth.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

from src.geometry.offline_iris_wrapper import infer_convex_region_from_scene_occupancy


def _build_anchor_regions(pos, center, radii, theta, occ, cfg, region_stride=4):
    """Return a list of (seg_start, seg_end, A, b), one convex region PER point.

    The ellipse branch produces one ellipse per trajectory point (anchor k is at
    pos[k+1]), so we build a safe region for every interior point (j=1..H-2) using
    its own ellipse.  This makes the covered set continuous (no gaps between
    anchors), so the projection cannot leave interpolated holes that clip a wall.
    """
    H = pos.shape[0]
    N = center.shape[0]
    window_half = float(cfg.get("window_half", 0.25))
    safety_margin = float(cfg.get("safety_margin", 0.0))
    include_boundary = bool(cfg.get("include_boundary", True))
    dilation = int(cfg.get("dilation", 1))
    boundary_jitter = int(cfg.get("boundary_jitter", 1))

    regions = []
    for j in range(1, H - 1):            # interior trajectory points
        a = j - 1                        # ellipse anchored at pos[j]
        if a >= N:
            break
        c = center[a]
        r1 = radii[a, 0]
        r2 = radii[a, 1]
        th = theta[a]
        A, b, _ = infer_convex_region_from_scene_occupancy(
            occ, c, r1, r2, th,
            window_half=window_half, safety_margin=safety_margin,
            include_boundary=include_boundary,
            dilation=dilation, boundary_jitter=boundary_jitter)
        if A is None or b is None or len(A) < 3:
            continue
        regions.append((j, j + 1, np.asarray(A, dtype=float), np.asarray(b, dtype=float)))
    return regions


def project_to_convex_regions(pos, center, radii, theta, occ, cfg,
                              lambda_smooth=1.0, region_stride=4, solver="OSQP"):
    """Project one trajectory into the convex safe regions via QP.

    pos (H,2), center (N,2), radii (N,2), theta (N), occ (H,W).  Returns (H,2).
    """
    H = pos.shape[0]
    x0 = np.asarray(pos, dtype=float)
    x = cp.Variable((H, 2))
    obj = 0.5 * cp.sum_squares(x - x0)
    acc = x[2:] - 2.0 * x[1:-1] + x[:-2]
    obj = obj + float(lambda_smooth) * cp.sum_squares(acc)

    cons = [x[0, :] == x0[0], x[-1, :] == x0[-1]]
    for (a, seg_end, A, b) in _build_anchor_regions(x0, center, radii, theta, occ,
                                                    cfg, region_stride):
        for s in range(a, seg_end):
            cons.append(A @ x[s, :] <= b)

    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError:
        prob.solve(solver="ECOS")
    if x.value is None:
        return x0
    return np.asarray(x.value, dtype=float)
