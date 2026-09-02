"""Analytical convex (safe) region construction from an ellipse + obstacle map.

This module mirrors the Neural-IRIS "generate_safe_region" logic: given a
prior ellipse (shape matrix P and centre c), it searches the obstacle boundary
points in the ellipse's metric, greedily builds separating halfspaces from the
nearest obstacle points, and filters out every obstacle point that the new
halfspace already keeps outside the safe region.  The result is a convex
polytope

    P = { x in R^2 | A x <= b }

that contains the ellipse and does not cut any obstacle.

Unlike Neural-IRIS, this project's P is the *shape* matrix of the ellipse
(i.e. the ellipse is p = P u + c with ||u|| <= 1), so the quadratic form is
Q = P^{-2}.  We therefore compute Q = P^{-1} P^{-1} internally instead of
P.T @ P as the reference implementation does for its inverse-shape matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection


def ellipse_quadratic_form(P: np.ndarray, eps: float = 1e-10) -> np.ndarray | None:
    """Return the quadratic-form matrix Q of an ellipse p = P u + c.

    The ellipse is (p - c)^T Q (p - c) <= 1 with Q = P^{-2}.
    P is symmetrised first; returns None if P is singular.
    """
    P = np.asarray(P, dtype=float).reshape(2, 2)
    P = 0.5 * (P + P.T)
    try:
        P_inv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(P_inv)):
        return None
    Q = P_inv.T @ P_inv
    if not np.all(np.isfinite(Q)) or np.linalg.det(P) < eps:
        return None
    return Q


def generate_convex_region(
    P: np.ndarray,
    c: np.ndarray,
    obs_points: np.ndarray,
    patch_size: int = 128,
    safety_margin: float = 0.5,
    include_boundary: bool = True,
    dedup_eps: float = 1e-6,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate halfspaces (A, b) of a convex region from an ellipse + obstacles.

    Parameters
    ----------
    P : (2,2) shape matrix of the ellipse (p = P u + c, ||u|| <= 1).
    c : (2,) centre of the ellipse.
    obs_points : (N,2) obstacle boundary points in the same coordinate frame.
    patch_size : size of the local occupancy patch (used when bounds is None).
    safety_margin : extra inward offset (in the same units as obs_points).
    include_boundary : if True, add box constraints around the region.
    dedup_eps : tolerance used when filtering obstacle points behind a new face.
    bounds : optional (xmin, xmax, ymin, ymax) for the box constraints.  When
        provided, it overrides patch_size for the boundary box.

    Returns
    -------
    (A, b) : array (M,2) and (M,) with A x <= b.
    """
    c = np.asarray(c, dtype=float).reshape(2)
    obs_points = np.asarray(obs_points, dtype=float).reshape(-1, 2)
    A_list: list[np.ndarray] = []
    b_list: list[float] = []

    if bounds is None:
        xmin, xmax, ymin, ymax = 0.0, float(patch_size), 0.0, float(patch_size)
    else:
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)

    if include_boundary:
        A_list.extend([np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
                       np.array([0.0, 1.0]), np.array([0.0, -1.0])])
        b_list.extend([
            xmax - safety_margin,
            safety_margin - xmin,
            ymax - safety_margin,
            safety_margin - ymin,
        ])

    if len(obs_points) == 0:
        return np.asarray(A_list, dtype=float), np.asarray(b_list, dtype=float)

    Q = ellipse_quadratic_form(P)
    if Q is None:
        return np.asarray(A_list, dtype=float), np.asarray(b_list, dtype=float)

    # Metric distance to the ellipse: (obs-c)^T Q (obs-c) == 1 on the ellipse.
    diffs = obs_points - c
    dists = np.sum((diffs @ Q) * diffs, axis=1)
    active_obs = obs_points[np.argsort(dists)]

    while len(active_obs) > 0:
        obs = active_obs[0]
        normal = Q @ (obs - c)
        norm_n = np.linalg.norm(normal)
        if norm_n > 1e-6:
            normal = normal / norm_n
            b_val = float(np.dot(normal, obs) - safety_margin)
            A_list.append(normal)
            b_list.append(b_val)

            # Obstacle-point filtering: drop every point that the newly inserted
            # halfspace already puts outside the safe region (n^T x > b_val).
            rest = active_obs[1:]
            if len(rest):
                dist_along = np.dot(rest, normal)
                mask = dist_along <= (b_val + dedup_eps)
                active_obs = rest[mask]
            else:
                active_obs = np.empty((0, 2), dtype=float)
        else:
            active_obs = active_obs[1:]

    return np.asarray(A_list, dtype=float), np.asarray(b_list, dtype=float)


def halfspaces_to_vertices(
    A: np.ndarray,
    b: np.ndarray,
    interior_point: np.ndarray | None = None,
) -> np.ndarray | None:
    """Convert A x <= b to counter-clockwise polygon vertices.

    Uses scipy.spatial.HalfspaceIntersection with interior_point as a
    strictly-feasible point (the ellipse centre usually).  Returns None if the
    polytope is empty, degenerate, or the intersection fails.
    """
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(A) < 3:
        return None

    # Drop redundant/num-degenerate rows.
    norms = np.linalg.norm(A, axis=1)
    keep = norms > 1e-8
    if not np.any(keep):
        return None
    A = A[keep]
    b = b[keep]
    if len(A) < 3:
        return None

    if interior_point is None:
        interior_point = np.zeros(2, dtype=float)
    interior_point = np.asarray(interior_point, dtype=float).reshape(2)

    # scipy convention: [A, -b] represents A x - b <= 0  (i.e. A x <= b).
    halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        intersections = np.asarray(hs.intersections, dtype=float)
        if len(intersections) < 3:
            return None
        hull = ConvexHull(intersections)
        return intersections[hull.vertices]
    except Exception:
        return None


def project_halfspaces_to_world(
    A_pix: np.ndarray,
    b_pix: np.ndarray,
    anchor_world: np.ndarray,
    local_res: float,
    patch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert patch-pixel halfspaces A_pix x_pix <= b_pix to world coords.

    The patch-to-world mapping is
        p_pix = local_res * (p_world - anchor) + (half - 0.5)     (per-axis)
    so a halfspace n^T p_pix <= b becomes
        (local_res * n)^T p_world
            <= b - n^T (half-0.5) + local_res * n^T anchor.
    """
    A_pix = np.asarray(A_pix, dtype=float).reshape(-1, 2)
    b_pix = np.asarray(b_pix, dtype=float).reshape(-1)
    anchor_world = np.asarray(anchor_world, dtype=float).reshape(2)
    half = float(patch_size) / 2.0
    s = np.array([half - 0.5, half - 0.5], dtype=float)

    A_world = local_res * A_pix
    b_world = b_pix - np.dot(A_pix, s) + np.dot(A_world, anchor_world)
    return A_world, b_world


def generate_convex_region_world(
    P_pix: np.ndarray,
    c_pix: np.ndarray,
    obs_points_pix: np.ndarray,
    anchor_world: np.ndarray,
    local_res: float,
    patch_size: int = 128,
    safety_margin: float = 0.5,
    include_boundary: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Convenience: generate convex region in world coordinates.

    Returns (A_world, b_world, vertices_world); vertices_world is None when the
    region is degenerate.
    """
    A_pix, b_pix = generate_convex_region(
        P_pix, c_pix, obs_points_pix,
        patch_size=patch_size, safety_margin=safety_margin,
        include_boundary=include_boundary,
    )
    A_world, b_world = project_halfspaces_to_world(
        A_pix, b_pix, anchor_world, local_res, patch_size=patch_size
    )
    c_world = anchor_world + (c_pix - (float(patch_size) / 2.0 - 0.5)) / local_res
    vertices_world = halfspaces_to_vertices(A_world, b_world, c_world)
    return A_world, b_world, vertices_world


def ellipse_params_to_shape(center, r1, r2, theta):
    """Build the ellipse shape matrix P (p = P u + c) from (r1, r2, theta).

    The ellipse axes are r1 (major) and r2 (minor); theta is the angle of the
    major axis from the +x axis (radians).
    """
    center = np.asarray(center, dtype=float).reshape(2)
    r1 = float(r1)
    r2 = float(r2)
    theta = float(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=float)
    P = R @ np.diag([r1, r2]) @ R.T
    return P


def generate_convex_region_from_params(
    params: np.ndarray,
    obs_points: np.ndarray,
    patch_size: int = 128,
    safety_margin: float = 0.5,
    include_boundary: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate convex region halfspaces from ellipse parameters + obstacle points.

    params is (cx, cy, r1, r2, theta) in the same coordinate frame as obs_points
    (typically patch pixels).
    """
    params = np.asarray(params, dtype=float).reshape(5)
    center = params[0:2]
    P = ellipse_params_to_shape(center, params[2], params[3], params[4])
    return generate_convex_region(
        P, center, obs_points,
        patch_size=patch_size, safety_margin=safety_margin,
        include_boundary=include_boundary,
    )


def generate_convex_region_from_params_world(
    params: np.ndarray,
    obs_points_pix: np.ndarray,
    anchor_world: np.ndarray,
    local_res: float,
    patch_size: int = 128,
    safety_margin: float = 0.5,
    include_boundary: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """World-frame convenience using ellipse params (cx, cy, r1, r2, theta).

    The params are in patch-pixel coordinates; the returned A/b and vertices are
    in world coordinates.  This is the direct entry point for a model that
    predicts ellipse params from a local occupancy crop.
    """
    A_pix, b_pix = generate_convex_region_from_params(
        params, obs_points_pix,
        patch_size=patch_size, safety_margin=safety_margin,
        include_boundary=include_boundary,
    )
    A_world, b_world = project_halfspaces_to_world(
        A_pix, b_pix, anchor_world, local_res, patch_size=patch_size
    )
    c_pix = np.asarray(params, dtype=float).reshape(5)[0:2]
    c_world = anchor_world + (c_pix - (float(patch_size) / 2.0 - 0.5)) / local_res
    vertices_world = halfspaces_to_vertices(A_world, b_world, c_world)
    return A_world, b_world, vertices_world
