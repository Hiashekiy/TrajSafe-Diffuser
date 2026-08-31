"""Offline IRIS (Iterative Regional Inflation) solver for ellipse labels.

This ports the real IRIS ellipse solver from the original IRIS dataset generator
(the maximum-volume inscribed ellipse inside an obstacle-safe convex polytope),
rather than using the trained Neural-IRIS network.  It works directly on a 128x128
occupancy patch and returns the inellipse (P, c) in patch-pixel coordinates.
"""

import warnings

import numpy as np
import scipy.ndimage as ndimage

warnings.filterwarnings("ignore")

try:
    import cvxpy as cp
    CVXOPT_AVAILABLE = False  # cvxopt not required; cvxpy CLARABEL/SCS is used
except Exception:
    cp = None
    CVXOPT_AVAILABLE = False


def extract_obstacle_constraints(patch, dilation_iters=1, boundary_jitter=1):
    """Dilate obstacles and return (obstacle_mask, boundary_points)."""
    struct = np.ones((3, 3), dtype=bool)
    obstacle_mask = np.asarray(patch, dtype=np.uint8) == 1
    if dilation_iters > 0:
        obstacle_mask = ndimage.binary_dilation(obstacle_mask, structure=struct, iterations=dilation_iters)
    boundary_mask = obstacle_mask ^ ndimage.binary_erosion(obstacle_mask, structure=struct)
    py, px = np.where(boundary_mask)
    boundary_points = np.column_stack((px, py)).astype(float)
    if boundary_jitter > 0 and len(boundary_points) > 0:
        offsets = []
        for dy in range(-boundary_jitter, boundary_jitter + 1):
            for dx in range(-boundary_jitter, boundary_jitter + 1):
                if dx == 0 and dy == 0:
                    continue
                offsets.append((dx, dy))
        dense = [boundary_points]
        for dx, dy in offsets:
            shifted = boundary_points + np.array([dx, dy], dtype=float)
            shifted[:, 0] = np.clip(shifted[:, 0], 0, patch.shape[1] - 1)
            shifted[:, 1] = np.clip(shifted[:, 1], 0, patch.shape[0] - 1)
            dense.append(shifted)
        boundary_points = np.unique(np.vstack(dense), axis=0)
    return obstacle_mask, boundary_points


def _solve_iris_once(P_val, c_val, obs_points, bounds, K_bins, max_iters):
    xmin, xmax, ymin, ymax = bounds
    P_val = np.asarray(P_val, dtype=float).reshape(2, 2)
    c_val = np.asarray(c_val, dtype=float).reshape(2)
    for _ in range(max_iters):
        try:
            P_inv = np.linalg.inv(P_val)
            P_inv2 = P_inv.T @ P_inv
        except np.linalg.LinAlgError:
            break
        obs_shifted = obs_points - c_val
        obs_trans = obs_shifted @ P_inv.T
        dists_trans = np.linalg.norm(obs_trans, axis=1)
        angles = np.arctan2(obs_trans[:, 1], obs_trans[:, 0])
        bin_indices = np.floor((angles + np.pi) / (2 * np.pi) * K_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, K_bins - 1)
        A, b = [], []
        A.extend([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        b.extend([xmax, -xmin, ymax, -ymin])
        for k in range(K_bins):
            mask = bin_indices == k
            if not np.any(mask):
                continue
            sector_dists = dists_trans[mask]
            min_idx = np.argmin(sector_dists)
            active_obs = obs_points[mask][min_idx]
            n = P_inv2 @ (active_obs - c_val)
            norm_n = np.linalg.norm(n)
            if norm_n > 1e-6:
                n_norm = n / norm_n
                A.append(n_norm)
                b.append(np.dot(n_norm, active_obs))
        A_mat = np.array(A, dtype=float)
        b_vec = np.array(b, dtype=float)
        P = cp.Variable((2, 2), PSD=True)
        c = cp.Variable(2)
        constraints = [cp.norm(P @ A_mat[j]) + A_mat[j] @ c <= b_vec[j] for j in range(len(A_mat))]
        objective = cp.Maximize(cp.log_det(P))
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate") or P.value is None:
                prob.solve(solver=cp.SCS, max_iters=500, eps=1e-3, verbose=False)
        except Exception:
            break
        if prob.status not in ("optimal", "optimal_inaccurate") or P.value is None or c.value is None:
            break
        vol_old = np.linalg.det(P_val)
        vol_new = np.linalg.det(P.value)
        P_val = np.asarray(P.value, dtype=float)
        c_val = np.asarray(c.value, dtype=float)
        if vol_old > 1e-8 and (vol_new - vol_old) / vol_old < 0.05:
            break
    return P_val, c_val


def solve_iris_offline(obs_points, bounds=(0, 128, 0, 128), seed=(64.0, 64.0), max_iters=15, K_bins=32):
    """Run the offline IRIS solver; returns (P [2,2], c [2]) in patch-pixel coords."""
    seed_c = np.asarray(seed, dtype=float).reshape(2)
    xmin, xmax, ymin, ymax = bounds
    obs_points = np.asarray(obs_points, dtype=float).reshape(-1, 2)
    if len(obs_points) == 0:
        radius = min(xmax - xmin, ymax - ymin) / 2.0
        return np.eye(2) * radius, seed_c
    dists = np.linalg.norm(obs_points - seed_c, axis=1)
    min_dist = np.min(dists)
    init_radius = max(min_dist * 0.5, 1.0)
    P_val = np.eye(2) * init_radius
    c_val = seed_c.copy()
    return _solve_iris_once(P_val, c_val, obs_points, bounds, K_bins, max_iters)


def is_ellipse_safe(P, c, obstacle_mask, interior_samples=48, boundary_samples=48):
    """Check ellipse does not cross an obstacle (boundary + interior samples)."""
    P = np.asarray(P, dtype=float)
    try:
        P = (P + P.T) / 2.0
        if np.linalg.det(P) <= 1e-10:
            return False
    except np.linalg.LinAlgError:
        return False
    angles = np.linspace(0.0, 2.0 * np.pi, boundary_samples, endpoint=False)
    unit_b = np.vstack((np.cos(angles), np.sin(angles)))
    boundary_points = (P @ unit_b).T + c
    rng = np.random.default_rng()
    radii = np.sqrt(rng.random(interior_samples))
    theta = rng.uniform(0.0, 2.0 * np.pi, interior_samples)
    unit_i = np.vstack((radii * np.cos(theta), radii * np.sin(theta)))
    interior_points = (P @ unit_i).T + c
    points = np.vstack((boundary_points, interior_points))
    xs = np.rint(points[:, 0]).astype(int)
    ys = np.rint(points[:, 1]).astype(int)
    valid = ((xs >= 0) & (xs < obstacle_mask.shape[1]) & (ys >= 0) & (ys < obstacle_mask.shape[0]))
    if not np.all(valid):
        return False
    return not np.any(obstacle_mask[ys, xs])


def is_trivial_ellipse(P, c, patch_size, area_threshold=16.0, center_threshold=2.0):
    P = np.asarray(P, dtype=float)
    c = np.asarray(c, dtype=float)
    if P.shape != (2, 2) or c.shape != (2,):
        return True
    det_P = np.linalg.det(P)
    if det_P <= 0:
        return True
    if det_P <= area_threshold:
        return True
    center = np.array([patch_size / 2.0, patch_size / 2.0], dtype=float)
    if np.linalg.norm(c - center) > center_threshold:
        return False
    return False


def is_anchor_inside_ellipse(P, c, anchor, tol=1.0001):
    P = np.asarray(P, dtype=float)
    c = np.asarray(c, dtype=float)
    diff = np.asarray(anchor, dtype=float) - c
    try:
        u = np.linalg.solve(P, diff)
    except np.linalg.LinAlgError:
        return False
    return float(np.linalg.norm(u)) <= tol


def P_to_ellipse_params(P, c):
    """Convert inellipse (P, c) to (center, Q, params [cx,cy,r1,r2,theta]).

    Ellipse boundary: p = P u + c, ||u|| <= 1, so P eigenvalues are semi-axis
    lengths.  Q = P^{-2} for the form (p-c)^T Q (p-c) <= 1.  r1 >= r2.
    """
    P = (np.asarray(P, dtype=float) + np.asarray(P, dtype=float).T) / 2.0
    w, v = np.linalg.eigh(P)
    idx_big = int(np.argmax(w))
    idx_small = 1 - idx_big
    r1 = float(max(w[idx_big], 1e-6))
    r2 = float(max(w[idx_small], 1e-6))
    v1 = v[:, idx_big]
    theta = np.arctan2(v1[1], v1[0])
    center = np.asarray(c, dtype=float).reshape(2)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Q = R @ np.diag([1.0 / (r1 ** 2), 1.0 / (r2 ** 2)]) @ R.T
    return center, Q, np.array([center[0], center[1], r1, r2, theta], dtype=float)
