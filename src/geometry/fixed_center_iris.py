"""Fixed-centre IRIS: maximum-volume inscribed ellipse at a GIVEN centre (V2).

V1's offline IRIS solved

    max_{P, c}  log det P        s.t.  ||P A_j|| + A_j^T c <= b_j

so the label's centre c was free to move.  V2 fixes the centre geometrically
(c_i = gamma_m(s_i), docs/V2.md section 21), so the label MUST be recomputed
with c held constant:

    max_P  log det P             s.t.  ||P A_j|| + A_j^T c_i <= b_j

The returned label is the 4-vector [log a, log b, cos 2t, sin 2t] with a >= b,
which is exactly what EllipseShapeHead predicts.  There is no centre output and
no way for the solver to move the centre.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - cvxpy is a hard dependency of stage 4
    cp = None

__all__ = [
    "scene_obstacle_points",
    "solve_fixed_center_iris",
    "P_to_shape4",
    "fixed_center_shape4",
    "ellipse_is_inside_free",
]

DEFAULT_EXTENT = (-1.0, 1.0, -1.0, 1.0)


def scene_obstacle_points(occ: np.ndarray, center_scene, window_half: float = 0.25,
                          dilation: int = 1, boundary_jitter: int = 1,
                          extent=DEFAULT_EXTENT):
    """Obstacle boundary points around a scene point, in SCENE coordinates.

    The occupancy map spans 'extent' and stores 1 = obstacle.  A local window is
    cropped around the centre, the obstacle region is dilated and its boundary
    extracted; the pixel coordinates are then converted to scene units.
    """
    from .iris_solver import extract_obstacle_constraints

    occ = np.asarray(occ)
    h, w = occ.shape
    x0, x1, y0, y1 = extent
    dx = (x1 - x0) / w
    dy = (y1 - y0) / h
    cx, cy = float(center_scene[0]), float(center_scene[1])

    half_x = int(np.ceil(window_half / dx))
    half_y = int(np.ceil(window_half / dy))
    pxc = (cx - x0) / (x1 - x0) * w - 0.5
    pyc = (cy - y0) / (y1 - y0) * h - 0.5
    gx0 = max(0, int(np.round(pxc)) - half_x)
    gx1 = min(w, int(np.round(pxc)) + half_x + 1)
    gy0 = max(0, int(np.round(pyc)) - half_y)
    gy1 = min(h, int(np.round(pyc)) + half_y + 1)
    if gx1 <= gx0 or gy1 <= gy0:
        return np.empty((0, 2), dtype=float)

    sub = (occ[gy0:gy1, gx0:gx1] > 0.5).astype(np.uint8)
    _, pts = extract_obstacle_constraints(sub, dilation_iters=int(dilation),
                                          boundary_jitter=int(boundary_jitter))
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)
    gx = pts[:, 0] + gx0
    gy = pts[:, 1] + gy0
    sx = x0 + (gx + 0.5) * dx
    sy = y0 + (gy + 0.5) * dy
    return np.column_stack([sx, sy]).astype(float)


def solve_fixed_center_iris(obs_points: np.ndarray, center_scene,
                            bounds=DEFAULT_EXTENT, K_bins: int = 32,
                            max_iters: int = 15, growth_tol: float = 0.05,
                            min_radius: float = 1e-3) -> np.ndarray:
    """Return the inellipse matrix P (2x2, PSD) around the FIXED centre.

    Ellipse boundary: x = P u + c, ||u|| <= 1, so the singular values of P are
    the semi-axis lengths.  The centre is never optimised: it is a constant in
    the constraint ||P A_j|| + A_j^T c <= b_j.
    """
    if cp is None:  # pragma: no cover
        raise ImportError("cvxpy is required for fixed-centre IRIS labels")

    c = np.asarray(center_scene, dtype=float).reshape(2)
    xmin, xmax, ymin, ymax = bounds
    obs = np.asarray(obs_points, dtype=float).reshape(-1, 2)

    if len(obs) == 0:
        radius = min(c[0] - xmin, xmax - c[0], c[1] - ymin, ymax - c[1])
        return np.eye(2) * max(float(radius), min_radius)

    dists = np.linalg.norm(obs - c, axis=1)
    P_val = np.eye(2) * max(float(dists.min()) * 0.5, min_radius)

    for _ in range(int(max_iters)):
        try:
            P_inv = np.linalg.inv(P_val)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate
            break
        P_inv2 = P_inv.T @ P_inv
        shifted = obs - c
        trans = shifted @ P_inv.T
        d = np.linalg.norm(trans, axis=1)
        ang = np.arctan2(trans[:, 1], trans[:, 0])
        bins = np.clip(np.floor((ang + np.pi) / (2 * np.pi) * K_bins).astype(int),
                       0, K_bins - 1)

        A = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        b = [xmax, -xmin, ymax, -ymin]
        for k in range(int(K_bins)):
            mask = bins == k
            if not mask.any():
                continue
            p = obs[mask][int(np.argmin(d[mask]))]
            n = P_inv2 @ (p - c)
            norm_n = float(np.linalg.norm(n))
            if norm_n > 1e-9:
                n = n / norm_n
                A.append([float(n[0]), float(n[1])])
                b.append(float(n @ p))
        A_mat = np.asarray(A, dtype=float)
        b_vec = np.asarray(b, dtype=float)

        P = cp.Variable((2, 2), PSD=True)
        constraints = [cp.norm(P @ A_mat[j]) + A_mat[j] @ c <= b_vec[j]
                       for j in range(len(A_mat))]
        prob = cp.Problem(cp.Maximize(cp.log_det(P)), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate") or P.value is None:
                prob.solve(solver=cp.SCS, max_iters=1000, eps=1e-4, verbose=False)
        except Exception:
            break
        if (prob.status not in ("optimal", "optimal_inaccurate")
                or P.value is None):
            break
        P_new = np.asarray(P.value, dtype=float)
        vol_old = float(np.linalg.det(P_val))
        vol_new = float(np.linalg.det(P_new))
        P_val = P_new
        if vol_old > 1e-9 and (vol_new - vol_old) / vol_old < growth_tol:
            break
    return P_val


def P_to_shape4(P: np.ndarray) -> np.ndarray:
    """Inellipse P -> [log a, log b, cos 2t, sin 2t] with a >= b (major first)."""
    P = np.asarray(P, dtype=float).reshape(2, 2)
    P = 0.5 * (P + P.T)
    w, v = np.linalg.eigh(P)
    big = int(np.argmax(w))
    small = 1 - big
    a = float(max(w[big], 1e-8))
    b = float(max(w[small], 1e-8))
    v1 = v[:, big]
    theta = float(np.arctan2(v1[1], v1[0]))
    return np.array([np.log(a), np.log(b), np.cos(2 * theta), np.sin(2 * theta)],
                    dtype=np.float64)


def shape4_to_P(shape4) -> np.ndarray:
    """Inverse of P_to_shape4 (used by the safety checks and the plots)."""
    shape4 = np.asarray(shape4, dtype=float).reshape(4)
    a = float(np.exp(shape4[0]))
    b = float(np.exp(shape4[1]))
    theta = 0.5 * float(np.arctan2(shape4[3], shape4[2]))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R @ np.diag([a, b]) @ R.T


def ellipse_is_inside_free(P, center_scene, occ, extent=DEFAULT_EXTENT,
                           boundary_samples: int = 64,
                           interior_samples: int = 64, seed: int = 0) -> bool:
    """Conservative check that the whole ellipse lies in free map cells."""
    occ = np.asarray(occ)
    h, w = occ.shape
    x0, x1, y0, y1 = extent
    P = np.asarray(P, dtype=float)
    c = np.asarray(center_scene, dtype=float).reshape(2)
    ang = np.linspace(0.0, 2.0 * np.pi, int(boundary_samples), endpoint=False)
    unit = np.vstack([np.cos(ang), np.sin(ang)])
    pts = (P @ unit).T + c
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(int(interior_samples)))
    th = rng.uniform(0.0, 2.0 * np.pi, int(interior_samples))
    inner = np.vstack([r * np.cos(th), r * np.sin(th)])
    pts = np.vstack([pts, (P @ inner).T + c])
    ix = np.floor((pts[:, 0] - x0) / (x1 - x0) * w).astype(int)
    iy = np.floor((pts[:, 1] - y0) / (y1 - y0) * h).astype(int)
    if np.any(ix < 0) or np.any(ix >= w) or np.any(iy < 0) or np.any(iy >= h):
        return False
    return not occ[iy, ix].astype(bool).any()


def fixed_center_shape4(occ: np.ndarray, center_scene, window_half: float = 0.25,
                        safety_dilation: int = 1, boundary_jitter: int = 1,
                        extent=DEFAULT_EXTENT, K_bins: int = 32,
                        max_iters: int = 15, shrink: float = 0.85,
                        max_shrink_steps: int = 8,
                        require_safe: bool = True
                        ) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
    """Full label pipeline at one fixed centre.

    Returns (shape4 or None, P, safe).  When the solved ellipse would touch an
    obstacle cell it is shrunk until it is safe, so the stored label can never
    be a collision.
    """
    obs = scene_obstacle_points(occ, center_scene, window_half=window_half,
                                dilation=safety_dilation,
                                boundary_jitter=boundary_jitter, extent=extent)
    P = solve_fixed_center_iris(obs, center_scene, bounds=extent, K_bins=K_bins,
                                max_iters=max_iters)
    safe = ellipse_is_inside_free(P, center_scene, occ, extent=extent)
    steps = 0
    while require_safe and not safe and steps < int(max_shrink_steps):
        P = P * float(shrink)
        steps += 1
        safe = ellipse_is_inside_free(P, center_scene, occ, extent=extent)
    if not np.isfinite(P).all():
        return None, P, False
    return P_to_shape4(P), P, bool(safe or not require_safe)
