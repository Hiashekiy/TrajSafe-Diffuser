"""Augmented-Lagrangian segment safety loss for Phase 3 joint fine-tuning.

Implements the V5 design:
    L_AL = lambda_dual * mean(V) + (rho / 2) * mean(Q)
with a soft active-set over the boundary halfspaces of the local convex region,
a per-sample diffusion timestep gate, and a detached convex region (A,b) so the
gradient only flows to the predicted trajectory (p -> z0 -> trajectory head).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from src.geometry.iris_solver import extract_obstacle_constraints
from src.geometry.convex_region import ellipse_params_to_shape, generate_convex_region

EPS = 1e-8

# scene maze order (matches src/datasets/scene_dataset.py MAZE_NAMES)
_MAZE_NAMES = ["umaze", "medium", "large"]

# obstacle boundary point cache, keyed by (maze_id, dilation, boundary_jitter).
# Scene maps are fixed per maze id, so we extract obstacle points once and reuse
# them across batches/epochs.
_OBSTACLE_POINT_CACHE: dict = {}


def al_timestep_weight(t, T, start_ratio=0.60, full_ratio=0.20):
    """Per-sample w_AL(t) in [0,1].  Reverse diffusion: t goes large -> small."""
    t = torch.as_tensor(t, dtype=torch.float32)
    t_start = float(start_ratio) * (T - 1)
    t_full = float(full_ratio) * (T - 1)
    w = (t_start - t) / max(t_start - t_full, 1e-6)
    return w.clamp(0.0, 1.0)


def _scene_obstacle_points(occ, extent=(-1.0, 1.0, -1.0, 1.0),
                           dilation=1, boundary_jitter=1, cache_key=None,
                           cache_path=None):
    """Obstacle boundary points of a full scene occupancy map in scene coords.

    Results are cached by cache_key (e.g. maze_id) in memory; if cache_path is
    given and the .npy already exists it is loaded directly from the dataset.
    """
    if cache_key is not None:
        key = (cache_key, int(dilation), int(boundary_jitter))
        if key in _OBSTACLE_POINT_CACHE:
            return _OBSTACLE_POINT_CACHE[key]

    if cache_path is not None and os.path.exists(cache_path):
        out = np.asarray(np.load(cache_path), dtype=float).reshape(-1, 2)
    else:
        occ = np.asarray(occ)
        H, W = occ.shape
        x0, x1, y0, y1 = extent
        _, pts = extract_obstacle_constraints(
            np.asarray(occ, dtype=np.uint8),
            dilation_iters=dilation, boundary_jitter=boundary_jitter)
        if len(pts) == 0:
            out = np.empty((0, 2), dtype=float)
        else:
            dx = (float(x1) - float(x0)) / W
            dy = (float(y1) - float(y0)) / H
            gx = pts[:, 0].astype(float)
            gy = pts[:, 1].astype(float)
            sx = x0 + (gx + 0.5) * dx
            sy = y0 + (gy + 0.5) * dy
            out = np.column_stack([sx, sy]).astype(float)
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, out)

    if cache_key is not None:
        _OBSTACLE_POINT_CACHE[(cache_key, int(dilation), int(boundary_jitter))] = out
    return out


def _convex_region_for_ellipse(occ, center, r1, r2, theta, cfg,
                               obs=None, extent=(-1.0, 1.0, -1.0, 1.0)):
    """Build A,b of the local convex region around one predicted ellipse."""
    cx, cy = float(center[0]), float(center[1])
    r1, r2, theta = float(r1), float(r2), float(theta)
    if not (np.isfinite(cx + cy + r1 + r2 + theta) and r1 > 0 and r2 > 0):
        return None, None
    window_half = float(cfg.get("window_half", 0.25))
    safety_margin = float(cfg.get("safety_margin", 0.0))
    include_boundary = bool(cfg.get("include_boundary", True))

    if obs is None:
        obs = _scene_obstacle_points(
            occ, extent=extent,
            dilation=int(cfg.get("dilation", 1)),
            boundary_jitter=int(cfg.get("boundary_jitter", 1)))
    if len(obs):
        sel = (np.abs(obs[:, 0] - cx) <= window_half) &               (np.abs(obs[:, 1] - cy) <= window_half)
        obs_local = obs[sel]
    else:
        obs_local = np.empty((0, 2), dtype=float)

    P = ellipse_params_to_shape([cx, cy], r1, r2, theta)
    return generate_convex_region(
        P, np.array([cx, cy], dtype=float), obs_local,
        bounds=(cx - window_half, cx + window_half,
                cy - window_half, cy + window_half),
        safety_margin=safety_margin,
        include_boundary=include_boundary,
    )


def _segment_al_loss(A, b, p0, p1, beta):
    """Soft active-set AL term for a batch of segments sharing one (A,b).

    A : (M,2) detached; b : (M,).  p0 : (S,2), p1 : (S,2) batched endpoints
    (S segments).  Returns (V0, V1, Q0, Q1) each shape (S,).
    """
    # Move A,b to the segment device once; S small (region_stride).
    A = torch.as_tensor(A, dtype=torch.float32, device=p0.device)
    b = torch.as_tensor(b, dtype=torch.float32, device=p0.device)
    norms = torch.linalg.norm(A, dim=-1, keepdim=True).clamp(min=EPS)
    A_norm = A / norms                       # (M,2)
    b_n = b / norms.squeeze(-1)              # (M,)

    # r = A_norm @ p - b_n, batched over S segments: (S,M) = p @ A_norm.T - b_n
    r0 = p0 @ A_norm.T - b_n                 # (S,M)
    r1 = p1 @ A_norm.T - b_n                 # (S,M)

    alpha0 = torch.softmax(beta * r0, dim=-1)
    alpha1 = torch.softmax(beta * r1, dim=-1)

    v0 = torch.relu(r0)
    v1 = torch.relu(r1)
    V0 = (alpha0 * v0).sum(-1)               # (S,)
    V1 = (alpha1 * v1).sum(-1)
    Q0 = (alpha0 * v0.square()).sum(-1)
    Q1 = (alpha1 * v1.square()).sum(-1)
    return V0, V1, Q0, Q1


def al_safety_loss(pos_pred, ellipse_center, ellipse_radii, ellipse_theta,
                   map_tensor, t, cfg, device, dual, num_timesteps,
                   region_stride=4, maze_ids=None, maps_dir=None):
    """Compute the AL safety loss over predicted ellipses + scene occupancy.

    Returns a dict with L_AL, mean_V, mean_Q, w_active.
    A,b are built from the predicted ellipse with the scene occupancy map and
    detached, so the loss only back-propagates through pos_pred.
    """
    pos_pred = pos_pred.float()
    ellipse_center = ellipse_center.float()
    ellipse_radii = ellipse_radii.float()
    ellipse_theta = ellipse_theta.float()

    H = pos_pred.shape[1]
    Nseg = H - 1

    start_ratio = float(cfg.get("start_t_ratio", 0.60))
    full_ratio = float(cfg.get("full_t_ratio", 0.20))
    w_t = al_timestep_weight(t, num_timesteps, start_ratio, full_ratio).to(device)
    active = (w_t > 0.0).nonzero(as_tuple=False).flatten().tolist()
    if not active:
        zero = torch.zeros((), device=device, requires_grad=False)
        return {"L_AL": zero, "mean_V": zero, "mean_Q": zero, "w_active": 0.0}

    beta = float(cfg.get("softmax_beta", 15.0))
    rho = float(cfg.get("rho", 5.0))
    dual = float(dual)

    V_all, Q_all = [], []
    sample_terms = []   # (w, V0, V1, Q0, Q1) non-detached for the loss
    for b in active:
        occ_np = map_tensor[b, 0].detach().cpu().numpy()
        cache_key = None
        if maze_ids is not None:
            cache_key = int(maze_ids[b].detach().cpu().item())
        cache_path = None
        if maps_dir is not None and cache_key is not None:
            cache_path = os.path.join(maps_dir, f"{_MAZE_NAMES[cache_key]}_obstacle_points.npy")
        obs_map = _scene_obstacle_points(
            occ_np, extent=(-1.0, 1.0, -1.0, 1.0),
            dilation=int(cfg.get("dilation", 1)),
            boundary_jitter=int(cfg.get("boundary_jitter", 1)),
            cache_key=cache_key, cache_path=cache_path)
        # Build one convex region at each anchor ellipse and reuse it for the
        # next region_stride segments (avoids one region construction per segment).
        anchor_stride = max(1, int(region_stride))
        n_anchor = (Nseg + anchor_stride - 1) // anchor_stride
        n_ellipse = ellipse_center.shape[1]
        for ai in range(n_anchor):
            a = ai * anchor_stride
            if a >= n_ellipse:
                break
            center = ellipse_center[b, a].detach().cpu().numpy()
            rad = ellipse_radii[b, a].detach().cpu().numpy()
            theta = ellipse_theta[b, a].detach().cpu().item()
            A, bnd = _convex_region_for_ellipse(occ_np, center, rad[0], rad[1],
                                                theta, cfg, obs=obs_map)
            if A is None or bnd is None or len(A) < 3:
                continue
            seg_end = min(a + anchor_stride, Nseg)
            # Vectorise all segments in this anchor group (they share (A,b)).
            p0 = pos_pred[b, a:seg_end]            # (S,2)
            p1 = pos_pred[b, a + 1:seg_end + 1]    # (S,2)
            V0, V1, Q0, Q1 = _segment_al_loss(A, bnd, p0, p1, beta)   # (S,)
            wb = float(w_t[b].item())
            sample_terms.append((wb, V0, V1, Q0, Q1))
            V_all.extend([V0.detach(), V1.detach()])
            Q_all.extend([Q0.detach(), Q1.detach()])

    if not sample_terms:
        zero = torch.zeros((), device=device, requires_grad=False)
        return {"L_AL": zero, "mean_V": zero, "mean_Q": zero, "w_active": 0.0}

    mean_V = torch.cat(V_all).mean() if V_all else torch.zeros((), device=device)
    mean_Q = torch.cat(Q_all).mean() if Q_all else torch.zeros((), device=device)

    B_total = pos_pred.shape[0]
    L_AL = torch.cat([
        wb * (dual * (V0 + V1) + 0.5 * rho * (Q0 + Q1))
        for wb, V0, V1, Q0, Q1 in sample_terms
    ]).sum() / max(1, B_total)

    return {
        "L_AL": L_AL,
        "mean_V": mean_V,
        "mean_Q": mean_Q,
        "w_active": float(w_t[active].mean().item()),
    }
