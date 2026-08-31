"""Ellipse losses per the method doc (param, IoU, ellipse collision, anchor)."""

import torch
import torch.nn.functional as F


def _build_Q(center, r1, r2, theta, eps=1e-8):
    """center [B,H,2], r1/r2 [B,H], theta [B,H] -> [B,H,2,2]."""
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    R = torch.stack([torch.stack([cos, -sin], dim=-1),
                     torch.stack([sin, cos], dim=-1)], dim=-2)  # [B,H,2,2]
    D = torch.stack([1.0 / (r1 ** 2 + eps), 1.0 / (r2 ** 2 + eps)], dim=-1)  # [B,H,2]
    D = torch.diag_embed(D)
    return R @ D @ R.transpose(-1, -2)


def _sample_occupancy(map_tensor, points, extent):
    """points [B,H,N,N,2] world -> occupancy [B,H,N,N]."""
    x0, x1, y0, y1 = extent
    nx = (points[..., 0] - x0) / (x1 - x0) * 2.0 - 1.0
    ny = (points[..., 1] - y0) / (y1 - y0) * 2.0 - 1.0
    grid = torch.stack([nx, ny], dim=-1)  # [B,H,N,N,2]
    B, H, N, _, _ = grid.shape
    g = grid.reshape(B, H * N * N, 1, 2)
    out = F.grid_sample(map_tensor, g, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.squeeze(1).squeeze(-1).reshape(B, H, N, N)


def ellipse_loss(pred_center, pred_r1, pred_r2, pred_theta, pred_dir,
                 gt_center, gt_r, gt_Q, gt_valid, x0, map_tensor, extent,
                 state_norm, cfg_loss, cfg_geom, grid_n=21):
    """All in world coords except pred_center which is normalized."""
    B, H = pred_r1.shape
    device = pred_r1.device
    valid = gt_valid.float()                # [B,H]
    # convert predicted center (normalized) to world
    dtype = pred_r1.dtype
    mins = torch.as_tensor(state_norm.mins[2:4], device=device, dtype=dtype)
    maxs = torch.as_tensor(state_norm.maxs[2:4], device=device, dtype=dtype)
    epsn = state_norm.eps
    center_world = (pred_center + 1.0) / 2.0 * (maxs - mins + epsn) + mins
    Q_pred = _build_Q(center_world, pred_r1, pred_r2, pred_theta)
    # derive gt theta from gt_Q: direction of largest axis -> eigenvector of Q with smallest eigenvalue
    # compute via 2x2 eigh
    ev = torch.linalg.eigh(gt_Q)
    w = ev.eigenvalues
    v = ev.eigenvectors
    theta_gt = torch.atan2(v[..., 1, 0], v[..., 0, 0])  # axis with smaller eigenvalue
    v_gt = torch.stack([torch.cos(2 * theta_gt), torch.sin(2 * theta_gt)], dim=-1)

    # 1) parameter losses
    l_c = (F.smooth_l1_loss(center_world, gt_center, reduction="none").sum(-1) * valid).sum() / (valid.sum() + 1e-8)
    l_r = (((pred_r1 - gt_r[..., 0]) ** 2 + (pred_r2 - gt_r[..., 1]) ** 2) * valid).sum() / (valid.sum() + 1e-8)
    l_theta = (((pred_dir - v_gt) ** 2).sum(-1) * valid).sum() / (valid.sum() + 1e-8)
    lam_c = cfg_loss.get("lambda_c", 1.0)
    lam_r = cfg_loss.get("lambda_r", 1.0)
    lam_theta = cfg_loss.get("lambda_theta", 1.0)
    L_param = lam_c * l_c + lam_r * l_r + lam_theta * l_theta

    # 2) IoU and ellipse-collision on a local grid
    margin = cfg_loss.get("ellipse_margin", 1.0)
    max_r = torch.maximum(pred_r1, pred_r2)
    s = (1.5 * max_r + margin).clamp(min=0.5)          # [B,H]
    off = torch.linspace(-1.0, 1.0, grid_n, device=device)
    ox, oy = torch.meshgrid(off, off, indexing="ij")
    off2 = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # [N*N,2]
    # grid points [B,H,N*N,2]
    grid_pts = center_world[..., None, :] + off2[None, None, :, :] * s[..., None, None]
    gpx = grid_pts[..., 0]
    gpy = grid_pts[..., 1]

    # quadratic form value for predicted ellipse
    dx = gpx - center_world[..., 0:1]
    dy = gpy - center_world[..., 1:2]
    Q11 = Q_pred[..., 0, 0]
    Q12 = Q_pred[..., 0, 1]
    Q22 = Q_pred[..., 1, 1]
    quad_pred = Q11[..., None] * dx * dx + 2 * Q12[..., None] * dx * dy + Q22[..., None] * dy * dy
    M_pred = torch.sigmoid(10.0 * (1.0 - quad_pred))

    # GT mask (constant, bool)
    dxg = gpx - gt_center[..., 0:1]
    dyg = gpy - gt_center[..., 1:2]
    Qg11 = gt_Q[..., 0, 0]
    Qg12 = gt_Q[..., 0, 1]
    Qg22 = gt_Q[..., 1, 1]
    quad_gt = Qg11[..., None] * dxg * dxg + 2 * Qg12[..., None] * dxg * dyg + Qg22[..., None] * dyg * dyg
    M_gt = (quad_gt <= 1.0).float()

    # occupancy sampled at grid points
    occ = _sample_occupancy(map_tensor, grid_pts.reshape(B, H, grid_n, grid_n, 2), extent)  # [B,H,N,N]
    occ = occ.reshape(B, H, grid_n * grid_n)

    inter = (M_pred * M_gt).sum(-1)
    union = (M_pred + M_gt - M_pred * M_gt).sum(-1)
    iou = inter / (union + 1e-8)
    l_iou = ((1.0 - iou) * valid).sum() / (valid.sum() + 1e-8)

    grid_count = float(grid_n * grid_n)
    l_ecol_raw = ((M_pred * occ).sum(-1) * valid).sum() / (valid.sum() + 1e-8)
    # Normalized (per grid point) so the logged value is ~0.0-1.0 and comparable
    # to the other loss terms, while l_ecol_raw keeps the original semantics.
    l_ecol = l_ecol_raw / grid_count

    # 3) anchor loss: clean waypoint inside predicted ellipse
    p_gt = (x0[:, :, 2:4] + 1.0) / 2.0 * (maxs - mins + epsn) + mins  # [B,H,2]
    d_anchor = (p_gt - center_world)[..., None, :] @ Q_pred @ (p_gt - center_world)[..., :, None]
    d_anchor = d_anchor.squeeze(-1).squeeze(-1)
    l_anchor = ((torch.relu(d_anchor - 1.0)) * valid).sum() / (valid.sum() + 1e-8)

    return {
        "L_param": L_param,
        "L_iou": l_iou,
        "L_ecol": l_ecol,
        "L_ecol_raw": l_ecol_raw,
        "L_anchor": l_anchor,
        "L_center": l_c,
        "L_radius": l_r,
        "L_theta": l_theta,
        "center_world": center_world,
        "Q_pred": Q_pred,
    }


