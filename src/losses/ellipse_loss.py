"""Ellipse losses: parameter regression (center/radius/orientation) + long-axis
guidance + soft IoU.  All values are in the scene frame [-1,1]^2.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def _gt_double_angle(gt_Q):
    """Derive the GT double-angle vector [cos2t, sin2t] from the ellipse Q.

    The long axis is the eigenvector of Q with the SMALLEST eigenvalue.
    """
    ev = torch.linalg.eigh(gt_Q)
    v = ev.eigenvectors
    t = torch.atan2(v[..., 1, 0], v[..., 0, 0])
    return torch.stack([torch.cos(2 * t), torch.sin(2 * t)], dim=-1)


def _build_Q(center, r1, r2, theta, eps=1e-8):
    """Ellipse quadratic form Q [B,N,2,2] from center, radii, theta (scene coords)."""
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    R = torch.stack([torch.stack([cos, -sin], dim=-1),
                     torch.stack([sin, cos], dim=-1)], dim=-2)   # [B,N,2,2]
    D = torch.stack([1.0 / (r1 ** 2 + eps), 1.0 / (r2 ** 2 + eps)], dim=-1)
    D = torch.diag_embed(D)
    return R @ D @ R.transpose(-1, -2)


def ellipse_param_loss(pred_center, pred_r1, pred_r2, pred_dir,
                       gt_center, gt_r, gt_Q, gt_valid, cfg_loss):
    """Parameter losses: center (SmoothL1), radii (MSE), orientation (double-angle MSE)."""
    valid = gt_valid.float()
    denom = valid.sum() + EPS
    l_c = (F.smooth_l1_loss(pred_center, gt_center, reduction="none").sum(-1) * valid).sum() / denom
    l_r = (((pred_r1 - gt_r[..., 0]) ** 2 + (pred_r2 - gt_r[..., 1]) ** 2) * valid).sum() / denom
    v_gt = _gt_double_angle(gt_Q)
    l_t = (((pred_dir - v_gt) ** 2).sum(-1) * valid).sum() / denom
    lam_c = cfg_loss.get("lambda_c", 1.0)
    lam_r = cfg_loss.get("lambda_r", 1.0)
    lam_t = cfg_loss.get("lambda_theta", 1.0)
    return {
        "L_param": lam_c * l_c + lam_r * l_r + lam_t * l_t,
        "L_center": l_c, "L_radius": l_r, "L_theta": l_t,
    }


def ellipse_axis_loss(pred_center, pred_r1, pred_r2, pred_theta, pos_pred):
    """Long-axis guidance: pull the intermediate trajectory toward the predicted
    major axis of each waypoint safe ellipse.  Near-circular ellipses are masked.
    """
    u2 = torch.stack([-torch.sin(pred_theta), torch.cos(pred_theta)], dim=-1)
    d_axis = (u2 * (pos_pred - pred_center)).sum(-1)
    e = torch.sqrt((1.0 - pred_r2 ** 2 / (pred_r1 ** 2 + EPS)).clamp(min=0.0))
    mask = (e > 1e-2).float()
    mid = slice(0, -1)
    em = e[:, mid] * mask[:, mid]
    num = (em * F.smooth_l1_loss(d_axis[:, mid], torch.zeros_like(d_axis[:, mid]),
                                 reduction="none")).sum()
    den = em.sum() + EPS
    if den.detach().item() <= 1e-6:
        return torch.zeros((), device=d_axis.device, requires_grad=True)
    return num / den


def ellipse_iou_loss(pred_center, pred_r1, pred_r2, pred_theta,
                     gt_center, gt_r, gt_Q, gt_valid, cfg_loss, grid_n=21):
    """Soft IoU between the predicted ellipse mask and the GT ellipse mask.

    Region-level loss that jointly constrains center / radius / orientation.
    """
    valid = gt_valid.float()
    margin = cfg_loss.get("ellipse_margin", 1.0)
    max_r = torch.maximum(torch.maximum(pred_r1, pred_r2),
                          torch.maximum(gt_r[..., 0], gt_r[..., 1]))   # [B,N]
    s = (1.5 * max_r + margin).clamp(min=0.5)                          # [B,N]
    off = torch.linspace(-1.0, 1.0, grid_n, device=pred_center.device)
    ox, oy = torch.meshgrid(off, off, indexing="ij")
    off2 = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)       # [G*G,2]
    grid_pts = pred_center[..., None, :] + off2[None, None, :, :] * s[..., None, None]
    gpx = grid_pts[..., 0]
    gpy = grid_pts[..., 1]
    Q_pred = _build_Q(pred_center, pred_r1, pred_r2, pred_theta)
    dx = gpx - pred_center[..., 0:1]
    dy = gpy - pred_center[..., 1:2]
    quad_pred = (Q_pred[..., 0, 0, None] * dx * dx + 2 * Q_pred[..., 0, 1, None] * dx * dy
                 + Q_pred[..., 1, 1, None] * dy * dy)
    M_pred = torch.sigmoid(10.0 * (1.0 - quad_pred))                    # [B,N,G*G]
    dxg = gpx - gt_center[..., 0:1]
    dyg = gpy - gt_center[..., 1:2]
    quad_gt = (gt_Q[..., 0, 0, None] * dxg * dxg + 2 * gt_Q[..., 0, 1, None] * dxg * dyg
               + gt_Q[..., 1, 1, None] * dyg * dyg)
    M_gt = (quad_gt <= 1.0).float()                                    # [B,N,G*G]
    inter = (M_pred * M_gt).sum(-1)
    union = (M_pred + M_gt - M_pred * M_gt).sum(-1)
    iou = inter / (union + 1e-8)
    l_iou = ((1.0 - iou) * valid).sum() / (valid.sum() + 1e-8)
    return l_iou
