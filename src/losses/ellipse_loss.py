"""Ellipse losses: parameter regression (center/radius/orientation), soft IoU,
ellipse-collision, anchor containment, and trajectory-ellipse alignment.

All values are in the scene frame [-1,1]^2.
"""

import torch
import torch.nn.functional as F

from src.geometry.scene_frame import sample_sdf_scene

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


def _local_grid(pred_center, pred_r1, pred_r2, cfg_loss, grid_n=21, gt_r=None):
    """Build a local grid around each predicted ellipse centre.

    Returns (grid_pts [B,N,G*G,2], s [B,N]).
    """
    margin = cfg_loss.get("ellipse_margin", 1.0)
    max_r = torch.maximum(pred_r1, pred_r2)
    if gt_r is not None:
        max_r = torch.maximum(max_r, torch.maximum(gt_r[..., 0], gt_r[..., 1]))
    s = (1.5 * max_r + margin).clamp(min=0.5)                          # [B,N]
    off = torch.linspace(-1.0, 1.0, grid_n, device=pred_center.device)
    ox, oy = torch.meshgrid(off, off, indexing="ij")
    off2 = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)       # [G*G,2]
    grid_pts = pred_center[..., None, :] + off2[None, None, :, :] * s[..., None, None]
    return grid_pts, s


def _pred_soft_mask(grid_pts, pred_center, pred_r1, pred_r2, pred_theta, cfg_loss):
    """Soft ellipse mask on the local grid: M(u) = sigmoid(tau*(1 - (u-c)^T Q (u-c)))."""
    tau = cfg_loss.get("soft_mask_tau", 10.0)
    Q_pred = _build_Q(pred_center, pred_r1, pred_r2, pred_theta)
    c = pred_center[..., None, :]
    dx = grid_pts[..., 0] - c[..., 0]
    dy = grid_pts[..., 1] - c[..., 1]
    quad = (Q_pred[..., 0, 0, None] * dx * dx + 2 * Q_pred[..., 0, 1, None] * dx * dy
            + Q_pred[..., 1, 1, None] * dy * dy)
    return torch.sigmoid(tau * (1.0 - quad))                           # [B,N,G*G]


def ellipse_param_loss(pred_center, pred_r1, pred_r2, pred_dir,
                       gt_center, gt_r, gt_Q, gt_valid, cfg_loss):
    """Parameter losses: center (SmoothL1), radii (MSE), orientation (double-angle MSE).

    The angle term is masked for near-circular ellipses (angle is meaningless).
    """
    valid = gt_valid.float()
    denom = valid.sum() + EPS
    l_c = (F.smooth_l1_loss(pred_center, gt_center, reduction="none").sum(-1) * valid).sum() / denom
    l_r = (((pred_r1 - gt_r[..., 0]) ** 2 + (pred_r2 - gt_r[..., 1]) ** 2) * valid).sum() / denom

    v_gt = _gt_double_angle(gt_Q)
    if cfg_loss.get("near_circle_angle_mask", True):
        tau = cfg_loss.get("near_circle_tau", 0.1)
        ecc = (pred_r1 - pred_r2) / (pred_r1 + pred_r2 + EPS)
        angle_valid = (ecc > tau).float()
    else:
        angle_valid = torch.ones_like(pred_r1)
    denom_t = (valid * angle_valid).sum() + EPS
    l_t = (((pred_dir - v_gt) ** 2).sum(-1) * valid * angle_valid).sum() / denom_t

    lam_c = cfg_loss.get("lambda_c", 1.0)
    lam_r = cfg_loss.get("lambda_r", 1.0)
    lam_t = cfg_loss.get("lambda_theta", 1.0)
    return {
        "L_param": lam_c * l_c + lam_r * l_r + lam_t * l_t,
        "L_center": l_c, "L_radius": l_r, "L_theta": l_t,
    }


def ellipse_align_loss(pred_r1, pred_r2, pred_theta, pos_pred, gt_valid=None,
                       near_circle_tau=0.1, mask_near_circle=True):
    """Trajectory-ellipse alignment: 1 - |v_k^T a_k| with near-circular masking.

    v_k is the unit tangent of the predicted clean trajectory at interior points,
    and a_k = [cos theta, sin theta] is the predicted ellipse long-axis direction.
    """
    H = pos_pred.shape[1]
    # tangent at interior positions 1..H-2
    v = pos_pred[:, 2:] - pos_pred[:, :-2]             # [B, H-2, 2]
    v = v / (v.norm(dim=-1, keepdim=True) + EPS)
    theta_sl = pred_theta[:, :H - 2]                   # [B, H-2]
    a = torch.stack([torch.cos(theta_sl), torch.sin(theta_sl)], dim=-1)
    align = 1.0 - (v * a).sum(-1).abs()

    mask = torch.ones_like(align)
    if mask_near_circle:
        r1_sl = pred_r1[:, :H - 2]
        r2_sl = pred_r2[:, :H - 2]
        near = (r1_sl - r2_sl) / (r1_sl + r2_sl + EPS)
        mask = (near > near_circle_tau).float()
    if gt_valid is not None:
        mask = mask * gt_valid[:, :H - 2].float()
    denom = mask.sum() + EPS
    if denom.detach().item() <= 1e-6:
        return torch.zeros((), device=pos_pred.device, requires_grad=True)
    return (align * mask).sum() / denom


def ellipse_axis_loss(pred_center, pred_r1, pred_r2, pred_theta, pos_pred,
                      gt_valid=None, cfg_loss=None, **kwargs):
    """Backward-compatible alias for the alignment loss."""
    del pred_center
    near_circle_tau = (cfg_loss.get("near_circle_tau", 0.1)
                       if cfg_loss is not None else 0.1)
    return ellipse_align_loss(pred_r1, pred_r2, pred_theta, pos_pred,
                              gt_valid=gt_valid, near_circle_tau=near_circle_tau)


def ellipse_iou_loss(pred_center, pred_r1, pred_r2, pred_theta,
                     gt_center, gt_r, gt_Q, gt_valid, cfg_loss, grid_n=21):
    """Soft IoU between the predicted ellipse mask and the GT ellipse mask."""
    valid = gt_valid.float()
    grid_pts, s = _local_grid(pred_center, pred_r1, pred_r2, cfg_loss, grid_n, gt_r=gt_r)
    M_pred = _pred_soft_mask(grid_pts, pred_center, pred_r1, pred_r2, pred_theta, cfg_loss)
    gpx = grid_pts[..., 0]
    gpy = grid_pts[..., 1]
    dxg = gpx - gt_center[..., 0:1]
    dyg = gpy - gt_center[..., 1:2]
    quad_gt = (gt_Q[..., 0, 0, None] * dxg * dxg + 2 * gt_Q[..., 0, 1, None] * dxg * dyg
               + gt_Q[..., 1, 1, None] * dyg * dyg)
    M_gt = (quad_gt <= 1.0).float()                    # [B,N,G*G]
    inter = (M_pred * M_gt).sum(-1)
    union = (M_pred + M_gt - M_pred * M_gt).sum(-1)
    iou = inter / (union + 1e-8)
    l_iou = ((1.0 - iou) * valid).sum() / (valid.sum() + 1e-8)
    return l_iou


def ellipse_collision_loss(pred_center, pred_r1, pred_r2, pred_theta,
                           sdf_tensor, gt_valid, cfg_loss, grid_n=21):
    """Penalise the predicted soft ellipse mask overlapping obstacles."""
    valid = gt_valid.float()
    grid_pts, s = _local_grid(pred_center, pred_r1, pred_r2, cfg_loss, grid_n)
    M_pred = _pred_soft_mask(grid_pts, pred_center, pred_r1, pred_r2, pred_theta, cfg_loss)
    B, N, _ = pred_center.shape
    gp = grid_pts.reshape(B, N * grid_n * grid_n, 2)
    d = sample_sdf_scene(sdf_tensor, gp)               # [B, N*G*G]
    d = d.reshape(B, N, grid_n * grid_n)
    O = (d <= 0.0).float()
    coll = (M_pred * O).sum(-1) / (grid_n * grid_n)    # [B,N]
    l = (coll * valid).sum() / (valid.sum() + EPS)
    return l


def ellipse_anchor_loss(pred_center, pred_r1, pred_r2, pred_theta, pred_anchor,
                        gt_valid, cfg_loss):
    """Anchor containment: ReLU[(p_hat - c)^T Q (p_hat - c) - 1]."""
    del cfg_loss
    valid = gt_valid.float()
    Q = _build_Q(pred_center, pred_r1, pred_r2, pred_theta)
    d = pred_anchor - pred_center
    quad = torch.einsum("bni,bnij,bnj->bn", d, Q, d)
    loss = torch.relu(quad - 1.0)
    return (loss * valid).sum() / (valid.sum() + EPS)
