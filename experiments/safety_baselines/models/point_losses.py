"""Coordinate losses on direct trajectory points; no spline operations."""
import torch
import torch.nn.functional as F
from src.models.trajsafe.boundary import boundary_targets

def point_x0_loss(q_hat: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Direct point-coordinate loss with no decoding or fitting."""
    if q_hat.shape != q_gt.shape:
        raise ValueError("point shapes differ: %s vs %s"
                         % (tuple(q_hat.shape), tuple(q_gt.shape)))
    interior_hat, interior_gt = q_hat[:, 1:-1], q_gt[:, 1:-1]
    if interior_hat.numel() == 0:              # degenerate C == 2
        return q_hat.sum() * 0.0
    return F.mse_loss(interior_hat, interior_gt)

def point_smoothness_loss(q_pred: torch.Tensor, q_gt: torch.Tensor,
                            acc_weight: float = 0.25,
                            jerk_weight: float = 1.0,
                            eps: float = 1e-3) -> torch.Tensor:
    """Direct point-coordinate loss with no decoding or fitting."""
    velocity = q_pred[:, 1:] - q_pred[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]

    gt_velocity = q_gt[:, 1:] - q_gt[:, :-1]
    step_scale = gt_velocity.norm(dim=-1).mean(dim=1, keepdim=True)
    step_scale = step_scale.detach().clamp_min(1e-4)[:, :, None]
    acceleration = acceleration / step_scale
    jerk = jerk / step_scale

    acc_norm = (acceleration.square().sum(dim=-1) + eps ** 2).sqrt().sub(eps)
    jerk_norm = (jerk.square().sum(dim=-1) + eps ** 2).sqrt().sub(eps)
    loss_acc = torch.log1p(acc_norm).mean()
    loss_jerk = torch.log1p(jerk_norm).mean()
    return acc_weight * loss_acc + jerk_weight * loss_jerk

def boundary_point_loss(q_raw: torch.Tensor, q_gt: torch.Tensor,
                          cond: torch.Tensor, ws: torch.Tensor,
                          wg: torch.Tensor) -> torch.Tensor:
    """Direct point-coordinate loss with no decoding or fitting."""
    if q_raw.shape != q_gt.shape:
        raise ValueError("point shapes differ: %s vs %s"
                         % (tuple(q_raw.shape), tuple(q_gt.shape)))
    target_s, target_g = boundary_targets(q_gt, cond)
    ws = ws.to(q_raw.dtype)[None, :, None]
    wg = wg.to(q_raw.dtype)[None, :, None]
    err_s = ((q_raw - target_s) ** 2).sum(dim=-1) * ws[:, :, 0]
    err_g = ((q_raw - target_g) ** 2).sum(dim=-1) * wg[:, :, 0]
    denom = (ws.sum() + wg.sum()).clamp_min(1e-9)
    per_sample = (err_s.sum(dim=1) + err_g.sum(dim=1)) / denom
    return per_sample.mean()
