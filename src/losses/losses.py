"""Losses for the control-space TrajSafe-Diffuser.

    L = lambda_traj    * L_traj      (MSE on the decoded 128-point curve)
      + lambda_control * L_ctrl      (MSE on the 30 interior B-spline controls)
      + lambda_coarse  * L_coarse
      + lambda_smooth  * L_smooth
      + lambda_topo    * L_topo
      + lambda_shape   * L_shape
      + lambda_iou     * L_iou
      + lambda_safe    * L_safe

There is NO alignment loss: the ellipse centre is the FIXED Skeleton centre
``c_i = Gamma_m(i/127)``, so there is nothing to align and nothing to predict.
No progress target, no centre target and no alignment weight exist anywhere in
the training chain.

``L_shape`` is the ellipse-parameter loss and touches only the Ellipse Shape
Head; the centre is deliberately NOT part of it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.geometry.ellipse_raster import ellipse_soft_mask

__all__ = [
    "trajectory_smoothness_loss",
    "trajectory_x0_loss",
    "control_x0_loss",
    "topology_ce",
    "ellipse_shape_loss",
    "ellipse_iou_loss",
    "ellipse_safety_loss",
]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the valid ENTRIES; the mask broadcasts over trailing dims."""
    mask = mask.to(values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def trajectory_x0_loss(p_hat: torch.Tensor, p_gt: torch.Tensor) -> torch.Tensor:
    """MSE on the interior waypoints (endpoints are hard-conditioned)."""
    return F.mse_loss(p_hat[:, 1:-1], p_gt[:, 1:-1])


def control_x0_loss(q_hat: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """MSE on the 30 INTERIOR B-spline controls.

    The first and the last control are hard-conditioned to start / goal
    (clamped knots => curve endpoints), so they carry no training signal and
    must not contribute to the loss.
    """
    if q_hat.shape != q_gt.shape:
        raise ValueError("control shapes differ: %s vs %s"
                         % (tuple(q_hat.shape), tuple(q_gt.shape)))
    return F.mse_loss(q_hat[:, 1:-1], q_gt[:, 1:-1])


def trajectory_smoothness_loss(p_pred: torch.Tensor, p_gt: torch.Tensor,
                               acc_weight: float = 0.25,
                               jerk_weight: float = 1.0,
                               eps: float = 1e-3) -> torch.Tensor:
    """Penalise geometric acceleration and high-frequency jerk.

    Both finite differences are scaled by the detached mean GT step length,
    so the regulariser is resolution independent.
    """
    velocity = p_pred[:, 1:] - p_pred[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]

    gt_velocity = p_gt[:, 1:] - p_gt[:, :-1]
    step_scale = gt_velocity.norm(dim=-1).mean(dim=1, keepdim=True)
    step_scale = step_scale.detach().clamp_min(1e-4)[:, :, None]
    acceleration = acceleration / step_scale
    jerk = jerk / step_scale

    acc_norm = (acceleration.square().sum(dim=-1) + eps ** 2).sqrt().sub(eps)
    jerk_norm = (jerk.square().sum(dim=-1) + eps ** 2).sqrt().sub(eps)
    loss_acc = torch.log1p(acc_norm).mean()
    loss_jerk = torch.log1p(jerk_norm).mean()
    return acc_weight * loss_acc + jerk_weight * loss_jerk


def topology_ce(pi: torch.Tensor, best: torch.Tensor,
                sample_mask: torch.Tensor) -> torch.Tensor:
    """L_topo = CE(pi, m*) with m* = argmin_m nDTW(P0, S_m)."""
    log_pi = torch.log(pi.clamp_min(1e-12))
    per = -log_pi.gather(1, best[:, None].long()).squeeze(1)
    return _masked_mean(per, sample_mask)


def ellipse_shape_loss(shape4: torch.Tensor, shape4_gt: torch.Tensor,
                       shape_valid: torch.Tensor,
                       sample_mask: torch.Tensor) -> torch.Tensor:
    """L_shape = 1/N_valid sum_{valid} ||shape_i - shape_i^*||^2."""
    per = ((shape4 - shape4_gt) ** 2).mean(dim=-1)               # [B,H]
    valid = shape_valid & sample_mask[:, None]
    return _masked_mean(per, valid)


def ellipse_iou_loss(center: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                     theta: torch.Tensor, gt_mask: torch.Tensor,
                     shape_valid: torch.Tensor, sample_mask: torch.Tensor,
                     raster_res: int = 64, tau: float = 10.0,
                     eps: float = 1e-6) -> torch.Tensor:
    """Fuzzy soft-IoU loss on the complete ellipse, only where a label exists.

    The prediction is rasterised with the SAME soft rasteriser that produced the
    GT mask.  The GT mask is built at training time from the DETACHED fixed
    Skeleton centres and the offline ``shape4`` labels.
    """
    pred = ellipse_soft_mask(center, a, b, theta, int(raster_res), float(tau))
    gt = gt_mask.to(pred.dtype)
    inter = torch.minimum(pred, gt).sum(dim=(-1, -2))
    union = torch.maximum(pred, gt).sum(dim=(-1, -2))
    per = 1.0 - (inter + float(eps)) / (union + float(eps))       # [B,H]
    valid = shape_valid & sample_mask[:, None]
    return _masked_mean(per, valid)


def ellipse_safety_loss(center, a, b, theta, occ, raster_res=64, tau=10.0,
                        chunk_size=32, cvar_fraction=0.2, cvar_weight=1.0,
                        eps=1e-6, sample_mask=None):
    """Mean + CVaR of ``1 - free-area-fraction`` over the full ellipses.

    ``u_i`` is the unsafe fraction of the complete theoretical soft ellipse and
    ``L_safe = Mean(u) + lambda_CVaR * CVaR(u)`` exactly as in the report.
    """
    if occ.dim() == 3:
        occ = occ.unsqueeze(1)
    B, K = center.shape[0], center.shape[1]
    occ_r = F.adaptive_max_pool2d(occ.float(),
                                  output_size=(int(raster_res), int(raster_res)))
    free = 1.0 - occ_r[:, 0, None]
    cell_area = (2.0 / float(raster_res)) ** 2
    tau_t = torch.as_tensor(tau, device=occ.device, dtype=occ.dtype)
    soft_area_factor = F.softplus(tau_t) / tau_t
    weight = (torch.ones(B, device=center.device) if sample_mask is None
              else sample_mask.to(center.dtype))
    unsafe_chunks = []
    for start in range(0, K, int(chunk_size)):
        end = min(start + int(chunk_size), K)
        mask = ellipse_soft_mask(center[:, start:end], a[:, start:end],
                                 b[:, start:end], theta[:, start:end],
                                 int(raster_res), float(tau))
        free_cells = (mask * free).sum(dim=(-1, -2))
        full_cells = (math.pi * a[:, start:end] * b[:, start:end]
                      * soft_area_factor / cell_area)
        free_ratio = (free_cells / (full_cells + eps)).clamp(0.0, 1.0)
        unsafe_chunks.append(1.0 - free_ratio)
    unsafe = torch.cat(unsafe_chunks, dim=1)                       # [B,K]
    weighted = unsafe * weight[:, None]
    mean = weighted.sum() / weight.sum().clamp_min(1.0) / float(K)
    tail = max(1, math.ceil(K * float(cvar_fraction)))
    cvar = torch.topk(weighted, k=tail, dim=1, largest=True,
                      sorted=False).values.mean(dim=1)
    cvar = _masked_mean(cvar, weight)
    return mean + float(cvar_weight) * cvar, mean, cvar
