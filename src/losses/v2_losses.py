"""V2 loss functions.

    L = L_P + lam_s L_smooth + lam_T L_topo + lam_P L_prog
          + lam_E L_shape + lam_I L_iou + lam_S L_safe

There is deliberately NO centre-safety loss and no lambda_center_safe
(docs/V2.md sections 31/32): the centre is c_i = gamma_m(s_i), a geometric
consequence of the selected topology, so it cannot leave the free space and
there is nothing left for a centre loss to fix.

Every ellipse loss takes the centre and the 4-vector shape as SEPARATE
arguments.  The V1 form ellipse_mask_losses(p_pred, e_pred) computed
center = p_pred + e_pred[..., :2] internally; that construction is gone.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.geometry.ellipse_raster import ellipse_soft_mask
from src.models.skeleton.path_ops import shape4_to_abtheta

__all__ = [
    "trajectory_x0_loss",
    "trajectory_smoothness_loss",
    "topology_soft_ce",
    "progress_loss",
    "ellipse_shape_loss",
    "ellipse_mask_losses",
]


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------


def trajectory_x0_loss(p_hat: torch.Tensor, p_gt: torch.Tensor) -> torch.Tensor:
    """MSE on the interior waypoints (the endpoints are hard-conditioned)."""
    return F.mse_loss(p_hat[:, 1:-1], p_gt[:, 1:-1])


def trajectory_smoothness_loss(p_pred, p_gt, acc_weight=0.25, jerk_weight=1.0,
                               eps=1e-3):
    """Identical to V1 (train.py:trajectory_smoothness_loss).

    Penalises geometric acceleration and high-frequency jerk, scaled by the
    detached mean GT step length.  A test compares this function against the V1
    implementation numerically, so the regularization cannot drift.
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


# ---------------------------------------------------------------------------
# topology / progress / shape
# ---------------------------------------------------------------------------


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def topology_soft_ce(pi: torch.Tensor, q: torch.Tensor,
                     sample_mask: torch.Tensor) -> torch.Tensor:
    """L_topo = -(1/B) sum_b sum_m q_bm log pi_bm  (soft target, never one-hot).

    Invalid candidate slots have pi = 0 and q = 0, so they contribute exactly
    zero (the clamp on pi keeps the 0 * log(0) product finite).
    """
    log_pi = torch.log(pi.clamp_min(1e-12))
    per_sample = -(q * log_pi).sum(dim=-1)
    return _masked_mean(per_sample, sample_mask)


def progress_loss(s: torch.Tensor, s_gt: torch.Tensor,
                  sample_mask: torch.Tensor) -> torch.Tensor:
    """SmoothL1(s, s_GT) averaged over the K ellipse anchors."""
    per = F.smooth_l1_loss(s, s_gt, reduction="none").mean(dim=-1)
    return _masked_mean(per, sample_mask)


def ellipse_shape_loss(shape4: torch.Tensor, shape4_gt: torch.Tensor,
                       shape_valid: torch.Tensor,
                       sample_mask: torch.Tensor) -> torch.Tensor:
    """MSE on [log a, log b, cos 2t, sin 2t], only where a label exists."""
    per = ((shape4 - shape4_gt) ** 2).mean(dim=-1)          # [B,K]
    weight = shape_valid.to(per.dtype) * sample_mask[:, None].to(per.dtype)
    return (per * weight).sum() / weight.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# ellipse raster losses
# ---------------------------------------------------------------------------


def ellipse_mask_losses(center, shape4, occ, gt_mask, raster_res=64, tau=10.0,
                        chunk_size=32, safe_cvar_fraction=0.2,
                        safe_cvar_weight=1.0, eps=1e-6,
                        sample_mask=None):
    """Return (IoU loss, safety loss, safety mean, safety CVaR).

    center [B,K,2] and shape4 [B,K,4] are separate inputs: this function never
    reconstructs the centre from a prediction, so no loss can move it.

    Safety is one minus the fraction of the ellipse's complete theoretical soft
    area that lies in free raster cells, so area outside the map and obstacle
    area are both unsafe.  The CVaR term is computed per trajectory.
    """
    if center.dim() != 3 or center.shape[-1] != 2:
        raise ValueError("center must be [B,K,2], got %s" % (tuple(center.shape),))
    if shape4.dim() != 3 or shape4.shape[-1] != 4:
        raise ValueError("shape4 must be [B,K,4], got %s" % (tuple(shape4.shape),))
    if shape4.shape[:2] != center.shape[:2]:
        raise ValueError("center and shape4 disagree on [B,K]")
    if raster_res <= 0:
        raise ValueError("raster_res must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0.0 < safe_cvar_fraction <= 1.0:
        raise ValueError("safe_cvar_fraction must be in (0, 1]")
    if gt_mask.dim() != 4 or gt_mask.shape[:2] != center.shape[:2]:
        raise ValueError("gt_mask must be [B,K,R,R]")
    if gt_mask.shape[-2:] != (raster_res, raster_res):
        raise ValueError("gt_mask resolution does not match raster_res")

    if occ.dim() == 3:
        occ = occ.unsqueeze(1)
    B, K = center.shape[0], center.shape[1]
    a, b, theta = shape4_to_abtheta(shape4)

    occ_r = F.adaptive_max_pool2d(occ.float(), output_size=(raster_res, raster_res))
    free = 1.0 - occ_r[:, 0, None]                         # [B,1,R,R]
    cell_area = (2.0 / raster_res) ** 2
    tau_t = torch.as_tensor(tau, device=occ.device, dtype=occ.dtype)
    soft_area_factor = F.softplus(tau_t) / tau_t

    iou_sum = center.new_zeros(())
    unsafe_chunks = []
    mask_weight = (torch.ones(B, device=center.device) if sample_mask is None
                   else sample_mask.to(center.dtype))
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        mask = ellipse_soft_mask(center[:, start:end], a[:, start:end],
                                 b[:, start:end], theta[:, start:end],
                                 raster_res, tau)
        target = (gt_mask[:, start:end].to(mask.dtype) / 255.0).detach()
        inter = torch.minimum(mask, target).sum(dim=(-1, -2))
        union = torch.maximum(mask, target).sum(dim=(-1, -2))
        iou_sum = iou_sum + ((1.0 - (inter + eps) / (union + eps))
                             * mask_weight[:, None]).sum()

        free_cells = (mask * free).sum(dim=(-1, -2))
        full_cells = (math.pi * a[:, start:end] * b[:, start:end]
                      * soft_area_factor / cell_area)
        free_ratio = (free_cells / (full_cells + eps)).clamp(0.0, 1.0)
        unsafe_chunks.append(1.0 - free_ratio)

    denom = float(max(mask_weight.sum().item(), 1.0) * K)
    unsafe = torch.cat(unsafe_chunks, dim=1)               # [B,K]
    weighted = unsafe * mask_weight[:, None]
    loss_safe_mean = weighted.sum() / mask_weight.sum().clamp_min(1.0) / K
    tail_count = max(1, math.ceil(K * safe_cvar_fraction))
    cvar = torch.topk(weighted, k=tail_count, dim=1, largest=True,
                      sorted=False).values.mean(dim=1)
    loss_safe_cvar = _masked_mean(cvar, mask_weight)
    loss_safe = loss_safe_mean + safe_cvar_weight * loss_safe_cvar
    return iou_sum / denom, loss_safe, loss_safe_mean, loss_safe_cvar
