"""Losses for the control-space TrajSafe-Diffuser.

    L = lambda_ctrl   * L_ctrl      (raw final controls vs GT controls)
      + lambda_coarse * L_coarse    (raw coarse controls vs GT controls)
      + lambda_smooth * L_smooth    (2nd/3rd differences OF THE CONTROLS)
      + lambda_bound  * L_boundary  (raw controls near start/goal)
      + lambda_topo   * L_topo
      + lambda_shape  * L_shape
      + lambda_iou    * L_iou
      + lambda_safe   * L_safe

There is NO ``L_traj``: the decoded 128-point curve never enters the loss, and
nothing is ever decoded inside the training chain.  ``L_smooth`` is computed on
the CONTROL POLYGON (second and third differences of ``Q``), and ``L_boundary``
supervises the RAW control polygon around both endpoints against the GT local
control-polygon shape translated to this sample's start/goal::

    T^s_i = S + (Q^GT_i - Q^GT_0)
    T^g_j = G + (Q^GT_j - Q^GT_{C-1})

so the network itself has to produce a sane local polygon instead of relying on
the (fixed, untrained) boundary decoder.  The correction profile is read from
``model.boundary_decoder`` - one single definition, never duplicated here.

There is also NO alignment loss: the ellipse centre is the FIXED Skeleton centre
``c_i = Gamma_m(i/(Q-1))``, so there is nothing to align and nothing to predict.
No progress target and no centre target exist anywhere in the training chain.
``L_shape`` touches only the Ellipse Shape Head; the centre is not part of it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.geometry.ellipse_raster import ellipse_soft_mask
from src.models.trajsafe.boundary import boundary_targets

__all__ = [
    "control_x0_loss",
    "control_smoothness_loss",
    "boundary_control_loss",
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


def control_x0_loss(q_hat: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """MSE on the INTERIOR B-spline controls (``Q_1 .. Q_{C-2}``).

    The two endpoint controls are EXCLUDED because, in the control-token model,
    the network's raw predictions ``Q~_0`` / ``Q~_{C-1}`` are NOT hard-
    conditioned: they are free outputs that are (a) supervised by
    ``L_boundary`` against the GT local polygon and (b) made exact afterwards by
    the fixed BoundaryDecoder.  Adding them here would double-count the
    endpoints and pull all of ``Q_1..Q_3`` towards the start.

    Used for BOTH ``L_ctrl`` and ``L_coarse``; both take the RAW (pre boundary
    decoder) control polygon.
    """
    if q_hat.shape != q_gt.shape:
        raise ValueError("control shapes differ: %s vs %s"
                         % (tuple(q_hat.shape), tuple(q_gt.shape)))
    interior_hat, interior_gt = q_hat[:, 1:-1], q_gt[:, 1:-1]
    if interior_hat.numel() == 0:              # degenerate C == 2
        return q_hat.sum() * 0.0
    return F.mse_loss(interior_hat, interior_gt)


def control_smoothness_loss(q_pred: torch.Tensor, q_gt: torch.Tensor,
                            acc_weight: float = 0.25,
                            jerk_weight: float = 1.0,
                            eps: float = 1e-3) -> torch.Tensor:
    """Penalise the second and third differences of the CONTROL polygon.

        Delta^2 Q_i = Q_{i+2} - 2 Q_{i+1} + Q_i
        Delta^3 Q_i = Q_{i+3} - 3 Q_{i+2} + 3 Q_{i+1} - Q_i

    Both are scaled by the detached mean GT control step length, so the
    regulariser does not depend on the control count C.  NOTHING is decoded:
    the control polygon itself has to be smooth.
    """
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


def boundary_control_loss(q_raw: torch.Tensor, q_gt: torch.Tensor,
                          cond: torch.Tensor, ws: torch.Tensor,
                          wg: torch.Tensor) -> torch.Tensor:
    """Weighted local supervision of the RAW controls at both ends.

    Per sample::

        l_b = [ sum_i w^s_i ||Q~_i - T^s_i||^2
              + sum_i w^g_i ||Q~_i - T^g_i||^2 ] / (2 sum_i w_i)

    and the returned value is the MEAN over the batch, so the term is
    batch-size invariant exactly like every other loss in this file (summing
    the batch without dividing would silently scale ``lambda_boundary`` by the
    batch size).

    ``T^s`` / ``T^g`` are the GT control polygons rigidly translated to THIS
    sample's start / goal (see :func:`boundary_targets`), so the loss enforces
    the GT *shape* of the local polygon, not a collapse onto S/G.
    ``ws`` / ``wg`` come from the model's fixed boundary decoder.
    """
    if q_raw.shape != q_gt.shape:
        raise ValueError("control shapes differ: %s vs %s"
                         % (tuple(q_raw.shape), tuple(q_gt.shape)))
    target_s, target_g = boundary_targets(q_gt, cond)
    ws = ws.to(q_raw.dtype)[None, :, None]
    wg = wg.to(q_raw.dtype)[None, :, None]
    err_s = ((q_raw - target_s) ** 2).sum(dim=-1) * ws[:, :, 0]
    err_g = ((q_raw - target_g) ** 2).sum(dim=-1) * wg[:, :, 0]
    denom = (ws.sum() + wg.sum()).clamp_min(1e-9)
    per_sample = (err_s.sum(dim=1) + err_g.sum(dim=1)) / denom
    return per_sample.mean()


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
