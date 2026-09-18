"""V3 losses (docs sections 24-29).

    L = lam_traj  L_traj      final x0 vs GT (interior waypoints)
      + lam_coarse L_coarse   coarse x0 vs GT (interior waypoints)
      + lam_smooth L_smooth   V1 regulariser (identical formula)
      + lam_topo   L_topo     CE(pi, m*)          - no soft target any more
      + lam_align  L_align    SmoothL1(gamma_m(s_i), p_i^*)   - no progress GT
      + lam_gap    L_gap      gap smoothness (weak)
      + lam_safe   L_safe     full-ellipse occupancy safety (mean + CVaR)
      + lam_area   L_area     push the ellipse to grow, normalised by a_max b_max
      + lam_ratio  L_ratio    aspect-ratio cap (weak)
      + lam_inside L_inside   trajectory inside the ellipse (off by default)

Deleted relative to V2: L_shape (no fixed-centre IRIS GT), the GT IoU term (the
centre moves with gamma_m(s)), and all progress_gt supervision.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.geometry.ellipse_raster import ellipse_soft_mask
from src.losses.v2_losses import trajectory_smoothness_loss

__all__ = [
    "trajectory_smoothness_loss",
    "trajectory_x0_loss",
    "topology_ce",
    "align_loss",
    "gap_loss",
    "ellipse_safety_loss",
    "ellipse_area_loss",
    "ellipse_ratio_loss",
    "ellipse_inside_loss",
]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the valid entries; the mask broadcasts over trailing dims."""
    mask = mask.to(values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def trajectory_x0_loss(p_hat: torch.Tensor, p_gt: torch.Tensor) -> torch.Tensor:
    """MSE on the interior waypoints (endpoints are hard-conditioned)."""
    return F.mse_loss(p_hat[:, 1:-1], p_gt[:, 1:-1])


def topology_ce(pi: torch.Tensor, best: torch.Tensor,
                sample_mask: torch.Tensor) -> torch.Tensor:
    """L_topo = CE(pi, m*): cross entropy against the single nDTW-best label."""
    log_pi = torch.log(pi.clamp_min(1e-12))
    per = -log_pi.gather(1, best[:, None].long()).squeeze(1)
    return _masked_mean(per, sample_mask)


def align_loss(centers: torch.Tensor, p_gt: torch.Tensor,
               sample_mask: torch.Tensor) -> torch.Tensor:
    """L_align = mean_i SmoothL1(c_i, p_i^*).

    The centres are pulled onto the GT trajectory while staying ON the skeleton,
    because c_i = gamma_m(s_i) is a point of the committed chain by construction.
    """
    per = F.smooth_l1_loss(centers, p_gt, reduction="none").mean(dim=-1)
    return _masked_mean(per, sample_mask)


def gap_loss(s: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    """Weak regulariser against wildly uneven progress gaps."""
    if s.shape[1] < 3:
        return s.new_zeros(())
    delta = s[:, 1:] - s[:, :-1]
    per = (delta[:, 1:] - delta[:, :-1]).pow(2).mean(dim=-1)
    return _masked_mean(per, sample_mask)


def ellipse_safety_loss(center, a, b, theta, occ, raster_res=64, tau=10.0,
                        chunk_size=32, cvar_fraction=0.2, cvar_weight=1.0,
                        eps=1e-6, sample_mask=None):
    """Mean + CVaR of 1 - free-area-fraction over the full ellipses.

    Unchanged in spirit from V1/V2 (docs section 24): the whole soft ellipse is
    rasterised, the map is conservatively max-pooled the same way, and anything
    outside [-1,1]^2 counts as unsafe.  No random point sampling.
    """
    if occ.dim() == 3:
        occ = occ.unsqueeze(1)
    B, K = center.shape[0], center.shape[1]
    occ_r = F.adaptive_max_pool2d(occ.float(), output_size=(raster_res, raster_res))
    free = 1.0 - occ_r[:, 0, None]
    cell_area = (2.0 / raster_res) ** 2
    tau_t = torch.as_tensor(tau, device=occ.device, dtype=occ.dtype)
    soft_area_factor = F.softplus(tau_t) / tau_t
    weight = (torch.ones(B, device=center.device) if sample_mask is None
              else sample_mask.to(center.dtype))
    unsafe_chunks = []
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        mask = ellipse_soft_mask(center[:, start:end], a[:, start:end],
                                 b[:, start:end], theta[:, start:end],
                                 raster_res, tau)
        free_cells = (mask * free).sum(dim=(-1, -2))
        full_cells = (math.pi * a[:, start:end] * b[:, start:end]
                      * soft_area_factor / cell_area)
        free_ratio = (free_cells / (full_cells + eps)).clamp(0.0, 1.0)
        unsafe_chunks.append(1.0 - free_ratio)
    unsafe = torch.cat(unsafe_chunks, dim=1)
    weighted = unsafe * weight[:, None]
    mean = weighted.sum() / weight.sum().clamp_min(1.0) / float(K)
    tail = max(1, math.ceil(K * cvar_fraction))
    cvar = torch.topk(weighted, k=tail, dim=1, largest=True,
                      sorted=False).values.mean(dim=1)
    cvar = _masked_mean(cvar, weight)
    return mean + cvar_weight * cvar, mean, cvar


def ellipse_area_loss(a: torch.Tensor, b: torch.Tensor, a_max: float,
                      b_max: float = None, sample_mask=None) -> torch.Tensor:
    """L_area = masked_mean(1 - a_i b_i / (a_max b_max)): bounded, so it cannot
    diverge, and exactly 0 when nothing is valid.

    The normaliser is the LARGEST AREA the head can produce, a_max * b_max, not
    a_max^2.  With b_max = 0.30 and a_max = 0.80 the old normaliser could never
    exceed 0.375, so the loss could not reach 0 even with maximal ellipses.
    """
    if b_max is None:
        b_max = a_max
    per = 1.0 - (a * b) / (float(a_max) * float(b_max))
    per = per.mean(dim=1)
    if sample_mask is None:
        return per.mean()
    return _masked_mean(per, sample_mask)


def ellipse_ratio_loss(a: torch.Tensor, b: torch.Tensor, r_max: float = 4.0,
                       sample_mask=None) -> torch.Tensor:
    per = F.relu(a / b.clamp_min(1e-6) - float(r_max)).pow(2).mean(dim=1)
    if sample_mask is None:
        return per.mean()
    return _masked_mean(per, sample_mask)


def ellipse_inside_loss(center, a, b, theta, points, rho=0.8,
                        sample_mask=None) -> torch.Tensor:
    """Softplus(q - rho), q being the ellipse-frame quadratic form of the points."""
    rel = points - center
    ct, st = torch.cos(theta), torch.sin(theta)
    xr = ct * rel[..., 0] + st * rel[..., 1]
    yr = -st * rel[..., 0] + ct * rel[..., 1]
    q = (xr / a.clamp_min(1e-6)) ** 2 + (yr / b.clamp_min(1e-6)) ** 2
    per = F.softplus(q - float(rho)).mean(dim=1)
    if sample_mask is None:
        return per.mean()
    return _masked_mean(per, sample_mask)
