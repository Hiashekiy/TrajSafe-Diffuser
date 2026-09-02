"""Consensus guidance application for reverse diffusion / joint training.

J_guide is a soft "stay near the local consensus centerline" potential.  During
inference it is applied as a gradient correction to z0_pred; during Phase-3
training it is additionally used as a differentiable objective so that the model
learns trajectories that are already consistent with the consensus geometry.
"""

from __future__ import annotations

import torch

from src.diffusion.zerosum import zero_sum, integrate_positions
from .local_consensus import consensus_guidance_cost

EPS = 1e-8


def guidance_weight(t, T, start_ratio=0.40, full_ratio=0.10):
    """w_G(t) in [0,1].  Reverse diffusion: t goes large -> small."""
    t = torch.as_tensor(t, dtype=torch.float32)
    t_start = float(start_ratio) * (T - 1)
    t_full = float(full_ratio) * (T - 1)
    w = (t_start - t) / max(t_start - t_full, 1e-6)
    return w.clamp(0.0, 1.0)


def apply_consensus_guidance(z0_raw, base, start, center, radii, theta,
                             cfg, t, T, retain_graph=False):
    """Differentiable guidance step on z0 (used at sampling / inference).

    Parameters
    ----------
    z0_raw : [B,N,2] raw predicted residual from the model (detached constant).
    base   : [B,N,2] the deterministic bridge base increments.
    start  : [B,2] start position.
    center/radii/theta : predicted ellipses (detached geometry reference).
    cfg    : consensus_guidance config dict.
    t, T   : current diffusion timestep and total steps.
    retain_graph : whether to keep the graph (only needed for stacked unrolls).

    Returns
    -------
    (z0_proj_guided, stats).  z0_proj_guided is [B,N,2] zero-sum projected.
    """
    w_G = guidance_weight(t, T,
                          cfg.get("start_t_ratio", 0.40),
                          cfg.get("full_t_ratio", 0.10))
    if float(w_G.max().item()) <= 0.0:
        # no guidance from this timestep; keep the normal projection
        return zero_sum(z0_raw), {"w_G": 0.0, "shift_norm": 0.0}

    z = z0_raw.detach().requires_grad_(True)
    z0_proj = zero_sum(z)
    pos = integrate_positions(start, base + z0_proj)

    J, stats = consensus_guidance_cost(pos, center, radii, theta, cfg,
                                       detach_geometry=True)
    if not J.requires_grad:
        return zero_sum(z0_raw), {"w_G": float(w_G.mean().item()), "shift_norm": 0.0}

    grad = torch.autograd.grad(J, z, retain_graph=retain_graph,
                               create_graph=False, allow_unused=True)[0]
    if grad is None:
        return zero_sum(z0_raw), {"w_G": float(w_G.mean().item()), "shift_norm": 0.0}

    eta = float(cfg.get("eta_max", 0.05)) * w_G              # [B]
    z_guided = z - eta[:, None, None] * grad
    z0_proj_guided = zero_sum(z_guided)
    shift_norm = torch.norm(z0_proj_guided - z0_proj.detach(), dim=-1).mean().item()
    return z0_proj_guided, {"w_G": float(w_G.mean().item()), "shift_norm": shift_norm}


def apply_consensus_guidance_unrolled(z0_raw, base, start, center, radii, theta,
                                      cfg, t, T):
    """Differentiable guidance step that keeps the graph to z0_raw (training use).

    Unlike apply_consensus_guidance (inference), the source tensor is the model
    output z0_raw itself, so gradient flows back to the network through the
    guidance correction.  Consensus geometry is still detached (J_guide only
    corrects the trajectory).

    Returns (z0_proj_guided, pos_guided, stats).
    """
    w_G = guidance_weight(t, T,
                          cfg.get("start_t_ratio", 0.40),
                          cfg.get("full_t_ratio", 0.10))
    if float(w_G.max().item()) <= 0.0:
        z0_proj = zero_sum(z0_raw)
        pos = integrate_positions(start, base + z0_proj)
        return z0_proj, pos, {"w_G": 0.0, "shift_norm": 0.0}

    z0_proj = zero_sum(z0_raw)
    pos = integrate_positions(start, base + z0_proj)
    J, stats = consensus_guidance_cost(pos, center, radii, theta, cfg,
                                       detach_geometry=True)
    if not J.requires_grad:
        return z0_proj, pos, {"w_G": float(w_G.mean().item()), "shift_norm": 0.0}

    grad = torch.autograd.grad(J, z0_raw, retain_graph=True, create_graph=True,
                               allow_unused=True)[0]
    if grad is None:
        return z0_proj, pos, stats

    eta = float(cfg.get("eta_max", 0.05)) * w_G              # [B]
    z_guided = z0_raw - eta[:, None, None] * grad
    z0_proj_guided = zero_sum(z_guided)
    pos_guided = integrate_positions(start, base + z0_proj_guided)
    shift_norm = torch.norm(z0_proj_guided - z0_proj.detach(), dim=-1).mean().item()
    return z0_proj_guided, pos_guided, {
        "w_G": float(w_G.mean().item()), "shift_norm": shift_norm,
    }
