"""Consensus guidance application for reverse diffusion / joint training.

J_guide is a soft "stay near the local consensus centerline" potential.  It is
defined on the *integrated path* pos = integrate(start, base + z0_proj), and the
gradient is taken w.r.t. the path positions, then converted back to a zero-sum
residual.  This avoids the ill-conditioning of stepping directly on the residual
z: because pos is a cumulative sum of z, a small change in z moves every
downstream position, so a z-space gradient step can explode the trajectory.
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


def _clamp_endpoints(pos, start, goal):
    """Pin the first/last waypoint because the zero-sum bridge is hard-constrained."""
    pos = pos.clone()
    pos[:, 0, :] = start
    pos[:, -1, :] = goal
    return pos


def _path_to_residual(pos, base):
    """pos [B,H,2] -> zero-sum residual [B,N,2] (delta = pos diff, z = delta-base)."""
    delta = pos[:, 1:] - pos[:, :-1]
    return zero_sum(delta - base)


def apply_consensus_guidance(z0_raw, base, start, center, radii, theta,
                             cfg, t, T, retain_graph=False):
    """Differentiable guidance step on the integrated path (used at sampling).

    The guidance objective J_guide lives on the trajectory positions, so the
    gradient is taken w.r.t. pos (not the residual z).  The corrected path is
    then converted back to a zero-sum residual.

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

    z0_proj = zero_sum(z0_raw.detach())
    pos = integrate_positions(start, base + z0_proj)
    goal = pos[:, -1, :]                                    # [B,2] actual end waypoint
    pos_r = pos.detach().requires_grad_(True)

    J, stats = consensus_guidance_cost(pos_r, center, radii, theta, cfg,
                                       detach_geometry=True)
    if not J.requires_grad:
        return zero_sum(z0_raw), {"w_G": float(w_G.mean().item()), "shift_norm": 0.0}

    grad_pos = torch.autograd.grad(J, pos_r, retain_graph=retain_graph,
                                   create_graph=False, allow_unused=True)[0]
    if grad_pos is None:
        return zero_sum(z0_raw), {"w_G": float(w_G.mean().item()), "shift_norm": 0.0}

    eta = float(cfg.get("eta_max", 0.05)) * w_G              # [B]
    pos_guided = _clamp_endpoints(pos_r - eta[:, None, None] * grad_pos,
                                  start, goal)
    z0_proj_guided = _path_to_residual(pos_guided, base)
    shift_norm = torch.norm(pos_guided - pos.detach(), dim=-1).mean().item()
    return z0_proj_guided, {"w_G": float(w_G.mean().item()), "shift_norm": shift_norm}


def apply_consensus_guidance_unrolled(z0_raw, base, start, center, radii, theta,
                                      cfg, t, T):
    """Differentiable guidance step that keeps the graph to z0_raw (training use).

    Same position-space guidance as apply_consensus_guidance, but the source
    tensor is the model output z0_raw itself, so gradient flows back to the
    network through the guidance correction.  Consensus geometry is detached.

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
    goal = pos[:, -1, :]                                    # [B,2] actual end waypoint
    J, stats = consensus_guidance_cost(pos, center, radii, theta, cfg,
                                       detach_geometry=True)
    if not J.requires_grad:
        return z0_proj, pos, {"w_G": float(w_G.mean().item()), "shift_norm": 0.0}

    grad_pos = torch.autograd.grad(J, pos, retain_graph=True, create_graph=True,
                                   allow_unused=True)[0]
    if grad_pos is None:
        return z0_proj, pos, stats

    eta = float(cfg.get("eta_max", 0.05)) * w_G              # [B]
    pos_guided = _clamp_endpoints(pos - eta[:, None, None] * grad_pos,
                                  start, goal)
    z0_proj_guided = _path_to_residual(pos_guided, base)
    pos_guided = integrate_positions(start, base + z0_proj_guided)
    shift_norm = torch.norm(pos_guided - pos.detach(), dim=-1).mean().item()
    return z0_proj_guided, pos_guided, {
        "w_G": float(w_G.mean().item()), "shift_norm": shift_norm,
    }
