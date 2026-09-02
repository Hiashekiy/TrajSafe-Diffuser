"""Local ellipse consensus geometry for consensus guidance (J_guide).

Implements the V5 design (sections 22-26):
  - positional weights [1, 2, 4, 2, 1]
  - eccentricity weighting e_j = sqrt(1 - r2_j^2 / r1_j^2)
  - consensus center, double-angle direction, minor radius, confidence gamma.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-8


def eccentricity(r1, r2, eps=EPS):
    """e = sqrt(1 - r2^2 / r1^2), clamped to [0,1]."""
    r1 = torch.clamp(r1, min=eps)
    r2 = torch.clamp(r2, min=0.0)
    e = torch.sqrt(torch.clamp(1.0 - (r2 * r2) / (r1 * r1 + eps), min=0.0, max=1.0))
    return e


def compute_consensus_geometry(center, radii, theta, window=2,
                               pos_weights=(1.0, 2.0, 4.0, 2.0, 1.0),
                               eps=EPS):
    """Build per-ellipse local consensus geometry.

    Parameters are in the scene frame:
      center : [B,N,2] ellipse centres
      radii  : [B,N,2] (r1 >= r2)
      theta  : [B,N] long-axis angle (radians)

    Returns a dict with [B,N] / [B,N,2] arrays:
      cbar, ubar, r2bar, gamma, q, ebar, theta_bar.
    """
    center = center.float()
    radii = radii.float()
    theta = theta.float()
    B, N, _ = center.shape
    device = center.device
    if N == 0:
        return {}

    offsets = torch.arange(-window, window + 1, device=device, dtype=torch.long)
    pw = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    e = eccentricity(radii[..., 0], radii[..., 1])          # [B,N]

    # Vectorized local window gather over all k at once.
    W = offsets.numel()
    base = torch.arange(N, device=device, dtype=torch.long)[:, None]   # [N,1]
    neigh = base + offsets[None, :]                                     # [N,W]
    valid = (neigh >= 0) & (neigh < N)                                  # [N,W]
    idx = neigh.clamp(0, N - 1)                                         # [N,W]

    center_g = center[:, idx]                                         # [B,N,W,2]
    e_g = e[:, idx]                                                   # [B,N,W]
    theta_g = theta[:, idx]                                           # [B,N,W]
    r2_g = radii[:, idx, 1]                                           # [B,N,W]

    weight = pw[None, None, :] * valid.float()[None, :, :] * e_g      # [B,N,W]
    wsum = weight.sum(-1, keepdim=True) + eps                         # [B,N,1]

    cbar = (weight[..., None] * center_g).sum(-2) / wsum              # [B,N,2]

    cos2 = torch.cos(2.0 * theta_g)
    sin2 = torch.sin(2.0 * theta_g)
    C = (weight * cos2).sum(-1) / wsum.squeeze(-1)                    # [B,N]
    S = (weight * sin2).sum(-1) / wsum.squeeze(-1)                    # [B,N]
    r2bar = (weight * r2_g).sum(-1) / wsum.squeeze(-1)                # [B,N]

    wpos = pw[None, None, :] * valid.float()[None, :, :]              # [1,N,W]
    ebar = (wpos * e_g).sum(-1) / wpos.sum(-1).clamp(min=eps)         # [B,N]

    theta_bar = 0.5 * torch.atan2(S, C)
    ubar_x = torch.cos(theta_bar)
    ubar_y = torch.sin(theta_bar)
    ubar = torch.stack([ubar_x, ubar_y], dim=-1)             # [B,N,2]
    q = torch.sqrt(C * C + S * S)                            # [B,N]
    gamma = ebar * q
    # normal to the consensus direction
    nbar = torch.stack([-ubar_y, ubar_x], dim=-1)            # [B,N,2]
    return {
        "cbar": cbar, "ubar": ubar, "nbar": nbar,
        "r2bar": r2bar, "theta_bar": theta_bar,
        "eccentricity": e, "ebar": ebar, "q": q, "gamma": gamma,
    }


def huber(x, delta=1.0):
    """Huber: 0.5*x^2 if |x|<=delta else delta*(|x|-0.5*delta)."""
    a = 0.5 * x * x
    b = delta * (x.abs() - 0.5 * delta)
    return torch.where(x.abs() <= delta, a, b)


def consensus_guidance_cost(traj, center, radii, theta, cfg,
                            detach_geometry=True, eps=EPS):
    """J_guide = mean_k gamma_k * huber(delta_k), geometry detached.

    traj : [B,H,2] predicted positions (scene).
    center/radii/theta : predicted ellipses [B,N,2]/[B,N,2]/[B,N].
    cfg  : dict with huber_delta, normalize_by_minor_axis, window_radius,
           positional_weights.
    """
    traj = traj.float()
    center = center.float()
    radii = radii.float()
    theta = theta.float()

    window = int(cfg.get("window_radius", 2))
    pw = cfg.get("positional_weights", [1.0, 2.0, 4.0, 2.0, 1.0])
    geoms = center if not detach_geometry else center.detach()
    radii_g = radii if not detach_geometry else radii.detach()
    theta_g = theta if not detach_geometry else theta.detach()
    geom = compute_consensus_geometry(geoms, radii_g, theta_g,
                                      window=window, pos_weights=pw, eps=eps)

    H = traj.shape[1]
    # interior trajectory points 1..H-2; ellipse index k=i-1
    n_pts = max(0, H - 2)
    if n_pts == 0 or geom.get("cbar") is None:
        return torch.zeros((), device=traj.device, requires_grad=True), {}

    p = traj[:, 1:H - 1]                                     # [B,H-2,2]
    k = slice(0, n_pts)
    cbar = geom["cbar"][:, k]                                # [B,H-2,2]
    nbar = geom["nbar"][:, k]
    r2bar = geom["r2bar"][:, k]
    gamma = geom["gamma"][:, k]

    d_perp = (nbar[..., 0] * (p[..., 0] - cbar[..., 0])
              + nbar[..., 1] * (p[..., 1] - cbar[..., 1]))
    if bool(cfg.get("normalize_by_minor_axis", True)):
        delta = d_perp / (r2bar + eps)
    else:
        delta = d_perp

    rho = huber(delta, float(cfg.get("huber_delta", 1.0)))
    J = (gamma * rho).mean()

    stats = {
        "gamma_mean": gamma.mean().detach(),
        "q_mean": geom["q"][:, k].mean().detach(),
        "delta_abs_mean": delta.abs().mean().detach(),
        "d_perp_mean": d_perp.abs().mean().detach(),
    }
    return J, stats
