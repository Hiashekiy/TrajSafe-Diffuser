"""Tensor helpers for the V2 ellipse-centre geometry.

The single definition of an ellipse centre in V2 is

    c_i = gamma_m(s_i)

i.e. a point on the SELECTED candidate polyline at normalized arc length s_i.
These helpers evaluate that map differentiably with respect to s (the progress
head output), so the shape head and the trajectory refinement receive gradients
through the centre, while the polyline itself is data.
"""

from __future__ import annotations

import torch

__all__ = ["path_arclength", "gather_path_points", "shape4_to_abtheta",
           "abtheta_to_shape4"]


def path_arclength(coords: torch.Tensor) -> torch.Tensor:
    """coords [B, L, 2] -> cumulative arc length [B, L] (starts at 0)."""
    if coords.shape[1] == 1:
        return torch.zeros(coords.shape[0], 1, device=coords.device,
                           dtype=coords.dtype)
    seg = torch.linalg.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
    zero = torch.zeros_like(seg[:, :1])
    return torch.cat([zero, torch.cumsum(seg, dim=1)], dim=1)


def gather_path_points(coords: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """gamma_m(s): coords [B, L, 2], s [B, K] in [0, 1] -> points [B, K, 2].

    Piecewise-linear interpolation at normalized arc length s, differentiable
    with respect to s.
    """
    B, L, _ = coords.shape
    cum = path_arclength(coords)
    total = cum[:, -1:].clamp_min(1e-9)
    target = s.clamp(0.0, 1.0) * total
    idx = torch.searchsorted(cum.contiguous(), target.contiguous(), right=True)
    idx = idx.clamp(1, max(L - 1, 1))
    c0 = cum.gather(1, idx - 1)
    c1 = cum.gather(1, idx)
    t = ((target - c0) / (c1 - c0).clamp_min(1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    ii = idx.unsqueeze(-1).expand(-1, -1, 2)
    p0 = coords.gather(1, ii - 1)
    p1 = coords.gather(1, ii)
    return p0 + t * (p1 - p0)


def shape4_to_abtheta(shape4: torch.Tensor):
    """[..., 4] = [log a, log b, cos 2t, sin 2t] -> (a, b, theta).

    a >= b is enforced structurally and (cos, sin) is renormalised, so the
    returned angles are always consistent with the returned axes.
    """
    log_a = shape4[..., 0]
    log_b = shape4[..., 1]
    big = torch.maximum(log_a, log_b)
    small = torch.minimum(log_a, log_b)
    a = torch.exp(big.clamp(-8.0, 8.0))
    b = torch.exp(small.clamp(-8.0, 8.0))
    u = shape4[..., 2]
    v = shape4[..., 3]
    # atan2 is scale invariant, so (cos 2t, sin 2t) is used directly.  The
    # degenerate direction (0, 0) - possible at initialisation, when the shape
    # head emits exactly zero - would make d atan2 NaN, so it is replaced by a
    # well defined constant direction.
    norm2 = u * u + v * v
    safe = norm2 > 1e-12
    uu = torch.where(safe, u, torch.ones_like(u))
    vv = torch.where(safe, v, torch.zeros_like(v))
    theta = 0.5 * torch.atan2(vv, uu)
    return a, b, theta


def abtheta_to_shape4(a: torch.Tensor, b: torch.Tensor,
                      theta: torch.Tensor) -> torch.Tensor:
    """(a, b, theta) -> [..., 4] with a >= b (major axis first)."""
    big = torch.maximum(a, b)
    small = torch.minimum(a, b)
    two_t = 2.0 * theta
    return torch.stack([torch.log(big.clamp_min(1e-8)),
                        torch.log(small.clamp_min(1e-8)),
                        torch.cos(two_t), torch.sin(two_t)], dim=-1)
