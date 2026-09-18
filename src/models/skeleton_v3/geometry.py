"""Geometry helpers for V3: gamma_m(s) on the DENSE safe polyline.

The dense geometry of a candidate is the raw cell chain (padded to a fixed
length for batching).  Interpolating on it - never on the 128-point network
feature path - is what makes the ellipse centre structurally safe, because every
segment of the chain stays inside free space.
"""

from __future__ import annotations

import torch

__all__ = ["dense_arclength", "gather_dense_path_points"]


def dense_arclength(coords: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """coords [B,G,2] padded, lengths [B] valid point count -> cum [B,G].

    The padding tail contributes zero length, so the cumulative arc length is
    flat after the last valid point.
    """
    B, G, _ = coords.shape
    if G < 2:
        return torch.zeros(B, 1, device=coords.device, dtype=coords.dtype)
    seg = torch.linalg.norm(coords[:, 1:] - coords[:, :-1], dim=-1)     # [B,G-1]
    idx = torch.arange(G - 1, device=coords.device)[None]
    seg = seg * (idx < (lengths[:, None] - 1)).to(seg.dtype)
    zero = torch.zeros_like(seg[:, :1])
    return torch.cat([zero, torch.cumsum(seg, dim=1)], dim=1)


def gather_dense_path_points(coords: torch.Tensor, lengths: torch.Tensor,
                             s: torch.Tensor) -> torch.Tensor:
    """gamma_m(s) on the dense chain: coords [B,G,2], lengths [B], s [B,K].

    Differentiable with respect to s.
    """
    B, G, _ = coords.shape
    lengths = lengths.clamp(min=1, max=G)
    cum = dense_arclength(coords, lengths)
    last = (lengths - 1).clamp(min=0).long()
    total = cum.gather(1, last[:, None]).clamp_min(1e-9)               # [B,1]
    target = s.clamp(0.0, 1.0) * total
    idx = torch.searchsorted(cum.contiguous(), target.contiguous(), right=True)
    hi = last[:, None].expand_as(idx)
    idx = torch.minimum(torch.clamp(idx, min=1), hi.clamp_min(1))
    c0 = cum.gather(1, idx - 1)
    c1 = cum.gather(1, idx)
    t = ((target - c0) / (c1 - c0).clamp_min(1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    ii = idx.unsqueeze(-1).expand(-1, -1, 2)
    p0 = coords.gather(1, ii - 1)
    p1 = coords.gather(1, ii)
    return p0 + t * (p1 - p0)
