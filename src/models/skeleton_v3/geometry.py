"""Skeleton-curve geometry for the report-faithful TrajSafe-Diffuser.

``Gamma_m`` is the complete ordered dense geometry of a candidate.  It is pure
geometry data: it never enters the network.  Its ONLY use is the centre decode

    s_i -> c_i = Gamma_m(s_i)

implemented as arc-length interpolation on the dense safe polyline (report
section 10.2 / 16 / 27.1).  Interpolating on the 128-point network feature path
instead would allow a chord to cut an obstacle corner.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["dense_arclength", "gather_dense_path_points", "CurveDecoder"]


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
    """Gamma_m(s) on the dense chain: coords [B,G,2], lengths [B], s [B,K].

    Differentiable with respect to ``s``.  Degenerate candidates (length < 2)
    return the first padded point, which keeps the forward finite; the planner
    masks these rows and falls back to the coarse trajectory.
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


class CurveDecoder(nn.Module):
    """Parameter-free module wrapper around :func:`gather_dense_path_points`."""

    def forward(self, geometry: torch.Tensor, lengths: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        return gather_dense_path_points(geometry, lengths, s)
