"""EllipseShapeHead: the 2D safety shape at a FIXED skeleton centre.

docs/V2.md sections 24/26.  V2 has no centre head at all: the centre is
c_i = gamma_m(s_i), a geometric consequence of the selected topology.  The head
only predicts the four shape numbers, using the centre as a geometry query into
the fine scene memory:

    q_i = phi(c_i) + h_i^prog
    r_i = MLP(geo_cross_attn(q_i, geo_mem))
    shape4 = [log a, log b, cos 2t, sin 2t]   with a >= b and ||(cos, sin)|| = 1
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .traj_blocks import CrossAttention

__all__ = ["EllipseShapeHead", "raw_to_shape4"]


def raw_to_shape4(raw: torch.Tensor) -> torch.Tensor:
    """raw [...,4] = [l_a, l_b, u, v] -> [log a, log b, cos 2t, sin 2t], a >= b.

    The direction is always renormalised to a unit vector.  The degenerate
    (u, v) = (0, 0) - reachable at initialisation - falls back to (1, 0)
    instead of producing a zero vector with an undefined angle.
    """
    big = torch.maximum(raw[..., 0], raw[..., 1])
    small = torch.minimum(raw[..., 0], raw[..., 1])
    u = raw[..., 2:3]
    v = raw[..., 3:4]
    norm2 = u * u + v * v
    safe = norm2 > 1e-12
    inv = 1.0 / torch.sqrt(norm2 + 1e-12)
    u = torch.where(safe, u * inv, torch.ones_like(u))
    v = torch.where(safe, v * inv, torch.zeros_like(v))
    return torch.cat([big[..., None], small[..., None], u, v], dim=-1)


class EllipseShapeHead(nn.Module):
    def __init__(self, d_model, num_heads, spatial_pe, hidden=256, dropout=0.0,
                 geo_mem_res=32, geo_sigma=0.25, geo_bias_clip=8.0):
        super().__init__()
        self.spatial_pe = spatial_pe
        self.geo_mem_res = int(geo_mem_res)
        self.geo_sigma = float(geo_sigma)
        self.geo_bias_clip = float(geo_bias_clip)
        self.geo_ca = CrossAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    def _geo_bias(self, center, ab):
        """Noise-aware spatial bias [B,1,K,G] (same form as V1's geometry bias)."""
        from ..joint.joint_planner import scene_grid_centres

        grid = scene_grid_centres(self.geo_mem_res, center.device)
        grid = grid.to(center.dtype)
        dist2 = ((center[:, :, None, :] - grid[None, None, :, :]) ** 2).sum(dim=-1)
        strength = (ab ** 2).to(center.dtype)[:, None, None]
        bias = -strength * dist2 / (2.0 * self.geo_sigma ** 2)
        return bias.clamp(-self.geo_bias_clip, 0.0)[:, None]

    def forward(self, center, progress_feat, geo_mem, ab=None):
        """center [B,K,2]; progress_feat [B,K,D]; geo_mem [B,G,D] or None."""
        q = self.spatial_pe(center) + progress_feat
        if geo_mem is not None:
            bias = self._geo_bias(center, ab) if ab is not None else None
            q = self.norm(q + self.geo_ca(q, geo_mem, bias))
        return raw_to_shape4(self.mlp(q))
