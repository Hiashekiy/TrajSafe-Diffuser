"""Ellipse Head (docs sections 15/16): bounded physical axes at a fixed centre.

    q_i = F_i^prog + phi(c_i)
    attention into the V1 fine geometry memory C_E with the V1 spatial bias
        B_ij = -alpha_t ||c_i - q_j||^2 / (2 sigma^2)
    raw [r_a, r_b, u, v] ->
        b = b_min + (b_max - b_min) sigma(r_b)
        a = b + (a_max - b) sigma(r_a)          =>  b_min <= b <= a <= a_max
        theta = 0.5 atan2(v', u')

shape4 = [log a, log b, cos 2t, sin 2t] is derived for token encoding and
visualisation only - it has NO ground-truth supervision in V3.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..joint.joint_blocks import AdaLN
from .blocks import CrossAttention

__all__ = ["EllipseHead"]


class EllipseHead(nn.Module):
    def __init__(self, d_model, num_heads, spatial_pe, hidden=256, dropout=0.0,
                 b_min=0.02, b_max=0.30, a_max=0.80, geo_mem_res=32,
                 geo_sigma=0.25, geo_bias_clip=8.0):
        super().__init__()
        self.spatial_pe = spatial_pe
        self.b_min = float(b_min)
        self.b_max = float(b_max)
        self.a_max = float(a_max)
        self.geo_mem_res = int(geo_mem_res)
        self.geo_sigma = float(geo_sigma)
        self.geo_bias_clip = float(geo_bias_clip)
        self.n_q = AdaLN(d_model, d_model)
        self.geo_ca = CrossAttention(d_model, num_heads, dropout)
        self.n_out = AdaLN(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    def _geo_bias(self, center, ab):
        from ..joint.joint_planner import scene_grid_centres

        grid = scene_grid_centres(self.geo_mem_res, center.device).to(center.dtype)
        dist2 = ((center[:, :, None, :] - grid[None, None, :, :]) ** 2).sum(-1)
        strength = (ab ** 2).to(center.dtype)[:, None, None]
        bias = -strength * dist2 / (2.0 * self.geo_sigma ** 2)
        return bias.clamp(-self.geo_bias_clip, 0.0)[:, None]

    def forward(self, progress_feat, center, geo_mem, h_t, ab):
        """progress_feat [B,K,D]; center [B,K,2]; geo_mem [B,G,D] or None."""
        q = progress_feat + self.spatial_pe(center)
        att = None
        if geo_mem is not None:
            bias = self._geo_bias(center, ab) if ab is not None else None
            att = self.geo_ca(self.n_q(q, h_t), geo_mem, bias)
            q = q + att
        raw = self.mlp(self.n_out(q, h_t))
        r_a, r_b = raw[..., 0], raw[..., 1]
        b = self.b_min + (self.b_max - self.b_min) * torch.sigmoid(r_b)
        a = b + (self.a_max - b) * torch.sigmoid(r_a)
        uv = raw[..., 2:4]
        norm = torch.sqrt((uv * uv).sum(dim=-1, keepdim=True) + 1e-12)
        u_hat = uv / norm
        theta = 0.5 * torch.atan2(u_hat[..., 1], u_hat[..., 0])
        shape4 = torch.stack([torch.log(a.clamp_min(1e-6)),
                              torch.log(b.clamp_min(1e-6)),
                              u_hat[..., 0], u_hat[..., 1]], dim=-1)
        return {"a": a, "b": b, "theta": theta, "shape4": shape4,
                "geo_attn": att}
