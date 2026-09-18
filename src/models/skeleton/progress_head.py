"""ProgressHead: where the K ellipses sit along the selected topology.

docs/V2.md sections 19/20/21.  The head predicts the K-1 GAPS between
consecutive ellipse centres and normalises them, so monotonicity and the
boundary conditions are structural rather than learned:

    w_i   = softplus(u_i) + eps
    d_i   = w_i / sum_j w_j
    s_0 = 0,  s_i = sum_{j<i} d_j,  s_{K-1} = 1

Together with c_i = gamma_m(s_i) this is the only definition of an ellipse
centre in V2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .traj_blocks import CrossAttention

__all__ = ["ProgressHead"]


class ProgressHead(nn.Module):
    def __init__(self, d_model, num_heads, hidden=256, dropout=0.0, eps=1e-4):
        super().__init__()
        self.eps = float(eps)
        self.cross = CrossAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, traj_feat, path_feat):
        """traj_feat [B,H,D] (query), path_feat [B,L,D] (selected topology).

        Returns (s [B,H] with s_0 = 0 and s_{H-1} = 1, fused [B,H,D]).
        """
        fused = self.norm(traj_feat + self.cross(traj_feat, path_feat))
        u = self.mlp(fused).squeeze(-1)                     # [B,H]
        w = F.softplus(u[:, :-1]) + self.eps                # K-1 positive gaps
        delta = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        zero = torch.zeros_like(delta[:, :1])
        s = torch.cat([zero, torch.cumsum(delta, dim=1)], dim=1)
        return s, fused
