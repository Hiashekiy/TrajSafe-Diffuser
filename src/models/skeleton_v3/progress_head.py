"""Progress Head (docs section 13): monotone arc length on the chosen path.

    F_prog = H_t + CrossAttention(Q = H_t, K = V = H_m^S)      (with AdaLN(h_t))
    u_i    -> w_i = softplus(u_i) + eps -> normalised gaps -> cumulative s

so 0 = s_1 < s_2 < ... < s_H = 1 holds STRUCTURALLY for every input, and there
is no progress ground truth to supervise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..joint.joint_blocks import AdaLN
from .blocks import CrossAttention

__all__ = ["ProgressHead"]


class ProgressHead(nn.Module):
    def __init__(self, d_model, num_heads, hidden=256, dropout=0.0, eps=1e-4):
        super().__init__()
        self.eps = float(eps)
        self.n_q = AdaLN(d_model, d_model)
        self.cross = CrossAttention(d_model, num_heads, dropout)
        self.n_out = AdaLN(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, traj_feat, path_feat, h_t):
        """traj_feat [B,H,D]; path_feat [B,L,D] of the selected candidate.

        Returns (s [B,H] with s_0 = 0 and s_{H-1} = 1, fused [B,H,D]).
        """
        att = self.cross(self.n_q(traj_feat, h_t), path_feat)
        fused = traj_feat + att
        u = self.mlp(self.n_out(fused, h_t)).squeeze(-1)
        w = F.softplus(u[:, :-1]) + self.eps
        delta = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        zero = torch.zeros_like(delta[:, :1])
        s = torch.cat([zero, torch.cumsum(delta, dim=1)], dim=1)
        return s, fused
