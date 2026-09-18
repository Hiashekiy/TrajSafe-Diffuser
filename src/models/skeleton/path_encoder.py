"""PathEncoder: candidate polylines -> tokens pooled against the trajectory.

docs/V2.md sections 13/1-3.  A candidate is already a complete ordered polyline
[L, 5] = [x, y, tx, ty, u], so no graph network is needed:

    h_mj = MLP_path(f_j^m) + PE(j)
    A_m  = MHA(Q = h_m, K = V = H_tau)
    g_m  = MeanPool(LN(h_m + A_m))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..position_encoding import Sinusoidal1DPositionEmbedding
from .traj_blocks import CrossAttention

__all__ = ["PathEncoder"]


class PathEncoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden=128, dropout=0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.index_pe = Sinusoidal1DPositionEmbedding(d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.attn = CrossAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, candidate_paths, traj_feat):
        """candidate_paths [B,M,L,5]; traj_feat [B,H,D].

        Returns (tokens [B,M,L,D], pooled [B,M,D]).
        """
        B, M, L, _ = candidate_paths.shape
        H = traj_feat.shape[1]
        dev = candidate_paths.device

        f = candidate_paths.reshape(B * M, L, 5)
        idx = torch.arange(L, device=dev, dtype=torch.long)
        h = self.norm_in(self.mlp(f) + self.index_pe(idx)[None])

        mem = traj_feat[:, None].expand(B, M, H, self.d_model).reshape(B * M, H, self.d_model)
        h = self.norm(h + self.attn(h, mem))
        h = self.norm_out(h + self.ffn(h))
        h = h.reshape(B, M, L, self.d_model)
        return h, h.mean(dim=2)
