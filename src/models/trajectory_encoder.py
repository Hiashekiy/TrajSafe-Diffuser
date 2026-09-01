"""Trajectory Token Encoder.

Each diffusion token is one zero-sum residual  z_k [B,N,2]  (N = H-1).
The token only carries:
    phi_z(z_k)  +  Sinusoidal1DPosition(k)  +  SinusoidalTimestep(t)
No velocity / acceleration / p_k position embedding.
"""

import torch
import torch.nn as nn

from .position_encoding import Sinusoidal1DPositionEmbedding, SinusoidalTimestepEmbedding


class PreNormBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        norm = self.ln(x)
        out, _ = self.attn(norm, norm, norm, need_weights=False)
        return x + out


class TrajectoryEncoder(nn.Module):
    def __init__(self, horizon, d_model=128, num_heads=8, num_layers=2,
                 ffn_dim=512, dropout=0.1):
        super().__init__()
        self.horizon = horizon          # H = number of positions (tokens = H-1)
        self.d_model = d_model
        self.z_linear = nn.Linear(2, d_model)
        self.pos_embed = Sinusoidal1DPositionEmbedding(d_model)
        self.time_embed = SinusoidalTimestepEmbedding(d_model)
        self.blocks = nn.ModuleList([
            PreNormBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, z_t, t, cond=None):
        """z_t [B,N,2], t [B] -> [B,N,C]."""
        B, N, _ = z_t.shape
        h = self.z_linear(z_t.float())
        idx = torch.arange(N, device=z_t.device)
        h = h + self.pos_embed(idx)[None]
        h = h + self.time_embed(t)[:, None]
        for blk in self.blocks:
            h = blk(h)
        return h
