"""Trajectory Token Encoder."""

import torch
import torch.nn as nn

from .position_encoding import Sinusoidal2DPositionEmbedding, SinusoidalTimestepEmbedding


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
    def __init__(self, horizon, state_dim=6, d_model=128, num_heads=8,
                 num_layers=2, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.motion_dim = 4  # [ax,ay,vx,vy]
        self.pos_dim = 2     # [x,y]
        self.motion_linear = nn.Linear(self.motion_dim, d_model)
        self.pos_embed = Sinusoidal2DPositionEmbedding(d_model)
        self.index_embed = nn.Embedding(horizon, d_model)
        self.time_embed = SinusoidalTimestepEmbedding(d_model)
        self.blocks = nn.ModuleList([
            PreNormBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x_t, t, cond=None):
        # x_t [B,H,6], t [B].
        # NOTE: this matches the archived Janner Diffuser / RGG reference: the
        # denoiser is UNCONDITIONAL; start/goal conditioning is applied via
        # endpoint inpainting (apply_endpoint_condition) during training/sampling.
        B, H, _ = x_t.shape
        motion = x_t[..., [0, 1, 4, 5]]  # [B,H,4]
        pos = x_t[..., [2, 3]]           # [B,H,2]
        h = self.motion_linear(motion) + self.pos_embed(pos)
        idx_emb = self.index_embed(torch.arange(H, device=x_t.device))[None]  # [1,H,C]
        h = h + idx_emb
        t_emb = self.time_embed(t)[:, None]  # [B,1,C]
        h = h + t_emb
        for blk in self.blocks:
            h = blk(h)
        return h
