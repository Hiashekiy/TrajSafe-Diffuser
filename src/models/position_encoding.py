"""Position / timestep / index embeddings."""

import math

import torch
import torch.nn as nn


class Sinusoidal2DPositionEmbedding(nn.Module):
    """2D world-coordinate positional embedding.

    Produces d_model features via a fixed set of sin/cos frequencies per axis;
    the concatenation is projected to d_model.
    """

    def __init__(self, d_model, num_freqs=None):
        super().__init__()
        if num_freqs is None:
            num_freqs = max(4, d_model // 4)
        self.num_freqs = num_freqs
        self.out_dim = num_freqs * 4
        self.proj = nn.Linear(self.out_dim, d_model)

    def forward(self, xy):
        # xy [...,2] -> [...,d_model]
        x = torch.as_tensor(xy, dtype=torch.float32).to(self.proj.weight.device)
        freqs = torch.linspace(0.0, 1.0, self.num_freqs, device=x.device) * math.pi
        fx = x[..., 0:1] * freqs
        fy = x[..., 1:2] * freqs
        feats = torch.cat([torch.sin(fx), torch.cos(fx), torch.sin(fy), torch.cos(fy)], dim=-1)
        return self.proj(feats)


class SinusoidalTimestepEmbedding(nn.Module):
    """Scalar diffusion timestep embedding -> d_model."""

    def __init__(self, d_model, max_period=10000.0):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(nn.Linear(half * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        args = t[:, None] * self.freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


