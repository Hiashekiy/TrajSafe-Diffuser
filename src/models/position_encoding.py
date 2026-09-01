"""Position / timestep / index embeddings."""

import math

import torch
import torch.nn as nn


class Sinusoidal2DPositionEmbedding(nn.Module):
    """Classic Transformer sinusoidal position embedding, applied per-axis to a 2D coord.

    For each axis a in {x, y}, and each frequency index i:
        angle = (a * scale) / base^(2i / half)
    and we emit sin(angle), cos(angle), interleaved per axis.

    ``scale`` (default 128) lets the continuous [-1,1] 256-cell grid be resolved;
    the raw 1/10000^(2i/d) has max frequency 1, too coarse for continuous coords.
    """

    def __init__(self, d_model, base=10000.0, scale=128.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        self.scale = scale
        half = d_model // 4              # per-axis frequency count (2 axes * sin/cos)
        self.half = half
        freqs = torch.exp(-math.log(base) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, xy):
        x = torch.as_tensor(xy, dtype=torch.float32).to(self.proj.weight.device)
        fx = (x[..., 0:1] * self.scale) * self.freqs
        fy = (x[..., 1:2] * self.scale) * self.freqs
        feats = torch.stack([torch.sin(fx), torch.cos(fx), torch.sin(fy), torch.cos(fy)], dim=-1)
        feats = feats.reshape(*feats.shape[:-2], -1)
        return self.proj(feats)


class Sinusoidal1DPositionEmbedding(nn.Module):
    """Classic Transformer 1D sinusoidal position embedding for integer token indices.

    PE(pos, 2i)   = sin( pos / base^(2i/(d/2)) )
    PE(pos, 2i+1) = cos( pos / base^(2i/(d/2)) )
    """

    def __init__(self, d_model, base=10000.0):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.exp(-math.log(base) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)   # [half]

    def forward(self, pos):
        """pos [..., L] integers -> [..., L, d_model]."""
        args = pos[..., :, None] * self.freqs                   # [..., L, half]
        emb = torch.stack([torch.sin(args), torch.cos(args)], dim=-1)  # [..., L, half, 2]
        return emb.reshape(*emb.shape[:-2], -1)                 # [..., L, d_model]


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


class Sinusoidal2DRelativePositionEmbedding(nn.Module):
    """Relative 2D sinusoidal position embedding for small scene-space offsets.

    The ellipse branch feeds Local_2 + PE_rel(delta_x, delta_y).  The offsets are
    very small (a 9x9 window at 64 res covers about +/-0.125 scene units), so the
    raw ``base=10000`` frequencies are too coarse to distinguish them.  We multiply
    each offset by ``scale`` (default 128) before the sin/cos encoding, matching
    the approach used by ``Sinusoidal2DPositionEmbedding`` for absolute coords.
    """

    def __init__(self, d_model, base=10000.0, scale=128.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        self.scale = scale
        half = d_model // 4              # per-axis frequency count (2 axes * sin/cos)
        self.half = half
        freqs = torch.exp(-math.log(base) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, xy):
        x = torch.as_tensor(xy, dtype=torch.float32).to(self.proj.weight.device)
        dx = x[..., 0:1] * self.scale
        dy = x[..., 1:2] * self.scale
        fx = dx * self.freqs
        fy = dy * self.freqs
        feats = torch.stack([torch.sin(fx), torch.cos(fx), torch.sin(fy), torch.cos(fy)], dim=-1)
        feats = feats.reshape(*feats.shape[:-2], -1)
        return self.proj(feats)
