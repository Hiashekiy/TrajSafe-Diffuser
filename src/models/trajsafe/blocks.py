"""Transformer blocks for the report-faithful TrajSafe-Diffuser.

Only the blocks that the implementation report actually defines live here:

    TrajSelfAttention   self-attention over H trajectory tokens with a
                        learnable per-head relative waypoint-index bias
                        B_ij^h = b_h(|i - j|)  (report section 8.1)
    CrossAttention      plain Q/KV attention (no spatial bias)
    TrajBlock           AdaLN(self) -> AdaLN(global-map cross) -> AdaLN(FFN),
                        each sublayer residual (report section 8.1 / 22.1)
    MatchBlock          the ONE shared trajectory-skeleton cross-attention plus
                        the LN -> FFN_match residual (report section 12.1)

Explicitly NOT here any more: the interleaved trajectory/ellipse
``JointFusionBlock``, token type embeddings, role embeddings, or a second
trajectory-skeleton cross-attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import AdaLN, _MHABase

__all__ = ["TrajSelfAttention", "CrossAttention", "TrajBlock", "MatchBlock"]


class TrajSelfAttention(_MHABase):
    """Self-attention over H trajectory tokens with a learnable relative bias."""

    def __init__(self, d_model: int, num_heads: int, horizon: int,
                 dropout: float = 0.0):
        super().__init__(d_model, num_heads, dropout)
        self.horizon = int(horizon)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.b_horizon = nn.Parameter(torch.zeros(self.horizon, num_heads))
        kd = (torch.arange(self.horizon)[:, None]
              - torch.arange(self.horizon)[None, :]).abs()
        self.register_buffer("_kd", kd.clamp(0, self.horizon - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        bias = self.b_horizon[self._kd].permute(2, 0, 1)[None]
        return self.attend(q, k, v, bias.expand(B, self.num_heads, L, L))


class CrossAttention(_MHABase):
    """Q from ``x``, K/V from ``mem``; optional additive score bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__(d_model, num_heads, dropout)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor, mem: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        k, v = self.kv(mem).chunk(2, dim=-1)
        return self.attend(self.q(x), k, v, bias)


class TrajBlock(nn.Module):
    """Pre-norm AdaLN block: self-attention + global-map cross + FFN.

    The same module class is used by the Trajectory Backbone (N_T = 8) and by
    the Final Denoiser (N_F = 3); the two stacks own independent parameters.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, horizon: int,
                 dropout: float = 0.0):
        super().__init__()
        self.n1 = AdaLN(d_model, d_model)
        self.sa = TrajSelfAttention(d_model, num_heads, horizon, dropout)
        self.n2 = AdaLN(d_model, d_model)
        self.ca = CrossAttention(d_model, num_heads, dropout)
        self.n3 = AdaLN(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, global_mem: torch.Tensor,
                h_t: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.n1(x, h_t))
        x = x + self.ca(self.n2(x, h_t), global_mem)
        x = x + self.ffn(self.n3(x, h_t))
        return x


class MatchBlock(nn.Module):
    """The single shared trajectory-skeleton matching block.

    For every candidate m (batched on the M dimension, parameters shared):

        A_m = CrossAttention(Q=AdaLN(H_traj, h_t), K=H_m^S, V=H_m^S)
        U_m = H_traj + A_m
        R_m = U_m + FFN_match(LN_match(U_m))

    No Chamfer feature, no candidate length, no coarse-trajectory token, and no
    second cross-attention: R is the shared feature used by both the topology
    head and the progress head.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.scale = math.sqrt(self.head_dim)
        self.adaln = AdaLN(d_model, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout))

    def forward(self, h_traj: torch.Tensor, h_skel: torch.Tensor,
                h_t: torch.Tensor) -> torch.Tensor:
        """h_traj [B,H,D], h_skel [B,M,L,D] -> R [B,M,H,D]."""
        B, H, D = h_traj.shape
        M = h_skel.shape[1]
        L = h_skel.shape[2]

        q = self.q(self.adaln(h_traj, h_t))                  # [B,H,D]
        kv = self.kv(h_skel)                                 # [B,M,L,2D]
        k, v = kv.chunk(2, dim=-1)

        q = q[:, None].expand(B, M, H, D).reshape(B * M, H, D)
        k = k.reshape(B * M, L, D)
        v = v.reshape(B * M, L, D)

        q = q.view(B * M, H, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * M, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * M, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attn = self.drop(F.softmax(scores, dim=-1))
        att = torch.matmul(attn, v).transpose(1, 2).reshape(B * M, H, D)
        att = self.out(att).view(B, M, H, D)

        u = h_traj[:, None, :, :] + att                       # [B,M,H,D]
        r = u + self.ffn(self.ln(u))
        return r
