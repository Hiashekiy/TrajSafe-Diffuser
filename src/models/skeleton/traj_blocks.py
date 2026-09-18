"""Trajectory-only diffusion transformer blocks (V2).

V2 keeps exactly ONE diffusion state, the trajectory x_t.  These blocks are the
trajectory backbone and the ellipse-conditioned refinement head:

    TrajBlock        self-attention over H waypoints + scene cross-attention
    RefineBlock      + cross-attention from the trajectory to the ellipse tokens
                       (this is where the safety ellipses feed back into the
                        same trajectory diffusion, docs/V2.md section 27)

AdaLN and the manual multi-head attention core are reused from the frozen V1
package: the V1 files themselves are never modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..joint.joint_blocks import AdaLN, _MHABase

__all__ = ["TrajSelfAttention", "CrossAttention", "TrajBlock", "RefineBlock"]


class TrajSelfAttention(_MHABase):
    """Self-attention over H trajectory tokens with a learnable relative bias."""

    def __init__(self, d_model, num_heads, horizon, dropout=0.0):
        super().__init__(d_model, num_heads, dropout)
        self.horizon = int(horizon)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.b_horizon = nn.Parameter(torch.zeros(self.horizon, num_heads))
        kd = (torch.arange(self.horizon)[:, None]
              - torch.arange(self.horizon)[None, :]).abs()
        self.register_buffer("_kd", kd.clamp(0, self.horizon - 1))

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        bias = self.b_horizon[self._kd].permute(2, 0, 1)[None]
        bias = bias.expand(B, self.num_heads, L, L)
        return self.attend(q, k, v, bias)


class CrossAttention(_MHABase):
    """Standard cross-attention: Q from x, K/V from mem."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__(d_model, num_heads, dropout)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)

    def forward(self, x, mem, bias=None):
        k, v = self.kv(mem).chunk(2, dim=-1)
        return self.attend(self.q(x), k, v, bias)


class TrajBlock(nn.Module):
    """Pre-norm AdaLN block: self-attention + scene cross-attention + FFN."""

    def __init__(self, d_model, num_heads, ff_dim, horizon, dropout=0.0):
        super().__init__()
        self.n1 = AdaLN(d_model, d_model)
        self.sa = TrajSelfAttention(d_model, num_heads, horizon, dropout)
        self.n2 = AdaLN(d_model, d_model)
        self.ca = CrossAttention(d_model, num_heads, dropout)
        self.n3 = AdaLN(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout))

    def forward(self, x, global_mem, h_t):
        x = x + self.sa(self.n1(x, h_t))
        x = x + self.ca(self.n2(x, h_t), global_mem)
        x = x + self.ffn(self.n3(x, h_t))
        return x


class RefineBlock(nn.Module):
    """Second-half block: the trajectory also attends to the ellipse tokens.

    Q = trajectory tokens, K = V = ellipse tokens H_E.  This is the only place
    where the ellipse sequence influences the trajectory diffusion, and it
    happens inside the SAME diffusion process (no second diffusion state).
    """

    def __init__(self, d_model, num_heads, ff_dim, horizon, dropout=0.0):
        super().__init__()
        self.n1 = AdaLN(d_model, d_model)
        self.sa = TrajSelfAttention(d_model, num_heads, horizon, dropout)
        self.n2 = AdaLN(d_model, d_model)
        self.ca = CrossAttention(d_model, num_heads, dropout)
        self.ne = AdaLN(d_model, d_model)
        self.ea = CrossAttention(d_model, num_heads, dropout)
        self.n3 = AdaLN(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout))

    def forward(self, x, global_mem, h_t, ellipse_tokens):
        x = x + self.sa(self.n1(x, h_t))
        x = x + self.ca(self.n2(x, h_t), global_mem)
        x = x + self.ea(self.ne(x, h_t), ellipse_tokens)
        x = x + self.ffn(self.n3(x, h_t))
        return x
