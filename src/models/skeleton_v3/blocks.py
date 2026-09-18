"""V3 network blocks.

Reuses the frozen V1 conditioning design (AdaLN + the manual multi-head
attention core) and the V1 Joint Transformer idea, but the timestep enters as a
GLOBAL conditioning signal in every sublayer - never as a token that is simply
added:

    AdaLN(x, h_t) = (1 + gamma(h_t)) LN(x) + beta(h_t)

Modules here:
    TrajSelfAttention   relative horizon bias over H trajectory tokens
    CrossAttention      plain Q/KV attention
    TrajBlock           self-attn -> scene cross-attn -> FFN   (backbone)
    JointFusionBlock    interleaved [T1,E1,...,TH,EH] self-attn with horizon /
                        type / same-pair biases -> scene cross-attn -> FFN
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..joint.joint_blocks import AdaLN, _MHABase

__all__ = ["TrajSelfAttention", "CrossAttention", "TrajBlock", "JointFusionBlock"]


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
        return self.attend(q, k, v, bias.expand(B, self.num_heads, L, L))


class CrossAttention(_MHABase):
    """Q from x, K/V from mem (optional additive score bias)."""

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


class JointFusionBlock(nn.Module):
    """V1-style interleaved trajectory/ellipse fusion block.

    The 2H joint sequence keeps the structural biases of V1:
        B_horizon(|k_i - k_j|) + B_type(r_i, r_j) + B_pair(same index, T <-> E)
    """

    def __init__(self, d_model, num_heads, ff_dim, horizon, dropout=0.0):
        super().__init__()
        self.horizon = int(horizon)
        L = 2 * self.horizon
        self.n1 = AdaLN(d_model, d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.num_heads = int(num_heads)
        self.head_dim = d_model // self.num_heads
        import math
        self.scale = math.sqrt(self.head_dim)
        self.b_horizon = nn.Parameter(torch.zeros(self.horizon, num_heads))
        self.b_type = nn.Parameter(torch.zeros(2, 2, num_heads))
        self.b_pair = nn.Parameter(torch.zeros(self.horizon, num_heads))
        ks = torch.arange(self.horizon).repeat_interleave(2)
        rs = torch.arange(2).repeat(self.horizon)
        kd = (ks[:, None] - ks[None, :]).abs().clamp(0, self.horizon - 1)
        same_pair = ((ks[:, None] == ks[None, :]) & (rs[:, None] != rs[None, :]))
        self.register_buffer("_ks", ks)
        self.register_buffer("_rs", rs)
        self.register_buffer("_kd", kd)
        self.register_buffer("_same_pair", same_pair)
        assert L == len(ks)

        self.n2 = AdaLN(d_model, d_model)
        self.ca = CrossAttention(d_model, num_heads, dropout)
        self.n3 = AdaLN(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout))

    def forward(self, z, global_mem, h_t):
        B, L, D = z.shape
        h = self.n1(z, h_t)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        bh = self.b_horizon[self._kd]
        bt = self.b_type[self._rs[:, None], self._rs[None, :]]
        pair = self.b_pair[self._ks[:, None]] * self._same_pair[..., None]
        bias = (bh + bt + pair).permute(2, 0, 1)[None]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        scores = scores + bias.expand(B, self.num_heads, L, L)
        attn = self.drop(torch.softmax(scores, dim=-1))
        o = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        z = z + self.out(o)
        z = z + self.ca(self.n2(z, h_t), global_mem)
        z = z + self.ffn(self.n3(z, h_t))
        return z
