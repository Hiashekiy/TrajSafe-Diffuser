"""Joint Diffusion Transformer blocks (docs/联合扩散.md #15-#21).

Per block (pre-norm + AdaLN, residual):

    Z'   = Z  + JointSelfAttention(  AdaLN(Z, h_t)  )      # relative bias inside
    Z''  = Z' + SceneCrossAttention( AdaLN(Z', h_t),
                                     Q=Z', K=V=C_scene )
    Z+   = Z''+ FFN(                 AdaLN(Z'', h_t) )

h_t is the diffusion-timestep embedding; AdaLN modulates each sublayer with
per-channel (gamma, beta) so t controls how the block treats noisy tokens.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """LayerNorm modulated by h_t: out = (1 + gamma) * LN(x) + beta."""

    def __init__(self, d_model, time_dim, zero_init=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod = nn.Linear(time_dim, 2 * d_model)
        if zero_init:
            nn.init.zeros_(self.mod.weight)
            nn.init.zeros_(self.mod.bias)

    def forward(self, x, h_t):
        gamma, beta = self.mod(h_t).chunk(2, dim=-1)   # each [B,d]
        return (1.0 + gamma[:, None, :]) * self.norm(x) + beta[:, None, :]


class _MHABase(nn.Module):
    """Shared manual multi-head attention (pre-projected q/k/v, bias injection)."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x, L):
        return x.view(x.shape[0], L, self.num_heads, self.head_dim).transpose(1, 2)

    def attend(self, q, k, v, bias=None):
        """q/k/v [B,L,D] (k/v same L or longer), bias [B,h,Li,Lj] or None."""
        Bi, Lq, _ = q.shape
        Lj = k.shape[1]
        q = self._split(q, Lq)
        k = self._split(k, Lj)
        v = self._split(v, Lj)
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale   # [B,h,Lq,Lj]
        if bias is not None:
            scores = scores + bias
        attn = self.drop(F.softmax(scores, dim=-1))
        o = torch.matmul(attn, v)                                     # [B,h,Lq,dh]
        o = o.transpose(1, 2).reshape(Bi, Lq, self.d_model)
        return self.out(o)


class JointSelfAttention(_MHABase):
    """Self-attention over the 2H joint tokens with structural relative bias.

    bias_ij = B_horizon(|k_i - k_j|) + B_type(r_i, r_j) [+ B_pair when the two
    tokens share the same planning index but different type (T_k <-> E_k)].
    """

    def __init__(self, d_model, num_heads, horizon, dropout=0.0):
        super().__init__(d_model, num_heads, dropout)
        self.H = int(horizon)
        L = 2 * self.H
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # learnable relative biases (per head)
        self.b_horizon = nn.Parameter(torch.zeros(self.H, num_heads))
        self.b_type = nn.Parameter(torch.zeros(2, 2, num_heads))
        self.b_pair = nn.Parameter(torch.zeros(self.H, num_heads))
        # token meta: k index and type per joint-sequence position
        ks = torch.arange(self.H).repeat_interleave(2)          # [L]
        rs = torch.arange(2).repeat(self.H)                     # [L]
        kd = (ks[:, None] - ks[None, :]).abs().clamp(0, self.H - 1)
        same_pair = ((ks[:, None] == ks[None, :]) &
                     (rs[:, None] != rs[None, :]))              # [L,L]
        self.register_buffer("_ks", ks)
        self.register_buffer("_rs", rs)
        self.register_buffer("_kd", kd)
        self.register_buffer("_same_pair", same_pair)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        Hd = self.num_heads
        bh = self.b_horizon[self._kd]                                  # [L,L,h]
        bt = self.b_type[self._rs[:, None], self._rs[None, :]]         # [L,L,h]
        pair = self.b_pair[self._ks[:, None]] * self._same_pair[..., None]
        bias = (bh + bt + pair).permute(2, 0, 1)[None].expand(B, Hd, L, L)
        return self.attend(q, k, v, bias)


class JointCrossAttention(_MHABase):
    """Q from joint tokens, K=V from C_scene (start+goal+256 map tokens)."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__(d_model, num_heads, dropout)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)

    def forward(self, x, mem):
        q = self.q(x)
        k, v = self.kv(mem).chunk(2, dim=-1)
        return self.attend(q, k, v)


class JointBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, horizon, dropout=0.0):
        super().__init__()
        self.n1 = AdaLN(d_model, d_model)
        self.sa = JointSelfAttention(d_model, num_heads, horizon, dropout)
        self.n2 = AdaLN(d_model, d_model)
        self.ca = JointCrossAttention(d_model, num_heads, dropout)
        self.n3 = AdaLN(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout))

    def forward(self, x, mem, h_t):
        x = x + self.sa(self.n1(x, h_t))
        x = x + self.ca(self.n2(x, h_t), mem)
        x = x + self.ffn(self.n3(x, h_t))
        return x
