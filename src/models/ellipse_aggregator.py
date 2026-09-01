"""Ellipse Aggregator: set-attention over the local geometric tokens.

For each trajectory point the ellipse branch sees 81 local tokens
(Local_2 + PE_rel).  A shared learnable query attends to that set, producing
a per-point summarised geometry vector e_k [B,N,C].
"""

import torch
import torch.nn as nn


class _SetAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN block (residual)."""

    def __init__(self, d_model, num_heads, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, q, kv):
        nq = self.ln_q(q)
        nkv = self.ln_kv(kv)
        out, _ = self.attn(nq, nkv, nkv, need_weights=False)
        h = q + out
        h = h + self.ffn(self.ln_ffn(h))
        return h


class EllipseAggregator(nn.Module):
    """Set-attention over the 81 local tokens around each predicted waypoint."""

    def __init__(self, d_model=128, num_heads=8, num_layers=2, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        # one shared learnable query used for every trajectory point / batch
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            _SetAttnBlock(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, local_tokens):
        """local_tokens [B,N,nl,C] (Local_2 + PE_rel) -> [B,N,C]."""
        B, N, nl, C = local_tokens.shape
        q = self.query.unsqueeze(0).expand(B * N, 1, C)
        kv = local_tokens.reshape(B * N, nl, C)
        h = q
        for blk in self.blocks:
            h = blk(h, kv)
        return h.reshape(B, N, C)
