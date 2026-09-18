"""Static path encoder for V3 (docs section 9).

The encoder is STATIC: it never binds to the trajectory, so the same tokens are
reused by the Skeleton Selector and by the Progress Head.  Trajectory-path
matching is the selector's job, not the encoder's.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..position_encoding import Sinusoidal1DPositionEmbedding

__all__ = ["StaticPathEncoder"]


class StaticPathEncoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden=128, dropout=0.0, layers=2):
        super().__init__()
        self.d_model = int(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.index_pe = Sinusoidal1DPositionEmbedding(d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList()
        for _ in range(int(layers)):
            self.blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, num_heads,
                                              dropout=dropout, batch_first=True),
                "n2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(),
                    nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)),
            }))
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, candidate_features):
        """candidate_features [B,M,L,5] -> (tokens [B,M,L,D], pooled [B,M,D])."""
        B, M, L, _ = candidate_features.shape
        f = candidate_features.reshape(B * M, L, 5)
        idx = torch.arange(L, device=f.device, dtype=torch.long)
        h = self.norm_in(self.mlp(f) + self.index_pe(idx)[None])
        for blk in self.blocks:
            y = blk["n1"](h)
            y, _ = blk["attn"](y, y, y, need_weights=False)
            h = h + y
            h = h + blk["ffn"](blk["n2"](h))
        h = self.norm_out(h)
        h = h.reshape(B, M, L, self.d_model)
        return h, h.mean(dim=2)
