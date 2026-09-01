"""Residual Head: map decoder features to the per-token zero-sum residual Z.

Since the decoder produces one token per increment (N = H-1), this head outputs
[B, N, 2] directly (no slicing / aggregation needed).
"""

import torch.nn as nn


class TrajectoryHead(nn.Module):
    def __init__(self, d_model=128, ffn_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, 2),
        )

    def forward(self, x):
        """x [B,N,C] -> [B,N,2]."""
        return self.mlp(x)
