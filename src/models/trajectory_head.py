"""Trajectory Head: map decoder features to 6D clean trajectory state."""

import torch.nn as nn


class TrajectoryHead(nn.Module):
    def __init__(self, d_model=128, ffn_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, 6),
        )

    def forward(self, x):
        return self.mlp(x)
