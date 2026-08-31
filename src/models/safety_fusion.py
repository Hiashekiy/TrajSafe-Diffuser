"""Safety Feature Fusion: combine trajectory context with local scene feature."""

import torch.nn as nn


class SafetyFusion(nn.Module):
    def __init__(self, d_model=128, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model * 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, F_traj, A_t):
        import torch
        h = self.ln(torch.cat([F_traj, A_t], dim=-1))
        return self.mlp(h)
