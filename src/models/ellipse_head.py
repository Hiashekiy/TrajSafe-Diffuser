"""Ellipse Head: predict center, radii, and double-angle direction."""

import torch
import torch.nn as nn


class EllipseHead(nn.Module):
    def __init__(self, d_model=128, ffn_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, 6),
        )

    def forward(self, S_t):
        raw = self.mlp(S_t)  # [B,H,6]
        cx = raw[..., 0]
        cy = raw[..., 1]
        rho1 = raw[..., 2]
        rho2 = raw[..., 3]
        dx = raw[..., 4]
        dy = raw[..., 5]
        r_raw1 = torch.nn.functional.softplus(rho1) + 1e-6
        r_raw2 = torch.nn.functional.softplus(rho2) + 1e-6
        r1 = torch.maximum(r_raw1, r_raw2)
        r2 = torch.minimum(r_raw1, r_raw2)
        norm = torch.sqrt(dx * dx + dy * dy) + 1e-8
        dir_x = dx / norm
        dir_y = dy / norm
        theta = 0.5 * torch.atan2(dir_y, dir_x)
        return {
            "center": torch.stack([cx, cy], dim=-1),
            "r1": r1,
            "r2": r2,
            "dir": torch.stack([dir_x, dir_y], dim=-1),
            "theta": theta,
            "raw": raw,
        }
