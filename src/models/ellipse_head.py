"""Ellipse Head: predict center offset, radii, and double-angle direction.

The head consumes the aggregated local geometry feature e_k together with the
predicted waypoint (anchor) p_hat.  It regresses a 6-dim raw vector
    [dx, dy, rho1, rho2, u, v]
and restores:
    center  = p_hat + [dx, dy]
    r1, r2  = softplus(rho) sorted so r1 >= r2 > 0
    dir     = normalize([u, v])            (double-angle vector)
    theta   = 0.5 * atan2(dir_y, dir_x)
"""

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

    def forward(self, feat, p_hat):
        """feat [B,N,C], p_hat [B,N,2] -> ellipse dict.

        p_hat is the predicted clean trajectory point (anchor) -- NOT detached.
        """
        raw = self.mlp(feat)          # [B,N,6]
        delta = raw[..., 0:2]         # [dx, dy]
        rho1 = raw[..., 2]
        rho2 = raw[..., 3]
        u = raw[..., 4]
        v = raw[..., 5]

        center = p_hat + delta
        r_raw1 = torch.nn.functional.softplus(rho1) + 1e-6
        r_raw2 = torch.nn.functional.softplus(rho2) + 1e-6
        r1 = torch.maximum(r_raw1, r_raw2)
        r2 = torch.minimum(r_raw1, r_raw2)

        norm = torch.sqrt(u * u + v * v) + 1e-8
        dir_x = u / norm
        dir_y = v / norm
        theta = 0.5 * torch.atan2(dir_y, dir_x)

        return {
            "center": center,
            "r1": r1,
            "r2": r2,
            "dir": torch.stack([dir_x, dir_y], dim=-1),
            "theta": theta,
            "raw": raw,
        }
