"""Final feature fusion and clean-trajectory denoiser (report sections 21/22).

    Z_i = [h_i^traj ; h_i^path ; h_i^ell]        (3D channels)
    F   = MLP_fuse(Z)                           (LN(3D) -> 2D -> GELU -> D)
    H_clean = FinalDenoiser_{N_F}(F, C_G, h_t)

``h_path`` is the output of ``PathFeatureHead`` (the renamed former MLP_prog);
it is NOT a learned-progress feature.  The progress feeding the ellipse centre
decoder is the fixed buffer ``s_i = i/127`` inside the planner.

The Final Denoiser blocks have exactly the structure of the Trajectory Backbone
``TrajBlock`` (AdaLN self-attention -> AdaLN global-map cross-attention ->
AdaLN FFN) but completely independent parameters.  There is no interleaved
trajectory/ellipse token stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import TrajBlock

__all__ = ["FusionMLP", "FinalDenoiser"]


class FusionMLP(nn.Module):
    """Channel-only fusion of the three H-waypoint latents."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, h_traj: torch.Tensor, h_path: torch.Tensor,
                h_ell: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_traj, h_path, h_ell], dim=-1))


class FinalDenoiser(nn.Module):
    """Independent stack of ``TrajBlock`` modules (N_F = 3 for the baseline)."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, horizon: int,
                 blocks: int = 3, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TrajBlock(d_model, num_heads, ff_dim, horizon, dropout)
            for _ in range(int(blocks))
        ])

    def forward(self, f: torch.Tensor, c_g: torch.Tensor,
                h_t: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            f = blk(f, c_g, h_t)
        return f
