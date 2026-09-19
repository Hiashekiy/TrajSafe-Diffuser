"""Ellipse geometry query (report sections 17 and 18).

    e_i^c = MLP_C(Phi_xy(c_i))
    q_i^E = h_i^prog + e_i^c
    A^E   = CenterBiasedCrossAttention(AdaLN(q^E, h_t), C_E, B_geo)
    H_ell = q^E + A^E

The spatial bias uses the *diffusion strength* exactly as specified:

    B_ij = Clamp(-alpha_bar_t * ||c_i - q_j^E||^2 / (2 sigma_geo^2),
                 -b_clip, 0)

where the schedule argument ``ab`` is sqrt(alpha_bar_t), therefore the code
MUST use ``ab ** 2`` (this is the ``bar alpha_t = (sqrt bar alpha_t)^2`` note in
the report).  The bias is shared by all attention heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...geometry.ellipse_raster import scene_grid_centres
from ..joint.joint_blocks import AdaLN, _MHABase
from .encoders import CoordMLP

__all__ = ["EllipseGeometry"]


class EllipseGeometry(_MHABase):
    """Center embedding + one center-biased cross-attention into C_E."""

    def __init__(self, d_model: int, num_heads: int, spatial_pe: nn.Module,
                 geo_mem_res: int = 32, geo_sigma: float = 0.25,
                 geo_bias_clip: float = 8.0, hidden: int = 256,
                 dropout: float = 0.0):
        super().__init__(d_model, num_heads, dropout)
        self.spatial_pe = spatial_pe
        self.geo_mem_res = int(geo_mem_res)
        self.geo_sigma = float(geo_sigma)
        self.geo_bias_clip = float(geo_bias_clip)
        self.mlp_c = CoordMLP(d_model, hidden)
        self.adaln = AdaLN(d_model, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)

    def _geo_bias(self, center: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        """center [B,H,2], ab [B] = sqrt(alpha_bar) -> bias [B,1,H,N_E]."""
        grid = scene_grid_centres(self.geo_mem_res, center.device,
                                  dtype=center.dtype)
        dist2 = ((center[:, :, None, :] - grid[None, None, :, :]) ** 2).sum(-1)
        strength = (ab.to(center.dtype) ** 2)[:, None, None]
        bias = -strength * dist2 / (2.0 * self.geo_sigma ** 2)
        return bias.clamp(-self.geo_bias_clip, 0.0)[:, None]

    def forward(self, h_prog: torch.Tensor, center: torch.Tensor,
                geo_mem: torch.Tensor | None, h_t: torch.Tensor,
                ab: torch.Tensor | None):
        """h_prog [B,H,D], center [B,H,2] -> (H_ell [B,H,D], A_E or None)."""
        q_e = h_prog + self.mlp_c(self.spatial_pe(center))
        if geo_mem is None:
            # The report defines A^E = 0 when C_E is unavailable.
            return q_e, None
        bias = self._geo_bias(center, ab) if ab is not None else None
        k, v = self.kv(geo_mem).chunk(2, dim=-1)
        att = self.attend(self.q(self.adaln(q_e, h_t)), k, v, bias)
        return q_e + att, att
