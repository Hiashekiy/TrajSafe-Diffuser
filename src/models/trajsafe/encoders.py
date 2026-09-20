"""Coordinate encoders for the report-faithful TrajSafe-Diffuser.

    SpatialPE / Phi_xy   shared 2D sinusoidal coordinate encoding with one
                         internal Linear(D, D); the SAME instance is used by
                         trajectory points, skeleton points, map grid centres
                         and ellipse centres (report sections 3.1 / 3.4).
    IndexPE / PE_1D      parameter-free integer index encoding.

``CoordMLP`` is the independent projection used after Phi_xy for coordinates
that are direct network inputs:

    CoordMLP(x) = Linear(D, 256) -> SiLU -> Linear(256, D)

``MLP_T`` (trajectory), ``MLP_S`` (skeleton) and ``MLP_C`` (ellipse centre)
have the same structure but never share parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CoordMLP", "TrajectoryEncoder", "SkeletonEncoder"]


class CoordMLP(nn.Module):
    """Linear(D, hidden) -> SiLU -> Linear(hidden, D)."""

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrajectoryEncoder(nn.Module):
    """Control-token encoder: ``T_i^0 = MLP_T(Phi_xy(q_i)) + PE_1D(i)``.

    Despite the historical attribute name, this is the **Control Encoder** of
    the control-space model: it is applied to the C control points
    ``Q_t [B,C,2]`` (2D points, so the shared Phi_xy coordinate encoding is the
    right input).  The index embedding is computed on the fly, so the module
    carries no token-count-shaped buffer and the same weights serve both the new
    C control tokens and the legacy L = 128 curve tokens.
    """

    def __init__(self, d_model: int, spatial_pe: nn.Module, index_pe: nn.Module,
                 horizon: int, hidden: int = 256):
        super().__init__()
        self.spatial_pe = spatial_pe
        self.index_pe = index_pe
        self.horizon = int(horizon)
        self.mlp_t = CoordMLP(d_model, hidden)

    def forward(self, p_t: torch.Tensor) -> torch.Tensor:
        L = p_t.shape[-2]
        idx = torch.arange(L, device=p_t.device, dtype=torch.long)
        return self.mlp_t(self.spatial_pe(p_t)) + self.index_pe(idx)[None]


class SkeletonEncoder(nn.Module):
    """Static Skeleton Transformer (report section 11.1).

    Input is ONLY the 2D coordinate sequence; no tangent, handcrafted progress
    or path-length feature is concatenated.

        S^0 = LN_in(MLP_S(Phi_xy(s)) + PE_1D(j))
        S   = SkeletonTransformer_{N_S}(S^0)     # pre-norm MHA + FFN
        H_S = LN_out(S)

    The transformer never sees h_t, C_G or C_E.
    """

    def __init__(self, d_model: int, num_heads: int, spatial_pe: nn.Module,
                 index_pe: nn.Module, blocks: int = 2, ffn_dim: int = 512,
                 dropout: float = 0.0, hidden: int = 256):
        super().__init__()
        self.d_model = int(d_model)
        self.spatial_pe = spatial_pe
        self.index_pe = index_pe
        self.mlp_s = CoordMLP(d_model, hidden)
        self.norm_in = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=int(ffn_dim),
            dropout=float(dropout), activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(blocks),
                                             norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)

    def forward(self, candidate_xy: torch.Tensor) -> torch.Tensor:
        """candidate_xy [B,M,L,2] -> H_S [B,M,L,D]."""
        B, M, L, _ = candidate_xy.shape
        f = candidate_xy.reshape(B * M, L, 2)
        idx = torch.arange(L, device=f.device, dtype=torch.long)
        h = self.norm_in(self.mlp_s(self.spatial_pe(f)) + self.index_pe(idx)[None])
        h = self.encoder(h)
        return h.reshape(B, M, L, self.d_model)
