"""Task heads for the report-faithful TrajSafe-Diffuser.

    TopologyHead       R [B,M,H,D] -> MeanPool_H -> MLP_score -> l_m -> pi
                       (masked softmax; invalid logits are -inf)
    ProgressHead       R_use [B,H,D] -> MLP_prog -> Head_prog ->
                       positive gaps -> cumulative monotone s [B,H]
    EllipseShapeHead   H_ell [B,H,D] -> AdaLN -> MLP -> [l1,l2,u,v]
                       -> shape4 = [log a, log b, cos 2t, sin 2t]

There is deliberately NO ellipse-centre output anywhere in these heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...geometry.ellipse_shape import raw_to_shape4, shape4_to_abtheta
from ..joint.joint_blocks import AdaLN

__all__ = ["TopologyHead", "ProgressHead", "EllipseShapeHead"]


class TopologyHead(nn.Module):
    """Candidate-level scorer; parameters are shared over every waypoint."""

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, r: torch.Tensor, candidate_mask: torch.Tensor):
        """r [B,M,H,D], candidate_mask [B,M] bool.

        Returns a dict with:
            logits [B,M]  invalid entries are -inf
            pi     [B,M]  masked softmax; an all-invalid row is all zeros
        """
        if r.dim() != 4:
            raise ValueError("TopologyHead expects R [B,M,H,D]")
        if candidate_mask.shape != r.shape[:2]:
            raise ValueError("candidate_mask must be [B,M]")
        z = r.mean(dim=2)                                  # [B,M,D]
        logits = self.score(z).squeeze(-1)                 # [B,M]
        masked = logits.masked_fill(~candidate_mask, float("-inf"))

        any_valid = candidate_mask.any(dim=-1, keepdim=True)
        row_max = masked.max(dim=-1, keepdim=True).values
        safe_max = torch.where(any_valid, row_max,
                               torch.zeros_like(row_max))
        exp = torch.exp(masked - safe_max)
        exp = torch.where(candidate_mask, exp, torch.zeros_like(exp))
        denom = exp.sum(dim=-1, keepdim=True)
        pi = torch.where(any_valid,
                         exp / denom.clamp_min(1e-12),
                         torch.zeros_like(exp))
        return {"logits": masked, "pi": pi}


class ProgressHead(nn.Module):
    """Token-wise progress head; no cross-attention and no temporal mixing.

    ``MLP_prog`` is a per-waypoint projection of the SHARED matching feature
    R_use.  Only the first H-1 waypoints feed ``Head_prog`` (the last raw scalar
    would correspond to a non-existent interval).
    """

    def __init__(self, d_model: int, hidden: int = 256, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)
        self.mlp_prog = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), d_model),
        )
        self.head_prog = nn.Sequential(
            nn.Linear(d_model, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, r_use: torch.Tensor):
        """r_use [B,H,D] -> (H_prog [B,H,D], s [B,H])."""
        if r_use.dim() != 3:
            raise ValueError("ProgressHead expects R_use [B,H,D]")
        h_prog = self.mlp_prog(r_use)
        u = self.head_prog(h_prog[:, :-1]).squeeze(-1)      # [B,H-1]
        w = F.softplus(u) + self.eps
        delta = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        zero = torch.zeros_like(delta[:, :1])
        s = torch.cat([zero, torch.cumsum(delta, dim=1)], dim=1)   # [B,H]
        return h_prog, s


class EllipseShapeHead(nn.Module):
    """AdaLN-conditioned per-waypoint MLP with the stable 4-parameter output."""

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.adaln = AdaLN(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), 4),
            nn.Dropout(float(dropout)),
        )

    def forward(self, h_ell: torch.Tensor, h_t: torch.Tensor):
        """h_ell [B,H,D] -> raw [B,H,4], shape4 [B,H,4], a/b/theta [B,H]."""
        raw = self.mlp(self.adaln(h_ell, h_t))
        shape4 = raw_to_shape4(raw)
        a, b, theta = shape4_to_abtheta(shape4)
        return {"raw": raw, "shape4": shape4, "a": a, "b": b, "theta": theta}
