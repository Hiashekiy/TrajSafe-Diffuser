"""Task heads for the control-space TrajSafe-Diffuser.

    TopologyHead        R [B,M,H,D] -> MeanPool_H -> MLP_score -> l_m -> pi
                        (masked softmax; invalid logits are -inf)
    PathFeatureHead     R_use [B,H,D] -> MLP_path -> H_path [B,H,D]
                        Pure feature transform.  The old learned per-waypoint
                        progress MLP with its cumulative softplus parameterisation
                        is GONE: the ellipse centres now come from the fixed
                        progress s_i = i/127 on the selected dense Skeleton curve.
    TrajectoryToControlHead
                        fixed endpoint-constrained LS projection of a decoded
                        128-point curve onto the 32 B-spline controls
                        (re-exported from ``src.geometry.bspline``).
    EllipseShapeHead    H_ell [B,H,D] -> AdaLN -> MLP -> [l1,l2,u,v]
                        -> shape4 = [log a, log b, cos 2t, sin 2t]

There is deliberately NO ellipse-centre output and NO progress output anywhere
in these heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...geometry.bspline import TrajectoryToControlHead
from ...geometry.ellipse_shape import raw_to_shape4, shape4_to_abtheta
from ..common.blocks import AdaLN

__all__ = ["TopologyHead", "PathFeatureHead", "EllipseShapeHead",
           "TrajectoryToControlHead"]


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


class PathFeatureHead(nn.Module):
    """Per-waypoint feature transform of the shared matching feature R_use.

    This is the former ``MLP_prog`` with its semantics fixed: it produces a
    path feature, NOT a progress distribution.  It is followed by the fixed
    buffer ``s_i = i/127`` in the planner; nothing here is monotone, cumulative
    or supervised by an alignment loss.
    """

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), d_model),
        )

    def forward(self, r_use: torch.Tensor) -> torch.Tensor:
        """r_use [B,H,D] -> H_path [B,H,D]."""
        if r_use.dim() != 3:
            raise ValueError("PathFeatureHead expects R_use [B,H,D]")
        return self.net(r_use)


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
