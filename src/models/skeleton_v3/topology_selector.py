"""Skeleton Selector (docs section 10).

    l_m = MLP[ g_tau, g_m^S, d_m, L_m, TimeProj(h_t) ]
    pi  = masked_softmax(l_m)          invalid: l_m = -inf

Recomputed at EVERY diffusion timestep - there is no commit step and no cache.
The trajectory features are NOT detached: L_coarse already supervises the coarse
prediction directly, so letting L_topo shape the backbone is intended here
(unlike V2, which had to protect a frozen refine stage).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .path_encoder import StaticPathEncoder

__all__ = ["TopologySelector", "chamfer_mean_distance"]


def chamfer_mean_distance(pred: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """(1/H) sum_i min_j ||pred_i - path_mj||; pred [B,H,2], path [B,M,L,2]."""
    diff = pred[:, None, :, None, :] - path[:, :, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)
    return dist.min(dim=-1).values.mean(dim=-1)


class TopologySelector(nn.Module):
    def __init__(self, d_model, num_heads, hidden=256, dropout=0.0,
                 path_hidden=128, path_layers=2):
        super().__init__()
        self.path_encoder = StaticPathEncoder(d_model, num_heads,
                                              hidden=path_hidden,
                                              dropout=dropout,
                                              layers=path_layers)
        self.time_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.score = nn.Sequential(
            nn.Linear(3 * d_model + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, traj_feat, coarse_p0, h_t, candidate_features,
                candidate_mask, candidate_lengths):
        path_feat, g_m = self.path_encoder(candidate_features)
        g_tau = traj_feat.mean(dim=1)[:, None, :].expand_as(g_m)
        d_m = chamfer_mean_distance(coarse_p0.detach(),
                                    candidate_features[..., :2])
        t_proj = self.time_proj(h_t)[:, None, :].expand_as(g_m)
        feat = torch.cat([g_tau, g_m, d_m[..., None],
                          candidate_lengths[..., None], t_proj], dim=-1)
        logits = self.score(feat).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, float("-inf"))
        pi = torch.softmax(logits, dim=-1)
        pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
        return {"logits": logits, "pi": pi, "path_feat": path_feat,
                "geo_dist": d_m}
