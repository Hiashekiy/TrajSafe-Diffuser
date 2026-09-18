"""TopologySelector: score every complete candidate path (no graph net).

docs/V2.md sections 13/4 and 14.  The selector never predicts a branch
sequence, it only ranks the candidate topologies that the geometry layer
already certified as complete, connected and collision free:

    S_m = MLP_score([g_m, g_tau, d_m, L_m]),   pi = masked_softmax(S)

V2 stage 1 detaches the trajectory features (config
topology.detach_trajectory_feature), so the topology loss can never pull the
diffusion backbone towards one skeleton in order to make classification easier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .path_encoder import PathEncoder

__all__ = ["TopologySelector", "chamfer_mean_distance"]


def chamfer_mean_distance(pred: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """(1/H) sum_i min_j ||pred_i - path_mj||; pred [B,H,2], path [B,M,L,2]."""
    diff = pred[:, None, :, None, :] - path[:, :, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)              # [B,M,H,L]
    return dist.min(dim=-1).values.mean(dim=-1)         # [B,M]


class TopologySelector(nn.Module):
    def __init__(self, d_model, num_heads, hidden=256, dropout=0.0,
                 path_hidden=128, detach_trajectory_feature=True):
        super().__init__()
        self.detach = bool(detach_trajectory_feature)
        self.path_encoder = PathEncoder(d_model, num_heads, hidden=path_hidden,
                                        dropout=dropout)
        self.score = nn.Sequential(
            nn.Linear(2 * d_model + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, traj_feat, x0_p, candidate_paths, candidate_mask,
                candidate_lengths):
        """traj_feat [B,H,D]; x0_p [B,H,2]; candidate_paths [B,M,L,5];
        candidate_mask [B,M] bool; candidate_lengths [B,M]."""
        tr = traj_feat.detach() if self.detach else traj_feat
        path_feat, g_m = self.path_encoder(candidate_paths, tr)
        g_tau = tr.mean(dim=1)[:, None, :].expand_as(g_m)
        d_m = chamfer_mean_distance(x0_p.detach(), candidate_paths[..., :2])
        feat = torch.cat([g_m, g_tau, d_m[..., None],
                          candidate_lengths[..., None]], dim=-1)
        logits = self.score(feat).squeeze(-1)                    # [B,M]
        logits = logits.masked_fill(~candidate_mask, float("-inf"))
        pi = torch.softmax(logits, dim=-1)
        pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
        return {"logits": logits, "pi": pi, "path_feat": path_feat,
                "geo_dist": d_m}
