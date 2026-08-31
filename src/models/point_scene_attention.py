"""Point-Scene Cross-Attention: each waypoint query attends to local scene."""

import torch.nn as nn


class PointSceneAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.q_proj = nn.Linear(d_model, d_model)

    def forward(self, F_traj, local_scene):
        """F_traj [B,H,C], local_scene [B,H,Nl,C] -> [B,H,C]."""
        B, H, C = F_traj.shape
        q = self.q_proj(F_traj).reshape(B * H, 1, C)
        kv = local_scene.reshape(B * H, local_scene.shape[2], C)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out.reshape(B, H, C)
