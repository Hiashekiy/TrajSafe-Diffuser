"""Local Scene Sampler: sample a window around each waypoint from the full-res
scene feature, and add a learned relative position encoding per window cell.

Uses grid_sample with a per-batch grid (shape [B, N*w*w, 2]) so the 256x256
feature map is NOT replicated across B*N -- this is memory-safe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSceneSampler(nn.Module):
    def __init__(self, local_window=7, d_model=128, res=256):
        super().__init__()
        self.local_window = local_window
        self.res = res
        w = local_window
        self.rel_pos = nn.Embedding(w * w, d_model)
        self.abs_pos = nn.Linear(2, d_model)     # absolute waypoint position encoding
        offs = (torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0) * (2.0 / res)
        self.register_buffer("offs", offs)
        ox, oy = torch.meshgrid(offs, offs, indexing="xy")
        self.register_buffer("offs_xy", torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1))  # [w*w,2]

    def forward(self, scene_local_pe, p_scene):
        """scene_local_pe [B,C,256,256]; p_scene [B,N,2] -> [B,N,w*w,C]."""
        B, C, Hs, Ws = scene_local_pe.shape
        N = p_scene.shape[1]
        w = self.local_window
        grid = p_scene[:, :, None, :] + self.offs_xy[None, None, :, :]   # [B,N,w*w,2]
        grid = grid.reshape(B, N * w * w, 2).unsqueeze(2)                # [B,N*w*w,1,2]
        sampled = F.grid_sample(scene_local_pe, grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2)                    # [B,N*w*w,C]
        idx = torch.arange(w * w, device=p_scene.device)
        rel = self.rel_pos(idx)[None, None, :, :]                        # [1,1,w*w,C]
        pos_emb = self.abs_pos(p_scene)[:, :, None, :]                   # [B,N,1,C]
        sampled = sampled.reshape(B, N, w * w, C) + rel + pos_emb
        return sampled
