"""Local Scene Sampler: grid_sample a window around each waypoint position."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSceneSampler(nn.Module):
    def __init__(self, local_window=5):
        super().__init__()
        self.local_window = local_window

    def forward(self, scene_map_pe, p_world, extent):
        """scene_map_pe [B,C,Hs,Ws]; p_world [B,H,2]; extent (x0,x1,y0,y1).

        Returns [B,H,Nl,C].
        """
        B, C, Hs, Ws = scene_map_pe.shape
        H = p_world.shape[1]
        w = self.local_window
        x0, x1, y0, y1 = extent
        nx = (p_world[..., 0] - x0) / (x1 - x0) * 2.0 - 1.0
        ny = (p_world[..., 1] - y0) / (y1 - y0) * 2.0 - 1.0
        offs_x = (torch.arange(w, device=p_world.device, dtype=torch.float32) - (w - 1) / 2.0) * (2.0 / Ws)
        offs_y = (torch.arange(w, device=p_world.device, dtype=torch.float32) - (w - 1) / 2.0) * (2.0 / Hs)
        gx = (nx[..., None, None] + offs_x[None, None, :, None]).expand(B, H, w, w)
        gy = (ny[..., None, None] + offs_y[None, None, None, :]).expand(B, H, w, w)
        grid = torch.stack([gx, gy], dim=-1).reshape(B * H, w, w, 2)
        # expand scene map over the B*H batch
        scene_b = scene_map_pe.repeat_interleave(H, dim=0).contiguous()
        sampled = F.grid_sample(scene_b, grid, mode="bilinear", padding_mode="border",
                                align_corners=False)
        sampled = sampled.permute(0, 2, 3, 1).reshape(B * H, w * w, C)
        return sampled.reshape(B, H, w * w, C)
