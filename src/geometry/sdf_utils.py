"""Signed distance field (SDF) sampling utilities.

Positive = free space, negative = inside an obstacle.  Only differentiable
bilinear sampling (sample_sdf_torch) is used by the active pipeline.
"""
import torch
import torch.nn.functional as F


def sample_sdf_torch(sdf_map, points, extent):
    """Differentiable bilinear sample of an SDF field at world points.

    sdf_map: [B,1,ny,nx] tensor.  points: [B,H,2] (obs-frame x,y).
    extent: (x0,x1,y0,y1).  Returns [B,H].
    """
    x0, x1, y0, y1 = extent
    xs = (points[..., 0] - x0) / (x1 - x0) * 2.0 - 1.0
    ys = (points[..., 1] - y0) / (y1 - y0) * 2.0 - 1.0
    grid = torch.stack([xs, ys], dim=-1)          # [B,H,2]
    grid = grid.unsqueeze(2)                       # [B,H,1,2]
    out = F.grid_sample(sdf_map, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.squeeze(1).squeeze(-1)              # [B,H]
