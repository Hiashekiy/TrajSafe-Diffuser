"""Differentiable soft rasterisation of safety ellipses (V2).

One implementation, used by
  * the training losses (IoU against the GT mask, full-ellipse safety),
  * the offline GT-mask construction in the dataset,
  * the evaluation metrics,
so a predicted ellipse and a ground-truth ellipse are always rasterised with
exactly the same convention.

The map spans scene [-1, 1]^2 and the raster has 'res' cells per axis; the
occupancy grid is max-pooled the same way so cell (i, j) is unsafe as soon as
any original cell inside it is an obstacle.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["scene_grid_centres", "ellipse_soft_mask", "occupancy_raster"]


def scene_grid_centres(res: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Cell centres of a res x res grid over scene [-1, 1]^2 -> [res*res, 2]."""
    xs = torch.linspace(-1.0, 1.0, res + 1, device=device, dtype=dtype)[:-1] + 1.0 / res
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def occupancy_raster(occ: torch.Tensor, res: int) -> torch.Tensor:
    """occ [B,1,H,W] (1 = obstacle) -> max-pooled free mask [B,1,res,res]."""
    if occ.dim() == 3:
        occ = occ.unsqueeze(1)
    return 1.0 - F.adaptive_max_pool2d(occ.float(), output_size=(res, res))


def ellipse_soft_mask(center: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                      theta: torch.Tensor, res: int, tau: float = 10.0
                      ) -> torch.Tensor:
    """center [B,K,2], a/b/theta [B,K] -> soft masks [B,K,res,res] in [0,1].

    mask = sigmoid(tau * (1 - q)),  q = (x_r/a)^2 + (y_r/b)^2 with the raster
    rotated into the ellipse frame.
    """
    B, K = center.shape[0], center.shape[1]
    grid = scene_grid_centres(res, device=center.device, dtype=center.dtype)
    gx = grid[:, 0].reshape(res, res)[None, None]
    gy = grid[:, 1].reshape(res, res)[None, None]
    dx = gx - center[..., 0][:, :, None, None]
    dy = gy - center[..., 1][:, :, None, None]
    ct = torch.cos(theta)[:, :, None, None]
    st = torch.sin(theta)[:, :, None, None]
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    ac = a.clamp_min(1e-6)[:, :, None, None]
    bc = b.clamp_min(1e-6)[:, :, None, None]
    q = (xr / ac) ** 2 + (yr / bc) ** 2
    return torch.sigmoid(float(tau) * (1.0 - q))
