"""Trajectory losses in the scene frame [-1,1]^2."""

import torch
import torch.nn.functional as F

from src.geometry.scene_frame import sample_sdf_scene


def l_z(z0_pred, z0_gt):
    """MSE between the projected predicted residual and the GT residual."""
    return F.mse_loss(z0_pred.float(), z0_gt.float())


def l_p(pos_pred, pos_gt):
    """MSE on the intermediate waypoints (excludes start and goal).

    pos [B,H,2] scene.  intermediate = indices 1 .. H-2 (k=1..N-1).
    """
    return F.mse_loss(pos_pred[:, 1:-1].float(), pos_gt[:, 1:-1].float())


def l_smooth(z0_pred):
    """Second-difference Huber on the residual (equivalent to positions)."""
    if z0_pred.shape[1] < 3:
        return torch.zeros((), device=z0_pred.device, requires_grad=True)
    acc = z0_pred[:, 2:] - 2.0 * z0_pred[:, 1:-1] + z0_pred[:, :-2]
    return F.smooth_l1_loss(acc, torch.zeros_like(acc))


def l_collision(pos_pred, sdf_map, margin=0.0, sigma=0.1):
    """Softplus penalty for intermediate positions in / too close to obstacles.

    pos_pred [B,H,2] scene coords; sdf_map [B,1,256,256] scene SDF.
    """
    mid = pos_pred[:, 1:-1]
    d = sample_sdf_scene(sdf_map, mid)
    return F.softplus((margin - d) / sigma).mean()
