"""Trajectory generation losses: diffusion reconstruction, collision, smoothness."""

import torch
import torch.nn.functional as F

from src.geometry.sdf_utils import sample_sdf_torch


def l_diff(x0_pred, x0, lambda_var=1.0, lambda_var_vel=1.0):
    """Trajectory reconstruction loss.

    Returns MSE(x0_pred, x0) PLUS a temporal-variation term that penalises
    "constant-collapse" solutions: if the model predicts a single point for the
    whole trajectory, the consecutive-difference term is 0 while the GT
    difference is non-zero, so the loss is high.  This directly counters the
    mean-collapse problem of plain MSE.
    """
    x0_pred = x0_pred.float()
    x0 = x0.float()
    base = F.mse_loss(x0_pred, x0)
    # variation along the horizon (what changes from one waypoint to the next)
    d_pred = x0_pred[:, 1:] - x0_pred[:, :-1]
    d_gt = x0[:, 1:] - x0[:, :-1]
    var = F.mse_loss(d_pred, d_gt)
    # emphasise variation on the position+velocity dims (2:6) so a constant
    # position/velocity path is strongly penalised
    var_vel = F.mse_loss(d_pred[:, :, 2:6], d_gt[:, :, 2:6])
    return base + float(lambda_var) * var + float(lambda_var_vel) * var_vel


def l_collision(x0_pred, sdf_map, extent, margin=0.0, sigma=0.1, state_norm=None):
    """Penalise predicted positions that are inside / too close to obstacles.

    x0_pred [B,H,6] normalized; positions converted to world before sampling SDF.
    """
    device = x0_pred.device
    dtype = x0_pred.dtype
    mins = torch.as_tensor(state_norm.mins[2:4], device=device, dtype=dtype)
    maxs = torch.as_tensor(state_norm.maxs[2:4], device=device, dtype=dtype)
    eps = state_norm.eps
    p_pred = (x0_pred[:, :, 2:4] + 1.0) / 2.0 * (maxs - mins + eps) + mins
    d = sample_sdf_torch(sdf_map, p_pred, extent)
    return torch.nn.functional.softplus((margin - d) / sigma).mean()


def l_smooth(x0_pred, state_norm):
    device = x0_pred.device
    dtype = x0_pred.dtype
    mins = torch.as_tensor(state_norm.mins[2:4], device=device, dtype=dtype)
    maxs = torch.as_tensor(state_norm.maxs[2:4], device=device, dtype=dtype)
    eps = state_norm.eps
    p = (x0_pred[:, :, 2:4] + 1.0) / 2.0 * (maxs - mins + eps) + mins
    if p.shape[1] < 3:
        return torch.zeros((), device=device, requires_grad=True)
    acc = p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]
    return (acc ** 2).sum(-1).mean()
