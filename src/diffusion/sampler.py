"""Reverse-diffusion sampler with start/goal reimposition."""

import torch

from .conditioning import apply_endpoint_condition


def _reimpose_conditions(x, cond, horizon):
    # cond [B,2,2] = (start_xy_norm, goal_xy_norm).  Replace position dims 2,3.
    return apply_endpoint_condition(x, cond, horizon, obs_start=2, obs_end=4)


def sample(model, map_tensor, schedule, cond, n_samples, device="cuda",
           steps=None, reimpose=True, return_traj=False, return_timesteps=False):
    """Sample clean trajectories conditioned on start/goal positions.

    cond: [B,2,2] normalized (start_xy, goal_xy).  steps None => full schedule.
    Returns (x0 [B,H,6] normalized, traj_log[, t_log]).
    When ``return_timesteps``, also returns the diffusion-time ``t`` used at
    each recorded step (same length as ``traj_log``).
    """
    model.eval()
    B = cond.shape[0]
    horizon = model.horizon
    if steps is None or steps >= schedule.num_timesteps:
        timesteps = list(range(schedule.num_timesteps - 1, -1, -1))
    else:
        idx = torch.linspace(0, schedule.num_timesteps - 1, steps).long().tolist()
        timesteps = list(reversed(idx))
    x = torch.randn(B, horizon, 6, device=device, dtype=torch.float32)
    traj_log = []
    t_log = []
    with torch.no_grad():
        for t in timesteps:
            tb = torch.full((B,), t, device=device, dtype=torch.long)
            out = model(x, tb, map_tensor, cond=cond)
            x0_pred = out["x0_pred"]
            if t == 0:
                x = x0_pred.detach().float()
            else:
                t_idx = torch.full((1,), t, device=device, dtype=torch.long)
                alpha_t = schedule.alphas[t_idx][0].item()
                sqrt_ab = schedule.sqrt_alphas_cumprod[t_idx][0].item()
                sqrt_one_ma = schedule.sqrt_one_minus_alphas_cumprod[t_idx][0].item()
                beta_t = schedule.betas[t_idx][0].item()
                eps = (x.double() - sqrt_ab * x0_pred.double()) / sqrt_one_ma
                prev = (x.double() - beta_t / sqrt_one_ma * eps) / (alpha_t ** 0.5)
                pvar = schedule.posterior_variance[t_idx][0].item()
                prev = prev + (pvar ** 0.5) * torch.randn_like(prev)
                x = prev.float()
            if reimpose:
                x = _reimpose_conditions(x, cond, horizon)
            if return_traj:
                traj_log.append(x.detach().clone())
                t_log.append(t)
    if return_timesteps:
        return x, traj_log, t_log
    return x, traj_log
