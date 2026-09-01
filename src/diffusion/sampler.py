"""Deterministic zero-sum reverse sampler.

All state stays in the zero-sum subspace: Z_T, Z_t, eps, Z_{t-1} are all zero-sum,
so the integrated positions always satisfy p_0 = start and p_N = goal.
Returns scene-frame positions [B,H,2].
"""

import torch

from .zerosum import zero_sum, integrate_positions, compute_base


def sample(model, map_tensor, schedule, cond, n_samples, device="cuda",
           steps=None, return_traj=False, return_timesteps=False):
    model.eval()
    B = cond.shape[0]
    H = model.horizon
    N = H - 1
    start = cond[:, 0]
    goal = cond[:, 1]
    g = goal - start
    base = compute_base(g, N)

    if steps is None or steps >= schedule.num_timesteps:
        timesteps = list(range(schedule.num_timesteps - 1, -1, -1))
    else:
        idx = torch.linspace(0, schedule.num_timesteps - 1, steps).long().tolist()
        timesteps = list(reversed(idx))

    z = torch.randn(B, N, 2, device=device, dtype=torch.float32)
    z = zero_sum(z)
    traj_log = []
    t_log = []
    with torch.no_grad():
        for t in timesteps:
            delta_t = base + z
            pos_t = integrate_positions(start, delta_t)         # [B,H,2]
            tb = torch.full((B,), t, device=device, dtype=torch.long)
            out = model(z, tb, map_tensor, cond=cond)
            z0_pred = zero_sum(out["z0_pred"])
            t_i = torch.full((1,), t, device=device, dtype=torch.long)
            alpha_t = float(schedule.alphas[t_i][0])
            sqrt_ab = float(schedule.sqrt_alphas_cumprod[t_i][0])
            sqrt_one = float(schedule.sqrt_one_minus_alphas_cumprod[t_i][0])
            beta_t = float(schedule.betas[t_i][0])
            eps = (z.double() - sqrt_ab * z0_pred.double()) / sqrt_one
            z = ((z.double() - beta_t / sqrt_one * eps) / (alpha_t ** 0.5)).float()
            if return_traj:
                traj_log.append(pos_t.detach().clone())
                t_log.append(t)
    if return_timesteps:
        return pos_t, traj_log, t_log
    return pos_t, traj_log
