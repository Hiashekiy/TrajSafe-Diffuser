"""Deterministic zero-sum reverse sampler.

All state stays in the zero-sum subspace: Z_T, Z_t, eps, Z_{t-1} are all zero-sum,
so the integrated positions always satisfy p_0 = start and p_N = goal.
Returns scene-frame positions [B,H,2].

Optionally it applies the V5 consensus guidance (J_guide) during later (low
noise) reverse steps: a differentiable gradient correction on z0_pred using the
local ellipse consensus geometry, followed by a zero-sum projection.
"""

import torch

from .zerosum import zero_sum, integrate_positions, compute_base
from src.guidance.consensus_guidance import apply_consensus_guidance, guidance_weight


def sample(model, map_tensor, schedule, cond, n_samples, device="cuda",
           steps=None, return_traj=False, return_timesteps=False,
           guidance_cfg=None, return_guidance=False):
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

    use_guidance = bool(guidance_cfg and guidance_cfg.get("enabled", True))

    z = torch.randn(B, N, 2, device=device, dtype=torch.float32)
    z = zero_sum(z)
    traj_log = []
    t_log = []
    guidance_stats = []
    pos_pred_final = None
    for t in timesteps:
        delta_t = base + z
        pos_t = integrate_positions(start, delta_t)         # [B,H,2]
        tb = torch.full((B,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            out = model(z, tb, map_tensor, cond=cond)
        z0_raw = out["z0_pred"]                              # detached

        if use_guidance and out.get("ellipse_center") is not None:
            tt = torch.tensor([t], device=device)
            w_G = guidance_weight(tt, schedule.num_timesteps,
                                  guidance_cfg.get("start_t_ratio", 0.40),
                                  guidance_cfg.get("full_t_ratio", 0.10))
            active = float(w_G.max().item()) > 0.0
        else:
            active = False

        if active:
            z0_pred, stats = apply_consensus_guidance(
                z0_raw.detach(), base, start,
                out["ellipse_center"], out["ellipse_radii"],
                out["ellipse_theta"],
                guidance_cfg, tt, schedule.num_timesteps)
            guidance_stats.append(stats)
            z0_pred = z0_pred.detach()
        else:
            z0_pred = zero_sum(z0_raw)

        # The deliverable is the path integrated from the (possibly guided)
        # clean residual -- guidance acts on the predicted path, not on a raw z.
        pos_pred_final = integrate_positions(start, base + z0_pred)

        with torch.no_grad():
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

    final = pos_pred_final if pos_pred_final is not None else pos_t
    if return_timesteps:
        if return_guidance:
            return final, traj_log, t_log, guidance_stats
        return final, traj_log, t_log
    if return_guidance:
        return final, traj_log, guidance_stats
    return final, traj_log
