"""DDIM reverse sampler for the V1 joint model (docs/联合扩散.md #28).

The model predicts clean values directly (x0 heads).  With the shared T=16
cosine schedule sqrt(alpha_bar_15) ~ 0.003, epsilon-based DDPM/DDIM divides by
sqrt(alpha_bar) and is numerically unstable, so the reverse pass is the DDIM
formulation driven by x0 predictions:

    eps_t    = (x_t - sqrt(ab_t) x0_hat) / sqrt(1-ab_t)
    x_{t-1}  = sqrt(ab_{t-1}) x0_hat + sqrt(1-ab_{t-1}) eps_t

Trajectory endpoints are hard-conditioned: after every reverse step p_0 and
p_{H-1} are overwritten with the exact scene start/goal (same convention used
in training), so they always hold.

Returns scene-frame P [B,H,2] and the 6D ellipse repr E6 [B,H,6].
"""
import torch


def sample_joint(model, schedule, cond, map_tensor, device="cuda",
                 steps=None, seed=None):
    """cond [B,2,2] scene (start,goal); map_tensor [B,1,256,256]."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    B, H = cond.shape[0], model.horizon
    T = schedule.num_timesteps
    if steps is None:
        ts = list(range(T))
    else:
        idx = torch.linspace(0, T - 1, steps).long().tolist()
        ts = sorted(set(idx))
    start, goal = cond[:, 0], cond[:, 1]

    sqrt_ab = schedule.sqrt_alphas_cumprod.detach().cpu().tolist()
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod.detach().cpu().tolist()

    p = torch.randn(B, H, 2, device=device, dtype=torch.float32)
    e = torch.randn(B, H, 6, device=device, dtype=torch.float32)
    p[:, 0] = start
    p[:, -1] = goal

    with torch.no_grad():
        for t in reversed(range(T)):
            if t not in ts:
                continue
            tb = torch.full((B,), t, device=device, dtype=torch.long)
            ab = torch.full((B,), float(sqrt_ab[t]), device=device, dtype=torch.float32)
            out = model(p, e, map_tensor, cond, tb, ab)
            x0_p = out["x0_p"]
            x0_e = out["x0_e"]

            if t == 0:
                p, e = x0_p, x0_e
            else:
                sa_t, s1_t = float(sqrt_ab[t]), float(sqrt_1ma[t])
                sa_prev, s1_prev = float(sqrt_ab[t - 1]), float(sqrt_1ma[t - 1])
                # eps implied by the x0 prediction, then deterministic DDIM step
                eps_p = (p - sa_t * x0_p) / s1_t
                eps_e = (e - sa_t * x0_e) / s1_t
                p = sa_prev * x0_p + s1_prev * eps_p
                e = sa_prev * x0_e + s1_prev * eps_e
            # hard endpoint conditioning (P only)
            p[:, 0] = start
            p[:, -1] = goal
    return p, e
