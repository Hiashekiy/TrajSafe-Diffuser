"""DDIM reverse sampler for the V1 joint model (docs/联合扩散.md #28).

The model predicts clean values directly (x0 heads).  With the shared T=16
cosine schedule sqrt(alpha_bar_15) ~ 0.003, epsilon-based DDPM/DDIM divides by
sqrt(alpha_bar) and is numerically unstable, so the reverse pass is the DDIM
formulation driven by x0 predictions.

Full run (steps=None): x_15 -> x_14 -> ... -> x_0 exactly one level per step.

Sub-sampled run (--steps S < T): S network forwards that JUMP between the
selected levels.  For selected times t_0 > t_1 > ... > 0 each transition goes
directly x_t -> x_s where s is the NEXT actually-selected timestep:

    eps_hat  = (x_t - sqrt(ab_t) x0_hat) / sqrt(1-ab_t)
    x_s      = sqrt(ab_s) x0_hat + sqrt(1-ab_s) eps_hat     (s > 0)
    x_0      = x0_hat                                        (final level)

(Previously the code always used s = t-1, so with --steps the state drifted:
it produced x_{t-1} but labelled/used it at the next selected t.)

Trajectory endpoints are hard-conditioned: after every reverse step p_0 and
p_{H-1} are overwritten with the exact scene start/goal (training convention),
so they always hold.

Returns scene-frame P [B,H,2] and the 6D ellipse repr E6 [B,H,6].
"""
import torch


def _pick_times(T, steps):
    if steps is None or steps >= T:
        return None                       # full per-step run
    idx = torch.linspace(0, T - 1, steps).long().tolist()
    times = sorted(set(idx))
    # always anchor the run at the top level and end at the clean level 0
    if T - 1 not in times:
        times.append(T - 1)
    if 0 not in times:
        times.append(0)
    return sorted(times)


def sample_joint(model, schedule, cond, map_tensor, device="cuda",
                 steps=None, seed=None):
    """cond [B,2,2] scene (start,goal); map_tensor [B,1,256,256]."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    B, H = cond.shape[0], model.horizon
    T = schedule.num_timesteps
    start, goal = cond[:, 0], cond[:, 1]

    sqrt_ab = schedule.sqrt_alphas_cumprod.detach().cpu().tolist()
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod.detach().cpu().tolist()

    times = _pick_times(T, steps)

    p = torch.randn(B, H, 2, device=device, dtype=torch.float32)
    e = torch.randn(B, H, 6, device=device, dtype=torch.float32)
    p[:, 0] = start
    p[:, -1] = goal

    def endpoints(x):
        x[:, 0] = start
        x[:, -1] = goal
        return x

    with torch.no_grad():
        if times is None:
            # ---- full per-step run: x_{t} -> x_{t-1} (unchanged default) ----
            for t in reversed(range(T)):
                tb = torch.full((B,), t, device=device, dtype=torch.long)
                ab = torch.full((B,), float(sqrt_ab[t]), device=device, dtype=torch.float32)
                out = model(p, e, map_tensor, cond, tb, ab)
                x0_p, x0_e = out["x0_p"], out["x0_e"]
                if t == 0:
                    p, e = x0_p, x0_e
                else:
                    sa_t, s1_t = float(sqrt_ab[t]), float(sqrt_1ma[t])
                    sa_prev, s1_prev = float(sqrt_ab[t - 1]), float(sqrt_1ma[t - 1])
                    eps_p = (p - sa_t * x0_p) / s1_t
                    eps_e = (e - sa_t * x0_e) / s1_t
                    p = sa_prev * x0_p + s1_prev * eps_p
                    e = sa_prev * x0_e + s1_prev * eps_e
                p = endpoints(p)
        else:
            # ---- sub-sampled run: x_{times[i]} -> x_{times[i-1]} ----
            for i in range(len(times) - 1, 0, -1):
                t, s = times[i], times[i - 1]
                tb = torch.full((B,), t, device=device, dtype=torch.long)
                ab = torch.full((B,), float(sqrt_ab[t]), device=device, dtype=torch.float32)
                out = model(p, e, map_tensor, cond, tb, ab)
                x0_p, x0_e = out["x0_p"], out["x0_e"]
                if s == 0:
                    p, e = x0_p, x0_e
                else:
                    sa_t, s1_t = float(sqrt_ab[t]), float(sqrt_1ma[t])
                    sa_s, s1_s = float(sqrt_ab[s]), float(sqrt_1ma[s])
                    eps_p = (p - sa_t * x0_p) / s1_t
                    eps_e = (e - sa_t * x0_e) / s1_t
                    p = sa_s * x0_p + s1_s * eps_p
                    e = sa_s * x0_e + s1_s * eps_e
                p = endpoints(p)
    return p, e
