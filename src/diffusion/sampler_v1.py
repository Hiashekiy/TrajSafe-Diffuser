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

from src.diffusion.alm_guidance import alm_correct
from src.geometry.convex_corridor import EllipseRegionBuilder
from src.geometry.ellipse_center_repair import (
    EllipseCenterRepair,
    reencode_ellipse_centers,
)


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
                 steps=None, seed=None, alm_config=None,
                 return_alm_stats=False):
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

    alm_cfg = alm_config or {}
    alm_enabled = bool(alm_cfg.get("enabled", False))
    corridor_builder = (EllipseRegionBuilder(map_tensor, alm_cfg)
                        if alm_enabled else None)
    center_repair_enabled = alm_enabled and bool(
        alm_cfg.get("center_repair", True))
    center_repairer = (EllipseCenterRepair(corridor_builder)
                       if center_repair_enabled else None)
    rho = float(alm_cfg.get("rho", alm_cfg.get("rho_init", 5.0)))
    alm_start_t = int(alm_cfg.get("start_t", 7))
    collected_stats = []

    def endpoints(x):
        x[:, 0] = start
        x[:, -1] = goal
        return x

    def guide_clean_prediction(x0_p, x0_e, t):
        x0_p = endpoints(x0_p)
        if not alm_enabled or t > alm_start_t:
            return x0_p, x0_e
        center_stats = {}
        if center_repair_enabled:
            repair = center_repairer(x0_p, x0_e, start)
            x0_e = repair.ellipse
            physical_centers = repair.centers
            point_A, point_b = repair.A, repair.b
            point_mask, point_valid = repair.face_mask, repair.valid
            center_stats = repair.stats
        else:
            clean_e = torch.nan_to_num(x0_e, nan=0.0, posinf=0.0, neginf=0.0)
            physical_centers = x0_p + clean_e[..., :2]
            point_A, point_b, point_mask, point_valid = corridor_builder(
                x0_p, clean_e)
        # Region C_i (from ellipse i) constrains the incoming segment
        # [p_{i-1}, p_i]. C_0 is unused because p_0 is the fixed start.
        A, b = point_A[:, 1:], point_b[:, 1:]
        face_mask, valid = point_mask[:, 1:], point_valid[:, 1:]
        enforce_mask = corridor_builder.segment_needs_guidance(x0_p)
        # Corridors change at every reverse level, so their dual variables must
        # not inherit pressure from geometrically different old constraints.
        step_lam = torch.zeros(B, H - 1, device=device, dtype=x0_p.dtype)
        x0_p, _, stats = alm_correct(
            x0_p, A, b, face_mask, valid, step_lam, rho,
            step_size=float(alm_cfg.get("step_size", 0.03)),
            inner_steps=int(alm_cfg.get("inner_steps", 4)),
            max_grad_norm=float(alm_cfg.get("max_grad_norm", 1.0)),
            max_correction_per_step=float(
                alm_cfg.get("max_correction_per_step", 0.10)),
            enforce_mask=enforce_mask,
            collect_stats=return_alm_stats,
        )
        if stats is not None:
            post_collision_mask = corridor_builder.segment_needs_guidance(x0_p)
            stats["physical_collision_rate_after"] = (
                post_collision_mask.float().mean())
            stats["new_physical_collision_rate"] = (
                post_collision_mask & ~enforce_mask).float().mean()
            collected_stats.append((t, {**center_stats, **stats}))

        # E stores centre offsets (c = p + delta_c). Preserve the physical
        # ellipse centres after moving P so the joint P/E state stays coherent.
        x0_e = reencode_ellipse_centers(x0_e, physical_centers, x0_p)
        return endpoints(x0_p), x0_e

    with torch.no_grad():
        if times is None:
            # ---- full per-step run: x_{t} -> x_{t-1} (unchanged default) ----
            for t in reversed(range(T)):
                tb = torch.full((B,), t, device=device, dtype=torch.long)
                ab = torch.full((B,), float(sqrt_ab[t]), device=device, dtype=torch.float32)
                out = model(p, e, map_tensor, cond, tb, ab)
                x0_p, x0_e = out["x0_p"], out["x0_e"]
                x0_p, x0_e = guide_clean_prediction(x0_p, x0_e, t)
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
                x0_p, x0_e = guide_clean_prediction(x0_p, x0_e, t)
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
    if not return_alm_stats:
        return p, e
    if not collected_stats:
        return p, e, {}
    summary = {}
    for key in collected_stats[0][1]:
        values = torch.stack([item[key] for _, item in collected_stats])
        reducer = (torch.max if key.startswith("max_") or key.endswith("_max")
                   else torch.mean)
        summary[key] = float(reducer(values).cpu())
    summary["guided_reverse_steps"] = len(collected_stats)
    summary["rho"] = rho
    summary["per_step"] = [
        {"t": t, **{key: float(value.cpu()) for key, value in stats.items()}}
        for t, stats in collected_stats
    ]
    return p, e, summary
