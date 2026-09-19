"""V3 DDIM sampler (report section 24).

Only the trajectory P_t is a diffusion state.  Every reverse timestep runs the
WHOLE report network, including a fresh topology choice:

    for t in reverse_times:
        P0_hat = model(P_t, occ, cond, t, ab, candidates)   # argmax(pi) routing
        P_t    = ddim_step(P_t, P0_hat, t)

There is no commit timestep, no cached selection and no ellipse diffusion state.
The trailing ``0 -> -1`` transition is not optional: it sets P = P0_hat exactly.
"""

from __future__ import annotations

import torch

__all__ = ["sample_v3", "pick_times"]


def pick_times(T: int, steps):
    if steps is None or steps >= T:
        return None
    idx = torch.linspace(0, T - 1, steps).long().tolist()
    times = sorted(set(int(v) for v in idx))
    if T - 1 not in times:
        times.append(T - 1)
    if 0 not in times:
        times.append(0)
    return sorted(times)


@torch.no_grad()
def sample_v3(model, schedule, cond, occ, candidate_xy, candidate_mask,
              geometry, geometry_lengths, device="cuda", steps=None, seed=None,
              return_trace=False, alm_guidance=None):
    """cond [B,2,2]; occ [B,1,R,R]; candidate_xy [B,M,L,2];
    candidate_mask [B,M]; geometry [B,M,G,2]; geometry_lengths [B,M].

    ``alm_guidance`` is an optional inference-time callback
    ``fn(x0, out, p_t, t) -> (x0_corrected, info)``.  When it is omitted the
    sampler is exactly the report-faithful DDIM loop (only the trajectory is a
    diffusion state).
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    model.eval()
    # Be defensive: callers frequently pass CPU batch tensors while the model is
    # on CUDA; mismatch shows up as masked_fill(-inf) on a different device.
    cond = cond.to(device)
    occ = occ.to(device)
    candidate_xy = candidate_xy.to(device)
    candidate_mask = candidate_mask.to(device)
    geometry = geometry.to(device)
    geometry_lengths = geometry_lengths.to(device)
    B, H = cond.shape[0], model.horizon
    T = schedule.num_timesteps
    dev = device
    start, goal = cond[:, 0], cond[:, 1]
    sqrt_ab = schedule.sqrt_alphas_cumprod.detach().to(dev).float()
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod.detach().to(dev).float()
    times = pick_times(T, steps)
    if times is None:
        pairs = [(t, t - 1) for t in range(T - 1, -1, -1)]
    else:
        pairs = [(times[i], times[i - 1]) for i in range(len(times) - 1, 0, -1)]
        # close the schedule with the clean transition 0 -> -1 so P_0 is x0(0)
        pairs.append((times[0], -1))

    p = torch.randn(B, H, 2, device=dev, dtype=torch.float32)
    p[:, 0] = start
    p[:, -1] = goal
    has_cand = candidate_mask.any(dim=1)
    trace = []
    last = None

    for t, s_t in pairs:
        tb = torch.full((B,), int(t), device=dev, dtype=torch.long)
        ab = sqrt_ab[int(t)].expand(B).contiguous()
        out = model.forward_all(p, occ, cond, tb, ab, candidate_xy,
                                candidate_mask, geometry, geometry_lengths,
                                select_index=None)
        x0 = out["final"]
        x0_raw = x0
        guide_info = None
        if alm_guidance is not None:
            x0, guide_info = alm_guidance(x0, out, p, int(t))
        last = out
        if int(s_t) < 0:
            p = x0
        else:
            sa_t, s1_t = sqrt_ab[int(t)], sqrt_1ma[int(t)]
            sa_s, s1_s = sqrt_ab[int(s_t)], sqrt_1ma[int(s_t)]
            eps = (p - sa_t * x0) / s1_t
            p = sa_s * x0 + s1_s * eps
        p[:, 0] = start
        p[:, -1] = goal
        if return_trace:
            trace.append({
                "t": int(t), "s": int(s_t) if int(s_t) >= 0 else -1,
                "p": p.detach().cpu().clone(),
                "coarse": out["coarse"].detach().cpu().clone(),
                "final": x0.detach().cpu().clone(),
                "final_raw": x0_raw.detach().cpu().clone(),
                "guide": guide_info,
                "selected_idx": out["selected_idx"].detach().cpu().clone(),
                "pi": out["topo"]["pi"].detach().cpu().clone(),
                "progress": out["ellipse"]["progress"].detach().cpu().clone(),
                "ellipse_center": out["ellipse"]["center"].detach().cpu().clone(),
                "ellipse_a": out["ellipse"]["a"].detach().cpu().clone(),
                "ellipse_b": out["ellipse"]["b"].detach().cpu().clone(),
                "ellipse_theta": out["ellipse"]["theta"].detach().cpu().clone(),
                "ellipse_shape4": out["ellipse"]["shape4"].detach().cpu().clone(),
            })

    result = {"p": p, "has_candidate": has_cand}
    if last is not None:
        result["selected_idx"] = last["selected_idx"]
        result["topology_pi"] = last["topo"]["pi"]
        result["progress"] = last["ellipse"]["progress"]
        result["ellipse_center"] = last["ellipse"]["center"]
        result["ellipse_a"] = last["ellipse"]["a"]
        result["ellipse_b"] = last["ellipse"]["b"]
        result["ellipse_theta"] = last["ellipse"]["theta"]
        result["ellipse_shape4"] = last["ellipse"]["shape4"]
    if return_trace:
        result["trace"] = trace
    return result
