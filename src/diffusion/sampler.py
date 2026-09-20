"""DDIM sampler for the CONTROL-space diffusion state.

The diffusion state is the 32-control cubic B-spline polygon::

    q = torch.randn(B, 32, 2); q[:, 0] = start; q[:, -1] = goal
    for t in reverse_times:
        out = model.forward_all(q, ...)     # argmax(pi) routing
        q0  = out["control"]                # predicted clean control polygon
        q   = ddim_step(q, q0, t)
        q   = hard_control_endpoints(q, cond)
    p = model.bspline.decode_controls(q)    # only at the very end

Every reverse timestep runs the WHOLE network (including a fresh topology
choice).  There is no commit timestep and no cached selection.  The trailing
``0 -> -1`` transition is not optional: it sets q = q0 exactly.

The old waypoint-based ALM guidance is NOT migrated to control space; passing
``alm_guidance`` raises instead of silently applying wrong semantics.
"""

from __future__ import annotations

import torch

__all__ = ["sample", "pick_times"]


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
def sample(model, schedule, cond, occ, candidate_xy, candidate_mask,
           geometry, geometry_lengths, device="cuda", steps=None, seed=None,
           return_trace=False, alm_guidance=None):
    """cond [B,2,2]; occ [B,1,R,R]; candidate_xy [B,M,L,2];
    candidate_mask [B,M]; geometry [B,M,G,2]; geometry_lengths [B,M].
    """
    if alm_guidance is not None:
        raise RuntimeError(
            "B-spline control-space ALM 尚未迁移，本版本禁用旧 waypoint ALM "
            "(set alm.enabled=false / do not pass alm_guidance)")
    if seed is not None:
        torch.manual_seed(int(seed))
    model.eval()
    cond = cond.to(device)
    occ = occ.to(device)
    candidate_xy = candidate_xy.to(device)
    candidate_mask = candidate_mask.to(device)
    geometry = geometry.to(device)
    geometry_lengths = geometry_lengths.to(device)
    B, C = cond.shape[0], int(model.num_controls)
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
        # close the schedule with the clean transition 0 -> -1 so q_0 is x0(0)
        pairs.append((times[0], -1))

    q = torch.randn(B, C, 2, device=dev, dtype=torch.float32)
    q[:, 0] = start
    q[:, -1] = goal
    has_cand = candidate_mask.any(dim=1)
    trace = []
    last = None

    for t, s_t in pairs:
        tb = torch.full((B,), int(t), device=dev, dtype=torch.long)
        ab = sqrt_ab[int(t)].expand(B).contiguous()
        out = model.forward_all(q, occ, cond, tb, ab, candidate_xy,
                                candidate_mask, geometry, geometry_lengths,
                                select_index=None)
        q0 = out["control"]                      # clean control prediction
        last = out
        if int(s_t) < 0:
            q = q0
        else:
            sa_t, s1_t = sqrt_ab[int(t)], sqrt_1ma[int(t)]
            sa_s, s1_s = sqrt_ab[int(s_t)], sqrt_1ma[int(s_t)]
            eps = (q - sa_t * q0) / s1_t
            q = sa_s * q0 + s1_s * eps
        q = model.hard_control_endpoints(q, cond)
        if return_trace:
            trace.append({
                "t": int(t), "s": int(s_t) if int(s_t) >= 0 else -1,
                "q": q.detach().cpu().clone(),
                "p": model.bspline.decode_controls(q).detach().cpu().clone(),
                "q0": q0.detach().cpu().clone(),
                "final": model.bspline.decode_controls(q0).detach().cpu().clone(),
                "coarse": out["coarse"].detach().cpu().clone(),
                "selected_idx": out["selected_idx"].detach().cpu().clone(),
                "pi": out["topo"]["pi"].detach().cpu().clone(),
                "progress": out["ellipse"]["progress"].detach().cpu().clone(),
                "ellipse_center": out["ellipse"]["center"].detach().cpu().clone(),
                "ellipse_a": out["ellipse"]["a"].detach().cpu().clone(),
                "ellipse_b": out["ellipse"]["b"].detach().cpu().clone(),
                "ellipse_theta": out["ellipse"]["theta"].detach().cpu().clone(),
                "ellipse_shape4": out["ellipse"]["shape4"].detach().cpu().clone(),
            })

    p = model.bspline.decode_controls(q)
    result = {"control": q, "p": p, "final": p, "has_candidate": has_cand}
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
