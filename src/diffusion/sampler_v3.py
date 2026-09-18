"""V3 DDIM sampler (docs section 20/21).

Every reverse timestep runs the FULL network, including a fresh skeleton choice:

    for t in reverse_times:
        base   = encode_trajectory(P_t, map, t)
        coarse = head_p(base)                  # first, ordinary x0 prediction
        pi     = score_all_skeletons(coarse, base, candidates, t)
        m      = argmax(pi)                    # recomputed EVERY step
        s      = progress(base, selected_path, t)
        c      = gamma(selected_dense_geometry, s)
        E      = ellipse_head(c, base, geometry_memory, t)
        h_fin  = joint_fusion(base, E, scene, t)
        P0     = head_p(h_fin)                 # SAME head, real output
        P_t    = ddim_step(P_t, P0, t)

No commit_t, no committed flag, no cached selected path/features.  The last
transition (next index < 0) sets P = x0 exactly, for both the full and the
sub-sampled schedule.
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
def sample_v3(model, schedule, cond, occ, candidate_features, candidate_mask,
              candidate_lengths, geometry, geometry_lengths, device="cuda",
              steps=None, seed=None, selection="argmax", return_trace=False):
    """cond [B,2,2]; occ [B,1,R,R]; candidate_features [B,M,L,5];
    candidate_mask [B,M]; geometry [B,M,G,2]; geometry_lengths [B,M]."""
    if selection not in ("argmax", "sample"):
        raise ValueError("selection must be 'argmax' or 'sample'")
    if seed is not None:
        torch.manual_seed(int(seed))
    model.eval()
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

    p = torch.randn(B, H, 2, device=dev, dtype=torch.float32)
    p[:, 0] = start
    p[:, -1] = goal
    has_cand = candidate_mask.any(dim=1)
    trace = []
    last = None

    for t, s_t in pairs:
        tb = torch.full((B,), int(t), device=dev, dtype=torch.long)
        ab = sqrt_ab[int(t)].expand(B).contiguous()
        out = model.forward_all(p, occ, cond, tb, ab, candidate_features,
                                candidate_mask, candidate_lengths, geometry,
                                geometry_lengths, select_index=None)
        if selection == "sample":
            pi = out["topo"]["pi"]
            idx = torch.multinomial(pi.clamp_min(1e-12), 1).squeeze(-1)
            idx = torch.where(has_cand, idx, out["selected_idx"])
            ar = torch.arange(B, device=dev)
            path_feat = out["topo"]["path_feat"][ar, idx]
            geom = geometry[ar, idx]
            glen = geometry_lengths[ar, idx]
            ell = model.build_ellipses(out["base"], out["coarse"], path_feat,
                                       geom, glen, ab)
            h_final = model.fuse(out["base"], ell["tokens"])
            final = model.final_trajectory(h_final, cond)
            out["ellipse"] = ell
            out["final"] = final
            out["selected_idx"] = idx
        x0 = out["final"]
        last = out
        # The last transition must land EXACTLY on the clean prediction, for the
        # full schedule (next index -1) and for a sub-sampled one (next index 0,
        # where alpha_bar_0 is not exactly 1).  Section 21.
        if int(s_t) <= 0:
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
                "selected_idx": out["selected_idx"].detach().cpu().clone(),
                "pi": out["topo"]["pi"].detach().cpu().clone(),
                "progress": out["ellipse"]["progress"].detach().cpu().clone(),
                "ellipse_center": out["ellipse"]["center"].detach().cpu().clone(),
                "ellipse_a": out["ellipse"]["a"].detach().cpu().clone(),
                "ellipse_b": out["ellipse"]["b"].detach().cpu().clone(),
                "ellipse_theta": out["ellipse"]["theta"].detach().cpu().clone(),
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
    if return_trace:
        result["trace"] = trace
    return result
