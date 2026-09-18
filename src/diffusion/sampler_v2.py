"""DDIM reverse sampler for V2 (Skeleton-Topology-Grounded Trajectory Diffusion).

The diffusion state is ONLY the trajectory:

    p = torch.randn(B, H, 2)        # there is no ellipse state any more

Reverse pass (docs/V2.md sections 12/17/18/33/34):

    t = 15 ... 8    trajectory only
    first t <= commit_t:
                    base = encode_trajectory(p, ...)
                    logits = score_candidates(base, ...)
                    m ~ Cat(pi) or argmax(pi)      -> COMMITTED ONCE
    t = 7 ... 0     refine_with_path(base, P_m):
                    progress s_i^t = f_prog(H_tau^t, P_m)
                    c_i^t = gamma_m(s_i^t)
                    E_i^t = shape head at c_i^t
                    x0_p = refined trajectory
    p <- DDIM(p, x0_p)              (trajectory only)

The selected topology is never re-sampled and candidate coordinates are never
averaged: the softmax only produces a probability, the geometry always commits
to one real candidate.

The commit rule is "the first selected timestep with t <= commit_t", not
"t == commit_t", so a sub-sampled run (--steps S) that skips level 7 still
commits.
"""

from __future__ import annotations

import torch

__all__ = ["sample_v2", "pick_times"]


def pick_times(T: int, steps):
    """Timesteps actually visited by a sub-sampled run (None = every level)."""
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
def sample_v2(model, schedule, cond, map_tensor, candidate_paths,
              candidate_mask, candidate_lengths=None, device="cuda",
              steps=None, seed=None, commit_t=7, selection="sample",
              return_trace=False):
    """cond [B,2,2]; map_tensor [B,1,256,256]; candidate_paths [B,M,L,5];
    candidate_mask [B,M]; candidate_lengths [B,M]."""
    if selection not in ("sample", "argmax"):
        raise ValueError("selection must be 'sample' or 'argmax', got %r" % selection)
    if seed is not None:
        torch.manual_seed(int(seed))
    model.eval()

    B, H = cond.shape[0], model.horizon
    T = schedule.num_timesteps
    M = candidate_paths.shape[1]
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

    if candidate_lengths is None:
        candidate_lengths = torch.zeros(B, M, device=dev, dtype=torch.float32)
    has_cand = candidate_mask.any(dim=1)                 # [B]

    committed = False
    selected_path = None
    selected_feat = None
    selected_idx = torch.zeros(B, dtype=torch.long, device=dev)
    committed_at = torch.full((B,), -1, dtype=torch.long, device=dev)
    pi_commit = torch.zeros(B, M, device=dev, dtype=torch.float32)
    last_refine = None
    trace = []

    def endpoints(x):
        x = x.clone()
        x[:, 0] = start
        x[:, -1] = goal
        return x

    for t, s in pairs:
        tb = torch.full((B,), int(t), device=dev, dtype=torch.long)
        ab = sqrt_ab[int(t)].expand(B).contiguous()
        base = model.encode_trajectory(p, map_tensor, cond, tb, ab)

        if (not committed) and int(t) <= int(commit_t):
            topo = model.score_candidates(base, candidate_paths, candidate_mask,
                                          candidate_lengths)
            pi = topo["pi"]
            if selection == "argmax":
                idx = pi.argmax(dim=-1)
            else:
                idx = torch.multinomial(pi.clamp_min(1e-12), 1).squeeze(-1)
            idx = torch.where(has_cand, idx, torch.zeros_like(idx))
            ar = torch.arange(B, device=dev)
            selected_idx = idx
            pi_commit = pi
            # per-row: rows without any candidate are never committed
            committed_at = torch.where(has_cand,
                                       torch.full_like(committed_at, int(t)),
                                       committed_at)
            selected_path = candidate_paths[ar, idx][..., :2].contiguous()
            selected_feat = topo["path_feat"][ar, idx].contiguous()
            committed = True

        if selected_path is None:
            x0_p = base["x0_p_base"]
        else:
            ref = model.refine_with_path(base, selected_path, selected_feat)
            last_refine = ref
            x0_p = torch.where(has_cand[:, None, None], ref["x0_p"],
                               base["x0_p_base"])
        x0_p = endpoints(x0_p)

        if int(s) < 0:
            p = x0_p
        else:
            sa_t, s1_t = sqrt_ab[int(t)], sqrt_1ma[int(t)]
            sa_s, s1_s = sqrt_ab[int(s)], sqrt_1ma[int(s)]
            eps = (p - sa_t * x0_p) / s1_t
            p = sa_s * x0_p + s1_s * eps
        p = endpoints(p)

        if return_trace:
            # Full per-step record for the dashboard replay.  "p" is the noisy
            # state at the START of this step (exactly what the model consumed),
            # and the ellipse fields are None before the topology is committed.
            ref = last_refine if selected_path is not None else None
            trace.append({
                "t": int(t),
                "s": int(s) if int(s) >= 0 else -1,
                "p": p.detach().cpu().clone(),
                "x0_p": x0_p.detach().cpu().clone(),
                "x0_p_base": base["x0_p_base"].detach().cpu().clone(),
                "committed": bool(committed),
                "selected_idx": selected_idx.detach().cpu().clone(),
                "pi": pi_commit.detach().cpu().clone(),
                "progress": (ref["progress"].detach().cpu().clone()
                             if ref is not None else None),
                "ellipse_center": (ref["ellipse_center"].detach().cpu().clone()
                                   if ref is not None else None),
                "ellipse_shape4": (ref["ellipse_shape4"].detach().cpu().clone()
                                   if ref is not None else None),
            })

    out = {
        "p": p,
        "selected_idx": selected_idx,
        "committed_at": committed_at,
        "topology_pi": pi_commit,
        "has_candidate": has_cand,
    }
    if last_refine is not None:
        out["progress"] = last_refine["progress"]
        out["ellipse_center"] = last_refine["ellipse_center"]
        out["ellipse_shape4"] = last_refine["ellipse_shape4"]
    else:                                   # pragma: no cover - defensive
        out["progress"] = torch.zeros(B, H, device=dev)
        out["ellipse_center"] = torch.zeros(B, H, 2, device=dev)
        out["ellipse_shape4"] = torch.zeros(B, H, 4, device=dev)
    if return_trace:
        out["trace"] = trace
    return out
