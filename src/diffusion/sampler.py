"""DDIM sampler for the CONTROL-space diffusion state with guided B-spline ALM.

The diffusion state is the C-control cubic B-spline polygon (C = num_controls,
read from the config)::

    q = torch.randn(B, C, 2); q[:, 0] = start; q[:, -1] = goal
    for t in reverse_times:
        out = model.forward_all(q, ...)     # argmax(pi) or the FROZEN topology
        q0  = out["control"]                # predicted clean control polygon
        q   = ddim_step(q, q0)
        q   = hard_control_endpoints(q, cond)
    p = model.bspline.decode_controls(q)    # dense validation only

The sampler is agnostic to how the network turns the control polygon into
``out["control"]`` (control-token chain or the legacy curve-token chain).

Three-stage state machine (report sections 1-3, 9, 10, 31, 32, 35, 36):

    STATE 1  WARMUP
        plain reverse diffusion, no corridor, no ALM

    STATE 2  TRY_ACTIVATE   (from ``alm.warmup_reverse_steps`` onwards)
        one-shot topology choice -> 128 convex regions -> overlap check ->
        point-seeded gap bridge -> frozen corridor + exact B-spline constraint
        pack.  The FIRST candidate (by pi) that closes is accepted; on failure
        the sampler keeps denoising and retries for at most
        ``alm.max_activation_delay_steps`` more reverse forwards.

    STATE 3  GUIDED
        topology / corridor / bridge cells / pack are FROZEN; every reverse step
        runs the control-space ALM on THAT step's fresh ``Q0_raw`` and the DDIM
        update consumes ``Q0_safe``.  The dual is warm-started across steps.

There is NO extra final projection at ``t = 0``: the last guided step already
returns ``Q0_safe``.  The trailing ``0 -> -1`` transition sets ``q = q0_used``.

The count that drives the warm-up is the number of executed reverse network
forwards, never an absolute timestep value (DDIM sub-sampling safe).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..geometry.bspline_constraints import build_constraint_pack
from ..geometry.convex_region import EllipseRegionBuilder
from ..geometry.safety_corridor import (SCENE_TO_METER, build_safety_corridor,
                                        progress_alignment_stats)
from .bspline_alm import bspline_alm_correct

__all__ = ["sample", "pick_times", "ActivationResult", "try_activate_corridor",
           "dense_validation", "ablation_configs"]

ABLATIONS = {
    "A": "raw",
    "RAW": "raw",
    "B": "final_only",
    "FINAL_ONLY": "final_only",
    "C": "guided_bspline",
    "GUIDED": "guided_bspline",
    "GUIDED_BSPLINE": "guided_bspline",
    "D": "no_bridge",
    "NO_BRIDGE": "no_bridge",
}


def ablation_configs(alm_cfg=None, corridor_cfg=None, name="C"):
    """Ablation presets of report section 48.

        A  raw diffusion                         (ALM off)
        B  final-step-only ALM                   (control experiment)
        C  warmup + frozen corridor + per-step ALM (the actual method)
        D  C with the gap bridge disabled

    Returns ``(alm_cfg, corridor_cfg)`` ready for :func:`sample`.
    """
    key = str(name).upper()
    if key not in ABLATIONS:
        raise ValueError("unknown ablation %r (use A/B/C/D)" % name)
    mode = ABLATIONS[key]
    alm = dict(alm_cfg or {})
    corridor = dict(corridor_cfg or {})
    if mode == "raw":
        alm["enabled"] = False
        alm["mode"] = "guided_bspline"
    elif mode == "final_only":
        alm["enabled"] = True
        alm["mode"] = "final_only"
    elif mode == "guided_bspline":
        alm["enabled"] = True
        alm["mode"] = "guided_bspline"
    else:                                            # no_bridge
        alm["enabled"] = True
        alm["mode"] = "guided_bspline"
        bridge = dict(corridor.get("bridge") or {})
        bridge["enabled"] = False
        corridor["bridge"] = bridge
    return alm, corridor

_DEFAULT_REGION_KEYS = (
    "safety_margin", "obstacle_window_half", "obstacle_dilation",
    "corridor_chunk_size", "ellipse_axis_min", "ellipse_axis_max",
    "filter_eps", "guidance_dilation_cells", "guidance_occupancy_threshold",
)


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


# ---------------------------------------------------------------------------
# activation
# ---------------------------------------------------------------------------


@dataclass
class ActivationResult:
    activated: torch.Tensor          # [B] bool, newly activated this step
    topology_idx: torch.Tensor       # [B] long, the frozen index
    corridors: list                  # list[SafetyCorridor | None]
    q0_raw: torch.Tensor             # [B,C,2], reference for the ALM this step
    attempts: torch.Tensor           # [B] long
    failure_reason: list             # list[str]
    stats: dict


def _region_builder(occ_row: torch.Tensor, cfg: dict) -> EllipseRegionBuilder:
    return EllipseRegionBuilder(occ_row, cfg)


def try_activate_corridor(
    model,
    q: torch.Tensor,
    occ: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    ab: torch.Tensor,
    candidate_xy: torch.Tensor,
    candidate_mask: torch.Tensor,
    geometry: torch.Tensor,
    geometry_lengths: torch.Tensor,
    current_out: dict,
    corridor_cfg: dict,
    builder_cfg: dict,
    pending: torch.Tensor,
    tried: torch.Tensor,
    anchors: torch.Tensor,
) -> ActivationResult:
    """Try to close a safety corridor for every sample flagged in ``pending``.

    Candidates are attempted in descending ``pi`` order.  A candidate that does
    not close makes way for the next one; a candidate that closes is frozen for
    the rest of that sample (no topology change is allowed in the guided phase).
    """
    B = q.shape[0]
    device = q.device
    trials = max(1, int(corridor_cfg.get("topology_trials", 4)))
    pi = current_out["topo"]["pi"]
    score = pi.masked_fill(~candidate_mask, float("-inf"))
    order = torch.argsort(score, dim=-1, descending=True)[:, :trials]
    order = torch.clamp(order, min=0)

    activated = torch.zeros(B, dtype=torch.bool, device=device)
    topology_idx = torch.zeros(B, dtype=torch.long, device=device)
    corridors: list = [None] * B
    q0_raw = current_out["control"].clone()
    attempts = torch.zeros(B, dtype=torch.long, device=device)
    failure = ["exhausted"] * B
    stats = {"trials": trials, "candidate_trials": [],
             "region_face_counts": [], "overlap_min": [], "overlap_mean": []}

    builders: dict = {}
    active = pending.clone()
    for k in range(trials):
        if not bool(active.any()):
            break
        if k == 0:
            out_k = current_out
        else:
            # fewer candidates than trials: the extra rounds re-propose the last
            # candidate and are rejected by the ``tried`` bookkeeping below
            idx_k = order[:, min(k, order.shape[1] - 1)]
            out_k = model.forward_all(
                q, occ, cond, t, ab, candidate_xy, candidate_mask, geometry,
                geometry_lengths, select_index=idx_k)
        stats["candidate_trials"].append(int(k))
        for b in range(B):
            if not bool(active[b]):
                continue
            cand = int(order[b, min(k, order.shape[1] - 1)])
            if bool(tried[b, cand]):
                active[b] = False
                failure[b] = "no_more_candidates"
                continue
            tried[b, cand] = True
            attempts[b] += 1
            if b not in builders:
                builders[b] = _region_builder(occ[b:b + 1], builder_cfg)
            centers = out_k["ellipse"]["center"][b].detach()
            shape4 = out_k["ellipse"]["shape4"][b].detach()
            gamma = geometry[b, cand]
            glen = int(geometry_lengths[b, cand].item())
            corridor = build_safety_corridor(
                builders[b], centers, shape4, anchors, gamma=gamma,
                gamma_lengths=max(glen, 1), config=corridor_cfg)
            stats["region_face_counts"].append(int(
                sum(c.face_count for c in corridor.cells)))
            if corridor.valid:
                activated[b] = True
                active[b] = False
                topology_idx[b] = cand
                corridors[b] = corridor
                q0_raw[b] = out_k["control"][b]
                failure[b] = ""
                ov = corridor.overlap_ratio
                if ov:
                    stats["overlap_min"].append(float(min(ov)))
                    stats["overlap_mean"].append(float(sum(ov) / len(ov)))
            else:
                failure[b] = corridor.failure_reason or "corridor_invalid"
    stats["overlap_min"] = (min(stats["overlap_min"])
                            if stats["overlap_min"] else None)
    stats["overlap_mean"] = (sum(stats["overlap_mean"])
                             / len(stats["overlap_mean"])
                             if stats["overlap_mean"] else None)
    return ActivationResult(activated=activated, topology_idx=topology_idx,
                            corridors=corridors, q0_raw=q0_raw,
                            attempts=attempts, failure_reason=failure,
                            stats=stats)


# ---------------------------------------------------------------------------
# dense final validation (verification only, never projection)
# ---------------------------------------------------------------------------


def _dense_decode(codec, q: torch.Tensor, num_points: int) -> torch.Tensor:
    params = torch.linspace(0.0, 1.0, int(num_points), device=q.device)
    basis = codec.basis_at(params).to(dtype=q.dtype, device=q.device)
    return torch.einsum("pk,bkd->bpd", basis, q)


def _free_mask(occ_row: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``occ_row [1,R,R]``, ``points [N,2]`` -> ``[N]`` bool free-space."""
    grid = points.reshape(1, 1, -1, 2)
    value = F.grid_sample(occ_row[None].to(points.dtype), grid, mode="bilinear",
                          padding_mode="border", align_corners=False)
    return (value[0, 0, 0] <= 0.5) & (points.abs() <= 1.0).all(dim=-1)


def _point_constraint_table(pack, params: torch.Tensor, sample: int):
    num = int(pack.num_pieces[sample].item())
    if num == 0:
        return None
    iv = pack.intervals[sample, :num]
    idx = torch.searchsorted(iv[:, 0].contiguous(), params, right=True) - 1
    idx = idx.clamp(0, num - 1)
    return (pack.piece_A[sample, :num], pack.piece_b[sample, :num],
            pack.face_mask[sample, :num], idx)


def _point_violation(pack, points: torch.Tensor, params: torch.Tensor,
                     sample: int) -> float:
    table = _point_constraint_table(pack, params, sample)
    if table is None:
        return 0.0
    A, b, face, idx = table
    value = (A[idx] * points[:, None, :]).sum(dim=-1) - b[idx]
    value = value.masked_fill(~face[idx], -1e9)
    return float(value.max())


def _membership_rate(pack, params: torch.Tensor, points: torch.Tensor,
                     sample: int, tol: float = 1e-6):
    table = _point_constraint_table(pack, params, sample)
    if table is None:
        return None
    A, b, face, idx = table
    value = (A[idx] * points[:, None, :]).sum(dim=-1) - b[idx]
    value = value.masked_fill(~face[idx], -torch.inf)
    return float((value.max(dim=-1).values <= tol).to(torch.float32).mean())


def dense_validation(model, codec, q, pack, occ, cond, num_points=512,
                     cfg=None):
    """Verify the FINAL trajectory across a dense parameter grid."""
    cfg = dict(cfg or {})
    tol = float(cfg.get("constraint_tol", 1e-3))
    num_points = max(int(num_points), 2 * codec.curve_points)
    p = _dense_decode(codec, q, num_points)
    params = torch.linspace(0.0, 1.0, num_points, device=q.device)
    out = []
    for b in range(q.shape[0]):
        free = _free_mask(occ[b], p[b])
        metrics = {
            "final_collision": bool(not free.all()),
            "final_free_rate": float(free.to(torch.float32).mean()),
            "endpoint_error": float((p[b, 0] - cond[b, 0]).norm()
                                    + (p[b, -1] - cond[b, 1]).norm()),
            "dense_points": int(num_points),
        }
        if pack is not None and int(pack.num_pieces[b]) > 0:
            g = _point_violation(pack, p[b], params, b)
            metrics["final_max_constraint_violation"] = float(g)
            metrics["final_constraint_feasible"] = bool(g <= tol)
            metrics["final_corridor_membership_rate"] = _membership_rate(
                pack, params, p[b], b)
        else:
            metrics["final_max_constraint_violation"] = None
            metrics["final_constraint_feasible"] = None
            metrics["final_corridor_membership_rate"] = None
        out.append(metrics)
    return out


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample(model, schedule, cond, occ, candidate_xy, candidate_mask,
           geometry, geometry_lengths, device="cuda", steps=None, seed=None,
           return_trace=False, alm_guidance=None, alm_config=None,
           corridor_config=None):
    """cond [B,2,2]; occ [B,1,R,R]; candidate_xy [B,M,L,2];
    candidate_mask [B,M]; geometry [B,M,G,2]; geometry_lengths [B,M].

    ``alm_guidance`` belongs to the LEGACY waypoint ALM and is refused; the
    B-spline control-space ALM is configured through ``alm_config`` /
    ``corridor_config`` (the ``alm`` / ``corridor`` sections of the YAML).
    """
    if alm_guidance is not None:
        raise RuntimeError(
            "alm_guidance 是旧 waypoint ALM 语义，禁止套用到 32-control "
            "B-spline；请改用 alm_config/corridor_config（guided_bspline）")
    if seed is not None:
        torch.manual_seed(int(seed))
    model.eval()
    cond = cond.to(device)
    occ = occ.to(device)
    candidate_xy = candidate_xy.to(device)
    candidate_mask = candidate_mask.to(device)
    geometry = geometry.to(device)
    geometry_lengths = geometry_lengths.to(device)

    alm_cfg = dict(alm_config or {})
    alm_cfg.setdefault("scene_to_meter", SCENE_TO_METER)
    corridor_cfg = dict(corridor_config or {})
    alm_enabled = bool(alm_cfg.get("enabled", False))
    mode = str(alm_cfg.get("mode", "guided_bspline"))
    dense_verify_points = int(alm_cfg.get("dense_verify_points", 512))
    inherit_dual = bool(alm_cfg.get("inherit_dual_across_reverse_steps", True))
    builder_cfg = {k: alm_cfg[k] for k in _DEFAULT_REGION_KEYS if k in alm_cfg}
    builder_cfg.update(corridor_cfg.get("region") or {})

    B, C = cond.shape[0], int(model.num_controls)
    H = int(model.horizon)
    # the ellipse / safety branch runs on Q Skeleton queries, not on the curve
    Q = int(getattr(model, "num_safety_queries", H))
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
        pairs.append((times[0], -1))

    warmup = int(alm_cfg.get("warmup_reverse_steps", 3))
    max_delay = int(alm_cfg.get("max_activation_delay_steps", 2))
    activation_inner = int(alm_cfg.get("activation_inner_steps", 6))
    inner_steps = int(alm_cfg.get("inner_steps", 3))
    if not alm_enabled:
        warmup = 10 ** 9
    elif mode == "final_only":
        # ablation B: the corridor is only built on the LAST reverse forward
        warmup = max(0, len(pairs) - 1)
        max_delay = 0
    warmup = max(0, warmup)

    anchors = torch.linspace(0.0, 1.0, Q, device=dev)

    q = torch.randn(B, C, 2, device=dev, dtype=torch.float32)
    q[:, 0] = start
    q[:, -1] = goal
    has_cand = candidate_mask.any(dim=1)

    guided = torch.zeros(B, dtype=torch.bool, device=dev)
    frozen_idx = torch.zeros(B, dtype=torch.long, device=dev)
    tried = torch.zeros(B, candidate_mask.shape[1], dtype=torch.bool, device=dev)
    corridors: list = [None] * B
    pack = None
    lam = None
    activation_delay = torch.zeros(B, dtype=torch.long, device=dev)
    activation_step = torch.full((B,), -1, dtype=torch.long, device=dev)
    alm_status = ["disabled" if not alm_enabled else "inactive"] * B
    activation_info = {
        "warmup_reverse_steps": warmup,
        "max_activation_delay_steps": max_delay,
        "topology_fallback": [False] * B,
        "attempts": [0] * B,
        "failure_reason": [None] * B,
        "candidate_trials": [],
        "region_face_counts": [],
        "overlap_min": None, "overlap_mean": None,
    }
    exhausted = torch.zeros(B, dtype=torch.bool, device=dev)

    trace = []
    last = None
    reverse_forward_count = 0

    for t, s_t in pairs:
        tb = torch.full((B,), int(t), device=dev, dtype=torch.long)
        ab = sqrt_ab[int(t)].expand(B).contiguous()

        all_guided = bool(guided.all()) and bool(guided.any())
        sel = frozen_idx if all_guided else None
        out = model.forward_all(q, occ, cond, tb, ab, candidate_xy,
                                candidate_mask, geometry, geometry_lengths,
                                select_index=sel)
        last = out
        q0_used = out["control"]
        idx_used = out["selected_idx"].clone()
        just_activated = torch.zeros(B, dtype=torch.bool, device=dev)
        step_alm_stats = None
        step_q0_raw = out["control"].clone()

        if alm_enabled and not bool(guided.all()) \
                and reverse_forward_count >= warmup:
            pending = ~guided & ~exhausted
            if bool(pending.any()):
                top1 = out["topo"]["pi"].argmax(dim=-1)
                result = try_activate_corridor(
                    model, q, occ, cond, tb, ab, candidate_xy, candidate_mask,
                    geometry, geometry_lengths, out, corridor_cfg, builder_cfg,
                    pending, tried, anchors)
                newly = result.activated
                for b in range(B):
                    if bool(pending[b]):
                        activation_info["attempts"][b] += int(result.attempts[b])
                        activation_info["failure_reason"][b] = \
                            result.failure_reason[b] or None
                if bool(newly.any()):
                    guided = guided | newly
                    frozen_idx = torch.where(newly, result.topology_idx,
                                             frozen_idx)
                    for b in range(B):
                        if bool(newly[b]):
                            corridors[b] = result.corridors[b]
                            alm_status[b] = "guided"
                            activation_step[b] = reverse_forward_count
                            activation_info["failure_reason"][b] = None
                            activation_info["topology_fallback"][b] = bool(
                                int(result.topology_idx[b]) != int(top1[b]))
                    q0_used = torch.where(newly[:, None, None], result.q0_raw,
                                          out["control"])
                    step_q0_raw = q0_used.clone()
                    just_activated = newly
                    pack = build_constraint_pack(model.bspline, corridors,
                                                 device=dev,
                                                 dtype=torch.float32)
                    lam = None
                    activation_info["candidate_trials"] = \
                        result.stats["candidate_trials"]
                    activation_info["region_face_counts"] = \
                        result.stats["region_face_counts"]
                    activation_info["overlap_min"] = result.stats["overlap_min"]
                    activation_info["overlap_mean"] = result.stats["overlap_mean"]
                    idx_used = torch.where(newly, result.topology_idx, idx_used)
                still = pending & ~newly
                activation_delay = torch.where(still, activation_delay + 1,
                                               activation_delay)
                failed = still & (activation_delay > max_delay)
                if bool(failed.any()):
                    exhausted = exhausted | failed
                    for b in range(B):
                        if bool(failed[b]):
                            alm_status[b] = "activation_failed"

        if alm_enabled and bool(guided.any()):
            budget = torch.where(
                just_activated,
                torch.full((B,), activation_inner, device=dev, dtype=torch.long),
                torch.full((B,), inner_steps, device=dev, dtype=torch.long))
            budget = torch.where(
                guided, budget, torch.zeros(B, dtype=torch.long, device=dev))
            q0_used, lam, step_alm_stats = bspline_alm_correct(
                q0_used, pack, model.bspline,
                lam if inherit_dual else None, alm_cfg,
                inner_steps=max(activation_inner, inner_steps),
                max_steps=budget)

        if int(s_t) < 0:
            q = q0_used
        else:
            sa_t, s1_t = sqrt_ab[int(t)], sqrt_1ma[int(t)]
            sa_s, s1_s = sqrt_ab[int(s_t)], sqrt_1ma[int(s_t)]
            eps = (q - sa_t * q0_used) / s1_t
            q = sa_s * q0_used + s1_s * eps
        q = model.hard_control_endpoints(q, cond)
        reverse_forward_count += 1

        if return_trace:
            trace.append({
                "t": int(t), "s": int(s_t) if int(s_t) >= 0 else -1,
                "q": q.detach().cpu().clone(),
                "q0_raw": step_q0_raw.detach().cpu().clone(),
                "q0_safe": q0_used.detach().cpu().clone(),
                "q0": out["control"].detach().cpu().clone(),
                "p": model.bspline.decode_controls(q).detach().cpu().clone(),
                "p_raw": model.bspline.decode_controls(
                    step_q0_raw).detach().cpu().clone(),
                "p_safe": model.bspline.decode_controls(
                    q0_used).detach().cpu().clone(),
                "final": model.bspline.decode_controls(
                    q0_used).detach().cpu().clone(),
                "coarse": out["coarse"].detach().cpu().clone(),
                "selected_idx": idx_used.detach().cpu().clone(),
                "pi": out["topo"]["pi"].detach().cpu().clone(),
                "progress": out["ellipse"]["progress"].detach().cpu().clone(),
                "ellipse_center": out["ellipse"]["center"].detach().cpu().clone(),
                "ellipse_a": out["ellipse"]["a"].detach().cpu().clone(),
                "ellipse_b": out["ellipse"]["b"].detach().cpu().clone(),
                "ellipse_theta": out["ellipse"]["theta"].detach().cpu().clone(),
                "ellipse_shape4": out["ellipse"]["shape4"].detach().cpu().clone(),
                "guided": guided.detach().cpu().clone(),
                "frozen_topology_idx": frozen_idx.detach().cpu().clone(),
                "alm_active": bool(step_alm_stats is not None),
                "alm_stats": ({k: v.detach().cpu().clone()
                               for k, v in step_alm_stats.items()}
                              if step_alm_stats is not None else None),
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
    if bool(guided.any()):
        result["selected_idx"] = frozen_idx
    result["frozen_topology_idx"] = frozen_idx
    result["guided"] = guided
    result["activation_step"] = activation_step
    result["alm_status"] = alm_status
    result["activation_info"] = activation_info
    result["corridors"] = [c.to_dict() if c is not None else None
                           for c in corridors]
    result["pack_summary"] = pack.summary() if pack is not None else None
    # the frozen pack itself, for offline diagnostics / QP cross-checks
    result["pack"] = pack
    result["alm_settings"] = {
        "enabled": alm_enabled, "mode": mode, "warmup_reverse_steps": warmup,
        "activation_inner_steps": activation_inner, "inner_steps": inner_steps,
        "rho": float(alm_cfg.get("rho", 5.0)),
        "step_size": float(alm_cfg.get("step_size", 0.05)),
        "constraint_tol": float(alm_cfg.get("constraint_tol", 1e-3)),
        "max_curve_step_scene": float(
            alm_cfg.get("max_curve_step_scene", 0.01)),
        "scene_to_meter": float(alm_cfg.get("scene_to_meter", SCENE_TO_METER)),
    }
    if return_trace:
        result["trace"] = trace

    # ---- progress alignment diagnostics (V1 u <-> s correspondence) --------
    from ..models.trajsafe.geometry import gather_dense_path_points
    s_grid = torch.linspace(0.0, 1.0, H, device=dev)[None]
    # ``P_i = C(i/(H-1))`` of the FINAL curve vs ``c_i = Gamma_m(i/(H-1))``
    final_curve = model.bspline.decode_controls(q)
    alignment = []
    sel_final = (frozen_idx if bool(guided.any())
                 else (last["selected_idx"] if last is not None
                       else torch.zeros(B, dtype=torch.long, device=dev)))
    for b in range(B):
        sel_b = int(sel_final[b])
        n = max(1, int(geometry_lengths[b, sel_b].item()))
        if n >= 2:
            c = gather_dense_path_points(
                geometry[b, sel_b][None],
                torch.tensor([n], device=dev, dtype=torch.long),
                s_grid)[0].detach().cpu().numpy()
            alignment.append(progress_alignment_stats(
                final_curve[b].detach().cpu().numpy(), c))
        else:
            alignment.append(progress_alignment_stats(None, None))
    result["progress_alignment"] = alignment

    # ---- dense final validation (verification only, no projection) --------
    result["final_validation"] = dense_validation(
        model, model.bspline, q, pack, occ, cond,
        num_points=dense_verify_points, cfg=alm_cfg)
    return result
