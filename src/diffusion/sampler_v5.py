"""Four-step inference sampler for the new single-path framework (v5).

    step 1  plain denoise: no path, no safety stream, no feedback
            -> x0_1 -> 128-point curve
    (once)  PDG-Focal A* on the INFLATED safe grid, guided by x0_1
            -> initial_search_path; active_path = it (arc-length resampled 128)
    step 2  active_path + 64 ellipses, NO feedback -> x0_2
    step 3  latest verified active_path (+ feedback from step 2 if valid)
    step 4  latest verified active_path (+ latest valid feedback)

After every step >= 2: decode -> collision check on the safe grid -> (only if a
collision exists) build the corridor from the predicted ellipses -> ALM ->
GLOBAL validation -> update ``active_path`` and the feedback only when it passes.
No AMENDMENT of the noise schedule: training keeps T=16, inference executes
``steps`` (default 4) DDIM forwards over a subsampled schedule.

Scope note for tonight's run: the corridor is still built over the WHOLE ellipse
set (the "local collision interval" version needs ``build_local_constraint_pack``,
which is tracked as the remaining S4 item); the collision-interval detection and
its diagnostics are already computed here.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .alm_local import (alm_closed_loop, collision_intervals,
                        collision_mask)
from .sampler import pick_times, effective_warmup_steps

__all__ = ["sample_v5", "ddim_step_v5"]

_SCENE_TO_METER = 80.0


def ddim_step_v5(q, x0, t, s, sched, cond):
    if s < 0:
        q = x0
    else:
        sa_t = sched.sqrt_alphas_cumprod[t].view(-1, 1, 1).to(q.dtype)
        s1_t = sched.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1).to(q.dtype)
        sa_s = sched.sqrt_alphas_cumprod[s].view(-1, 1, 1).to(q.dtype)
        s1_s = sched.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1).to(q.dtype)
        eps = (q - sa_t * x0) / s1_t.clamp_min(1e-8)
        q = sa_s * x0 + s1_s * eps
    q = q.clone()
    q[:, 0] = cond[:, 0]
    q[:, -1] = cond[:, 1]
    return q


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dense_resample(p, n):
    """Arc-length resample of a polyline to ``n`` points (numpy)."""
    if torch.is_tensor(p):
        p = p.detach().cpu().numpy()
    p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-12:
        return np.repeat(p[:1], n, axis=0)
    s = np.linspace(0.0, cum[-1], int(n))
    return np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], -1)


def _curve_collision_frac(curves, occ, n_dense=None):
    """Per-sample collision fraction of a [B,K,2] curve batch (numpy).

    The curve is checked on the SAME dense grid the ALM verification uses
    (``alm.dense_verify_points``, default 512): checking only the 128 decoded
    points lets a curve graze an obstacle corner *between* two samples.
    """
    if torch.is_tensor(curves):
        curves = curves.detach().cpu().numpy()
    curves = np.asarray(curves)
    B = int(curves.shape[0])
    out = []
    n = int(n_dense) if n_dense else None
    for b in range(B):
        c = curves[b] if n is None else _dense_resample(curves[b], n)
        hit, _ = collision_mask(occ[b], c)
        out.append(float(hit.mean()))
    return np.asarray(out)


def _build_safe_grids(occ: torch.Tensor, cfg: dict):
    """One SafeGrid + neighbour table per sample (project occupancy semantics)."""
    from ..grid_search.grid_map import GridSpec, build_safe_grid
    from ..grid_search.grid_neighbors import build_neighbor_table
    gcfg = dict(cfg.get("grid_search") or {})
    s2m = float(gcfg.get("scene_to_meter", _SCENE_TO_METER))
    occ_np = occ.detach().cpu().numpy()
    B, _, H, W = occ_np.shape
    spec = GridSpec(H, W, 2.0 * s2m / max(H, W), s2m)
    out = []
    t0 = time.perf_counter()
    for b in range(B):
        safe = build_safe_grid(
            occ_np[b, 0].astype(np.uint8), spec,
            robot_radius_m=float(gcfg.get("robot_radius_m", 0.8)),
            safety_margin_m=float(gcfg.get("safety_margin_m", 0.3)),
            already_inflated_cells=float(gcfg.get("already_inflated_cells", 0.0)))
        nbr = build_neighbor_table(safe.free, spec.cell_m)
        out.append((safe, nbr))
    return out, spec, (time.perf_counter() - t0) * 1000.0


def _search_once(grids, spec, cond, x0_curve, cfg, b):
    """PDG-Focal A* for sample ``b``; returns (path_scene[128,2], stats dict)."""
    from ..grid_search import diffusion_path, progress_guidance
    from ..grid_search.baselines import astar, focal_astar, pdg_focal_astar
    gcfg = dict(cfg.get("grid_search") or {})
    safe, nbr = grids[b]
    start = cond[b, 0].detach().cpu().numpy().astype(np.float64)
    goal = cond[b, 1].detach().cpu().numpy().astype(np.float64)
    curve = x0_curve[b].detach().cpu().numpy().astype(np.float64)
    stats = {"guide_valid": False, "fallback": None, "expanded": 0,
             "search_ms": 0.0, "guide_ms": 0.0, "status": "NA"}
    t0 = time.perf_counter()
    try:
        ref = diffusion_path.prepare_reference_path(curve, spec, start_world=start,
                                     goal_world=goal)
        if bool(gcfg.get("lazy_guidance", True)):
            # compute H_D only for the nodes the search touches (28 ms -> 1.6 ms)
            from ..grid_search.lazy_guidance import LazyGuidance
            field = LazyGuidance(ref, spec, J=int(gcfg.get("J", 3)))
        else:
            field = progress_guidance.build_progress_guidance(
                ref, spec, J=int(gcfg.get("J", 3)))
        s_idx = int(spec.cell_to_index(spec.world_to_cell(start)))
        g_idx = int(spec.cell_to_index(spec.world_to_cell(goal)))
        res = None
        if field.valid:
            res = pdg_focal_astar(safe, s_idx, g_idx, field,
                                  w=float(gcfg.get("focal_w", 1.2)),
                                  alpha_p=float(gcfg.get("alpha_p", 1.0)),
                                  neighbors=nbr)
            stats["guide_valid"] = True
        if res is None or res.status != "SUCCESS":
            res = focal_astar(safe, s_idx, g_idx, w=1.2, secondary="clearance",
                              neighbors=nbr)
            stats["fallback"] = "focal_clearance"
            if res.status != "SUCCESS":
                res = astar(safe, s_idx, g_idx, nbr)
                stats["fallback"] = "astar"
        stats["status"] = res.status
        stats["expanded"] = res.stats.get("expanded_nodes", 0)
        if res.status == "SUCCESS":
            rc = spec.index_to_rc(res.path)
            pts = spec.cell_to_world(rc.astype(np.float64)[:, ::-1])
            pts[0] = start
            pts[-1] = goal
            # ONE arc-length resample of the raw grid path to exactly 128 points
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            if cum[-1] <= 1e-9:
                raise ValueError("degenerate search path")
            s = np.linspace(0.0, cum[-1], 128)
            pts = np.stack([np.interp(s, cum, pts[:, k]) for k in range(2)], -1)
            pts[0], pts[-1] = start, goal
            # ---- push the grid path towards the middle of the free channel ---
            # A* hugs obstacle corners; the ellipses are centred on this path, so
            # a wall-hugging line makes the local regions degenerate / stop
            # overlapping (invalid_base_region / bridge_gamma_failed).  Centering
            # is a clearance-gradient ascent with pinned endpoints.
            cen_cfg = dict(((cfg.get("grid_search") or {}).get("center_path")
                            or {}))
            if bool(cen_cfg.get("enabled", True)):
                try:
                    from ..grid_search.path_centering import center_path
                    g_b = grids[b] if isinstance(grids, (list, tuple)) else grids
                    if isinstance(g_b, (list, tuple)):
                        g_b = g_b[0]          # (SafeGrid, neighbour_table)
                    raw_free = getattr(g_b, "raw_free", None)
                    free_b = (np.asarray(raw_free, dtype=bool) if raw_free
                              is not None else np.asarray(g_b.free, dtype=bool))
                    pts, cen_st = center_path(
                        pts, free_b, float(spec.cell_m),
                        iters=int(cen_cfg.get("iters", 40)),
                        step_cells=float(cen_cfg.get("step_cells", 1.0)),
                        smooth=float(cen_cfg.get("smooth", 0.25)))
                    stats["clearance_before"] = cen_st["clearance_before"]
                    stats["clearance_after"] = cen_st["clearance_after"]
                    pts[0], pts[-1] = start, goal
                except Exception as exc:
                    stats["center_error"] = repr(exc)[:80]
            stats["path_len_m"] = float(cum[-1] * spec.scene_to_meter)
            stats["search_ms"] = (time.perf_counter() - t0) * 1000.0
            return pts.astype(np.float32), stats
    except Exception as exc:                      # never let search kill sampling
        import traceback as _tb
        stats["search_tb"] = _tb.format_exc()[-700:]
        stats["search_error"] = repr(exc)[:160]
    stats["search_ms"] = (time.perf_counter() - t0) * 1000.0
    u = np.linspace(0.0, 1.0, 128)[:, None]
    straight = (start[None] * (1 - u) + goal[None] * u).astype(np.float32)
    stats["fallback"] = stats["fallback"] or "straight"
    return straight, stats


@torch.no_grad()
def sample_v5(model, sched, batch, cfg, device="cuda", steps=None, seed=None,
              use_heading=True, use_feedback=True):
    """Run the four-step flow; returns the final curve + a per-step trace.

    ``use_heading`` / ``use_feedback`` exist for the S7 ablations (they only
    remove a conditioning stream, nothing else changes).
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    model.eval()
    cond = batch["cond"].to(device)
    occ = batch["occupancy"].to(device)
    heading = batch.get("heading")
    hv = batch.get("heading_valid")
    heading = heading.to(device) if (heading is not None and use_heading) else None
    hv = hv.to(device) if (hv is not None and use_heading) else None
    if not use_feedback:
        fb_disable = True
    else:
        fb_disable = False
    B, C = cond.shape[0], int(model.num_controls)
    T = int(sched.num_timesteps)
    steps = int(steps if steps is not None
                else (cfg["diffusion"].get("inference_steps", 4)))
    times = pick_times(T, steps)
    if times is None:                     # steps >= T: the FULL schedule
        times = list(range(T))
    pairs = [(times[i], times[i - 1]) for i in range(len(times) - 1, 0, -1)]
    pairs.append((times[0], -1))

    grids, spec, grid_ms = _build_safe_grids(occ, cfg)
    q = torch.randn(B, C, 2, device=device)
    q[:, 0] = cond[:, 0]
    q[:, -1] = cond[:, 1]

    active_path = None
    fb_delta = torch.zeros(B, C, 2, device=device)
    fb_valid = torch.zeros(B, dtype=torch.bool, device=device)
    trace, search_stats = [], []
    t_start = time.perf_counter()

    q0_safe = None
    lam_prev = None
    # last trajectory that actually passed the dense verification (curves, not
    # just "the model was asked to do something"): the only thing we may fall
    # back to without losing the safety guarantee.
    last_verified = None            # [B,K,2] curve that passed the dense check
    verified_ok = torch.zeros(B, dtype=torch.bool, device=device)
    for i, (t, s) in enumerate(pairs):
        tb = torch.full((B,), int(t), device=device, dtype=torch.long)
        ab = sched.sqrt_alphas_cumprod[int(t)].expand(B).to(device)
        t_net0 = time.perf_counter()
        out = model(q, occ, cond, tb, ab, active_path=active_path,
                    feedback_delta=(None if fb_disable else
                                    (fb_delta if i >= 2 else None)),
                    feedback_valid=(None if fb_disable else
                                    (fb_valid if i >= 2 else None)),
                    heading=heading, heading_valid=hv)
        net_ms = (time.perf_counter() - t_net0) * 1000.0
        x0 = out["control"]
        curve = out["final"]

        step = {"step": i + 1, "t": int(t), "net_ms": net_ms,
                "has_path": bool(active_path is not None),
                "has_safety": bool(out["has_safety"][0].item()),
                "feedback_valid": bool(fb_valid.any().item()),
                "raw_curve": curve.detach().cpu().numpy(),
                "ellipse": {k: v.detach().cpu().numpy()
                            for k, v in (out["ellipse"] or {}).items()
                            if torch.is_tensor(v) and v.ndim >= 2},
                "safe_curve": None,
                "collision": [], "alm": None, "search": None}

        if i == 0:
            # ---- ONE global search, guided by this x0 ------------------------
            paths, stats = [], []
            for b in range(B):
                p, st = _search_once(grids, spec, cond,
                                     curve.detach(), cfg, b)
                paths.append(torch.from_numpy(p).to(device))
                stats.append(st)
            active_path = torch.stack(paths).to(device)
            search_stats = stats
            step["search"] = stats
        else:
            # §5.2 / §6.1: NO collision -> NO corridor, NO ALM, NO feedback
            any_coll = False
            for b in range(B):
                c = curve[b].detach().cpu().numpy()
                hit_b, _ = collision_mask(occ[b], c)
                if hit_b.any():
                    any_coll = True
                    break
            if not any_coll:
                alm_info = {"alm": False, "alm_ok": False,
                            "reason": "no_collision"}
                q0_safe = x0
                fb_valid = torch.zeros(B, dtype=torch.bool, device=device)
            else:
                # LEGACY pipeline (corridor + bridge + pack + ALM + dense
                # validation) reused as-is; the searched path is the bridge
                # skeleton and the dual is warm-started across reverse steps.
                q0_safe, lam_prev, alm_info = alm_closed_loop(
                    model, occ, cond, x0, out["ellipse"], cfg,
                    gamma=active_path, lam=lam_prev)
            step["alm"] = alm_info
            step["safe_curve"] = (model.bspline.decode_controls(q0_safe)
                                  .detach().cpu().numpy()
                                  if alm_info.get("alm") else None)
            # per-sample: only the samples whose correction passed the dense
            # verification may update the path / feed back / be kept as output
            per = alm_info.get("per_sample") or []
            ok_b = torch.tensor([bool(q.get("ok")) for q in per], device=device,
                                dtype=torch.bool) if per else torch.zeros(
                                    B, dtype=torch.bool, device=device)
            if bool(ok_b.any()):
                safe_all = model.bspline.decode_controls(q0_safe)
                if last_verified is None:
                    last_verified = safe_all.clone()
                else:
                    last_verified = torch.where(ok_b[:, None, None], safe_all,
                                                last_verified)
                verified_ok = verified_ok | ok_b
                if active_path is not None:
                    active_path = torch.where(ok_b[:, None, None], safe_all,
                                              active_path)
                fb_delta = (q0_safe - x0).detach()
                fb_valid = ok_b
            else:
                fb_valid = torch.zeros(B, dtype=torch.bool, device=device)
        # collision diagnostics on the raw prediction
        for b in range(B):
            c = curve[b].detach().cpu().numpy()
            hit, oob = collision_mask(occ[b], c)
            step["collision"].append({
                "frac": float(hit.mean()), "oob": float(oob.mean()),
                "intervals": collision_intervals(hit,
                                                  n_query=int(model.num_safety_queries))})
        trace.append(step)
        # legacy parity (sampler.py:802): the ALM-corrected polygon enters the
        # DDIM transition; only fall back to the raw x0 when ALM did not run.
        q0_ddim = (q0_safe if (i > 0 and q0_safe is not None
                               and step["alm"].get("alm_ok")) else x0)
        step["q0_ddim_is_safe"] = bool(q0_ddim is not x0)
        q = ddim_step_v5(q, q0_ddim, tb, s, sched, cond)

    total_ms = (time.perf_counter() - t_start) * 1000.0
    final_curve = model.bspline.decode_controls(q)
    # ---- final output selection -------------------------------------------
    # 1) the last densely-verified curve, when one exists (zero collisions on the
    #    same checker the verification used);
    # 2) otherwise the final prediction, but if that one still collides and a
    #    verified curve exists we already took it above;
    # 3) the centred search path is only used when nothing was ever verified AND
    #    the prediction collides (it is collision-free by construction).
    _dense_n = int((cfg.get("alm") or {}).get("dense_verify_points", 512))
    curve_coll = _curve_collision_frac(final_curve, occ, _dense_n)
    src = ["model"] * B
    if active_path is not None:
        ap = active_path
    else:
        ap = final_curve
    out_curves = final_curve.clone()
    for b in range(B):
        # keep the model output whenever it is already collision-free; otherwise
        # substitute the last densely-verified curve, and only as a last resort
        # the (collision-free by construction) centred search path.
        if float(curve_coll[b]) <= 0.0:
            continue
        if bool(verified_ok[b]) and last_verified is not None:
            out_curves[b] = last_verified[b]
            src[b] = "last_verified"
        else:
            out_curves[b] = ap[b]
            src[b] = "search_path"
    final_curve = out_curves
    final_source = src
    final_coll = _curve_collision_frac(final_curve, occ, _dense_n)
    return {"final": final_curve, "control": q, "trace": trace,
            "final_source": final_source,
            "final_collision_frac": [float(x) for x in final_coll],
            "search": search_stats, "grid_ms": grid_ms,
            "total_ms": total_ms, "steps": steps,
            "initial_path": None if not trace[0].get("search") else
            [s.get("status") for s in trace[0]["search"]]}
