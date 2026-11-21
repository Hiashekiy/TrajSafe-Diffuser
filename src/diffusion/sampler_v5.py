"""Four-step inference sampler for the new single-path framework (v5).

    step 1  plain denoise: no path, no safety stream, no feedback
            -> x0_1 -> 128-point curve
    (once)  Field-guided search on the INFLATED safe grid, guided by x0_1
            -> initial_search_path; active_path = it (arc-length resampled 128)
    step 2  active_path + configured safety ellipses, NO feedback -> x0_2
    step 3  latest verified active_path (+ feedback from step 2 if valid)
    step 4  latest verified active_path (+ latest valid feedback)

After every step >= 2: decode -> collision check on the network occupancy -> (only if a
collision exists) build the corridor from the predicted ellipses -> ALM ->
GLOBAL validation -> update ``active_path`` and the feedback only when it passes.
No AMENDMENT of the noise schedule: training keeps T=16, inference executes
``steps`` (default 4) DDIM forwards over a subsampled schedule.

Search fields use only the footprint-inflated occupancy. Network conditioning
and the existing downstream ALM map retain the original occupancy.
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
        # Treat the outside as occupied before applying the footprint radius.
        from scipy.ndimage import distance_transform_edt
        boundary_distance = distance_transform_edt(
            np.pad(safe.raw_free, 1, constant_values=False))[1:-1, 1:-1]
        safe.free &= boundary_distance > safe.inflation_cells
        out.append((safe, None))
    return out, spec, (time.perf_counter() - t0) * 1000.0


def _search_once(grids, spec, cond, x0_curve, cfg, b):
    """Field-guided search on the footprint-inflated map; network map is unchanged."""
    from ..grid_search.field_search import search_path
    safe, _ = grids[b]
    return search_path(
        safe, spec, cond[b, 0].detach().cpu().numpy(),
        cond[b, 1].detach().cpu().numpy(),
        x0_curve[b].detach().cpu().numpy(), cfg.get("grid_search") or {})


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
