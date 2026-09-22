"""Train the control-space TrajSafe-Diffuser on the CARLA v1 snapshot.

    L = lam_ctrl   L_ctrl     (raw final controls   vs control_gt)
      + lam_coarse L_coarse   (raw coarse controls  vs control_gt)
      + lam_smooth L_smooth   (2nd/3rd differences of the CONTROL polygon)
      + lam_bound  L_boundary (raw controls near start/goal vs GT local shape)
      + lam_topo   L_topo
      + lam_shape  L_shape
      + lam_iou    L_iou
      + lam_safe   L_safe
      + lam_alm    L_alm      (decoded curve vs the offline ALM corridor)
      + lam_fbsafe L_fbsafe   (CONTINUOUS Bezier corridor violation of the
                               SECOND (feedback-conditioned) raw prediction)
      + lam_curve  L_curve    (2nd/3rd differences of the DECODED curve)

Historical safety feedback (``model.feedback.enabled`` + ``train.feedback``)
adds a real two-step rollout per batch, exactly mirroring inference::

    q_t --network--> Q0_raw(t) --ALM--> Q0_safe(t) --DDIM--> q_s
        --feedback (Q0_safe, Delta, valid)--> network --> Q0_raw(s) --> LOSS

The first stage runs under ``no_grad`` and the loss is computed on the SECOND
network's OWN raw output: the network can never learn to let the ALM clean up
after it.  There is deliberately NO distillation term ``Q_next ~ Q_safe_prev``.

The diffusion state is the C-control polygon Q_t (``control_gt``, C from the
config).  The network runs on the control tokens themselves; NO decoded
trajectory point ever enters the network or the loss (there is no L_traj), and
the 128-point curve only exists at the very end for plotting/metrics.  There is
NO alignment loss, no learned progress and no ellipse-centre label: the ellipse
centre is the fixed Skeleton centre Gamma_m(i/(Q-1)).

Training routing uses m* = argmin_m nDTW(curve_gt, S_m) (cached
``topology_best``); inference routing uses argmax(pi).

Usage:
  python train.py --config configs/config.yaml
  python train.py --config configs/config.yaml --max-batches 3
  python train.py --config configs/config.yaml --overfit 32 --epochs 300
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config, num_controls
from src.utils.seed import set_seed
from src.utils.checkpoint import (ARCH_CONTROL_SPACE, ARCH_LEGACY_CURVE,
                                  detect_architecture, load_checkpoint,
                                  save_checkpoint)
from src.diffusion.bspline_alm import bspline_alm_correct
from src.diffusion.schedule import NoiseSchedule
from src.geometry.bspline_constraints import build_constraint_pack_from_regions
from src.models.trajsafe import TrajSafePlanner
from src.datasets.carla_spline_dataset import make_loader, make_collate
from src.geometry.ellipse_raster import ellipse_soft_mask
from src.geometry.ellipse_shape import shape4_to_abtheta
from src.losses.losses import (alm_corridor_loss, boundary_control_loss,
                               control_smoothness_loss, control_x0_loss,
                               curve_smoothness_loss, ellipse_iou_loss,
                               ellipse_safety_loss, ellipse_shape_loss,
                               feedback_safety_loss, pack_violation,
                               topology_ce)
from src.models.trajsafe.boundary import boundary_targets

LOSS_KEYS = ["Lctrl", "Lcoarse", "Lsmooth", "Lboundary", "Ltopo", "Lshape",
             "Liou", "Lsafe", "Lalm", "Lfbsafe", "Lcurve"]
LOSS_WEIGHT_KEYS = {
    "Lctrl": "lambda_control", "Lcoarse": "lambda_coarse",
    "Lsmooth": "lambda_smooth", "Lboundary": "lambda_boundary",
    "Ltopo": "lambda_topology", "Lshape": "lambda_shape",
    "Liou": "lambda_iou", "Lsafe": "lambda_safe", "Lalm": "lambda_alm",
    "Lfbsafe": "lambda_feedback_safe", "Lcurve": "lambda_curve_smooth",
}
DEFAULT_LOSS_WEIGHTS = {
    "lambda_control": 0.2, "lambda_coarse": 0.5, "lambda_smooth": 0.08,
    "lambda_boundary": 0.2, "lambda_topology": 0.25, "lambda_shape": 0.08,
    "lambda_iou": 0.25, "lambda_safe": 0.15, "lambda_alm": 0.3,
    "lambda_feedback_safe": 0.3, "lambda_curve_smooth": 0.1,
}
# Metres per scene unit.  40.0 for the 80 m carla_v1 crop, 80.0 for the 160 m
# carla_full_160_256 windows.  This is DATA, so it is read from
# data.scene_to_meter in main(); the literal below only covers configs written
# before that key existed.  Everything else in the repository works in SCENE
# units, so this constant affects REPORTING only.
SCENE_TO_METER = 40.0


def set_scene_to_meter(cfg, default: float = 40.0) -> float:
    """Read data.scene_to_meter (metres per scene unit) into the global."""
    global SCENE_TO_METER
    try:
        SCENE_TO_METER = float((cfg.get("data") or {}).get(
            "scene_to_meter", default))
    except (TypeError, ValueError):
        SCENE_TO_METER = float(default)
    return SCENE_TO_METER


def _broadcast(x0, v):
    return v.reshape(v.shape[0], *([1] * (x0.dim() - 1)))


def add_noise(x0, t, schedule):
    ab = schedule.sqrt_alphas_cumprod[t].to(x0.device).float()
    s1 = schedule.sqrt_one_minus_alphas_cumprod[t].to(x0.device).float()
    eps = torch.randn_like(x0)
    return _broadcast(x0, ab) * x0 + _broadcast(x0, s1) * eps, eps


def _point_collision(points: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
    """points [B,H,2], occ [B,1,R,R] (1 = obstacle) -> [B,H] bool collision."""
    R = occ.shape[-1]
    cell = 2.0 / float(R)
    ix = torch.floor((points[..., 0] + 1.0) / cell)
    iy = torch.floor((points[..., 1] + 1.0) / cell)
    inside = ((ix >= 0) & (ix < R) & (iy >= 0) & (iy < R))
    ix = ix.clamp(0, R - 1).long()
    iy = iy.clamp(0, R - 1).long()
    b = torch.arange(points.shape[0], device=points.device)[:, None]
    hit = occ[b, 0, iy, ix] > 0.5
    return hit | (~inside)


def metrics(batch, out, occ, device):
    """Evaluation metrics that are NOT just the training losses."""
    p_gt = batch["pos"].to(device)
    q_gt = batch["control_gt"].to(device)
    cond = batch["cond"].to(device)
    best = batch["topology_best"].to(device)
    has_cand = batch["has_candidate"].to(device)
    ell = out["ellipse"]
    final = out["final"]
    control = out["control"]
    res = {}
    with torch.no_grad():
        # predicted topology accuracy against the cached m*
        pred_idx = out["topo"]["pi"].argmax(dim=-1)
        if bool(has_cand.any()):
            res["pred_topo_best_rate"] = float(
                (pred_idx[has_cand] == best[has_cand]).float().mean())
        else:
            res["pred_topo_best_rate"] = float("nan")
        err = torch.linalg.norm(final - p_gt, dim=-1)              # [B,H]
        res["curve_rmse_m"] = float(err.pow(2).mean().sqrt()) * SCENE_TO_METER
        res["curve_max_err_m"] = float(err.max()) * SCENE_TO_METER
        q_err = torch.linalg.norm(control - q_gt, dim=-1)[:, 1:-1]
        res["ctrl_rmse_m"] = float(q_err.pow(2).mean().sqrt()) * SCENE_TO_METER
        # boundary window: how well the RAW polygon keeps the GT local shape
        ws, wg = out["boundary_weights"]
        target_s, target_g = boundary_targets(q_gt, cond)
        raw = out["q_raw_final"]
        ds = torch.linalg.norm(raw - target_s, dim=-1) * (ws[None, :] > 0)
        dg = torch.linalg.norm(raw - target_g, dim=-1) * (wg[None, :] > 0)
        res["boundary_raw_err_m"] = float(
            torch.maximum(ds.max(dim=1).values, dg.max(dim=1).values).mean()
        ) * SCENE_TO_METER
        goal_err = torch.linalg.norm(final[:, -1] - cond[:, 1], dim=-1)
        res["goal_error_m"] = float(goal_err.mean()) * SCENE_TO_METER
        coll = _point_collision(final, occ)
        res["collision_rate"] = float(coll.float().mean())
        center = ell["center"]
        res["ellipse_center_free_rate"] = float(
            (~_point_collision(center, occ)).float().mean())
        a, b, theta = ell["a"], ell["b"], ell["theta"]
        res64 = 64
        mask = ellipse_soft_mask(center, a, b, theta, res64, 10.0)
        free = 1.0 - torch.nn.functional.adaptive_max_pool2d(
            occ.float(), output_size=(res64, res64))[:, 0]
        inside = (mask * free[:, None]).sum(dim=(-1, -2))
        total = mask.sum(dim=(-1, -2)).clamp_min(1e-6)
        res["ellipse_free_frac"] = float((inside / total).mean())
        res["ellipse_area_mean"] = float((np.pi * a * b).mean())
    return res


def rollout_timesteps(B: int, num_timesteps: int, two_step: bool,
                      device="cpu") -> torch.Tensor:
    """The step-1 reverse times of a batch.

    With the two-step rollout the FIRST forward must be at ``t >= 1``: the second
    stage needs a real next reverse step, and ``t = 0`` would give ``s = t = 0``,
    i.e. the degenerate "0 -> 0" update whose ``q_s`` is exactly ``q_t`` (a
    duplicated forward with nothing to learn).  Without the rollout every
    timestep is available, exactly as before.
    """
    low = 1 if two_step else 0
    return torch.randint(int(low), int(num_timesteps), (int(B),), device=device)


def rollout_pair(t: torch.Tensor, num_timesteps: int):
    """The ``(t, s)`` reverse-step pair the second stage rolls out.

    V1 always takes the IMMEDIATELY next reverse step (``s = t - 1``), which is
    what the 16-step sampler does with ``steps=None``.  If training ever needs
    DDIM sub-sampling, the pair must be drawn from the SAME schedule the sampler
    uses (``src.diffusion.sampler.pick_times``) instead of hard-coding ``t - 1``;
    this helper is the single place to change.
    """
    t = t.long()
    return t, (t - 1).clamp(0, max(int(num_timesteps) - 1, 0))


def corridor_fit(points: torch.Tensor, cell_a: torch.Tensor, cell_b: torch.Tensor,
                 cell_valid: torch.Tensor, stride: int = 8) -> float:
    """Fraction of a Skeleton polyline that lies INSIDE the offline corridor.

    Point ``p`` is inside cell ``i`` iff ``A_i p <= b_i`` for every face, so

        violation(p) = min_{valid i} max_f (A_if . p - b_if)

    is positive exactly outside the corridor (the same half-space semantics as
    ``alm_corridor_loss``).  This is a DIAGNOSTIC for the training/inference gap
    of review item 3: the offline corridor was built on the GT route ``m*``, so
    when the second rollout step is routed by the network's own ``argmax(pi)``
    (``train.feedback.topology: "pi"``) this number tells whether that Skeleton
    is still inside the corridor the ALM will project onto.  A value well below
    1 means the corridor does not belong to the chosen Skeleton.
    """
    pts = points[::max(1, int(stride))]
    if pts.numel() == 0 or cell_a.shape[0] == 0:
        return 0.0
    nrm = cell_a.norm(dim=-1).clamp_min(1e-9)
    signed = (torch.einsum("cfk,pk->pcf", cell_a.to(pts.dtype), pts)
              - cell_b.to(pts.dtype)[None]) / nrm[None]
    worst = signed.amax(dim=-1)                       # [P,C] best face per cell
    worst = worst.masked_fill(~cell_valid[None].to(worst.device), float("inf"))
    violation = worst.amin(dim=1)                     # [P] nearest cell
    violation = torch.nan_to_num(violation, nan=0.0, posinf=0.0, neginf=0.0)
    return float((violation <= 0).to(torch.float32).mean())


def feedback_rollout(model, schedule, batch, out1, q_t, t, device,
                     fb_cfg, alm_cfg):
    """One REAL ALM + DDIM step, then the feedback-conditioned second forward.

        Q0_raw(t) = out1["control"]
        Q0_safe(t), lam, stats = ALM(Q0_raw(t), offline corridor pack)
        q_s = DDIM(q_t, Q0_safe(t), t -> t-1)
        feedback = (Q0_safe(t), Q0_safe(t) - Q0_raw(t), valid)
        out2 = network(q_s, feedback)

    Everything before the second forward pass is ``no_grad``: the gradients of
    the feedback losses reach ONLY the second network prediction, never the ALM,
    the DDIM step or the first prediction (the design note's section 12).

    The feedback flag is a REAL verification result, not a constant: a sample
    whose ALM output still violates the exact Bezier pack carries no feedback
    (``valid = 0``), exactly like the sampler's section 4.4 rule.  When the raw
    prediction was already feasible the network is told ``delta = 0`` ("you were
    already safe, no correction was needed").

    ``fb_cfg["topology"]`` selects the Skeleton of the SECOND forward:

        ``"expert"`` (default)  the cached m* = argmin nDTW(curve_gt, S_m), i.e.
                                the Skeleton the OFFLINE corridor was built on -
                                always compatible with the constraint pack;
        ``"pi"``                the FIRST forward's own ``argmax(pi)``, i.e. the
                                routing inference would use.  This is closer to
                                the deployed loop but the offline corridor may
                                NOT belong to the chosen Skeleton, so it is an
                                opt-in experiment (watch ``fb_topo_match`` and
                                ``fb_valid_rate``), not the default.

    Returns ``(out2, pack, diag)``.
    """
    cond = batch["cond"].to(device)
    occ = batch["occupancy"].to(device)
    cand_xy = batch["candidate_xy"].to(device)
    cm = batch["candidate_mask"].to(device)
    geo = batch["candidate_geometry"].to(device)
    gl = batch["candidate_geometry_lengths"].to(device)
    best = batch["topology_best"].to(device)
    cell_a = batch["alm_cell_a"].to(device)
    cell_b = batch["alm_cell_b"].to(device)
    cell_valid = batch["alm_cell_valid"].to(device).bool()
    alm_valid = batch["alm_valid"].to(device).bool()
    B = q_t.shape[0]

    # the offline corridor cache IS a region table, so the training pack is the
    # very same object (exact Bezier extraction + linear inequalities) the
    # inference ALM projects onto
    anchors = torch.linspace(0.0, 1.0, cell_valid.shape[1], device=device)
    pack = build_constraint_pack_from_regions(
        model.bspline, cell_a, cell_b, cell_valid, anchors=anchors,
        device=device, dtype=torch.float32,
        margin=float(alm_cfg.get("constraint_margin", 0.0)),
        sample_valid=alm_valid)

    # a correction is only trusted as history when the ALM output passes the
    # exact pack; ``feedback.accept_tol`` (or ``alm.feedback_accept_tol``)
    # relaxes the strict ``constraint_tol`` to "approximately feasible"
    tol = float(fb_cfg.get("accept_tol",
                           alm_cfg.get("feedback_accept_tol",
                                       alm_cfg.get("constraint_tol", 1e-3))))
    inner = int(fb_cfg.get("alm_inner_steps", alm_cfg.get("inner_steps", 3)))
    q0_raw1 = out1["control"].detach()

    # the Skeleton of the SECOND forward (see the docstring)
    topo_mode = str(fb_cfg.get("topology", "expert") or "expert").lower()
    if topo_mode in ("pi", "argmax", "pred", "predicted"):
        select2 = out1["topo"]["pi"].argmax(dim=-1).detach()
    elif topo_mode in ("expert", "best", "gt"):
        select2 = best
    else:
        raise ValueError("train.feedback.topology must be 'expert' or 'pi', "
                         "got %r" % (fb_cfg.get("topology"),))
    topo_match = (float((select2 == best).to(torch.float32).mean())
                  if B else 1.0)
    # how well the chosen Skeleton sits inside the offline corridor (only worth
    # paying for when the routing is NOT the corridor's own expert route)
    check_fit = bool(fb_cfg.get("check_corridor_fit", topo_mode != "expert"))
    topo_fit = None
    if check_fit:
        fits = []
        for b in range(B):
            if not bool(alm_valid[b]):
                continue
            n = int(gl[b, int(select2[b])].item())
            if n < 2:
                continue
            fits.append(corridor_fit(
                geo[b, int(select2[b]), :n].detach(), cell_a[b].detach(),
                cell_b[b].detach(), cell_valid[b].detach()))
        topo_fit = (sum(fits) / len(fits)) if fits else 0.0

    with torch.no_grad():
        q0_safe1, _, alm_stats = bspline_alm_correct(
            q0_raw1, pack, model.bspline, None, alm_cfg,
            inner_steps=max(1, inner))
        before = alm_stats["max_violation_before"]
        after = alm_stats["max_violation_after"]
        correction = alm_stats["mean_curve_correction_scene"]
        has_pack = pack.num_pieces > 0
        reliable = has_pack & (after <= tol)
        already_ok = reliable & (before <= tol)
        keep = already_ok[:, None, None]
        fb_control = torch.where(keep, q0_raw1, q0_safe1)
        fb_delta = torch.where(keep, torch.zeros_like(q0_safe1),
                               q0_safe1 - q0_raw1)
        fb_valid = reliable

        # the sampler spends its first reverse forwards WITHOUT feedback, so the
        # network must keep both behaviours alive
        if bool(fb_cfg.get("simulate_warmup", False)):
            fb_valid = torch.zeros_like(fb_valid)
        drop = float(fb_cfg.get("drop_prob", 0.0) or 0.0)
        if model.training and drop > 0.0:
            fb_valid = fb_valid & (torch.rand(B, device=device) >= drop)

        # ---- the real DDIM update the sampler would perform ----------------
        s_idx = rollout_pair(t, schedule.num_timesteps)[1]
        sa_t = schedule.sqrt_alphas_cumprod[t].to(device).float()
        s1_t = schedule.sqrt_one_minus_alphas_cumprod[t].to(device).float()
        sa_s = schedule.sqrt_alphas_cumprod[s_idx].to(device).float()
        s1_s = schedule.sqrt_one_minus_alphas_cumprod[s_idx].to(device).float()
        eps = ((q_t - _broadcast(q_t, sa_t) * q0_safe1)
               / _broadcast(q_t, s1_t).clamp_min(1e-12))
        q_s = (_broadcast(q_t, sa_s) * q0_safe1
               + _broadcast(q_t, s1_s) * eps)
        q_s = model.hard_control_endpoints(q_s, cond)

    ab_s = schedule.sqrt_alphas_cumprod[s_idx].to(device)
    out2 = model.forward_all(
        q_s, occ, cond, s_idx, ab_s, cand_xy, cm, geo, gl, select_index=select2,
        feedback_control=fb_control.detach(), feedback_delta=fb_delta.detach(),
        feedback_valid=fb_valid)
    diag = {
        "feedback_valid": fb_valid,
        "raw_violation": before,
        "mean_violation": pack_violation(q0_raw1, pack, reduction="mean"),
        "safe_violation": after,
        "correction": correction,
        "delta_norm": (fb_delta.detach().norm(dim=-1).amax(dim=1)
                       * fb_valid.to(fb_delta.dtype)),
        "has_pack": has_pack,
        "topo_mode": topo_mode,
        "topo_match": topo_match,
        "topo_corridor_fit": topo_fit,
        "step_pair": (int(t.min()), int(s_idx.min())),
    }
    return out2, pack, diag


def batch_losses(batch, model, schedule, lcfg, device, alm_cfg=None,
                 fb_cfg=None):
    fb_cfg = dict(fb_cfg or {})
    alm_cfg = dict(alm_cfg or {})
    # the two-step rollout exists only when the model was built with feedback
    rollout = (bool(getattr(model, "feedback_enabled", False))
               and bool(fb_cfg.get("rollout", True)))
    q0 = batch["control_gt"].to(device)
    cond = batch["cond"].to(device)
    occ = batch["occupancy"].to(device)
    cand_xy = batch["candidate_xy"].to(device)
    cm = batch["candidate_mask"].to(device)
    geo = batch["candidate_geometry"].to(device)
    gl = batch["candidate_geometry_lengths"].to(device)
    best = batch["topology_best"].to(device)
    has_cand = batch["has_candidate"].to(device)
    shape_gt = batch["ellipse_shape4_gt"].to(device)
    shape_valid = batch["shape_valid"].to(device).bool()
    B = q0.shape[0]

    # with the rollout t >= 1 so the second stage always has a real next step
    t = rollout_timesteps(B, schedule.num_timesteps, rollout, device=device)
    q_t, _ = add_noise(q0, t, schedule)
    q_t = model.hard_control_endpoints(q_t, cond)
    ab = schedule.sqrt_alphas_cumprod[t].to(device)

    out = model.forward_all(q_t, occ, cond, t, ab, cand_xy, cm, geo, gl,
                            select_index=best)
    ell = out["ellipse"]

    # The fixed boundary-decoder profile is the ONLY definition of the
    # correction window; the loss reads it from the model, never from the YAML.
    ws, wg = model.boundary_decoder.weights(
        q0.shape[1], device=device, dtype=q0.dtype)
    out["boundary_weights"] = (ws, wg)

    l_ctrl = control_x0_loss(out["q_raw_final"], q0)
    l_coarse = control_x0_loss(out["q_coarse_raw"], q0)
    l_smooth = control_smoothness_loss(
        out["q_raw_final"], q0,
        acc_weight=float(lcfg.get("smooth_acc_weight", 0.25)),
        jerk_weight=float(lcfg.get("smooth_jerk_weight", 1.0)))
    l_boundary = (
        boundary_control_loss(out["q_raw_final"], q0, cond, ws, wg)
        + float(lcfg.get("boundary_coarse_weight", 0.5))
        * boundary_control_loss(out["q_coarse_raw"], q0, cond, ws, wg))
    l_topo = topology_ce(out["topo"]["pi"], best, has_cand)
    l_shape = ellipse_shape_loss(ell["shape4"], shape_gt, shape_valid, has_cand)

    # GT soft mask from the DETACHED fixed Skeleton centres + offline shape
    mask_res = int(lcfg.get("ellipse_safe_res", 64))
    mask_tau = float(lcfg.get("ellipse_mask_tau", 10.0))
    a_gt, b_gt, theta_gt = shape4_to_abtheta(shape_gt)
    gt_mask = ellipse_soft_mask(ell["center"].detach(), a_gt, b_gt, theta_gt,
                                mask_res, mask_tau)
    gt_mask = gt_mask * shape_valid[..., None, None].to(gt_mask.dtype)
    l_iou = ellipse_iou_loss(
        ell["center"], ell["a"], ell["b"], ell["theta"], gt_mask,
        shape_valid, has_cand, raster_res=mask_res, tau=mask_tau)
    l_safe, safe_mean, safe_cvar = ellipse_safety_loss(
        ell["center"], ell["a"], ell["b"], ell["theta"], occ,
        raster_res=mask_res, tau=mask_tau,
        chunk_size=int(lcfg.get("ellipse_safe_chunk", 32)),
        cvar_fraction=float(lcfg.get("safe_cvar_fraction", 0.2)),
        cvar_weight=float(lcfg.get("safe_cvar_weight", 1.0)),
        sample_mask=has_cand)

    # --- ALM corridor: the TRAINING-TIME counterpart of the inference ALM ---
    # out["final"] is the decoded 128-point curve (B_128 @ Q_final), so the
    # gradient reaches the control polygon.  The corridor is the same object
    # bspline_alm_correct() projects onto, precomputed offline; alm_valid gates
    # out the samples whose corridor never closed.
    l_alm = alm_corridor_loss(
        out["final"], batch["alm_cell_a"].to(device),
        batch["alm_cell_b"].to(device),
        batch["alm_cell_valid"].to(device).bool(),
        batch["alm_valid"].to(device).bool())

    raw1 = {"Lctrl": l_ctrl, "Lcoarse": l_coarse,
            "Lsmooth": l_smooth, "Lboundary": l_boundary, "Ltopo": l_topo,
            "Lshape": l_shape, "Liou": l_iou, "Lsafe": l_safe, "Lalm": l_alm,
            "Lfbsafe": q0.sum() * 0.0, "Lcurve": q0.sum() * 0.0}
    weights = {k: float(lcfg.get(key, DEFAULT_LOSS_WEIGHTS[key]))
               for k, key in LOSS_WEIGHT_KEYS.items()}
    total1 = sum(weights[k] * raw1[k] for k in LOSS_KEYS)

    stats = {
        "safe_mean": float(safe_mean.detach()),
        "safe_cvar": float(safe_cvar.detach()),
        "a_mean": float(ell["a"].mean().detach()),
        "b_mean": float(ell["b"].mean().detach()),
    }

    # ---- step 2: the historical-safety-feedback rollout -------------------
    # The SECOND network's OWN raw output is supervised (safety + smoothness +
    # expert shape).  No distillation term ties it to the ALM output, so the
    # network cannot learn the shortcut "copy what the ALM did".
    raw = dict(raw1)
    total2 = None
    if rollout:
        out2, pack, diag = feedback_rollout(model, schedule, batch, out, q_t, t,
                                            device, fb_cfg, alm_cfg)
        w2 = float(fb_cfg.get("step2_weight", 1.0))
        raw2 = {
            "Lctrl": w2 * control_x0_loss(out2["q_raw_final"], q0),
            "Lcoarse": w2 * control_x0_loss(out2["q_coarse_raw"], q0),
            "Lboundary": w2 * (
                boundary_control_loss(out2["q_raw_final"], q0, cond, ws, wg)
                + float(lcfg.get("boundary_coarse_weight", 0.5))
                * boundary_control_loss(out2["q_coarse_raw"], q0, cond, ws, wg)),
            # the polygon the NEXT reverse step would hand to the ALM
            "Lfbsafe": feedback_safety_loss(
                out2["control"], pack,
                margin=float(lcfg.get("feedback_margin", 0.0)),
                sample_mask=diag["has_pack"]),
            "Lcurve": curve_smoothness_loss(
                out2["control"], q0, model.bspline.basis,
                acc_weight=float(lcfg.get("feedback_curve_acc_weight", 0.25)),
                jerk_weight=float(lcfg.get("feedback_curve_jerk_weight", 1.0))),
        }
        for key, value in raw2.items():
            raw[key] = raw1[key] + value
        # only the step-2 terms exist in raw2 (the rest are step-1 only)
        total2 = sum(weights[k] * raw2[k] for k in raw2)
        # ``fb_mean_violation`` is the "how much of the trajectory grazes the
        # corridor" companion of ``fb_raw_violation`` (the deepest point); it is
        # a DIAGNOSTIC only - the training term stays the max violation.
        mean_violation = diag["mean_violation"].detach()
        mask = diag["has_pack"]
        stats.update({
            "fb_valid_rate": float(diag["feedback_valid"].float().mean()),
            "fb_raw_violation": float(diag["raw_violation"].mean()),
            "fb_mean_violation": (float(mean_violation[mask].mean())
                                  if bool(mask.any()) else 0.0),
            "fb_safe_violation": float(diag["safe_violation"].mean()),
            "fb_correction": float(diag["correction"].mean()),
            "fb_delta_norm": float(diag["delta_norm"].mean()),
            "fb_topo_match": float(diag["topo_match"]),
            "fb_t_min": float(diag["step_pair"][0]),
        })
        if diag["topo_corridor_fit"] is not None:
            stats["fb_topo_corridor_fit"] = float(diag["topo_corridor_fit"])

    total = total1 if total2 is None else total1 + total2
    # The two parts touch DISJOINT graphs (the rollout is detached), so the
    # trainer back-propagates them separately: identical gradients, roughly half
    # the peak activation memory of one backward over the sum.
    stats["loss_parts"] = [total1] if total2 is None else [total1, total2]
    return raw, weights, total, out, stats


def validate(model, schedule, loader, lcfg, device, max_batches,
             alm_cfg=None, fb_cfg=None):
    model.eval()
    acc, met, n = {}, {}, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            raw, weights, total, out, stats = batch_losses(
                batch, model, schedule, lcfg, device, alm_cfg=alm_cfg,
                fb_cfg=fb_cfg)
            for k in LOSS_KEYS:
                acc[k] = acc.get(k, 0.0) + float(raw[k].detach())
            acc["total"] = acc.get("total", 0.0) + float(total)
            for k, v in metrics(batch, out, batch["occupancy"].to(device),
                                device).items():
                met[k] = met.get(k, 0.0) + float(v)
            n += 1
    model.train()
    div = max(n, 1)
    summary = {k: v / div for k, v in acc.items()}
    summary.update({k: v / div for k, v in met.items()})
    return summary


# --------------------------------------------------------------------- plots
def _to_px(points, res):
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * res - 0.5


def save_preview(model, schedule, batch, device, out_png, index=0, steps=8,
                 seed=0):
    """One qualitative preview: occupancy, GT/pred curve+controls, ellipses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.diffusion.sampler import sample as ddim_sample

    model.eval()
    with torch.no_grad():
        out = ddim_sample(
            model, schedule, batch["cond"].to(device),
            batch["occupancy"].to(device), batch["candidate_xy"].to(device),
            batch["candidate_mask"].to(device),
            batch["candidate_geometry"].to(device),
            batch["candidate_geometry_lengths"].to(device),
            device=device, steps=steps, seed=seed)
    res = int(batch["occupancy"].shape[-1])
    occ = batch["occupancy"][index, 0].cpu().numpy()
    cond = batch["cond"][index].cpu().numpy()
    p_gt = batch["pos"][index].cpu().numpy()
    q_gt = batch["control_gt"][index].cpu().numpy()
    curve = out["p"][index].cpu().numpy()
    q_pred = out["control"][index].cpu().numpy()
    sel = int(out["selected_idx"][index])
    glen = int(batch["candidate_geometry_lengths"][index, sel])
    gamma = batch["candidate_geometry"][index, sel, :max(glen, 2)].cpu().numpy()
    center = out["ellipse_center"][index].cpu().numpy()
    a = out["ellipse_a"][index].cpu().numpy()
    b = out["ellipse_b"][index].cpu().numpy()
    theta = out["ellipse_theta"][index].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=110)
    for ax in axes:
        ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
        ax.plot(*_to_px(gamma, res).T, color="#1f77b4", lw=1.1,
                label="selected Skeleton")
        ax.plot(*_to_px(cond, res).T, linestyle="none", marker="*", ms=14,
                color="k", label="start / goal")
    ax = axes[0]
    ax.plot(*_to_px(p_gt, res).T, color="#2ca02c", lw=2.0, label="GT curve")
    ax.plot(*_to_px(curve, res).T, color="#d62728", lw=1.6, ls="--",
            label="pred curve")
    ax.plot(*_to_px(q_gt, res).T, color="#2ca02c", lw=0.8, alpha=0.6,
            marker="o", ms=2, label="GT controls")
    ax.plot(*_to_px(q_pred, res).T, color="#d62728", lw=0.8, alpha=0.6,
            marker="o", ms=2, label="pred controls")
    ax.set_title("curve + control polygons", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    ang = np.linspace(0, 2 * np.pi, 48)
    cp = _to_px(center, res)
    for k in range(0, len(center), 4):
        ct, st = np.cos(theta[k]), np.sin(theta[k])
        ex = a[k] * np.cos(ang) * res / 2.0
        ey = b[k] * np.sin(ang) * res / 2.0
        ax.plot(ct * ex - st * ey + cp[k, 0], st * ex + ct * ey + cp[k, 1],
                color="#d62728", lw=0.7, alpha=0.8)
    ax.scatter(cp[:, 0], cp[:, 1], s=1.5, c="#d62728")
    ax.plot(*_to_px(curve, res).T, color="#d62728", lw=1.4, ls="--")
    ax.set_title("128 fixed Skeleton centres + ellipses", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    model.train()
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--overfit", type=int, default=0,
                    help="train on the first N samples only")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps (effective batch "
                         "= batch_size * accum)")
    ap.add_argument("--init-best-val", type=float, default=None,
                    help="seed best_val when resuming so an older (better) "
                         "checkpoint is not overwritten by a worse one")
    ap.add_argument("--init-best-task", type=float, default=None,
                    help="seed the task-metric best score when resuming")
    ap.add_argument("--steps-per-save", type=int, default=200,
                    help="also write latest.pt every N optimizer steps "
                         "(crash resilience); 0 disables")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N samples of each split")
    ap.add_argument("--data-root", default=None,
                    help="override data.processed_root")
    ap.add_argument("--preview-png", default=None)
    ap.add_argument("--preview-steps", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    env, data_cfg = cfg["env"], cfg["data"]
    model_cfg, diff_cfg = cfg["model"], cfg["diffusion"]
    loss_cfg, train_cfg = cfg["loss"], cfg["train"]
    alm_cfg = dict(cfg.get("alm") or {})
    fb_cfg = dict(train_cfg.get("feedback") or {})
    print("[train] scene_to_meter=%.1f m/unit (reporting only)"
          % set_scene_to_meter(cfg), flush=True)

    set_seed(int(env.get("seed", 42)))
    # cuDNN's engine search can fail ("FIND was unable to find an engine") when
    # another GPU process (CARLA) holds most of the memory, so it is OFF by
    # default and can be re-enabled from the config.
    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = False
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available()
               and env.get("device", "cuda") == "cuda" else "cpu"))
    print("[train] device=%s" % device, flush=True)

    processed_root = args.data_root or data_cfg.get(
        "processed_root", "data/carla_processed")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))
    batch_size = int(args.batch_size or data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))
    n_ctrl = num_controls(cfg)

    if args.overfit:
        train_loader, train_ds = make_loader(
            "train", processed_root, batch_size=min(batch_size, args.overfit),
            shuffle=True, num_workers=num_workers, geometry_points=geo_points,
            limit=args.overfit, num_controls=n_ctrl)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=min(batch_size, args.overfit), shuffle=True,
            num_workers=num_workers, drop_last=False,
            collate_fn=make_collate(train_ds))
    else:
        train_loader, train_ds = make_loader(
            "train", processed_root, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, geometry_points=geo_points,
            limit=args.limit, num_controls=n_ctrl)
    val_loader = val_ds = None
    if not args.no_val:
        val_loader, val_ds = make_loader(
            "val", processed_root, batch_size=batch_size, shuffle=False,
            num_workers=0, geometry_points=geo_points, limit=args.limit,
            num_controls=n_ctrl)
    print("[data] train=%d val=%s batch=%d geometry_points=%d controls=%d"
          % (len(train_ds), len(val_ds) if val_ds else "-", batch_size,
             geo_points, n_ctrl), flush=True)

    schedule = NoiseSchedule(
        diff_cfg["timesteps"],
        beta_schedule=diff_cfg.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02)).to(device)
    model = TrajSafePlanner(model_cfg, cfg.get("ellipse_label"),
                            cfg.get("bspline")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] params=%.2fM controls=%d safety_queries=%d mode=%s "
          "traj_blocks=%d skeleton_blocks=%d final_blocks=%d curve=%d "
          "boundary_profile=%s feedback=%s"
          % (n_params / 1e6, model.num_controls, model.num_safety_queries,
             "control_space" if model.control_space else "legacy_curve",
             model.traj_blocks, model.skeleton_blocks, model.final_blocks,
             model.curve_points,
             [float(v) for v in model.boundary_decoder.profile.tolist()],
             ("on(hidden=%d, rollout=%s, drop=%.2f)"
              % (model.feedback_hidden, bool(fb_cfg.get("rollout", True)),
                 float(fb_cfg.get("drop_prob", 0.0) or 0.0)))
             if getattr(model, "feedback_enabled", False) else "off"),
          flush=True)
    if getattr(model, "feedback_enabled", False) and alm_cfg and not alm_cfg.get("enabled", False):
        print("[warn] model.feedback.enabled is on but alm.enabled is off: the "
              "training rollout still uses the OFFLINE corridor pack, while "
              "inference would never activate one", flush=True)

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    log_interval = (args.log_interval if args.log_interval is not None
                    else int(train_cfg.get("log_interval", 20)))
    ckpt_dir = args.ckpt_dir or train_cfg.get("ckpt_dir",
                                              "outputs/bspline_carla/ckpt")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(ckpt_dir))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    print("[ckpt] %s" % ckpt_dir, flush=True)

    lr = float(args.lr if args.lr is not None else train_cfg["lr"])
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    start_epoch = 0
    if args.resume:
        ck = load_checkpoint(args.resume, model, optim, map_location=device)
        start_epoch = int(ck.get("epoch", 0)) + 1
        ckpt_arch = detect_architecture(ck.get("model_state", ck))
        running = (ARCH_CONTROL_SPACE if model.control_space
                   else ARCH_LEGACY_CURVE)
        if ckpt_arch != running:
            print("[resume] NOTE: the checkpoint was written by the %s model but "
                  "the config runs %s: the shared weights are reused and the new "
                  "modules (safety_query_head / safety_cross_attention) start "
                  "from their initialisation" % (ckpt_arch, running), flush=True)
        print("[resume] epoch %d" % start_epoch, flush=True)
    if args.init_best_val is not None:
        best_val = float(args.init_best_val)
        print("[resume] seed best_val=%.6f" % best_val, flush=True)
    if args.init_best_task is not None:
        best_task = float(args.init_best_task)
        print("[resume] seed best_task=%.6f" % best_task, flush=True)

    grad_clip = float(train_cfg.get("grad_clip", 0.0)) or None
    eval_every = int(train_cfg.get("eval_every", 1))
    save_every = int(train_cfg.get("save_every", 5))
    max_hours = (args.max_hours if args.max_hours is not None
                 else train_cfg.get("max_hours"))
    best_val = float("inf")
    best_epoch = -1
    best_task = float("inf")
    best_task_epoch = -1
    t0 = time.time()
    history = []
    summary_path = os.path.join(out_dir, "training_summary.json")
    latest = os.path.join(ckpt_dir, "latest.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    best_task_path = os.path.join(ckpt_dir, "best_task.pt")

    def write_summary(extra=None):
        payload = {
            "model": "TrajSafePlanner-controlspace-%d" % model.num_controls,
            "config": args.config,
            "processed_root": processed_root,
            "device": str(device),
            "params_m": n_params / 1e6,
            "num_controls": int(model.num_controls),
            "num_safety_queries": int(model.num_safety_queries),
            "control_space": bool(model.control_space),
            "boundary_profile": [float(v) for v in
                                 model.boundary_decoder.profile.tolist()],
            "lr": lr,
            "batch_size": batch_size,
            "train_samples": int(len(train_ds)),
            "val_samples": int(len(val_ds)) if val_ds else 0,
            "epochs_requested": epochs,
            "epochs_done": start_epoch + len(history),
            "best_val_total": (None if best_val == float("inf") else best_val),
            "best_epoch": best_epoch,
            "latest_ckpt": latest,
            "best_ckpt": (best_path if os.path.exists(best_path) else None),
            "best_task_ckpt": (best_task_path
                               if os.path.exists(best_task_path) else None),
            "best_task_score": (None if best_task == float("inf")
                                else best_task),
            "best_task_epoch": best_task_epoch,
            "elapsed_seconds": time.time() - t0,
            "history": history,
            "loss_weights": {k: float(loss_cfg.get(m, DEFAULT_LOSS_WEIGHTS[m]))
                             for k, m in LOSS_WEIGHT_KEYS.items()},
            "alm_enabled": bool((cfg.get("alm") or {}).get("enabled", False)),
            "feedback": {
                "enabled": bool(getattr(model, "feedback_enabled", False)),
                "hidden": int(getattr(model, "feedback_hidden", 0)),
                "rollout": bool(fb_cfg.get("rollout", True)),
                "drop_prob": float(fb_cfg.get("drop_prob", 0.0) or 0.0),
                "step2_weight": float(fb_cfg.get("step2_weight", 1.0)),
            },
        }
        if extra:
            payload.update(extra)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    preview_batch = None
    for epoch in range(start_epoch, epochs):
        model.train()
        acc, n_steps, wacc, gnorms = {}, 0, {}, []
        macc = {}
        accum = max(1, int(args.accum))
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            # The GPU is shared with other processes on this machine, so a
            # transient CUDA failure must not kill the whole night: retry a few
            # times, then let the supervisor restart from latest.pt.
            attempt = 0
            while True:
                try:
                    raw, weights, total, out, stats = batch_losses(
                        batch, model, schedule, loss_cfg, device,
                        alm_cfg=alm_cfg, fb_cfg=fb_cfg)
                    if not bool(torch.isfinite(total)):
                        raise RuntimeError(
                            "non-finite total loss at epoch %d step %d"
                            % (epoch, step))
                    # gradient accumulation keeps the effective batch size while
                    # lowering the peak activation memory.  The rollout's two
                    # parts live on disjoint graphs, so they are back-propagated
                    # separately (same gradients, lower peak memory).
                    for part in stats.get("loss_parts") or [total]:
                        (part / accum).backward()
                    break
                except RuntimeError as exc:
                    msg = str(exc)
                    low = msg.lower()
                    transient = ("out of memory" in low
                                 or "unable to find an engine" in low
                                 or "unknown error" in low
                                 or "cuda error" in low
                                 or "memory allocation failure" in low)
                    if (not transient) or attempt >= 4:
                        raise
                    attempt += 1
                    print("[warn] transient CUDA failure (%s); retry %d/4"
                          % (msg[:100].replace("\n", " "), attempt), flush=True)
                    optim.zero_grad(set_to_none=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(10.0)
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip if grad_clip else 1e9))
                if not np.isfinite(gnorm):
                    raise RuntimeError(
                        "non-finite gradient norm at epoch %d step %d"
                        % (epoch, step))
                gnorms.append(gnorm)
                optim.step()
                optim.zero_grad(set_to_none=True)
            for k in LOSS_KEYS:
                acc[k] = acc.get(k, 0.0) + float(raw[k].detach())
                wacc[k] = wacc.get(k, 0.0) + weights[k] * float(raw[k].detach())
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    macc[k] = macc.get(k, 0.0) + float(v)
            n_steps += 1
            if preview_batch is None:
                preview_batch = {k: (v.detach().cpu().clone()
                                     if torch.is_tensor(v) else v)
                                 for k, v in batch.items()}
            if log_interval and (step + 1) % log_interval == 0:
                avg_raw = {k: v / n_steps for k, v in acc.items()}
                avg_w = {k: v / n_steps for k, v in wacc.items()}
                print("[e%d s%d/%d] %s | %s | gnorm=%.3f t=%.0fs"
                      % (epoch, step + 1, len(train_loader),
                         " ".join("%s=%.4f" % (k, avg_raw[k])
                                  for k in LOSS_KEYS),
                         " ".join("w%s=%.4f" % (k, avg_w[k])
                                  for k in LOSS_KEYS),
                         sum(gnorms) / len(gnorms), time.time() - t0),
                      flush=True)
            if (args.steps_per_save and n_steps % int(args.steps_per_save) == 0
                    and (step + 1) % accum == 0):
                save_checkpoint(latest, model, optim, epoch, cfg)
        if n_steps == 0:
            print("[epoch %d] no training step (empty loader)" % epoch, flush=True)
            break
        avg_raw = {k: v / max(n_steps, 1) for k, v in acc.items()}
        avg_extra = {k: v / max(n_steps, 1) for k, v in macc.items()}
        line = ("[epoch %d/%d] gnorm=%.3f " % (
            epoch, epochs, sum(gnorms) / max(len(gnorms), 1))
            + " ".join("%s=%.4f" % (k, avg_raw[k]) for k in LOSS_KEYS))
        if avg_extra:
            line += " | fb " + " ".join(
                "%s=%.4f" % (k, avg_extra[k]) for k in sorted(avg_extra))
        entry = {"epoch": epoch, "train": {k: float(avg_raw[k])
                                           for k in LOSS_KEYS},
                 "seconds": time.time() - t0}
        if avg_extra:
            entry["train"].update({k: float(v) for k, v in avg_extra.items()})
        if val_loader is not None and ((epoch + 1) % eval_every == 0
                                       or epoch == epochs - 1):
            v = validate(model, schedule, val_loader, loss_cfg, device,
                         int(train_cfg.get("val_batches", 12)),
                         alm_cfg=alm_cfg, fb_cfg=fb_cfg)
            entry["val"] = {k: float(x) for k, x in v.items()}
            line += " | val " + " ".join(
                "%s=%.4f" % (k, v[k]) for k in LOSS_KEYS if k in v)
            line += " | rmse_m=%.4f topo=%.3f coll=%.4f" % (
                v.get("curve_rmse_m", float("nan")),
                v.get("pred_topo_best_rate", float("nan")),
                v.get("collision_rate", float("nan")))
            if v["total"] < best_val:
                best_val = v["total"]
                best_epoch = epoch
                save_checkpoint(best_path, model, optim, epoch, cfg)
                line += " (best)"
            # The val CE of the topology head overfits early while the task
            # metrics keep improving, so a second checkpoint tracks the task
            # objective (curve RMSE in meters + one scene unit * collision
            # rate, so the balance stays scale-consistent across datasets).
            task = (float(v.get("curve_rmse_m", float("inf")))
                    + SCENE_TO_METER * float(v.get("collision_rate", 0.0)))
            if task < best_task:
                best_task = task
                best_task_epoch = epoch
                save_checkpoint(best_task_path, model, optim, epoch, cfg)
                line += " (task-best %.4f)" % task
        print(line, flush=True)
        save_checkpoint(latest, model, optim, epoch, cfg)
        if save_every and (epoch + 1) % save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, "epoch_%d.pt" % (epoch + 1)),
                            model, optim, epoch, cfg)
        history.append(entry)
        write_summary()
        if max_hours and (time.time() - t0) / 3600.0 >= float(max_hours):
            print("[stop] wall-clock budget %.2fh reached after epoch %d"
                  % (float(max_hours), epoch), flush=True)
            break

    if not os.path.exists(best_path) and val_loader is not None:
        # guarantee a best.pt even if validation never triggered
        v = validate(model, schedule, val_loader, loss_cfg, device,
                     int(train_cfg.get("val_batches", 12)),
                     alm_cfg=alm_cfg, fb_cfg=fb_cfg)
        best_val = v["total"]
        best_epoch = start_epoch + len(history) - 1
        save_checkpoint(best_path, model, optim, best_epoch, cfg)
        print("[best] forced validation total=%.4f" % best_val, flush=True)

    if args.preview_png and preview_batch is not None:
        try:
            save_preview(model, schedule, preview_batch, device,
                         args.preview_png, index=0, steps=args.preview_steps)
            print("[preview] %s" % args.preview_png, flush=True)
        except Exception as exc:                              # pragma: no cover
            print("[preview] failed: %r" % (exc,), flush=True)
    write_summary({"finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print("[done] epochs=%d best val L=%.4f best_epoch=%d ckpt=%s"
          % (start_epoch + len(history), best_val, best_epoch, ckpt_dir),
          flush=True)


if __name__ == "__main__":
    main()
