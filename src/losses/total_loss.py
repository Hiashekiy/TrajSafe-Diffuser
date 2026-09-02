"""Total training objective following the sequential (trajectory -> ellipse) design.

Three phases:
  "traj"    : total = L_z + L_p + L_smooth (no collision loss)
  "ellipse" : total = L_E
  "joint"   : total = lambda_smooth_joint * L_smooth + L_E
                     + w_epoch(e) * w_AL(t) * L_AL

L_traj = lam_z*L_Z + lam_p*L_p + lam_s*L_smooth
L_E    = lam_param*L_param + lam_iou*L_iou + lam_coll*L_ecoll + lam_anchor*L_anchor

Phase 3 (V5) explicitly drops L_Z, L_p, L_align, L_col and J_guide from the
training loss; safety is enforced by the Augmented-Lagrangian segment constraint
L_AL whose convex region (A,b) is detached so it only shapes the trajectory.
Consensus guidance (J_guide) is NOT a loss term: it is applied as a gradient
correction on z0 in the training/sampling forward pass (see train.py / sampler).
"""

import torch

from src.diffusion.zerosum import compute_z0, zero_sum
from .trajectory_loss import l_z, l_p, l_smooth, l_collision
from .ellipse_loss import (ellipse_param_loss, ellipse_align_loss,
                           ellipse_iou_loss, ellipse_collision_loss,
                           ellipse_anchor_loss)
from .al_loss import al_safety_loss


def _schedule_lambda(epoch, cfg, key):
    warm = int(cfg.get(f"{key}_warmup_epochs", 0))
    ramp = int(cfg.get(f"{key}_ramp_epochs", 10))
    maxv = float(cfg.get(f"{key}_max", cfg.get(f"lambda_{key}_max", 0.0)))
    if maxv <= 0.0:
        return 0.0
    if epoch < warm:
        return 0.0
    return maxv * min(1.0, (epoch - warm + 1) / max(1, ramp))


def _joint_ellipse_lambda(epoch, ecfg):
    ramp = bool(ecfg.get("ramp", ecfg.get("joint_ellipse_ramp", True)))
    target = float(ecfg.get("lambda_max", ecfg.get("joint_ellipse_lambda_max", 1.0)))
    if not ramp:
        return target
    ratio = float(ecfg.get("ramp_ratio", ecfg.get("joint_ellipse_ramp_ratio", 0.2)))
    ramp_epochs = int(ecfg.get("ramp_epochs", ecfg.get("joint_ellipse_ramp_epochs",
                                                       max(1, int(round(10 * ratio))))))
    return target * min(1.0, (epoch + 1) / max(1, ramp_epochs))


def _epoch_ramp(epoch, al_cfg):
    """w_epoch(e) = min(1, e / E_ramp), with e starting at 0.

    This strictly follows the V5 formula: the first epoch (e=0) has weight 0,
    and the weight reaches 1.0 at e = E_ramp.
    """
    ratio = float(al_cfg.get("epoch_ramp_ratio", 0.2))
    ramp_epochs = int(al_cfg.get("epoch_ramp_epochs", max(1, int(round(10 * ratio)))))
    return min(1.0, max(0, epoch) / max(1, ramp_epochs))


def total_loss(model_out, batch, cfg_loss, device="cuda", phase="joint", epoch=0,
               ellipse_cfg=None, diffusion_t=None, num_timesteps=None,
               al_state=None, al_cfg=None, joint_loss_cfg=None):
    pos_gt = batch["pos"].float()
    cond = batch["cond"].float()
    z0_gt, _, _, _ = compute_z0(pos_gt, cond)

    z0_pred = model_out["z0_pred"]
    z0_proj = zero_sum(z0_pred)
    pos_pred = model_out["pos_pred"]

    L_Z = l_z(z0_proj, z0_gt)
    L_p = l_p(pos_pred, pos_gt)
    L_sm = l_smooth(z0_proj)
    # V5: L_col is no longer a core Phase-3 loss item (the convex-region AL term is
    # the safety constraint).  Keep a zero placeholder for logging compatibility.
    L_col = pos_pred.new_zeros(())

    lam_z = cfg_loss.get("lambda_z", 1.0)
    lam_p = cfg_loss.get("lambda_p", 0.5)
    lam_s = cfg_loss.get("lambda_smooth", 0.05)
    lam_col = float(cfg_loss.get("lambda_collision",
                                  cfg_loss.get("lambda_collision_max", 0.0)))

    # ---- ellipse objective ----
    has_ellipse = model_out.get("ellipse_center") is not None
    if has_ellipse:
        ecfg = ellipse_cfg if ellipse_cfg is not None else cfg_loss.get("ellipse_loss", cfg_loss)
        valid = batch["ellipse_valid"]
        gt_center = batch["ellipse_params"][..., 0:2]
        gt_r = batch["ellipse_params"][..., 2:4]
        gt_Q = batch["ellipse_Q"].float()

        el = ellipse_param_loss(
            pred_center=model_out["ellipse_center"],
            pred_r1=model_out["ellipse_radii"][..., 0],
            pred_r2=model_out["ellipse_radii"][..., 1],
            pred_dir=model_out["ellipse_dir"],
            gt_center=gt_center, gt_r=gt_r, gt_Q=gt_Q, gt_valid=valid,
            cfg_loss=ecfg)
        L_param = el["L_param"]

        L_iou = ellipse_iou_loss(
            pred_center=model_out["ellipse_center"],
            pred_r1=model_out["ellipse_radii"][..., 0],
            pred_r2=model_out["ellipse_radii"][..., 1],
            pred_theta=model_out["ellipse_theta"],
            gt_center=gt_center, gt_r=gt_r, gt_Q=gt_Q, gt_valid=valid,
            cfg_loss=ecfg)

        L_ecoll = ellipse_collision_loss(
            pred_center=model_out["ellipse_center"],
            pred_r1=model_out["ellipse_radii"][..., 0],
            pred_r2=model_out["ellipse_radii"][..., 1],
            pred_theta=model_out["ellipse_theta"],
            sdf_tensor=batch["sdf_tensor"], gt_valid=valid, cfg_loss=ecfg)

        L_anchor = ellipse_anchor_loss(
            pred_center=model_out["ellipse_center"],
            pred_r1=model_out["ellipse_radii"][..., 0],
            pred_r2=model_out["ellipse_radii"][..., 1],
            pred_theta=model_out["ellipse_theta"],
            pred_anchor=pos_pred[:, 1:], gt_valid=valid, cfg_loss=ecfg)

        L_align = ellipse_align_loss(
            pred_r1=model_out["ellipse_radii"][..., 0],
            pred_r2=model_out["ellipse_radii"][..., 1],
            pred_theta=model_out["ellipse_theta"],
            pos_pred=pos_pred, gt_valid=valid,
            near_circle_tau=ecfg.get("near_circle_tau", 0.1),
            mask_near_circle=ecfg.get("near_circle_angle_mask", True))

        lam_param = ecfg.get("lambda_param", 10.0)
        lam_iou = ecfg.get("lambda_iou", 5.0)
        lam_coll = ecfg.get("lambda_coll", 2.0)
        lam_anchor = ecfg.get("lambda_anchor", 5.0)
        L_E = (lam_param * L_param + lam_iou * L_iou
               + lam_coll * L_ecoll + lam_anchor * L_anchor)
        lam_E = _joint_ellipse_lambda(epoch, ecfg)
    else:
        L_param = _zero_like(torch.zeros((), device=device))
        L_iou = L_param
        L_ecoll = L_param
        L_anchor = L_param
        L_align = L_param
        L_E = L_param
        lam_E = 0.0

    lam_align = cfg_loss.get("lambda_align", 0.1)

    traj_loss = lam_z * L_Z + lam_p * L_p + lam_s * L_sm

    if phase == "traj":
        total = traj_loss
        L_AL = _zero_like(torch.zeros((), device=device))
        al_mean_V = L_AL
        al_mean_Q = L_AL
        al_w_active = 0.0
    elif phase == "ellipse":
        total = L_E
        L_AL = _zero_like(torch.zeros((), device=device))
        al_mean_V = L_AL
        al_mean_Q = L_AL
        al_w_active = 0.0
    elif phase == "joint":
        # V5: drop L_Z/L_p/L_align/L_col (main), keep L_smooth + L_E + AL.
        jlc = joint_loss_cfg or {}
        lam_sm_joint = float(jlc.get("lambda_smooth",
                                     cfg_loss.get("lambda_smooth_joint", 0.05)))
        total = lam_sm_joint * L_sm + L_E
        L_AL = _zero_like(torch.zeros((), device=device))
        al_mean_V = L_AL
        al_mean_Q = L_AL
        al_w_active = 0.0
        w_epoch = _epoch_ramp(epoch, al_cfg or {})

        if w_epoch > 0.0 and al_cfg and al_cfg.get("enabled", True) and has_ellipse                 and diffusion_t is not None and num_timesteps is not None:
            dual = (al_state or {}).get("dual", float(al_cfg.get("dual_init", 0.1)))
            out_al = al_safety_loss(
                pos_pred, model_out["ellipse_center"],
                model_out["ellipse_radii"], model_out["ellipse_theta"],
                batch["map_tensor"], diffusion_t, al_cfg, device, dual,
                num_timesteps,
                segment_stride=int(al_cfg.get("segment_stride", 1)))
            L_AL = out_al["L_AL"]
            al_mean_V = out_al["mean_V"]
            al_mean_Q = out_al["mean_Q"]
            al_w_active = out_al["w_active"]
            total = total + w_epoch * L_AL
    else:
        raise ValueError(f"unknown phase {phase}")

    return {
        "total": total, "L_Z": L_Z, "L_p": L_p, "L_smooth": L_sm, "L_col": L_col,
        "L_param": L_param, "L_iou": L_iou, "L_ecoll": L_ecoll,
        "L_anchor": L_anchor, "L_align": L_align, "L_ellipse": L_E,
        "lambda_E": lam_E, "phase": phase,
        "L_AL": L_AL, "al_mean_V": al_mean_V, "al_mean_Q": al_mean_Q,
        "al_w_active": al_w_active,
    }


def _zero_like(x):
    return x
