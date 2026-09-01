"""Total training objective following the sequential (trajectory -> ellipse) design.

Three phases:
  "traj"    : total = L_z + L_p + L_smooth (no collision loss)
  "ellipse" : total = L_E
  "joint"   : total = L_traj + lambda_E * L_E + lambda_align * L_align

L_traj = lam_z*L_Z + lam_p*L_p + lam_s*L_smooth
Phase 3 additionally enables lam_col*L_collision.
L_E    = lam_param*L_param + lam_iou*L_iou + lam_coll*L_ecoll + lam_anchor*L_anchor

In Phase 3 the ellipse objective flows back into the trajectory network
(L_ecoll / L_anchor / L_iou all depend on p_hat via Local_2 and c = p_hat + delta).
"""

import torch

from src.diffusion.zerosum import compute_z0, zero_sum
from .trajectory_loss import l_z, l_p, l_smooth, l_collision
from .ellipse_loss import (ellipse_param_loss, ellipse_align_loss,
                           ellipse_iou_loss, ellipse_collision_loss,
                           ellipse_anchor_loss)


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


def total_loss(model_out, batch, cfg_loss, device="cuda", phase="joint", epoch=0,
               ellipse_cfg=None):
    pos_gt = batch["pos"].float()
    cond = batch["cond"].float()
    z0_gt, _, _, _ = compute_z0(pos_gt, cond)

    z0_pred = model_out["z0_pred"]
    z0_proj = zero_sum(z0_pred)
    pos_pred = model_out["pos_pred"]

    L_Z = l_z(z0_proj, z0_gt)
    L_p = l_p(pos_pred, pos_gt)
    L_sm = l_smooth(z0_proj)
    # Phase 1 intentionally learns reconstruction and smoothness only.
    # Trajectory collision avoidance is introduced in the joint phase.
    if phase == "joint":
        L_col = l_collision(pos_pred, batch["sdf_tensor"],
                            margin=cfg_loss.get("collision_margin", 0.0),
                            sigma=cfg_loss.get("collision_sigma", 0.1))
    else:
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
    elif phase == "ellipse":
        total = L_E
    elif phase == "joint":
        total = traj_loss + lam_col * L_col + lam_E * L_E + lam_align * L_align
    else:
        raise ValueError(f"unknown phase {phase}")

    return {
        "total": total, "L_Z": L_Z, "L_p": L_p, "L_smooth": L_sm, "L_col": L_col,
        "L_param": L_param, "L_iou": L_iou, "L_ecoll": L_ecoll,
        "L_anchor": L_anchor, "L_align": L_align, "L_ellipse": L_E,
        "lambda_E": lam_E, "phase": phase,
    }


def _zero_like(x):
    return x
