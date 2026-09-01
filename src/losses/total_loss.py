"""Total training objective with GradNorm automatic weighting (no hand-set weights for the
main balancing losses).

- L_Z             : fixed anchor weight (lambda_z).
- L_p / L_param / L_iou / L_smooth : GradNorm dynamic weights (gradient-magnitude balanced).
- L_axis          : schedule ramp.
- L_col           : Lagrangian dynamic weight (constraint).
"""

import torch

from src.diffusion.zerosum import compute_z0, zero_sum, integrate_positions
from .trajectory_loss import l_z, l_p, l_smooth, l_collision
from .ellipse_loss import ellipse_param_loss, ellipse_axis_loss, ellipse_iou_loss


def _schedule_lambda(epoch, cfg, key):
    warm = int(cfg.get(f"{key}_warmup_epochs", 0))
    ramp = int(cfg.get(f"{key}_ramp_epochs", 10))
    maxv = float(cfg.get(f"{key}_max", 0.0))
    if maxv <= 0.0:
        return 0.0
    if epoch < warm:
        return 0.0
    return maxv * min(1.0, (epoch - warm + 1) / max(1, ramp))


def total_loss(model_out, batch, cfg_loss, device="cuda", warmup=False, epoch=0,
               gradnorm=None, shared_params=None, lag_col=None):
    pos_gt = batch["pos"].float()
    cond = batch["cond"].float()
    z0_gt, _, base, _ = compute_z0(pos_gt, cond)
    z0_pred = model_out["z0_pred"]
    z0_proj = zero_sum(z0_pred)
    delta_pred = base + z0_proj
    pos_pred = integrate_positions(cond[:, 0], delta_pred)

    L_Z = l_z(z0_proj, z0_gt)
    L_p = l_p(pos_pred, pos_gt)
    L_sm = l_smooth(z0_proj)

    lam_z = cfg_loss.get("lambda_z", 1.0)
    lam_s = cfg_loss.get("lambda_smooth", 0.05)

    if warmup:
        total = lam_z * L_Z + lam_s * L_sm
        zero = torch.zeros((), device=device)
        return {
            "total": total, "L_Z": L_Z, "L_p": L_p, "L_smooth": L_sm,
            "L_param": zero, "L_iou": zero, "L_axis": zero, "L_col": zero,
            "L_ellipse": zero,
        }

    el = ellipse_param_loss(
        pred_center=model_out["ellipse_center"],
        pred_r1=model_out["ellipse_radii"][..., 0],
        pred_r2=model_out["ellipse_radii"][..., 1],
        pred_dir=model_out["ellipse_dir"],
        gt_center=batch["ellipse_params"][..., 0:2],
        gt_r=batch["ellipse_params"][..., 2:4],
        gt_Q=batch["ellipse_Q"].float(),
        gt_valid=batch["ellipse_valid"],
        cfg_loss=cfg_loss)
    L_param = el["L_param"]

    L_iou = ellipse_iou_loss(
        pred_center=model_out["ellipse_center"],
        pred_r1=model_out["ellipse_radii"][..., 0],
        pred_r2=model_out["ellipse_radii"][..., 1],
        pred_theta=model_out["ellipse_theta"],
        gt_center=batch["ellipse_params"][..., 0:2],
        gt_r=batch["ellipse_params"][..., 2:4],
        gt_Q=batch["ellipse_Q"].float(),
        gt_valid=batch["ellipse_valid"],
        cfg_loss=cfg_loss)

    L_axis = ellipse_axis_loss(
        pred_center=model_out["ellipse_center"],
        pred_r1=model_out["ellipse_radii"][..., 0],
        pred_r2=model_out["ellipse_radii"][..., 1],
        pred_theta=model_out["ellipse_theta"],
        pos_pred=pos_pred[:, 1:])

    L_col = l_collision(pos_pred, batch["sdf_tensor"],
                        margin=cfg_loss.get("collision_margin", 0.0),
                        sigma=cfg_loss.get("collision_sigma", 0.1))

    # GradNorm dynamic weights for the main auxiliary tasks
    gn_keys = {"L_p": L_p, "L_param": L_param, "L_iou": L_iou, "L_smooth": L_sm}
    if gradnorm is not None and shared_params is not None:
        gn = gradnorm.weights(gn_keys, shared_params)
        w_p, w_param, w_iou, w_smooth = gn["L_p"], gn["L_param"], gn["L_iou"], gn["L_smooth"]
    else:
        w_p = cfg_loss.get("lambda_p", 0.5)
        w_param = cfg_loss.get("lambda_ellipse", 0.5)
        w_iou = cfg_loss.get("lambda_iou", 0.3)
        w_smooth = lam_s

    w_axis = _schedule_lambda(epoch, cfg_loss, "axis")
    if lag_col is not None:
        w_col = lag_col.step(L_col)
    else:
        w_col = _schedule_lambda(epoch, cfg_loss, "collision")

    total = (lam_z * L_Z + w_smooth * L_sm + w_p * L_p
             + w_param * L_param + w_iou * L_iou
             + w_axis * L_axis + w_col * L_col)

    return {
        "total": total, "L_Z": L_Z, "L_p": L_p, "L_smooth": L_sm,
        "L_param": L_param, "L_iou": L_iou, "L_axis": L_axis, "L_col": L_col,
        "L_ellipse": w_param * L_param + w_iou * L_iou,
        "w_p": w_p, "w_param": w_param, "w_iou": w_iou, "w_smooth": w_smooth,
        "w_axis": w_axis, "w_col": w_col,
    }
