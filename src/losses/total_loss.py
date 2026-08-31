"""Total training objective combining all losses."""

import torch

from .ellipse_loss import ellipse_loss
from .trajectory_loss import l_collision, l_diff, l_smooth


def total_loss(model_out, batch, map_tensor, sdf_map, extent, state_norm,
               cfg_loss, cfg_geom, device="cuda", warmup=False):
    """Compute the total loss.

    warmup=True: only train the trajectory reconstruction branch (L_diff + a tiny
    L_smooth) so the diffusion backbone first learns to produce a *spread-out*
    trajectory before the safety/geometry losses are turned on.  This avoids the
    "safe-point collapse" observed when geometric losses dominate from step 0.
    """
    x0_pred = model_out["x0_pred"]
    x0 = batch["traj"].float()
    gt_center = batch["ellipse_params"][..., 0:2]
    gt_r = batch["ellipse_params"][..., 2:4]
    gt_Q = batch["ellipse_Q"].float()
    gt_valid = batch["ellipse_valid"]
    device = x0_pred.device

    # ---- trajectory core losses (always) ----
    L_diff = l_diff(x0_pred, x0,
                    lambda_var=cfg_loss.get("lambda_var", 1.0),
                    lambda_var_vel=cfg_loss.get("lambda_var_vel", 1.0))
    L_smooth = l_smooth(x0_pred, state_norm)

    lam_diff = cfg_loss.get("lambda_diff", 1.0)
    lam_s = cfg_loss.get("lambda_s", 0.1)

    if warmup:
        total = lam_diff * L_diff + lam_s * L_smooth
        zero = torch.zeros((), device=device)
        return {
            "total": total, "L_diff": L_diff, "L_ellipse": zero,
            "L_param": zero, "L_iou": zero, "L_ecol": zero, "L_ecol_raw": zero,
            "L_anchor": zero, "L_col": zero, "L_smooth": L_smooth,
        }

    sdf_b = sdf_map.expand(x0_pred.shape[0], -1, -1, -1).contiguous()
    L_col = l_collision(x0_pred, sdf_b, extent,
                        margin=cfg_loss.get("collision_margin", 0.0),
                        sigma=cfg_loss.get("collision_sigma", 0.1),
                        state_norm=state_norm)

    el = ellipse_loss(
        pred_center=model_out["ellipse_center"],
        pred_r1=model_out["ellipse_radii"][..., 0],
        pred_r2=model_out["ellipse_radii"][..., 1],
        pred_theta=model_out["ellipse_theta"],
        pred_dir=model_out["ellipse_dir"],
        gt_center=gt_center, gt_r=gt_r, gt_Q=gt_Q, gt_valid=gt_valid,
        x0=x0, map_tensor=map_tensor.expand(x0_pred.shape[0], -1, -1, -1).contiguous(),
        extent=extent, state_norm=state_norm, cfg_loss=cfg_loss, cfg_geom=cfg_geom)

    lam_e = cfg_loss.get("lambda_e", 1.0)
    lam_col = cfg_loss.get("lambda_col", 1.0)
    lam_param = cfg_loss.get("lambda_param", 1.0)
    lam_iou = cfg_loss.get("lambda_iou", 1.0)
    lam_ecol = cfg_loss.get("lambda_ecol", 1.0)
    lam_anchor = cfg_loss.get("lambda_anchor", 1.0)

    L_ellipse = (lam_param * el["L_param"] + lam_iou * el["L_iou"]
                 + lam_ecol * el["L_ecol_raw"] + lam_anchor * el["L_anchor"])
    total = lam_diff * L_diff + lam_e * L_ellipse + lam_col * L_col + lam_s * L_smooth
    return {
        "total": total, "L_diff": L_diff, "L_ellipse": L_ellipse,
        "L_param": el["L_param"], "L_iou": el["L_iou"], "L_ecol": el["L_ecol"],
        "L_ecol_raw": el["L_ecol_raw"],
        "L_anchor": el["L_anchor"], "L_col": L_col, "L_smooth": L_smooth,
    }
