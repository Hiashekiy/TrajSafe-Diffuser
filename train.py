"""Train the V1 joint trajectory–ellipse diffusion transformer.

docs/联合扩散.md #26-#27:
  * P and E are diffused with independent Gaussian noise but the SAME alpha_bar
    schedule; model f(P_t,E_t,M,s,g,t) -> (eps_P_hat, eps_E_hat).
  * Loss L = L_P + lambda_e L_E + lambda_smooth L_smooth
             + lambda_iou L_iou + lambda_safe L_safe.
    Endpoint slots of P are hard-conditioned inputs, so they are excluded from
    L_P. L_smooth directly regularizes normalized acceleration and jerk. L_iou
    matches precomputed GT soft masks. L_safe combines the mean unsafe fraction
    with the CVaR of the worst anchors within each trajectory.
  * Hard endpoints: after noising, P's first/last waypoint are overwritten with
    the exact scene start/goal (matches the sampler's inpainting convention).

Usage:
  python train.py --config configs/config_v1.yaml
  python train.py --config configs/config_v1.yaml --epochs 2 --resume outputs/ckpt_v1/epoch_10.pt
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.diffusion.schedule import NoiseSchedule
from src.models.joint import JointPlanner
from src.datasets.joint_dataset import make_loader
from src.geometry.ellipse_utils import physical_ellipse_center
from src.geometry.scene_frame import sample_sdf_scene


def _broadcast(x0, v):
    """v [B] -> broadcastable over x0 [B, ...]."""
    return v.reshape(v.shape[0], *([1] * (x0.dim() - 1)))


def _loss_grad_ratio(loss_a, loss_b, param):
    """||grad loss_a|| / ||grad loss_b|| on a shared param (balance probe)."""
    ga = torch.autograd.grad(loss_a, param, retain_graph=True)[0]
    gb = torch.autograd.grad(loss_b, param, retain_graph=True)[0]
    return float((ga.norm() / (gb.norm() + 1e-8)).item())


def add_noise(x0, t, schedule):
    """x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps.  Returns (x_t float32, eps)."""
    dev = x0.device
    ab = schedule.sqrt_alphas_cumprod[t].to(dev).float()
    s1 = schedule.sqrt_one_minus_alphas_cumprod[t].to(dev).float()
    eps = torch.randn_like(x0)
    x_t = _broadcast(x0, ab) * x0 + _broadcast(x0, s1) * eps
    return x_t, eps


def hard_endpoints(p_t, cond):
    """Overwrite first/last waypoints with exact start/goal (scene)."""
    p_t = p_t.clone()
    p_t[:, 0] = cond[:, 0]
    p_t[:, -1] = cond[:, 1]
    return p_t


def trajectory_smoothness_loss(p_pred, p_gt, acc_weight=0.25,
                               jerk_weight=1.0, eps=1e-3):
    """Penalize geometric acceleration and high-frequency trajectory jerk.

    Unlike matching the target second difference, this does not reproduce
    local noise in the demonstration. The target is used only to establish a
    per-sample step-length scale, detached from autograd. A weaker acceleration
    term promotes smooth curvature, while the stronger third-difference term
    specifically suppresses alternating, high-frequency bends.
    """
    velocity = p_pred[:, 1:] - p_pred[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]

    gt_velocity = p_gt[:, 1:] - p_gt[:, :-1]
    step_scale = gt_velocity.norm(dim=-1).mean(dim=1, keepdim=True)
    step_scale = step_scale.detach().clamp_min(1e-4)[:, :, None]
    acceleration = acceleration / step_scale
    jerk = jerk / step_scale

    # Charbonnier vector norm keeps a useful response to small zigzags without
    # a singular derivative at zero. log1p makes the auxiliary loss robust to
    # the very large, random x0 predictions seen early in training.
    acc_norm = (acceleration.square().sum(dim=-1) + eps ** 2).sqrt().sub(eps)
    jerk_norm = (jerk.square().sum(dim=-1) + eps ** 2).sqrt().sub(eps)
    loss_acc = torch.log1p(acc_norm).mean()
    loss_jerk = torch.log1p(jerk_norm).mean()
    return acc_weight * loss_acc + jerk_weight * loss_jerk


def ellipse_regression_loss(p_pred, p_gt, e_pred, e_gt, absolute=False):
    """Regress the physical ellipse centre and the remaining E6 parameters.

    E6 stores the centre either as an offset from its trajectory anchor
    (default) or as an absolute scene coordinate (``absolute=True``).  In the
    offset form, regressing that offset directly conflicts with mask
    supervision whenever ``p_pred`` differs from ``p_gt``; compare absolute
    centres instead, using the detached predicted trajectory anchor exactly as
    the mask losses do.  In the absolute form the physical centre is simply
    ``e6[..., :2]``.  The 2/6 and 4/6 weights preserve the scale of the former
    six-component mean squared error.
    """
    center_pred = physical_ellipse_center(p_pred.detach(), e_pred, absolute)
    center_gt = physical_ellipse_center(p_gt, e_gt, absolute)
    loss_center = F.mse_loss(center_pred, center_gt)
    loss_shape = F.mse_loss(e_pred[..., 2:], e_gt[..., 2:])
    return (2.0 * loss_center + 4.0 * loss_shape) / 6.0


def ellipse_mask_losses(p_pred, e_pred, occ, gt_mask, raster_res=64,
                        tau=10.0, chunk_size=32, safe_cvar_fraction=0.2,
                        safe_cvar_weight=1.0, eps=1e-6, absolute=False):
    """Return IoU and mean-plus-CVaR full-ellipse safety losses.

    The map is conservatively max-pooled to ``raster_res`` and every ellipse is
    evaluated against the full raster. Chunking only limits peak memory; it does
    not sample ellipse points. The precomputed GT mask is a constant uint8
    tensor. The trajectory anchor is detached, so neither mask loss has a
    direct gradient path to the predicted trajectory.

    Safety is one minus the fraction of the ellipse's *complete theoretical
    soft area* that lies in valid free raster cells. Consequently obstacle area
    and area outside [-1,1]^2 are both unsafe; an off-map or vanishing raster
    mask can no longer obtain zero loss.
    """
    if occ.dim() == 3:
        occ = occ.unsqueeze(1)
    if occ.dim() != 4 or occ.shape[1] != 1:
        raise ValueError(f"occ must have shape [B,1,H,W] or [B,H,W], got {tuple(occ.shape)}")
    if raster_res <= 0:
        raise ValueError(f"raster_res must be positive, got {raster_res}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if not 0.0 < safe_cvar_fraction <= 1.0:
        raise ValueError(
            f"safe_cvar_fraction must be in (0, 1], got {safe_cvar_fraction}"
        )
    if safe_cvar_weight < 0.0:
        raise ValueError(
            f"safe_cvar_weight must be non-negative, got {safe_cvar_weight}"
        )
    if gt_mask.dim() != 4 or gt_mask.shape[:2] != p_pred.shape[:2]:
        raise ValueError(
            f"gt_mask must have shape [B,H,R,R], got {tuple(gt_mask.shape)}"
        )
    if gt_mask.shape[-2:] != (raster_res, raster_res):
        raise ValueError(
            f"gt_mask resolution {tuple(gt_mask.shape[-2:])} != {raster_res}"
        )

    # Do not let ellipse collision avoidance drag the trajectory away from its
    # supervised path. The relative centre offset (or the absolute centre) and
    # all other ellipse parameters remain differentiable.
    center = physical_ellipse_center(p_pred.detach(), e_pred, absolute)
    a = torch.exp(torch.clamp(e_pred[..., 2], -6.0, 0.7))
    b = torch.exp(torch.clamp(e_pred[..., 3], -6.0, 0.7))
    theta = 0.5 * torch.atan2(e_pred[..., 5], e_pred[..., 4])

    occ_r = F.adaptive_max_pool2d(occ, output_size=(raster_res, raster_res))
    coord = ((torch.arange(raster_res, device=occ.device, dtype=occ.dtype) + 0.5)
             * (2.0 / raster_res) - 1.0)
    gy, gx = torch.meshgrid(coord, coord, indexing="ij")
    gx = gx[None, None]
    gy = gy[None, None]
    free = 1.0 - occ_r[:, 0, None]

    # Integral over R^2 of sigmoid(tau * (1 - q)) is
    # pi*a*b*softplus(tau)/tau. Convert that scene area to raster-cell units so
    # it is directly comparable to sums over the mask.
    cell_area = (2.0 / raster_res) ** 2
    tau_t = torch.as_tensor(tau, device=occ.device, dtype=occ.dtype)
    soft_area_factor = F.softplus(tau_t) / tau_t

    iou_sum = p_pred.new_zeros(())
    unsafe_chunks = []
    horizon = p_pred.shape[1]
    for start in range(0, horizon, chunk_size):
        end = min(start + chunk_size, horizon)
        cx = center[:, start:end, 0, None, None]
        cy = center[:, start:end, 1, None, None]
        dx = gx - cx
        dy = gy - cy

        ct = torch.cos(theta[:, start:end])[:, :, None, None]
        st = torch.sin(theta[:, start:end])[:, :, None, None]
        xr = ct * dx + st * dy
        yr = -st * dx + ct * dy
        ac = a[:, start:end, None, None]
        bc = b[:, start:end, None, None]
        q = (xr / ac).square() + (yr / bc).square()
        mask = torch.sigmoid(tau * (1.0 - q))

        # GT masks were rasterized offline with the same grid/tau and quantized
        # to uint8. detach() documents and enforces their target-only role.
        target_mask = (gt_mask[:, start:end].to(mask.dtype) / 255.0).detach()
        # Fuzzy-set IoU. min/max gives IoU=1 for identical soft masks, unlike
        # product-based "soft IoU", whose self-overlap is below one wherever
        # boundary pixels are fractional.
        intersection = torch.minimum(mask, target_mask).sum(dim=(-1, -2))
        union = torch.maximum(mask, target_mask).sum(dim=(-1, -2))
        iou_sum = iou_sum + (1.0 - (intersection + eps) / (union + eps)).sum()

        free_cells = (mask * free).sum(dim=(-1, -2))
        full_soft_cells = (
            torch.pi * a[:, start:end] * b[:, start:end]
            * soft_area_factor / cell_area
        )
        free_ratio = (free_cells / (full_soft_cells + eps)).clamp(0.0, 1.0)
        unsafe_chunks.append(1.0 - free_ratio)

    denom = p_pred.shape[0] * horizon
    unsafe_per_anchor = torch.cat(unsafe_chunks, dim=1)
    loss_safe_mean = unsafe_per_anchor.mean()
    # CVaR is computed independently for each trajectory so a difficult sample
    # cannot be hidden by safer samples elsewhere in the batch.  Using a tail
    # mean instead of a hard maximum keeps gradients distributed and stable.
    tail_count = max(1, math.ceil(horizon * safe_cvar_fraction))
    loss_safe_cvar = torch.topk(
        unsafe_per_anchor, k=tail_count, dim=1, largest=True, sorted=False
    ).values.mean()
    loss_safe = loss_safe_mean + safe_cvar_weight * loss_safe_cvar
    return iou_sum / denom, loss_safe, loss_safe_mean, loss_safe_cvar


def ellipse_center_safety_loss(p_pred, e_pred, sdf, margin=0.02,
                               cvar_fraction=0.2, cvar_weight=1.0):
    """Abs-centre safety: penalise ellipse centres inside walls / off-map.

    The physical centre is ``e_pred[..., :2]`` (absolute representation),
    independent of the trajectory, so the gradient flows only to the ellipse
    branch.  Keep ``lambda_center_safe`` small so this auxiliary term does not
    unbalance the joint P/E training.  A log-hinge bounds early outliers and a
    CVaR tail focuses on the worst anchors, so the term only performs local
    corrections instead of distorting the whole trajectory.
    ``sdf`` must be [B,1,H,W] in scene units (free > 0).
    """
    center = e_pred[..., :2]

    def _penalty(clearance):
        violation = torch.relu(float(margin) - clearance)
        return torch.log1p(violation / float(margin))

    def _summarise(penalty):
        mean = penalty.mean()
        tail = max(1, math.ceil(penalty.shape[1] * cvar_fraction))
        cvar = torch.topk(penalty, k=tail, dim=1, largest=True,
                          sorted=False).values.mean()
        return mean + float(cvar_weight) * cvar, mean, cvar

    clearance = sample_sdf_scene(sdf, center)
    boundary = 1.0 - center.abs().amax(dim=-1)
    penalty = _penalty(torch.minimum(clearance, boundary))
    loss, mean, cvar = _summarise(penalty)
    return loss, mean, cvar



def batch_losses(batch, model, schedule, lambda_e, lambda_smooth,
                 lambda_iou, lambda_safe,
                 smooth_acc_weight, smooth_jerk_weight,
                 safe_res, safe_tau, safe_chunk, safe_cvar_fraction,
                 safe_cvar_weight, device, absolute=False,
                 lambda_center_safe=0.0, center_safe_margin=0.02,
                 center_safe_cvar_fraction=0.2, center_safe_cvar_weight=1.0):
    """One batch -> component losses and their weighted total."""
    p0 = batch["pos"].to(device)
    e0 = batch["e6"].to(device)
    cond = batch["cond"].to(device)
    occ = batch["map_tensor"].to(device)
    gt_mask = batch["ellipse_mask"].to(device)
    B = p0.shape[0]
    t = torch.randint(0, schedule.num_timesteps, (B,), device=device)

    p_t, _ = add_noise(p0, t, schedule)
    p_t = hard_endpoints(p_t, cond)
    e_t, _ = add_noise(e0, t, schedule)

    ab = schedule.sqrt_alphas_cumprod[t].to(device)
    out = model(p_t, e_t, occ, cond, t, ab)

    # Preserve the original regression losses exactly. Only the auxiliary
    # losses use the trajectory as it will actually appear during sampling.
    p_hat_raw = out["x0_p"]
    e_hat = out["x0_e"]
    p_hat = hard_endpoints(p_hat_raw, cond)
    loss_p = F.mse_loss(p_hat_raw[:, 1:-1], p0[:, 1:-1])
    loss_e = ellipse_regression_loss(p_hat, p0, e_hat, e0, absolute=absolute)
    loss_smooth = trajectory_smoothness_loss(
        p_hat, p0, acc_weight=smooth_acc_weight,
        jerk_weight=smooth_jerk_weight,
    )
    loss_iou, loss_safe, loss_safe_mean, loss_safe_cvar = ellipse_mask_losses(
        p_hat, e_hat, occ, gt_mask, raster_res=safe_res, tau=safe_tau,
        chunk_size=safe_chunk, safe_cvar_fraction=safe_cvar_fraction,
        safe_cvar_weight=safe_cvar_weight, absolute=absolute,
    )
    # Absolute-centre safety: auxiliary, weighted to keep P/E balanced.
    if absolute:
        sdf = batch["sdf_tensor"].to(device)
        loss_center, center_mean, center_cvar = ellipse_center_safety_loss(
            p_hat, e_hat, sdf, margin=center_safe_margin,
            cvar_fraction=center_safe_cvar_fraction,
            cvar_weight=center_safe_cvar_weight,
        )
    else:
        zero = p_hat.new_zeros(())
        loss_center, center_mean, center_cvar = zero, zero, zero
    total = (loss_p + lambda_e * loss_e
             + lambda_smooth * loss_smooth + lambda_iou * loss_iou
             + lambda_safe * loss_safe
             + lambda_center_safe * loss_center)
    return (loss_p, loss_e, loss_smooth, loss_iou, loss_safe,
            loss_safe_mean, loss_safe_cvar,
            loss_center, center_mean, center_cvar, total)


def validate(model, schedule, val_loader, lambda_e, lambda_smooth,
             lambda_iou, lambda_safe,
             smooth_acc_weight, smooth_jerk_weight,
             safe_res, safe_tau, safe_chunk, safe_cvar_fraction,
             safe_cvar_weight, device, max_batches, absolute=False,
             lambda_center_safe=0.0, center_safe_margin=0.02,
             center_safe_cvar_fraction=0.2, center_safe_cvar_weight=1.0):
    model.eval()
    s_p = s_e = s_smooth = s_iou = 0.0
    s_safe = s_safe_mean = s_safe_cvar = n = 0.0
    s_center = s_center_mean = s_center_cvar = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            (lp, le, ls, liou, lsafe, lsafe_mean, lsafe_cvar,
             lcenter, lcenter_mean, lcenter_cvar, _) = batch_losses(
                batch, model, schedule, lambda_e, lambda_smooth,
                lambda_iou, lambda_safe, smooth_acc_weight,
                smooth_jerk_weight, safe_res, safe_tau, safe_chunk,
                safe_cvar_fraction, safe_cvar_weight, device, absolute,
                lambda_center_safe, center_safe_margin,
                center_safe_cvar_fraction, center_safe_cvar_weight,
            )
            s_p += float(lp)
            s_e += float(le)
            s_smooth += float(ls)
            s_iou += float(liou)
            s_safe += float(lsafe)
            s_safe_mean += float(lsafe_mean)
            s_safe_cvar += float(lsafe_cvar)
            s_center += float(lcenter)
            s_center_mean += float(lcenter_mean)
            s_center_cvar += float(lcenter_cvar)
            n += 1.0
    model.train()
    denom = max(n, 1.0)
    return (s_p / denom, s_e / denom, s_smooth / denom,
            s_iou / denom, s_safe / denom, s_safe_mean / denom,
            s_safe_cvar / denom, s_center / denom, s_center_mean / denom,
            s_center_cvar / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v1.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap steps per epoch (smoke tests)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    env, data_cfg = cfg["env"], cfg["data"]
    center_absolute = data_cfg.get("ellipse_center_mode", "offset") == "absolute"
    model_cfg, diff_cfg = cfg["model"], cfg["diffusion"]
    loss_cfg, train_cfg = cfg["loss"], cfg["train"]

    set_seed(int(env["seed"]))
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() and env.get("device", "cuda") == "cuda" else "cpu"))
    print(f"[train] device={device}", flush=True)

    base = data_cfg["base"]
    train_loader, train_ds = make_loader(os.path.join(base, "train"),
                                         data_cfg["batch_size"], True,
                                         data_cfg.get("num_workers", 0))
    val_loader, val_ds = make_loader(os.path.join(base, "val"),
                                     data_cfg["batch_size"], False, 0)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} (H={model_cfg['horizon']})", flush=True)

    schedule = NoiseSchedule(diff_cfg["timesteps"],
                             beta_schedule=diff_cfg.get("beta_schedule", "squaredcos_cap_v2"),
                             beta_start=diff_cfg.get("beta_start", 0.0001),
                             beta_end=diff_cfg.get("beta_end", 0.02)).to(device)
    model = JointPlanner(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params / 1e6:.2f}M", flush=True)

    lambda_e = float(loss_cfg.get("lambda_e", 1.0))
    lambda_smooth = float(loss_cfg.get("lambda_smooth", 0.1))
    lambda_iou = float(loss_cfg.get("lambda_iou", 0.5))
    lambda_safe = float(loss_cfg.get("lambda_safe", 0.2))
    smooth_acc_weight = float(loss_cfg.get("smooth_acc_weight", 0.25))
    smooth_jerk_weight = float(loss_cfg.get("smooth_jerk_weight", 1.0))
    safe_res = int(loss_cfg.get("ellipse_safe_res", 64))
    safe_tau = float(loss_cfg.get("ellipse_mask_tau", 10.0))
    safe_chunk = int(loss_cfg.get("ellipse_safe_chunk", 32))
    safe_cvar_fraction = float(loss_cfg.get("safe_cvar_fraction", 0.2))
    safe_cvar_weight = float(loss_cfg.get("safe_cvar_weight", 1.0))
    lambda_center_safe = float(loss_cfg.get("lambda_center_safe", 0.0))
    center_safe_margin = float(loss_cfg.get("ellipse_center_safe_margin", 0.02))
    center_safe_cvar_fraction = float(loss_cfg.get("center_safe_cvar_fraction", 0.2))
    center_safe_cvar_weight = float(loss_cfg.get("center_safe_cvar_weight", 1.0))
    for split_name, dataset in (("train", train_ds), ("val", val_ds)):
        if dataset.mask_res != safe_res or abs(dataset.mask_tau - safe_tau) > 1e-6:
            raise ValueError(
                f"{split_name} GT masks use res={dataset.mask_res}, tau={dataset.mask_tau}, "
                f"but config requests res={safe_res}, tau={safe_tau}; rerun "
                "scripts/data/09_precompute_gt_ellipse_masks.py"
            )
    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    log_interval = args.log_interval if args.log_interval is not None else int(train_cfg["log_interval"])
    ckpt_dir = train_cfg["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                              weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    start_epoch = 0
    if args.resume:
        ck = load_checkpoint(args.resume, model, optim, map_location=device)
        start_epoch = int(ck.get("epoch", 0)) + 1
        print(f"[resume] epoch {start_epoch} from {args.resume}", flush=True)

    grad_clip = float(train_cfg.get("grad_clip", 0.0)) or None
    eval_every = int(train_cfg.get("eval_every", epochs + 1))
    save_every = int(train_cfg.get("save_every", max(1, epochs // 10)))
    best_val = float("inf")

    t_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        ep_lp = ep_le = ep_ls = ep_liou = ep_lsafe = 0.0
        ep_lsafe_mean = ep_lsafe_cvar = 0.0
        ep_lcenter = ep_lcenter_mean = ep_lcenter_cvar = 0.0
        last_center_grad_ratio = float("nan")
        n_steps = 0
        for step, batch in enumerate(train_loader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            (lp, le, ls, liou, lsafe, lsafe_mean, lsafe_cvar,
             lcenter, lcenter_mean, lcenter_cvar, loss) = batch_losses(
                batch, model, schedule, lambda_e, lambda_smooth,
                lambda_iou, lambda_safe, smooth_acc_weight,
                smooth_jerk_weight, safe_res, safe_tau, safe_chunk,
                safe_cvar_fraction, safe_cvar_weight, device, center_absolute,
                lambda_center_safe, center_safe_margin,
                center_safe_cvar_fraction, center_safe_cvar_weight,
            )
            optim.zero_grad()
            if center_absolute and (step + 1) % log_interval == 0:
                last_center_grad_ratio = _loss_grad_ratio(
                    lcenter, le, model.head_e.weight)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            ep_lp += float(lp.detach())
            ep_le += float(le.detach())
            ep_ls += float(ls.detach())
            ep_liou += float(liou.detach())
            ep_lsafe += float(lsafe.detach())
            ep_lsafe_mean += float(lsafe_mean.detach())
            ep_lsafe_cvar += float(lsafe_cvar.detach())
            ep_lcenter += float(lcenter.detach())
            ep_lcenter_mean += float(lcenter_mean.detach())
            ep_lcenter_cvar += float(lcenter_cvar.detach())
            n_steps += 1
            if (step + 1) % log_interval == 0:
                el = time.time() - t_start
                lp_value = float(lp.detach())
                le_value = float(le.detach())
                ls_value = float(ls.detach())
                liou_value = float(liou.detach())
                lsafe_value = float(lsafe.detach())
                lsafe_mean_value = float(lsafe_mean.detach())
                lsafe_cvar_value = float(lsafe_cvar.detach())
                loss_value = float(loss.detach())
                print(f"[e{epoch} s{step + 1}/{len(train_loader)}] "
                      f"Lp={lp_value:.4f} Le={le_value:.4f} "
                      f"Ls={ls_value:.4f} Liou={liou_value:.4f} "
                      f"Lsafe={lsafe_value:.4f} "
                      f"LsafeMean={lsafe_mean_value:.4f} "
                      f"LsafeCVaR={lsafe_cvar_value:.4f} "
                      f"wLs={lambda_smooth * ls_value:.4f} "
                      f"wLiou={lambda_iou * liou_value:.4f} "
                      f"wLsafe={lambda_safe * lsafe_value:.4f} "
                      f"wLsafeMean={lambda_safe * lsafe_mean_value:.4f} "
                      f"wLsafeCVaR={lambda_safe * safe_cvar_weight * lsafe_cvar_value:.4f} "
                      f"L={loss_value:.4f} "
                      f"t={el:.0f}s", flush=True)

        denom = max(n_steps, 1)
        avg_lp, avg_le = ep_lp / denom, ep_le / denom
        avg_ls, avg_liou = ep_ls / denom, ep_liou / denom
        avg_lsafe = ep_lsafe / denom
        avg_lsafe_mean = ep_lsafe_mean / denom
        avg_lsafe_cvar = ep_lsafe_cvar / denom
        avg_lcenter = ep_lcenter / denom
        avg_lcenter_mean = ep_lcenter_mean / denom
        avg_lcenter_cvar = ep_lcenter_cvar / denom
        avg_total = (avg_lp + lambda_e * avg_le
                     + lambda_smooth * avg_ls + lambda_iou * avg_liou
                     + lambda_safe * avg_lsafe
                     + lambda_center_safe * avg_lcenter)
        line = (f"[epoch {epoch}/{epochs}] train Lp={avg_lp:.4f} Le={avg_le:.4f} "
                f"Ls={avg_ls:.4f} Liou={avg_liou:.4f} Lsafe={avg_lsafe:.4f} "
                f"LsafeMean={avg_lsafe_mean:.4f} "
                f"LsafeCVaR={avg_lsafe_cvar:.4f} "
                f"wLs={lambda_smooth * avg_ls:.4f} "
                f"wLiou={lambda_iou * avg_liou:.4f} "
                f"wLsafe={lambda_safe * avg_lsafe:.4f} "
                f"wLsafeMean={lambda_safe * avg_lsafe_mean:.4f} "
                f"wLsafeCVaR={lambda_safe * safe_cvar_weight * avg_lsafe_cvar:.4f} "
                f"Lcenter={avg_lcenter:.4f} "
                 f"wLcenter={lambda_center_safe * avg_lcenter:.4f} "
                 f"gradRatio={last_center_grad_ratio:.3f} "
                 f"L={avg_total:.4f}")
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            (vp, ve, vs, viou, vsafe, vsafe_mean, vsafe_cvar,
             vcenter, vcenter_mean, vcenter_cvar) = validate(
                model, schedule, val_loader, lambda_e, lambda_smooth,
                lambda_iou, lambda_safe, smooth_acc_weight,
                smooth_jerk_weight, safe_res, safe_tau, safe_chunk,
                safe_cvar_fraction, safe_cvar_weight, device, center_absolute,
                int(train_cfg.get("val_batches", 20)),
                lambda_center_safe, center_safe_margin,
                center_safe_cvar_fraction, center_safe_cvar_weight,
            )
            vtot = (vp + lambda_e * ve
                    + lambda_smooth * vs + lambda_iou * viou
                    + lambda_safe * vsafe
                    + lambda_center_safe * vcenter)
            line += (f" | val Lp={vp:.4f} Le={ve:.4f} Ls={vs:.4f} "
                     f"Liou={viou:.4f} Lsafe={vsafe:.4f} "
                     f"LsafeMean={vsafe_mean:.4f} "
                     f"LsafeCVaR={vsafe_cvar:.4f} "
                     f"wLs={lambda_smooth * vs:.4f} "
                     f"wLiou={lambda_iou * viou:.4f} "
                     f"wLsafe={lambda_safe * vsafe:.4f} "
                     f"wLsafeMean={lambda_safe * vsafe_mean:.4f} "
                     f"wLsafeCVaR={lambda_safe * safe_cvar_weight * vsafe_cvar:.4f} "
                     f"L={vtot:.4f}")
            if vtot < best_val:
                best_val = vtot
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model,
                                optim, epoch, cfg)
                line += " (best)"
        print(line, flush=True)
        save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), model, optim, epoch, cfg)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt"),
                            model, optim, epoch, cfg)

    print(f"[done] {epochs} epochs, best val L={best_val:.4f}, ckpt_dir={ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
