"""Train the V1 joint trajectory–ellipse diffusion transformer.

docs/联合扩散.md #26-#27:
  * P and E are diffused with independent Gaussian noise but the SAME alpha_bar
    schedule; model f(P_t,E_t,M,s,g,t) -> (eps_P_hat, eps_E_hat).
  * Loss L = L_P + lambda_e L_E + lambda_smooth L_smooth
             + lambda_safe L_safe.
    Endpoint slots of P are hard-conditioned inputs, so they are excluded from
    L_P. L_smooth matches predicted and target second differences. L_safe is
    the occupancy ratio inside each predicted soft ellipse mask.
  * Hard endpoints: after noising, P's first/last waypoint are overwritten with
    the exact scene start/goal (matches the sampler's inpainting convention).

Usage:
  python train.py --config configs/config_v1.yaml
  python train.py --config configs/config_v1.yaml --epochs 2 --resume outputs/ckpt_v1/epoch_10.pt
"""
import argparse
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


def _broadcast(x0, v):
    """v [B] -> broadcastable over x0 [B, ...]."""
    return v.reshape(v.shape[0], *([1] * (x0.dim() - 1)))


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


def trajectory_smoothness_loss(p_pred, p_gt):
    """Match trajectory second differences without flattening genuine turns."""
    d2_pred = p_pred[:, 2:] - 2.0 * p_pred[:, 1:-1] + p_pred[:, :-2]
    d2_gt = p_gt[:, 2:] - 2.0 * p_gt[:, 1:-1] + p_gt[:, :-2]
    return F.smooth_l1_loss(d2_pred, d2_gt)


def ellipse_safety_loss(p_pred, e_pred, occ, raster_res=64, tau=10.0,
                        chunk_size=32, eps=1e-6):
    """Mean obstacle-overlap ratio of differentiable predicted ellipse masks.

    The map is conservatively max-pooled to ``raster_res`` and every ellipse is
    evaluated against the full raster. Chunking only limits peak memory; it does
    not sample ellipse points. Gradients flow through both ``p_pred`` and every
    ellipse parameter, including the trajectory-relative centre offset.
    """
    if occ.dim() == 3:
        occ = occ.unsqueeze(1)
    if occ.dim() != 4 or occ.shape[1] != 1:
        raise ValueError(f"occ must have shape [B,1,H,W] or [B,H,W], got {tuple(occ.shape)}")
    if raster_res <= 0:
        raise ValueError(f"raster_res must be positive, got {raster_res}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    center = p_pred + e_pred[..., :2]
    a = torch.exp(torch.clamp(e_pred[..., 2], -6.0, 0.7))
    b = torch.exp(torch.clamp(e_pred[..., 3], -6.0, 0.7))
    theta = 0.5 * torch.atan2(e_pred[..., 5], e_pred[..., 4])

    occ_r = F.adaptive_max_pool2d(occ, output_size=(raster_res, raster_res))
    coord = ((torch.arange(raster_res, device=occ.device, dtype=occ.dtype) + 0.5)
             * (2.0 / raster_res) - 1.0)
    gy, gx = torch.meshgrid(coord, coord, indexing="ij")
    gx = gx[None, None]
    gy = gy[None, None]
    obstacle = occ_r[:, 0, None]

    ratio_sum = p_pred.new_zeros(())
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

        intersection = (mask * obstacle).sum(dim=(-1, -2))
        ellipse_area = mask.sum(dim=(-1, -2))
        ratio_sum = ratio_sum + (intersection / (ellipse_area + eps)).sum()

    return ratio_sum / (p_pred.shape[0] * horizon)


def batch_losses(batch, model, schedule, lambda_e, lambda_smooth, lambda_safe,
                 safe_res, safe_tau, safe_chunk, device):
    """One batch -> four component losses and their weighted total."""
    p0 = batch["pos"].to(device)
    e0 = batch["e6"].to(device)
    cond = batch["cond"].to(device)
    occ = batch["map_tensor"].to(device)
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
    loss_e = F.mse_loss(e_hat, e0)
    loss_smooth = trajectory_smoothness_loss(p_hat, p0)
    loss_safe = ellipse_safety_loss(
        p_hat, e_hat, occ, raster_res=safe_res, tau=safe_tau,
        chunk_size=safe_chunk,
    )
    total = (loss_p + lambda_e * loss_e
             + lambda_smooth * loss_smooth + lambda_safe * loss_safe)
    return loss_p, loss_e, loss_smooth, loss_safe, total


def validate(model, schedule, val_loader, lambda_e, lambda_smooth, lambda_safe,
             safe_res, safe_tau, safe_chunk, device, max_batches):
    model.eval()
    s_p = s_e = s_smooth = s_safe = n = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            lp, le, ls, lsafe, _ = batch_losses(
                batch, model, schedule, lambda_e, lambda_smooth, lambda_safe,
                safe_res, safe_tau, safe_chunk, device,
            )
            s_p += float(lp)
            s_e += float(le)
            s_smooth += float(ls)
            s_safe += float(lsafe)
            n += 1.0
    model.train()
    denom = max(n, 1.0)
    return s_p / denom, s_e / denom, s_smooth / denom, s_safe / denom


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
    lambda_safe = float(loss_cfg.get("lambda_safe", 1.0))
    safe_res = int(loss_cfg.get("ellipse_safe_res", 64))
    safe_tau = float(loss_cfg.get("ellipse_mask_tau", 10.0))
    safe_chunk = int(loss_cfg.get("ellipse_safe_chunk", 32))
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
        ep_lp = ep_le = ep_ls = ep_lsafe = 0.0
        n_steps = 0
        for step, batch in enumerate(train_loader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            lp, le, ls, lsafe, loss = batch_losses(
                batch, model, schedule, lambda_e, lambda_smooth, lambda_safe,
                safe_res, safe_tau, safe_chunk, device,
            )
            optim.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            ep_lp += float(lp.detach())
            ep_le += float(le.detach())
            ep_ls += float(ls.detach())
            ep_lsafe += float(lsafe.detach())
            n_steps += 1
            if (step + 1) % log_interval == 0:
                el = time.time() - t_start
                lp_value = float(lp.detach())
                le_value = float(le.detach())
                ls_value = float(ls.detach())
                lsafe_value = float(lsafe.detach())
                loss_value = float(loss.detach())
                print(f"[e{epoch} s{step + 1}/{len(train_loader)}] "
                      f"Lp={lp_value:.4f} Le={le_value:.4f} "
                      f"Ls={ls_value:.4f} Lsafe={lsafe_value:.4f} "
                      f"wLs={lambda_smooth * ls_value:.4f} "
                      f"wLsafe={lambda_safe * lsafe_value:.4f} "
                      f"L={loss_value:.4f} "
                      f"t={el:.0f}s", flush=True)

        denom = max(n_steps, 1)
        avg_lp, avg_le = ep_lp / denom, ep_le / denom
        avg_ls, avg_lsafe = ep_ls / denom, ep_lsafe / denom
        avg_total = (avg_lp + lambda_e * avg_le
                     + lambda_smooth * avg_ls + lambda_safe * avg_lsafe)
        line = (f"[epoch {epoch}/{epochs}] train Lp={avg_lp:.4f} Le={avg_le:.4f} "
                f"Ls={avg_ls:.4f} Lsafe={avg_lsafe:.4f} "
                f"wLs={lambda_smooth * avg_ls:.4f} "
                f"wLsafe={lambda_safe * avg_lsafe:.4f} L={avg_total:.4f}")
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            vp, ve, vs, vsafe = validate(
                model, schedule, val_loader, lambda_e, lambda_smooth,
                lambda_safe, safe_res, safe_tau, safe_chunk, device,
                int(train_cfg.get("val_batches", 20)),
            )
            vtot = (vp + lambda_e * ve
                    + lambda_smooth * vs + lambda_safe * vsafe)
            line += (f" | val Lp={vp:.4f} Le={ve:.4f} Ls={vs:.4f} "
                     f"Lsafe={vsafe:.4f} wLs={lambda_smooth * vs:.4f} "
                     f"wLsafe={lambda_safe * vsafe:.4f} L={vtot:.4f}")
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
