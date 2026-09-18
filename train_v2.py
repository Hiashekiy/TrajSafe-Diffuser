"""Train V2: Skeleton-Topology-Grounded Trajectory Diffusion.

    L = L_P + lam_s L_smooth + lam_T L_topo + lam_P L_prog
          + lam_E L_shape + lam_I L_iou + lam_S L_safe

Teacher forcing (docs/V2.md section 29): the topology selector is trained
against the soft target q_m, while progress / ellipse / trajectory refinement
always receive the GT-best candidate m* = argmin_m nDTW, so a wrong early
selector can never poison the other heads.  Scheduled sampling is deliberately
NOT part of this version.

Usage:
  python train_v2.py --config configs/config_v2_skeleton.yaml
  python train_v2.py --config configs/config_v2_skeleton.yaml --epochs 2 --max-batches 5
"""
import argparse
import os
import sys
import time

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.diffusion.schedule import NoiseSchedule
from src.models.skeleton import SkeletonPlanner
from src.datasets.skeleton_dataset import make_loader
from src.losses.v2_losses import (ellipse_mask_losses, ellipse_shape_loss,
                                  progress_loss, topology_soft_ce,
                                  trajectory_smoothness_loss,
                                  trajectory_x0_loss)


def _broadcast(x0, v):
    return v.reshape(v.shape[0], *([1] * (x0.dim() - 1)))


def add_noise(x0, t, schedule):
    """x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps (same as V1)."""
    ab = schedule.sqrt_alphas_cumprod[t].to(x0.device).float()
    s1 = schedule.sqrt_one_minus_alphas_cumprod[t].to(x0.device).float()
    eps = torch.randn_like(x0)
    return _broadcast(x0, ab) * x0 + _broadcast(x0, s1) * eps, eps


def hard_endpoints(p, cond):
    p = p.clone()
    p[:, 0] = cond[:, 0]
    p[:, -1] = cond[:, 1]
    return p


def batch_losses(batch, model, schedule, lcfg, device):
    p0 = batch["pos"].to(device)
    cond = batch["cond"].to(device)
    occ = batch["map_tensor"].to(device)
    cand = batch["candidate_paths"].to(device)
    cand_mask = batch["candidate_mask"].to(device)
    cand_len = batch["candidate_lengths"].to(device)
    q = batch["topology_target"].to(device)
    best = batch["topology_best"].to(device)
    prog_gt = batch["progress_gt"].to(device)
    shape_gt = batch["ellipse_shape4_gt"].to(device)
    shape_ok = batch["shape_valid"].to(device)
    gt_mask = batch["ellipse_mask"].to(device)
    has_cand = batch["has_candidate"].to(device)
    B = p0.shape[0]

    t = torch.randint(0, schedule.num_timesteps, (B,), device=device)
    p_t, _ = add_noise(p0, t, schedule)
    p_t = hard_endpoints(p_t, cond)
    ab = schedule.sqrt_alphas_cumprod[t].to(device)

    base = model.encode_trajectory(p_t, occ, cond, t, ab)
    topo = model.score_candidates(base, cand, cand_mask, cand_len)

    ar = torch.arange(B, device=device)
    sel_idx = torch.where(has_cand, best, torch.zeros_like(best))
    sel_path = cand[ar, sel_idx][..., :2].contiguous()
    sel_feat = topo["path_feat"][ar, sel_idx].contiguous()
    ref = model.refine_with_path(base, sel_path, sel_feat)

    x0_p = torch.where(has_cand[:, None, None], ref["x0_p"], base["x0_p_base"])
    x0_p = hard_endpoints(x0_p, cond)

    loss_p = trajectory_x0_loss(x0_p, p0)
    loss_smooth = trajectory_smoothness_loss(
        x0_p, p0, acc_weight=float(lcfg.get("smooth_acc_weight", 0.25)),
        jerk_weight=float(lcfg.get("smooth_jerk_weight", 1.0)))
    loss_topo = topology_soft_ce(topo["pi"], q, has_cand)
    loss_prog = progress_loss(ref["progress"], prog_gt, has_cand)
    loss_shape = ellipse_shape_loss(ref["ellipse_shape4"], shape_gt, shape_ok,
                                    has_cand)
    loss_iou, loss_safe, loss_safe_mean, loss_safe_cvar = ellipse_mask_losses(
        ref["ellipse_center"], ref["ellipse_shape4"], occ, gt_mask,
        raster_res=int(lcfg.get("ellipse_safe_res", 64)),
        tau=float(lcfg.get("ellipse_mask_tau", 10.0)),
        chunk_size=int(lcfg.get("ellipse_safe_chunk", 32)),
        safe_cvar_fraction=float(lcfg.get("safe_cvar_fraction", 0.2)),
        safe_cvar_weight=float(lcfg.get("safe_cvar_weight", 1.0)),
        sample_mask=has_cand)

    total = (float(lcfg.get("lambda_traj", 1.0)) * loss_p
             + float(lcfg.get("lambda_smooth", 0.1)) * loss_smooth
             + float(lcfg.get("lambda_topology", 1.0)) * loss_topo
             + float(lcfg.get("lambda_progress", 0.5)) * loss_prog
             + float(lcfg.get("lambda_shape", 1.0)) * loss_shape
             + float(lcfg.get("lambda_iou", 0.25)) * loss_iou
             + float(lcfg.get("lambda_safe", 0.1)) * loss_safe)
    parts = {"Lp": loss_p, "Lsmooth": loss_smooth, "Ltopo": loss_topo,
             "Lprog": loss_prog, "Lshape": loss_shape, "Liou": loss_iou,
             "Lsafe": loss_safe, "LsafeMean": loss_safe_mean,
             "LsafeCVaR": loss_safe_cvar, "total": total}
    return parts


def validate(model, schedule, loader, lcfg, device, max_batches):
    model.eval()
    acc = {}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            parts = batch_losses(batch, model, schedule, lcfg, device)
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach())
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def _fmt(parts, prefix=""):
    keys = ["Lp", "Lsmooth", "Ltopo", "Lprog", "Lshape", "Liou", "Lsafe", "total"]
    return prefix + " ".join("%s=%.4f" % (k, parts[k]) for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v2_skeleton.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--no-val", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env, data_cfg = cfg["env"], cfg["data"]
    model_cfg, diff_cfg = cfg["model"], cfg["diffusion"]
    loss_cfg, train_cfg = cfg["loss"], cfg["train"]

    set_seed(int(env.get("seed", 42)))
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available()
               and env.get("device", "cuda") == "cuda" else "cpu"))
    print("[train_v2] device=%s" % device, flush=True)

    source = data_cfg.get("source", "data/processed_scene_v1")
    base_dir = data_cfg.get("base", "data/processed_scene_v2")
    shape_dir = os.path.join(base_dir, "skeletons")
    mask_res = int(loss_cfg.get("ellipse_safe_res", 64))
    mask_tau = float(loss_cfg.get("ellipse_mask_tau", 10.0))

    train_loader, train_ds = make_loader(
        "train", source, base_dir, data_cfg.get("batch_size", 32), True,
        data_cfg.get("num_workers", 0), mask_res=mask_res, mask_tau=mask_tau,
        shape_dir=shape_dir)
    val_loader = val_ds = None
    if not args.no_val:
        val_loader, val_ds = make_loader(
            "val", source, base_dir, data_cfg.get("batch_size", 32), False, 0,
            mask_res=mask_res, mask_tau=mask_tau, shape_dir=shape_dir)
    print("[data] train=%d val=%s" % (len(train_ds),
                                      len(val_ds) if val_ds else "-"), flush=True)

    schedule = NoiseSchedule(
        diff_cfg["timesteps"],
        beta_schedule=diff_cfg.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02)).to(device)
    model = SkeletonPlanner(model_cfg, cfg.get("topology")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] params=%.2fM  traj_blocks=%d refine_blocks=%d"
          % (n_params / 1e6, model.traj_blocks, model.refine_blocks), flush=True)

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    log_interval = (args.log_interval if args.log_interval is not None
                    else int(train_cfg.get("log_interval", 20)))
    ckpt_dir = train_cfg.get("ckpt_dir", "outputs/ckpt_v2_skeleton")
    os.makedirs(ckpt_dir, exist_ok=True)

    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                              weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    start_epoch = 0
    if args.resume:
        ck = load_checkpoint(args.resume, model, optim, map_location=device)
        start_epoch = int(ck.get("epoch", 0)) + 1
        print("[resume] epoch %d" % start_epoch, flush=True)

    grad_clip = float(train_cfg.get("grad_clip", 0.0)) or None
    eval_every = int(train_cfg.get("eval_every", 10))
    save_every = int(train_cfg.get("save_every", 10))
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        acc, n_steps = {}, 0
        for step, batch in enumerate(train_loader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            parts = batch_losses(batch, model, schedule, loss_cfg, device)
            optim.zero_grad()
            parts["total"].backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach())
            n_steps += 1
            if (step + 1) % log_interval == 0:
                avg = {k: v / n_steps for k, v in acc.items()}
                print("[e%d s%d/%d] %s t=%.0fs"
                      % (epoch, step + 1, len(train_loader), _fmt(avg),
                         time.time() - t0), flush=True)
        avg = {k: v / max(n_steps, 1) for k, v in acc.items()}
        line = "[epoch %d/%d] train %s" % (epoch, epochs, _fmt(avg))
        if val_loader is not None and ((epoch + 1) % eval_every == 0
                                       or epoch == epochs - 1):
            vparts = validate(model, schedule, val_loader, loss_cfg, device,
                              int(train_cfg.get("val_batches", 25)))
            line += " | val " + _fmt(vparts)
            if vparts["total"] < best_val:
                best_val = vparts["total"]
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optim,
                                epoch, cfg)
                line += " (best)"
        print(line, flush=True)
        save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), model, optim,
                        epoch, cfg)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, "epoch_%d.pt" % (epoch + 1)),
                            model, optim, epoch, cfg)
    print("[done] best val L=%.4f ckpt=%s" % (best_val, ckpt_dir), flush=True)


if __name__ == "__main__":
    main()
