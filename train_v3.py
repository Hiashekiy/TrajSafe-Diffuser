"""Train V3.

    L = lam_traj L_traj + lam_coarse L_coarse + lam_smooth L_smooth
      + lam_topo L_topo + lam_align L_align + lam_shape L_shape
      + lam_iou L_iou + lam_safe L_safe

``L_align`` compares the decoded ellipse centre c_i = Gamma(s_i) directly with
the GT trajectory p_i^GT (SmoothL1).  There is no GT skeleton projection, no s*,
no ellipse_center_gt and no progress_gt in the training dependency chain.
``L_shape`` regulates only the 4 ellipse parameters.

One forward per batch runs the WHOLE report pipeline.  Training routing uses
m* = argmin_m nDTW(P0, S_m) (the cached topology_best label); inference routing
uses argmax(pi).  There is no predicted-skeleton warm-up schedule, because the
report defines exactly these two routing modes.  Only the trajectory is a
diffusion state.

Usage:
  python train_v3.py --config configs/config_v3_skeleton.yaml
  python train_v3.py --config configs/config_v3_skeleton.yaml --epochs 1 --max-batches 3
"""
import argparse
import os
import sys
import time

import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.diffusion.schedule import NoiseSchedule
from src.models.skeleton_v3 import SkeletonPlannerV3
from src.datasets.skeleton_dataset_v3 import make_loader
from src.losses.v3_losses import (center_alignment_loss, ellipse_iou_loss,
                                  ellipse_safety_loss, ellipse_shape_loss,
                                  topology_ce, trajectory_smoothness_loss,
                                  trajectory_x0_loss)

LOSS_KEYS = ["Ltraj", "Lcoarse", "Lsmooth", "Ltopo", "Lalign", "Lshape",
             "Liou", "Lsafe"]


def _broadcast(x0, v):
    return v.reshape(v.shape[0], *([1] * (x0.dim() - 1)))


def add_noise(x0, t, schedule):
    ab = schedule.sqrt_alphas_cumprod[t].to(x0.device).float()
    s1 = schedule.sqrt_one_minus_alphas_cumprod[t].to(x0.device).float()
    eps = torch.randn_like(x0)
    return _broadcast(x0, ab) * x0 + _broadcast(x0, s1) * eps, eps


def batch_losses(batch, model, schedule, lcfg, device):
    p0 = batch["pos"].to(device)
    cond = batch["cond"].to(device)
    occ = batch["map_tensor"].to(device)
    cand_xy = batch["candidate_xy"].to(device)
    cm = batch["candidate_mask"].to(device)
    geo = batch["candidate_geometry"].to(device)
    gl = batch["candidate_geometry_lengths"].to(device)
    best = batch["topology_best"].to(device)
    has_cand = batch["has_candidate"].to(device)
    shape_gt = batch["ellipse_shape4_gt"].to(device)
    shape_valid = batch["shape_valid"].to(device).bool()
    gt_mask = batch["ellipse_mask"].to(device)
    B = p0.shape[0]

    t = torch.randint(0, schedule.num_timesteps, (B,), device=device)
    p_t, _ = add_noise(p0, t, schedule)
    p_t = model.hard_endpoints(p_t, cond)
    ab = schedule.sqrt_alphas_cumprod[t].to(device)

    out = model.forward_all(p_t, occ, cond, t, ab, cand_xy, cm, geo, gl,
                            select_index=best)
    ell = out["ellipse"]

    l_traj = trajectory_x0_loss(out["final"], p0)
    l_coarse = trajectory_x0_loss(out["coarse"], p0)
    l_smooth = trajectory_smoothness_loss(
        out["final"], p0, acc_weight=float(lcfg.get("smooth_acc_weight", 0.25)),
        jerk_weight=float(lcfg.get("smooth_jerk_weight", 1.0)))
    l_topo = topology_ce(out["topo"]["pi"], best, has_cand)
    l_align = center_alignment_loss(ell["center"], p0, has_cand)
    l_shape = ellipse_shape_loss(ell["shape4"], shape_gt, shape_valid, has_cand)
    l_iou = ellipse_iou_loss(
        ell["center"], ell["a"], ell["b"], ell["theta"], gt_mask,
        shape_valid, has_cand,
        raster_res=int(lcfg.get("ellipse_safe_res", 64)),
        tau=float(lcfg.get("ellipse_mask_tau", 10.0)))
    l_safe, safe_mean, safe_cvar = ellipse_safety_loss(
        ell["center"], ell["a"], ell["b"], ell["theta"], occ,
        raster_res=int(lcfg.get("ellipse_safe_res", 64)),
        tau=float(lcfg.get("ellipse_mask_tau", 10.0)),
        chunk_size=int(lcfg.get("ellipse_safe_chunk", 32)),
        cvar_fraction=float(lcfg.get("safe_cvar_fraction", 0.2)),
        cvar_weight=float(lcfg.get("safe_cvar_weight", 1.0)),
        sample_mask=has_cand)

    raw = {"Ltraj": l_traj, "Lcoarse": l_coarse, "Lsmooth": l_smooth,
           "Ltopo": l_topo, "Lalign": l_align, "Lshape": l_shape,
           "Liou": l_iou, "Lsafe": l_safe}
    weights = {"Ltraj": float(lcfg.get("lambda_traj", 1.0)),
               "Lcoarse": float(lcfg.get("lambda_coarse", 0.25)),
               "Lsmooth": float(lcfg.get("lambda_smooth", 0.1)),
               "Ltopo": float(lcfg.get("lambda_topology", 0.5)),
               "Lalign": float(lcfg.get("lambda_align", 1.0 / 3.0)),
               "Lshape": float(lcfg.get("lambda_shape", 2.0 / 3.0)),
               "Liou": float(lcfg.get("lambda_iou", 0.25)),
               "Lsafe": float(lcfg.get("lambda_safe", 0.1))}
    total = sum(weights[k] * raw[k] for k in LOSS_KEYS)
    valid_sel = has_cand
    stats = {
        "sel_best_rate": float((out["selected_idx"] == best)[valid_sel].float().mean())
        if bool(valid_sel.any()) else 1.0,
        "a_mean": float(ell["a"].mean().detach()),
        "b_mean": float(ell["b"].mean().detach()),
        "safe_mean": float(safe_mean.detach()),
        "safe_cvar": float(safe_cvar.detach()),
    }
    return raw, weights, total, stats


def validate(model, schedule, loader, lcfg, device, max_batches):
    model.eval()
    acc, n = {}, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            raw, weights, total, _ = batch_losses(batch, model, schedule, lcfg,
                                                  device)
            for k in LOSS_KEYS:
                acc[k] = acc.get(k, 0.0) + float(raw[k].detach())
            acc["total"] = acc.get("total", 0.0) + float(total)
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v3_skeleton.yaml")
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    env, data_cfg = cfg["env"], cfg["data"]
    model_cfg, diff_cfg = cfg["model"], cfg["diffusion"]
    loss_cfg, train_cfg = cfg["loss"], cfg["train"]

    set_seed(int(env.get("seed", 42)))
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available()
               and env.get("device", "cuda") == "cuda" else "cpu"))
    print("[train_v3] device=%s" % device, flush=True)

    source = data_cfg.get("source", "data/processed_scene_v1")
    base_dir = data_cfg.get("base", "data/processed_scene_v3")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))
    mask_res = int(data_cfg.get("ellipse_mask_res", 64))
    mask_tau = float(data_cfg.get("ellipse_mask_tau", 10.0))
    mazes = data_cfg.get("mazes")

    train_loader, train_ds = make_loader(
        "train", source, base_dir, data_cfg.get("batch_size", 32), True,
        data_cfg.get("num_workers", 0), geometry_points=geo_points,
        ellipse_mask_res=mask_res, ellipse_mask_tau=mask_tau, mazes=mazes)
    if args.overfit:
        base_ds = train_loader.dataset
        train_ds = torch.utils.data.Subset(base_ds, list(range(args.overfit)))
        from src.datasets.skeleton_dataset_v3 import make_collate
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=min(data_cfg.get("batch_size", 32), args.overfit),
            shuffle=True, collate_fn=make_collate(base_ds))
    val_loader = val_ds = None
    if not args.no_val:
        val_loader, val_ds = make_loader(
            "val", source, base_dir, data_cfg.get("batch_size", 32), False, 0,
            geometry_points=geo_points, ellipse_mask_res=mask_res,
            ellipse_mask_tau=mask_tau, mazes=mazes)
    print("[data] train=%d val=%s geometry_points=%d"
          % (len(train_ds), len(val_ds) if val_ds else "-", geo_points), flush=True)

    schedule = NoiseSchedule(
        diff_cfg["timesteps"],
        beta_schedule=diff_cfg.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02)).to(device)
    model = SkeletonPlannerV3(model_cfg, cfg.get("ellipse_label")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] params=%.2fM traj_blocks=%d skeleton_blocks=%d final_blocks=%d"
          % (n_params / 1e6, model.traj_blocks, model.skeleton_blocks,
             model.final_blocks), flush=True)

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    log_interval = (args.log_interval if args.log_interval is not None
                    else int(train_cfg.get("log_interval", 20)))
    ckpt_dir = args.ckpt_dir or train_cfg.get("ckpt_dir", "outputs/ckpt_v3_skeleton")
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[ckpt] %s" % ckpt_dir, flush=True)

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
        acc, n_steps, wacc, gnorms = {}, 0, {}, []
        for step, batch in enumerate(train_loader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            raw, weights, total, stats = batch_losses(batch, model, schedule,
                                                      loss_cfg, device)
            optim.zero_grad()
            total.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip if grad_clip else 1e9))
            gnorms.append(gnorm)
            optim.step()
            for k in LOSS_KEYS:
                acc[k] = acc.get(k, 0.0) + float(raw[k].detach())
                wacc[k] = wacc.get(k, 0.0) + weights[k] * float(raw[k].detach())
            n_steps += 1
            if (step + 1) % log_interval == 0:
                avg_raw = {k: v / n_steps for k, v in acc.items()}
                avg_w = {k: v / n_steps for k, v in wacc.items()}
                print("[e%d s%d/%d] %s | %s | gnorm=%.3f t=%.0fs"
                      % (epoch, step + 1, len(train_loader),
                         " ".join("%s=%.4f" % (k, avg_raw[k]) for k in LOSS_KEYS),
                         " ".join("w%s=%.4f" % (k, avg_w[k]) for k in LOSS_KEYS),
                         sum(gnorms) / len(gnorms), time.time() - t0), flush=True)
        avg_raw = {k: v / max(n_steps, 1) for k, v in acc.items()}
        avg_w = {k: v / max(n_steps, 1) for k, v in wacc.items()}
        line = ("[epoch %d/%d] gnorm=%.3f " % (epoch, epochs,
                                               sum(gnorms) / max(len(gnorms), 1))
                + " ".join("%s=%.4f" % (k, avg_raw[k]) for k in LOSS_KEYS)
                + " | " + " ".join("w%s=%.4f" % (k, avg_w[k]) for k in LOSS_KEYS))
        if val_loader is not None and ((epoch + 1) % eval_every == 0
                                       or epoch == epochs - 1):
            v = validate(model, schedule, val_loader, loss_cfg, device,
                         int(train_cfg.get("val_batches", 25)))
            line += " | val " + " ".join("%s=%.4f" % (k, v[k]) for k in LOSS_KEYS)
            if v["total"] < best_val:
                best_val = v["total"]
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optim,
                                epoch, cfg)
                line += " (best)"
        print(line, flush=True)
        save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), model, optim,
                        epoch, cfg)
        if save_every and (epoch + 1) % save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, "epoch_%d.pt" % (epoch + 1)),
                            model, optim, epoch, cfg)
        if args.max_hours and (time.time() - t0) / 3600.0 >= args.max_hours:
            print("[stop] wall-clock budget %.2fh reached after epoch %d"
                  % (args.max_hours, epoch), flush=True)
            break
    print("[done] best val L=%.4f ckpt=%s" % (best_val, ckpt_dir), flush=True)


if __name__ == "__main__":
    main()
