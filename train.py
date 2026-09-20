"""Train the control-space TrajSafe-Diffuser on the CARLA v1 snapshot.

    L = lam_traj   L_traj    (decoded 128-point curve  vs curve_gt)
      + lam_ctrl   L_ctrl    (30 interior B-spline controls vs control_gt)
      + lam_coarse L_coarse
      + lam_smooth L_smooth
      + lam_topo   L_topo
      + lam_shape  L_shape
      + lam_iou    L_iou
      + lam_safe   L_safe

The diffusion state is the 32-control polygon Q_t (``control_gt``).  The
network still runs on the decoded 128-point curve.  There is NO alignment loss,
no learned progress and no ellipse-centre label: the ellipse centre is the
fixed Skeleton centre Gamma_m(i/127).

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

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.diffusion.schedule import NoiseSchedule
from src.models.trajsafe import TrajSafePlanner
from src.datasets.carla_spline_dataset import make_loader, make_collate
from src.geometry.ellipse_raster import ellipse_soft_mask
from src.geometry.ellipse_shape import shape4_to_abtheta
from src.losses.losses import (ellipse_iou_loss, ellipse_safety_loss,
                               ellipse_shape_loss, topology_ce,
                               trajectory_smoothness_loss, trajectory_x0_loss,
                               control_x0_loss)

LOSS_KEYS = ["Ltraj", "Lctrl", "Lcoarse", "Lsmooth", "Ltopo", "Lshape",
             "Liou", "Lsafe"]
SCENE_TO_METER = 40.0


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
    cond = batch["cond"].to(device)
    best = batch["topology_best"].to(device)
    has_cand = batch["has_candidate"].to(device)
    ell = out["ellipse"]
    final = out["final"]
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
        q_err = torch.linalg.norm(out["control"] - batch["control_gt"].to(device),
                                  dim=-1)[:, 1:-1]
        res["ctrl_rmse_m"] = float(q_err.pow(2).mean().sqrt()) * SCENE_TO_METER
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


def batch_losses(batch, model, schedule, lcfg, device):
    q0 = batch["control_gt"].to(device)
    p_gt = batch["pos"].to(device)
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

    t = torch.randint(0, schedule.num_timesteps, (B,), device=device)
    q_t, _ = add_noise(q0, t, schedule)
    q_t = model.hard_control_endpoints(q_t, cond)
    ab = schedule.sqrt_alphas_cumprod[t].to(device)

    out = model.forward_all(q_t, occ, cond, t, ab, cand_xy, cm, geo, gl,
                            select_index=best)
    ell = out["ellipse"]

    l_traj = trajectory_x0_loss(out["final"], p_gt)
    l_ctrl = control_x0_loss(out["control"], q0)
    l_coarse = trajectory_x0_loss(out["coarse"], p_gt)
    l_smooth = trajectory_smoothness_loss(
        out["final"], p_gt, acc_weight=float(lcfg.get("smooth_acc_weight", 0.25)),
        jerk_weight=float(lcfg.get("smooth_jerk_weight", 1.0)))
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

    raw = {"Ltraj": l_traj, "Lctrl": l_ctrl, "Lcoarse": l_coarse,
           "Lsmooth": l_smooth, "Ltopo": l_topo, "Lshape": l_shape,
           "Liou": l_iou, "Lsafe": l_safe}
    weights = {"Ltraj": float(lcfg.get("lambda_traj", 1.0)),
               "Lctrl": float(lcfg.get("lambda_control", 0.2)),
               "Lcoarse": float(lcfg.get("lambda_coarse", 0.5)),
               "Lsmooth": float(lcfg.get("lambda_smooth", 0.08)),
               "Ltopo": float(lcfg.get("lambda_topology", 0.25)),
               "Lshape": float(lcfg.get("lambda_shape", 0.08)),
               "Liou": float(lcfg.get("lambda_iou", 0.25)),
               "Lsafe": float(lcfg.get("lambda_safe", 0.15))}
    total = sum(weights[k] * raw[k] for k in LOSS_KEYS)
    stats = {
        "safe_mean": float(safe_mean.detach()),
        "safe_cvar": float(safe_cvar.detach()),
        "a_mean": float(ell["a"].mean().detach()),
        "b_mean": float(ell["b"].mean().detach()),
    }
    return raw, weights, total, out, stats


def validate(model, schedule, loader, lcfg, device, max_batches):
    model.eval()
    acc, met, n = {}, {}, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            raw, weights, total, out, stats = batch_losses(
                batch, model, schedule, lcfg, device)
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

    if args.overfit:
        train_loader, train_ds = make_loader(
            "train", processed_root, batch_size=min(batch_size, args.overfit),
            shuffle=True, num_workers=num_workers, geometry_points=geo_points,
            limit=args.overfit)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=min(batch_size, args.overfit), shuffle=True,
            num_workers=num_workers, drop_last=False,
            collate_fn=make_collate(train_ds))
    else:
        train_loader, train_ds = make_loader(
            "train", processed_root, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, geometry_points=geo_points,
            limit=args.limit)
    val_loader = val_ds = None
    if not args.no_val:
        val_loader, val_ds = make_loader(
            "val", processed_root, batch_size=batch_size, shuffle=False,
            num_workers=0, geometry_points=geo_points, limit=args.limit)
    print("[data] train=%d val=%s batch=%d geometry_points=%d"
          % (len(train_ds), len(val_ds) if val_ds else "-", batch_size,
             geo_points), flush=True)

    schedule = NoiseSchedule(
        diff_cfg["timesteps"],
        beta_schedule=diff_cfg.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02)).to(device)
    model = TrajSafePlanner(model_cfg, cfg.get("ellipse_label"),
                            cfg.get("bspline")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] params=%.2fM controls=%d traj_blocks=%d skeleton_blocks=%d "
          "final_blocks=%d curve=%d"
          % (n_params / 1e6, model.num_controls, model.traj_blocks,
             model.skeleton_blocks, model.final_blocks, model.horizon),
          flush=True)

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
            "model": "TrajSafePlanner-controlspace-32",
            "config": args.config,
            "processed_root": processed_root,
            "device": str(device),
            "params_m": n_params / 1e6,
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
            "loss_weights": {k: float(loss_cfg.get(m, 0.0)) for k, m in [
                ("Ltraj", "lambda_traj"), ("Lctrl", "lambda_control"),
                ("Lcoarse", "lambda_coarse"), ("Lsmooth", "lambda_smooth"),
                ("Ltopo", "lambda_topology"), ("Lshape", "lambda_shape"),
                ("Liou", "lambda_iou"), ("Lsafe", "lambda_safe")]},
            "alm_enabled": bool((cfg.get("alm") or {}).get("enabled", False)),
        }
        if extra:
            payload.update(extra)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    preview_batch = None
    for epoch in range(start_epoch, epochs):
        model.train()
        acc, n_steps, wacc, gnorms = {}, 0, {}, []
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
                        batch, model, schedule, loss_cfg, device)
                    if not bool(torch.isfinite(total)):
                        raise RuntimeError(
                            "non-finite total loss at epoch %d step %d"
                            % (epoch, step))
                    # gradient accumulation keeps the effective batch size while
                    # lowering the peak activation memory
                    (total / accum).backward()
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
        line = ("[epoch %d/%d] gnorm=%.3f " % (
            epoch, epochs, sum(gnorms) / max(len(gnorms), 1))
            + " ".join("%s=%.4f" % (k, avg_raw[k]) for k in LOSS_KEYS))
        entry = {"epoch": epoch, "train": {k: float(avg_raw[k])
                                           for k in LOSS_KEYS},
                 "seconds": time.time() - t0}
        if val_loader is not None and ((epoch + 1) % eval_every == 0
                                       or epoch == epochs - 1):
            v = validate(model, schedule, val_loader, loss_cfg, device,
                         int(train_cfg.get("val_batches", 12)))
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
            # objective (curve RMSE in meters + 40 m * collision rate).
            task = (float(v.get("curve_rmse_m", float("inf")))
                    + 40.0 * float(v.get("collision_rate", 0.0)))
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
                     int(train_cfg.get("val_batches", 12)))
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
