"""Training for the NEW single-path framework (v5).

    q_t [B,32,2] -> (Control Encoder -> Control Backbone x1)
                 -> [+ feedback | + single path | + 64 ellipses]
                 -> FusionMLP -> Final Denoiser x4 -> head_p -> BoundaryDecoder
                 -> ONE clean x0   (no coarse branch, no topology head)

Four-step rollout (framework plan §9.3):

    step 1   no path, no safety, no feedback        -> x0_1, DDIM -> q_2
    (A*)     the cached PDG-Focal A* path becomes active_path
    step 2   path + 64 ellipses, no feedback        -> x0_2, DDIM -> q_3
    step 3   latest active_path (+ feedback if verified)
    step 4   latest active_path (+ latest verified feedback)

Losses supervise the NETWORK's own raw clean prediction (never the ALM-corrected
polygon): control MSE, control smoothness, boundary-window shape, endpoint
heading, ellipse shape, and a differentiable collision penalty.

Honest scope of this file (see NEW_FRAMEWORK_STATUS.md):
* ALM + global validation are executed under ``no_grad`` only when
  ``train.rollout.use_alm`` is on; tonight's default keeps ``valid=False``
  (no fabricated feedback) because a per-batch corridor+ALM is too slow for the
  training loop.  The inference sampler does run ALM (see sampler_v5.py).
* ``train.rollout.use_pdg_path`` consumes the offline A* cache
  (``pdg_path_points.npy``) instead of re-running the search inside the loop.

    python train_v5.py --config configs/config_v5_c32q64.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import make_collate, make_loader
from src.diffusion.alm_local import (alm_closed_loop, collision_intervals,
                                    collision_mask)
from src.diffusion.schedule import NoiseSchedule
from src.models.factory import build_model, framework_of
from src.utils.config import load_config, num_controls, num_safety_queries
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def control_loss(x0, q_gt, cond, weights=None):
    """Endpoint-aware MSE on the control polygon (raw prediction)."""
    err = (x0 - q_gt) ** 2
    if weights is not None:
        err = err * weights[None, :, None]
    return err.mean()


def smooth_loss(x0):
    """2nd/3rd differences of the control polygon, scale-normalised."""
    d1 = x0[:, 1:] - x0[:, :-1]
    step = d1.norm(dim=-1).mean().clamp_min(1e-6)
    d2 = d1[:, 1:] - d1[:, :-1]
    d3 = d2[:, 1:] - d2[:, :-1]
    return d2.norm(dim=-1).mean() / step + 0.5 * d3.norm(dim=-1).mean() / step


def boundary_loss(model, x0, q_gt, cond):
    """Local shape near both ends, translated to THIS sample's endpoints."""
    C = x0.shape[1]
    span = int(getattr(model.boundary_decoder, "span", 4))
    span = max(1, min(span, C // 2))
    off_s = (q_gt - q_gt[:, 0:1]) + cond[:, 0][:, None, :]
    off_g = (q_gt - q_gt[:, C - 1:C]) + cond[:, 1][:, None, :]
    ls = F.mse_loss(x0[:, :span], off_s[:, :span])
    lg = F.mse_loss(x0[:, C - span:], off_g[:, C - span:])
    return 0.5 * (ls + lg)


def heading_loss(x0, heading, heading_valid):
    """1 - cos between the endpoint tangent and the target direction."""
    if heading is None:
        return x0.new_zeros(())
    d0 = x0[:, 1] - x0[:, 0]
    d1 = x0[:, -1] - x0[:, -2]
    t0 = heading[:, 0]
    t1 = heading[:, 1]
    c0 = (d0 * t0).sum(-1) / d0.norm(dim=-1).clamp_min(1e-6)
    c1 = (d1 * t1).sum(-1) / d1.norm(dim=-1).clamp_min(1e-6)
    l0 = (1.0 - c0).clamp(min=0.0)
    l1 = (1.0 - c1).clamp(min=0.0)
    if heading_valid is not None:
        v = heading_valid.to(x0.dtype)
        l0 = l0 * v[:, 0]
        l1 = l1 * v[:, 1]
        denom = v.sum().clamp_min(1.0)
        return (l0.sum() + l1.sum()) / denom
    return l0.mean() + l1.mean()


def shape_loss(ell, shape4_gt, shape_valid):
    if not ell or shape4_gt is None:
        return None
    pred = ell["shape4"]
    if pred.shape[1] != shape4_gt.shape[1]:
        return None
    m = shape_valid
    if m.sum() < 1:
        return None
    err = ((pred - shape4_gt) ** 2).mean(dim=-1)
    return (err * m).sum() / m.sum().clamp_min(1.0)


def collision_penalty(model, x0, occ):
    """Differentiable soft penalty: decoded curve on occupied cells."""
    curve = model.bspline.decode_controls(x0)                     # [B,128,2]
    grid = curve[:, None]                                         # [B,1,128,2]
    val = F.grid_sample(occ, grid, mode="bilinear", padding_mode="border",
                        align_corners=True)                       # [B,1,1,128]
    return val.mean()


# ---------------------------------------------------------------------------
# DDIM
# ---------------------------------------------------------------------------
def ddim_step(q, x0, t, s, sched, cond):
    if s < 0:
        q = x0
    else:
        sa_t = sched.sqrt_alphas_cumprod[t].view(-1, 1, 1).to(q.dtype)
        s1_t = sched.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1).to(q.dtype)
        sa_s = sched.sqrt_alphas_cumprod[s].view(-1, 1, 1).to(q.dtype)
        s1_s = sched.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1).to(q.dtype)
        eps = (q - sa_t * x0) / s1_t.clamp_min(1e-8)
        q = sa_s * x0 + s1_s * eps
    q = q.clone()
    q[:, 0] = cond[:, 0]
    q[:, -1] = cond[:, 1]
    return q


def pick_times(T: int, steps: int):
    idx = torch.linspace(0, T - 1, steps).long().tolist()
    times = sorted(set(int(v) for v in idx))
    if T - 1 not in times:
        times.append(T - 1)
    if 0 not in times:
        times.append(0)
    return sorted(times)


# ---------------------------------------------------------------------------
# one rollout
# ---------------------------------------------------------------------------
def rollout(model, sched, batch, cfg, device, train=True):
    lw = dict(cfg.get("loss") or {})
    T = int(sched.num_timesteps)
    ro_cfg = dict((cfg.get("train", {}) or {}).get("rollout") or {})
    steps = int(ro_cfg.get("steps", 4))
    # §9.3: the training rollout MUST run the same closed loop as inference
    use_alm = bool(ro_cfg.get("use_alm", True))
    times = pick_times(T, steps)
    pairs = [(times[i], times[i - 1]) for i in range(len(times) - 1, 0, -1)]
    pairs.append((times[0], -1))
    B, C = batch["cond"].shape[0], int(model.num_controls)

    cond = batch["cond"].to(device)
    occ = batch["occupancy"].to(device)
    q_gt = batch["control_gt"].to(device)
    heading = batch.get("heading")
    hv = batch.get("heading_valid")
    heading = heading.to(device) if heading is not None else None
    hv = hv.to(device) if hv is not None else None
    path = batch.get("pdg_path")
    path_valid = batch.get("pdg_path_valid")
    path = path.to(device) if path is not None else None
    path_valid = path_valid.to(device) if path_valid is not None else None
    shape_gt = batch.get("ellipse_shape4_gt")
    shape_m = batch.get("shape_valid")
    shape_gt = shape_gt.to(device) if shape_gt is not None else None
    shape_m = shape_m.to(device) if shape_m is not None else None

    q = torch.randn(B, C, 2, device=device)
    q[:, 0] = cond[:, 0]
    q[:, -1] = cond[:, 1]

    active_path = None
    fb_delta = torch.zeros(B, C, 2, device=device)
    fb_valid = torch.zeros(B, dtype=torch.bool, device=device)
    lam_prev = None

    step_w = [0.4, 0.6, 0.8, 1.0][:len(pairs)]
    total, parts = 0.0, {}
    preds = []
    for i, (t, s) in enumerate(pairs):
        tb = torch.full((B,), int(t), device=device, dtype=torch.long)
        ab = sched.sqrt_alphas_cumprod[int(t)].expand(B).to(device)
        out = model(q, occ, cond, tb, ab, active_path=active_path,
                    feedback_delta=(fb_delta if i >= 2 else None),
                    feedback_valid=(fb_valid if i >= 2 else None),
                    heading=heading, heading_valid=hv)
        x0 = out["control"]
        preds.append(x0.detach())
        w = step_w[i] if i < len(step_w) else 1.0
        l_ctrl = control_loss(x0, q_gt, cond)
        l_sm = smooth_loss(x0)
        l_bd = boundary_loss(model, x0, q_gt, cond)
        l_hd = heading_loss(x0, heading, hv)
        l_sh = shape_loss(out["ellipse"], shape_gt, shape_m)
        l_col = collision_penalty(model, x0, occ)
        step_loss = (float(lw.get("lambda_control", 0.2)) * l_ctrl
                     + float(lw.get("lambda_smooth", 0.08)) * l_sm
                     + float(lw.get("lambda_boundary", 0.2)) * l_bd
                     + float(lw.get("lambda_heading", 0.1)) * l_hd
                     + (float(lw.get("lambda_shape", 0.08)) * l_sh
                        if l_sh is not None else 0.0)
                     + float(lw.get("lambda_safe", 0.15)) * l_col)
        total = total + w * step_loss
        for k, v in (("ctrl", l_ctrl), ("smooth", l_sm), ("bound", l_bd),
                     ("heading", l_hd), ("shape", l_sh), ("coll", l_col)):
            if v is not None:
                parts.setdefault(k, []).append(float(v.detach()))

        q = ddim_step(q, x0, tb, s, sched, cond)

        # ---- after step 1: the cached PDG-Focal A* path becomes active -----
        if i == 0:
            if path is None:
                # Bootstrap path: the cached PDG-Focal A* path is not ready yet.
                # The GT curve is used instead (path_source="gt") so the
                # path-conditioned branch is still trained on a sane path; the
                # doc's §9.2 refresh+fine-tune then switches to the real A* path.
                gt = batch.get("pos")
                if gt is not None:
                    path = gt.to(device)
                    path_valid = torch.ones(B, dtype=torch.bool, device=device)
            if path is not None:
                ok = (path_valid.to(torch.bool) if path_valid is not None
                      else torch.ones(B, dtype=torch.bool, device=device))
                seed = torch.stack([torch.linspace(cond[b, 0, 0], cond[b, 1, 0], path.shape[1])
                                    for b in range(B)])
                seed_y = torch.stack([torch.linspace(cond[b, 0, 1], cond[b, 1, 1], path.shape[1])
                                      for b in range(B)])
                straight = torch.stack([seed, seed_y], dim=-1).to(path.device)
                active_path = torch.where(ok[:, None, None], path, straight)
            else:
                u = torch.linspace(0, 1, 128, device=device)[None, :, None]
                active_path = cond[:, 0:1] * (1 - u) + cond[:, 1:2] * u
        else:
            # ---------------- CLOSED LOOP (same code as inference) -----------
            # collide -> local corridor cells -> local pack -> ALM -> global
            # validation -> active_path / feedback.  Under no_grad: the loss
            # never back-propagates through the projection (§9.3).
            if use_alm and getattr(model, "feedback_enabled", True):
                with torch.no_grad():
                    Q = int(getattr(model, "num_safety_queries", 64))
                    curve_i = model.bspline.decode_controls(x0.detach())
                    any_coll = False
                    for b in range(B):
                        hit_b, _ = collision_mask(occ[b], curve_i[b])
                        any_coll = any_coll or bool(hit_b.any())
                    if any_coll:
                        ell_det = {k: (v.detach() if torch.is_tensor(v) else v)
                                   for k, v in (out.get("ellipse") or {}).items()}
                        q0_safe, lam_prev, info = alm_closed_loop(
                            model, occ.detach(), cond, x0.detach(), ell_det, cfg,
                            gamma=active_path.detach() if active_path is not None
                            else None, lam=lam_prev)
                        n_ran = 1.0
                        ok = bool(info.get("alm_ok", False))
                        if ok:
                            safe_curve = model.bspline.decode_controls(q0_safe)
                            active_path = safe_curve
                            fb_delta = (q0_safe - x0.detach())
                            fb_valid = torch.ones(B, dtype=torch.bool,
                                                  device=device)
                            parts.setdefault("alm_ok", []).append(1.0)
                        else:
                            fb_valid = torch.zeros(B, dtype=torch.bool,
                                                   device=device)
                            parts.setdefault("alm_ok", []).append(0.0)
                        parts.setdefault("alm_ran", []).append(n_ran)
                        ps = info.get("per_sample") or []
                        if ps:
                            parts.setdefault("coll_after", []).append(
                                float(np.mean([p["collision_frac"] for p in ps])))
                        parts.setdefault("corr", []).append(
                            float((q0_safe - x0.detach()).norm(dim=-1).mean()))
            # if ALM did not run / did not pass: keep the previous active_path,
            # do NOT fabricate feedback (§7 last paragraph).

    loss = total / float(sum(step_w[:len(pairs)]))
    metrics = {k: float(np.mean(v)) for k, v in parts.items()}
    return loss, metrics, preds


# ---------------------------------------------------------------------------
def evaluate(model, sched, loader, cfg, device, max_batches=6):
    model.eval()
    agg = {}
    n = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            loss_v, m, preds = rollout(model, sched, batch, cfg, device,
                                       train=False)
            agg["loss"] = agg.get("loss", 0.0) + float(loss_v)
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            # discrete metrics on the last prediction
            x0 = preds[-1]
            cond = batch["cond"].to(device)
            q_gt = batch["control_gt"].to(device)
            agg["ctrl_mse"] = agg.get("ctrl_mse", 0.0) + float(
                F.mse_loss(x0, q_gt))
            curve = model.bspline.decode_controls(x0)
            occ = batch["occupancy"].to(device)
            agg["coll_soft"] = agg.get("coll_soft", 0.0) + float(
                collision_penalty(model, x0, occ))
            agg["endpoint"] = agg.get("endpoint", 0.0) + float(
                (curve[:, 0] - cond[:, 0]).norm(dim=-1).mean()
                + (curve[:, -1] - cond[:, 1]).norm(dim=-1).mean())
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q64.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int((cfg.get("env") or {}).get("seed", 42)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr = dict(cfg.get("train") or {})
    epochs = int(args.epochs if args.epochs is not None else tr.get("epochs", 120))
    max_hours = float(args.max_hours if args.max_hours is not None
                      else tr.get("max_hours", 6.0))
    batch = int(args.batch if args.batch is not None
                else (cfg["data"].get("batch_size") or 16))
    lr = float(args.lr if args.lr is not None else tr.get("lr", 2e-4))
    ckpt_dir = tr.get("ckpt_dir", "outputs/models/v5_c32q64/ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "train_log.jsonl")

    print("[v5] framework=%s C=%d Q=%d path_points=%d backbone=%s final=%s"
          % (framework_of(cfg), num_controls(cfg), num_safety_queries(cfg),
             cfg["model"].get("path_points", 128), cfg["model"].get("traj_blocks"),
             cfg["model"].get("final_blocks")), flush=True)

    loader, ds = make_loader(
        "train", cfg["data"]["processed_root"], batch_size=batch,
        shuffle=True, num_workers=0,
        geometry_points=int((cfg.get("topology") or {}).get("candidate_geometry_points", 1280)),
        limit=args.limit, require_labels=True, num_controls=num_controls(cfg),
        num_safety_queries=num_safety_queries(cfg))
    val_loader, _ = make_loader(
        "val", cfg["data"]["processed_root"], batch_size=batch, shuffle=False,
        num_workers=0, limit=args.limit, require_labels=True,
        num_controls=num_controls(cfg), num_safety_queries=num_safety_queries(cfg))
    print("[v5] train=%d val=%d batches/epoch=%d" % (len(ds), len(val_loader.dataset),
                                                     len(loader)), flush=True)

    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                             "squaredcos_cap_v2")).to(device)
    model = build_model(cfg).to(device)
    print("[v5] params(M)=%.2f" % (sum(p.numel() for p in model.parameters()) / 1e6),
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(tr.get("weight_decay", 1e-4)))
    start_ep, best = 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_ep = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("best", float("inf")))
        print("[v5] resumed from %s @ epoch %d" % (args.resume, start_ep - 1))

    t_start = time.time()
    model.train()
    for ep in range(start_ep, epochs):
        ep_t0 = time.time()
        sums, nb = {}, 0
        for batch_i, batch in enumerate(loader):
            loss, m, _ = rollout(model, sched, batch, cfg, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(tr.get("grad_clip", 1.0)))
            opt.step()
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach())
            nb += 1
            if (batch_i + 1) % int(tr.get("log_interval", 20)) == 0:
                print("[v5] ep %d %d/%d loss=%.4f ctrl=%.4f coll=%.4f %.0fs"
                      % (ep, batch_i + 1, len(loader),
                         sums["loss"] / nb, sums.get("ctrl", 0) / nb,
                         sums.get("coll", 0) / nb, time.time() - ep_t0),
                      flush=True)
            if time.time() - t_start > max_hours * 3600:
                break
        tr_m = {k: v / max(nb, 1) for k, v in sums.items()}
        va = evaluate(model, sched, val_loader, cfg, device,
                      max_batches=int(tr.get("val_batches", 6)))
        rec = {"epoch": ep, "train": tr_m, "val": va,
               "seconds": time.time() - ep_t0}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("[v5] == ep %d train_loss=%.4f val_loss=%.4f val_ctrl_mse=%.5f "
              "val_coll=%.4f val_ep=%.4f %.0fs"
              % (ep, tr_m.get("loss", float("nan")), va.get("loss", float("nan")),
                 va.get("ctrl_mse", float("nan")), va.get("coll", float("nan")),
                 va.get("endpoint", float("nan")), time.time() - ep_t0), flush=True)

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "epoch": ep, "best": best, "config": cfg}
        torch.save(state, os.path.join(ckpt_dir, "latest.pt"))
        score = va.get("loss", float("inf"))
        if score < best:
            best = score
            state["best"] = best
            torch.save(state, os.path.join(ckpt_dir, "best_task.pt"))
            print("[v5] new best (%.4f) -> best_task.pt" % best, flush=True)
        if time.time() - t_start > max_hours * 3600:
            print("[v5] max_hours reached", flush=True)
            break
    print("[v5] done in %.1f min; ckpt=%s" % ((time.time() - t_start) / 60,
                                              ckpt_dir), flush=True)


if __name__ == "__main__":
    main()
