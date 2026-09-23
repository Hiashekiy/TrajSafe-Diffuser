"""plot_campaign.py - figures for the 160k8p A/B historical-feedback campaign.

Three PNGs:

``campaign_panels.png``      model x sample trajectory panels.  Runs the FULL
                             inference loop (DDIM + online corridor + per-step
                             ALM) for every model on the SAME test samples
                             (GPU), and draws occupancy, frozen corridor cells,
                             GT curve/controls, predicted curve/controls,
                             predicted ellipses and start/goal.
``campaign_per_step.png``    report-section-18 curves from
                             ``outputs/campaign_testset_eval.json`` (no GPU):
                             raw constraint violation, ALM correction magnitude
                             and the ALM-induced roughening, per reverse step.
``campaign_val_curves.png``  training curves from each stage's
                             ``training_summary.json`` (no GPU): task =
                             rmse_m + 80*collision, rmse, collision, and the
                             feedback acceptance rate.

Usage:

    python scripts/plot_campaign.py --samples 4 --steps 16
    python scripts/plot_campaign.py --skip-panels          # CPU only

The model list defaults to the three campaign arms plus the cross-cache
reference; ``--model NAME=ckpt[::config]`` overrides it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sample import draw_ellipses, plot_corridor, to_px  # noqa: E402

DEFAULT_MODELS = [
    "A_oneshot=outputs/campaign_a_oneshot/ckpt/best_task.pt",
    "B2_feedback=outputs/campaign_b2_feedback/ckpt/best_task.pt",
    "B1_base=outputs/campaign_b1_base/ckpt/best_task.pt"
    "::configs/config_160k8p_s1.yaml",
    "REF_160k8=outputs/bspline_carla_160k8/ckpt/best_task.pt",
]
STAGES = [("A_oneshot", "outputs/campaign_a_oneshot"),
          ("B1_base", "outputs/campaign_b1_base"),
          ("B2_feedback", "outputs/campaign_b2_feedback")]


# --------------------------------------------------------------------- panels
def sample_models(specs, config, split, num, steps, seed, device):
    """Run the sampler for every model on the SAME samples; return the outputs."""
    from src.utils.config import load_config, num_controls, num_safety_queries
    from src.utils.checkpoint import load_model
    from src.diffusion.schedule import NoiseSchedule
    from src.diffusion.sampler import sample
    from src.datasets.carla_spline_dataset import (CarlaSplineDataset,
                                                   make_collate)
    from eval_campaign_testset import parse_model

    cfg = load_config(config)
    data_cfg = cfg["data"]
    ds = CarlaSplineDataset(split, data_cfg.get("processed_root"),
                            geometry_points=1280, limit=int(num),
                            num_controls=num_controls(cfg),
                            num_safety_queries=num_safety_queries(cfg))
    batch = make_collate(ds)([ds[i] for i in range(len(ds))])
    schedule = NoiseSchedule(
        cfg["diffusion"]["timesteps"],
        beta_schedule=cfg["diffusion"].get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=cfg["diffusion"].get("beta_start", 1e-4),
        beta_end=cfg["diffusion"].get("beta_end", 0.02)).to(device)

    outs = []
    for spec in specs:
        name, ckpt, mcfg_path = parse_model(spec, config)
        path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
        if not os.path.exists(path):
            print("[plot] SKIP %s: %s missing" % (name, ckpt))
            continue
        mcfg = load_config(mcfg_path)
        model, _, _ = load_model(mcfg, path, device=device, verbose=False)
        out = sample(model, schedule, batch["cond"], batch["occupancy"],
                     batch["candidate_xy"], batch["candidate_mask"],
                     batch["candidate_geometry"],
                     batch["candidate_geometry_lengths"], device=device,
                     steps=steps, seed=seed, return_trace=True,
                     alm_config=mcfg.get("alm"),
                     corridor_config=mcfg.get("corridor"))
        print("[plot] sampled %s" % name, flush=True)
        outs.append((name, out))
    return batch, outs, float(data_cfg.get("scene_to_meter", 40.0))


def plot_panels(batch, outs, out_png, split, steps, scene_to_meter):
    n = int(batch["cond"].shape[0])
    res = int(batch["occupancy"].shape[-1])
    fig, axes = plt.subplots(n, len(outs), figsize=(3.4 * len(outs), 3.4 * n),
                             dpi=110, squeeze=False)
    occ = batch["occupancy"].cpu().numpy()
    cond = batch["cond"].cpu().numpy()
    gt = batch["pos"].cpu().numpy()
    q_gt = batch["control_gt"].cpu().numpy()
    for r in range(n):
        for c, (name, out) in enumerate(outs):
            ax = axes[r][c]
            ax.imshow(occ[r, 0], origin="lower", cmap="gray_r",
                      interpolation="nearest")
            plot_corridor(ax, out["corridors"][r], res)
            draw_ellipses(ax, out["ellipse_center"][r].cpu().numpy(),
                          out["ellipse_a"][r].cpu().numpy(),
                          out["ellipse_b"][r].cpu().numpy(),
                          out["ellipse_theta"][r].cpu().numpy(), res, stride=8)
            ax.plot(*to_px(gt[r], res).T, color="#2ca02c", lw=1.8,
                    label="GT curve")
            ax.plot(*to_px(out["p"][r].cpu().numpy(), res).T, color="#d62728",
                    lw=1.5, ls="--", label="pred curve")
            ax.plot(*to_px(q_gt[r], res).T, color="#2ca02c", lw=0.6,
                    marker="o", ms=1.6, alpha=0.5)
            ax.plot(*to_px(out["control"][r].cpu().numpy(), res).T,
                    color="#d62728", lw=0.6, marker="o", ms=1.6, alpha=0.6)
            ax.plot(*to_px(cond[r], res).T, linestyle="none", marker="*",
                    ms=11, color="k")
            fv = out["final_validation"][r]
            viol = fv.get("final_max_constraint_violation")
            ax.set_title("%s | s%d  coll=%s viol=%s rmse=%.1fm"
                         % (name, r, int(bool(fv.get("final_collision"))),
                            "-" if viol is None else "%.3f" % viol,
                            float(np.linalg.norm(
                                out["p"][r].cpu().numpy() - gt[r], axis=-1)
                                .mean()) * scene_to_meter), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("160k8p campaign: %s split, %d reverse steps "
                 "(corridor = each model's own prediction)"
                 % (split, steps), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------ per step
def plot_per_step(eval_json, out_png):
    with open(eval_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    results = data["results"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=110)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, r in enumerate(results):
        idx = [j for j, v in enumerate(r["per_step_raw_violation"])
               if v is not None]
        raw = [r["per_step_raw_violation"][j] for j in idx]
        corr = [r["per_step_alm_correction"][j] for j in idx]
        sr = [r["per_step_smooth_raw"][j] for j in idx]
        ss = [r["per_step_smooth_safe"][j] for j in idx]
        col = colors[i % 10]
        axes[0].plot(idx, raw, "-o", ms=3, color=col, label=r["name"])
        axes[1].plot(idx, corr, "-o", ms=3, color=col, label=r["name"])
        rough = [100.0 * (s / max(sr[k], 1e-12) - 1.0) for k, s in
                 enumerate(ss)]
        axes[2].plot(idx, rough, "-o", ms=3, color=col, label=r["name"])
    axes[0].set_title("raw prediction: max constraint violation\n"
                      "(before the ALM)", fontsize=9)
    axes[1].set_title("ALM correction magnitude\n(curve units, scene)",
                      fontsize=9)
    axes[2].set_title("ALM-induced curve roughening\n"
                      "(2nd difference, safe vs raw, %)", fontsize=9)
    for ax, logy in zip(axes, (True, True, False)):
        ax.set_xlabel("reverse-step index")
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    axes[2].axhline(0.0, color="k", lw=0.8)
    fig.suptitle("160k8p campaign: per-reverse-step diagnostics "
                 "(%s split, %d samples, %d steps, seed %s)"
                 % (data["split"], data["samples"], data["steps"],
                    data["seed"]), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------ training
def plot_training(out_png, stages=STAGES):
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), dpi=110)
    colors = {"A_oneshot": "#d62728", "B1_base": "#1f77b4",
              "B2_feedback": "#2ca02c"}
    for name, out_dir in stages:
        path = os.path.join(ROOT, out_dir, "training_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        hist = summary["history"]
        ep = [e["epoch"] for e in hist]
        rmse = [e.get("val", {}).get("curve_rmse_m") for e in hist]
        coll = [e.get("val", {}).get("collision_rate") for e in hist]
        task = [None if a is None or b is None else a + 80.0 * b
                for a, b in zip(rmse, coll)]
        fbv = [e.get("train", {}).get("fb_valid_rate") for e in hist]
        col = colors.get(name, "k")
        axes[0].plot(ep, task, color=col, lw=1.4, label=name)
        axes[1].plot(ep, rmse, color=col, lw=1.4, label=name)
        axes[2].plot(ep, coll, color=col, lw=1.4, label=name)
        if any(v is not None for v in fbv):
            axes[3].plot(ep, fbv, color=col, lw=1.4, label=name)
        best = min((t, e) for t, e in zip(task, ep) if t is not None)
        axes[0].plot([best[1]], [best[0]], "*", ms=12, color=col)
    axes[0].set_title("val task = rmse_m + 80*collision\n(* = best_task epoch)",
                      fontsize=9)
    axes[1].set_title("val curve RMSE (m)", fontsize=9)
    axes[2].set_title("val collision rate", fontsize=9)
    axes[3].set_title("train fb_valid_rate\n(feedback-trained stages only)",
                      fontsize=9)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("160k8p campaign: training curves "
                 "(A: 58 ep wall-clock capped, B1: 100 ep, B2: 100 + 37 ep)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_160k8p.yaml")
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-json", default="outputs/campaign_testset_eval.json")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--skip-panels", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    made = []

    if not args.skip_panels:
        device = args.device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        specs = args.model or DEFAULT_MODELS
        batch, outs, s2m = sample_models(specs, args.config, args.split,
                                         args.samples, args.steps, args.seed,
                                         device)
        if outs:
            png = os.path.join(out_dir, "campaign_panels.png")
            plot_panels(batch, outs, png, args.split, args.steps, s2m)
            made.append(png)
            print("[plot] %s" % png, flush=True)

    eval_json = args.eval_json if os.path.isabs(args.eval_json) \
        else os.path.join(ROOT, args.eval_json)
    if os.path.exists(eval_json):
        png = os.path.join(out_dir, "campaign_per_step.png")
        plot_per_step(eval_json, png)
        made.append(png)
        print("[plot] %s" % png, flush=True)
    png = os.path.join(out_dir, "campaign_val_curves.png")
    plot_training(png)
    made.append(png)
    print("[plot] %s" % png, flush=True)
    print("[plot] done: %d figure(s)" % len(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
