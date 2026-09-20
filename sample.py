"""Sample the control-space (C B-spline controls) TrajSafe-Diffuser.

    python sample.py --config configs/config.yaml \
        --ckpt outputs/bspline_carla/ckpt/best.pt --split test --num 6 --seed 0

Inference routing is always argmax(pi).  The DDIM loop runs on the C-control
polygon; the 128-point curve is obtained by decoding the final controls.

``--arch auto`` (default) picks the forward chain from the checkpoint itself: a
checkpoint written BEFORE the control-space refactor is replayed through the
legacy 128-curve-token chain, so it keeps working unchanged; a new checkpoint
uses the control-token chain.  ``--arch control`` / ``--arch legacy`` forces one.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config, num_controls, num_safety_queries
from src.utils.checkpoint import load_model
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,
                                               make_collate)


def to_px(points, res):
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * res - 0.5


def draw_ellipses(ax, center, a, b, theta, res, stride=4, color="#d62728"):
    ang = np.linspace(0, 2 * np.pi, 48)
    scale = res / 2.0
    c = to_px(center, res)
    for k in range(0, len(center), max(1, stride)):
        ct, st = np.cos(theta[k]), np.sin(theta[k])
        ex = a[k] * np.cos(ang) * scale
        ey = b[k] * np.sin(ang) * scale
        ax.plot(ct * ex - st * ey + c[k, 0], st * ex + ct * ey + c[k, 1],
                color=color, lw=0.7, alpha=0.85)


def plot_samples(occ, cond, path, traj, gt, ell, out_png, title,
                 controls=None, gt_controls=None, stride=4, corridor=None,
                 raw_curve=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = occ.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=110)
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    plot_corridor(ax, corridor, res)
    if path is not None and len(path):
        ax.plot(*to_px(path, res).T, color="#1f77b4", lw=1.2,
                label="selected skeleton")
    ax.plot(*to_px(gt, res).T, color="#2ca02c", lw=1.6, label="GT curve")
    if raw_curve is not None:
        ax.plot(*to_px(np.asarray(raw_curve), res).T, color="#ff5ebf", lw=1.0,
                ls=":", label="raw clean x0 (before ALM)")
    ax.plot(*to_px(traj, res).T, color="#d62728", lw=1.7, ls="--",
            label="ALM safe x0 (DDIM)")

    if controls is not None:
        ax.plot(*to_px(controls, res).T, color="#d62728", lw=0.7, alpha=0.6,
                marker="o", ms=2, label="pred controls")
    if gt_controls is not None:
        ax.plot(*to_px(gt_controls, res).T, color="#2ca02c", lw=0.7, alpha=0.5,
                marker="o", ms=2, label="GT controls")
    draw_ellipses(ax, ell["center"], ell["a"], ell["b"], ell["theta"], res, stride)
    cp = to_px(ell["center"], res)
    ax.scatter(cp[:, 0], cp[:, 1], s=1.5, c="#d62728")
    sp = to_px(cond, res)
    ax.scatter(sp[:, 0], sp[:, 1], marker="*", s=150, c="k", zorder=6)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_trace(occ, trace, geometry, geometry_lengths, out_png, title):
    """Per-timestep replay of the control-space DDIM loop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = occ.shape[0]
    cols = 4
    rows = int(np.ceil(len(trace) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows),
                             dpi=110, squeeze=False)
    for i, step in enumerate(trace):
        ax = axes[i // cols][i % cols]
        ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
        j = int(np.asarray(step["selected_idx"]).reshape(-1)[0])
        n = max(2, int(geometry_lengths[j]))
        ax.plot(*to_px(geometry[j][:n], res).T, color="#1f77b4", lw=1.1)
        ax.plot(*to_px(np.asarray(step["coarse"]), res).T,
                color="#ff9f1c", lw=1.0, ls="--")
        ax.plot(*to_px(np.asarray(step["final"]), res).T,
                color="#2ca02c", lw=1.4)
        ax.plot(*to_px(np.asarray(step["p"]), res).T,
                color="#d62728", lw=0.9, ls=":")
        if step.get("alm_active") and "p_raw" in step:
            ax.plot(*to_px(np.asarray(step["p_raw"]), res).T,
                    color="#ff5ebf", lw=0.8, ls=":", alpha=0.9)
            ax.plot(*to_px(np.asarray(step["p_safe"]), res).T,
                    color="#25c8ff", lw=1.1, alpha=0.9)
        cp = to_px(np.asarray(step["ellipse_center"]), res)
        ax.scatter(cp[:, 0], cp[:, 1], s=1.2, c="#d62728")
        ax.set_title("t=%d->%d  m=%d  %s  pi(m)=%.2f"
                     % (step["t"], step["s"], j,
                        "ALM" if step.get("alm_active") else "raw",
                        float(np.asarray(step["pi"]).reshape(-1)[j])),
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(len(trace), rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return axes


def plot_corridor(ax, corridor, res, color="#25c8ff"):
    """Draw the frozen corridor: network cells light, bridge cells highlighted."""
    import matplotlib.patches as mpatches

    if not corridor or not corridor.get("cells"):
        return
    for cell in corridor["cells"]:
        poly = cell.get("polygon") or []
        if len(poly) < 3:
            continue
        pts = to_px(np.asarray(poly, dtype=np.float64), res)
        is_bridge = cell["source"] == "bridge"
        ax.add_patch(mpatches.Polygon(
            pts, closed=True, fill=True,
            facecolor="#ffd166" if is_bridge else color,
            edgecolor="#ff9f1c" if is_bridge else "#2fb9ff",
            alpha=0.30 if is_bridge else 0.07,
            linewidth=1.2 if is_bridge else 0.5,
            zorder=2 if is_bridge else 1))
    for gap in corridor.get("bridge_gaps") or []:
        c = to_px(np.asarray([gap["center"]], dtype=np.float64), res)[0]
        ax.scatter([c[0]], [c[1]], s=30, marker="X", c="#ff9f1c",
                   edgecolors="k", linewidths=0.4, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-trace-plot", action="store_true")
    ap.add_argument("--out", default="outputs/bspline_carla")
    ap.add_argument("--ablation", default=None,
                    choices=["A", "B", "C", "D"],
                    help="report section 48 ablation preset "
                         "(A raw / B final-only / C guided / D no-bridge). "
                         "Default: the configs/config.yaml settings.")
    ap.add_argument("--arch", default="auto",
                    choices=["auto", "control", "legacy"],
                    help="forward chain: auto = from the checkpoint "
                         "(a pre-refactor checkpoint replays the legacy "
                         "128-curve-token chain), control = control tokens, "
                         "legacy = 128 curve tokens")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    processed_root = cfg["data"].get("processed_root", "data/carla_processed")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))
    n_ctrl = num_controls(cfg)
    n_safety = num_safety_queries(cfg)

    ds = CarlaSplineDataset(args.split, processed_root,
                            geometry_points=geo_points, require_labels=False,
                            num_controls=n_ctrl, num_safety_queries=n_safety)
    idxs = list(range(args.offset, min(args.offset + args.num, len(ds))))
    if not idxs:
        raise SystemExit("offset beyond the split")
    batch = make_collate(ds)([ds[i] for i in idxs])
    print("[sample] %d samples from %s" % (len(idxs), args.split), flush=True)

    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model, ckpt, _ = load_model(cfg, args.ckpt, arch=args.arch, device=device)
    model.eval()
    print("[sample] ckpt epoch=%s controls=%d arch=%s"
          % (ckpt.get("epoch") if isinstance(ckpt, dict) else None,
             model.num_controls,
             "control_space" if model.control_space else "legacy_curve"),
          flush=True)

    cond = batch["cond"].to(device)
    occ = batch["occupancy"].to(device)
    geom_cpu = batch["candidate_geometry"].numpy()
    glen_cpu = batch["candidate_geometry_lengths"].numpy()

    alm_cfg, corridor_cfg = cfg.get("alm") or {}, cfg.get("corridor") or {}
    if args.ablation:
        from src.diffusion.sampler import ablation_configs
        alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg,
                                                args.ablation)
        print("[sample] ablation %s -> alm.mode=%s enabled=%s bridge=%s"
              % (args.ablation, alm_cfg.get("mode"), alm_cfg.get("enabled"),
                 (corridor_cfg.get("bridge") or {}).get("enabled")),
              flush=True)
    out = sample(model, schedule, cond, occ,
                 batch["candidate_xy"].to(device),
                 batch["candidate_mask"].to(device),
                 batch["candidate_geometry"].to(device),
                 batch["candidate_geometry_lengths"].to(device),
                 device=device, steps=args.steps, seed=args.seed,
                 return_trace=True, alm_config=alm_cfg,
                 corridor_config=corridor_cfg)

    os.makedirs(args.out, exist_ok=True)
    tag = "%s_%d_s%d" % (args.split, args.offset, args.seed)
    if args.ablation:
        tag += "_abl%s" % args.ablation
    np.savez_compressed(
        os.path.join(args.out, "samples_%s.npz" % tag),
        p=out["p"].cpu().numpy(),
        control=out["control"].cpu().numpy(),
        control_gt=batch["control_gt"].numpy(),
        curve_gt=batch["pos"].numpy(),
        center=out["ellipse_center"].cpu().numpy(),
        a=out["ellipse_a"].cpu().numpy(),
        b=out["ellipse_b"].cpu().numpy(),
        theta=out["ellipse_theta"].cpu().numpy(),
        progress=out["progress"].cpu().numpy(),
        selected_idx=out["selected_idx"].cpu().numpy(),
        pi=out["topology_pi"].cpu().numpy(),
        cond=cond.cpu().numpy(),
        idx=np.asarray(idxs))
    for k, i in enumerate(idxs):
        sel = int(out["selected_idx"][k])
        n = max(2, int(glen_cpu[k, sel]))
        alm_stats = out["trace"][-1].get("alm_stats") if out["trace"] else None
        title = "%s #%d  m=%d  alm=%s" % (
            args.split, i, sel, out["alm_status"][k])
        if alm_stats is not None:
            title += "  v %.4f->%.4f" % (
                float(alm_stats["max_violation_before"][k]),
                float(alm_stats["max_violation_after"][k]))
        plot_samples(
            batch["occupancy"][k, 0].numpy(), cond[k].cpu().numpy(),
            geom_cpu[k, sel, :n], out["p"][k].cpu().numpy(),
            batch["pos"][k].numpy(),
            {"center": out["ellipse_center"][k].cpu().numpy(),
             "a": out["ellipse_a"][k].cpu().numpy(),
             "b": out["ellipse_b"][k].cpu().numpy(),
             "theta": out["ellipse_theta"][k].cpu().numpy()},
            os.path.join(args.out, "samples_%s_%d.png" % (tag, i)),
            title,
            controls=out["control"][k].cpu().numpy(),
            gt_controls=batch["control_gt"][k].numpy(),
            corridor=out["corridors"][k],
            raw_curve=(out["trace"][-1]["p_raw"][k].numpy()
                       if out["trace"] else None))
        if not args.no_trace_plot:
            trace_k = [{**step,
                        "coarse": step["coarse"][k].numpy(),
                        "final": step["final"][k].numpy(),
                        "p": step["p"][k].numpy(),
                        "q": step["q"][k].numpy(),
                        "selected_idx": step["selected_idx"][k].numpy(),
                        "pi": step["pi"][k].numpy(),
                        "p_raw": step["p_raw"][k].numpy(),
                        "p_safe": step["p_safe"][k].numpy(),
                        "alm_active": bool(step["alm_active"]),
                        "ellipse_center": step["ellipse_center"][k].numpy()}
                       for step in out["trace"]]
            plot_trace(batch["occupancy"][k, 0].numpy(), trace_k, geom_cpu[k],
                       glen_cpu[k],
                       os.path.join(args.out, "trace_%s_%d.png" % (tag, i)),
                       "reverse replay #%d (orange=coarse, green=safe, "
                       "cyan=ALM safe, magenta=raw)" % i)
    with open(os.path.join(args.out, "samples_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "epoch": ckpt.get("epoch"),
                   "arch": ("control_space" if model.control_space
                            else "legacy_curve"),
                   "num_controls": int(model.num_controls),
                   "num_safety_queries": int(model.num_safety_queries),
                   "seed": args.seed, "steps": args.steps,
                   "ablation": args.ablation,
                   "selected_idx": out["selected_idx"].cpu().tolist(),
                   "pi": out["topology_pi"].cpu().tolist(),
                   "alm_status": list(out["alm_status"]),
                   "activation_step": out["activation_step"].cpu().tolist(),
                   "frozen_topology_idx":
                       out["frozen_topology_idx"].cpu().tolist(),
                   "alm_settings": out["alm_settings"],
                   "activation_info": {
                       "attempts": out["activation_info"]["attempts"],
                       "topology_fallback":
                           out["activation_info"]["topology_fallback"],
                       "failure_reason":
                           out["activation_info"]["failure_reason"],
                       "overlap_min": out["activation_info"]["overlap_min"],
                       "overlap_mean": out["activation_info"]["overlap_mean"],
                       "region_face_counts":
                           out["activation_info"]["region_face_counts"],
                   },
                   "pack_summary": out["pack_summary"],
                   "progress_alignment": out["progress_alignment"],
                   "final_validation": out["final_validation"],
                   "corridor_summary": [
                       None if c is None else {
                           "valid": c["valid"],
                           "base_cell_count": c["base_cell_count"],
                           "bridge_cell_count": c["bridge_cell_count"],
                           "min_overlap": min(c["overlap_ratio"])
                           if c["overlap_ratio"] else None,
                           "mean_overlap": (sum(c["overlap_ratio"])
                                            / len(c["overlap_ratio"]))
                           if c["overlap_ratio"] else None,
                       }
                       for c in out["corridors"]],
                   "per_step_selection": [
                       [int(x["selected_idx"][b]) for x in out["trace"]]
                       for b in range(len(idxs))],
                   "per_step_violation": [
                       [None if x["alm_stats"] is None else
                        [float(x["alm_stats"]["max_violation_before"][b]),
                         float(x["alm_stats"]["max_violation_after"][b])]
                        for x in out["trace"]]
                       for b in range(len(idxs))]}, f, indent=2)
    print("saved to", args.out, flush=True)


if __name__ == "__main__":
    main()
