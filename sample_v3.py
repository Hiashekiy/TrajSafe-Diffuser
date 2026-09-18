"""Sample V3 (Skeleton-Topology-Grounded Trajectory Diffusion, dynamic).

    python sample_v3.py --config configs/config_v3_skeleton.yaml \
        --ckpt outputs/ckpt_v3_skeleton/best.pt --split test --num 6 --seed 0

Produces, per sample:
    samples_<tag>_<idx>.png    map + selected skeleton + trajectory + ellipses
    trace_<tag>_<idx>.png      per-timestep replay: coarse vs final trajectory,
                               the FRESHLY selected skeleton and the ellipse
                               centres (spec section 33 geometry visualisation)
and one npz/json with the raw arrays and the per-step selected candidate, which
is what makes a "skeleton jumping between timesteps" visible.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler_v3 import sample_v3
from src.models.skeleton_v3 import SkeletonPlannerV3
from src.datasets.skeleton_dataset_v3 import (MAZE_NAMES, SkeletonDatasetV3,
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


def plot_samples(occ, cond, path, traj, ell, out_png, title, stride=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = occ.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=110)
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    ax.plot(*to_px(path, res).T, color="#1f77b4", lw=1.3, label="selected skeleton")
    ax.plot(*to_px(traj, res).T, color="#2ca02c", lw=1.8, label="trajectory")
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
    """Per-timestep replay: coarse (dashed) vs final, plus the selected chain.

    ``trace`` is the per-sample trace: every step already has the batch axis
    removed, so the trajectory arrays are [H, 2] and are plotted in full (the
    old code took element 0 and silently drew a single waypoint).
    """
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
        cp = to_px(np.asarray(step["ellipse_center"]), res)
        ax.scatter(cp[:, 0], cp[:, 1], s=1.2, c="#d62728")
        ax.set_title("t=%d->%d  m=%d  pi(m)=%.2f"
                     % (step["t"], step["s"], j,
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
    return axes        # returned so tests can inspect what was actually drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v3_skeleton.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selection", default="argmax", choices=("argmax", "sample"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-trace-plot", action="store_true")
    ap.add_argument("--out", default="outputs/v3_samples")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    source = cfg["data"].get("source", "data/processed_scene_v1")
    base = cfg["data"].get("base", "data/processed_scene_v3")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))

    ds = SkeletonDatasetV3(args.split, source, base, geometry_points=geo_points)
    idxs = list(range(args.offset, min(args.offset + args.num, len(ds))))
    if not idxs:
        raise SystemExit("offset beyond the split")
    batch = make_collate(ds)([ds[i] for i in idxs])
    print("[sample_v3] %d samples from %s (geometry_points=%d)"
          % (len(idxs), args.split, geo_points), flush=True)

    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model = SkeletonPlannerV3(cfg["model"], cfg.get("ellipse")).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    print("[sample_v3] ckpt epoch=%s" % ckpt.get("epoch"), flush=True)

    cond = batch["cond"].to(device)
    occ = batch["map_tensor"].to(device)
    geom_cpu = batch["candidate_geometry"].numpy()
    glen_cpu = batch["candidate_geometry_lengths"].numpy()
    out = sample_v3(model, schedule, cond, occ,
                    batch["candidate_features"].to(device),
                    batch["candidate_mask"].to(device),
                    batch["candidate_lengths"].to(device),
                    batch["candidate_geometry"].to(device),
                    batch["candidate_geometry_lengths"].to(device),
                    device=device, steps=args.steps, seed=args.seed,
                    selection=args.selection, return_trace=True)

    os.makedirs(args.out, exist_ok=True)
    tag = "%s_%d_s%d_%s" % (args.split, args.offset, args.seed, args.selection)
    np.savez_compressed(
        os.path.join(args.out, "samples_%s.npz" % tag),
        p=out["p"].cpu().numpy(),
        center=out["ellipse_center"].cpu().numpy(),
        a=out["ellipse_a"].cpu().numpy(),
        b=out["ellipse_b"].cpu().numpy(),
        theta=out["ellipse_theta"].cpu().numpy(),
        progress=out["progress"].cpu().numpy(),
        selected_idx=out["selected_idx"].cpu().numpy(),
        pi=out["topology_pi"].cpu().numpy(),
        cond=cond.cpu().numpy(),
        idx=np.asarray(idxs))
    per_maze = {}
    for k, i in enumerate(idxs):
        per_maze.setdefault(int(batch["maze_id"][k]), []).append(k)
    for maze, rows in per_maze.items():
        occ_map = ds.maps[maze][0, 0].numpy()
        for k in rows:
            sel = int(out["selected_idx"][k])
            n = max(2, int(glen_cpu[k, sel]))
            plot_samples(
                occ_map, cond[k].cpu().numpy(), geom_cpu[k, sel, :n],
                out["p"][k].cpu().numpy(),
                {"center": out["ellipse_center"][k].cpu().numpy(),
                 "a": out["ellipse_a"][k].cpu().numpy(),
                 "b": out["ellipse_b"][k].cpu().numpy(),
                 "theta": out["ellipse_theta"][k].cpu().numpy()},
                os.path.join(args.out, "samples_%s_%d.png" % (tag, idxs[k])),
                "V3 %s #%d  m=%d" % (MAZE_NAMES[maze], idxs[k], sel))
            if not args.no_trace_plot:
                # the trace is a list of per-STEP dicts with a batch axis: slice
                # sample k out of every step and drop the batch dim, so every
                # array below is [H, 2] / [H] / [M]
                trace_k = [{**step,
                            "coarse": step["coarse"][k].numpy(),
                            "final": step["final"][k].numpy(),
                            "p": step["p"][k].numpy(),
                            "selected_idx": step["selected_idx"][k].numpy(),
                            "pi": step["pi"][k].numpy(),
                            "ellipse_center": step["ellipse_center"][k].numpy()}
                           for step in out["trace"]]
                plot_trace(occ_map, trace_k, geom_cpu[k], glen_cpu[k],
                           os.path.join(args.out,
                                        "trace_%s_%d.png" % (tag, idxs[k])),
                           "V3 reverse replay #%d (dashed=coarse, solid=final)" % idxs[k])
    with open(os.path.join(args.out, "samples_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "epoch": ckpt.get("epoch"),
                   "seed": args.seed, "selection": args.selection,
                   "steps": args.steps,
                   "selected_idx": out["selected_idx"].cpu().tolist(),
                   "pi": out["topology_pi"].cpu().tolist(),
                   "per_step_selection": [
                       [int(x["selected_idx"][b]) for x in out["trace"]]
                       for b in range(len(idxs))]}, f, indent=2)
    print("saved to", args.out, flush=True)


if __name__ == "__main__":
    main()
