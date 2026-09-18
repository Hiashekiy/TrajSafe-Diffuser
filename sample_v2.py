"""Sample V2 (Skeleton-Topology-Grounded Trajectory Diffusion).

    python sample_v2.py --config configs/config_v2_skeleton.yaml \
        --ckpt outputs/ckpt_v2_skeleton/best.pt --split test --num 8 \
        --steps 16 --seed 0 --selection sample

Writes a figure with the occupancy map, the selected skeleton topology, the
sampled trajectory and the safety ellipses, plus an .npz with the raw arrays.
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
from src.diffusion.sampler_v2 import sample_v2
from src.models.skeleton import SkeletonPlanner
from src.datasets.skeleton_dataset import make_loader
from src.models.skeleton.path_ops import shape4_to_abtheta
from src.geometry.skeleton_graph import load_graph_npz


def to_pixels(points, size):
    """Scene [-1,1]^2 -> pixel-centre coordinates of a size x size map."""
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * size - 0.5


def plot_samples(occ, results, conds, paths, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = occ.shape[0]
    B = len(results["p"])
    cols = min(4, B)
    rows = int(np.ceil(B / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows),
                             dpi=110, squeeze=False)
    for b in range(B):
        ax = axes[b // cols][b % cols]
        ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
        sel = to_pixels(paths[b], size)
        ax.plot(sel[:, 0], sel[:, 1], color="#1f77b4", linewidth=1.4,
                label="selected topology")
        p = to_pixels(results["p"][b].detach().cpu().numpy(), size)
        ax.plot(p[:, 0], p[:, 1], color="#2ca02c", linewidth=1.8,
                label="trajectory")
        center = to_pixels(results["ellipse_center"][b].detach().cpu().numpy(),
                           size)
        sh = results["ellipse_shape4"][b].detach().cpu().numpy()
        a = np.exp(sh[:, 0])
        bb = np.exp(sh[:, 1])
        th = 0.5 * np.arctan2(sh[:, 3], sh[:, 2])
        ang = np.linspace(0, 2 * np.pi, 48)
        scale = size / 2.0                     # scene units -> pixels
        for k in range(0, len(center), 4):
            ct, st = np.cos(th[k]), np.sin(th[k])
            ex = a[k] * np.cos(ang) * scale
            ey = bb[k] * np.sin(ang) * scale
            xs = ct * ex - st * ey + center[k, 0]
            ys = st * ex + ct * ey + center[k, 1]
            ax.plot(xs, ys, color="#d62728", linewidth=0.6, alpha=0.8)
        ax.scatter([center[:, 0]], [center[:, 1]], s=2, c="#d62728")
        sg = to_pixels(conds[b], size)
        ax.scatter(sg[:, 0], sg[:, 1], marker="*", s=110, c="k", zorder=5)
        ax.set_title("m=%d  t_commit=%d" % (int(results["selected_idx"][b]),
                                            int(results["committed_at"][b])),
                     fontsize=9)
        if b == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v2_skeleton.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None,
                    help="sub-sampled reverse steps (None = every level)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selection", default=None, help="sample | argmax")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/v2_samples")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = (args.device if args.device else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    topo_cfg = cfg.get("topology", {})
    selection = args.selection or topo_cfg.get("selection", "sample")
    commit_t = int(topo_cfg.get("commit_t", 7))

    source = cfg["data"].get("source", "data/processed_scene_v1")
    base_dir = cfg["data"].get("base", "data/processed_scene_v2")
    loader, ds = make_loader(args.split, source, base_dir,
                             batch_size=args.num, shuffle=False, num_workers=0,
                             mask_res=int(cfg["loss"].get("ellipse_safe_res", 64)),
                             mask_tau=float(cfg["loss"].get("ellipse_mask_tau", 10.0)))

    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model = SkeletonPlanner(cfg["model"], topo_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    print("[sample_v2] loaded %s" % args.ckpt, flush=True)

    batch = next(iter(loader))
    sl = slice(args.offset, args.offset + args.num)
    cond = batch["cond"][sl].to(device)
    occ = batch["map_tensor"][sl].to(device)
    cand = batch["candidate_paths"][sl].to(device)
    cand_mask = batch["candidate_mask"][sl].to(device)
    cand_len = batch["candidate_lengths"][sl].to(device)

    out = sample_v2(model, schedule, cond, occ, cand, cand_mask, cand_len,
                    device=device, steps=args.steps, seed=args.seed,
                    commit_t=commit_t, selection=selection)

    os.makedirs(args.out, exist_ok=True)
    tag = "%s_%s_s%d_%s" % (args.split, args.offset, args.seed, selection)
    npz_path = os.path.join(args.out, "samples_" + tag + ".npz")
    np.savez_compressed(
        npz_path,
        p=out["p"].cpu().numpy(),
        ellipse_center=out["ellipse_center"].cpu().numpy(),
        ellipse_shape4=out["ellipse_shape4"].cpu().numpy(),
        progress=out["progress"].cpu().numpy(),
        selected_idx=out["selected_idx"].cpu().numpy(),
        committed_at=out["committed_at"].cpu().numpy(),
        topology_pi=out["topology_pi"].cpu().numpy(),
        candidate_paths=cand.cpu().numpy(),
        candidate_mask=cand_mask.cpu().numpy(),
        cond=cond.cpu().numpy(),
    )
    per_map = {}
    for b in range(cond.shape[0]):
        mid = int(batch["maze_id"][args.offset + b])
        per_map.setdefault(mid, []).append(b)
    for mid, rows in per_map.items():
        occ_map = ds.maps[mid][0, 0].numpy()
        sub = {k: v[rows] for k, v in
               (("p", out["p"]), ("ellipse_center", out["ellipse_center"]),
                ("ellipse_shape4", out["ellipse_shape4"]),
                ("selected_idx", out["selected_idx"]),
                ("committed_at", out["committed_at"]))}
        paths = cand[rows].cpu().numpy()
        idx = out["selected_idx"][rows].cpu().numpy()
        sel = np.stack([paths[i, idx[i], :, :2] for i in range(len(rows))])
        plot_samples(occ_map, sub, cond[rows].cpu().numpy(), sel,
                     os.path.join(args.out, "samples_%s_map%d.png" % (tag, mid)),
                     "V2 samples (%s)" % tag)
    with open(os.path.join(args.out, "samples_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "selection": selection,
                   "commit_t": commit_t, "steps": args.steps,
                   "selected_idx": out["selected_idx"].cpu().tolist(),
                   "committed_at": out["committed_at"].cpu().tolist(),
                   "topology_pi": out["topology_pi"].cpu().tolist()}, f, indent=2)
    print("saved", npz_path, flush=True)


if __name__ == "__main__":
    main()
