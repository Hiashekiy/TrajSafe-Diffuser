"""Detailed reverse-diffusion visualization on a mixed-scene test sample.

Reads the mixed [0,8]^2 dataset, loads the checkpoint, and draws the
final x0 + predicted ellipses plus the noise->clean reverse-diffusion grid.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.datasets.mixed_dataset import EXTENT
from src.datasets.data_io import load_maze, unnorm_positions
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import draw_traj, set_map_limits


def draw_ellipse_patches(ax, c_world, r1, r2, theta, every=8, alpha=0.7, lw=1.2):
    """Draw predicted ellipses on ax.

    c_world : (H,2) world centers; r1/r2/theta : (H,) arrays.
    """
    for j in range(0, len(c_world), every):
        if not np.isfinite(r1[j] + r2[j]) or r1[j] <= 0:
            continue
        e = Ellipse((c_world[j, 0], c_world[j, 1]), 2 * r1[j], 2 * r2[j],
                    angle=np.degrees(theta[j]), fill=False, edgecolor="tab:red",
                    lw=lw, alpha=alpha)
        ax.add_patch(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pt")
    ap.add_argument("--maze", default="umaze", choices=["umaze", "medium", "large"])
    ap.add_argument("--test-idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--mode", default="image", choices=["image", "video"],
                    help="output mode; video also records the denoising process")
    ap.add_argument("--no-ellipses", action="store_true",
                    help="do not draw predicted ellipses")
    ap.add_argument("--out-reverse", default="outputs/test_reverse_diffusion.png")
    ap.add_argument("--out-single", default="outputs/test_sample_single.png")
    ap.add_argument("--video-out", default="outputs/test_reverse_diffusion.gif",
                    help="output video path (.gif or .mp4)")
    ap.add_argument("--fps", type=int, default=8, help="video frames per second")
    ap.add_argument("--video-every", type=int, default=1,
                    help="record every N denoising steps in the video")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed; omit for a different random sampling noise each run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and cfg["env"].get("device", "cuda") == "cuda" else "cpu"

    norm, occ, sdf, conds = load_maze(args.maze, split="test")
    cond = conds[args.test_idx:args.test_idx + 1]
    cond_t = torch.as_tensor(cond, dtype=torch.float32).to(device)
    ext = EXTENT

    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"], beta_end=cfg["diffusion"]["beta_end"]).to(device)
    geom = dict(cfg["geometry"]); geom["extent"] = list(EXTENT)
    model = Planner(cfg["model"], geom, norm).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)

    map_t = torch.as_tensor(occ, dtype=torch.float32).to(device)[None, None]
    x0, traj_log, t_log = sample(model, map_t, schedule, cond_t, 1, device=device,
                                     steps=args.steps, return_traj=True, return_timesteps=True)

    draw_ellipses = not args.no_ellipses

    def predict_ellipse_at(xk_t, t_k):
        """Run the model on one intermediate x_k to get its ellipses."""
        with torch.no_grad():
            tb = torch.full((1,), int(t_k), device=device, dtype=torch.long)
            predk = model(xk_t, tb, map_t, cond=cond_t)
        ck = unnorm_positions(predk["ellipse_center"].cpu().numpy(), norm)
        r1k = predk["ellipse_radii"][..., 0].cpu().numpy()
        r2k = predk["ellipse_radii"][..., 1].cpu().numpy()
        thk = predk["ellipse_theta"].cpu().numpy()
        return ck[0], r1k[0], r2k[0], thk[0]  # (H,2), (H,), (H,), (H,)

    # ---- (A) single final test sample: map + final trajectory + predicted ellipses ----
    with torch.no_grad():
        t0 = torch.zeros(x0.shape[0], device=device, dtype=torch.long)
        pred = model(x0, t0, map_t, cond=cond_t)
    c_norm = pred["ellipse_center"].cpu().numpy()
    r1 = pred["ellipse_radii"][..., 0].cpu().numpy()
    r2 = pred["ellipse_radii"][..., 1].cpu().numpy()
    theta = pred["ellipse_theta"].cpu().numpy()
    final_world = unnorm_positions(x0.cpu().numpy()[..., 2:4], norm)   # [1,H,2]
    c_world = unnorm_positions(c_norm, norm)                          # [1,H,2]

    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.imshow(occ, origin="lower", extent=(ext[0], ext[1], ext[2], ext[3]), cmap="gray_r", alpha=0.8)
    pos = final_world[0]
    vel1 = x0.cpu().numpy()[0, :, 4:6]
    draw_traj(ax1, pos, velocities=vel1, marker_every=0, arrow_every=0, lw=1.6)
    if draw_ellipses:
        draw_ellipse_patches(ax1, c_world[0], r1[0], r2[0], theta[0], every=8, alpha=0.8, lw=1.2)
    ax1.set_title(f"{args.maze} test #{args.test_idx}: final x0 + ellipses")
    ax1.set_aspect("equal"); set_map_limits(ax1, ext); ax1.legend(loc="upper right", fontsize=8)
    fig1.tight_layout()
    os.makedirs(os.path.dirname(args.out_single) or ".", exist_ok=True)
    fig1.savefig(args.out_single, dpi=120, bbox_inches="tight")
    print("saved", args.out_single)

    # ---- (B) output mode: video ----
    if args.mode == "video":
        if args.video_every <= 1:
            frame_idx = list(range(len(traj_log)))
        else:
            frame_idx = list(range(0, len(traj_log), args.video_every))
            if frame_idx[-1] != len(traj_log) - 1:
                frame_idx.append(len(traj_log) - 1)

        fig, ax = plt.subplots(figsize=(7, 7))

        def animate(i):
            idx = frame_idx[i]
            ax.clear()
            ax.imshow(occ, origin="lower", extent=(ext[0], ext[1], ext[2], ext[3]),
                      cmap="gray_r", alpha=0.8)
            xk_t = traj_log[idx]
            xk = xk_t.cpu().numpy()[0]
            posk = unnorm_positions(xk[:, 2:4], norm)
            velk = xk[:, 4:6]
            draw_traj(ax, posk, velocities=velk, marker_every=0, arrow_every=0, lw=1.4)
            if draw_ellipses:
                ck, r1k, r2k, thk = predict_ellipse_at(xk_t, t_log[idx])
                draw_ellipse_patches(ax, ck, r1k, r2k, thk, every=8, alpha=0.6, lw=1.0)
            ax.set_title(f"t = {t_log[idx]}  (step {idx})", fontsize=12)
            ax.set_aspect("equal")
            set_map_limits(ax, ext)
            return (ax,)

        anim = FuncAnimation(fig, animate, frames=len(frame_idx), blit=False)

        out_video = args.video_out
        os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
        if out_video.endswith(".gif"):
            writer = PillowWriter(fps=args.fps)
        elif out_video.endswith(".mp4"):
            writer = FFMpegWriter(fps=args.fps)
        else:
            raise ValueError("--video-out should end with .gif or .mp4")
        anim.save(out_video, writer=writer, dpi=120)
        plt.close(fig)
        print("saved", out_video)
        return

    # ---- (C) image mode: reverse-diffusion snapshots ----
    n_snap = min(int(args.grid), len(traj_log))
    idxs = np.linspace(0, len(traj_log) - 1, n_snap).astype(int)
    ncol = 4
    nrow = (n_snap + ncol - 1) // ncol
    fig2, axes2 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes2 = np.array(axes2).reshape(-1)
    for k, i in enumerate(idxs):
        ax = axes2[k]
        xk_t = traj_log[i]
        xk = xk_t.cpu().numpy()[0]
        posk = unnorm_positions(xk[:, 2:4], norm)
        ax.imshow(occ, origin="lower", extent=(ext[0], ext[1], ext[2], ext[3]),
                  cmap="gray_r", alpha=0.75)
        velk = xk[:, 4:6]
        draw_traj(ax, posk, velocities=velk, marker_every=0, arrow_every=0, lw=1.2)
        if draw_ellipses:
            ck, r1k, r2k, thk = predict_ellipse_at(xk_t, t_log[i])
            draw_ellipse_patches(ax, ck, r1k, r2k, thk, every=8, alpha=0.6, lw=1.0)
        ax.set_title(f"t = {t_log[i]}  (step {i})")
        ax.set_aspect("equal"); set_map_limits(ax, ext)
    for k in range(n_snap, len(axes2)):
        axes2[k].axis("off")
    fig2.suptitle("Reverse diffusion: noise -> clean trajectory (mixed-scene model)", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(args.out_reverse, dpi=120, bbox_inches="tight")
    print("saved", args.out_reverse)


if __name__ == "__main__":
    main()
