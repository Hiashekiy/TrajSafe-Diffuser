"""Reverse-diffusion visualization for the zero-sum bridge planner (scene coords)."""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.datasets.scene_dataset import load_maze_scene
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.diffusion.zerosum import compute_base, zero_sum
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import draw_traj, set_map_limits


def draw_ellipse_patches(ax, c, r1, r2, theta, every=8, alpha=0.7, lw=1.2):
    for j in range(0, len(c), every):
        if not np.isfinite(r1[j] + r2[j]) or r1[j] <= 0:
            continue
        e = Ellipse((c[j, 0], c[j, 1]), 2 * r1[j], 2 * r2[j], angle=np.degrees(theta[j]),
                    fill=False, edgecolor="tab:red", lw=lw, alpha=alpha)
        ax.add_patch(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pt")
    ap.add_argument("--maze", default="umaze", choices=["umaze", "medium", "large"])
    ap.add_argument("--test-idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--mode", default="image", choices=["image", "video"])
    ap.add_argument("--no-ellipses", action="store_true")
    ap.add_argument("--out-reverse", default="outputs/test_reverse_diffusion.png")
    ap.add_argument("--out-single", default="outputs/test_sample_single.png")
    ap.add_argument("--video-out", default="outputs/test_reverse_diffusion.gif")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--video-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and cfg["env"].get("device", "cuda") == "cuda" else "cpu"
    extent = next(m["extent"] for m in cfg["mazes"] if m["name"] == args.maze)
    extent = next(m["extent"] for m in cfg["mazes"] if m["name"] == args.maze)
    frame, occ, sdf, conds = load_maze_scene(args.maze, split="test", extent=extent)
    cond = conds[args.test_idx:args.test_idx + 1]
    cond_t = torch.as_tensor(cond, dtype=torch.float32).to(device)
    model = Planner(cfg["model"], cfg["geometry"], None).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"],
                             beta_end=cfg["diffusion"]["beta_end"]).to(device)
    map_t = torch.as_tensor(occ, dtype=torch.float32).to(device)[None, None]
    pos_scene, traj_log, t_log = sample(model, map_t, schedule, cond_t, 1, device=device,
                                        steps=args.steps, return_traj=True, return_timesteps=True)
    start = cond_t[:, 0]; goal = cond_t[:, 1]; g = goal - start
    N = pos_scene.shape[1] - 1; base = compute_base(g, N)

    def predict_ellipse(xk_t, tk):
        d = xk_t[:, 1:] - xk_t[:, :-1]
        z = zero_sum(d - base)
        with torch.no_grad():
            pred = model(z, torch.full((1,), int(tk), device=device, dtype=torch.long), map_t, cond_t)
        return pred["ellipse_center"].cpu().numpy()[0], pred["ellipse_radii"][..., 0].cpu().numpy()[0],                pred["ellipse_radii"][..., 1].cpu().numpy()[0], pred["ellipse_theta"].cpu().numpy()[0]

    draw_ellipses = not args.no_ellipses
    with torch.no_grad():
        c, r1, r2, th = predict_ellipse(pos_scene, 0)
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r", alpha=0.8)
    draw_traj(ax1, pos_scene[0].cpu().numpy(), marker_every=0, arrow_every=0, lw=1.6)
    if draw_ellipses:
        draw_ellipse_patches(ax1, c, r1, r2, th, every=8)
    ax1.set_title(f"{args.maze} test #{args.test_idx}: final trajectory + ellipses")
    ax1.set_aspect("equal"); set_map_limits(ax1, (-1, 1, -1, 1)); ax1.legend(loc="upper right", fontsize=8)
    fig1.tight_layout()
    os.makedirs(os.path.dirname(args.out_single) or ".", exist_ok=True)
    fig1.savefig(args.out_single, dpi=120, bbox_inches="tight")
    print("saved", args.out_single)

    if args.mode == "video":
        frame_idx = list(range(len(traj_log))) if args.video_every <= 1 else             list(range(0, len(traj_log), args.video_every))
        if frame_idx[-1] != len(traj_log) - 1:
            frame_idx.append(len(traj_log) - 1)
        fig, ax = plt.subplots(figsize=(7, 7))
        def animate(i):
            idx = frame_idx[i]; ax.clear()
            ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r", alpha=0.8)
            xk = traj_log[idx].cpu().numpy()[0]
            draw_traj(ax, xk, marker_every=0, arrow_every=0, lw=1.4)
            if draw_ellipses:
                ck, r1k, r2k, thk = predict_ellipse(traj_log[idx], t_log[idx])
                draw_ellipse_patches(ax, ck, r1k, r2k, thk, every=8)
            ax.set_title(f"t = {t_log[idx]}  (step {idx})", fontsize=12)
            ax.set_aspect("equal"); set_map_limits(ax, (-1, 1, -1, 1))
            return (ax,)
        anim = FuncAnimation(fig, animate, frames=len(frame_idx), blit=False)
        os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
        writer = PillowWriter(fps=args.fps) if args.video_out.endswith(".gif") else FFMpegWriter(fps=args.fps)
        anim.save(args.video_out, writer=writer, dpi=120)
        print("saved", args.video_out)
        return

    n_snap = min(int(args.grid), len(traj_log))
    idxs = np.linspace(0, len(traj_log) - 1, n_snap).astype(int)
    ncol = 4; nrow = (n_snap + ncol - 1) // ncol
    fig2, axes2 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes2 = np.array(axes2).reshape(-1)
    for k, i in enumerate(idxs):
        ax = axes2[k]; xk = traj_log[i].cpu().numpy()[0]
        ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r", alpha=0.75)
        draw_traj(ax, xk, marker_every=0, arrow_every=0, lw=1.2)
        if draw_ellipses:
            ck, r1k, r2k, thk = predict_ellipse(traj_log[i], t_log[i])
            draw_ellipse_patches(ax, ck, r1k, r2k, thk, every=8)
        ax.set_title(f"t = {t_log[i]} (step {i})")
        ax.set_aspect("equal"); set_map_limits(ax, (-1, 1, -1, 1))
    for k in range(n_snap, len(axes2)):
        axes2[k].axis("off")
    fig2.tight_layout()
    fig2.savefig(args.out_reverse, dpi=120, bbox_inches="tight")
    print("saved", args.out_reverse)


if __name__ == "__main__":
    main()
