"""Interactive Maze2D planner demo.

Draw walls/obstacles with the mouse, select a start and a goal, then press 'p'
to run the trained mixed-scene Planner and watch the reverse-diffusion process
and the final trajectory.

Controls:
  left-drag   : fill a rectangle with walls
  right-drag / 'e' click : erase a rectangle
  's' then click : set start
  'g' then click : set goal
  'w'         : switch to wall drawing mode
  'e'         : switch to erase mode
  'c'         : clear the map and start/goal
  'p'         : run planning (uses the checkpoint)
  'q'         : quit
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Disable matplotlib's built-in key bindings that conflict with our controls:
# 's' (save), 'q' (quit), 'p' (pan), 'g' (grid), etc. are taken by the app.
mpl.rcParams['keymap.save'] = []
mpl.rcParams['keymap.quit'] = []
mpl.rcParams['keymap.pan'] = []
mpl.rcParams['keymap.grid'] = []
mpl.rcParams['keymap.xscale'] = []
mpl.rcParams['keymap.yscale'] = []

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.datasets.normalization import load_normalization
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import draw_traj

EXTENT = (0.0, 8.0, 0.0, 8.0)


def world_to_grid(x, y, n):
    gx = int((x - EXTENT[0]) / (EXTENT[1] - EXTENT[0]) * n)
    gy = int((y - EXTENT[2]) / (EXTENT[3] - EXTENT[2]) * n)
    return max(0, min(n - 1, gx)), max(0, min(n - 1, gy))


def grid_to_world(gx, gy, n):
    x = EXTENT[0] + (gx + 0.5) / n * (EXTENT[1] - EXTENT[0])
    y = EXTENT[2] + (gy + 0.5) / n * (EXTENT[3] - EXTENT[2])
    return x, y


def world_to_norm(p, norm):
    mins = np.asarray(norm.mins[2:4], dtype=np.float64)
    maxs = np.asarray(norm.maxs[2:4], dtype=np.float64)
    eps = norm.eps
    cc = (np.asarray(p, dtype=np.float64) - mins) / (maxs - mins + eps)
    return (2.0 * cc - 1.0)


def draw_ellipses(ax, c_world, r1, r2, theta, every=8, alpha=0.8, lw=1.0):
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
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--grid-size", type=int, default=80)
    ap.add_argument("--fps-anim", type=float, default=30.0)
    ap.add_argument("--no-ellipses", action="store_true", help="do not draw predicted ellipses on the final trajectory")
    ap.add_argument("--seed", type=int, default=None, help="random seed; omit for random sampling")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and cfg["env"].get("device", "cuda") == "cuda" else "cpu"

    norm, _ = load_normalization(os.path.join("data/processed/mixed", "normalization.json"))
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"], beta_end=cfg["diffusion"]["beta_end"]).to(device)
    geom = dict(cfg["geometry"]); geom["extent"] = list(EXTENT)
    model = Planner(cfg["model"], geom, norm).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()

    n = int(args.grid_size)
    grid = np.zeros((n, n), dtype=np.uint8)  # 1 = wall
    state = {"mode": "wall", "start": None, "goal": None,
             "left_down": False, "right_down": False,
             "anchor": None, "last_cell": None}

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Interactive Maze2D Planner")

    def unnorm_positions(p):
        p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
        mins = np.asarray(norm.mins[2:4], dtype=np.float64)
        maxs = np.asarray(norm.maxs[2:4], dtype=np.float64)
        eps = norm.eps
        return (p + 1.0) / 2.0 * (maxs - mins + eps) + mins

    def draw_map(title="Draw the maze: s=start, g=goal, p=plan, c=clear"):
        ax.clear()
        ax.imshow(grid, origin="lower", extent=EXTENT, cmap="gray_r", alpha=0.95,
                  interpolation="nearest", zorder=0)
        if state["start"] is not None:
            ax.scatter(*state["start"], c="lime", marker="*", s=200, zorder=4, label="start")
        if state["goal"] is not None:
            ax.scatter(*state["goal"], c="red", marker="*", s=200, zorder=4, label="goal")
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        if state["start"] is not None or state["goal"] is not None:
            ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw()

    def paint(gx, gy, value):
        grid[gy, gx] = value
        draw_map()
        fig.canvas.draw_idle()

    def fill_rect(value):
        a = state["anchor"]; b = state["last_cell"]
        if a is None or b is None:
            return
        x0, x1 = sorted([a[0], b[0]])
        y0, y1 = sorted([a[1], b[1]])
        grid[y0:y1 + 1, x0:x1 + 1] = value
        draw_map()
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return
        gx, gy = world_to_grid(event.xdata, event.ydata, n)
        if event.button == 1:
            state["left_down"] = True
            state["anchor"] = (gx, gy)
            state["last_cell"] = (gx, gy)
            mode = state["mode"]
            if mode == "wall":
                fill_rect(1)
            elif mode == "erase":
                fill_rect(0)
            elif mode == "start":
                state["start"] = (event.xdata, event.ydata)
                state["mode"] = "wall"
                print("start set", state["start"])
                draw_map()
            elif mode == "goal":
                state["goal"] = (event.xdata, event.ydata)
                state["mode"] = "wall"
                print("goal set", state["goal"])
                draw_map()
        elif event.button == 3:
            state["right_down"] = True
            state["anchor"] = (gx, gy)
            state["last_cell"] = (gx, gy)
            fill_rect(0)

    def on_release(event):
        if event.button == 1:
            state["left_down"] = False
            state["anchor"] = None
            state["last_cell"] = None
        elif event.button == 3:
            state["right_down"] = False
            state["anchor"] = None
            state["last_cell"] = None

    def on_motion(event):
        if event.inaxes != ax:
            return
        gx, gy = world_to_grid(event.xdata, event.ydata, n)
        if state["right_down"]:
            state["last_cell"] = (gx, gy)
            fill_rect(0)
        elif state["left_down"]:
            state["last_cell"] = (gx, gy)
            mode = state["mode"]
            if mode == "wall":
                fill_rect(1)
            elif mode == "erase":
                fill_rect(0)

    def on_key(event):
        k = event.key
        if k == "s":
            state["mode"] = "start"
            print("click to set start")
        elif k == "g":
            state["mode"] = "goal"
            print("click to set goal")
        elif k == "w":
            state["mode"] = "wall"
            print("mode: wall")
        elif k == "e":
            state["mode"] = "erase"
            print("mode: erase")
        elif k == "c":
            grid[:] = 0
            state["start"] = None
            state["goal"] = None
            state["mode"] = "wall"
            print("cleared")
            draw_map()
        elif k == "q":
            plt.close(fig)
            print("quit")
        elif k == "p":
            run_planning()

    def run_planning():
        if state["start"] is None:
            print("please set start first (press s then click)")
            return
        if state["goal"] is None:
            print("please set goal first (press g then click)")
            return
        print("running planning ...")
        map_t = torch.as_tensor(grid, dtype=torch.float32).to(device)[None, None]
        cond = np.stack([
            world_to_norm(state["start"], norm),
            world_to_norm(state["goal"], norm),
        ])[None].astype(np.float32)
        cond_t = torch.as_tensor(cond, dtype=torch.float32).to(device)
        with torch.no_grad():
            x0, traj_log, t_log = sample(model, map_t, schedule, cond_t, 1, device=device,
                                         steps=args.steps, return_traj=True, return_timesteps=True)

        sleep = max(0.0, 1.0 / max(1.0, args.fps_anim))
        for i, xk in enumerate(traj_log):
            draw_map(f"denoising t={t_log[i]}  step {i}/{len(traj_log)-1}")
            xk_np = xk.cpu().numpy()[0]
            posk = unnorm_positions(xk_np[:, 2:4])
            velk = xk_np[:, 4:6]
            draw_traj(ax, posk, velocities=velk, marker_every=0, arrow_every=0, lw=1.2)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(sleep)

        # final trajectory
        draw_map("final trajectory")
        x0_np = x0.cpu().numpy()[0]
        pos0 = unnorm_positions(x0_np[:, 2:4])
        vel0 = x0_np[:, 4:6]
        draw_traj(ax, pos0, velocities=vel0, marker_every=0, arrow_every=0, lw=1.6)
        if not args.no_ellipses:
            t0 = torch.zeros((1,), device=device, dtype=torch.long)
            with torch.no_grad():
                pred = model(x0, t0, map_t, cond=cond_t)
            c_norm = pred["ellipse_center"].cpu().numpy()
            r1 = pred["ellipse_radii"][..., 0].cpu().numpy()
            r2 = pred["ellipse_radii"][..., 1].cpu().numpy()
            theta = pred["ellipse_theta"].cpu().numpy()
            c_world = unnorm_positions(c_norm[0])
            draw_ellipses(ax, c_world, r1[0], r2[0], theta[0], every=8)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        print("done. press p to replan, c to clear, q to quit")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_map()
    print(__doc__)
    print("Press 's' then click for start, 'g' then click for goal, 'p' to plan.")
    plt.show()


if __name__ == "__main__":
    main()
