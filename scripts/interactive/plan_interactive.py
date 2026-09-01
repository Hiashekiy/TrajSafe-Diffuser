"""Interactive Maze2D planner on the scene frame [-1,1]^2.

Draw walls/obstacles with the mouse, set start/goal, press 'p' to run the model.
Controls: s=start, g=goal, w=wall, e=erase, c=clear, p=plan, q=quit.
"""
import argparse, os, sys, time
import numpy as np
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
mpl.rcParams['keymap.save'] = []; mpl.rcParams['keymap.quit'] = []
mpl.rcParams['keymap.pan'] = []; mpl.rcParams['keymap.grid'] = []
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.diffusion.zerosum import compute_base, zero_sum
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import draw_traj

EXTENT = (-1.0, 1.0, -1.0, 1.0)
N = 256


def scene_to_grid(x, y, n):
    gx = int((x - EXTENT[0]) / (EXTENT[1] - EXTENT[0]) * n)
    gy = int((y - EXTENT[2]) / (EXTENT[3] - EXTENT[2]) * n)
    return max(0, min(n - 1, gx)), max(0, min(n - 1, gy))


def grid_to_scene(gx, gy, n):
    return EXTENT[0] + (gx + 0.5) / n * (EXTENT[1] - EXTENT[0]),            EXTENT[2] + (gy + 0.5) / n * (EXTENT[3] - EXTENT[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pt")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--grid-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and cfg["env"].get("device", "cuda") == "cuda" else "cpu"
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"], beta_end=cfg["diffusion"]["beta_end"]).to(device)
    model = Planner(cfg["model"], cfg["geometry"], None).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()

    n = int(args.grid_size)
    grid = np.zeros((n, n), dtype=np.uint8)
    state = {"mode": "wall", "start": None, "goal": None,
             "left_down": False, "anchor": None, "last": None}
    fig, ax = plt.subplots(figsize=(8, 8))

    def draw_map(title="Draw: s=start, g=goal, p=plan, c=clear"):
        ax.clear()
        ax.imshow(grid, origin="lower", extent=EXTENT, cmap="gray_r", alpha=0.95, interpolation="nearest")
        if state["start"] is not None:
            ax.scatter(*state["start"], c="lime", marker="*", s=200, zorder=4, label="start")
        if state["goal"] is not None:
            ax.scatter(*state["goal"], c="red", marker="*", s=200, zorder=4, label="goal")
        ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        if state["start"] is not None or state["goal"] is not None:
            ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw()

    def paint(gx, gy, value):
        grid[gy, gx] = value

    def fill_rect(value):
        a, b = state["anchor"], state["last"]
        if a is None or b is None:
            return
        x0, x1 = sorted([a[0], b[0]]); y0, y1 = sorted([a[1], b[1]])
        grid[y0:y1 + 1, x0:x1 + 1] = value
        draw_map(); fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return
        gx, gy = scene_to_grid(event.xdata, event.ydata, n)
        if event.button == 1:
            state["left_down"] = True; state["anchor"] = (gx, gy); state["last"] = (gx, gy)
            mode = state["mode"]
            if mode == "wall":
                fill_rect(1)
            elif mode == "erase":
                fill_rect(0)
            elif mode == "start":
                state["start"] = (event.xdata, event.ydata); state["mode"] = "wall"; draw_map()
            elif mode == "goal":
                state["goal"] = (event.xdata, event.ydata); state["mode"] = "wall"; draw_map()
        elif event.button == 3:
            state["anchor"] = (gx, gy); state["last"] = (gx, gy); fill_rect(0)

    def on_release(event):
        if event.button == 1:
            state["left_down"] = False; state["anchor"] = None; state["last"] = None
        elif event.button == 3:
            state["anchor"] = None; state["last"] = None

    def on_motion(event):
        if event.inaxes != ax:
            return
        gx, gy = scene_to_grid(event.xdata, event.ydata, n)
        state["last"] = (gx, gy)
        if state["left_down"]:
            mode = state["mode"]
            fill_rect(1 if mode == "wall" else 0)

    def run_planning():
        if state["start"] is None or state["goal"] is None:
            print("set start (s) and goal (g) first"); return
        map_t = torch.as_tensor(grid, dtype=torch.float32).to(device)[None, None]
        cond = np.array([[state["start"], state["goal"]]], dtype=np.float32)
        cond_t = torch.as_tensor(cond, dtype=torch.float32).to(device)
        pos_scene, traj_log, t_log = sample(model, map_t, schedule, cond_t, 1, device=device,
                                            steps=args.steps, return_traj=True, return_timesteps=True)
        for i, xk in enumerate(traj_log):
            draw_map(f"denoising t={t_log[i]} step {i}/{len(traj_log)-1}")
            draw_traj(ax, xk.cpu().numpy()[0], marker_every=0, arrow_every=0, lw=1.2)
            fig.canvas.draw_idle(); fig.canvas.flush_events(); time.sleep(0.02)
        draw_map("final trajectory")
        pos0 = pos_scene.cpu().numpy()[0]
        draw_traj(ax, pos0, marker_every=0, arrow_every=0, lw=1.6)
        start = cond_t[:, 0]; goal = cond_t[:, 1]
        g = goal - start; Nb = pos_scene.shape[1] - 1; base = compute_base(g, Nb)
        delta = pos_scene[:, 1:] - pos_scene[:, :-1]
        z = zero_sum(delta - base)
        with torch.no_grad():
            pred = model(z, torch.zeros((1,), device=device, dtype=torch.long), map_t, cond_t)
        c = pred["ellipse_center"].cpu().numpy()[0]; r1 = pred["ellipse_radii"][..., 0].cpu().numpy()[0]
        r2 = pred["ellipse_radii"][..., 1].cpu().numpy()[0]; th = pred["ellipse_theta"].cpu().numpy()[0]
        for j in range(0, len(c), 8):
            if r1[j] > 0:
                ax.add_patch(Ellipse((c[j, 0], c[j, 1]), 2 * r1[j], 2 * r2[j], angle=np.degrees(th[j]),
                                     fill=False, edgecolor="tab:red", lw=1.0, alpha=0.7))
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        print("done. p=replan, c=clear, q=quit")

    def on_key(event):
        k = event.key
        if k == "s":
            state["mode"] = "start"
        elif k == "g":
            state["mode"] = "goal"
        elif k == "w":
            state["mode"] = "wall"
        elif k == "e":
            state["mode"] = "erase"
        elif k == "c":
            grid[:] = 0; state["start"] = None; state["goal"] = None; state["mode"] = "wall"; draw_map()
        elif k == "q":
            plt.close(fig)
        elif k == "p":
            run_planning()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_map()
    print(__doc__)
    plt.show()


if __name__ == "__main__":
    main()
