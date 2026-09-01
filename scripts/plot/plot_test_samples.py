"""Sample on test conditions and visualize map + trajectories + ellipses (scene coords)."""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.datasets.scene_dataset import load_maze_scene
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.diffusion.zerosum import compute_base, zero_sum
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import draw_traj, set_map_limits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze", default="umaze", choices=["umaze", "medium", "large"])
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pt")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    case_rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extent = next(m["extent"] for m in cfg["mazes"] if m["name"] == args.maze)
    frame, occ, sdf, conds = load_maze_scene(args.maze, split="test", n=args.n, rng=case_rng, extent=extent)
    cond_t = torch.as_tensor(conds, dtype=torch.float32).to(device)
    model = Planner(cfg["model"], cfg["geometry"], None).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"],
                             beta_end=cfg["diffusion"]["beta_end"]).to(device)
    map_t = torch.as_tensor(occ, dtype=torch.float32).to(device)[None, None]
    pos_scene, _ = sample(model, map_t, schedule, cond_t, args.n, device=device, steps=args.steps)
    pos_np = pos_scene.cpu().numpy()
    start = cond_t[:, 0]; goal = cond_t[:, 1]
    g = goal - start; N = pos_scene.shape[1] - 1; base = compute_base(g, N)
    delta = pos_scene[:, 1:] - pos_scene[:, :-1]
    z = zero_sum(delta - base)
    with torch.no_grad():
        pred = model(z, torch.zeros(pos_scene.shape[0], device=device, dtype=torch.long), map_t, cond_t)
    c = pred["ellipse_center"].cpu().numpy(); r1 = pred["ellipse_radii"][..., 0].cpu().numpy()
    r2 = pred["ellipse_radii"][..., 1].cpu().numpy(); th = pred["ellipse_theta"].cpu().numpy()
    ncol = 2; nrow = max(1, (args.n + 1) // 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes.ravel(), range(args.n)):
        ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r", alpha=0.9)
        draw_traj(ax, pos_np[i], marker_every=0, arrow_every=0)
        for j in range(0, len(pos_np[i]), 8):
            if not np.isfinite(r1[i][j] + r2[i][j]) or r1[i][j] <= 0:
                continue
            e = Ellipse((c[i][j][0], c[i][j][1]), 2 * r1[i][j], 2 * r2[i][j],
                        angle=np.degrees(th[i][j]), fill=False, edgecolor="tab:red", lw=1.0, alpha=0.7)
            ax.add_patch(e)
        ax.set_title(f"{args.maze} #{i}", fontsize=10)
        ax.set_aspect("equal"); set_map_limits(ax, (-1, 1, -1, 1)); ax.legend(fontsize=6)
    for k in range(args.n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"zero-sum bridge samples on {args.maze}", fontsize=13)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/test_sample_visual_{args.maze}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
