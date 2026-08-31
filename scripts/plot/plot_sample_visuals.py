"""Quick visualization: map + trajectory + GT ellipses (corrected axes)."""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from src.utils.config import load_config
from src.geometry.d4rl_geometry import build_occupancy_grid
from src.geometry.d4rl_coordinates import get_wall_centers_qpos, particle_clearance, MUJOCO_MARGIN
from src.datasets.normalization import load_normalization
from src.utils.visualization import draw_traj, set_map_limits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of data samples to plot")
    ap.add_argument("--idxs", type=str, default=None, help="comma-separated sample indices; overrides --n")
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    data = cfg["data"]; geom = cfg["geometry"]
    ext = tuple(geom["extent"]); res = float(geom["global_res"])
    proc = data["processed_dir"]
    occ, sdf, _ = build_occupancy_grid(data["maze"], extent=ext, global_res=res, inflate_particle=geom.get("inflate_particle", True))
    norm, _ = load_normalization(os.path.join(proc, "normalization.json"))
    mins = np.asarray(norm.mins[2:4]); maxs = np.asarray(norm.maxs[2:4]); eps = norm.eps
    wc = get_wall_centers_qpos(data["maze"])

    tdir = os.path.join(proc, "train")
    if args.idxs is not None:
        idxs = [int(x.strip()) for x in args.idxs.split(",") if x.strip()]
    else:
        n_total = len(np.load(os.path.join(tdir, "trajectories.npy")))
        n_plot = max(1, min(args.n, n_total))
        idxs = np.linspace(0, n_total - 1, n_plot).astype(int).tolist()

    ncol = 2
    nrow = max(1, (len(idxs) + 1) // 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, idx in zip(axes.ravel(), idxs):
        tdir = os.path.join(proc, "train")
        traj = np.load(os.path.join(tdir, "trajectories.npy"))[idx]
        eparams = np.load(os.path.join(tdir, "ellipse_params.npy"))[idx]
        ev = np.load(os.path.join(tdir, "ellipse_valid.npy"))[idx]
        pos = (traj[:, 2:4] + 1.0) / 2.0 * (maxs - mins + eps) + mins
        clear = particle_clearance(pos, wc)
        coll = float(np.mean(clear <= MUJOCO_MARGIN))
        ax.imshow(occ, origin="lower", extent=(ext[0], ext[1], ext[2], ext[3]), cmap="gray_r", alpha=0.8)
        vel = traj[:, 4:6]
        draw_traj(ax, pos, velocities=vel, marker_every=0, arrow_every=0)
        # sample ellipses every 12 waypoints for clarity
        for j in range(0, len(eparams), 12):
            if not ev[j]: continue
            cx, cy, r1, r2, th = eparams[j]
            e = Ellipse((cx, cy), 2*r1, 2*r2, angle=np.degrees(th), fill=False, edgecolor="tab:red", lw=1.2, alpha=0.8)
            ax.add_patch(e)
        ax.set_title(f"sample #{idx}  coll_rate={coll:.3f}  valid={int(ev.sum())}/{len(ev)}")
        ax.set_aspect("equal"); set_map_limits(ax, ext); ax.legend(loc="upper right", fontsize=7)
        print(f"sample {idx}: collision_rate={coll:.4f}, mean_clearance={float(clear.mean()):.4f}, p05={float(np.percentile(clear,5)):.4f}")
    for k in range(len(idxs), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Maze2D data sample (corrected axes): map + trajectory + sampled IRIS ellipses", fontsize=13)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out = "outputs/sample_visual_grid.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)

if __name__ == "__main__":
    main()
