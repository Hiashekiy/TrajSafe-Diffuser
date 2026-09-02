"""Diagnose the Phase-3 local-consensus centerline guidance.

For a given checkpoint / maze it
  * samples the zero-sum bridge (guidance ON, matching plot_test.py),
  * re-runs the ellipse head at t=0 on the reconstructed z so the predicted
    ellipses match the final trajectory (same as plot_test.py),
  * computes the V5 local consensus geometry (cbar / ubar / nbar / r2bar /
    gamma / q), draws the consensus centerline on top of the trajectory +
    ellipses, and draws per-anchor deviation connectors from the trajectory
    to the consensus center,
  * prints a quantitative summary (J_guide, mean |d_perp|, mean |delta|,
    mean gamma, fraction of Huber-saturated points),
  * optionally samples with guidance OFF and overlaid both trajectories to
    show what J_guide actually moved.

Outputs:
    outputs/centerline_guide_{maze}.png       centerline + deviation overlay
    outputs/centerline_compare_{maze}.png     guidance ON vs OFF (if --compare)
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.datasets.scene_dataset import load_maze_scene
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.diffusion.zerosum import compute_base, zero_sum
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import draw_traj, set_map_limits
from src.geometry.scene_frame import sample_sdf_scene
from src.geometry.offline_iris_wrapper import infer_convex_region_from_scene_occupancy
from src.guidance.local_consensus import compute_consensus_geometry, consensus_guidance_cost


def scene_ellipse_prediction(pos_scene, cond, map_t, model, device):
    """Reconstruct z from the sampled positions and predict ellipses at t=0."""
    start = cond[:, 0]; goal = cond[:, 1]
    g = goal - start
    N = pos_scene.shape[1] - 1
    base = compute_base(g, N)
    delta = pos_scene[:, 1:] - pos_scene[:, :-1]
    z = zero_sum(delta - base)
    with torch.no_grad():
        t0 = torch.zeros(pos_scene.shape[0], device=device, dtype=torch.long)
        pred = model(z, t0, map_t, cond)
    c = pred["ellipse_center"].cpu().numpy()
    r1 = pred["ellipse_radii"][..., 0].cpu().numpy()
    r2 = pred["ellipse_radii"][..., 1].cpu().numpy()
    th = pred["ellipse_theta"].cpu().numpy()
    return c, r1, r2, th


def draw_centerline(ax, cbar, ubar, nbar, r2bar, gamma, p_match,
                    show_dir=True, show_dev=True, centerline_cmap="cool",
                    dev_color="orange", dir_color="cyan"):
    """Overlay consensus centerline, consensus directions, and deviations."""
    # consensus centerline
    x = cbar[:, 0]; y = cbar[:, 1]
    pts = np.stack([x, y], axis=1)
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=centerline_cmap, linewidths=1.4, alpha=0.95, zorder=4)
    lc.set_array(gamma[:-1])
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)
    ax.scatter(x, y, c=gamma, cmap=centerline_cmap, s=12, edgecolors="k",
               linewidths=0.3, zorder=5, label="consensus center")

    if show_dir:
        # long-axis (consensus) direction segment at every 6th anchor
        for j in range(0, cbar.shape[0], 6):
            L = 0.10
            dx = ubar[j, 0] * L; dy = ubar[j, 1] * L
            ax.plot([cbar[j, 0] - dx, cbar[j, 0] + dx],
                    [cbar[j, 1] - dy, cbar[j, 1] + dy],
                    color=dir_color, lw=1.0, alpha=0.9, zorder=5)

    if show_dev:
        # connector from matched interior trajectory point to consensus center
        nd = p_match.shape[0]
        for k in range(0, nd, 3):
            ax.plot([p_match[k, 0], cbar[k, 0]], [p_match[k, 1], cbar[k, 1]],
                    color=dev_color, lw=0.8, alpha=0.75, zorder=3)
        ax.scatter(p_match[::3, 0], p_match[::3, 1], c="tab:orange", s=10,
                   edgecolors="k", linewidths=0.3, zorder=5, label="traj pt")
    return lc


def per_anchor_metrics(pos, cbar, nbar, r2bar, gamma, huber_delta=1.0,
                          min_minor_axis=0.05):
    """Match guidance indices: p = pos[:, 1:H-1], cbar/nbar/r2bar/gamma = [:, 0:H-2]."""
    H = pos.shape[0]
    p = pos[1:H - 1]                       # [H-2,2]
    k = slice(0, H - 2)
    cb = cbar[k]; nb = nbar[k]; r2 = r2bar[k]; ga = gamma[k]
    d_perp = nb[:, 0] * (p[:, 0] - cb[:, 0]) + nb[:, 1] * (p[:, 1] - cb[:, 1])
    r2_eff = np.maximum(r2, min_minor_axis)
    delta = d_perp / (r2_eff + 1e-8)
    rho = np.where(np.abs(delta) <= huber_delta,
                   0.5 * delta * delta,
                   huber_delta * (np.abs(delta) - 0.5 * huber_delta))
    J = float((ga * rho).mean())
    stats = {
        "J_guide": J,
        "mean_gamma": float(ga.mean()),
        "median_gamma": float(np.median(ga)),
        "mean_d_perp": float(np.abs(d_perp).mean()),
        "mean_abs_delta": float(np.abs(delta).mean()),
        "frac_saturated": float((np.abs(delta) > huber_delta).mean()),
        "max_abs_delta": float(np.abs(delta).max()),
    }
    return p, d_perp, delta, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze", default="large", choices=["umaze", "medium", "large"])
    ap.add_argument("--ckpt", default="outputs/ckpt/joint/best.pt")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--convex", action="store_true",
                    help="also overlay the analytical convex safe regions")
    ap.add_argument("--compare", action="store_true",
                    help="also sample with guidance disabled and overlay both")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    case_rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extent = next(m["extent"] for m in cfg["mazes"] if m["name"] == args.maze)
    frame, occ, sdf, conds = load_maze_scene(args.maze, split="test", n=args.n,
                                             rng=case_rng, extent=extent)
    cond_t = torch.as_tensor(conds, dtype=torch.float32).to(device)
    model = Planner(cfg["model"], cfg["geometry"], None).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"]["beta_schedule"],
                             beta_start=cfg["diffusion"]["beta_start"],
                             beta_end=cfg["diffusion"]["beta_end"]).to(device)
    map_t = torch.as_tensor(occ, dtype=torch.float32).to(device)[None, None]
    sdf_t = torch.as_tensor(sdf, dtype=torch.float32).to(device)[None, None]

    gcfg = cfg.get("consensus_guidance", {})
    # guidance ON
    pos_on, traj_log, gstats = sample(model, map_t, schedule, cond_t, args.n,
                                       device=device, steps=args.steps,
                                       guidance_cfg=gcfg, return_guidance=True)
    pos_on = pos_on.cpu().numpy()
    # guidance OFF
    gcfg_off = dict(gcfg); gcfg_off["enabled"] = False
    pos_off, _, _ = sample(model, map_t, schedule, cond_t, args.n, device=device,
                           steps=args.steps, guidance_cfg=gcfg_off, return_guidance=True)
    pos_off = pos_off.cpu().numpy()

    c, r1, r2, th = scene_ellipse_prediction(torch.as_tensor(pos_on).to(device), cond_t, map_t, model, device)
    # consensus geometry in numpy
    ct = torch.as_tensor(c); rt = torch.as_tensor(np.stack([r1, r2], -1)); tt = torch.as_tensor(th)
    geom = compute_consensus_geometry(ct, rt, tt, window=int(gcfg.get("window_radius", 2)),
                                      pos_weights=gcfg.get("positional_weights", [1,2,4,2,1]),
                                      use_curvature_gate=bool(gcfg.get("use_curvature_gate", False)),
                                      curvature_kappa=float(gcfg.get("curvature_kappa", 1.5)))
    cbar = geom["cbar"].cpu().numpy(); ubar = geom["ubar"].cpu().numpy()
    nbar = geom["nbar"].cpu().numpy(); r2bar = geom["r2bar"].cpu().numpy()
    gamma = geom["gamma"].cpu().numpy(); q = geom["q"].cpu().numpy()

    # collision metrics
    def coll_rate(pos):
        out = []
        for i in range(args.n):
            d = sample_sdf_scene(sdf_t, torch.as_tensor(pos[i], dtype=torch.float32).to(device)[None]).cpu().numpy()[0]
            out.append(float(np.mean(d <= 0.0)))
        return out
    coll_on = coll_rate(pos_on); coll_off = coll_rate(pos_off)

    # ---- figure 1: centerline + deviation ----
    ncol = 2; nrow = max(1, (args.n + 1) // 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6 * nrow))
    axes = np.array(axes).reshape(-1)
    allstats = []
    for ax, i in zip(axes.ravel(), range(args.n)):
        ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r", alpha=0.9)
        draw_traj(ax, pos_on[i], marker_every=0, arrow_every=0)
        for j in range(0, len(pos_on[i]), 8):
            if not np.isfinite(c[i][j]).all() or not np.isfinite(r1[i][j] + r2[i][j]) or r1[i][j] <= 0:
                continue
            e = Ellipse((c[i][j][0], c[i][j][1]), 2 * r1[i][j], 2 * r2[i][j],
                        angle=np.degrees(th[i][j]), fill=False, edgecolor="tab:red",
                        lw=1.0, alpha=0.7, zorder=2)
            ax.add_patch(e)
        if args.convex:
            for j in range(0, len(pos_on[i]), 8):
                if not np.isfinite(c[i][j]).all() or not np.isfinite(r1[i][j] + r2[i][j]) or r1[i][j] <= 0:
                    continue
                A, b, verts = infer_convex_region_from_scene_occupancy(
                    occ, c[i][j], r1[i][j], r2[i][j], th[i][j])
                if verts is not None and len(verts) >= 3:
                    ax.add_patch(Polygon(verts, closed=True, facecolor="lightgreen",
                                         edgecolor="green", alpha=0.35, lw=0.8, zorder=1))
        p, d_perp, delta, stats = per_anchor_metrics(pos_on[i], cbar[i], nbar[i],
                                                     r2bar[i], gamma[i],
                                                     huber_delta=float(gcfg.get("huber_delta", 1.0)),
                                                     min_minor_axis=float(gcfg.get("min_minor_axis", 0.05)))
        allstats.append(stats)
        draw_centerline(ax, cbar[i], ubar[i], nbar[i], r2bar[i], gamma[i], p)
        ax.set_title(f"{args.maze} #{i} coll={coll_on[i]:.3f} "
                     f"Jg={stats['J_guide']:.3f} |d|={stats['mean_d_perp']:.3f} "
                     f"γ={stats['mean_gamma']:.2f}", fontsize=9)
        ax.set_aspect("equal"); set_map_limits(ax, (-1, 1, -1, 1))
        ax.legend(fontsize=5, loc="upper right")
    for k in range(args.n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"consensus centerline guidance — {args.maze} (scene [-1,1]^2)", fontsize=13)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out1 = f"outputs/centerline_guide_{args.maze}.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", out1)

    # ---- figure 2: guidance ON vs OFF ----
    if args.compare:
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6 * nrow))
        axes = np.array(axes).reshape(-1)
        for ax, i in zip(axes.ravel(), range(args.n)):
            ax.imshow(occ, origin="lower", extent=(-1, 1, -1, 1), cmap="gray_r", alpha=0.9)
            draw_traj(ax, pos_off[i], marker_every=0, arrow_every=0, alpha=0.55, lw=1.6, label="guidance OFF")
            draw_traj(ax, pos_on[i], marker_every=0, arrow_every=0, lw=2.2, label="guidance ON")
            from matplotlib.lines import Line2D
            handles = [Line2D([], [], color="tab:gray", lw=1.6, label="guidance OFF"),
                       Line2D([], [], color="tab:blue", lw=2.2, label="guidance ON")]
            ax.set_title(f"{args.maze} #{i} coll_off={coll_off[i]:.3f} coll_on={coll_on[i]:.3f}", fontsize=9)
            ax.set_aspect("equal"); set_map_limits(ax, (-1, 1, -1, 1))
            ax.legend(handles=handles, fontsize=6, loc="upper right")
        for k in range(args.n, len(axes)):
            axes[k].axis("off")
        fig.suptitle(f"consensus guidance ON vs OFF — {args.maze}", fontsize=13)
        fig.tight_layout()
        out2 = f"outputs/centerline_compare_{args.maze}.png"
        fig.savefig(out2, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("saved", out2)

    # ---- summary ----
    print("\n=== per-sample consensus-centerline metrics (guidance ON) ===")
    for i, st in enumerate(allstats):
        print(f"#{i}: J_guide={st['J_guide']:.4f} mean_gamma={st['mean_gamma']:.3f} "
              f"median_gamma={st['median_gamma']:.3f} mean|d_perp|={st['mean_d_perp']:.4f} "
              f"mean|delta|={st['mean_abs_delta']:.4f} frac_sat={st['frac_saturated']:.2f} "
              f"max|delta|={st['max_abs_delta']:.3f}")
    print("\n=== guidance w_G / shift over reverse steps ===")
    if gstats:
        ws = [s.get('w_G', 0.0) for s in gstats]
        sh = [s.get('shift_norm', 0.0) for s in gstats]
        print("num active steps:", len(ws), " first:", ws[0], " last:", ws[-1])
        print("max w_G:", max(ws) if ws else 0.0, " max shift_norm:", max(sh) if sh else 0.0)
    print("\n=== collision (frac points in wall, guidance ON vs OFF) ===")
    for i in range(args.n):
        print(f"#{i}: off={coll_off[i]:.3f}  on={coll_on[i]:.3f}")
    print("seed =", args.seed)


if __name__ == "__main__":
    main()
