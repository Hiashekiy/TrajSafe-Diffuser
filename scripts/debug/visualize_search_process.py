"""Visualise what the step-1 path search actually does, on real test samples.

Two figures per sample:

  search_fields_<id>.png   ingredients + outcome (8 panels)
      1) occupancy + start/goal + the step-1 prediction (= guidance reference)
      2) guidance distance field d (m)      -> how far each cell is from the reference
      3) guidance progress field u (0..1)   -> where along the reference it projects
      4) geodesic h field (m to the goal)   -> the heuristic the search really uses
      5) every expanded cell, coloured by expansion order, + the searched path
      6) h of the expanded cells vs expansion index (the descent)
      7) raw grid path vs the centred path that feeds active_path
      8) stage timings of this search branch

  search_process_<id>.png  the frontier growing over expansion snapshots.

    python scripts/debug/visualize_search_process.py --ids 1004,9008
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

from src.datasets.carla_spline_dataset import make_loader          # noqa: E402
from src.diffusion.schedule import NoiseSchedule                     # noqa: E402
from src.models.factory import build_model                           # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402

fa = importlib.import_module("src.grid_search.focal_astar")
dp = importlib.import_module("src.grid_search.diffusion_path")
lg = importlib.import_module("src.grid_search.lazy_guidance")
pc = importlib.import_module("src.grid_search.path_centering")
gh = importlib.import_module("src.grid_search.geodesic_heuristic")

SNAP_FRAMES = 8


def masked(img, free):
    out = np.array(img, dtype=np.float64)
    out[~free] = np.nan
    return out


def draw_map(ax, occ):
    ax.imshow(occ, origin="lower", cmap="gray_r", alpha=0.55,
              interpolation="nearest")
    h, w = occ.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q128.yaml")
    ap.add_argument("--ckpt", default="outputs/models/v5_c32q128/ckpt/final_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default="1004,9008")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="outputs/search_vis")
    args = ap.parse_args()

    from src.diffusion.sampler_v5 import _build_safe_grids

    cfg = load_config(args.config)
    gcfg = dict(cfg.get("grid_search") or {})
    os.makedirs(os.path.join(ROOT, args.outdir), exist_ok=True)
    want = [int(x) for x in args.ids.split(",")]
    loader, _ = make_loader(
        args.split, cfg["data"]["processed_root"], batch_size=1, shuffle=False,
        num_workers=0, limit=None, require_labels=False,
        num_controls=num_controls(cfg), num_safety_queries=num_safety_queries(cfg))
    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                             "squaredcos_cap_v2")).to(args.device)
    model = build_model(cfg).to(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()

    for batch in loader:
        sid = int(batch["sample_id"][0])
        if sid not in want:
            continue
        cond = batch["cond"].to(args.device)
        occ_t = batch["occupancy"].to(args.device)
        heading = batch.get("heading")
        hv = batch.get("heading_valid")
        heading = heading.to(args.device) if heading is not None else None
        hv = hv.to(args.device) if hv is not None else None
        occ = occ_t[0, 0].cpu().numpy().astype(np.uint8)

        grids, spec, grid_ms = _build_safe_grids(occ_t, cfg)
        safe, nbr = grids[0]
        start_w = cond[0, 0].cpu().numpy().astype(np.float64)
        goal_w = cond[0, 1].cpu().numpy().astype(np.float64)

        # ---- step-1 prediction = the guidance reference --------------------
        T = int(sched.num_timesteps)
        q = torch.randn(1, int(model.num_controls), 2, device=args.device)
        q[:, 0] = cond[:, 0]
        q[:, -1] = cond[:, -1]
        tb = torch.full((1,), T - 1, device=args.device, dtype=torch.long)
        ab = sched.sqrt_alphas_cumprod[T - 1].expand(1).to(args.device)
        with torch.no_grad():
            out = model(q, occ_t, cond, tb, ab, active_path=None, feedback_delta=None,
                        feedback_valid=None, heading=heading, heading_valid=hv)
        curve = out["final"][0].detach().cpu().numpy().astype(np.float64)

        # ---- reference + the field the search really sees ------------------
        t0 = time.perf_counter()
        ref = dp.prepare_reference_path(curve, spec, start_world=start_w,
                                        goal_world=goal_w)
        prepare_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        field = lg.LazyGuidance(ref, spec, J=int(gcfg.get("J", 3)))
        lazy_ms = (time.perf_counter() - t0) * 1000.0
        # full-grid values straight from the lazy field (no coverage radius, so
        # the drawing shows the same numbers the search would get)
        d_all, u_all = field._values(np.arange(spec.height * spec.width))
        d0 = np.asarray(d_all)[:, 0].reshape(spec.height, spec.width)
        u0 = np.asarray(u_all)[:, 0].reshape(spec.height, spec.width)

        s_idx = int(spec.cell_to_index(spec.world_to_cell(start_w)))
        g_idx = int(spec.cell_to_index(spec.world_to_cell(goal_w)))
        goal_rc = spec.index_to_rc(np.array([g_idx]))[0]
        t0 = time.perf_counter()
        h_tab = gh.geodesic_distance_field(safe.free, float(spec.cell_m), goal_rc)
        geo_ms = (time.perf_counter() - t0) * 1000.0
        h_img = h_tab.reshape(spec.height, spec.width) if h_tab is not None else None

        # ---- the search, with exactly the production config ----------------
        search_cfg = fa.SearchConfig(
            w=float(gcfg.get("focal_w", 1.0)), heuristic="octile",
            secondary=str(gcfg.get("secondary", "guidance")),
            guidance=field, alpha_d=1.0,
            alpha_p=float(gcfg.get("alpha_p", 1.0)), h_table=h_tab,
            h_tie_tol_cells=float(gcfg.get("h_tie_tol_cells", 0.0)),
            snapshots=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000))
        t0 = time.perf_counter()
        res = fa.focal_search(safe, s_idx, g_idx, search_cfg, nbr)
        search_ms = (time.perf_counter() - t0) * 1000.0
        print("[vis] sample %d: %s expanded=%d cost=%.1f m  search %.1f ms"
              % (sid, res.status, res.stats["expanded_nodes"], res.cost_m, search_ms))

        # ---- path -> world -> 128 points -> centred ------------------------
        rc_path = spec.index_to_rc(res.path)
        pts_w = spec.cell_to_world(rc_path.astype(np.float64)[:, ::-1])
        pts_w[0], pts_w[-1] = start_w, goal_w
        seg = np.linalg.norm(np.diff(pts_w, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.linspace(0.0, cum[-1], 128)
        resampled = np.stack([np.interp(s, cum, pts_w[:, k]) for k in range(2)], -1)
        resampled[0], resampled[-1] = start_w, goal_w
        t0 = time.perf_counter()
        centered, cen_st = pc.center_path(
            resampled, safe.raw_free, float(spec.cell_m),
            iters=int((gcfg.get("center_path") or {}).get("iters", 40)),
            clearance=getattr(safe, "raw_clearance_m", None))
        center_ms = (time.perf_counter() - t0) * 1000.0

        def to_px(p):
            return (np.asarray(p) + 1.0) / 2.0 * spec.width - 0.5

        start_px, goal_px = to_px(start_w), to_px(goal_w)
        rc_exp = spec.index_to_rc(res.expanded)
        free2 = safe.free

        # one shared zoom box so the thin corridors are readable
        _all = np.vstack([start_px, goal_px, to_px(ref.points), to_px(pts_w),
                          to_px(centered)])
        pad = 18.0
        vx0 = float(_all[:, 0].min()) - pad
        vx1 = float(_all[:, 0].max()) + pad
        vy0 = float(_all[:, 1].min()) - pad
        vy1 = float(_all[:, 1].max()) + pad
        # make the box square so an equal-aspect map fills the panel
        side = max(vx1 - vx0, vy1 - vy0)
        cx, cy = 0.5 * (vx0 + vx1), 0.5 * (vy0 + vy1)
        vx0, vx1 = cx - side / 2, cx + side / 2
        vy0, vy1 = cy - side / 2, cy + side / 2
        vx0 = max(-0.5, vx0); vx1 = min(spec.width - 0.5, vx1)
        vy0 = max(-0.5, vy0); vy1 = min(spec.height - 0.5, vy1)

        def set_view(ax):
            ax.set_xlim(vx0, vx1)
            ax.set_ylim(vy0, vy1)

        # =====================================================================
        fig, axes = plt.subplots(2, 4, figsize=(22.0, 10.4))
        axes = axes.ravel()

        ax = axes[0]
        draw_map(ax, occ)
        ax.plot(*to_px(curve).T, color="#ff7f0e", lw=1.7,
                label="step-1 prediction x0 (guidance reference)")
        ax.plot(*to_px(ref.points).T, color="#17becf", lw=1.0, ls="--",
                label="resampled reference")
        ax.scatter(*start_px, marker="*", s=220, c="#2ca02c", zorder=6, label="start")
        ax.scatter(*goal_px, marker="*", s=220, c="#d62728", zorder=6, label="goal")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        set_view(ax)
        ax.set_title("1) occupancy + prediction x0 = the guidance reference", fontsize=9)

        ax = axes[1]
        draw_map(ax, occ)
        im = ax.imshow(masked(d0, free2), origin="lower", cmap="viridis", alpha=0.85)
        ax.plot(*to_px(ref.points).T, color="w", lw=1.0)
        set_view(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("d (m)", fontsize=7)
        ax.set_title("2) guidance distance d (m): how far from the reference", fontsize=9)

        ax = axes[2]
        draw_map(ax, occ)
        im = ax.imshow(masked(u0, free2), origin="lower", cmap="plasma", alpha=0.85)
        ax.plot(*to_px(ref.points).T, color="w", lw=1.0)
        ax.scatter(*start_px, marker="*", s=90, c="w", edgecolors="k", zorder=6)
        ax.scatter(*goal_px, marker="*", s=90, c="r", edgecolors="k", zorder=6)
        set_view(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("u (0..1)", fontsize=7)
        ax.set_title("3) guidance progress u: projection along the reference", fontsize=9)

        ax = axes[3]
        draw_map(ax, occ)
        if h_img is not None:
            im = ax.imshow(masked(h_img, free2), origin="lower", cmap="coolwarm",
                           alpha=0.85)
            cs = ax.contour(np.where(free2, h_img, np.nan), levels=8, colors="k",
                            linewidths=0.4, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=5, fmt="%.0f")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("h (m)", fontsize=7)
        ax.scatter(*start_px, marker="*", s=90, c="w", edgecolors="k", zorder=6)
        ax.scatter(*goal_px, marker="*", s=90, c="r", edgecolors="k", zorder=6)
        set_view(ax)
        ax.set_title("4) geodesic h (m): TRUE remaining cost -> admissible", fontsize=9)

        ax = axes[4]
        draw_map(ax, occ)
        sc = ax.scatter(rc_exp[:, 1], rc_exp[:, 0], c=np.arange(len(rc_exp)),
                        cmap="plasma", s=11, marker="s", linewidths=0, alpha=0.95)
        ax.plot(*to_px(pts_w).T, color="#1f77b4", lw=2.0, label="searched path")
        ax.scatter(*start_px, marker="*", s=180, c="#2ca02c", zorder=6)
        ax.scatter(*goal_px, marker="*", s=180, c="#d62728", zorder=6)
        ax.legend(loc="upper right", fontsize=7)
        set_view(ax)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label("expansion order",
                                                                   fontsize=7)
        ax.set_title("5) %d expanded cells, coloured by expansion order" % len(rc_exp),
                     fontsize=9)

        ax = axes[5]
        order = np.arange(len(rc_exp))
        h_cells = (h_tab.reshape(-1)[res.expanded] if h_tab is not None
                   else np.zeros(len(rc_exp)))
        ax.plot(order, h_cells, color="#1f77b4", lw=1.0)
        ax.axhline(res.cost_m, color="#d62728", ls="--", lw=1.0,
                   label="path cost %.1f m" % res.cost_m)
        ax.set_xlabel("expansion index", fontsize=8)
        ax.set_ylabel("h of the expanded cell (m)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title("6) descent: remaining cost of the cell the search chose", fontsize=9)

        ax = axes[6]
        draw_map(ax, occ)
        ax.plot(*to_px(pts_w).T, color="#1f77b4", lw=1.3, label="raw grid path (A*)")
        ax.plot(*to_px(centered).T, color="#9467bd", lw=1.9, ls="--",
                label="centred -> active_path")
        ax.scatter(*start_px, marker="*", s=180, c="#2ca02c", zorder=6)
        ax.scatter(*goal_px, marker="*", s=180, c="#d62728", zorder=6)
        ax.legend(loc="upper right", fontsize=7)
        set_view(ax)
        ax.set_title("7) centring: clearance %.2f -> %.2f m"
                     % (cen_st.get("clearance_before") or float("nan"),
                        cen_st.get("clearance_after") or float("nan")), fontsize=9)

        ax = axes[7]
        names = ["geo field h", "reference prepare", "LazyGuidance build",
                 "search loop", "grid build (2 EDT+nbr)", "centring (40 iters)"]
        vals = [geo_ms, prepare_ms, lazy_ms, search_ms, grid_ms, center_ms]
        order_i = np.argsort(vals)
        bars = ax.barh([names[i] for i in order_i], [vals[i] for i in order_i],
                       color=["#4c72b0", "#dd8452", "#55a868", "#c44e52",
                              "#8172b3", "#937860"])
        vmax = max(vals)
        for b, i in zip(bars, order_i):
            ax.text(vals[i] + vmax * 0.02, b.get_y() + b.get_height() / 2,
                    "%.1f ms" % vals[i], va="center", fontsize=7)
        ax.set_xlabel("ms (single sample, batch=1)", fontsize=8)
        ax.set_xlim(0, vmax * 1.22)
        ax.set_title("8) where this search branch spends its time", fontsize=9)

        for ax in axes[:7]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("v5 step-1 search - sample %d   status=%s   expanded=%d   "
                     "path %.1f m   search %.1f ms"
                     % (sid, res.status, res.stats["expanded_nodes"], res.cost_m,
                        search_ms), fontsize=11)
        fig.tight_layout()
        p1 = os.path.join(ROOT, args.outdir, "search_fields_%d.png" % sid)
        fig.savefig(p1, dpi=115, bbox_inches="tight")
        plt.close(fig)

        # =====================================================================
        snaps = sorted(res.snapshots.items())
        if snaps:
            step = max(1, len(snaps) // SNAP_FRAMES)
            picks = snaps[::step][:SNAP_FRAMES - 1]
            # always end on the COMPLETE expansion set
            picks.append((res.stats["expanded_nodes"], res.expanded))
            ncol = len(picks)
            fig, axes2 = plt.subplots(1, ncol, figsize=(3.2 * ncol, 4.0))
            axes2 = np.atleast_1d(axes2)
            for ax, (k, expo) in zip(axes2, picks):
                draw_map(ax, occ)
                rc_e = spec.index_to_rc(expo)
                ax.scatter(rc_e[:, 1], rc_e[:, 0], s=8, marker="s", linewidths=0,
                           c=np.arange(len(rc_e)), cmap="plasma", alpha=0.95)
                ax.scatter(*start_px, marker="*", s=130, c="#2ca02c", zorder=6)
                ax.scatter(*goal_px, marker="*", s=130, c="#d62728", zorder=6)
                ax.set_xlim(vx0, vx1)
                ax.set_ylim(vy0, vy1)
                ax.set_title("after %d expansions" % k, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle("sample %d - the search frontier growing (colour = expansion "
                         "order; the whole search is %d expansions)"
                         % (sid, res.stats["expanded_nodes"]), fontsize=10)
            fig.tight_layout()
            p2 = os.path.join(ROOT, args.outdir, "search_process_%d.png" % sid)
            fig.savefig(p2, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print("[vis] wrote", p1, "and", p2)
        else:
            print("[vis] wrote", p1)


if __name__ == "__main__":
    main()
