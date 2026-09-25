#!/usr/bin/env python
"""plot_c48_regions_342.py - draw the 128 convex regions the ALM would use.

For sample 342 (the last surviving C=48 test failure) the corridor never closes:
``build_safety_corridor`` builds the 128 regions but rejects NINE of them
(``center_inside = False``), and one rejected base cell voids the whole corridor
(``invalid_base_region:113``) -> the pack stays empty -> the ALM never runs.

This renders, at the end of the official test protocol (test / chunk 32 / 16
steps / seed 0 / ``K4P_c48_oneshot:best_task``):

  A  the whole crop with ALL 128 region polygons (green = valid, red = rejected),
     plus GT / prediction / start / goal;
  B  a zoom on the rejected cluster: the polygons, their ellipse centres, and
     the obstacle cells;
  C  ONE rejected cell blown up, with the face that cuts its own centre out:
     the face line, the obstacle boundary point that generated it, and the two
     distances that make it invalid (available slack |Q d| vs safety_margin).

    python scripts/plot/plot_c48_regions_342.py --out outputs/fig_c48_failure
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from scipy.spatial import ConvexHull                  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate  # noqa: E402
from src.diffusion.sampler import sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.convex_region import EllipseRegionBuilder  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402

CFG = "configs/config_160k4p_c48_oneshot.yaml"
CKPT = "outputs/oneshot_k4p_c48/ckpt/best_task.pt"
CHUNK, STEPS, SEED, RES = 32, 16, 0, 256
CELL = 2.0 / RES
SCENE_TO_M = 80.0


def polygon_from_halfspaces(A: np.ndarray, b: np.ndarray, tol: float = 1e-6):
    """Vertices of ``{x : A x <= b}`` by pair enumeration (no interior point).

    scipy's HalfspaceIntersection needs a point strictly INSIDE; the whole point
    of this figure is that some cells do not contain their own centre, so the
    interior point cannot be used.
    """
    pts = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            M = np.stack([A[i], A[j]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            p = np.linalg.solve(M, b[[i, j]])
            if (A @ p - b).max() <= tol:
                pts.append(p)
    if len(pts) < 3:
        return None
    pts = np.asarray(pts)
    # float vertices repeat across face pairs: merge them before the hull
    keep = []
    for q in pts:
        if not keep or np.min(np.linalg.norm(np.asarray(keep) - q, axis=1)) > 1e-7:
            keep.append(q)
    pts = np.asarray(keep)
    if len(pts) < 3:
        return None
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:                                     # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/fig_c48_failure")
    ap.add_argument("--index", type=int, default=342)
    ap.add_argument("--step", type=int, default=-1,
                    help="trace step to draw (-1 = last)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = load_config(CFG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CarlaSplineDataset("test", cfg["data"]["processed_root"],
                            geometry_points=1280, num_controls=num_controls(cfg),
                            num_safety_queries=num_safety_queries(cfg))
    collate = make_collate(ds)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2"),
                             beta_start=cfg["diffusion"].get("beta_start", 1e-4),
                             beta_end=cfg["diffusion"].get("beta_end", 0.02)).to(device)
    model, _, _ = load_model(cfg, CKPT, device=device, verbose=False)
    lo = (args.index // CHUNK) * CHUNK
    hi = min(lo + CHUNK, len(ds))
    sub = collate([ds[i] for i in range(lo, hi)])
    with torch.no_grad():
        out = sample(model, schedule, sub["cond"], sub["occupancy"],
                     sub["candidate_xy"], sub["candidate_mask"],
                     sub["candidate_geometry"], sub["candidate_geometry_lengths"],
                     device=device, steps=STEPS, seed=SEED, return_trace=True,
                     alm_config=cfg.get("alm"), corridor_config=cfg.get("corridor"))
    j = args.index - lo
    tr = out["trace"][args.step]

    occ = sub["occupancy"][j, 0].float().cpu().numpy()
    wall = occ > 0.5
    cond = sub["cond"][j].float().cpu().numpy()
    gt = sub["pos"][j].float().cpu().numpy()
    pred = tr["final"][j].float().cpu().numpy()

    builder = EllipseRegionBuilder(sub["occupancy"][j:j + 1], dict(cfg.get("corridor")))
    A, b, mask, valid, diag = builder.build_from_ellipse(
        tr["ellipse_center"][j][None], tr["ellipse_shape4"][j][None],
        return_diagnostics=True)
    A, b, mask, valid = A[0].cpu().numpy(), b[0].cpu().numpy(), \
        mask[0].cpu().numpy(), valid[0].cpu().numpy()
    centers = tr["ellipse_center"][j].float().cpu().numpy()
    viol = diag["max_center_violation"][0].cpu().numpy()
    bad = np.where(~valid)[0]
    print("[regions] %d/128 valid, rejected %s" % (int(valid.sum()), bad.tolist()))
    print("[regions] rejection reason: center inside = %d/%d, bounded = %d/%d"
          % (int(diag["center_inside"][0].sum()), len(valid),
             int(diag["bounded"][0].sum()), len(valid)))

    polys, bad_polys = {}, {}
    for i in range(len(valid)):
        f = mask[i]
        if f.sum() < 3:
            continue
        poly = polygon_from_halfspaces(A[i][f], b[i][f])
        if poly is None:
            continue
        (polys if valid[i] else bad_polys)[i] = poly

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.0),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0]))

    # ---------------- A: everything ---------------------------------------
    ax = axes[0]
    ax.imshow(wall, origin="lower", extent=[-1, 1, -1, 1], cmap="Greys",
              vmin=0, vmax=1, alpha=0.75, interpolation="nearest", zorder=0)
    for i, poly in polys.items():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor="#2fb9ff", alpha=0.16,
                                edgecolor="#2fb9ff", lw=0.6, zorder=2))
    for i, poly in bad_polys.items():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor="#ff2f2f", alpha=0.30,
                                edgecolor="#ff2f2f", lw=1.1, zorder=3))
    ax.plot(gt[:, 0], gt[:, 1], "--", color="#ffcf5a", lw=1.8, label="GT", zorder=5)
    ax.plot(pred[:, 0], pred[:, 1], "-", color="#c7ff4a", lw=2.4,
            label="prediction (16 steps)", zorder=6)
    ax.scatter([cond[0, 0]], [cond[0, 1]], s=80, c="#3ddc84", edgecolors="k", zorder=7)
    ax.scatter([cond[1, 0]], [cond[1, 1]], s=100, marker="X", c="#ff4d4d",
               edgecolors="k", zorder=7)
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("A · all 128 convex regions   green = valid (%d) · red = REJECTED (%d)"
                 % (len(polys), len(bad_polys)), fontsize=10)

    # ---------------- B: the rejected cluster ------------------------------
    ax = axes[1]
    cx_all = centers[bad]
    x0, x1 = cx_all[:, 0].min() - 0.12, cx_all[:, 0].max() + 0.12
    y0, y1 = cx_all[:, 1].min() - 0.12, cx_all[:, 1].max() + 0.12
    ax.imshow(wall, origin="lower", extent=[-1, 1, -1, 1], cmap="Greys",
              vmin=0, vmax=1, alpha=0.75, interpolation="nearest", zorder=0)
    for i, poly in bad_polys.items():
        ax.add_patch(MplPolygon(poly, closed=True, facecolor="#ff2f2f", alpha=0.22,
                                edgecolor="#ff2f2f", lw=1.4, zorder=2))
        ax.text(poly[:, 0].mean(), poly[:, 1].mean(), str(i), fontsize=7,
                color="#ff2f2f", ha="center", va="center", zorder=5)
    for i in bad:
        ax.plot(centers[i, 0], centers[i, 1], "x", color="#0b6", ms=7, mew=2, zorder=6)
    ax.plot(pred[:, 0], pred[:, 1], "-", color="#c7ff4a", lw=2.2, zorder=4)
    ax.plot(gt[:, 0], gt[:, 1], "--", color="#ffcf5a", lw=1.4, zorder=4)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("B · the rejected cells   × = their ellipse centre\n"
                 "(a cell is rejected when its own centre violates one of its faces)",
                 fontsize=10)

    # ---------------- C: one cell, one face --------------------------------
    ax = axes[2]
    i_bad = int(bad[np.argmax(viol[bad])])
    f = mask[i_bad]
    A_i, b_i = A[i_bad][f], b[i_bad][f]
    c = centers[i_bad]
    v = A_i @ c - b_i
    k = int(np.argmax(v))                      # the face that cuts the centre out
    n_k, b_k = A_i[k], b_i[k]
    obs = n_k * (b_k + builder.margin)         # the obstacle boundary point it came from
    slack = float(n_k @ (obs - c))             # = margin - violation
    poly = bad_polys.get(i_bad)
    ax.imshow(wall, origin="lower", extent=[-1, 1, -1, 1], cmap="Greys",
              vmin=0, vmax=1, alpha=0.8, interpolation="nearest", zorder=0)
    if poly is not None:
        ax.add_patch(MplPolygon(poly, closed=True, facecolor="#ff2f2f", alpha=0.25,
                                edgecolor="#ff2f2f", lw=1.6, zorder=2))
    pad = 0.10
    ax.set_xlim(c[0] - pad, c[0] + pad)
    ax.set_ylim(c[1] - pad, c[1] + pad)
    # face line and the normal direction from the centre
    L = np.array([-n_k[1], n_k[0]])
    s = np.linspace(-0.25, 0.25, 2)
    pts = obs[None] + s[:, None] * L[None]
    ax.plot(pts[:, 0], pts[:, 1], "-", color="#ff8c00", lw=1.8, zorder=4,
            label="the cutting face  n·x = b")
    ax.plot([c[0], obs[0]], [c[1], obs[1]], ":", color="#0b6", lw=1.6, zorder=4)
    ax.plot(*obs, "o", color="#ff8c00", ms=6, zorder=5,
            label="obstacle boundary point obs")
    ax.plot(*c, "x", color="#0b6", ms=10, mew=2.4, zorder=6,
            label="ellipse centre (must be INSIDE)")
    ax.arrow(c[0], c[1], n_k[0] * slack, n_k[1] * slack, color="#0b6",
             width=0.0015, head_width=0.006, length_includes_head=True, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("C · rejected cell #%d  —  why\n"
                 "slack |Q·d| = %.4f scene = %.2f m   <   safety_margin = %.4f scene = %.2f m"
                 % (i_bad, slack, slack * SCENE_TO_M, builder.margin,
                    builder.margin * SCENE_TO_M), fontsize=10)
    ax.text(0.02, 0.02,
            "obs is %.2f m from the centre, but the face is placed\n"
            "safety_margin = %.2f m BEYOND obs toward the centre\n"
            "→ the plane passes behind the centre\n"
            "→ centre_inside = False → the WHOLE corridor is voided"
            % (slack * SCENE_TO_M, builder.margin * SCENE_TO_M),
            transform=ax.transAxes, fontsize=7.5, color="#222",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=3))

    fig.suptitle("sample %s · 128 convex regions at the end of the 16-step test protocol "
                 "— %d valid, %d rejected  (alm_status = %s, corridor = EMPTY)"
                 % ("test_%04d" % args.index, len(polys), len(bad_polys),
                    out["alm_status"][j]), fontsize=11.5, y=0.975)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.885, bottom=0.03, wspace=0.04)
    path = os.path.join(args.out, "c48_regions_%d.png" % args.index)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("[regions] wrote %s" % path)
    with open(os.path.join(args.out, "c48_regions_%d.json" % args.index), "w",
              encoding="utf-8") as fh:
        json.dump(dict(index=args.index, n_valid=int(valid.sum()),
                       rejected=[int(i) for i in bad],
                       center_inside=int(diag["center_inside"][0].sum()),
                       bounded=int(diag["bounded"][0].sum()),
                       margin=float(builder.margin),
                       worst_cell=i_bad, worst_violation=float(v[k]),
                       worst_slack=slack,
                       alm_status=str(out["alm_status"][j])), fh, indent=2)


if __name__ == "__main__":
    main()
