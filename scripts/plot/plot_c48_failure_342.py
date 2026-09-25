#!/usr/bin/env python
"""plot_c48_failure_342.py - draw the last surviving C=48 test failure.

Sample 342 is the ONLY collision left after the two fixes (map-boundary
halfspaces + a larger ALM trust region) and it looks nothing like the others:

  * ``alm_status = activation_failed`` -> the corridor never closed, the
    constraint pack is EMPTY and the ALM never ran (per-sample violation = nan);
  * the collision is a single dense point out of 512 whose bilinear occupancy
    value is 0.532 -- a half-cell GRAZE, not a wall hit.

The figure replays the OFFICIAL test protocol (test split, chunk 32, 16 reverse
steps, seed 0, ``K4P_c48_oneshot:best_task`` on the ``160k4p_c48`` cache) so the
curve is the eval's own realization, and renders:

  A  the whole 256^2 crop: occupancy, GT curve, predicted curve, the grazing
     point, start/goal;
  B  a zoom on the grazing point, drawn on the CELL GRID, with the obstacle
     cells and the 0.5 iso-contour of the bilinear occupancy -- the CONTOUR is
     the real collision criterion, and the curve only touches it because the
     bilinear test inflates every obstacle by up to half a cell.

    python scripts/plot/plot_c48_failure_342.py --out outputs/fig_c48_failure
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
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Circle                # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate  # noqa: E402
from src.diffusion.sampler import _dense_decode, _free_mask, sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402

CFG = "configs/config_160k4p_c48_oneshot.yaml"
CKPT = "outputs/oneshot_k4p_c48/ckpt/best_task.pt"
INDEX, CHUNK, STEPS, SEED, RES, DENSE = 342, 32, 16, 0, 256, 512
CELL = 2.0 / RES
SCENE_TO_M = 80.0


def bilinear(occ: np.ndarray, p: np.ndarray) -> np.ndarray:
    """``F.grid_sample(occ, p, bilinear, border, align_corners=False)``."""
    gx = (p[:, 0] + 1.0) * (RES / 2.0) - 0.5
    gy = (p[:, 1] + 1.0) * (RES / 2.0) - 0.5
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - x0, gy - y0
    cx0, cx1 = np.clip(x0, 0, RES - 1), np.clip(x0 + 1, 0, RES - 1)
    cy0, cy1 = np.clip(y0, 0, RES - 1), np.clip(y0 + 1, 0, RES - 1)
    return (occ[cy0, cx0] * (1 - fx) * (1 - fy) + occ[cy0, cx1] * fx * (1 - fy)
            + occ[cy1, cx0] * (1 - fx) * fy + occ[cy1, cx1] * fx * fy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/fig_c48_failure")
    ap.add_argument("--index", type=int, default=INDEX)
    ap.add_argument("--zoom-cells", type=int, default=14)
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
    model, ckpt, _ = load_model(cfg, CKPT, device=device, verbose=True)
    lo = (args.index // CHUNK) * CHUNK
    hi = min(lo + CHUNK, len(ds))
    sub = collate([ds[i] for i in range(lo, hi)])
    with torch.no_grad():
        out = sample(model, schedule, sub["cond"], sub["occupancy"],
                     sub["candidate_xy"], sub["candidate_mask"],
                     sub["candidate_geometry"], sub["candidate_geometry_lengths"],
                     device=device, steps=STEPS, seed=SEED, return_trace=False,
                     alm_config=cfg.get("alm"), corridor_config=cfg.get("corridor"))
    j = args.index - lo
    q = out["control"][j:j + 1]
    p = _dense_decode(model.bspline, q, DENSE)[0].float().cpu().numpy()
    gt = sub["pos"][j].float().cpu().numpy()
    cond = sub["cond"][j].float().cpu().numpy()
    occ = sub["occupancy"][j, 0].float().cpu().numpy()
    wall = occ > 0.5
    free = _free_mask(sub["occupancy"][j].cpu(), torch.as_tensor(p)).numpy()
    val = bilinear(occ, p)
    hit = np.where(~free)[0]
    jh = int(hit[np.argmax(val[hit])]) if len(hit) else int(val.argmax())
    alm_status = out["alm_status"][j]
    print("[plot] %s  free_rate=%.6f  collisions=%d  alm_status=%s"
          % ("test_%04d" % args.index, float(free.mean()), int((~free).sum()), alm_status))

    px, py = p[jh]
    wy, wx = np.where(wall)
    cx = -1 + (wx + 0.5) * CELL
    cy = -1 + (wy + 0.5) * CELL
    d_min_scene = float(np.hypot(cx - px, cy - py).min())
    d_centre = d_min_scene * SCENE_TO_M
    d_edge = (d_min_scene - 0.5 * CELL) * SCENE_TO_M
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    # ---------------- panel A: the whole crop -----------------------------
    ax = axes[0]
    ax.imshow(wall, origin="lower", extent=[-1, 1, -1, 1], cmap="Greys",
              vmin=0, vmax=1, alpha=0.85, interpolation="nearest")
    ax.plot(gt[:, 0], gt[:, 1], "--", color="#ffcf5a", lw=2.0, label="GT curve")
    ax.plot(p[:, 0], p[:, 1], "-", color="#c7ff4a", lw=2.2, label="prediction (16 steps)")
    ax.scatter([cond[0, 0]], [cond[0, 1]], s=90, c="#3ddc84", edgecolors="k",
               zorder=5, label="start")
    ax.scatter([cond[1, 0]], [cond[1, 1]], s=110, marker="X", c="#ff4d4d",
               edgecolors="k", zorder=5, label="goal")
    ax.add_patch(Circle((px, py), 0.045, fill=False, ec="#ff2fd0", lw=1.8, zorder=6))
    ax.set_title("A · sample %s — full 256² crop (test split, 16 steps, seed 0)"
                 % ("test_%04d" % args.index), fontsize=10)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.99, 0.02, "occupancy (black = obstacle) · the predicted curve never "
                        "leaves the crop", transform=ax.transAxes, ha="right",
            fontsize=8, color="#444")

    # ---------------- panel B: the graze, on the cell grid ----------------
    ax = axes[1]
    r = args.zoom_cells
    # cell index of a scene coordinate: i = floor((x+1) * RES/2)  (NOT *RES)
    ci = int(np.floor((px + 1) * (RES / 2.0)))
    cj = int(np.floor((py + 1) * (RES / 2.0)))
    i0 = max(0, min(ci - r, RES - 2 * r - 1))
    j0 = max(0, min(cj - r, RES - 2 * r - 1))
    i1, j1 = i0 + 2 * r + 1, j0 + 2 * r + 1
    x_lo, x_hi = -1 + i0 * CELL, -1 + i1 * CELL
    y_lo, y_hi = -1 + j0 * CELL, -1 + j1 * CELL
    sub_wall = wall[j0:j1, i0:i1]
    print("[plot] zoom window cells x[%d,%d) y[%d,%d)  obstacle cells inside = %d"
          % (i0, i1, j0, j1, int(sub_wall.sum())))
    ax.imshow(sub_wall, origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.85,
              extent=[x_lo, x_hi, y_lo, y_hi], interpolation="nearest", zorder=0)
    # cell grid + the 0.5 iso-contour of the bilinear occupancy
    for i in range(i0, i1 + 1):
        ax.axvline(-1 + i * CELL, color="#bbbbbb", lw=0.4, zorder=1)
    for jj in range(j0, j1 + 1):
        ax.axhline(-1 + jj * CELL, color="#bbbbbb", lw=0.4, zorder=1)
    fg = np.linspace(x_lo, x_hi, 300)
    fh = np.linspace(y_lo, y_hi, 300)
    GX, GY = np.meshgrid(fg, fh)
    GV = bilinear(occ, np.stack([GX.ravel(), GY.ravel()], axis=-1)).reshape(GX.shape)
    cs = ax.contour(GX, GY, GV, levels=[0.5], colors=["#ff2fd0"], linewidths=2.0,
                    zorder=3)
    ax.clabel(cs, fmt="0.5", fontsize=8)
    ax.plot(p[:, 0], p[:, 1], "-", color="#c7ff4a", lw=2.4, zorder=4)
    ax.plot(gt[:, 0], gt[:, 1], "--", color="#ffcf5a", lw=1.6, zorder=4)
    ax.plot([cx[np.argmin(np.hypot(cx - px, cy - py))]],
            [cy[np.argmin(np.hypot(cx - px, cy - py))]], marker="s", ms=5,
            mfc="none", mec="#0b6", mew=1.4, zorder=6)
    ax.scatter([px], [py], s=70, facecolors="none", edgecolors="#ff2fd0",
               linewidths=2.0, zorder=6)
    ax.annotate("the ONE colliding point (of 512)\nbilinear value = %.3f" % float(val[jh]),
                (px, py), textcoords="offset points", xytext=(30, -46), fontsize=9,
                color="#ff2fd0", arrowprops=dict(arrowstyle="->", color="#ff2fd0", lw=1.4))
    ax.annotate("nearest obstacle cell", (cx[np.argmin(np.hypot(cx - px, cy - py))],
                                          cy[np.argmin(np.hypot(cx - px, cy - py))]),
                textcoords="offset points", xytext=(18, 26), fontsize=8, color="#0b6",
                arrowprops=dict(arrowstyle="->", color="#0b6", lw=1.2))
    rect = plt.Rectangle((x_lo, y_lo), CELL, CELL, fill=False, ec="#0b6", lw=1.2,
                         zorder=6)
    ax.add_patch(rect)
    ax.set_title("B · zoom on the graze (1 cell = %.2f m)" % (CELL * SCENE_TO_M),
                 fontsize=10)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.98, "pink contour = `free <=> bilinear(occ) <= 0.5`\n"
                        "black cells = the actual obstacles",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")

    # ---------------- caption block ---------------------------------------
    fig.suptitle(
        "C=48 k=4 test failure #%d  (%s)    ALM status = %s  →  no corridor, ALM never ran"
        % (args.index, "test_%04d" % args.index, alm_status), fontsize=11.5, y=0.975)
    fig.text(0.5, 0.005,
             "free_rate = %.6f  (%d/512 points flagged)   |   colliding point: bilinear = %.3f, "
             "%.2f m from the nearest obstacle-cell centre (%.2f m from its edge, 1 cell = %.2f m)   |   "
             "max |curve| = %.3f, so this is a half-cell GRAZE, not leaving the map"
             % (float(free.mean()), int((~free).sum()), float(val[jh]), d_centre, d_edge,
                CELL * SCENE_TO_M, float(np.abs(p).max())),
             ha="center", fontsize=8.6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.075, wspace=0.05)
    path = os.path.join(args.out, "c48_test_failure_%d.png" % args.index)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("[plot] wrote %s" % path)
    with open(os.path.join(args.out, "c48_test_failure_%d.json" % args.index), "w",
              encoding="utf-8") as fh:
        json.dump(dict(index=args.index, split="test", steps=STEPS, seed=SEED,
                       chunk=CHUNK, model=CKPT, config=CFG, alm_status=str(alm_status),
                       free_rate=float(free.mean()), n_flagged=int((~free).sum()),
                       bilinear_at_worst=float(val[jh]),
                       nearest_obstacle_centre_m=d_centre,
                       nearest_obstacle_edge_m=d_edge,
                       max_abs_curve=float(np.abs(p).max())), fh, indent=2)


if __name__ == "__main__":
    main()
