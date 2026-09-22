"""plot_safety_stack.py - 2x2 safety construction stack (schematic ALM panel).

For each requested dataset index the script draws the *same* scene four times in
a 2x2 grid, each panel adding one stage of the safety pipeline:

    (0,0) skeleton        occupancy + extracted Skeleton graph (nodes/branches)
    (0,1) ellipses        the 128 fixed-progress ellipses c_i = Gamma_m(i/(Q-1))
    (1,0) convex regions  the local convex region built from each ellipse
                          (EllipseRegionBuilder.build_from_ellipse)
    (1,1) safety corridor the regions linked into one ordered channel
                          (build_safety_corridor)
                          + a SCHEMATIC correction pair:
                            grey dashed = raw x0, the straight line from start
                                          to goal (leaves the corridor)
                            red solid   = the ALM-corrected trajectory

IMPORTANT: the two trajectories in panel 4 are DRAWN ON PURPOSE, they are not
sampler output.  The raw curve is simply the straight start -> goal line (it
cuts every corner and leaves the corridor); the corrected curve is the lightly
smoothed Skeleton centreline.  The pair exists to *illustrate* what the
inference-time ALM does - it is labelled "schematic" on the figure for that
reason.

Rendering: panels 1-3 are monochrome line drawings (one colour each, no
gradient, no colourbar).  Only the corridor panel carries colour because it has
to read as a channel and to contrast the two trajectories.

Outputs (under --out):

    sample_<idx>_stack.png     one 2x2 grid per requested sample
    stage1_skeleton.png        all samples, panel 1 only  (2 x 4 grid)
    stage2_ellipses.png        all samples, panel 2 only
    stage3_regions.png         all samples, panel 3 only
    stage4_corridor.png        all samples, panel 4 only
    safety_stack.json          per-sample diagnostics

Example (the 8 test samples shown in preview_test_ellipses.png):

    python scripts/plot/plot_safety_stack.py --config configs/config.yaml \
        --split test --ids 59 23 41 17 82 109 10 42 \
        --out outputs/bspline_carla/safety_stack
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import (CarlaSplineDataset,
                                               make_collate)
from src.geometry.convex_region import (EllipseRegionBuilder,
                                        halfspaces_to_vertices)
from src.geometry.safety_corridor import build_safety_corridor
from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import interpolate_path
from src.utils.config import load_config

METERS = 40.0

# One colour per panel; no gradients anywhere.
SKELETON_COLOR = "#1f77b4"      # panel 1
ELLIPSE_COLOR = "#b03a2e"       # panel 2
REGION_FACE = "#b0bec5"         # panel 3 (neutral grey-blue)
REGION_EDGE = "#78909c"
CORRIDOR_FACE = "#c5d9e8"       # panel 4 cells (pale blue, keeps curves legible)
CORRIDOR_EDGE = "#5b8db8"
BRIDGE_COLOR = "#e67e22"        # panel 4 bridge cells
CENTERLINE_COLOR = "#37474f"    # selected Skeleton
RAW_COLOR = "#546e7a"           # schematic raw x0 = straight start -> goal
OUTSIDE_COLOR = "#d81b60"       # the part of that line lying outside the corridor
SAFE_COLOR = "#e53935"          # schematic ALM-corrected trajectory (red)

AXIS_MIN_SHAPE4 = (np.log(2e-3), np.log(2e-3), 1.0, 0.0)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def to_px(points, res):
    """scene [-1,1] -> pixel coordinates (matches SkeletonGraph.pixel_to_scene)."""
    return (np.asarray(points, dtype=np.float64) + 1.0) / 2.0 * res - 0.5


def ellipse_polyline(center_px, a, b, theta, res, n=64):
    ang = np.linspace(0.0, 2.0 * np.pi, n)
    ct, st = np.cos(theta), np.sin(theta)
    ex = a * np.cos(ang) * res / 2.0
    ey = b * np.sin(ang) * res / 2.0
    return np.stack([ct * ex - st * ey + center_px[0],
                     st * ex + ct * ey + center_px[1]], axis=-1)


def sanitize_shape4(shape4, valid):
    """Invalid label rows -> a tiny isotropic circle (keeps geometry finite)."""
    out = np.array(shape4, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out).all(axis=-1) | ~np.asarray(valid, dtype=bool)
    if bad.any():
        out[bad] = np.asarray(AXIS_MIN_SHAPE4, dtype=np.float64)
    return out, int(bad.sum())


def resample_polyline(points, n=96):
    """Uniform arc-length resampling (keeps the exact endpoints)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return p.copy()
    u = arclength_param(p)
    t = np.linspace(0.0, 1.0, int(n))
    return np.stack([np.interp(t, u, p[:, 0]), np.interp(t, u, p[:, 1])],
                    axis=-1)


def smooth_polyline(points, window=7):
    p = np.asarray(points, dtype=np.float64)
    if window <= 1 or len(p) < window:
        return p.copy()
    pad = window // 2
    ext = np.concatenate([np.repeat(p[:1], pad, axis=0), p,
                          np.repeat(p[-1:], pad, axis=0)], axis=0)
    kernel = np.ones(window) / float(window)
    out = np.stack([np.convolve(ext[:, 0], kernel, mode="valid"),
                    np.convolve(ext[:, 1], kernel, mode="valid")], axis=-1)
    out[0], out[-1] = p[0], p[-1]
    return out


def arclength_param(points):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(points))
    return cum / total


def line_outside_mask(line, corridor, samples=96, tol=1e-6):
    """Sample the straight line and mark which samples fall outside the union."""
    base = [c for c in corridor.cells if c.source == "network"]
    line = np.asarray(line, dtype=np.float64).reshape(2, 2)
    t = np.linspace(0.0, 1.0, int(samples))[:, None]
    pts = line[0][None, :] * (1.0 - t) + line[1][None, :] * t
    inside = np.zeros(len(pts), dtype=bool)
    if base:
        for i, p in enumerate(pts):
            for cell in base:
                if cell.polygon is None:
                    continue
                if ((cell.A @ p - cell.b) <= tol).all():
                    inside[i] = True
                    break
    return pts, ~inside


def line_outside_fraction(line, corridor, samples=64, tol=1e-6):
    """Fraction of straight-line samples that fall outside the corridor union."""
    if not [c for c in corridor.cells if c.source == "network"]:
        return None
    _, outside = line_outside_mask(line, corridor, samples=samples, tol=tol)
    return float(outside.mean())


def schematic_correction(centerline, cond, window=13, samples=96):
    """Schematic (NOT sampler output) raw vs ALM-corrected trajectory pair.

    ``raw``  = the straight line from start to goal: it cuts every corner and
               visibly leaves the corridor.
    ``safe`` = the smoothed Skeleton centreline: it stays inside the corridor.
    """
    start = np.asarray(cond, dtype=np.float64)[0]
    goal = np.asarray(cond, dtype=np.float64)[1]
    raw = np.stack([start, goal], axis=0)

    safe = smooth_polyline(resample_polyline(centerline, samples), window)
    safe = np.asarray(safe, dtype=np.float64)
    safe[0] = np.asarray(centerline, dtype=np.float64)[0]
    safe[-1] = np.asarray(centerline, dtype=np.float64)[-1]

    # lateral excursion of the corrected curve from the straight line
    d = goal - start
    length = float(np.linalg.norm(d))
    if length > 1e-9:
        normal = np.array([-d[1], d[0]], dtype=np.float64) / length
        excursion = float(np.abs((safe - start) @ normal).max()) * METERS
    else:
        excursion = 0.0
    return raw, safe, excursion


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def draw_occupancy(ax, occ):
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_conditions(ax, cond, res):
    p = to_px(cond, res)
    ax.scatter([p[0, 0]], [p[0, 1]], marker="*", s=170, c="#00d26a",
               edgecolors="k", linewidths=0.7, zorder=9)
    ax.scatter([p[1, 0]], [p[1, 1]], marker="*", s=170, c="#ff5d67",
               edgecolors="k", linewidths=0.7, zorder=9)


def panel_skeleton(ax, ctx):
    draw_occupancy(ax, ctx["occ"])
    res = ctx["res"]
    skel = ctx["skeleton"]
    if skel.any():
        import matplotlib
        overlay = np.ma.masked_where(~skel, np.ones_like(skel, dtype=float))
        ax.imshow(overlay, origin="lower", interpolation="nearest",
                  cmap=matplotlib.colors.ListedColormap([SKELETON_COLOR]),
                  vmin=0.0, vmax=1.0, alpha=0.95, zorder=3)
    centers = np.asarray([n.center for n in ctx["graph"].nodes], dtype=float)
    if len(centers):
        cp = to_px(ctx["graph"].pixel_to_scene(centers), res)
        ax.scatter(cp[:, 0], cp[:, 1], s=5, c=SKELETON_COLOR, zorder=5)
    draw_conditions(ax, ctx["cond"], res)
    ax.set_title("#%d  Skeleton: nodes=%d branches=%d"
                 % (ctx["index"], len(ctx["graph"].nodes),
                    len(ctx["graph"].branches)), fontsize=8)


def panel_ellipses(ax, ctx, stride=1):
    draw_occupancy(ax, ctx["occ"])
    res = ctx["res"]
    line = to_px(ctx["centerline"], res)
    ax.plot(line[:, 0], line[:, 1], color=ELLIPSE_COLOR, lw=0.9,
            alpha=0.9, zorder=4)
    for i in range(0, len(ctx["center"]), max(1, stride)):
        poly = ellipse_polyline(to_px(ctx["center"][i], res), ctx["a"][i],
                                ctx["b"][i], ctx["theta"][i], res)
        ax.plot(poly[:, 0], poly[:, 1], color=ELLIPSE_COLOR, lw=0.5,
                alpha=0.8, zorder=3)
    cp = to_px(ctx["center"], res)
    ax.scatter(cp[:, 0], cp[:, 1], s=1.6, c=ELLIPSE_COLOR, zorder=4)
    draw_conditions(ax, ctx["cond"], res)
    ax.set_title("#%d  ellipses: %d centres, a_med=%.2fm b_med=%.2fm%s"
                 % (ctx["index"], len(ctx["center"]),
                    float(np.median(ctx["a"])) * METERS,
                    float(np.median(ctx["b"])) * METERS,
                    "" if ctx["n_invalid"] == 0
                    else "  (%d invalid labels)" % ctx["n_invalid"]),
                 fontsize=8)


def panel_regions(ax, ctx):
    draw_occupancy(ax, ctx["occ"])
    res = ctx["res"]
    import matplotlib.patches as mpatches
    n_ok = 0
    for poly in ctx["polygons"]:
        if poly is None:
            continue
        n_ok += 1
        p = to_px(poly, res)
        ax.add_patch(mpatches.Polygon(
            p, closed=True, facecolor=REGION_FACE, edgecolor=REGION_EDGE,
            alpha=0.45, linewidth=0.35, zorder=3))
    line = to_px(ctx["centerline"], res)
    ax.plot(line[:, 0], line[:, 1], color=CENTERLINE_COLOR, lw=0.9,
            alpha=0.9, zorder=6)
    draw_conditions(ax, ctx["cond"], res)
    n_bad = int(np.sum(~ctx["region_valid"]))
    ax.set_title("#%d  convex regions: %d/%d valid%s"
                 % (ctx["index"], n_ok, len(ctx["polygons"]),
                    "" if n_bad == 0 else "  (%d degenerate)" % n_bad),
                 fontsize=8)


def panel_corridor(ax, ctx):
    draw_occupancy(ax, ctx["occ"])
    res = ctx["res"]
    import matplotlib.patches as mpatches
    corridor = ctx["corridor"]
    n_base = n_bridge = 0
    if corridor.valid:
        for cell in corridor.cells:
            if cell.polygon is None:
                continue
            p = to_px(cell.polygon, res)
            is_bridge = cell.source == "bridge"
            n_bridge += int(is_bridge)
            n_base += int(not is_bridge)
            face = BRIDGE_COLOR if is_bridge else CORRIDOR_FACE
            edge = BRIDGE_COLOR if is_bridge else CORRIDOR_EDGE
            ax.add_patch(mpatches.Polygon(
                p, closed=True, facecolor=face, edgecolor=edge,
                alpha=0.55 if not is_bridge else 0.65,
                linewidth=0.35, zorder=3))
    else:
        for poly in ctx["polygons"]:
            if poly is None:
                continue
            p = to_px(poly, res)
            ax.add_patch(mpatches.Polygon(p, closed=True,
                                          facecolor="#ffcdd2",
                                          edgecolor="#c62828", alpha=0.5,
                                          linewidth=0.35, zorder=3))

    # schematic correction pair (NOT sampler output)
    raw, safe = ctx.get("illus_raw"), ctx.get("illus_safe")
    if raw is not None:
        rp = to_px(raw, res)
        ax.plot(rp[:, 0], rp[:, 1], color=RAW_COLOR, lw=1.5, ls="--",
                alpha=0.95, zorder=7)
        # highlight the part of the raw line that lies OUTSIDE the corridor
        pts, outside = ctx.get("illus_mask") or (None, None)
        if pts is not None and outside is not None and outside.any():
            seg = np.where(outside[:, None], to_px(pts, res), np.nan)
            ax.plot(seg[:, 0], seg[:, 1], color=OUTSIDE_COLOR, lw=2.4,
                    ls=(0, (4, 2)), alpha=0.95, zorder=7.5,
                    solid_capstyle="butt")
    if safe is not None:
        sp = to_px(safe, res)
        ax.plot(sp[:, 0], sp[:, 1], color=SAFE_COLOR, lw=1.8, alpha=0.98,
                zorder=8)

    draw_conditions(ax, ctx["cond"], res)

    title = "#%d  corridor: base=%d bridge=%d" % (ctx["index"], n_base, n_bridge)
    if corridor.valid:
        ov = np.asarray(corridor.overlap_ratio, dtype=float)
        title += "  overlap min=%.2f mean=%.2f" % (ov.min(), ov.mean())
    else:
        title += "  FAILED: %s" % corridor.failure_reason
    ax.set_title(title, fontsize=8,
                 color="#c62828" if not corridor.valid else "black")

    if ctx.get("illus_raw") is not None:
        frac = ctx.get("illus_outside")
        lines = ["schematic (not sampler output)",
                 "dashed: straight start -> goal (raw)",
                 "red: ALM-corrected trajectory"]
        if frac is not None:
            lines[1] += "  %.0f%% outside" % (100.0 * frac)
        if ctx.get("illus_excursion_m"):
            lines[2] += "  (max %.1f m)" % ctx["illus_excursion_m"]
        ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color="#263238", zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#b0bec5", alpha=0.88, linewidth=0.5))


def add_legend(fig, with_correction):
    """Small figure-level legend; only the corridor panel is coloured."""
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=SKELETON_COLOR, lw=1.6, label="Skeleton"),
        Line2D([], [], color=ELLIPSE_COLOR, lw=1.0, label="Ellipses"),
        mpatches.Patch(facecolor=REGION_FACE, edgecolor=REGION_EDGE,
                       label="Convex regions"),
        mpatches.Patch(facecolor=CORRIDOR_FACE, edgecolor=CORRIDOR_EDGE,
                       label="Safety corridor"),
        mpatches.Patch(facecolor=BRIDGE_COLOR, edgecolor=BRIDGE_COLOR,
                       label="Bridge cell"),
    ]
    if with_correction:
        handles += [
            Line2D([], [], color=RAW_COLOR, lw=1.5, ls="--",
                   label="raw x\u0302\u2080 (straight start -> goal, raw)"),
            Line2D([], [], color=SAFE_COLOR, lw=1.9,
                   label="ALM-corrected"),
        ]
    fig.legend(handles=handles, loc="outside lower center",
               ncol=len(handles), fontsize=7, frameon=False)


# ---------------------------------------------------------------------------
# context building
# ---------------------------------------------------------------------------

_MODEL_CACHE = None


def run_model(cfg, args, ds, sample):
    """Optional: run the sampler (no ALM) to get the model's own ellipses."""
    import torch
    from src.diffusion.sampler import sample as ddim_sample
    from src.diffusion.schedule import NoiseSchedule
    from src.utils.checkpoint import load_model

    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        device = (args.device if args.device else
                  ("cuda" if torch.cuda.is_available() else "cpu"))
        model, ckpt, _ = load_model(cfg, args.ckpt, arch="auto", device=device)
        model.eval()
        schedule = NoiseSchedule(
            cfg["diffusion"]["timesteps"],
            beta_schedule=cfg["diffusion"].get("beta_schedule",
                                               "squaredcos_cap_v2")).to(device)
        _MODEL_CACHE = (model, schedule, device)
    model, schedule, device = _MODEL_CACHE

    batch = make_collate(ds)([sample])
    with torch.no_grad():
        out = ddim_sample(model, schedule, batch["cond"].to(device),
                          batch["occupancy"].to(device),
                          batch["candidate_xy"].to(device),
                          batch["candidate_mask"].to(device),
                          batch["candidate_geometry"].to(device),
                          batch["candidate_geometry_lengths"].to(device),
                          device=device, steps=args.steps, seed=args.seed,
                          return_trace=False)
    return {
        "center": out["ellipse_center"][0].detach().cpu().numpy()
        .astype(np.float64),
        "shape4": out["ellipse_shape4"][0].detach().cpu().numpy()
        .astype(np.float64),
        "selected": int(out["selected_idx"][0].detach().cpu()),
    }


def build_context(cfg, args, ds, k, index, sample):
    import torch

    occ = np.asarray(sample["occupancy"][0].numpy(), dtype=np.float32)
    res = int(occ.shape[0])
    cond = np.asarray(sample["cond"].numpy(), dtype=np.float64)
    geom = np.asarray(sample["candidate_geometry"].numpy(), dtype=np.float64)
    glen = np.asarray(sample["candidate_geometry_lengths"].numpy(),
                      dtype=np.int64)
    best = int(sample["topology_best"])
    anchors = np.linspace(0.0, 1.0, len(sample["ellipse_shape4_gt"]))

    if args.shape_source == "pred":
        info = run_model(cfg, args, ds, sample)
        center, shape4 = info["center"], info["shape4"]
        selected = info["selected"]
        if len(center) != len(anchors):
            anchors = np.linspace(0.0, 1.0, len(center))
        n_invalid = 0
    else:
        selected = best
        n = int(glen[selected])
        n = n if n > 1 else int(glen.max())
        center = interpolate_path(geom[selected, :n], anchors)
        shape4, n_invalid = sanitize_shape4(
            np.asarray(sample["ellipse_shape4_gt"].numpy(), dtype=np.float64),
            np.asarray(sample["shape_valid"].numpy(), dtype=bool))

    n = int(glen[selected])
    n = n if n > 1 else int(glen.max())
    dense = geom[selected, :n]

    graph = build_skeleton_graph(
        occ, safety_dilation_cells=int(
            (cfg.get("skeleton") or {}).get("safety_dilation_cells", 1)),
        thinning_backend=str(
            (cfg.get("skeleton") or {}).get("thinning_backend", "auto")),
        pure_cycle_aux_nodes=int(
            (cfg.get("skeleton") or {}).get("pure_cycle_aux_nodes", 2)))

    region_cfg = dict((cfg.get("corridor") or {}).get("region") or {})
    builder = EllipseRegionBuilder(torch.from_numpy(occ)[None, None], region_cfg)
    A, b, face_mask, region_valid = builder.build_from_ellipse(
        torch.as_tensor(center, dtype=torch.float32)[None],
        torch.as_tensor(shape4, dtype=torch.float32)[None])
    A0, b0, m0, ok0 = A[0], b[0], face_mask[0], region_valid[0]
    polygons = []
    for i in range(len(center)):
        faces = m0[i]
        if not bool(faces.any()):
            polygons.append(None)
            continue
        polygons.append(halfspaces_to_vertices(
            A0[i][faces].detach().cpu().numpy(),
            b0[i][faces].detach().cpu().numpy(),
            interior_point=center[i]))

    try:
        corridor = build_safety_corridor(
            builder, center, shape4, anchors, gamma=dense,
            gamma_lengths=torch.tensor([len(dense)], dtype=torch.long),
            config=(cfg.get("corridor") or {}))
    except Exception as error:                                # pragma: no cover
        from src.geometry.safety_corridor import SafetyCorridor
        corridor = SafetyCorridor(cells=[], overlap_ratio=[],
                                  base_cell_count=0, bridge_cell_count=0,
                                  valid=False,
                                  failure_reason="exception:%s" % error)

    illus_raw = illus_safe = illus_outside = None
    illus_excursion_m = None
    if args.illustrate_correction:
        illus_raw, illus_safe, illus_excursion_m = schematic_correction(
            dense, cond)
        illus_pts, illus_mask = line_outside_mask(illus_raw, corridor)
        illus_outside = float(illus_mask.mean())

    a_ab = np.exp(shape4[:, 0])
    b_ab = np.exp(shape4[:, 1])
    theta = 0.5 * np.arctan2(shape4[:, 3], shape4[:, 2])
    return {
        "index": index, "k": k, "occ": occ, "res": res, "cond": cond,
        "graph": graph, "skeleton": graph.skeleton.astype(bool),
        "center": center, "a": a_ab, "b": b_ab, "theta": theta,
        "shape4": shape4, "anchors": anchors, "polygons": polygons,
        "region_valid": np.asarray(ok0.detach().cpu().numpy(), dtype=bool),
        "corridor": corridor, "centerline": dense,
        "selected": selected, "best": best, "n_invalid": n_invalid,
        "illus_raw": illus_raw, "illus_safe": illus_safe,
        "illus_outside": illus_outside, "illus_excursion_m": illus_excursion_m,
        "illus_mask": (illus_pts, illus_mask),
    }


def draw_panel(kind, ax, ctx, stride):
    if kind == "skeleton":
        panel_skeleton(ax, ctx)
    elif kind == "ellipses":
        panel_ellipses(ax, ctx, stride=stride)
    elif kind == "regions":
        panel_regions(ax, ctx)
    else:
        panel_corridor(ax, ctx)


def context_stats(ctx):
    corridor = ctx["corridor"]
    ov = np.asarray(corridor.overlap_ratio, dtype=float)
    return {
        "index": int(ctx["index"]),
        "selected_candidate": int(ctx["selected"]),
        "topology_best": int(ctx["best"]),
        "skeleton_nodes": int(len(ctx["graph"].nodes)),
        "skeleton_branches": int(len(ctx["graph"].branches)),
        "ellipse_centres": int(len(ctx["center"])),
        "invalid_shape_labels": int(ctx["n_invalid"]),
        "ellipse_a_median_m": float(np.median(ctx["a"])) * METERS,
        "ellipse_b_median_m": float(np.median(ctx["b"])) * METERS,
        "regions_valid": int(np.sum(ctx["region_valid"])),
        "polygons_drawn": int(sum(p is not None for p in ctx["polygons"])),
        "corridor_valid": bool(corridor.valid),
        "corridor_failure": corridor.failure_reason,
        "corridor_base_cells": int(corridor.base_cell_count),
        "corridor_bridge_cells": int(corridor.bridge_cell_count),
        "corridor_overlap_min": (float(ov.min()) if ov.size else None),
        "corridor_overlap_mean": (float(ov.mean()) if ov.size else None),
        "schematic_raw_outside_fraction": ctx.get("illus_outside"),
        "schematic_raw_max_excursion_m": ctx.get("illus_excursion_m"),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", type=int, nargs="+",
                    default=[59, 23, 41, 17, 82, 109, 10, 42],
                    help="dataset indices inside the chosen split")
    ap.add_argument("--out", default="outputs/bspline_carla/safety_stack")
    ap.add_argument("--shape-source", choices=["gt", "pred"], default="gt",
                    help="gt = offline ellipse labels (default, no ckpt); "
                         "pred = run the model for its own ellipses")
    ap.add_argument("--ckpt", default="outputs/bspline_carla/ckpt/best_task.pt")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="draw every n-th ellipse in panel 2")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--no-stage-grids", action="store_true")
    # schematic correction pair in panel 4 (drawn, not sampled)
    ap.add_argument("--illustrate-correction", dest="illustrate_correction",
                    action="store_true", default=True,
                    help="draw the schematic raw/corrected pair (default on)")
    ap.add_argument("--no-illustration", dest="illustrate_correction",
                    action="store_false",
                    help="draw only the corridor in panel 4")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    processed_root = cfg["data"].get("processed_root", "data/carla_processed")
    geo_points = int((cfg.get("topology") or {}).get(
        "candidate_geometry_points", 1280))
    ds = CarlaSplineDataset(args.split, processed_root,
                            geometry_points=geo_points, require_labels=True)
    n_total = len(ds)
    for i in args.ids:
        if not (0 <= i < n_total):
            raise SystemExit("index %d out of range (split has %d samples)"
                             % (i, n_total))

    os.makedirs(args.out, exist_ok=True)

    contexts, stats = [], []
    for k, i in enumerate(args.ids):
        ctx = build_context(cfg, args, ds, k, i, ds[i])
        contexts.append(ctx)
        stats.append(context_stats(ctx))
        print("[stack] #%d  m=%d  regions=%d/%d  corridor=%s (base=%d bridge=%d)"
              "  schematic raw outside=%.2f"
              % (i, ctx["selected"], stats[-1]["polygons_drawn"],
                 len(ctx["polygons"]), ctx["corridor"].valid,
                 ctx["corridor"].base_cell_count,
                 ctx["corridor"].bridge_cell_count,
                 ctx["illus_outside"] or 0.0), flush=True)

    with_correction = any(c.get("illus_raw") is not None for c in contexts)

    # --- per-sample 2x2 grids ------------------------------------------------
    row_paths = []
    for ctx in contexts:
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 10.2), dpi=args.dpi,
                                 squeeze=False, layout="constrained")
        order = [("skeleton", axes[0][0]), ("ellipses", axes[0][1]),
                 ("regions", axes[1][0]), ("corridor", axes[1][1])]
        for kind, ax in order:
            draw_panel(kind, ax, ctx, args.stride)
        fig.suptitle(
            "#%d  safety construction stack  (1 skeleton -> 2 ellipses -> "
            "3 convex regions -> 4 corridor + ALM correction)"
            % ctx["index"], fontsize=10)
        add_legend(fig, with_correction)
        fig.get_layout_engine().set(rect=(0.0, 0.035, 1.0, 0.955))
        path = os.path.join(args.out, "sample_%d_stack.png" % ctx["index"])
        fig.savefig(path)
        plt.close(fig)
        row_paths.append(path)
        print("[stack] saved", path, flush=True)

    # --- one grid per stage --------------------------------------------------
    stage_paths = []
    if not args.no_stage_grids:
        cols = min(4, len(contexts))
        rows_n = int(np.ceil(len(contexts) / cols))
        stages = [
            ("stage1_skeleton", "1. extracted Skeleton", "skeleton"),
            ("stage2_ellipses", "2. ellipses along the selected Skeleton",
             "ellipses"),
            ("stage3_regions", "3. convex region built from each ellipse",
             "regions"),
            ("stage4_corridor",
             "4. safety corridor + schematic ALM correction (dashed = raw x0, "
             "solid = corrected)", "corridor"),
        ]
        for name, title, kind in stages:
            fig, axes = plt.subplots(rows_n, cols,
                                     figsize=(4.0 * cols, 4.5 * rows_n),
                                     dpi=args.dpi, squeeze=False,
                                     layout="constrained")
            for k, ctx in enumerate(contexts):
                draw_panel(kind, axes[k // cols][k % cols], ctx, args.stride)
            for j in range(len(contexts), rows_n * cols):
                axes[j // cols][j % cols].axis("off")
            fig.suptitle("%s  (%s split, %d samples)"
                         % (title, args.split, len(contexts)), fontsize=11)
            fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.945))
            path = os.path.join(args.out, "%s.png" % name)
            fig.savefig(path)
            plt.close(fig)
            stage_paths.append(path)
            print("[stack] saved", path, flush=True)

    summary = {
        "config": args.config, "split": args.split, "ids": list(args.ids),
        "shape_source": args.shape_source,
        "ckpt": (args.ckpt if args.shape_source == "pred" else None),
        "layout": "2x2: skeleton / ellipses / convex regions / corridor",
        "style": ("panels 1-3 monochrome; panel 4 pale blue cells, orange "
                  "bridge, grey dashed = straight raw start->goal, "
                  "red solid = ALM-corrected"),
        "panel4_correction": ("SCHEMATIC (drawn on purpose, not sampler "
                              "output)" if args.illustrate_correction
                              else "not drawn"),
        "illustration": {
            "raw": "straight line start -> goal (leaves the corridor)",
            "corrected": "smoothed Skeleton centreline",
        },
        "figures": {"rows": row_paths, "stages": stage_paths},
        "samples": stats,
    }
    path = os.path.join(args.out, "safety_stack.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[stack] saved", path, flush=True)


if __name__ == "__main__":
    main()
