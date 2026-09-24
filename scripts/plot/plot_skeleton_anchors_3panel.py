"""plot_skeleton_anchors_3panel.py - 3-panel schematic of the safety-domain stack.

Pure schematic (no dataset, no checkpoint).  A STRAIGHT diagonal Skeleton is cut
at Q fixed-progress anchors ``s_i = i/(Q-1)``; the same three stages of the
pipeline are drawn one per panel:

    panel 1  Skeleton Anchors        the trajectory split by the sampling lines
    panel 2  Safety Ellipses        one ellipse per anchor, major axis = tangent
    panel 3  Local Convex Regions   each ellipse replaced by a convex polygon

The geometry is deliberately the SAME sequence in all three panels, so the figure
reads as "same anchors, progressively more safety geometry".

Outputs (under --out):

    panel1_skeleton_anchors.png
    panel2_safety_ellipses.png
    panel3_convex_regions.png
    combined.png                three panels + the arrows between them

Usage:

    python scripts/plot/plot_skeleton_anchors_3panel.py --out outputs/fig_pipeline
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,
                                Ellipse as MEllipse, Polygon)
from scipy.spatial import ConvexHull

RED = "#d62728"
BLUE = "#2b7bba"
GREEN = "#2e9b3f"
BOX = "#f6f9fc"
EDGE = "#c9d6e2"

# straight diagonal Skeleton: START -> END (scene units)
START = np.array([-0.62, -0.30])
END = np.array([0.62, 0.30])
XLIM, YLIM = 1.25, 0.85
BOX_XY = (-XLIM + 0.02, -YLIM + 0.02)
BOX_WH = (2.0 * XLIM - 0.04, 2.0 * YLIM - 0.04)


# --------------------------------------------------------------------- geometry
def build(num_anchors: int = 4, seed: int = 11):
    """Return the shared geometry of the three panels."""
    rng = np.random.default_rng(seed)
    u = np.linspace(0.0, 1.0, num_anchors)
    p = START[None] + u[:, None] * (END - START)[None]      # anchors on Gamma
    direction = END - START
    length = float(np.linalg.norm(direction))
    t = direction / length
    n = np.array([-t[1], t[0]])
    ds = length / max(num_anchors - 1, 1)                   # mean anchor spacing
    ang = np.arctan2(t[1], t[0])

    # panel 2: ellipses; a_ax > ds/2 so consecutive ellipses OVERLAP into a chain
    a_ax = ds * (0.62 + 0.03 * rng.normal(size=num_anchors))    # semi-major
    b_ax = a_ax * 0.80                                          # semi-minor

    # panel 3: irregular convex polygon per anchor (superset of its ellipse).
    #   Evenly spread directions with a small jitter + jittered radii -> the
    #   chunky heptagon-ish regions of the reference figure, while ConvexHull
    #   keeps each one convex by construction.
    rot = np.array([[np.cos(ang), -np.sin(ang)],
                    [np.sin(ang), np.cos(ang)]])
    polys = []
    for k in range(num_anchors):
        m = int(rng.integers(7, 9))                     # 7..8 sides
        th = (np.arange(m) * (2.0 * np.pi / m)
              + rng.uniform(-0.15, 0.15, size=m))
        rad = 1.0 + 0.22 * rng.random(m)                # 1.00 .. 1.22
        loc = np.stack([a_ax[k] * rad * np.cos(th),
                        b_ax[k] * 1.05 * rad * np.sin(th)], axis=-1)
        pts = loc @ rot.T + p[k]
        polys.append(pts[ConvexHull(pts).vertices])
    return dict(u=u, anchors=p, t=t, n=n, a=a_ax, b=b_ax, ang=ang, polys=polys)


# --------------------------------------------------------------------- panels
def _panel_style(ax, title):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(-YLIM, YLIM)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(FancyBboxPatch(BOX_XY, *BOX_WH,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                linewidth=1.4, edgecolor=EDGE, facecolor=BOX,
                                zorder=0))
    ax.set_title(title, fontsize=13, pad=8)


def _skeleton(ax, g, solid):
    """The centreline: solid red in panel 1, dashed black afterwards."""
    pts = np.stack([START, END])
    if solid:
        ax.plot(pts[:, 0], pts[:, 1], color=RED, lw=1.9, zorder=3,
                solid_capstyle="round")
    else:
        ax.plot(pts[:, 0], pts[:, 1], color="#3a3a3a", lw=1.2,
                ls=(0, (5, 4)), zorder=3)
    ax.plot(g["anchors"][:, 0], g["anchors"][:, 1], "o", ms=7.0, mfc=RED,
            mec="white", mew=1.1, zorder=5)


def draw_panel(ax, g, stage: int):
    """stage 1 = anchors, 2 = ellipses, 3 = convex regions."""
    if stage == 1:
        _panel_style(ax, "Skeleton Anchors")
        for k in range(len(g["u"])):
            a = g["anchors"][k] - g["n"] * 0.30
            b = g["anchors"][k] + g["n"] * 0.30
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#3a3a3a", lw=1.0,
                    ls=(0, (5, 4)), zorder=2)
        _skeleton(ax, g, solid=True)
        ax.text(0.0, -0.60, r"$m^{*}$", fontsize=15, ha="center", va="center")
        ax.text(0.0, -0.755, "fixed progress   " + r"$s_i = i/(Q-1)$",
                fontsize=8.5, color="#555555", ha="center", va="center")
    elif stage == 2:
        _panel_style(ax, "Safety Ellipses")
        for k in range(len(g["u"])):
            ax.add_patch(MEllipse(g["anchors"][k], width=2.0 * g["a"][k],
                                  height=2.0 * g["b"][k],
                                  angle=np.degrees(g["ang"]),
                                  facecolor=BLUE, alpha=0.14, edgecolor=BLUE,
                                  lw=1.5, zorder=2))
        _skeleton(ax, g, solid=False)
        ax.text(0.0, -0.72, r"$\cdots$", fontsize=17, ha="center", va="center")
    else:
        _panel_style(ax, "Local Convex Regions")
        for poly in g["polys"]:
            ax.add_patch(Polygon(poly, closed=True, facecolor=GREEN, alpha=0.18,
                                 edgecolor=GREEN, lw=1.7, joinstyle="round",
                                 zorder=2))
        _skeleton(ax, g, solid=False)
        ax.text(0.0, -0.72, r"$\cdots$", fontsize=17, ha="center", va="center")
    return ax


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/fig_pipeline")
    ap.add_argument("--num-anchors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    g = build(num_anchors=args.num_anchors, seed=args.seed)
    names = ["panel1_skeleton_anchors", "panel2_safety_ellipses",
             "panel3_convex_regions"]

    # ---- three standalone panels -------------------------------------------
    pw, ph = 2.0 * XLIM, 2.0 * YLIM
    for stage, name in enumerate(names, start=1):
        figw = 3.3
        figh = figw * (ph / pw) / 0.845
        fig, ax = plt.subplots(figsize=(figw, figh), dpi=args.dpi)
        draw_panel(ax, g, stage)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.845, bottom=0.005)
        path = os.path.join(args.out, name + ".png")
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        print("wrote", path)

    # ---- combined strip -----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.92), dpi=args.dpi)
    for stage, ax in enumerate(axes, start=1):
        draw_panel(ax, g, stage)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.845, bottom=0.005,
                        wspace=0.36)
    for xf in (0.335, 0.663):
        fig.patches.append(FancyArrowPatch(
            (xf, 0.42), (xf + 0.030, 0.42), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=17, lw=1.7, color="black"))
    path = os.path.join(args.out, "combined.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
