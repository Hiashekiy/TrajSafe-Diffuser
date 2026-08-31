"""Minimal visualization helpers (matplotlib)."""

import os

import numpy as np


def plot_map_traj_ellipses(occupancy, extent, traj_world, ellipse_params, out_path,
                           wall_centers=None):
    """Save a PNG with map + trajectory + ellipses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    x0, x1, y0, y1 = extent
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(occupancy.T, origin="lower", extent=(x0, x1, y0, y1),
              cmap="gray_r", alpha=0.6)
    traj_world = np.asarray(traj_world, dtype=float)
    ax.plot(traj_world[:, 0], traj_world[:, 1], "-o", color="tab:blue",
            markersize=1.5, linewidth=1)
    for params in ellipse_params:
        cx, cy, r1, r2, theta = params
        if r1 <= 0 or r2 <= 0 or not np.isfinite(r1 + r2):
            continue
        e = Ellipse((cx, cy), width=2 * r1, height=2 * r2,
                    angle=np.degrees(theta), fill=False, edgecolor="tab:red",
                    linewidth=0.8)
        ax.add_patch(e)
    ax.set_aspect("equal")
    set_map_limits(ax, extent)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def draw_traj(ax, positions, velocities=None, cmap="viridis", lw=2.0, alpha=0.9,
              marker_every=0, arrow_every=0, arrow_len=0.35,
              start_color="lime", goal_color="red",
              start_marker="*", goal_marker="*", label="trajectory",
              colorbar_ax=None, colorbar_label="progress", zorder=2):
    """Draw a trajectory as a time-coloured line instead of a cloud of points.

    positions  : array (H,2) world positions.
    velocities : optional array (H,2).  Only used to draw small direction arrows.
    """
    from matplotlib.collections import LineCollection

    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    H = len(pos)
    if H < 2:
        return

    progress = np.linspace(0.0, 1.0, H)
    segments = np.stack([pos[:-1], pos[1:]], axis=1)  # (H-1,2,2)
    lc = LineCollection(segments, cmap=cmap, linewidths=lw, alpha=alpha)
    lc.set_array(progress[:-1])
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)

    # thin dark line under the coloured segments keeps the path readable on busy maps
    ax.plot(pos[:, 0], pos[:, 1], color="black", lw=0.8, alpha=0.15,
            zorder=zorder - 1)

    if marker_every > 0:
        idx = np.arange(0, H, marker_every)
        if idx[-1] != H - 1:
            idx = np.concatenate([idx, [H - 1]])
        ax.scatter(pos[idx, 0], pos[idx, 1], c=progress[idx], cmap=cmap, s=18,
                   edgecolors="k", linewidths=0.3, alpha=0.9, zorder=zorder + 1)

    if velocities is not None and arrow_every > 0:
        vel = np.asarray(velocities, dtype=float).reshape(-1, 2)
        if len(vel) == H:
            ai = np.arange(0, H, arrow_every)
            if ai[-1] != H - 1:
                ai = np.concatenate([ai, [H - 1]])
            vx = vel[ai, 0]
            vy = vel[ai, 1]
            scale = float(np.max(np.hypot(vx, vy)))
            if scale > 0:
                ax.quiver(pos[ai, 0], pos[ai, 1],
                          vx / scale * arrow_len, vy / scale * arrow_len,
                          angles="xy", scale_units="xy", scale=1.0,
                          color="black", width=0.0025, alpha=0.55,
                          zorder=zorder + 2)

    ax.scatter(pos[0, 0], pos[0, 1], c=start_color, s=70, marker=start_marker,
               zorder=zorder + 3, label="start")
    ax.scatter(pos[-1, 0], pos[-1, 1], c=goal_color, s=70, marker=goal_marker,
               zorder=zorder + 3, label="goal")

    if colorbar_ax is not None:
        try:
            fig = ax.get_figure()
            cb = fig.colorbar(lc, cax=colorbar_ax, orientation="vertical")
            cb.set_label(colorbar_label)
        except Exception:
            pass


def set_map_limits(ax, extent):
    """Fix the axes to exactly the map extent so out-of-map points are clipped.

    extent : (x0, x1, y0, y1).
    """
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
