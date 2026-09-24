"""Offline visualisation of a frozen plan (matplotlib, no CARLA needed)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

from .frame import LocalFrame
from .planner_adapter import PlanResult

__all__ = ["plot_plan_over_occupancy", "plot_executed_trace",
           "plot_modes_on_occupancy"]


def _px(scene_xy, flip: bool = True, res: int = 256):
    """Scene -> pixel.  flip=True matches the Diffusion Lens display (row 255 on top)."""
    grid = LocalFrame.scene_to_grid(np.asarray(scene_xy, dtype=np.float64), res)
    col = grid[..., 0]
    row = grid[..., 1]
    if flip:
        row = (res - 1) - row
    return col, row


def plot_plan_over_occupancy(result: PlanResult, occupancy: np.ndarray, path: str,
                             title: str = "", executed_xy: Optional[np.ndarray] = None,
                             flip: bool = True, frame=None,
                             show_planner_details: bool = True) -> str:
    """Draw the plan over the canonical occupancy.

    flip=True matches the Diffusion Lens display (canonical row 255 on top, so
    scene +y points up).  The IMAGE must be flipped together with the overlay;
    flipping only the overlay mirrors the curve into the occupied area, which
    is exactly the bug this docstring exists to prevent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(occupancy)
    if flip:
        image = np.flipud(image)
    fig, ax = plt.subplots(figsize=(8.4, 8.4), dpi=120)
    ax.imshow(image, cmap="gray_r", origin="upper",
              extent=(-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5))
    if show_planner_details:
        for polygon in result.corridor_scene:
            col, row = _px(polygon, flip)
            ax.fill(np.concatenate((col, col[:1])), np.concatenate((row, row[:1])),
                    color="lime", alpha=0.10, lw=0.0)
        for ellipse in result.ellipse_scene:
            center = np.asarray(ellipse["center"], dtype=np.float64).reshape(1, 2)
            log_a, log_b, c2, s2 = [float(v) for v in ellipse["shape4"][:4]]
            theta = 0.5 * np.arctan2(s2, c2)
            t = np.linspace(0.0, 2.0 * np.pi, 48)
            local = np.stack((np.exp(log_a) * np.cos(t), np.exp(log_b) * np.sin(t)), axis=1)
            scene = local @ np.array([[np.cos(theta), np.sin(theta)],
                                      [-np.sin(theta), np.cos(theta)]])
            scene = scene + center
            col, row = _px(scene, flip)
            ax.plot(col, row, "-", color="royalblue", lw=0.9, alpha=0.85)
        candidates = np.asarray(result.candidates_scene, dtype=np.float64)
        for index in range(candidates.shape[0]):
            col, row = _px(candidates[index], flip)
            ax.plot(col, row, "-", color="slategray", lw=1.0, alpha=0.8)
    col, row = _px(result.curve_scene, flip)
    ax.plot(col, row, "-", color="magenta", lw=2.2, label="guided B-spline")
    if show_planner_details and result.raw_curve_scene is not None \
            and len(result.raw_curve_scene):
        col, row = _px(result.raw_curve_scene, flip)
        ax.plot(col, row, "--", color="orange", lw=1.4, label="raw (pre-ALM)")
    col, row = _px(result.condition_scene, flip)
    ax.plot(col[0], row[0], "o", color="lime", ms=10, label="start")
    ax.plot(col[1], row[1], "o", color="red", ms=8, label="goal")
    if executed_xy is not None and len(executed_xy):
        executed_scene = (frame.to_scene(executed_xy)
                          if frame is not None else executed_xy)
        col, row = _px(executed_scene, flip)
        ax.plot(col, row, "-", color="deepskyblue", lw=1.8,
                label="executed CARLA trace")
    ax.set_xlim(0, occupancy.shape[1] - 1)
    ax.set_ylim(occupancy.shape[0] - 1, 0)
    ax.set_title(title or "TrajSafe-Diffuser plan vs canonical occupancy")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_executed_trace(history_csv: str, result: PlanResult, occupancy: np.ndarray,
                        path: str, frame=None) -> str:
    import csv as csv_mod

    xs, ys = [], []
    with open(history_csv, "r", encoding="utf-8") as handle:
        for row in csv_mod.DictReader(handle):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    executed_world = np.stack((np.array(xs), np.array(ys)), axis=1)
    return plot_plan_over_occupancy(result, occupancy, path,
                                    title="latest four-step plan vs executed CARLA trace",
                                    executed_xy=executed_world, frame=frame,
                                    show_planner_details=False)


def plot_modes_on_occupancy(occupancy: np.ndarray, panels, path: str, frame,
                            obstacles=None, title: str = "", flip: bool = True,
                            dpi: int = 140) -> str:
    """One occupancy map per mode, each with its own generated trajectory.

    panels entries: name, curve_world, planned_free (bool per curve point),
    executed_world (optional), note (optional).  Off-drivable-area samples are
    marked in red so the ablation reads at a glance without any CARLA render.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.flipud(np.asarray(occupancy)) if flip else np.asarray(occupancy)
    extent = (-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5)
    count = len(panels)
    fig, axes = plt.subplots(1, count, figsize=(6.0 * count, 6.3), dpi=dpi)
    axes = list(np.atleast_1d(axes))
    for ax, panel in zip(axes, panels):
        ax.imshow(image, cmap="gray_r", origin="upper", extent=extent)
        curve_scene = frame.to_scene(np.asarray(panel["curve_world"], dtype=np.float64))
        col, row = _px(curve_scene, flip)
        ax.plot(col, row, "-", color="magenta", lw=2.4, zorder=4,
                label="generated trajectory")
        free = panel.get("planned_free")
        if free is not None:
            off = ~np.asarray(free, dtype=bool)
            if off.any():
                ax.plot(col[off], row[off], "o", color="red", ms=3.4, zorder=6,
                        label="outside drivable area")
        executed = panel.get("executed_world")
        if executed is not None and len(executed) > 1:
            c2, r2 = _px(frame.to_scene(np.asarray(executed, dtype=np.float64)), flip)
            ax.plot(c2, r2, "-", color="deepskyblue", lw=1.8, zorder=5,
                    label="CARLA executed")
        if obstacles:
            from .overlay import obstacle_world_corners

            for box in obstacles:
                c3, r3 = _px(frame.to_scene(obstacle_world_corners(box, frame)), flip)
                ax.fill(np.concatenate((c3, c3[:1])), np.concatenate((r3, r3[:1])),
                        color="red", alpha=0.9, zorder=3)
        start_scene = np.asarray(panel["start_scene"], dtype=np.float64).reshape(1, 2)
        goal_scene = np.asarray(panel["goal_scene"], dtype=np.float64).reshape(1, 2)
        for pts, colour, size, name in ((start_scene, "lime", 9, "start"),
                                        (goal_scene, "red", 8, "goal")):
            c, r = _px(pts, flip)
            ax.plot(c, r, "o", color=colour, ms=size, zorder=7, label=name)
        ax.set_xlim(0, image.shape[1] - 1)
        ax.set_ylim(image.shape[0] - 1, 0)
        ax.set_title("%s   %s" % (panel["name"], panel.get("note", "")), fontsize=12)
        ax.legend(loc="lower left", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path

def plot_trajectories_together(occupancy: np.ndarray, entries, path: str, frame,
                               obstacles=None, title: str = "", flip: bool = True,
                               dpi: int = 150) -> str:
    """Both modes on ONE occupancy map, for a single-slide comparison.

    entries: {"name", "colour", "curve_world", "planned_free" (optional bool
    array), "executed_world" (optional), "linestyle" (optional)}.
    Samples outside the drivable area are over-plotted as red dots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.flipud(np.asarray(occupancy)) if flip else np.asarray(occupancy)
    fig, ax = plt.subplots(figsize=(7.6, 7.6), dpi=dpi)
    ax.imshow(image, cmap="gray_r", origin="upper",
              extent=(-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5))
    if obstacles:
        from .overlay import obstacle_world_corners

        for box in obstacles:
            c, r = _px(frame.to_scene(obstacle_world_corners(box, frame)), flip)
            ax.fill(np.concatenate((c, c[:1])), np.concatenate((r, r[:1])),
                    color="crimson", alpha=0.9, zorder=3)
    worst = []
    for entry in entries:
        curve = frame.to_scene(np.asarray(entry["curve_world"], dtype=np.float64))
        col, row = _px(curve, flip)
        ax.plot(col, row, entry.get("linestyle", "-"), color=entry["colour"],
                lw=2.8, zorder=5, label=entry["name"])
        free = entry.get("planned_free")
        off_count = 0
        if free is not None:
            off = ~np.asarray(free, dtype=bool)
            off_count = int(off.sum())
            if off.any():
                ax.plot(col[off], row[off], "o", color="red", ms=4.0, zorder=7,
                        label="%s: outside drivable area" % entry["name"])
        worst.append((entry["name"], off_count, len(curve)))
        executed = entry.get("executed_world")
        if executed is not None and len(executed) > 1:
            c2, r2 = _px(frame.to_scene(np.asarray(executed, dtype=np.float64)), flip)
            ax.plot(c2, r2, "-", color=entry["colour"], lw=1.1, alpha=0.55, zorder=4)
    start = np.asarray(entries[0]["start_scene"], dtype=np.float64).reshape(1, 2)
    goal = np.asarray(entries[0]["goal_scene"], dtype=np.float64).reshape(1, 2)
    cs, rs = _px(start, flip)
    cg, rg = _px(goal, flip)
    ax.plot(cs, rs, "o", color="lime", ms=11, zorder=8, label="start")
    ax.plot(cg, rg, "o", color="red", ms=10, zorder=8, label="goal")
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(image.shape[0] - 1, 0)
    ax.set_xlabel("canonical column  (scene x)")
    ax.set_ylabel("canonical row  (scene y)")
    if title:
        ax.set_title(title, fontsize=12)
    summary = "   ".join("%s: %d/%d outside" % (n, o, t) for n, o, t in worst)
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=9,
            color="white", bbox=dict(facecolor="black", alpha=0.65, pad=4))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
