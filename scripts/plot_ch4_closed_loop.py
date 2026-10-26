#!/usr/bin/env python
"""Generate compact, publication-oriented figures for Section 4.4.

The script reads recorded experiment JSON files only. It does not rerun sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch4_plot import configure_fonts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL = ROOT / "outputs/chapter4/runs_abl/T3_A-C48_alm__s0.json"
DEFAULT_FEEDBACK = ROOT / "outputs/chapter4/runs_abl/FeedbackOnly__s0.json"
DEFAULT_PROCESS = ROOT / "outputs/chapter4/figures/process_curves.png"
DEFAULT_FEEDBACK_FIG = ROOT / "outputs/chapter4/figures/feedback_state_comparison.png"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GRID = "#CBD5E1"
TEXT = "#1F2937"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def values(run: dict, name: str, stat: str = "mean", metres: bool = False):
    raw = (run.get("per_step", {}).get(name, {}) or {}).get(stat, [])
    array = np.asarray([np.nan if value is None else float(value) for value in raw])
    if metres:
        array *= float(run.get("scene_to_meter", 80.0))
    return array


def guided_steps(run: dict):
    steps = np.asarray(run["per_step"]["step_index"], dtype=int)
    warmup = int(run.get("warmup_reverse_steps", 3))
    mask = steps > warmup
    return steps[mask], mask


def style_axis(ax, panel: str, title: str, ylabel: str):
    ax.text(-0.12, 1.06, panel, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=TEXT, va="top")
    ax.set_title(title, fontsize=10.2, loc="left", pad=6, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=9.2)
    ax.set_xlabel("反向去噪步骤", fontsize=9.2)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def band(ax, x, run, name, color, metres=False):
    lo = values(run, name, "p25", metres)
    hi = values(run, name, "p75", metres)
    mask = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=0.12,
                    linewidth=0, zorder=1)


def plot_process(run: dict, out: Path):
    x, guided = guided_steps(run)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.45))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.10,
                        hspace=0.40, wspace=0.30)

    ax = axes[0, 0]
    raw = values(run, "raw_violation", metres=True)[guided]
    safe = values(run, "safe_violation", metres=True)[guided]
    band(ax, np.asarray(run["per_step"]["step_index"]), run,
         "raw_violation", ORANGE, metres=True)
    band(ax, np.asarray(run["per_step"]["step_index"]), run,
         "safe_violation", BLUE, metres=True)
    ax.plot(x, raw, color=ORANGE, marker="o", markersize=3.4,
            linewidth=1.55, linestyle="--", label="ALM 前")
    ax.plot(x, safe, color=BLUE, marker="s", markersize=3.2,
            linewidth=1.7, label="ALM 后")
    style_axis(ax, "(a)", "走廊约束违规量", "平均违规量 / m")
    ax.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper right")

    ax = axes[0, 1]
    correction = values(run, "correction", metres=True)[guided]
    band(ax, np.asarray(run["per_step"]["step_index"]), run,
         "correction", PURPLE, metres=True)
    ax.plot(x, correction, color=PURPLE, marker="D", markersize=3.2,
            linewidth=1.7)
    style_axis(ax, "(b)", "安全轨迹修正幅度", "平均修正距离 / m")

    ax = axes[1, 0]
    feasible = values(run, "feasible_rate")[guided] * 100.0
    ax.plot(x, feasible, color=GREEN, marker="^", markersize=3.8,
            linewidth=1.7)
    ax.set_ylim(max(0.0, np.nanmin(feasible) - 5.0), 101.0)
    style_axis(ax, "(c)", "逐步走廊可行率", "可行样本比例 / %")

    ax = axes[1, 1]
    smooth_raw = values(run, "smooth_raw", metres=True)[guided]
    smooth_safe = values(run, "smooth_safe", metres=True)[guided]
    ax.plot(x, smooth_raw, color=ORANGE, marker="o", markersize=3.4,
            linewidth=1.5, linestyle="--", label="ALM 前")
    ax.plot(x, smooth_safe, color=BLUE, marker="s", markersize=3.2,
            linewidth=1.7, label="ALM 后")
    style_axis(ax, "(d)", "轨迹平滑度", "平均二阶差分 / m")
    ax.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper right")

    for ax in axes.flat:
        ax.set_xticks([4, 8, 12, 16])
        ax.set_xlim(3.6, 16.4)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_feedback_comparison(feedback: dict, full: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.20,
                        wspace=0.32)

    labels = ["仅反馈", "完整方法"]
    colors = [PURPLE, BLUE]
    hatches = ["///", ""]

    collision = [100.0 * feedback["aggregate"]["collision_rate"],
                 100.0 * full["aggregate"]["collision_rate"]]
    infeasible = [100.0 * (1.0 - feedback["aggregate"]["sdf_feasible_rate"]),
                  100.0 * (1.0 - full["aggregate"]["sdf_feasible_rate"])]
    x = np.arange(2)
    width = 0.34
    bars1 = axes[0].bar(x - width / 2, collision, width, color=colors[0],
                        alpha=0.88, hatch=hatches[0], edgecolor="white",
                        label="碰撞率")
    bars2 = axes[0].bar(x + width / 2, infeasible, width, color=colors[1],
                        alpha=0.88, edgecolor="white", label="SDF 不可行率")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(collision + infeasible) * 1.35 + 0.2)
    axes[0].set_ylabel("轨迹比例 / %", fontsize=9.2)
    axes[0].set_title("(a) 最终安全结果", loc="left", fontsize=10.2,
                      fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8.0, loc="upper right")
    axes[0].grid(axis="y", color=GRID, linewidth=0.55, alpha=0.55)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].bar_label(bars1, fmt="%.2f", fontsize=7.8, padding=2)
    axes[0].bar_label(bars2, fmt="%.2f", fontsize=7.8, padding=2)

    metric_names = ["修正前违规", "修正后违规", "修正距离"]
    def last(run, key):
        return float(values(run, key, metres=True)[-1])
    feedback_values = [last(feedback, "raw_violation"),
                       last(feedback, "safe_violation"),
                       last(feedback, "correction")]
    full_values = [last(full, "raw_violation"),
                   last(full, "safe_violation"),
                   last(full, "correction")]
    x = np.arange(3)
    bars1 = axes[1].bar(x - width / 2, feedback_values, width,
                        color=PURPLE, alpha=0.88, hatch="///",
                        edgecolor="white", label="仅反馈")
    bars2 = axes[1].bar(x + width / 2, full_values, width,
                        color=BLUE, alpha=0.88, edgecolor="white",
                        label="完整方法")
    axes[1].set_xticks(x, metric_names, fontsize=8.1)
    axes[1].set_ylabel("距离 / m", fontsize=9.2)
    axes[1].set_title("(b) 第 16 步过程量", loc="left", fontsize=10.2,
                      fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8.0, loc="upper left")
    axes[1].grid(axis="y", color=GRID, linewidth=0.55, alpha=0.55)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, max(feedback_values + full_values) * 1.30)
    axes[1].bar_label(bars1, fmt="%.3f", fontsize=7.3, padding=2,
                      rotation=90)
    axes[1].bar_label(bars2, fmt="%.3f", fontsize=7.3, padding=2,
                      rotation=90)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", default=str(DEFAULT_FULL))
    parser.add_argument("--feedback", default=str(DEFAULT_FEEDBACK))
    parser.add_argument("--process-out", default=str(DEFAULT_PROCESS))
    parser.add_argument("--feedback-out", default=str(DEFAULT_FEEDBACK_FIG))
    args = parser.parse_args()

    configure_fonts(verbose=False)
    full = load(Path(args.full))
    feedback = load(Path(args.feedback))
    plot_process(full, Path(args.process_out))
    plot_feedback_comparison(feedback, full, Path(args.feedback_out))
    print(Path(args.process_out).resolve())
    print(Path(args.feedback_out).resolve())


if __name__ == "__main__":
    main()
