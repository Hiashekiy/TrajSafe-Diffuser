#!/usr/bin/env python
"""从 Diffusion Lens 缓存生成论文图（样本 293，自定义起终点）。

布局：5 个反向步骤面板**排成一行**，下方一行图例。论文是英文稿，默认输出
英文标签；把 ``LANG`` 改成 ``"zh"`` 即可退回中文标签（中英文字串都在
``LABELS`` 里，改一个字符即可切换）。

可选：``python scripts/plot_dashboard_case293.py [输出路径]`` 覆盖 ``OUTPUT``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from ch4_plot import configure_fonts


ROOT = Path(__file__).resolve().parents[1]
CACHE_BASE = ROOT / "diffusion-dashboard" / "cache-carla" / "K4P_c48_oneshot"
CACHE_STREAM = "best_task__test_0293__seed43__fce4123cd39a.json"
OCCUPANCY = ROOT / "data" / "carla_processed_160k4p_c48" / "test" / "occupancy.npy"
OUTPUT = ROOT / "outputs" / "ch4" / "figures" / "cases" / "case_293_custom_endpoints_alm.png"

# 第 1/4/8/12/16 个反向步骤，对应 t=15/12/8/4/0。
FRAME_INDICES = (0, 3, 7, 11, 15)
SCENE_TO_METER = 80.0

# "en" = 论文用英文标签（默认）；"zh" = 中文标签。
LANG = "en"

LABELS = {
    "en": {
        "step": "Step {n}",
        "warmup": "Step {n} (warm-up)",
        "violation": "violation {before:.2f}$\\rightarrow${after:.2f} m\n"
                     "correction {correction:.2f} m",
        "legend": ("Pre-ALM prediction", "Safe trajectory at this step",
                   "Continuous safe corridor", "Re-specified start",
                   "Re-specified goal"),
    },
    "zh": {
        "step": "第 {n} 步",
        "warmup": "第 {n} 步（预热）",
        "violation": "违规 {before:.2f}→{after:.2f} m\n修正 {correction:.2f} m",
        "legend": ("ALM 前原始预测", "当前步安全轨迹", "连续安全走廊",
                   "重新设置的起点", "重新设置的终点"),
    },
}


def load_payload() -> dict:
    """Windows 上缓存键保存在 NTFS alternate data stream 中。"""
    stream_path = f"{CACHE_BASE}:{CACHE_STREAM}"
    with open(stream_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(output: Path = OUTPUT) -> None:
    configure_fonts(verbose=False)
    text = LABELS[LANG]
    payload = load_payload()
    occupancy = np.load(OCCUPANCY, mmap_mode="r")[293]
    condition = np.asarray(payload["condition"], dtype=float)
    cells = payload["corridor"]["cells"]
    frames = payload["alm"]["frames"]

    expected = np.asarray(
        [[0.8948432803153992, -0.8210313320159912],
         [0.9211324453353882, 0.8169868588447571]], dtype=float)
    if not np.allclose(condition, expected, atol=1e-8):
        raise RuntimeError(f"缓存中的自定义起终点发生变化：{condition.tolist()}")
    if payload["steps"] != 16 or not payload["alm"]["enabled"]:
        raise RuntimeError("缓存不是图中所需的 16 步 ALM 运行")
    if payload["alm"].get("final_project"):
        raise RuntimeError("该图要求末端投影关闭，但缓存中已开启")

    # 单排：5 个正方形面板并排，图例放在整幅图的下方一行。面板上方留出一条
    # 标题带，用来放"步数 + 违规/修正数值"，避免文字压在图上。
    fig, axes = plt.subplots(1, len(FRAME_INDICES), figsize=(11.2, 3.35))
    fig.subplots_adjust(left=0.006, right=0.994, top=0.795, bottom=0.145,
                        wspace=0.045)

    for ax, frame_idx in zip(axes, FRAME_INDICES):
        frame = frames[frame_idx]
        raw = np.asarray(frame["raw_p"], dtype=float)
        safe = np.asarray(frame["safe_p"], dtype=float)

        ax.imshow(
            occupancy,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            zorder=0,
        )
        for cell in cells:
            polygon = np.asarray(cell.get("polygon") or [], dtype=float)
            if polygon.ndim == 2 and polygon.shape[0] >= 3:
                ax.add_patch(
                    Polygon(
                        polygon,
                        closed=True,
                        facecolor="#56B4E9",
                        edgecolor="#0072B2",
                        linewidth=0.28,
                        alpha=0.12,
                        zorder=1,
                    )
                )

        ax.plot(raw[:, 0], raw[:, 1], color="#D55E00", ls="--", lw=1.25,
                alpha=0.90, zorder=3)
        ax.plot(safe[:, 0], safe[:, 1], color="#0072B2", lw=1.85, zorder=4)
        ax.scatter(
            condition[0, 0], condition[0, 1], s=72, marker="o",
            facecolor="#4ade80", edgecolor="black", linewidth=1.0, zorder=6,
        )
        ax.scatter(
            condition[1, 0], condition[1, 1], s=125, marker="*",
            facecolor="#fb7185", edgecolor="black", linewidth=0.9, zorder=6,
        )

        step_no = frame_idx + 1
        stats = frame.get("stats") or {}
        # 标题带：第一行是步数，第二/三行是违规与修正量，全部在面板**上方**。
        ax.set_title(text["step"].format(n=step_no) if frame.get("alm_active")
                     else text["warmup"].format(n=step_no),
                     fontsize=12.0, pad=30)
        if frame.get("alm_active"):
            before = float(stats.get("max_violation_before", np.nan)) * SCENE_TO_METER
            after = float(stats.get("max_violation_after", np.nan)) * SCENE_TO_METER
            correction = float(stats.get("mean_curve_correction_m", np.nan))
            ax.text(
                0.5, 1.015,
                text["violation"].format(before=before, after=after,
                                         correction=correction),
                transform=ax.transAxes, fontsize=9.0, va="bottom", ha="center",
                color="#1F2937", linespacing=1.35,
            )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#94A3B8")
            spine.set_linewidth(0.65)

    handles = [
        Line2D([], [], color="#D55E00", ls="--", lw=1.6),
        Line2D([], [], color="#0072B2", lw=2.2),
        Line2D([], [], color="#56B4E9", lw=8, alpha=0.25),
        Line2D([], [], marker="o", color="none", markerfacecolor="#4ade80",
               markeredgecolor="black", markersize=9),
        Line2D([], [], marker="*", color="none", markerfacecolor="#fb7185",
               markeredgecolor="black", markersize=12),
    ]
    fig.legend(handles, list(text["legend"]), loc="lower center",
               bbox_to_anchor=(0.5, 0.002), ncol=len(handles), frameon=False,
               fontsize=11.5, handlelength=2.4, handletextpad=0.7,
               columnspacing=2.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT)
