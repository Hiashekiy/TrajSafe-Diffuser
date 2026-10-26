#!/usr/bin/env python
"""绘制第4章随机起终点补充实验（4张地图 × 3个距离档）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from ch4_plot import configure_fonts


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "ch4" / "random_endpoints" / "results.json"
OCCUPANCY = ROOT / "data" / "carla_processed_160k4p_c48" / "test" / "occupancy.npy"
OUTPUT = ROOT / "outputs" / "ch4" / "figures" / "random_endpoint_distances.png"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS),
                        help="eval_random_endpoints.py 生成的 results.json")
    parser.add_argument("--out", default=str(OUTPUT), help="输出 PNG 路径")
    parser.add_argument("--occupancy", default=str(OCCUPANCY), help="Occupancy array for the frozen experiment")
    parser.add_argument(
        "--pair-indices", default="0,1",
        help="每个距离档选出的两个样本下标（从0开始），例如 0,1 或 2,3")
    args = parser.parse_args()
    global_occupancy = Path(args.occupancy)
    try:
        pair_indices = tuple(int(token.strip())
                             for token in args.pair_indices.split(","))
    except ValueError as error:
        raise SystemExit("--pair-indices 必须是两个逗号分隔的非负整数") from error
    if len(pair_indices) != 2 or min(pair_indices) < 0:
        raise SystemExit("--pair-indices 必须正好包含两个非负整数，例如 0,1")

    configure_fonts(verbose=False)
    results_path = Path(args.results).resolve()
    output_path = Path(args.out).resolve()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    occupancy = np.load(global_occupancy, mmap_mode="r")
    maps = [int(row["map_index"]) for row in data["selected_maps"]]
    grouped = {}
    for record in data["records"]:
        grouped.setdefault((int(record["map_index"]), record["tier"]), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda row: int(row["pair_no"]))
    a, b = pair_indices
    columns = (("near", a), ("near", b),
               ("medium", a), ("medium", b),
               ("far", a), ("far", b))
    column_titles = ("近距离 1", "近距离 2", "中距离 1", "中距离 2",
                     "远距离 1", "远距离 2")
    fig, axes = plt.subplots(4, 6, figsize=(11.6, 7.55))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.015,
                        hspace=0.045, wspace=0.025)
    for row_no, map_index in enumerate(maps):
        for col_no, (tier, pair_index) in enumerate(columns):
            ax = axes[row_no, col_no]
            ax.imshow(occupancy[map_index], origin="lower", extent=(-1, 1, -1, 1),
                      cmap="gray_r", vmin=0, vmax=1, interpolation="nearest",
                      zorder=0)
            records = grouped[(map_index, tier)]
            if pair_index >= len(records):
                raise SystemExit(
                    f"地图 {map_index}/{tier} 只有 {len(records)} 个样本，"
                    f"不能选择下标 {pair_index}；请增大 --pairs-per-tier 后重跑实验")
            record = records[pair_index]
            result = record["result"]
            validation = result["final_validation"]
            raw = np.asarray(result["raw_final_curve"], dtype=float)
            final = np.asarray(result["final_curve"], dtype=float)
            condition = np.asarray(record["condition"], dtype=float)
            for cell in result["corridor_cells"]:
                polygon = np.asarray(cell.get("polygon") or [], dtype=float)
                if polygon.ndim == 2 and len(polygon) >= 3:
                    ax.add_patch(Polygon(
                        polygon, closed=True, facecolor="#38bdf8",
                        edgecolor="#0072B2", linewidth=0.20,
                        alpha=0.095, zorder=1))
            collided = bool(validation["final_collision"])
            ax.plot(raw[:, 0], raw[:, 1], color="#D55E00", ls="--",
                    lw=0.95, alpha=0.88, zorder=3)
            ax.plot(final[:, 0], final[:, 1],
                    color="#DC2626" if collided else "#0072B2",
                    ls="--" if collided else "-", lw=1.65, alpha=1.0,
                    zorder=4)
            ax.scatter(condition[0, 0], condition[0, 1], s=36, marker="o",
                       facecolor="#4ade80", edgecolor="black", linewidth=0.6,
                       zorder=6)
            ax.scatter(condition[1, 0], condition[1, 1], s=62, marker="*",
                       facecolor="#fb7185", edgecolor="black", linewidth=0.55,
                       zorder=6)
            if row_no == 0:
                ax.set_title(column_titles[col_no], fontsize=9.2, pad=3)
            ax.text(
                0.04, 0.94, f"{float(record['geodesic_m']):.1f} m",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
                color="#334155",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.82), zorder=8,
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#64748b")
                spine.set_linewidth(0.7)
            if col_no == 0:
                ax.set_ylabel(f"场景 {row_no + 1}", fontsize=9.3,
                              rotation=90, labelpad=5)

    legend = [
        Line2D([], [], color="#D55E00", ls="--", lw=1.0,
               label="ALM前原始预测"),
        Line2D([], [], color="#0072B2", lw=1.7, label="最终轨迹"),
        Line2D([], [], color="#38bdf8", lw=6, alpha=0.18,
               label="连续安全走廊"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#4ade80",
               markeredgecolor="black", markersize=7, label="随机起点"),
        Line2D([], [], marker="*", color="none", markerfacecolor="#fb7185",
               markeredgecolor="black", markersize=10, label="随机终点"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=5, fontsize=8.2, frameon=False, columnspacing=1.25,
               handlelength=2.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
