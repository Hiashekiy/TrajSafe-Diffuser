"""Build the markdown result tables of the PDG-Focal A* report from the CSVs.

    python scripts/grid_search/summarize_pdg_results.py --run outputs/grid_search/run_pdg

Every number here is computed from ``records_*.csv`` (one row per case x method),
never from a hand-written constant.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

SUITES = (
    ("main", "主对比：Dijkstra / A* / Focal A*(clearance) / Focal A*(index) / Ours"),
    ("ablation", "进度项消融 G0 / G1 / G2（同一 anchor、同一 w=1.2）"),
    ("synthetic", "合成验收场景 T1–T12（手工构造的参考折线）"),
    ("k_sweep", "K 扫描：用第 K 步反向扩散的中间 x0 作参考"),
    ("w_sweep", "Focal 权重 w 扫描"),
    ("alpha_sweep", "进度权重 alpha_p 扫描"),
)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        v = row.get(key)
        return np.nan if v in (None, "", "None") else float(v)
    except ValueError:
        return np.nan


def stats(rows, key):
    v = np.asarray([fnum(r, key) for r in rows], dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan
    return float(np.median(v))


def group(records):
    out = {}
    for r in records:
        out.setdefault(r["label"], []).append(r)
    return out


def cost_ratio(records):
    opt = {}
    for r in records:
        if r["label"] == "dijkstra" and fnum(r, "path_cost_m") > 0:
            opt[r["case_id"]] = fnum(r, "path_cost_m")
    out = {}
    for r in records:
        base = opt.get(r["case_id"])
        c = fnum(r, "path_cost_m")
        if base and np.isfinite(c):
            out.setdefault(r["label"], []).append(c / base)
    return {k: (float(np.median(v)), float(np.mean(v)), float(np.max(v)))
            for k, v in out.items()}


def suite_table(records, title):
    lines = ["### " + title, ""]
    head = ("| 配置 | case | 成功率 | 独立复核 | 扩展中位数 | 扩展均值 | 扩展 P95 | "
            "搜索 ms 中位数 | 搜索 ms P95 | 端到端 ms 中位数 | 端到端(不含引导) ms | "
            "代价中位数 m | 代价/最优 中位数 |")
    lines += [head, "|" + "|".join(["---"] * 13) + "|"]
    ratios = cost_ratio(records)
    for label, rows in group(records).items():
        n = len(rows)
        succ = np.mean([fnum(r, "success") for r in rows])
        valid = np.mean([fnum(r, "path_valid") for r in rows])
        exp = np.asarray([fnum(r, "expanded_nodes") for r in rows])
        e2e = np.asarray([fnum(r, "end_to_end_ms") for r in rows])
        guid = np.asarray([fnum(r, "guidance_build_time_ms") for r in rows])
        ms = np.asarray([fnum(r, "search_time_ms") for r in rows])
        cost = np.asarray([fnum(r, "path_cost_m") for r in rows])
        rr = ratios.get(label)
        lines.append("| %s | %d | %.3f | %.3f | %.1f | %.0f | %.0f | %.2f | %.1f | "
                     "%.1f | %.1f | %.2f | %s |"
                     % (label, n, succ, valid,
                        np.median(exp), np.mean(exp), np.percentile(exp, 95),
                        np.median(ms), np.percentile(ms, 95),
                        np.nanmedian(e2e), np.nanmedian(e2e - guid),
                        np.nanmedian(cost),
                        "%.3f" % rr[0] if rr else "-"))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = args.run
    out = args.out or os.path.join(run, "REPORT_TABLES.md")
    parts = ["# PDG-Focal A* 结果表（由 records_*.csv 自动生成）", ""]

    summary_path = os.path.join(run, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        cfg = summary.get("args", {})
        parts += ["## 运行配置", "",
                  "| 项 | 值 |", "|---|---|"]
        for k in ("cache", "processed", "split", "scene_to_meter", "cell_m",
                  "limit_main", "limit_sweep", "K", "w", "robot_radius",
                  "safety_margin", "already_inflated_cells", "total_forwards"):
            if k in cfg:
                parts.append("| %s | %s |" % (k, cfg[k]))
        parts.append("")
        ref = summary.get("suites", {}).get("main", {}).get("reference", {})
        if ref:
            parts += ["## 参考轨迹 / 引导场统计（main suite，200 case）", "",
                      "| 指标 | 值 |", "|---|---|"]
            for k, v in ref.items():
                parts.append("| %s | %s |"
                             % (k, ("%.4f" % v) if isinstance(v, float) else v))
            parts.append("")

    for tag, title in SUITES:
        rec_path = os.path.join(run, "records_%s.csv" % tag)
        if not os.path.exists(rec_path):
            continue
        parts.append(suite_table(read_csv(rec_path), title))

    # per-K reference safety
    k_path = os.path.join(run, "records_k_sweep.csv")
    if os.path.exists(k_path):
        rows = [r for r in read_csv(k_path) if r["label"].startswith("pdg_K")]
        parts += ["### 参考轨迹安全性 vs K（同一批 case）", "",
                  "| K | 参考轨迹碰撞率均值 | 存在穿障的 case 比例 | 参考长度中位数 m |",
                  "|---|---|---|---|"]
        for k in sorted({int(r["K"]) for r in rows}):
            sub = [x for x in rows if int(x["K"]) == k]
            coll = np.asarray([fnum(x, "reference_collision_rate")
                               for x in sub])
            parts.append("| %d | %.4f | %.3f | %.1f |"
                         % (k, float(np.nanmean(coll)),
                            float(np.mean(coll > 0)),
                            float(np.nanmedian([fnum(x, "reference_length_m")
                                                for x in sub]))))
        parts.append("")

    text = "\n".join(parts)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("[summary] ->", out)


if __name__ == "__main__":
    main()
