#!/usr/bin/env python
"""汇总 实验 1/2/3 的结果为论文可直接引用的表格（统一精度）。

只读取各实验已写出的 summary/CSV，不重新计算任何指标，避免汇总层引入新的
口径差异。

    python scripts/experiments/summarize.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = os.path.join(C.OUT_ROOT, "paper")
RES = C.OUT_ROOT


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _num(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return ("%." + str(digits) + "f") % value
    return str(value)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def experiment1():
    summary = _read(os.path.join(RES, "exp1", "summary.json"))
    if not summary:
        return None, ["_实验 1 尚未运行_"]
    order = ["none", "penalty", "primal_dual", "alm_sdf", "TrajSafe"]
    rows = []
    for tag in order:
        s = summary["summary"].get(tag)
        if not s:
            continue
        rows.append([s["name"], _num(s["collision_pct"], 2),
                     _num(s["clearance_m"], 3), _num(s["detour_ratio"], 4),
                     _num(s["nrmse_pct"], 3), _num(s["kappa_max_inv_m"], 4),
                     _num(s["kappa2_mean_inv_m2"], 5), str(s["n"])])
    header = ["Method", "Collision (%)", "Min clearance (m)",
              "Detour ratio", "NRMSE (%)", "Max curvature (1/m)",
              "Sq. curvature (1/m2)", "N"]
    out = [_table(header, rows), ""]
    eff = _read(os.path.join(RES, "exp1", "efficiency_summary.json"))
    if eff:
        acc = _read(os.path.join(C.CAMPAIGN, "efficiency_corrected_20261010",
                                 "timing.json"))
        arows = []
        label = [("steps16_stride1", "Standard", 16, 128),
                 ("steps4_stride1", "Fewer steps", 4, 128),
                 ("steps16_stride8", "Shared regions", 16, 17),
                 ("steps4_stride8", "Combined", 4, 17)]
        for key, name, steps, regions in label:
            cfgv = (acc or {}).get("acceleration", {}).get(key)
            if not cfgv:
                continue
            arows.append([name, str(steps), str(regions),
                          _num(cfgv.get("wall_ms"), 2),
                          "%d/%d" % (cfgv.get("collisions", 0), cfgv.get("n", 0))])
        out += ["**Table 1b — efficiency (reused historical interleaved timing, "
                "speed-up %.2fx)**" % eff["speedup"], "",
                _table(["Configuration", "DDIM steps", "Base regions",
                        "Time (ms/traj)", "Collisions"], arows), ""]
    return eff, out


def experiment2():
    abl = _read(os.path.join(RES, "exp2", "ablation_summary.json"))
    proc = _read(os.path.join(RES, "exp2", "feedback_process_summary.json"))
    out = []
    if abl:
        rows = []
        for row in abl["table"]:
            rows.append([row["config"], _num(row["safety_cond"]),
                         _num(row["history_feedback"]),
                         _num(row["alm_safety_update"]),
                         _num(row["collision_pct"], 2),
                         _num(row["clearance_m"], 3),
                         _num(row["detour_ratio"], 4),
                         _num(row["nrmse_pct"], 3), str(row["n"])])
        out += ["**Table 2 — component ablation (A0–A3)**", "",
                _table(["Config", "Safety cond.", "Hist. feedback",
                        "ALM update", "Collision (%)", "Min clearance (m)",
                        "Detour", "NRMSE (%)", "N"], rows), ""]
    if proc and proc.get("paired"):
        rows = []
        for key, label in (("pre_violation_m", "Pre-correction violation (m)"),
                           ("correction_m", "ALM correction (m)")):
            p = proc["paired"].get(key)
            if not p:
                continue
            rows.append([label, _num(p["mean_without"], 4),
                         _num(p["mean_with"], 4), _num(p["mean_diff"], 4),
                         _num(p.get("std_diff"), 4),
                         _num(p.get("wilcoxon_p"), 3), str(p["n_pairs"])])
        out += ["**Table 2b — paired process statistics, forward %d–16**"
                % proc["paired"]["pre_violation_m"]["forward_from"], "",
                _table(["Quantity", "Without FB", "With FB", "Diff",
                        "Std. diff", "Wilcoxon p", "Pairs"], rows), ""]
    return proc, out


def experiment3():
    """实验 3（ALM 迭代预算）：安全/质量表 + 计算开销表 + 配对 bootstrap。"""
    summary = _read(os.path.join(RES, "exp3", "table3_summary.json"))
    if not summary or "table1" not in summary:
        return None, ["_实验 3 尚未运行_"]
    order = ["I2", "I4", "I6", "I8", "I10", "Default"]
    t1 = {r["config"]: r for r in summary["table1"]}
    rows = []
    for tag in order:
        r = t1.get(tag)
        if not r:
            continue
        rows.append([tag, _num(r["constraint_feasible_pct"], 2),
                     _num(r["collision_pct"], 2),
                     "%d/%d" % (r["collisions"], r["n"]),
                     _num(r["clearance_m"], 3), _num(r["detour_ratio"], 4),
                     _num(r["nrmse_pct"], 3), _num(r["kappa_max_inv_m"], 4),
                     _num(r["kappa2_mean_inv_m2"], 5)])
    out = ["**Table 3 — ALM iteration budget: safety and trajectory quality**", "",
           _table(["Config", "Continuous feasible (%)", "Collision (%)",
                   "Collisions/N", "Min clearance (m)", "Detour", "NRMSE (%)",
                   "Max curvature (1/m)", "Sq. curvature (1/m2)"], rows), ""]
    t2 = {r["config"]: r for r in summary["table2"]}
    rows = []
    for tag in order:
        r = t2.get(tag)
        if not r:
            continue
        rows.append([tag, _num(r["iterations_mean"], 3),
                     _num(r["iterations_p95"], 2), _num(r["at_budget_pct"], 1),
                     _num(r["alm_ms_per_call"], 2),
                     _num(r.get("e2e_ms_batch32_median"), 2),
                     _num(r["e2e_ms_batch1"], 2),
                     _num(r.get("e2e_ms_batch1_median"), 2),
                     _num(r["e2e_ms_batch1_p95"], 2)])
    out += ["**Table 3b — cost and solver diagnostics**", "",
            _table(["Config", "Mean iters", "P95 iters", "At budget (%)",
                    "ALM ms/call", "E2E ms (b=32, med)", "E2E ms (b=1, mean)",
                    "E2E ms (b=1, med)", "E2E P95 (b=1)"], rows), ""]
    boot = summary.get("bootstrap_vs_default") or {}
    if boot:
        rows = []
        for tag in order:
            entry = boot.get(tag) or {}
            for key in ("constraint_feasible", "collision", "detour_ratio",
                        "nrmse_pct"):
                ci = entry.get(key)
                if not ci:
                    continue
                rows.append([tag, key, _num(ci["mean"], 4), _num(ci["lo"], 4),
                             _num(ci["hi"], 4), str(ci["n"])])
        if rows:
            out += ["**Table 3c — paired bootstrap vs Default (95% CI, "
                    "resampled by task)**", "",
                    _table(["Config", "Quantity", "Mean diff", "CI low",
                            "CI high", "Tasks"], rows), ""]
    return summary, out


def main() -> int:
    C.ensure_dir(OUT)
    sections = []
    eff, e1 = experiment1()
    proc, e2 = experiment2()
    t3, e3 = experiment3()
    sections += ["# TrajSafe-Diffuser 论文实验表格（自动汇总）", "",
                 "由 `scripts/experiments/summarize.py` 从各实验的原始结果生成；"
                 "所有数字均来自实际运行，未做任何人工填写。", "",
                 "## 实验 1：不同安全规划方法对比与计算效率", ""] + e1
    sections += ["## 实验 2：关键模块消融与历史反馈作用分析", ""] + e2
    sections += ["## 实验 3：ALM 迭代预算与安全性-计算效率权衡", ""] + e3
    path = os.path.join(OUT, "PAPER_TABLES.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections) + "\n")
    print("[summarize] wrote %s" % path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
