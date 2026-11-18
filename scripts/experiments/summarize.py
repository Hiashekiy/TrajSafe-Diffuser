#!/usr/bin/env python
"""汇总 实验 1/2/3 的结果为论文可直接引用的表格（统一精度）。

只读取各实验已写出的 summary/CSV，不重新计算任何指标，避免汇总层引入新的
口径差异。

    python scripts/experiments/summarize.py
"""

from __future__ import annotations

import csv
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
    """实验 3（单次 ALM 调用的迭代预算扫描）。

    数据源是 ``outputs/alm_budget_analysis``（缓存输入 + 预算回放），
    不是旧版的整链路预算表。
    """
    out_dir = os.path.join(C.ROOT, "outputs", "alm_budget_analysis")
    summary_csv = os.path.join(out_dir, "summary.csv")
    if not os.path.exists(summary_csv):
        return None, ["_实验 3 尚未运行_"]
    with open(summary_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    run_cfg = {}
    cfg_path = os.path.join(out_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as handle:
            run_cfg = json.load(handle)

    table_rows = []
    for r in rows:
        table_rows.append([
            "I%s" % r["budget"], _num(float(r["feasible_rate_pct"]), 2),
            "[%s, %s]" % (_num(float(r["feasible_ci_lo"]), 2),
                          _num(float(r["feasible_ci_hi"]), 2)),
            _num(float(r["violation_after_mean_scene"]), 5),
            _num(float(r["violation_after_p95_scene"]), 5),
            _num(float(r["correction_mean_m"]), 5),
            _num(float(r["iterations_mean"]), 2),
            str(r["n_calls"])])
    out = ["**Table 4 — ALM iteration budget vs constraint-correction quality** "
           "(main set: calls whose cached input already violated the corridor)",
           "",
           _table(["Max iters", "Feasible calls (%)", "95% CI",
                   "Residual violation (scene)", "P95", "Correction (m)",
                   "Actual iters", "Calls"], table_rows), ""]
    st = (run_cfg.get("stats") or {})
    man = (run_cfg.get("manifest") or {})
    if st:
        out += ["Coverage: cached calls %s (all valid), baseline-satisfied "
                "%.2f%%, main set %s calls / %s scenes / %s trajectories; "
                "parity check max |diff| = %s."
                % (st.get("n_cached_calls"), st.get("baseline_satisfied_pct") or 0.0,
                   st.get("n_calls_baseline_infeasible"), st.get("n_scenes_main"),
                   st.get("n_trajectories_main"),
                   (st.get("parity") or {}).get("max_abs_diff")), ""]
    if man:
        out += ["Model: `%s` (sha256 `%s`), branch `%s`."
                % (man.get("checkpoint"), man.get("checkpoint_hash"),
                   man.get("branch")), ""]
    return run_cfg, out


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
    sections += ["## 实验 3：ALM 迭代预算对安全约束修正效果的影响", ""] + e3
    path = os.path.join(OUT, "PAPER_TABLES.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections) + "\n")
    print("[summarize] wrote %s" % path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
