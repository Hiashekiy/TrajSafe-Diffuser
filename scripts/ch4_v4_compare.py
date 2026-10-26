#!/usr/bin/env python
"""Compare the v3 (A-C48) and v4 chapter-4 "Full" test runs, same protocol.

Reads ``outputs/chapter4_v4/runs/*.json`` (+ ``*.jsonl`` for the paired,
per-sample test) and the archived ``outputs/chapter4/runs/*.json`` as a third,
historical column, then writes ``outputs/chapter4_v4/COMPARE_v3_vs_v4.md``.

    python scripts/ch4_v4_compare.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = os.path.join(ROOT, "outputs", "chapter4_v4", "runs")
ARCH = os.path.join(ROOT, "outputs", "chapter4", "runs")
OUT = os.path.join(ROOT, "outputs", "chapter4_v4", "COMPARE_v3_vs_v4.md")

V3 = "Full (A-C48+ALM+FB+Proj)"
V4 = "Full (v4-map16+ALM+FB+Proj)"
L1 = "Full (v4-map16-L1+ALM+FB+Proj)"

#: aggregate keys reported in the table, in reading order
KEYS = [
    ("collision_rate", "碰撞率", "rate", 4),
    ("free_rate", "自由空间率", "rate", 4),
    ("rmse_m", "曲线 RMSE (m)", "m", 3),
    ("smoothness_m", "平滑度 2nd-diff (m)", "m", 4),
    ("endpoint_error", "端点误差", "m", 4),
    ("sdf_feasible_rate", "SDF 可行率", "rate", 4),
    ("sdf_mean_violation_m", "SDF 平均违反 (m)", "m", 4),
    ("sdf_p95_violation_m", "SDF p95 违反 (m)", "m", 4),
    ("sdf_min_distance_m", "最小障碍距离 (m)", "m", 3),
    ("offline_corridor_membership_rate", "离线走廊成员率", "rate", 4),
    ("offline_corridor_max_violation", "离线走廊最大违反", "raw", 4),
    ("online_corridor_membership_rate", "在线走廊成员率", "rate", 4),
    ("runtime_ms_mean", "单条推理 (ms)", "ms", 1),
    ("solver_failure_samples", "求解失败样本", "int", 0),
]


def load_runs(directory: str, method: str) -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "%s__s*.json" % method))):
        with open(path, "r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def load_records(directory: str, method: str, seed: int = 0) -> dict:
    path = os.path.join(directory, "%s__s%d.jsonl" % (method, seed))
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            out[int(rec["index"])] = rec
    return out


def stat(rows, key):
    vals = [float(r["aggregate"][key]) for r in rows if key in r["aggregate"]]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.stdev(vals)


def fmt(value, kind: str, digits: int) -> str:
    if value is None:
        return "—"
    mean, dev = value
    if kind == "int":
        return "%d" % round(mean)
    if kind == "rate":
        return "%.2f%%" % (100.0 * mean)
    if dev == 0.0:
        return ("%%.%df" % digits) % mean
    return ("%%.%df ± %%.%df" % (digits, digits)) % (mean, dev)


def paired_diff(method_a: str, method_b: str, key: str = "rmse_m") -> dict:
    """Per-sample paired difference (B - A) over the 420 test samples, seed 0."""
    a = load_records(NEW, method_a)
    b = load_records(NEW, method_b)
    common = sorted(set(a) & set(b))
    diffs, better, worse, same = [], 0, 0, 0
    for idx in common:
        va, vb = a[idx][key], b[idx][key]
        if va is None or vb is None:
            continue
        d = float(vb) - float(va)
        diffs.append(d)
        if d < -1e-9:
            better += 1
        elif d > 1e-9:
            worse += 1
        else:
            same += 1
    if not diffs:
        return {}
    n = len(diffs)
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd,
            "ci95": 1.96 * sd / (n ** 0.5) if n > 1 else 0.0,
            "better": better, "worse": worse, "same": same,
            "p50": st.median(diffs)}


def main() -> int:
    v3 = load_runs(NEW, V3)
    v4 = load_runs(NEW, V4)
    l1 = load_runs(NEW, L1)
    arch = load_runs(ARCH, V3)
    if not v3 or not v4:
        raise SystemExit("missing runs: v3=%d v4=%d (run scripts/ch4_v4_eval.py "
                         "and the v3 row first)" % (len(v3), len(v4)))

    lines = []
    add = lines.append
    add("# v3 (A-C48) vs v4 (map16+Δ反馈+L_safe) vs v4-L1（主干 1 层）-- 同协议 test 评测")
    add("")
    add("协议：`scripts/eval_chapter4.py`，split=test **420** 样本，chunk=32，"
        "steps=16，warmup=3，guidance=`alm`，`--feedback on --final-project on`，"
        "seed 0/1/2。三列都用**各自架构的原生代码**：v3 行在 "
        "`git worktree`（commit 74d6140，即 v4 改造前存档）里用 v3 代码跑，"
        "v4 / v4-L1 行在当前分支跑；数据根同为 "
        "`data/processed/160m-k4p-c48`。")
    add("")
    add("- v3: `outputs/models/oneshot_k4p_c48/ckpt/best_task.pt` + "
        "`configs/config_160k4p_c48_oneshot.yaml`（主干 8 层，7.52 M）")
    add("- v4: `outputs/models/v4_map16_c48/ckpt/best_task.pt` + "
        "`configs/config_160k4p_c48_v4_map16.yaml`（主干 8 层，6.52 M）")
    add("- v4-L1: `outputs/models/v4_map16_c48_l1/ckpt/best_task.pt` + "
        "`configs/config_160k4p_c48_v4_map16_l1.yaml`（主干 **1 层**，3.98 M）")
    add("")
    add("## 主表（mean ± std over seeds）")
    add("")
    add("| 指标 | v3 A-C48 (8层) | v4 map16 (8层) | v4-L1 (1层) |")
    add("|---|---|---|---|")
    for key, label, kind, digits in KEYS:
        add("| %s | %s | %s | %s |"
            % (label, fmt(stat(v3, key), kind, digits),
               fmt(stat(v4, key), kind, digits),
               fmt(stat(l1, key), kind, digits)))
    add("")

    for label_a, label_b, method_a, method_b in (
            ("v4 - v3", "RMSE_v4 - RMSE_v3", V3, V4),
            ("L1 - v4", "RMSE_L1 - RMSE_v4", V4, L1)):
        p = paired_diff(method_a, method_b)
        if not p:
            continue
        add("## 逐样本配对检验（420 条 test 样本，seed 0，配对量 = %s）" % label_b)
        add("")
        add("| 对比 | n | 均值差 | 标准差 | 95% CI | 中位差 | 后者更好 | 后者更差 | 持平 |")
        add("|---|---|---|---|---|---|---|---|---|")
        add("| %s | %d | %+.3f m | %.3f | ±%.3f | %+.3f | %d | %d | %d |"
            % (label_a, p["n"], p["mean"], p["sd"], p["ci95"], p["p50"],
               p["better"], p["worse"], p["same"]))
        add("")

    if arch:
        add("## 附：历史归档的 v3 行（commit 05f3781 快照，数据根已不存在，仅作一致性参考）")
        add("")
        add("| 指标 | 归档 v3 | 本次复跑 v3 |")
        add("|---|---|---|")
        for key in ("rmse_m", "collision_rate", "smoothness_m",
                    "offline_corridor_membership_rate", "runtime_ms_mean"):
            add("| %s | %s | %s |"
                % (key, fmt(stat(arch, key), "raw", 4), fmt(stat(v3, key), "raw", 4)))
        add("")

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
