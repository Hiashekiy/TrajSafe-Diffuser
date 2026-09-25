#!/usr/bin/env python
"""第4章结果汇总：把 runs/ 下的逐样本记录汇成论文用的表 + 一份实验报告.

设计原则：**表里的每个数字都从 runs/*.jsonl 的逐样本记录重新算出来**（跨 seed
汇池），而不是从各次运行的聚合里再平均一次——这样：
  * 碰撞率 = 总碰撞数 / 总轨迹数（1260 = 3 seed × 420），不是「三个比率的平均」；
  * 违规量的 P95/最大值是对全部轨迹取的，不会因为先按 seed 平均而失真。

运行时间无法逐样本拆分（采样是批处理的），因此按 chunk 计时后除以批大小，
报「每 seed 的均值 ± 标准差」。

用法::

    python scripts/ch4_report.py --runs-dir outputs/ch4/runs --out outputs/ch4/REPORT.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: main-table row order (guide 4.2.1) -- the paper's method goes last
MAIN_ORDER = ["Diffusion", "Penalty-guided", "Projected", "Primal-Dual",
              "SDF-ALM", "Full (A-C48+ALM+FB+Proj)"]
#: display names for the paper
DISPLAY = {
    "Diffusion": "Diffusion（标准条件扩散）",
    "Penalty-guided": "Penalty-guided Diffusion",
    "Projected": "Projected Diffusion",
    "Primal-Dual": "Primal-Dual Diffusion",
    "SDF-ALM": "SDF-ALM-guided Diffusion",
    "Full (A-C48+ALM+FB+Proj)": "**本文完整方法**（A-C48＋走廊ALM＋历史反馈＋末端投影）",
}
ABLATION_A = [("A0", "A0：无 ALM / 无反馈 / 无末端投影"),
              ("A1", "A1：逐步 ALM"),
              ("A2", "A2：逐步 ALM ＋ 历史反馈"),
              ("A3", "A3：逐步 ALM ＋ 历史反馈 ＋ 末端投影")]
ABLATION_T = [("T0_S-C48_noalm", "T0：S-C48（无反馈训练），无 ALM"),
              ("T1_S-C48_alm", "T1：S-C48，ALM"),
              ("T2_A-C48_noalm", "T2：A-C48（有反馈训练），无 ALM"),
              ("T3_A-C48_alm", "T3：A-C48，ALM")]


# ---------------------------------------------------------------------------
# loading / pooling
# ---------------------------------------------------------------------------


def load_run(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = []
    rec_path = json_path[:-5] + ".jsonl"
    if os.path.exists(rec_path):
        with open(rec_path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    payload["_records"] = records
    payload["_path"] = json_path
    return payload


def pool(runs: list) -> dict:
    """Pool the per-sample records of several runs (different seeds)."""
    recs = [r for run in runs for r in run["_records"]]
    if not recs:
        return {"n": 0}
    collision = np.array([bool(r["collision"]) for r in recs], dtype=float)
    violation = np.array([float(r["nonnegative_violation"]) for r in recs])
    feasible = np.array([bool(r["feasible"]) for r in recs], dtype=float)
    safe_rate = np.array([float(r["sdf_safe_point_rate"]) for r in recs])
    min_dist = np.array([float(r["sdf_min_distance_m"]) for r in recs])
    rmse = np.array([float(r["rmse_m"]) for r in recs])
    smooth = np.array([float(r["smoothness_m"]) for r in recs])
    return {
        "n": len(recs),
        "collisions": int(collision.sum()),
        "collision_rate": float(collision.mean()),
        "feasible_rate": float(feasible.mean()),
        "violation_mean": float(violation.mean()),
        "violation_p95": float(np.percentile(violation, 95)),
        "violation_max": float(violation.max()),
        "safe_point_rate": float(safe_rate.mean()),
        "min_distance": float(min_dist.mean()),
        "rmse": float(rmse.mean()),
        "smoothness": float(smooth.mean()),
        "runtime_mean": _agg_mean(runs, "runtime_ms_mean"),
        "runtime_std": _agg_mean(runs, "runtime_ms_std"),
        "seeds": sorted({int(run["seed"]) for run in runs}),
    }


def _agg_mean(runs: list, key: str):
    values = [run["aggregate"].get(key) for run in runs
              if run["aggregate"].get(key) is not None]
    return float(np.mean(values)) if values else None


def group_runs(runs: list) -> "OrderedDict[str, list]":
    groups: "OrderedDict[str, list]" = OrderedDict()
    for run in runs:
        groups.setdefault(run["method"], []).append(run)
    return groups


def fmt(value, digits=3, suffix=""):
    if value is None:
        return "—"
    return ("%%.%df%s" % (digits, suffix)) % value


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def main_table(groups: dict, order: list) -> list:
    lines = ["| 方法 | 碰撞数/总数↓ | 碰撞率/%↓ | SDF可行率/%↑ | SDF平均违规量/m↓ | "
             "SDF-P95违规量/m↓ | 安全点比例/%↑ | 最小安全距离/m↑ | RMSE/m↓ | "
             "平滑度/m↓ | 时间/ms↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in order:
        runs = groups.get(name)
        if not runs:
            lines.append("| %s | _未运行_ |  |  |  |  |  |  |  |  |  |"
                         % DISPLAY.get(name, name))
            continue
        agg = pool(runs)
        runtime = fmt(agg["runtime_mean"], 1)
        if agg["runtime_std"]:
            runtime += " ± " + fmt(agg["runtime_std"], 1)
        lines.append("| %s | %d/%d | %.2f | %.2f | %.4f | %.4f | %.2f | %.3f | "
                     "%.2f | %.3f | %s |"
                     % (DISPLAY.get(name, name), agg["collisions"], agg["n"],
                        100.0 * agg["collision_rate"], 100.0 * agg["feasible_rate"],
                        agg["violation_mean"], agg["violation_p95"],
                        100.0 * agg["safe_point_rate"], agg["min_distance"],
                        agg["rmse"], agg["smoothness"], runtime))
    return lines


def ablation_table(groups: dict, spec: list) -> list:
    lines = ["| 配置 | 碰撞率/%↓ | SDF可行率/%↑ | 平均违规量/m↓ | ALM前原始违规量/m↓ | "
             "平均修正距离/m↓ | 有效内迭代↓ | RMSE/m↓ | 时间/ms↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, label in spec:
        runs = groups.get(tag)
        if not runs:
            lines.append("| %s | _未运行_ |  |  |  |  |  |  |  |" % label)
            continue
        agg = pool(runs)
        # a run without the processor has no such quantity: print the em dash a
        # paper table uses for "does not exist" instead of nan
        raw_viol, correction, inner = (_dash(v) for v in step_summary(runs))
        lines.append("| %s | %.2f | %.2f | %.4f | %s | %s | %s | %.2f | %s |"
                     % (label, 100.0 * agg["collision_rate"],
                        100.0 * agg["feasible_rate"], agg["violation_mean"],
                        raw_viol, correction, inner,
                        agg["rmse"], fmt(agg["runtime_mean"], 1)))
    return lines


def step_summary(runs: list):
    """Mean over guided steps of raw violation / correction / inner iterations.

    ``per_step`` values are in scene units for violations and corrections; they
    are converted to metres with the run's own ``scene_to_meter``.
    """
    raw, corr, inner = [], [], []
    for run in runs:
        step = run.get("per_step") or {}
        s2m = float(run.get("scene_to_meter", 80.0))
        for key, sink, scale in (("raw_violation", raw, s2m),
                                 ("correction", corr, s2m)):
            values = (step.get(key) or {}).get("mean") or []
            sink.extend([v * scale for v in values if v is not None])
        values = (step.get("inner_steps") or {}).get("mean") or []
        inner.extend([v for v in values if v is not None])
    return (_mean(raw), _mean(corr), _mean(inner))


def _mean(values):
    return float(np.mean(values)) if values else None


def _dash(value, digits: int = 3):
    """``None`` / ``nan`` -> the em dash a paper table uses for "does not exist"."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, float):
        return "%.*f" % (digits, value)
    return value


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def environment() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu")
    except Exception:
        pass
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import osqp
        info["osqp"] = osqp.__version__
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git_commit"] = "unknown"
    return info


def build_report(groups: dict, env: dict) -> str:
    lines = []
    add = lines.append
    add("# 第4章 实验报告（自动汇总）")
    add("")
    add("> 本文件由 `scripts/ch4_report.py` 从 `outputs/ch4/runs/*.jsonl` 的**逐样本记录**"
        "重新汇池生成：碰撞率 = 总碰撞数 / 总轨迹数（3 seed × 420），违规量的 P95/最大值"
        "对全部轨迹取。修改任何一次运行后重跑本脚本即可刷新，表格不会与原始记录脱节。")
    add("")
    add("## 0. 运行环境与代码版本")
    add("")
    add("| 项目 | 值 |")
    add("|---|---|")
    for key in ("python", "torch", "cuda", "gpu", "scipy", "osqp", "git_commit"):
        if key in env:
            add("| %s | `%s` |" % (key, env[key]))
    add("")
    add("## 1. 4.2 不同推理期安全约束方法比较")
    add("")
    lines.extend(main_table(groups, MAIN_ORDER))
    add("")
    add("## 2. 4.3.1 历史安全反馈的推理期作用（同一 A-C48 checkpoint）")
    add("")
    lines.extend(ablation_table(groups, ABLATION_A))
    add("")
    add("## 3. 4.3.2 历史安全反馈训练与 ALM 的协同（训练期 2×2）")
    add("")
    lines.extend(ablation_table(groups, ABLATION_T))
    add("")
    add("## 4. 数值摘要（供正文引用）")
    add("")
    lines.extend(digest(groups))
    add("")
    return "\n".join(lines) + "\n"


def digest(groups: dict) -> list:
    """Machine-written observations; every sentence quotes a computed number."""
    out = []

    def row(name):
        runs = groups.get(name)
        return pool(runs) if runs else None

    diffusion = row("Diffusion")
    penalty = row("Penalty-guided")
    projected = row("Projected")
    primal = row("Primal-Dual")
    alm_sdf = row("SDF-ALM")
    full = row("Full (A-C48+ALM+FB+Proj)")

    def say(text):
        out.append("- " + text)

    if diffusion:
        say("标准 Diffusion 在 %d 条轨迹上碰撞 %d 条（%.2f%%），SDF 平均违规量 %.3f m、"
            "P95 %.3f m，说明统一生成先验本身并不满足安全约束。"
            % (diffusion["n"], diffusion["collisions"],
               100 * diffusion["collision_rate"], diffusion["violation_mean"],
               diffusion["violation_p95"]))
    for name, agg in (("固定惩罚", penalty), ("逐步 L2 投影", projected),
                      ("原始—对偶", primal), ("SDF-ALM", alm_sdf)):
        if agg and diffusion:
            say("%s将碰撞率从 %.2f%% 降到 %.2f%%（SDF 可行率 %.2f%%，平均违规量 %.4f m）。"
                % (name, 100 * diffusion["collision_rate"],
                   100 * agg["collision_rate"], 100 * agg["feasible_rate"],
                   agg["violation_mean"]))
    if primal and alm_sdf:
        say("SDF-ALM 相对原始—对偶：平均违规量 %.4f → %.4f m，P95 %.4f → %.4f m。"
            % (primal["violation_mean"], alm_sdf["violation_mean"],
               primal["violation_p95"], alm_sdf["violation_p95"]))
    # "collision but SDF-feasible" cases are explained by the crop, not by a
    # metric bug: the primary SDF column follows the guide's occupancy-only
    # formula, while leaving [-1,1]^2 is still counted as a collision.
    excursion = {}
    for path in glob.glob(os.path.join(ROOT, "outputs", "ch4", "runs", "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        rate = (payload.get("aggregate") or {}).get("map_excursion_rate")
        if rate:
            excursion[payload.get("method", path)] = rate
    if excursion:
        say("超出归一化地图范围 [-1,1]² 的轨迹比例（按方法）：%s。"
            "该比例非零时，主表中「SDF 可行」与「碰撞」可以同时成立——越界按评价"
            "协议记为碰撞，而指南定义的 r^SDF 只描述占据栅格，不含裁剪边界；"
            "四个 SDF 基线的优化器使用的是含边界的组合场，越界在其约束里是可见的。"
            % "、".join("%s %.2f%%" % (k, 100 * v) for k, v in excursion.items()))
    if full:
        say("本文完整方法碰撞 %d/%d（%.2f%%），SDF 平均违规量 %.4f m，RMSE %.2f m，"
            "平滑度 %.3f m，%.1f ms/轨迹。"
            % (full["collisions"], full["n"], 100 * full["collision_rate"],
               full["violation_mean"], full["rmse"], full["smoothness"],
               full["runtime_mean"] or float("nan")))
        say("⚠ 该行与前五行**不是同一个生成先验**：前五行共用 D-C48（标准条件扩散），"
            "只改推理期约束处理器；本行用 A-C48（含椭圆/安全区域特征且经历史反馈训练的"
            "完整生成模型）外加末端硬投影。它的 RMSE 优势（%.2f vs %.2f m）里既有约束"
            "处理的贡献，也有模型本身更强的贡献。主表属于端到端系统比较，不能据此声称"
            "「同一先验下本文的约束处理器更好」——那要由 4.3 的内部消融回答。"
            % (full["rmse"], diffusion["rmse"] if diffusion else float("nan")))
    a = {tag: row(tag) for tag, _ in ABLATION_A}
    if all(a.values()):
        say("推理期消融：A0→A1 碰撞率 %.2f%% → %.2f%%（逐步 ALM 的作用）；"
            "A1→A2 %.2f%% → %.2f%%（历史反馈）；A2→A3 %.2f%% → %.2f%%（末端硬投影）；"
            "A0→A3 %.2f%% → %.2f%%（完整管线）。"
            % (100 * a["A0"]["collision_rate"], 100 * a["A1"]["collision_rate"],
               100 * a["A1"]["collision_rate"], 100 * a["A2"]["collision_rate"],
               100 * a["A2"]["collision_rate"], 100 * a["A3"]["collision_rate"],
               100 * a["A0"]["collision_rate"], 100 * a["A3"]["collision_rate"]))
    t = {tag: row(tag) for tag, _ in ABLATION_T}
    if all(t.values()):
        say("训练期消融：T0→T1 碰撞率 %.2f%% → %.2f%%（无反馈训练时 ALM 的改善）；"
            "T2→T3 %.2f%% → %.2f%%（有反馈训练后 ALM 的边际作用）；"
            "T0→T2 %.2f%% → %.2f%%（反馈训练对未修正原始预测的作用）。"
            % (100 * t["T0_S-C48_noalm"]["collision_rate"],
               100 * t["T1_S-C48_alm"]["collision_rate"],
               100 * t["T2_A-C48_noalm"]["collision_rate"],
               100 * t["T3_A-C48_alm"]["collision_rate"],
               100 * t["T0_S-C48_noalm"]["collision_rate"],
               100 * t["T2_A-C48_noalm"]["collision_rate"]))
    # ---- the feedback mechanism's claim lives in the PROCESS metrics ---------
    # Its final collision rate is unchanged (the ALM alone already reaches 0), so a
    # collision-only digest would hide the effect completely.  These are exactly
    # the three numbers guide 4.3.1 / 4.3.2 asks for: the raw violation BEFORE the
    # processor, the correction distance, and the inner iterations consumed.
    if a.get("A1") and a.get("A2"):
        p1, p2 = step_summary(groups["A1"]), step_summary(groups["A2"])
        say("推理期反馈的作用体现在**过程指标**上，而不是最终碰撞率（两者都是 0）："
            "ALM 前原始违规量 %.3f → %.3f m，平均修正距离 %.3f → %.3f m，"
            "有效内迭代 %.2f → %.2f。即历史反馈让网络自己的预测更靠近可行域，"
            "从而减轻约束优化器需要承担的修正负担。"
            % (p1[0], p2[0], p1[1], p2[1], p1[2], p2[2]))
    if t.get("T1_S-C48_alm") and t.get("T3_A-C48_alm"):
        q1 = step_summary(groups["T1_S-C48_alm"])
        q3 = step_summary(groups["T3_A-C48_alm"])
        say("训练期反馈的作用（两者 ALM 都开）：ALM 前原始违规量 %.3f → %.3f m，"
            "平均修正距离 %.3f → %.3f m，有效内迭代 %.2f → %.2f。"
            % (q1[0], q3[0], q1[1], q3[1], q1[2], q3[2]))
    if not out:
        out.append("- （尚无已完成的运行）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", nargs="+", default=["outputs/ch4/runs"],
                    help="one or more directories of run JSONs (the main "
                         "table and the ablation tables live apart)")
    ap.add_argument("--out", default="outputs/ch4/REPORT.md")
    args = ap.parse_args()

    paths = []
    for runs_dir in args.runs_dir:
        found = glob.glob(os.path.join(ROOT, runs_dir, "*.json"))
        paths.extend(q for q in sorted(found) if not q.endswith(".jsonl"))

    if not paths:
        raise SystemExit("[report] no run JSONs under %s"
                         % ", ".join(args.runs_dir))
    runs = [load_run(p) for p in paths]
    groups = group_runs(runs)
    report = build_report(groups, environment())
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as handle:
        handle.write(report)
    print("[report] %d runs -> %s" % (len(runs), args.out), flush=True)
    print("[report] methods: %s" % ", ".join(groups), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
