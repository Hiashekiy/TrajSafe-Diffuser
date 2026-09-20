"""Build outputs/bspline_carla/NIGHT_RUN_REPORT.md from the run artefacts.

    python scripts/night_report.py --out outputs/bspline_carla

Reads (when present):
  data/carla_processed/cleaning_report.json
  data/carla_processed/preprocess_candidates_report.json
  data/carla_processed/preprocess_labels_report.json
  data/carla_processed/preprocess_report.json
  outputs/bspline_carla/overfit/training_summary.json
  outputs/bspline_carla/training_summary.json
  outputs/bspline_carla/train.log  (tail)
and writes the 20-point night report required by the task spec.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:                                   # pragma: no cover
        return {"_error": repr(exc)}


def _git(args):
    try:
        out = subprocess.run(["git"] + args, cwd=ROOT, capture_output=True,
                             text=True, timeout=30)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:                                   # pragma: no cover
        return "git failed: %r" % (exc,)


def _fmt(d, keys, prefix=""):
    lines = []
    for k in keys:
        if isinstance(d, dict) and k in d:
            v = d[k]
            if isinstance(v, float):
                lines.append("- %s%s = %.6g" % (prefix, k, v))
            else:
                lines.append("- %s%s = %s" % (prefix, k, v))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/bspline_carla")
    ap.add_argument("--processed", default="data/carla_processed")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    proc = os.path.abspath(args.processed)
    os.makedirs(out_dir, exist_ok=True)

    clean = _read_json(os.path.join(proc, "cleaning_report.json")) or {}
    cand = _read_json(os.path.join(proc, "preprocess_candidates_report.json")) or {}
    labels = _read_json(os.path.join(proc, "preprocess_labels_report.json")) or {}
    validate = _read_json(os.path.join(proc, "preprocess_report.json")) or {}
    overfit = _read_json(os.path.join(out_dir, "overfit",
                                      "training_summary.json")) or {}
    train = _read_json(os.path.join(out_dir, "training_summary.json")) or {}

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append("# CARLA + 32-control B-spline 夜间运行报告")
    L.append("")
    L.append("生成时间：%s" % now)
    L.append("")

    L.append("## 1. Git 状态")
    L.append("```")
    L.append("branch: %s" % _git(["branch", "--show-current"]))
    L.append("commit: %s" % _git(["log", "-1", "--oneline"]))
    L.append("--- status ---")
    L.append(_git(["status", "--porcelain"]) or "(clean)")
    L.append("```")

    L.append("## 2. 实际读到的数据量")
    L.append("```")
    L.append("scanned_records = %s" % clean.get("scanned_records"))
    L.append("existing_npz    = %s" % clean.get("existing_npz"))
    L.append("valid_samples   = %s" % clean.get("valid_samples"))
    L.append("manifest_entries= %s" % clean.get("manifest_entries"))
    L.append("```")

    L.append("## 3. 清洗掉多少 / 原因")
    L.append("```")
    L.append(json.dumps(clean.get("reason_histogram", {}), ensure_ascii=False))
    L.append("invalid_count = %s" % clean.get("invalid_count"))
    L.append("missing_npz   = %s" % clean.get("missing_npz"))
    L.append("```")

    L.append("## 4. train / val / test 数量与 episode")
    L.append("```")
    L.append("per_split_counts        = %s" % json.dumps(
        clean.get("per_split_counts", {})))
    L.append("unique_episodes_per_split = %s" % json.dumps(
        clean.get("unique_episodes_per_split", {})))
    L.append("split_leakage           = %s" % json.dumps(
        clean.get("split_leakage", [])))
    L.append("```")

    L.append("## 5. preprocessing 成功率")
    L.append("```")
    L.append(json.dumps(cand, ensure_ascii=False)[:2000])
    L.append("```")

    L.append("## 6. candidate empty rate")
    L.append("```")
    L.append("empty_candidate_count = %s" % cand.get("empty_candidate_count"))
    L.append("empty_rate            = %s" % cand.get("empty_rate"))
    L.append("per_split             = %s" % json.dumps(
        cand.get("per_split", {}))[:1500])
    L.append("```")

    L.append("## 7. ellipse label valid rate")
    L.append("```")
    L.append(json.dumps(labels, ensure_ascii=False)[:2000])
    L.append("```")

    L.append("## 8. B-spline endpoint-constrained fit")
    L.append("```")
    L.append("fit_rmse_meter       = %s" % json.dumps(
        clean.get("fit_rmse_meter", {})))
    L.append("fit_max_error_meter  = %s" % json.dumps(
        clean.get("fit_max_error_meter", {})))
    L.append("goal_vs_executed_end = %s" % json.dumps(
        clean.get("goal_to_executed_endpoint_meter", {})))
    L.append("```")

    L.append("## 9. 32-sample overfit")
    if overfit:
        L.append("```")
        L.append("epochs_done   = %s" % overfit.get("epochs_done"))
        L.append("best_val_total= %s" % overfit.get("best_val_total"))
        hist = overfit.get("history", [])
        if hist:
            L.append("first train   = %s" % json.dumps(hist[0].get("train", {})))
            L.append("last  train   = %s" % json.dumps(hist[-1].get("train", {})))
        L.append("```")
    else:
        L.append("(not found: %s)" % os.path.join(out_dir, "overfit",
                                                "training_summary.json"))

    L.append("## 10-15. 正式训练")
    L.append("```")
    L.append("epochs_done    = %s" % train.get("epochs_done"))
    L.append("train_samples  = %s" % train.get("train_samples"))
    L.append("val_samples    = %s" % train.get("val_samples"))
    L.append("best_epoch     = %s" % train.get("best_epoch"))
    L.append("best_val_total = %s" % train.get("best_val_total"))
    L.append("latest_ckpt    = %s" % train.get("latest_ckpt"))
    L.append("best_ckpt      = %s" % train.get("best_ckpt"))
    L.append("best_task_ckpt = %s (score %s @ epoch %s)"
             % (train.get("best_task_ckpt"), train.get("best_task_score"),
                train.get("best_task_epoch")))
    L.append("elapsed_seconds= %s" % train.get("elapsed_seconds"))
    hist = train.get("history", [])
    if hist:
        L.append("first train    = %s" % json.dumps(hist[0].get("train", {})))
        L.append("last  train    = %s" % json.dumps(hist[-1].get("train", {})))
        if "val" in hist[-1]:
            L.append("last  val      = %s" % json.dumps(hist[-1]["val"]))
    L.append("```")

    L.append("## 16. 验证器结论")
    L.append("```")
    L.append(json.dumps(validate, ensure_ascii=False)[:3000])
    L.append("```")

    L.append("## 16b. test split evaluation (evaluate.py)")
    L.append("```")
    for name in ("eval_test.json", "eval_test_latest.json"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            d = _read_json(p) or {}
            L.append("%s: %s" % (name, json.dumps(
                {k: d.get(k) for k in
                 ("curve_rmse_m", "ctrl_rmse_m", "traj_collision",
                  "ellipse_collision", "sel_best_rate", "pred_topo_best_rate",
                  "recall", "goal_dist_m")})))
    L.append("```")

    L.append("## 17. 预览图")
    for name in ("overfit_preview.png", "sample_preview.png"):
        p = os.path.join(out_dir, name)
        L.append("- %s : %s" % (p, "exists" if os.path.exists(p) else "MISSING"))

    L.append("## 18. 修改的文件")
    L.append("```")
    L.append(_git(["status", "--porcelain"]))
    L.append("```")

    L.append("## 19. 仍未做的事项")
    L.append("```")
    L.append("- 128 convex region 整体 safety corridor / convex-hull ALM / Bezier extraction")
    L.append("- control-space ALM（本轮明确未迁移）")
    L.append("- CARLA 在线闭环部署 / NPC 动态障碍 / 车辆动力学约束")
    L.append("- dashboard 的旧 waypoint 语义未同步改造（train/sample/evaluate CLI 优先）")
    L.append("```")

    L.append("## 20. ALM")
    L.append("```")
    L.append("ALM 本轮未迁移，保持关闭（config: alm.enabled=false；"
             "sampler 遇到 alm_guidance 会直接报错）。")
    L.append("```")

    L.append("## 21. 附加说明 / 偏差记录 (agent)")
    L.append("```")
    L.append("1. ellipse 标签采样参数: 为把单样本 11 s 降到 2.4 s, boundary 128->48,")
    L.append("   interior 4x32->2x12, binary_iters 12->8 (判定准则不变). 抽样对比:")
    L.append("   valid 数量一致, a/b 中位相对差 0.1%~1.4%.")
    L.append("2. best.pt 判据是 val total, 而 val total 被 topology CE 主导 (该 head 早期过拟合),")
    L.append("   因此额外保存 best_task.pt = val(curve_rmse_m + 40*collision_rate) 最优.")
    L.append("3. 严格碰撞判据 (128 点中任一落在障碍栅格): GT 基线 1.8%, 模型 DDIM 采样更高,")
    L.append("   逐点碰撞比例 GT 0.018% vs 模型约 0.7%~3%. 可用推理期 safety 引导 / best-of-N 改善.")
    L.append("4. occupancy 已统一 flipud; raw data/carla_v1 未被修改 (采集进程可继续).")
    L.append("5. ALM 未迁移: config alm.enabled=false, sampler 收到 alm_guidance 会直接报错.")
    L.append("6. run1 到 epoch 272 按 deadline 停止; 之后从 latest.pt 续训,")
    L.append("   ckpt/best_run1.pt 与 training_summary_run1.json 保留第一段最好结果.")
    L.append("```")

    log_path = os.path.join(out_dir, "train.log")
    if os.path.exists(log_path):
        L.append("## 附：train.log 末尾")
        L.append("```")
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-25:]
        L.append("".join(tail))
        L.append("```")

    path = os.path.join(out_dir, "NIGHT_RUN_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
