#!/usr/bin/env python
"""验证集超参数选择（指南 4.1.4 第 4/5 条：只能在验证集上选，选定后冻结）.

对每一种外部约束处理器，把候选网格**逐个**交给 ``scripts/eval_chapter4.py``
在 ``--split val`` 的一个固定子集上跑一遍（复用同一个评测实现，因此选参用的
指标与主表指标逐位一致），然后按下面的字典序规则挑唯一胜者：

    1. 碰撞数最少；
    2. 并列时 SDF 平均违规量最小；
    3. 再并列时 RMSE 最小；
    4. 再并列时修正距离（= 计算代价/轨迹偏移）最小。

规则是确定性的、事先写死的，不依赖肉眼看结果挑参数。完整网格会写进 JSON，
所以报告里可以展示「安全—质量—代价」的取舍，而不只是胜者那一行。

用法::

    python scripts/ch4_sweep_val.py --config <cfg> --ckpt <path> \\
        --guidance penalty --val-samples 256
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: the candidate grids of guide 4.1.2 §2/§3/§4/§5 (verbatim)
GRIDS = {
    "penalty": {
        "penalty_weight": [1, 5, 10, 20],
        "step_size": [0.02, 0.05, 0.10],
        "inner_steps": [5, 10],
    },
    "projection": {
        # the guide fixes "a small number of sequential convexification
        # iterations, chosen on the validation set"; each one costs a QP solve
        "inner_steps": [3, 5, 10],
    },
    "primal_dual": {
        "step_size": [0.02, 0.05, 0.10],
        "dual_step": [0.5, 1, 5],
        "inner_steps": [5, 10],
    },
    "alm_sdf": {
        "rho": [1, 5, 10],
        "step_size": [0.02, 0.05, 0.10],
        "inner_steps": [5, 10],
    },
}


def candidates(guidance: str):
    grid = GRIDS[guidance]
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def run_one(args, params: dict, out_json: str) -> dict:
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "eval_chapter4.py"),
           "--config", args.config, "--ckpt", args.ckpt,
           "--method", "%s_sweep" % args.guidance, "--guidance", args.guidance,
           "--split", args.split, "--samples", str(args.val_samples),
           "--chunk", str(args.chunk), "--steps", str(args.steps),
           "--seed", str(args.seed), "--device", args.device,
           "--out", out_json, "--md", out_json[:-5] + ".md",
           "--records", out_json[:-5] + ".jsonl"]
    if args.processed:
        cmd += ["--processed", args.processed]
    for key, value in params.items():
        cmd += ["--sdf-set", "%s=%s" % (key, value)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-6:])
        print("[sweep] FAILED %s\n%s" % (params, tail), flush=True)
        return {"params": params, "failed": True, "seconds": time.time() - started}
    with open(out_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    agg = payload["aggregate"]
    print("[sweep] %-46s coll=%2d feas=%6.2f%% viol=%.4f rmse=%.2f %.0fs"
          % (json.dumps(params), agg["collisions"],
             100.0 * (agg["sdf_feasible_rate"] or 0.0),
             agg["sdf_mean_violation_m"] or 0.0, agg["rmse_m"] or 0.0,
             time.time() - started), flush=True)
    return {"params": params, "failed": False, "seconds": time.time() - started,
            "aggregate": agg}


def rank_key(entry: dict):
    """The frozen lexicographic selection rule (see the module docstring)."""
    if entry.get("failed"):
        return (1, float("inf"), float("inf"), float("inf"), float("inf"))
    agg = entry["aggregate"]
    return (0,
            agg["collisions"],
            agg["sdf_mean_violation_m"] if agg["sdf_mean_violation_m"] is not None else float("inf"),
            agg["rmse_m"] if agg["rmse_m"] is not None else float("inf"),
            agg.get("sdf_point_mean_violation_m")
            if agg.get("sdf_point_mean_violation_m") is not None else float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--guidance", required=True, choices=list(GRIDS))
    ap.add_argument("--split", default="val")
    ap.add_argument("--val-samples", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(ROOT, "outputs", "ch4",
                                        "sweep_%s.json" % args.guidance)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    work_dir = os.path.join(ROOT, "outputs", "ch4", "sweep_%s" % args.guidance)
    os.makedirs(work_dir, exist_ok=True)

    entries = []
    for index, params in enumerate(candidates(args.guidance)):
        out_json = os.path.join(work_dir, "cand_%03d.json" % index)
        entries.append(run_one(args, params, out_json))
    ranked = sorted(entries, key=rank_key)
    winner = ranked[0]
    payload = {
        "guidance": args.guidance,
        "config": args.config,
        "checkpoint": args.ckpt,
        "split": args.split,
        "val_samples": args.val_samples,
        "chunk": args.chunk,
        "steps": args.steps,
        "seed": args.seed,
        "selection_rule": ("min collisions, then min mean SDF violation (m), "
                           "then min RMSE (m), then min point-mean violation"),
        "n_candidates": len(entries),
        "winner": winner["params"],
        "winner_aggregate": winner.get("aggregate"),
        "grid": [{k: v for k, v in entry.items() if k != "aggregate"}
                 | {"collisions": entry.get("aggregate", {}).get("collisions"),
                    "feasible_rate": entry.get("aggregate", {}).get(
                        "sdf_feasible_rate"),
                    "mean_violation_m": entry.get("aggregate", {}).get(
                        "sdf_mean_violation_m"),
                    "rmse_m": entry.get("aggregate", {}).get("rmse_m"),
                    "smoothness_m": entry.get("aggregate", {}).get("smoothness_m"),
                    "runtime_ms": entry.get("aggregate", {}).get("runtime_ms_mean")}
                for entry in ranked],
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    lines = ["# 验证集选参：%s" % args.guidance, "",
             "- checkpoint: `%s`" % args.ckpt,
             "- val 子集: %d 样本 (split=%s, chunk=%d, steps=%d, seed=%d)"
             % (args.val_samples, args.split, args.chunk, args.steps, args.seed),
             "- 选择规则: %s" % payload["selection_rule"],
             "- **冻结参数**: `%s`" % json.dumps(winner["params"]), "",
             "| 排名 | 参数 | 碰撞数 | SDF可行率 | 平均违规(m) | RMSE(m) | 平滑(m) | ms/traj |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rank, entry in enumerate(payload["grid"], start=1):
        lines.append("| %d | `%s` | %s | %s | %s | %s | %s | %s |"
                     % (rank, json.dumps(entry["params"]), entry["collisions"],
                        _fmt(entry["feasible_rate"], 100.0, "%.2f%%"),
                        _fmt(entry["mean_violation_m"], 1.0, "%.4f"),
                        _fmt(entry["rmse_m"], 1.0, "%.2f"),
                        _fmt(entry["smoothness_m"], 1.0, "%.3f"),
                        _fmt(entry["runtime_ms"], 1.0, "%.1f")))
    with open(out_path[:-5] + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("[sweep] winner %s -> %s" % (args.guidance, json.dumps(winner["params"])),
          flush=True)
    print("[sweep] wrote %s" % out_path, flush=True)
    return 0


def _fmt(value, scale, fmt):
    return "-" if value is None else fmt % (value * scale)


if __name__ == "__main__":
    raise SystemExit(main())
