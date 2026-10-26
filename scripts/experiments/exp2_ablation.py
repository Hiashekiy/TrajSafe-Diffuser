#!/usr/bin/env python
"""实验 2 第一部分：A0–A3 关键模块递进消融（实验/docs/实验2_*.md §三）。

Configurations are fixed by the guide and must not change:

    A0 Baseline      D  | no safety condition | no feedback | no ALM
    A1 + Safety      S  | safety condition    | no feedback | no ALM inference
    A2 + Feedback    A  | safety condition    | feedback    | ALM result NOT in
                                                            DDIM update / output
    A3 Full Model    A  | safety condition    | feedback    | ALM result in
                                                            DDIM update / output

A2 and A3 load the SAME checkpoint file (``A/ckpt/best_task.pt``); the only
difference is ``alm.safe_prediction_to_ddim``.  A2 still RUNS the ALM (its output
becomes the history feedback) -- it is not "ALM off".

Metrics come from ``metrics.enrich`` (shared with 实验 1/3): collision rate,
mean minimum clearance, detour ratio, NRMSE.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import metrics as M         # noqa: E402

OUT = os.path.join(C.OUT_ROOT, "exp2")
RAW = os.path.join(OUT, "raw_ablation")
EXP1_RAW = os.path.join(C.OUT_ROOT, "exp1", "raw")

ORDER = ["A0", "A1", "A2", "A3"]
CONFIGS = {
    "A0": dict(label="A0: Baseline", safety=False, feedback=False, alm_update=False, alm_feedback=False,
               config=C.CONFIG_D, ckpt=C.CKPT_D, guidance="none",
               reuse_exp1="none",
               alm=dict(force_feedback_valid_zero=True,
                        safe_prediction_to_ddim=True, final_project=False)),
    "A1": dict(label="A1: + Safety", safety=True, feedback=False, alm_update=False, alm_feedback=False,
               config=C.CONFIG_S, ckpt=C.CKPT_S, guidance="none",
               reuse_exp1=None,
               alm=dict(force_feedback_valid_zero=True,
                        safe_prediction_to_ddim=True, final_project=False)),
    "A2": dict(label="A2: + Feedback", safety=True, feedback=True, alm_update=False,
               alm_feedback=True,
               config=C.CONFIG_A, ckpt=C.CKPT_A, guidance="alm",
               reuse_exp1=None,
               alm=dict(force_feedback_valid_zero=False,
                        safe_prediction_to_ddim=False, final_project=False)),
    "A3": dict(label="A3: Full Model", safety=True, feedback=True, alm_update=True,
               alm_feedback=True,
               config=C.CONFIG_A, ckpt=C.CKPT_A, guidance="alm",
               reuse_exp1="TrajSafe",
               alm=dict(force_feedback_valid_zero=False,
                        safe_prediction_to_ddim=True, final_project=False)),
}


def _reuse_exp1(tag: str, seed: int) -> bool:
    stem = os.path.join(EXP1_RAW, "%s_seed%d.npz" % (tag, seed))
    return os.path.exists(stem)


def _copy_from_exp1(src_tag: str, dst_tag: str, seed: int) -> bool:
    stem_src = os.path.join(EXP1_RAW, "%s_seed%d" % (src_tag, seed))
    stem_dst = os.path.join(RAW, "%s_seed%d" % (dst_tag, seed))
    if not os.path.exists(stem_src + ".npz"):
        return False
    if os.path.exists(stem_dst + ".npz"):
        return True
    C.ensure_dir(RAW)
    rows = C.load_csv(stem_src + ".csv")
    for row in rows:
        row["method"] = dst_tag
    C.save_csv(stem_dst + ".csv", rows)
    with np.load(stem_src + ".npz") as data:
        C.save_arrays(stem_dst + ".npz", control=data["control"],
                      reference=data["reference"], endpoints=data["endpoints"])
    meta = json.load(open(stem_src + ".json", encoding="utf-8"))
    meta.setdefault("header", {})["reused_from"] = stem_src + ".npz"
    C.save_json(stem_dst + ".json", meta)
    return True


def _run(args) -> None:
    seeds = [int(s) for s in args.seeds]
    for tag in args.configs:
        spec = CONFIGS[tag]
        for seed in seeds:
            stem = os.path.join(RAW, "%s_seed%d" % (tag, seed))
            if os.path.exists(stem + ".npz") and not args.force:
                print("[exp2] skip %s_seed%d (exists)" % (tag, seed), flush=True)
                continue
            if (spec.get("reuse_exp1") and not args.force
                    and _copy_from_exp1(spec["reuse_exp1"], tag, seed)):
                print("[exp2] %s_seed%d reused from exp1/%s (identical model, "
                      "flags and noise)" % (tag, seed, spec["reuse_exp1"]),
                      flush=True)
                continue
            ctx = C.build_context(spec["config"], spec["ckpt"], split=args.split,
                                  processed=args.processed,
                                  limit=args.limit or None)
            res = C.run_split(ctx, method=tag, guidance=spec["guidance"],
                              alm_overrides=spec["alm"], steps=args.steps,
                              seed=seed, chunk=args.chunk)
            C.save_csv(stem + ".csv", res.rows)
            C.save_arrays(stem + ".npz", control=res.controls,
                          reference=res.reference, endpoints=res.endpoints)
            C.save_json(stem + ".json", dict(header=res.header,
                                             timing=res.timing))
            del ctx


def _load(tag: str):
    runs = []
    for path in sorted(glob.glob(os.path.join(RAW, "%s_seed*.npz" % tag))):
        stem = path[:-4]
        rows = C.load_csv(stem + ".csv")
        for row in rows:
            row["collision"] = str(row["collision"]).lower() == "true"
            row["sdf_min_distance_m"] = float(row["sdf_min_distance_m"])
            row["rmse_m"] = float(row["rmse_m"])
            row["seed"] = int(row["seed"])
            row["sample_id"] = int(row["sample_id"])
        data = np.load(path)
        runs.append(dict(rows=rows, control=data["control"],
                         reference=data["reference"],
                         endpoints=data["endpoints"],
                         meta=json.load(open(stem + ".json", encoding="utf-8"))))
    return runs


def _metrics(args) -> None:
    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                          processed=args.processed, limit=1)
    codec, sm = ctx.codec, ctx.scene_to_meter
    per_task, table, provenance = [], [], {}
    for tag in ORDER:
        runs = _load(tag)
        if not runs:
            print("[exp2] WARNING no run for %s" % tag, flush=True)
            continue
        pooled = []
        for run in runs:
            M.enrich(run["rows"], run["control"], run["reference"],
                     run["endpoints"], codec, sm,
                     step_m=args.geometry_step)
            pooled.extend(run["rows"])
        n = len(pooled)
        coll = sum(1 for r in pooled if r["collision"])
        stats = M.summarise(pooled, ("clearance_m", "detour_ratio", "nrmse_pct"))
        cf = CONFIGS[tag]
        table.append(dict(
            config=cf["label"], safety_cond=cf["safety"],
            history_feedback=cf["feedback"], alm_safety_update=cf["alm_update"],
            alm_builds_feedback=cf["alm_feedback"],
            collision_pct=100.0 * coll / n, collisions=coll, n=n,
            clearance_m=stats["clearance_m"]["mean"],
            clearance_valid_n=stats["clearance_m"]["valid_n"],
            detour_ratio=stats["detour_ratio"]["mean"],
            nrmse_pct=stats["nrmse_pct"]["mean"]))
        per_task.extend(pooled)
        provenance[tag] = [r["meta"]["header"] for r in runs]
    columns = ["config", "safety_cond", "history_feedback",
               "alm_builds_feedback", "alm_safety_update", "collision_pct",
               "clearance_m", "detour_ratio", "nrmse_pct", "collisions", "n",
               "clearance_valid_n"]
    C.save_csv(os.path.join(OUT, "table_ablation.csv"), table, columns)
    C.save_csv(os.path.join(OUT, "per_task_ablation.csv"), per_task,
               ["method", "seed", "sample_id", "collision", "clearance_m",
                "detour_ratio", "nrmse_pct", "rmse_m", "kappa_max_inv_m",
                "corridor_guided", "activation_step", "alm_status"])
    C.save_json(os.path.join(OUT, "ablation_summary.json"),
                dict(table=table, provenance=provenance,
                     step_m=args.geometry_step))
    _markdown(os.path.join(OUT, "table_ablation.md"), table, columns)
    print("[exp2] ablation table written (%d configs, %d tasks)"
          % (len(table), len(per_task)), flush=True)


def _markdown(path, rows, columns) -> None:
    def fmt(v):
        if isinstance(v, bool):
            return "√" if v else "×"
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return "%.4g" % v
        return str(v)

    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")
    C.ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "metrics"], required=True)
    ap.add_argument("--configs", nargs="*", default=ORDER, choices=ORDER)
    ap.add_argument("--seeds", nargs="*", default=["0", "1", "2"])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--geometry-step", type=float, default=M.GEOMETRY_STEP_M,
                    help="arc-length spacing (m) used to evaluate curvature")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    _run(args) if args.stage == "run" else _metrics(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
