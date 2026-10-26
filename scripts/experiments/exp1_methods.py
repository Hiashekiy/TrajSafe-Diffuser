#!/usr/bin/env python
"""实验 1：不同安全规划方法对比与计算效率（实验/docs/实验1.md）。

Two stages, on purpose:

``--stage run``     run the five methods x 3 seeds and SAVE the final B-spline
                    control polygons (the historical records did not keep them).
``--stage metrics`` recompute the full 6-metric table OFFLINE from those saved
                    polygons -- the same code for every method, no model call.

The existing efficiency measurements are merged in unchanged (实验 1 §5.5): no
new timing run is performed here.

    python scripts/experiments/exp1_methods.py --stage run
    python scripts/experiments/exp1_methods.py --stage metrics
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

OUT = os.path.join(C.OUT_ROOT, "exp1")
RAW = os.path.join(OUT, "raw")

#: method -> (display name, config, checkpoint, guidance, extra alm overrides)
METHODS = {
    "none": dict(name="Unconstrained Diffusion", config=C.CONFIG_D,
                 ckpt=C.CKPT_D, guidance="none"),
    "penalty": dict(name="Fixed Penalty Guidance", config=C.CONFIG_D,
                    ckpt=C.CKPT_D, guidance="penalty", sweep="penalty"),
    "primal_dual": dict(name="Primal-Dual Guidance", config=C.CONFIG_D,
                        ckpt=C.CKPT_D, guidance="primal_dual",
                        sweep="primal_dual"),
    "alm_sdf": dict(name="SDF-ALM Guidance", config=C.CONFIG_D,
                    ckpt=C.CKPT_D, guidance="alm_sdf", sweep="alm_sdf"),
    "TrajSafe": dict(name="TrajSafe-Diffuser", config=C.CONFIG_A,
                     ckpt=C.CKPT_A, guidance="alm",
                     alm=dict(force_feedback_valid_zero=False,
                              safe_prediction_to_ddim=True,
                              final_project=False)),
}
ORDER = ["none", "penalty", "primal_dual", "alm_sdf", "TrajSafe"]


def _run(args) -> None:
    seeds = [int(s) for s in args.seeds]
    for tag in args.methods:
        spec = METHODS[tag]
        if spec.get("sweep"):
            over = C.sweep_overrides(spec["sweep"])
            alm_over = dict(spec.get("alm") or {})
            alm_over.update({"force_feedback_valid_zero": False})
            ctx_kwargs = dict(alm_overrides=alm_over, sdf_overrides=over)
        else:
            ctx_kwargs = dict(alm_overrides=dict(spec.get("alm") or {}),
                              sdf_overrides={})
        for seed in seeds:
            stem = os.path.join(RAW, "%s_seed%d" % (tag, seed))
            if os.path.exists(stem + ".npz") and not args.force:
                print("[exp1] skip %s_seed%d (exists)" % (tag, seed), flush=True)
                continue
            ctx = C.build_context(spec["config"], spec["ckpt"], split=args.split,
                                  processed=args.processed,
                                  limit=args.limit or None)
            res = C.run_split(ctx, method=tag, guidance=spec["guidance"],
                              steps=args.steps, seed=seed, chunk=args.chunk,
                              **ctx_kwargs)
            C.save_csv(stem + ".csv", res.rows)
            C.save_arrays(stem + ".npz", control=res.controls,
                          reference=res.reference, endpoints=res.endpoints)
            C.save_json(stem + ".json", dict(header=res.header,
                                             timing=res.timing))
            del ctx


def _load_raw():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RAW, "*_seed*.npz"))):
        tag = os.path.basename(path).split("_seed")[0]
        stem = path[:-4]
        rows = C.load_csv(stem + ".csv")
        data = np.load(path)
        for row in rows:
            row["collision"] = str(row["collision"]).lower() == "true"
            row["sdf_min_distance_m"] = float(row["sdf_min_distance_m"])
            row["rmse_m"] = float(row["rmse_m"])
            row["seed"] = int(row["seed"])
            row["sample_id"] = int(row["sample_id"])
        runs.setdefault(tag, []).append(dict(
            rows=rows, control=data["control"], reference=data["reference"],
            endpoints=data["endpoints"], meta=json.load(
                open(stem + ".json", encoding="utf-8")),
            seed=int(rows[0]["seed"]) if rows else -1))
    return runs


def _metrics(args) -> None:
    runs = _load_raw()
    ctx = C.build_context(CONFIG_LEARNED, C.CKPT_A, split=args.split,
                          processed=args.processed, limit=1)
    codec, sm = ctx.codec, ctx.scene_to_meter
    per_task, summary, provenance = [], {}, {}
    for tag in ORDER:
        if tag not in runs:
            print("[exp1] WARNING no raw run for %s" % tag, flush=True)
            continue
        pooled = []
        for run in runs[tag]:
            M.enrich(run["rows"], run["control"], run["reference"],
                     run["endpoints"], codec, sm,
                     step_m=args.geometry_step)
            pooled.extend(run["rows"])
        # collision rate over ALL generated trajectories
        n = len(pooled)
        coll = sum(1 for r in pooled if r["collision"])
        stats = M.summarise(pooled, ("clearance_m", "detour_ratio", "nrmse_pct",
                                     "kappa_max_inv_m", "kappa2_mean_inv_m2",
                                     "length_m", "rmse_m"))
        summary[tag] = dict(
            name=METHODS[tag]["name"], n=n, collisions=coll,
            collision_pct=100.0 * coll / n,
            clearance_m=stats["clearance_m"]["mean"],
            clearance_valid_n=stats["clearance_m"]["valid_n"],
            detour_ratio=stats["detour_ratio"]["mean"],
            detour_valid_n=stats["detour_ratio"]["valid_n"],
            nrmse_pct=stats["nrmse_pct"]["mean"],
            kappa_max_inv_m=stats["kappa_max_inv_m"]["mean"],
            kappa2_mean_inv_m2=stats["kappa2_mean_inv_m2"]["mean"],
            length_m=stats["length_m"]["mean"],
            rmse_m=stats["rmse_m"]["mean"],
            degenerate_curves=sum(1 for r in pooled if r["degenerate_curve"]),
            micro_loop_curves=sum(1 for r in pooled if r["micro_loop_curve"]),
            curvature_step_m=METHODS and args.geometry_step,
            seeds=sorted({r["seed"] for r in pooled}),
            per_seed={s: dict(
                n=len([r for r in pooled if r["seed"] == s]),
                collisions=sum(1 for r in pooled
                               if r["seed"] == s and r["collision"]))
                for s in sorted({r["seed"] for r in pooled})})
        per_task.extend(pooled)
        provenance[tag] = [r["meta"]["header"] for r in runs[tag]]
    columns = ["method", "seed", "sample_id", "collision", "clearance_m",
               "sdf_min_distance_m", "detour_ratio", "nrmse_pct", "rmse_m",
               "ref_length_m", "length_m", "kappa_max_inv_m",
               "kappa2_mean_inv_m2", "degenerate_curve", "micro_loop_curve",
               "corridor_guided", "activation_step", "alm_status"]
    C.save_csv(os.path.join(OUT, "per_task_metrics.csv"), per_task, columns)
    C.save_json(os.path.join(OUT, "summary.json"),
                dict(summary=summary, provenance=provenance,
                     step_m=args.geometry_step,
                     clearance_rule="collision-free trajectories only",
                     nrmse_rule="RMSE / reference polyline length"))
    _write_table1(summary)
    _write_efficiency()
    print("[exp1] metrics done for %d methods, %d tasks"
          % (len(summary), len(per_task)), flush=True)


def _write_table1(summary: dict) -> None:
    header = ["method", "collision_pct", "clearance_m", "detour_ratio",
              "nrmse_pct", "kappa_max_inv_m", "kappa2_mean_inv_m2", "n",
              "clearance_valid_n"]
    rows = []
    for tag in ORDER:
        if tag not in summary:
            continue
        s = summary[tag]
        rows.append(dict(method=s["name"], collision_pct=s["collision_pct"],
                         clearance_m=s["clearance_m"],
                         detour_ratio=s["detour_ratio"], nrmse_pct=s["nrmse_pct"],
                         kappa_max_inv_m=s["kappa_max_inv_m"],
                         kappa2_mean_inv_m2=s["kappa2_mean_inv_m2"],
                         n=s["n"], clearance_valid_n=s["clearance_valid_n"]))
    C.save_csv(os.path.join(OUT, "table1_methods.csv"), rows, header)
    _write_markdown(os.path.join(OUT, "table1_methods.md"),
                    "表 1 不同安全规划方法的安全性与轨迹质量", rows, header)


def _write_efficiency() -> None:
    """Merge the EXISTING efficiency measurements (no new timing runs)."""
    src = os.path.join(C.CAMPAIGN, "efficiency_corrected_20261010",
                       "timing.json")
    if not os.path.exists(src):
        print("[exp1] no corrected timing.json -- efficiency table skipped",
              flush=True)
        return
    with open(src, encoding="utf-8") as handle:
        timing = json.load(handle)
    acc = timing.get("acceleration", {})
    label = {"steps16_stride1": ("Standard configuration", 16, 128),
             "steps4_stride1": ("Fewer sampling steps only", 4, 128),
             "steps16_stride8": ("Shared convex regions only", 16, 17),
             "steps4_stride8": ("Combined acceleration", 4, 17)}
    rows = []
    for key, (name, steps, _) in label.items():
        cfgv = acc.get(key)
        if not cfgv:
            continue
        rows.append(dict(
            configuration=name, ddim_steps=steps,
            base_regions=int(round(cfgv.get("base_cells", 0))),
            time_ms=cfgv.get("wall_ms"), collisions=cfgv.get("collisions"),
            n=cfgv.get("n"), feasible_rate=cfgv.get("feasible_rate")))
    header = ["configuration", "ddim_steps", "base_regions", "time_ms",
              "collisions", "n", "feasible_rate"]
    C.save_csv(os.path.join(OUT, "table2_efficiency.csv"), rows, header)
    _write_markdown(os.path.join(OUT, "table2_efficiency.md"),
                    "表 2 完整方法的采样步数与凸区域共享效率（沿用既有测时）",
                    rows, header)
    std = next((r for r in rows if r["configuration"].startswith("Standard")),
               None)
    comb = next((r for r in rows if r["configuration"].startswith("Combined")),
                None)
    if std and comb and comb["time_ms"]:
        C.save_json(os.path.join(OUT, "efficiency_summary.json"), dict(
            source=src, protocol=timing.get("protocol"),
            speedup=float(std["time_ms"]) / float(comb["time_ms"]),
            note="reused historical interleaved timing; no new timing run",
            method_level_times_present=bool(timing.get("methods")),
            method_level_times_used=False))


def _write_markdown(path: str, title: str, rows, columns) -> None:
    def fmt(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return "%.4g" % value
        return str(value)

    lines = ["# %s" % title, "", "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")
    C.ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


#: the learned (full) config, only used to obtain the B-spline codec offline
CONFIG_LEARNED = C.CONFIG_A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "metrics"], required=True)
    ap.add_argument("--methods", nargs="*", default=ORDER, choices=ORDER)
    ap.add_argument("--seeds", nargs="*", default=["0", "1", "2"])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N tasks (functional verification)")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--geometry-step", type=float, default=M.GEOMETRY_STEP_M,
                    help="arc-length spacing (m) used to evaluate curvature")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage == "run":
        _run(args)
    else:
        _metrics(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
