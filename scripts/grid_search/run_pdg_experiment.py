"""PDG-Focal A* experiment driver on the real CARLA test split.

Every planning case is (test sample, K), where K is the number of executed
reverse-diffusion forwards whose intermediate clean state ``x0`` becomes the
soft reference of the search.  All methods share:

    * the same occupancy bitmap and the same inflated safe grid,
    * the same 8-neighbour graph, no-corner-cutting rule and real edge costs,
    * the same start / goal and the same FOCAL weight ``w``,
    * the same independent path re-validation.

    python scripts/grid_search/run_pdg_experiment.py \
        --cache outputs/grid_search/diffusion_x0/test/diffusion_x0_test_0_420.npz \
        --processed data/processed/160m-k4p-c48 --split test \
        --out outputs/grid_search/run_pdg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
    __file__))))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset
from src.grid_search.benchmark import (Case, aggregate, evaluate_case,
                                       prepare_case, write_csv)
from src.grid_search.grid_map import GridSpec
from src.grid_search.visualization import render_case_figures

MAIN_METHODS = (
    ("dijkstra", "dijkstra", 1.0, 1.0, 0.0),
    ("astar", "astar", 1.0, 1.0, 0.0),
    ("focal_clearance", "focal_clearance", 1.2, 1.0, 0.0),
    ("focal_index", "focal_index", 1.2, 1.0, 0.0),
    ("pdg", "pdg", 1.2, 1.0, 1.0),
)
ABLATION_METHODS = (
    ("G0_no_guidance", "focal_clearance", 1.2, 1.0, 0.0),
    ("G1_distance_only", "pdg", 1.2, 1.0, 0.0),
    ("G2_distance_progress", "pdg", 1.2, 1.0, 1.0),
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--processed", default="data/processed/160m-k4p-c48")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene-to-meter", type=float, default=80.0)
    ap.add_argument("--robot-radius", type=float, default=0.8)
    ap.add_argument("--safety-margin", type=float, default=0.3)
    ap.add_argument("--already-inflated-cells", type=float, default=4.0)
    ap.add_argument("--limit-main", type=int, default=200)
    ap.add_argument("--limit-sweep", type=int, default=80)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--total-forwards", type=int, default=16)
    ap.add_argument("--w", type=float, default=1.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-sweeps", action="store_true")
    ap.add_argument("--figures", type=int, default=3)
    return ap.parse_args()


def load_cases(args):
    data = np.load(args.cache, allow_pickle=False)
    curves = data["curves"]            # [S,K,128,2]
    steps = list(data["steps"])
    cond = data["cond"]
    batch_ms = data["batch_ms"]
    idxs = data["indices"]
    times = data["times"] if "times" in data else None
    return curves, steps, cond, batch_ms, idxs, times


def make_case(args, spec, occupancy, cond_row, curve, k_index, k_value,
              sample_index, extra_ms):
    meta = {"sample": int(sample_index), "K": int(k_value),
            "k_index": int(k_index),
            "warmup_ms_est": float(extra_ms * k_value)}
    return Case(case_id="s%03d_K%d" % (sample_index, k_value), occupancy=occupancy,
                spec=spec, start_world=np.asarray(cond_row[0], np.float64),
                goal_world=np.asarray(cond_row[1], np.float64),
                ref_world=np.asarray(curve, np.float64), meta=meta)


def run_suite(name, cases, configs, args, out, save=True):
    records = []
    t0 = time.perf_counter()
    for i, case in enumerate(cases):
        prep = prepare_case(case, robot_radius_m=args.robot_radius,
                            safety_margin_m=args.safety_margin,
                            already_inflated_cells=args.already_inflated_cells)
        for label, method, w, ad, ap in configs:
            rec = evaluate_case(prep, label, method, w, ad, ap)
            e2e = (case.meta.get("warmup_ms_est", 0.0)
                   + prep.timings["preprocess_ms"]
                   + prep.timings["reference_ms"]
                   + prep.timings["guidance_ms"]
                   + rec["search_time_ms"] + rec["validation_time_ms"])
            rec["end_to_end_ms"] = e2e
            records.append(rec)
        if (i + 1) % 25 == 0:
            print("[%s] %d/%d cases, %.1f s"
                  % (name, i + 1, len(cases), time.perf_counter() - t0),
                  flush=True)
    agg = aggregate(records, ("label",))
    if save:
        write_csv(records, os.path.join(out, "records_%s.csv" % name))
        write_csv(agg, os.path.join(out, "aggregate_%s.csv" % name))
    return records, agg


def summarize(records):
    def _q(key, rs, fn=np.median):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return float(fn(vals)) if vals else None

    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    out = {}
    for label, rs in by_label.items():
        succ = [r for r in rs if r["success"]]
        out[label] = {
            "n": len(rs),
            "success_rate": float(np.mean([r["success"] for r in rs])),
            "path_valid_rate": float(np.mean([r["path_valid"] for r in rs])),
            "expanded_median": _q("expanded_nodes", rs, np.median),
            "expanded_mean": _q("expanded_nodes", rs, np.mean),
            "expanded_p95": _q("expanded_nodes", rs,
                               lambda v: np.percentile(v, 95)),
            "search_ms_median": _q("search_time_ms", rs, np.median),
            "search_ms_p95": _q("search_time_ms", rs, lambda v: np.percentile(v, 95)),
            "cost_median": _q("path_cost_m", rs, np.median),
            "cost_mean": _q("path_cost_m", rs, np.mean),
            "reopened_mean": _q("reopened_nodes", rs, np.mean),
            "peak_open_median": _q("peak_open_size", rs, np.median),
            "e2e_ms_median": _q("end_to_end_ms", rs, np.median),
            "n_with_colliding_reference": int(sum(
                1 for r in succ if (r.get("reference_collision_rate") or 0) > 0)),
        }
    ref = {}
    vals = [r for r in records]
    if vals:
        ref = {
            "reference_valid_rate": float(np.mean([r["reference_valid"]
                                                   for r in vals])),
            "reference_collision_rate_mean": float(np.mean([
                r["reference_collision_rate"] for r in vals
                if r["reference_collision_rate"] is not None])),
            "reference_collision_fraction_any": float(np.mean([
                1.0 if (r["reference_collision_rate"] or 0) > 0 else 0.0
                for r in vals])),
            "reference_length_median_m": float(np.median([
                r["reference_length_m"] for r in vals])),
            "guidance_coverage_mean": float(np.mean([
                r["guidance_coverage"] for r in vals])),
            "guidance_cover_frac_mean": float(np.mean([
                r.get("guidance_cover_frac", 0.0) for r in vals])),
            "guidance_ms_median": float(np.median([
                r["guidance_build_time_ms"] for r in vals])),
            "guidance_segments_median": float(np.median([
                r["guidance_segments"] for r in vals])),
            "preprocess_ms_median": float(np.median([
                r["preprocess_time_ms"] for r in vals])),
            "warmup_ms_median": float(np.median([
                r.get("warmup_ms_est", np.nan) for r in vals])),
        }
    return {"labels": out, "reference": ref}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    curves, steps, cond, batch_ms, idxs, times = load_cases(args)
    ds = CarlaSplineDataset(args.split, args.processed, require_labels=False)
    occ_all = np.asarray(ds.occupancy)
    res = occ_all.shape[-1]
    cell_m = 2.0 * args.scene_to_meter / res
    spec = GridSpec(res, res, cell_m, args.scene_to_meter)
    k_index = steps.index(args.K) if args.K in steps else len(steps) - 1
    print("[cfg] res=%d cell=%.3f m  K=%d (t=%s)  samples=%d"
          % (res, cell_m, args.K,
             int(times[k_index]) if times is not None else -1, len(idxs)),
          flush=True)

    summary = {"args": vars(args),
               "cell_m": cell_m,
               "inflation": {"robot_radius_m": args.robot_radius,
                             "safety_margin_m": args.safety_margin,
                             "already_inflated_cells":
                                 args.already_inflated_cells},
               "suites": {}}

    main_cases = []
    for pos in range(min(args.limit_main, len(idxs))):
        main_cases.append(make_case(args, spec, occ_all[pos], cond[pos],
                                    curves[pos, k_index], k_index, args.K,
                                    int(idxs[pos]), float(batch_ms[pos]) / float(args.total_forwards)))
    rec_main, agg_main = run_suite("main", main_cases, MAIN_METHODS, args,
                                   args.out)
    summary["suites"]["main"] = summarize(rec_main)

    rec_abl, agg_abl = run_suite("ablation", main_cases, ABLATION_METHODS, args,
                                 args.out)
    summary["suites"]["ablation"] = summarize(rec_abl)

    # --- synthetic acceptance scenes T1-T12 (crafted references) ----------
    from src.grid_search.scenes import scene_catalog
    syn_records = []
    for scene in scene_catalog():
        if scene.ref_world is None:
            continue
        case = Case(scene.ident, scene.occupancy, scene.spec,
                    scene.start_world, scene.goal_world, scene.ref_world,
                    meta={"scene": scene.name})
        prep = prepare_case(case, robot_radius_m=0.0, safety_margin_m=0.0,
                            already_inflated_cells=0.0)
        for label, method, w, ad, ap in (
                ("dijkstra", "dijkstra", 1.0, 1.0, 0.0),
                ("astar", "astar", 1.0, 1.0, 0.0),
                ("G0_no_guidance", "focal_clearance", 1.2, 1.0, 0.0),
                ("G1_distance_only", "pdg", 1.2, 1.0, 0.0),
                ("G2_distance_progress", "pdg", 1.2, 1.0, 1.0)):
            rec = evaluate_case(prep, label, method, w, ad, ap)
            syn_records.append(rec)
    write_csv(syn_records, os.path.join(args.out, "records_synthetic.csv"))
    write_csv(aggregate(syn_records, ("label",)),
              os.path.join(args.out, "aggregate_synthetic.csv"))
    summary["suites"]["synthetic"] = summarize(syn_records)

    if not args.skip_sweeps:
        sweep_positions = list(range(min(args.limit_sweep, len(idxs))))

        # --- K sweep: which intermediate x0 is worth using? ----------------
        k_cases = []
        for pos in sweep_positions:
            for j, k in enumerate(steps):
                k_cases.append((k, make_case(args, spec, occ_all[pos], cond[pos],
                                             curves[pos, j], j, k, int(idxs[pos]),
                                             float(batch_ms[pos]) / float(args.total_forwards))))
        records = []
        for k in steps:
            subset = [c for kk, c in k_cases if kk == k]
            rec, _ = run_suite("k%d" % k, subset, (
                ("pdg_K%d" % k, "pdg", args.w, 1.0, 1.0),
                ("pdg_dist_only_K%d" % k, "pdg", args.w, 1.0, 0.0),
                ("astar_ref_K%d" % k, "astar", 1.0, 1.0, 0.0),
            ), args, args.out, save=False)
            records.extend(rec)
        write_csv(records, os.path.join(args.out, "records_k_sweep.csv"))
        agg_k = aggregate(records, ("label",))
        write_csv(agg_k, os.path.join(args.out, "aggregate_k_sweep.csv"))
        summary["suites"]["k_sweep"] = summarize(records)

        # --- w sweep ------------------------------------------------------
        w_records = []
        for w in (1.0, 1.1, 1.2, 1.3, 1.5):
            cases_w = [make_case(args, spec, occ_all[p], cond[p],
                                 curves[p, k_index], k_index, args.K,
                                 int(idxs[p]), float(batch_ms[p]) / float(args.total_forwards))
                       for p in sweep_positions]
            rec, _ = run_suite("w%.1f" % w, cases_w, (
                ("pdg_w%.1f" % w, "pdg", w, 1.0, 1.0),
                ("focal_w%.1f" % w, "focal_clearance", w, 1.0, 0.0),
            ), args, args.out, save=False)
            w_records.extend(rec)
        write_csv(w_records, os.path.join(args.out, "records_w_sweep.csv"))
        agg_w = aggregate(w_records, ("label",))
        write_csv(agg_w, os.path.join(args.out, "aggregate_w_sweep.csv"))
        summary["suites"]["w_sweep"] = summarize(w_records)

        # --- alpha_p sweep ------------------------------------------------
        a_records = []
        for ap in (0.0, 0.5, 1.0, 2.0):
            cases_a = [make_case(args, spec, occ_all[p], cond[p],
                                 curves[p, k_index], k_index, args.K,
                                 int(idxs[p]), float(batch_ms[p]) / float(args.total_forwards))
                       for p in sweep_positions]
            rec, _ = run_suite("ap%.1f" % ap, cases_a, (
                ("pdg_ap%.1f" % ap, "pdg", args.w, 1.0, ap),
            ), args, args.out, save=False)
            a_records.extend(rec)
        write_csv(a_records, os.path.join(args.out, "records_alpha_sweep.csv"))
        agg_a = aggregate(a_records, ("label",))
        write_csv(agg_a, os.path.join(args.out, "aggregate_alpha_sweep.csv"))
        summary["suites"]["alpha_sweep"] = summarize(a_records)

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.figures:
        render_representatives(args, spec, occ_all, cond, curves, batch_ms,
                               idxs, k_index, rec_main)
    print("[done] ->", args.out, flush=True)


def data_t(steps, k_index):
    return {1: 15, 2: 14, 3: 13, 4: 12, 5: 11}.get(steps[k_index], "?")


def render_representatives(args, spec, occ_all, cond, curves, batch_ms, idxs,
                           k_index, rec_main):
    """Pick representative cases from the real main-suite data and plot them."""
    by_case = {}
    for r in rec_main:
        by_case.setdefault(r["case_id"], {})[r["label"]] = r
    scored = []
    for cid, labels in by_case.items():
        a = labels.get("astar")
        p = labels.get("pdg")
        if not a or not p or not a["success"] or not p["success"]:
            continue
        ratio = (a["expanded_nodes"] / max(p["expanded_nodes"], 1))
        scored.append((ratio, cid))
    scored.sort()
    if not scored:
        print("[fig] no successful case to plot", flush=True)
        return
    picks = [("best_reduction", scored[0][1]),
             ("median_reduction", scored[len(scored) // 2][1]),
             ("worst_increase", scored[-1][1])]

    figdir = os.path.join(args.out, "figures")
    for tag, cid in picks:
        sample = int(__import__("re").match(r"s(\d+)_K", cid).group(1))
        pos = int(np.where(idxs == sample)[0][0])
        case = make_case(args, spec, occ_all[pos], cond[pos],
                         curves[pos, k_index], k_index, args.K, sample,
                         float(batch_ms[pos]) / float(args.total_forwards))
        prep = prepare_case(case, robot_radius_m=args.robot_radius,
                            safety_margin_m=args.safety_margin,
                            already_inflated_cells=args.already_inflated_cells)
        for label, method, w, ad, ap in (("astar", "astar", 1.0, 1.0, 0.0),
                                         ("pdg", "pdg", 1.2, 1.0, 1.0),
                                         ("focal_clearance", "focal_clearance",
                                          1.2, 1.0, 0.0)):
            rec = evaluate_case(prep, label, method, w, ad, ap)
            render_case_figures(prep, rec["_result"], figdir,
                                prefix="%s_%s" % (tag, label))
        print("[fig] %s -> %s" % (tag, figdir), flush=True)


if __name__ == "__main__":
    main()




