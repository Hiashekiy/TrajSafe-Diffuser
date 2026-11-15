"""Fused guidance cost: put the diffusion term INTO the A* cost, with a weight.

    g(v) = sum(real moves) + lambda * sum_j-min[ d_j + alpha_p*L_ref*|u_j - u_g| ]
    f(v) = g(v) + octile(v)          (h0 stays admissible)

lambda = 0 -> plain A*.  Larger lambda trades real path length for staying on the
diffusion reference.  Results are reported as REAL path length (and its ratio to
the A* optimum) plus the penalty actually paid, so the trade-off is explicit.

    python scripts/grid_search/run_fused_guidance.py \
        --cache outputs/grid_search/diffusion_x0/test/diffusion_x0_test_0_420.npz \
        --out outputs/grid_search/run_fused
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
    __file__))))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset
from src.grid_search.benchmark import (Case, aggregate, evaluate_case,
                                       prepare_case, write_csv)
from src.grid_search.grid_map import GridSpec

LAMBDAS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def PY():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def real_cases(args):
    data = np.load(args.cache, allow_pickle=False)
    curves, steps, cond, idxs = (data["curves"], list(data["steps"]),
                                 data["cond"], data["indices"])
    ds = CarlaSplineDataset(args.split, args.processed, require_labels=False)
    occ_all = np.asarray(ds.occupancy)
    res = occ_all.shape[-1]
    spec = GridSpec(res, res, 2.0 * args.scene_to_meter / res,
                    args.scene_to_meter)
    ki = steps.index(args.K)
    out = []
    for pos in range(min(args.limit, len(idxs))):
        out.append(Case("s%03d_K%d" % (int(idxs[pos]), args.K), occ_all[pos],
                        spec, cond[pos, 0], cond[pos, 1], curves[pos, ki],
                        meta={"sample": int(idxs[pos]), "K": args.K,
                              "suite": "carla"}))
    return out


def hard_cases(args):
    from src.grid_search.hard_scenes import REFERENCE_KINDS, hard_scenes, reference_for
    out = []
    for scene in hard_scenes():
        for kind in REFERENCE_KINDS:
            out.append(Case("%s_%s" % (scene.ident, kind), scene.occupancy,
                            scene.spec, scene.start_world, scene.goal_world,
                            reference_for(scene, kind),
                            meta={"ident": scene.ident, "scene": scene.name,
                                  "reference": kind, "suite": "hard"}))
    return out


def run(cases, args, label_suite):
    rows = []
    for i, case in enumerate(cases):
        prep = prepare_case(case, robot_radius_m=args.robot_radius,
                            safety_margin_m=args.safety_margin,
                            already_inflated_cells=args.already_inflated_cells)
        base = evaluate_case(prep, "astar", "astar", 1.0, 1.0, 0.0)
        base_cost = base["path_cost_m"] or np.nan
        rows.append({**{k: v for k, v in base.items() if not k.startswith("_")},
                     "lambda": 0.0, "suite": label_suite,
                     "cost_ratio": 1.0})
        # FOCAL tie-break reference (the current Ours)
        focal = evaluate_case(prep, "pdg_focal", "pdg", 1.2, 1.0, 1.0)
        rows.append({**{k: v for k, v in focal.items() if not k.startswith("_")},
                     "suite": label_suite,
                     "cost_ratio": (focal["path_cost_m"] or np.nan) / base_cost})
        for lam in LAMBDAS:
            for ap in (1.0, 0.0):
                rec = evaluate_case(prep, "fused_lam%.2f_ap%.1f" % (lam, ap),
                                    "guided_astar", 1.0, 1.0, ap, lam=lam,
                                    secondary="index")
                rows.append({**{k: v for k, v in rec.items()
                                if not k.startswith("_")},
                             "suite": label_suite,
                             "cost_ratio": (rec["path_cost_m"] or np.nan)
                             / base_cost})
        if (i + 1) % 40 == 0:
            print("[fused] %s %d/%d" % (label_suite, i + 1, len(cases)),
                  flush=True)
    return rows


def summarize(rows):
    out = {}
    for label in {r["label"] for r in rows}:
        sub = [r for r in rows if r["label"] == label]
        exp = np.asarray([r["expanded_nodes"] for r in sub], float)
        ratio = np.asarray([r["cost_ratio"] for r in sub], float)
        ratio = ratio[np.isfinite(ratio)]
        out[label] = {
            "n": len(sub),
            "success": float(np.mean([r["success"] for r in sub])),
            "exp_median": float(np.median(exp)),
            "exp_mean": float(np.mean(exp)),
            "cost_ratio_median": float(np.median(ratio)) if len(ratio) else None,
            "cost_ratio_mean": float(np.mean(ratio)) if len(ratio) else None,
            "cost_ratio_max": float(np.max(ratio)) if len(ratio) else None,
            "penalty_median": float(np.median(
                [r["guide_penalty_m"] or 0.0 for r in sub])),
            "ms_median": float(np.median([r["search_time_ms"] for r in sub])),
        }
    return out


def tradeoff_figure(stats, outdir):
    plt = PY()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, suite, title in ((axes[0], "carla", "(a) 200 CARLA test cases"),
                             (axes[1], "hard", "(b) hard suite (6 scenes x 4 refs)")):
        sub = {k: v for k, v in stats.items() if v["suite"] == suite}
        base = sub.get("astar")
        if base:
            ax.scatter(base["exp_median"], 1.0, s=120, marker="s",
                       c="#1f77b4", label="A* (no guidance)", zorder=5)
        foc = sub.get("pdg_focal")
        if foc:
            ax.scatter(foc["exp_median"], foc["cost_ratio_median"], s=120,
                       marker="D", c="#7b4fb0",
                       label="FOCAL tie-break (current Ours)", zorder=5)
        for ap, marker, colour in ((1.0, "o", "#ff7f0e"), (0.0, "^", "#8c564b")):
            pts = [(v["exp_median"], v["cost_ratio_median"],
                    k.split("_lam")[1].split("_")[0])
                   for k, v in sub.items()
                   if "_ap%.1f" % ap in k]
            pts.sort(key=lambda p: float(p[2]))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=marker, color=colour, lw=1.4,
                    label="fused cost, alpha_p=%.0f" % ap)
            for x, y, lam in pts:
                ax.annotate("λ=%s" % lam, (x, y), textcoords="offset points",
                            xytext=(4, 6), fontsize=7)
        ax.set_xlabel("median expanded nodes")
        ax.set_ylabel("median path cost / A* optimum")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("fused guidance cost: how much real cost buys how much search",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_fused_tradeoff.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--processed", default="data/processed/160m-k4p-c48")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene-to-meter", type=float, default=80.0)
    ap.add_argument("--robot-radius", type=float, default=0.8)
    ap.add_argument("--safety-margin", type=float, default=0.3)
    ap.add_argument("--already-inflated-cells", type=float, default=4.0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    rows += run(real_cases(args), args, "carla")
    rows += run(hard_cases(args), args, "hard")
    write_csv(rows, os.path.join(args.out, "records_fused.csv"))

    stats = summarize(rows)
    for k in stats:
        stats[k]["suite"] = next(r["suite"] for r in rows if r["label"] == k)
    write_csv([{"label": k, **v} for k, v in stats.items()],
              os.path.join(args.out, "aggregate_fused.csv"))
    print("[fig]", tradeoff_figure(stats, args.out))

    print("\n%-22s %6s %6s %9s %9s %9s %8s" % (
        "config", "n", "succ", "exp_med", "cost/opt", "(mean)", "penalty_m"))
    for suite in ("carla", "hard"):
        print("--- %s ---" % suite)
        for label, v in sorted(stats.items(), key=lambda kv: kv[1]["exp_median"]):
            if v["suite"] != suite:
                continue
            print("%-22s %6d %6.3f %9.0f %9.4f %9.4f %8.2f"
                  % (label, v["n"], v["success"], v["exp_median"],
                     v["cost_ratio_median"] or np.nan,
                     v["cost_ratio_mean"] or np.nan, v["penalty_median"]))


if __name__ == "__main__":
    main()
