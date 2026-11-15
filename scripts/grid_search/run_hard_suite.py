"""Hard-scene benchmark: mazes / rooms / spiral / comb / long wall.

Each scene is solved with four DIFFERENT diffusion-like references
(optimal / detour / straight / decoy), because the guidance must survive a bad
prior.  All methods share the occupancy, the safe grid, the 8-neighbour graph
with the no-corner-cutting rule and the real edge costs.

    python scripts/grid_search/run_hard_suite.py --out outputs/grid_search/run_hard
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

from src.grid_search.benchmark import (Case, aggregate, evaluate_case,
                                       prepare_case, write_csv)
from src.grid_search.hard_scenes import REFERENCE_KINDS, hard_scenes, reference_for

CONFIGS = (
    ("dijkstra", "dijkstra", 1.0, 1.0, 0.0),
    ("astar", "astar", 1.0, 1.0, 0.0),
    ("focal_clearance", "focal_clearance", 1.2, 1.0, 0.0),
    ("pdg", "pdg", 1.2, 1.0, 1.0),
    ("pdg_dist_only", "pdg", 1.2, 1.0, 0.0),
)
COLOUR = {"astar": "#1f77b4", "focal_clearance": "#7b4fb0",
          "pdg": "#ff7f0e", "dijkstra": "#8c8c8c", "pdg_dist_only": "#2ca02c"}


def PY():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run(args):
    scenes = hard_scenes()
    records = []
    for scene in scenes:
        for kind in REFERENCE_KINDS:
            ref = reference_for(scene, kind)
            case = Case("%s_%s" % (scene.ident, kind), scene.occupancy,
                        scene.spec, scene.start_world, scene.goal_world, ref,
                        meta={"scene": scene.name, "reference": kind,
                              "ident": scene.ident})
            prep = prepare_case(case, robot_radius_m=args.robot_radius,
                                safety_margin_m=args.safety_margin,
                                already_inflated_cells=0.0)
            for label, method, w, ad, ap in CONFIGS:
                rec = evaluate_case(prep, label, method, w, ad, ap)
                rec["end_to_end_ms"] = (prep.timings["preprocess_ms"]
                                        + prep.timings["reference_ms"]
                                        + prep.timings["guidance_ms"]
                                        + rec["search_time_ms"]
                                        + rec["validation_time_ms"])
                records.append(rec)
            print("[hard] %-3s %-8s ref=%-8s  ref_len=%6.1f m  ref_coll=%.2f"
                  % (scene.ident, scene.name, kind, prep.ref.length_m,
                     records[-len(CONFIGS)]["reference_collision_rate"] or 0.0),
                  flush=True)
    return scenes, records


def figure_paths(args, scenes, outdir):
    """One panel per scene: all methods' returned paths."""
    plt = PY()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2), squeeze=False)
    for ax, scene in zip(axes.ravel(), scenes):
        case = Case(scene.ident, scene.occupancy, scene.spec, scene.start_world,
                    scene.goal_world, reference_for(scene, "detour"),
                    meta={"scene": scene.name})
        prep = prepare_case(case, robot_radius_m=args.robot_radius,
                            safety_margin_m=args.safety_margin,
                            already_inflated_cells=0.0)
        ax.imshow(np.asarray(scene.occupancy), origin="lower", cmap="gray_r",
                  interpolation="nearest", alpha=0.45)
        if prep.ref.valid:
            rc = scene.spec.index_to_rc(scene.spec.cell_to_index(prep.ref.points))
            ax.plot(rc[:, 1], rc[:, 0], color="#9ecae1", lw=1.4, ls="--",
                    zorder=2, label="diffusion x0 (detour)")
        for label, nice in (("astar", "A*"), ("focal_clearance", "Focal A*"),
                            ("pdg", "Ours")):
            rec = evaluate_case(prep, label, label, 1.2, 1.0,
                                1.0 if label == "pdg" else 0.0)
            p = scene.spec.index_to_rc(np.asarray(rec["_result"].path, np.int64))
            ax.plot(p[:, 1], p[:, 0], color=COLOUR[label], lw=2.4, zorder=5,
                    label="%s %.0f m" % (nice, rec["path_cost_m"]))
        s = scene.spec.world_to_cell(scene.start_world)
        g = scene.spec.world_to_cell(scene.goal_world)
        ax.scatter([s[0]], [s[1]], marker="*", s=260, c="#17a34a", zorder=9,
                   edgecolors="k", linewidths=0.6)
        ax.scatter([g[0]], [g[1]], marker="*", s=260, c="#e31a1c", zorder=9,
                   edgecolors="k", linewidths=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        rec_a = evaluate_case(prep, "astar", "astar", 1.2, 1.0, 0.0)
        rec_o = evaluate_case(prep, "pdg", "pdg", 1.2, 1.0, 1.0)
        ax.set_title("%s · %s\nA* %d exp / %.0f m   Ours %d exp / %.0f m"
                     % (scene.ident, scene.name, rec_a["expanded_nodes"],
                        rec_a["path_cost_m"], rec_o["expanded_nodes"],
                        rec_o["path_cost_m"]), fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("hard scenes: returned paths (reference = 'detour' prior)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_hard_paths.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def figure_stats(args, records, outdir):
    """Bars: expansions and cost, grouped by reference quality."""
    plt = PY()
    kinds = list(REFERENCE_KINDS)
    labels = [c[0] for c in CONFIGS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.16
    x = np.arange(len(kinds))
    for i, label in enumerate(labels):
        exp, cost = [], []
        for kind in kinds:
            sub = [r for r in records
                   if r["label"] == label and r["reference"] == kind]
            exp.append(np.median([r["expanded_nodes"] for r in sub]))
            cost.append(np.median([r["path_cost_m"] for r in sub
                                   if r["path_cost_m"]]))
        off = (i - (len(labels) - 1) / 2) * w
        axes[0].bar(x + off, exp, w, label=label, color=COLOUR.get(label))
        axes[1].bar(x + off, cost, w, label=label, color=COLOUR.get(label))
    for ax, title, ylabel in ((axes[0], "(a) search effort",
                               "median expanded nodes"),
                              (axes[1], "(b) path cost",
                               "median path cost (m)")):
        ax.set_xticks(x)
        ax.set_xticklabels(["reference: " + k for k in kinds], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("hard suite (6 scenes x 4 reference qualities x 5 methods)",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_hard_stats.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-radius", type=float, default=0.0)
    ap.add_argument("--safety-margin", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scenes, records = run(args)
    write_csv(records, os.path.join(args.out, "records_hard.csv"))
    agg = aggregate(records, ("label", "reference"))
    write_csv(agg, os.path.join(args.out, "aggregate_hard.csv"))
    print("[fig]", figure_paths(args, scenes, args.out))
    print("[fig]", figure_stats(args, records, args.out))

    print("\n== hard suite: median expanded nodes / cost by reference quality ==")
    for kind in REFERENCE_KINDS:
        line = []
        for label, _, _, _, _ in CONFIGS:
            sub = [r for r in records
                   if r["label"] == label and r["reference"] == kind]
            line.append("%s: %5.0f exp/%.0f m"
                        % (label, np.median([r["expanded_nodes"] for r in sub]),
                           np.median([r["path_cost_m"] for r in sub
                                      if r["path_cost_m"]])))
        print("ref=%-9s %s" % (kind, "  ".join(line)))


if __name__ == "__main__":
    main()
