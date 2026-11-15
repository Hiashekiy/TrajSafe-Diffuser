"""The two figures that matter: path comparison and cost comparison.

    python scripts/grid_search/plot_paths_and_cost.py \
        --cache outputs/grid_search/diffusion_x0/test/diffusion_x0_test_0_420.npz \
        --run   outputs/grid_search/run_pdg --samples 125 168 47

Produces
    fig_paths.png : same map, one panel per case, ALL methods' returned paths
    fig_cost.png  : path cost per method (selected cases) + 200-case median cost
Nothing else is written.
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
from src.grid_search.benchmark import Case, evaluate_case, prepare_case
from src.grid_search.grid_map import GridSpec

# (label, display, colour)
METHODS = (
    ("astar", "A*", "#1f77b4"),
    ("focal_clearance", "Focal A*", "#7b4fb0"),
    ("pdg", "Ours (PDG-Focal A*)", "#ff7f0e"),
)


def PY():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def build_case(spec, occ, cond, curve, sample, K):
    return Case("s%03d_K%d" % (sample, K), occ, spec, cond[0], cond[1], curve,
                meta={"sample": int(sample), "K": int(K),
                      "warmup_ms_est": 0.0})


def paths_figure(args, spec, occ_all, cond, curves, idxs, k_index, outdir):
    plt = PY()
    n = len(args.samples)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.6), squeeze=False)
    for ax, sample in zip(axes[0], args.samples):
        pos = int(np.where(idxs == sample)[0][0])
        prep = prepare_case(
            build_case(spec, occ_all[pos], cond[pos], curves[pos, k_index],
                       sample, args.K),
            robot_radius_m=args.robot_radius,
            safety_margin_m=args.safety_margin,
            already_inflated_cells=args.already_inflated_cells)
        occ = np.asarray(prep.case.occupancy)
        ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest",
                  vmin=0, vmax=1, alpha=0.55)

        # diffusion reference (soft guide, may cross obstacles)
        if prep.ref.valid:
            rc = spec.index_to_rc(spec.cell_to_index(prep.ref.points))
            ax.plot(rc[:, 1], rc[:, 0], color="#9ecae1", lw=1.6, ls="--",
                    zorder=2, label="diffusion x0 (reference)")

        costs = {}
        for label, nice, colour in METHODS:
            rec = evaluate_case(prep, label, label, 1.2, 1.0,
                                1.0 if label == "pdg" else 0.0)
            path = spec.index_to_rc(np.asarray(rec["_result"].path, np.int64))
            ax.plot(path[:, 1], path[:, 0], color=colour, lw=3.0, zorder=5,
                    label="%s  %.1f m" % (nice, rec["path_cost_m"]))
            costs[label] = rec["path_cost_m"]

        all_pts = [spec.index_to_rc(np.asarray(rec["_result"].path, np.int64))
                   for rec in
                   [evaluate_case(prep, m, m, 1.2, 1.0,
                                  1.0 if m == "pdg" else 0.0)
                    for m, _, _ in METHODS]]
        s = spec.world_to_cell(prep.case.start_world)
        g = spec.world_to_cell(prep.case.goal_world)
        ax.scatter([s[0]], [s[1]], marker="*", s=380, c="#17a34a", zorder=9,
                   edgecolors="k", linewidths=0.8, label="start")
        ax.scatter([g[0]], [g[1]], marker="*", s=380, c="#e31a1c", zorder=9,
                   edgecolors="k", linewidths=0.8, label="goal")

        # square crop around everything that is drawn (paths included!)
        all_pts.append(np.asarray([[s[1], s[0]], [g[1], g[0]]]))
        if prep.ref.valid:
            all_pts.append(spec.index_to_rc(spec.cell_to_index(prep.ref.points)))
        allp = np.concatenate(all_pts, axis=0)
        c = allp.mean(axis=0)
        half = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) / 2 + 8
        ax.set_xlim(c[1] - half, c[1] + half)
        ax.set_ylim(c[0] - half, c[0] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("sample %d  (K=%d)" % (sample, args.K), fontsize=13)
        handles, names = ax.get_legend_handles_labels()
        if sample == args.samples[0]:
            shared = list(zip(handles, names))
    fig.legend([h for h, _ in shared], [n for _, n in shared],
               loc="lower center", ncol=len(shared), fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = os.path.join(outdir, "fig_paths.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def cost_figure(args, outdir):
    """(a) cost per method for the shown cases, (b) 200-case cost statistics."""
    plt = PY()
    rows = list(csv.DictReader(open(os.path.join(args.run, "records_main.csv"),
                                   encoding="utf-8")))
    sel = {}
    for r in rows:
        if r["label"] in dict((m[0], 1) for m in METHODS):
            sel.setdefault(int(r["sample"]), {})[r["label"]] = float(
                r["path_cost_m"] or 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    labels = [m[1] for m in METHODS]
    colours = [m[2] for m in METHODS]
    x = np.arange(len(args.samples))
    w = 0.26
    for i, (label, nice, colour) in enumerate(METHODS):
        vals = [sel.get(s, {}).get(label, np.nan) for s in args.samples]
        bars = axes[0].bar(x + (i - 1) * w, vals, w, label=nice, color=colour)
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.8, "%.1f" % v,
                         ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["sample %d" % s for s in args.samples], fontsize=11)
    axes[0].set_ylabel("path cost (m)", fontsize=11)
    axes[0].set_title("(a) path cost on the cases shown in fig_paths", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # (b) 200-case statistics
    med, p95, ratio = [], [], []
    for label, _, _ in METHODS:
        v = np.asarray([float(r["path_cost_m"]) for r in rows
                        if r["label"] == label and r["path_cost_m"]], float)
        med.append(np.median(v))
        p95.append(np.percentile(v, 95))
        opt = {r["case_id"]: float(r["path_cost_m"]) for r in rows
               if r["label"] == "astar" and r["path_cost_m"]}
        rr = [float(r["path_cost_m"]) / opt[r["case_id"]] for r in rows
              if r["label"] == label and r["path_cost_m"] and
              r["case_id"] in opt]
        ratio.append(np.median(rr))
    x2 = np.arange(len(labels))
    b = axes[1].bar(x2 - 0.18, med, 0.34, color=colours, label="median cost (m)")
    b2 = axes[1].bar(x2 + 0.18, p95, 0.34, color=colours, alpha=0.45,
                     label="P95 cost (m)")
    for bars, vals in ((b, med), (b2, p95)):
        for bb, v in zip(bars, vals):
            axes[1].text(bb.get_x() + bb.get_width() / 2, v + 1.0, "%.1f" % v,
                         ha="center", fontsize=9)
    ax2 = axes[1].twinx()
    ax2.plot(x2, ratio, "ko-", lw=1.6, ms=6, label="median cost / A*")
    for xi, r in zip(x2, ratio):
        ax2.annotate("%.3f" % r, (xi, r), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9)
    ax2.set_ylabel("median cost / A*  (1.0 = optimal)", fontsize=10)
    ax2.set_ylim(0.98, max(ratio) + 0.02)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel("path cost (m)", fontsize=11)
    axes[1].set_title("(b) %.0f test cases: cost statistics" % len(
        {r["case_id"] for r in rows}), fontsize=11)
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_cost.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--processed", default="data/processed/160m-k4p-c48")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene-to-meter", type=float, default=80.0)
    ap.add_argument("--robot-radius", type=float, default=0.8)
    ap.add_argument("--safety-margin", type=float, default=0.3)
    ap.add_argument("--already-inflated-cells", type=float, default=4.0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--samples", type=int, nargs="+", default=[125, 168, 47])
    args = ap.parse_args()

    data = np.load(args.cache, allow_pickle=False)
    curves, steps = data["curves"], list(data["steps"])
    cond, idxs = data["cond"], data["indices"]
    ds = CarlaSplineDataset(args.split, args.processed, require_labels=False)
    occ_all = np.asarray(ds.occupancy)
    res = occ_all.shape[-1]
    spec = GridSpec(res, res, 2.0 * args.scene_to_meter / res,
                    args.scene_to_meter)
    outdir = os.path.join(args.run, "figures")
    os.makedirs(outdir, exist_ok=True)
    print("[fig]", paths_figure(args, spec, occ_all, cond, curves, idxs,
                                steps.index(args.K), outdir))
    print("[fig]", cost_figure(args, outdir))


if __name__ == "__main__":
    main()
