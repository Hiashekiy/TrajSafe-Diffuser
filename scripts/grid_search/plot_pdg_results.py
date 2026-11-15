"""Aggregate figures for the PDG-Focal A* experiment (real data only).

    python scripts/grid_search/plot_pdg_results.py --run outputs/grid_search/run_pdg
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
    __file__))))
sys.path.insert(0, ROOT)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(rows, key, label=None):
    out = []
    for r in rows:
        if label is not None and r.get("label") != label:
            continue
        try:
            out.append(float(r[key]))
        except (TypeError, ValueError):
            continue
    return np.asarray(out, dtype=np.float64)


def cost_ratio(records, label):
    """Per-case path cost divided by the Dijkstra optimum of the same case."""
    opt = {}
    for r in records:
        if r["label"] == "dijkstra" and r.get("path_cost_m"):
            opt[r["case_id"]] = float(r["path_cost_m"])
    ratios = []
    for r in records:
        if r["label"] != label or not r.get("path_cost_m"):
            continue
        base = opt.get(r["case_id"])
        if base:
            ratios.append(float(r["path_cost_m"]) / base)
    return np.asarray(ratios, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = args.run
    figdir = os.path.join(run, "figures")
    os.makedirs(figdir, exist_ok=True)

    main_rec = read_csv(os.path.join(run, "records_main.csv"))
    abl_rec = read_csv(os.path.join(run, "records_ablation.csv"))

    # ---- figure A: main comparison ---------------------------------------
    labels = ["dijkstra", "astar", "focal_clearance", "focal_index", "pdg"]
    med_exp = [np.median(num(main_rec, "expanded_nodes", l)) for l in labels]
    med_ms = [np.median(num(main_rec, "search_time_ms", l)) for l in labels]
    ratios = [np.median(cost_ratio(main_rec, l)) if len(cost_ratio(main_rec, l))
              else np.nan for l in labels]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(labels, med_exp, color="#4c72b0")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median expanded nodes (log)")
    axes[0].set_title("search effort")
    axes[1].bar(labels, med_ms, color="#dd8452")
    axes[1].set_ylabel("median search time (ms)")
    axes[1].set_title("search loop time")
    axes[2].bar(labels, ratios, color="#55a868")
    axes[2].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[2].set_ylabel("median path cost / Dijkstra optimum")
    axes[2].set_title("cost (focal bound)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.suptitle("PDG-Focal A* vs baselines (real CARLA test split, K=3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "A_main_comparison.png"), dpi=args.dpi)
    plt.close(fig)

    # ---- figure B: G0 / G1 / G2 ablation ---------------------------------
    abl_labels = ["G0_no_guidance", "G1_distance_only", "G2_distance_progress"]
    exp = [np.median(num(abl_rec, "expanded_nodes", l)) for l in abl_labels]
    ms = [np.median(num(abl_rec, "search_time_ms", l)) for l in abl_labels]
    rat = [np.median(cost_ratio(abl_rec, l)) if len(cost_ratio(abl_rec, l))
           else np.nan for l in abl_labels]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].bar(abl_labels, exp, color="#4c72b0")
    axes[0].set_ylabel("median expanded nodes")
    axes[0].set_title("G0/G1/G2 ablation: effort")
    axes[1].bar(abl_labels, ms, color="#dd8452")
    axes[1].set_ylabel("median search time (ms)")
    axes[1].set_title("ablation: time")
    axes[2].bar(abl_labels, rat, color="#55a868")
    axes[2].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[2].set_ylabel("cost / optimum")
    axes[2].set_title("ablation: cost")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.suptitle("distance vs distance+progress guidance (same w, same anchor)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "B_ablation.png"), dpi=args.dpi)
    plt.close(fig)

    # ---- figure C: K sweep ------------------------------------------------
    k_path = os.path.join(run, "records_k_sweep.csv")
    if os.path.exists(k_path):
        k_rec = read_csv(k_path)
        ks = sorted({int(r["K"]) for r in k_rec if r.get("K")})
        pdg_exp = [np.median(num([r for r in k_rec if int(r["K"]) == k],
                                 "expanded_nodes", "pdg_K%d" % k)) for k in ks]
        astar_exp = [np.median(num([r for r in k_rec if int(r["K"]) == k],
                                   "expanded_nodes", "astar_ref_K%d" % k))
                     for k in ks]
        ref_coll = [np.mean([float(r["reference_collision_rate"] or 0.0)
                             for r in k_rec if int(r["K"]) == k
                             and r["label"] == "pdg_K%d" % k]) for k in ks]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(ks, pdg_exp, "o-", label="PDG-Focal A*")
        axes[0].plot(ks, astar_exp, "s--", label="A* (same cases)")
        axes[0].set_xlabel("K (executed reverse-diffusion forwards)")
        axes[0].set_ylabel("median expanded nodes")
        axes[0].legend(fontsize=8)
        axes[0].set_title("guidance from earlier/later x0")
        axes[1].plot(ks, ref_coll, "^-", color="#c44e52")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("mean reference collision fraction")
        axes[1].set_title("how unsafe is the x0 reference?")
        fig.suptitle("K sweep on the real test split", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "C_k_sweep.png"), dpi=args.dpi)
        plt.close(fig)

    # ---- figure D: w sweep ------------------------------------------------
    w_path = os.path.join(run, "records_w_sweep.csv")
    if os.path.exists(w_path):
        w_rec = read_csv(w_path)
        ws = sorted({float(r["w"]) for r in w_rec})
        pdg_e = [np.median(num([r for r in w_rec if float(r["w"]) == w],
                               "expanded_nodes", "pdg_w%.1f" % w)) for w in ws]
        foc_e = [np.median(num([r for r in w_rec if float(r["w"]) == w],
                               "expanded_nodes", "focal_w%.1f" % w)) for w in ws]
        pdg_r = [np.median(cost_ratio(w_rec, "pdg_w%.1f" % w)) for w in ws]
        foc_r = [np.median(cost_ratio(w_rec, "focal_w%.1f" % w)) for w in ws]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(ws, pdg_e, "o-", label="PDG-Focal A*")
        axes[0].plot(ws, foc_e, "s--", label="Focal A* (clearance)")
        axes[0].set_xlabel("focal weight w")
        axes[0].set_ylabel("median expanded nodes")
        axes[0].legend(fontsize=8)
        axes[0].set_title("effort vs w")
        axes[1].plot(ws, pdg_r, "o-", label="PDG-Focal A*")
        axes[1].plot(ws, foc_r, "s--", label="Focal A* (clearance)")
        axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
        axes[1].set_xlabel("focal weight w")
        axes[1].set_ylabel("median cost / optimum")
        axes[1].set_title("cost vs w")
        axes[1].legend(fontsize=8)
        fig.suptitle("w sweep on the real test split", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "D_w_sweep.png"), dpi=args.dpi)
        plt.close(fig)

    # ---- figure E: alpha_p sweep -----------------------------------------
    a_path = os.path.join(run, "records_alpha_sweep.csv")
    if os.path.exists(a_path):
        a_rec = read_csv(a_path)
        aps = sorted({float(r["alpha_p"]) for r in a_rec})
        exp = [np.median(num([r for r in a_rec if float(r["alpha_p"]) == a],
                             "expanded_nodes", "pdg_ap%.1f" % a)) for a in aps]
        rat = [np.median(cost_ratio(a_rec, "pdg_ap%.1f" % a)) for a in aps]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(aps, exp, "o-")
        axes[0].set_xlabel("alpha_p (progress weight)")
        axes[0].set_ylabel("median expanded nodes")
        axes[0].set_title("effort vs alpha_p")
        axes[1].plot(aps, rat, "o-", color="#55a868")
        axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
        axes[1].set_xlabel("alpha_p")
        axes[1].set_ylabel("median cost / optimum")
        axes[1].set_title("cost vs alpha_p")
        fig.suptitle("progress-term weight sweep", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "E_alpha_sweep.png"), dpi=args.dpi)
        plt.close(fig)

    summary = os.path.join(run, "summary.json")
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            json.load(f)
    print("[plot] figures ->", figdir)


if __name__ == "__main__":
    main()
