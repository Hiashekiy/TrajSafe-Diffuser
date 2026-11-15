"""Re-plot the fused-guidance trade-off from an existing records CSV.

    python scripts/grid_search/plot_fused_tradeoff.py --run outputs/grid_search/run_fused
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np


def PY():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(
        open(os.path.join(args.run, "records_fused.csv"), encoding="utf-8")))
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["suite"], r["label"])].append(r)

    stats = {}
    for (suite, label), sub in groups.items():
        ratio = np.asarray([float(r["cost_ratio"]) for r in sub
                            if r["cost_ratio"] not in ("", "None")], float)
        stats[(suite, label)] = {
            "exp": float(np.median([float(r["expanded_nodes"]) for r in sub])),
            "ratio": float(np.median(ratio)) if len(ratio) else np.nan,
            "penalty": float(np.median([float(r["guide_penalty_m"] or 0)
                                        for r in sub])),
        }

    plt = PY()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, suite, title in (
            (axes[0], "carla", "(a) 100 CARLA test cases"),
            (axes[1], "hard", "(b) hard suite (6 scenes x 4 references)")):
        if ("carla" in {k[0] for k in stats}) and suite == "carla":
            n_cases = len({r["case_id"] for r in rows if r["suite"] == "carla"})
            title = "(a) %d CARLA test cases" % n_cases
        base = stats.get((suite, "astar"))
        if base:
            ax.scatter(base["exp"], 1.0, s=130, marker="s", c="#1f77b4",
                       label="A* (no guidance)", zorder=6)
        foc = stats.get((suite, "pdg_focal"))
        if foc:
            ax.scatter(foc["exp"], foc["ratio"], s=130, marker="D", c="#7b4fb0",
                       label="FOCAL tie-break (current Ours)", zorder=6)
        for ap, marker, colour in ((1.0, "o", "#ff7f0e"), (0.0, "^", "#8c564b")):
            pts = []
            for (s, label), v in stats.items():
                if s != suite or "_ap%.1f" % ap not in label:
                    continue
                lam = label.split("_lam")[1].split("_")[0]
                pts.append((float(lam), v["exp"], v["ratio"]))
            if not pts:
                continue
            pts.sort()
            ax.plot([p[1] for p in pts], [p[2] for p in pts], marker=marker,
                    color=colour, lw=1.5,
                    label="fused cost, alpha_p=%.0f" % ap)
            for lam, x, y in pts:
                ax.annotate("lam=%.2f" % lam, (x, y), textcoords="offset points",
                            xytext=(4, 6), fontsize=7)
        ax.set_xlabel("median expanded nodes  (less is better)")
        ax.set_ylabel("median path cost / A* optimum  (1.0 = optimal)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("fused guidance cost in g: search effort vs real path cost",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.run, "fig_fused_tradeoff.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", out)

    for suite in ("carla", "hard"):
        print("=== %s ===" % suite)
        for (s, label), v in sorted(stats.items(),
                                    key=lambda kv: kv[1]["exp"]):
            if s != suite:
                continue
            print("%-22s exp_med=%6.0f  cost/opt=%.4f  penalty=%6.1f m"
                  % (label, v["exp"], v["ratio"], v["penalty"]))


if __name__ == "__main__":
    main()
