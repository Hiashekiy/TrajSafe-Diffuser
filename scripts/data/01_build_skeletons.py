"""01_build_skeletons.py - step 1: occupancy map -> skeleton branch graph.

For every maze map it writes the cached free mask, skeleton and compressed
branch graph, plus an overlay figure for visual inspection.

    python scripts/data/01_build_skeletons.py --config configs/config.yaml

Outputs (data.skeleton/skeletons by default):

    <maze>.npz            free / skeleton / nodes / branches (load_graph_npz)
    <maze>_overlay.png    occupancy + skeleton + graph
    report.json           per-maze statistics of the extraction
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.skeleton_graph import build_skeleton_graph, save_graph_npz

MAZE_NAMES = ["umaze", "medium", "large"]


def plot_graph(occ, graph, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    ys, xs = np.nonzero(graph.skeleton)
    ax.scatter(xs, ys, s=0.7, c="#1f77b4", linewidths=0, label="skeleton")
    cmap = plt.get_cmap("tab20")
    for i, br in enumerate(graph.branches):
        pix = np.asarray(br.pixels, dtype=float)
        ax.plot(pix[:, 0], pix[:, 1], color=cmap(i % 20), linewidth=1.6,
                label="branches" if i == 0 else None)
    for n in graph.nodes:
        ax.plot([n.center[0]], [n.center[1]], marker="o", markersize=5,
                color="crimson", label="nodes" if n.idx == 0 else None)
        ax.text(n.center[0] + 2, n.center[1] + 2, str(n.idx), fontsize=7,
                color="crimson")
    ax.set_title(title)
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--scenes", default=None,
                    help="directory holding maps/<maze>.npy (default data.scenes_root)")
    ap.add_argument("--out", default=None,
                    help="output directory (default <data.skeleton>/skeletons)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scenes_root = args.scenes or cfg["data"].get("scenes", "data/scenes")
    out = args.out or os.path.join(cfg["data"].get("skeleton", "data/skeleton"),
                                   "skeletons")
    sk_cfg = cfg.get("skeleton", {})
    maps_dir = os.path.join(scenes_root, "maps")
    os.makedirs(out, exist_ok=True)

    report = {"scenes": scenes_root, "out": out, "skeleton": sk_cfg, "mazes": {}}
    for name in MAZE_NAMES:
        map_path = os.path.join(maps_dir, name + ".npy")
        if not os.path.exists(map_path):
            print("[skip] missing map %s" % map_path)
            continue
        occ = np.load(map_path)
        graph = build_skeleton_graph(
            occ,
            safety_dilation_cells=int(sk_cfg.get("safety_dilation_cells", 1)),
            thinning_backend=str(sk_cfg.get("thinning_backend", "auto")),
            pure_cycle_aux_nodes=int(sk_cfg.get("pure_cycle_aux_nodes", 2)),
        )
        graph.stats["map"] = name
        npz_path = os.path.join(out, name + ".npz")
        save_graph_npz(graph, npz_path)
        if not args.no_plot:
            plot_graph(occ, graph, os.path.join(out, name + "_overlay.png"),
                       "%s: %d nodes / %d branches" % (name, len(graph.nodes),
                                                       len(graph.branches)))
        report["mazes"][name] = graph.stats
        print("[%s] nodes=%d branches=%d skeleton_px=%d -> %s"
              % (name, len(graph.nodes), len(graph.branches),
                 graph.stats["skeleton_pixels"], npz_path))

    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DONE", out)


if __name__ == "__main__":
    main()
