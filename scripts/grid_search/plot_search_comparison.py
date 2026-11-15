"""Side-by-side visual comparison of the search behaviour of the methods.

For each case the figure shows the SAME map/reference and then the ACTUAL
expansion set + returned path of A*, Focal A* (clearance) and PDG-Focal A*.
Everything is cropped to the region the search actually touched, because on a
160 m x 160 m crop the interesting corridor is only a few dozen cells wide.
Red = the inflation ring (cells that are free on the raw map but blocked by the
robot radius), so the safety model itself is visible.

    python scripts/grid_search/plot_search_comparison.py \
        --cache outputs/grid_search/diffusion_x0/test/diffusion_x0_test_0_420.npz \
        --processed data/processed/160m-k4p-c48 --split test \
        --samples 168 125 47 --out outputs/grid_search/run_pdg/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
    __file__))))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset
from src.grid_search.benchmark import Case, evaluate_case, prepare_case
from src.grid_search.grid_map import GridSpec

METHODS = (
    ("astar", "A* (octile)"),
    ("focal_clearance", "Focal A* (clearance)"),
    ("pdg", "Ours PDG-Focal A*"),
)
PATH_COLOUR = {"astar": "#1f77b4", "focal_clearance": "#9467bd",
               "pdg": "#ff7f0e"}


def PY():
    """Lazy pyplot handle (Agg backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_map(ax, prep, ring=True):
    occ = np.asarray(prep.case.occupancy)
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest",
              vmin=0, vmax=1)
    if ring:
        ring_mask = np.ma.masked_where(
            ~(prep.safe.raw_free & ~prep.safe.free), np.ones_like(occ, float))
        ax.imshow(ring_mask, origin="lower", cmap="Reds", alpha=0.8,
                  interpolation="nearest", vmin=0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_endpoints(ax, prep):
    spec = prep.case.spec
    s = spec.world_to_cell(prep.case.start_world)
    g = spec.world_to_cell(prep.case.goal_world)
    ax.scatter([s[0]], [s[1]], marker="*", s=190, c="#17a34a", zorder=9,
               edgecolors="k", linewidths=0.6)
    ax.scatter([g[0]], [g[1]], marker="*", s=190, c="#d62728", zorder=9,
               edgecolors="k", linewidths=0.6)


def ref_rc(prep):
    spec = prep.case.spec
    if not prep.ref.valid:
        return None
    return spec.index_to_rc(spec.cell_to_index(prep.ref.points))


def draw_reference(ax, prep, dashed=False):
    rc = ref_rc(prep)
    if rc is None:
        return
    ax.plot(rc[:, 1], rc[:, 0], color="#2f6fd0", lw=1.4, alpha=0.95, zorder=5,
            ls="--" if dashed else "-")
    occ = np.asarray(prep.case.occupancy)
    hit = occ[rc[:, 0], rc[:, 1]].astype(bool)
    if hit.any():
        ax.scatter(rc[hit, 1], rc[hit, 0], s=26, marker="x", c="#d62728",
                   zorder=7, linewidths=1.2)


def draw_expansions(ax, prep, result, colour="#17a34a"):
    spec = prep.case.spec
    exp = np.asarray(result.expanded if result.expanded is not None else [],
                     dtype=np.int64)
    if len(exp):
        rc = spec.index_to_rc(exp)
        ax.scatter(rc[:, 1], rc[:, 0], s=2.2, c=colour, alpha=0.85, zorder=3,
                   edgecolors="none")


def draw_path(ax, prep, result, colour):
    if result.path is None:
        return
    spec = prep.case.spec
    p = spec.index_to_rc(np.asarray(result.path, dtype=np.int64))
    ax.plot(p[:, 1], p[:, 0], color=colour, lw=2.2, zorder=6)


def _bounds(pts, margin=10, size=None):
    w = h = size
    stacked = np.concatenate([p for p in pts if p is not None], axis=0)
    lo = np.maximum(stacked.min(axis=0) - margin, 0)
    hi = np.minimum(stacked.max(axis=0) + margin, np.array([h - 1, w - 1]))
    return lo[0], lo[1], hi[0], hi[1]          # r0, c0, r1, c1


def crop(axes, pts, margin=10, w=None, h=None):
    stacked = np.concatenate([p for p in pts if p is not None], axis=0)
    lo = np.maximum(stacked.min(axis=0) - margin, 0)
    hi = np.minimum(stacked.max(axis=0) + margin, np.array([h - 1, w - 1]))
    for ax in axes:
        ax.set_xlim(lo[1], hi[1])
        ax.set_ylim(lo[0], hi[0])


def _case(args, spec, occ, cond, curve, sample):
    return Case("s%03d_K%d" % (sample, args.K), occ, spec, cond[0], cond[1],
                curve, meta={"sample": int(sample), "K": int(args.K),
                             "warmup_ms_est": 0.0})


def compare_real(args, outdir):
    data = np.load(args.cache, allow_pickle=False)
    curves, steps = data["curves"], list(data["steps"])
    cond, idxs = data["cond"], data["indices"]
    ds = CarlaSplineDataset(args.split, args.processed, require_labels=False)
    occ_all = np.asarray(ds.occupancy)
    res = occ_all.shape[-1]
    cell_m = 2.0 * args.scene_to_meter / res
    spec = GridSpec(res, res, cell_m, args.scene_to_meter)
    k_index = steps.index(args.K)
    made = []

    for sample in args.samples:
        pos = int(np.where(idxs == sample)[0][0])
        prep = prepare_case(_case(args, spec, occ_all[pos], cond[pos],
                                  curves[pos, k_index], sample),
                            robot_radius_m=args.robot_radius,
                            safety_margin_m=args.safety_margin,
                            already_inflated_cells=args.already_inflated_cells)
        results = {label: evaluate_case(prep, label, label, 1.2, 1.0,
                                        1.0 if label == "pdg" else 0.0)
                   for label, _ in METHODS}

        pts = [ref_rc(prep)]
        for label, _ in METHODS:
            r = results[label]["_result"]
            if r.expanded is not None and len(r.expanded):
                pts.append(spec.index_to_rc(np.asarray(r.expanded)))
        bounds = _bounds(pts, 12, res)
        aspect = ((bounds[3] - bounds[1] + 1) / max(bounds[2] - bounds[0] + 1, 1))
        panel_w = 4.4
        panel_h = float(min(max(panel_w / max(aspect, 0.2), 1.7), 4.4))
        fig, axes = PY().subplots(1, 4, figsize=(4 * panel_w, panel_h))
        draw_map(axes[0], prep)
        draw_reference(axes[0], prep)
        draw_endpoints(axes[0], prep)
        axes[0].set_title("map + inflation ring (red)\nreference x0 K=%d, %.1f m"
                          % (args.K, prep.ref.length_m), fontsize=8)
        for ax, (label, nice) in zip(axes[1:], METHODS):
            r = results[label]
            draw_map(ax, prep, ring=False)
            draw_reference(ax, prep, dashed=True)
            draw_expansions(ax, prep, r["_result"])
            draw_path(ax, prep, r["_result"], PATH_COLOUR[label])
            draw_endpoints(ax, prep)
            ax.set_title("%s\n%d exp  %.1f ms  %.1f m  (%.2fx opt)"
                         % (nice, r["expanded_nodes"], r["search_time_ms"],
                            r["path_cost_m"],
                            r["path_cost_m"] / results["astar"]["path_cost_m"]),
                         fontsize=8)
        crop(axes, pts, margin=12, w=res, h=res)
        for ax in axes:
            ax.set_aspect("equal")
        fig.suptitle("sample %d   K=%d   w=1.2   cell=%.3f m   "
                     "(green = expanded, thick line = returned path)"
                     % (sample, args.K, cell_m), fontsize=9, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = os.path.join(outdir, "compare_case%03d.png" % sample)
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight")
        PY().close(fig)
        made.append(p)

        fig, axes = PY().subplots(1, 2, figsize=(2 * panel_w, panel_h + 0.4))
        for ax, title in zip(axes, ("expansions overlaid", "returned paths")):
            draw_map(ax, prep, ring=False)
            draw_reference(ax, prep, dashed=True)
            draw_endpoints(ax, prep)
            ax.set_title(title, fontsize=9)
        for label, nice in METHODS:
            r = results[label]["_result"]
            rc = spec.index_to_rc(np.asarray(r.expanded, dtype=np.int64))
            axes[0].scatter(rc[:, 1], rc[:, 0], s=1.8, c=PATH_COLOUR[label],
                            alpha=0.45, label="%s (%d)" % (nice, len(rc)),
                            edgecolors="none")
            draw_path(axes[1], prep, r, PATH_COLOUR[label])
            axes[1].scatter([], [], c=PATH_COLOUR[label],
                            label="%s %.1f m" % (nice.split(" ")[0], r.cost_m))
        axes[0].legend(loc="upper right", fontsize=7, markerscale=6)
        axes[1].legend(loc="upper right", fontsize=7)
        crop(axes, pts, margin=12, w=res, h=res)
        for ax in axes:
            ax.set_aspect("equal")
        fig.suptitle("sample %d: where each method actually looked" % sample,
                     fontsize=9, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p2 = os.path.join(outdir, "compare_overlay_%03d.png" % sample)
        fig.savefig(p2, dpi=args.dpi, bbox_inches="tight")
        PY().close(fig)
        made.append(p2)
    return made


def compare_synthetic(args, outdir):
    from src.grid_search.scenes import scene_catalog
    scenes = [s for s in scene_catalog() if s.ident in args.scenes]
    fig, axes = PY().subplots(2, len(scenes), figsize=(3.5 * len(scenes), 7.2),
                              squeeze=False)
    for j, scene in enumerate(scenes):
        case = Case(scene.ident, scene.occupancy, scene.spec,
                    scene.start_world, scene.goal_world, scene.ref_world,
                    meta={"scene": scene.name})
        prep = prepare_case(case, robot_radius_m=0.0, safety_margin_m=0.0,
                            already_inflated_cells=0.0)
        for i, (label, nice) in enumerate((("astar", "A*"), ("pdg", "Ours"))):
            ax = axes[i][j]
            draw_map(ax, prep, ring=False)
            draw_reference(ax, prep, dashed=True)
            rec = evaluate_case(prep, label, label, 1.2, 1.0,
                                1.0 if label == "pdg" else 0.0)
            draw_expansions(ax, prep, rec["_result"])
            draw_path(ax, prep, rec["_result"], PATH_COLOUR[label])
            draw_endpoints(ax, prep)
            ax.set_title("%s · %s\n%s, %d exp, %.1f m"
                         % (scene.ident, nice, rec["status"],
                            rec["expanded_nodes"], rec["path_cost_m"] or 0.0),
                         fontsize=8)
    fig.suptitle("synthetic acceptance scenes: expansions (green) and paths",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, "compare_synthetic.png")
    fig.savefig(p, dpi=args.dpi, bbox_inches="tight")
    PY().close(fig)
    return [p]


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
    ap.add_argument("--samples", type=int, nargs="+", default=[168, 125, 47])
    ap.add_argument("--scenes", nargs="+",
                    default=["T2", "T3", "T5", "T6", "T11"])
    ap.add_argument("--dpi", type=int, default=125)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    made = compare_real(args, args.out) + compare_synthetic(args, args.out)
    for p in made:
        print("[fig]", p)


if __name__ == "__main__":
    main()
