"""12_build_skeleton_shape_labels.py - V2 step 4: fixed-centre IRIS labels.

Every skeleton pixel is highly reused (the ellipse centre in V2 is always a
point of the skeleton), so the expensive fixed-centre IRIS solve is done ONCE
per (map, skeleton pixel) and cached as a per-map lookup:

    <base>/skeletons/<maze>_shape4.npy   [H, W, 4] float32
                                         [log a, log b, cos 2t, sin 2t]
    <base>/skeletons/<maze>_shape_valid.npy [H, W] bool
    <base>/skeletons/<maze>_shape_overlay.png

The continuous centre stays c_i = gamma_m(s_i); only the SUPERVISION target for
the shape is read from the nearest skeleton pixel (docs/V2.md section 25).

    python scripts/data/12_build_skeleton_shape_labels.py --config configs/config_v2_skeleton.yaml
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.fixed_center_iris import fixed_center_shape4, shape4_to_P
from src.geometry.skeleton_graph import load_graph_npz

MAZE_NAMES = ["umaze", "medium", "large"]


def plot_shapes(occ, skeleton, shape4, valid, path, title, stride=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    ys, xs = np.nonzero(skeleton)
    for k in range(0, len(ys), max(1, int(stride))):
        x, y = int(xs[k]), int(ys[k])
        if not valid[y, x]:
            continue
        c = np.array([(x + 0.5) * 2.0 / occ.shape[1] - 1.0,
                      (y + 0.5) * 2.0 / occ.shape[0] - 1.0])
        P = shape4_to_P(shape4[y, x])
        ang = np.linspace(0, 2 * np.pi, 64)
        pts = (P @ np.vstack([np.cos(ang), np.sin(ang)])).T + c
        px = (pts[:, 0] + 1.0) / 2.0 * occ.shape[1] - 0.5
        py = (pts[:, 1] + 1.0) / 2.0 * occ.shape[0] - 0.5
        ax.plot(px, py, color="#d62728", linewidth=0.7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v2_skeleton.yaml")
    ap.add_argument("--mazes", nargs="*", default=MAZE_NAMES)
    ap.add_argument("--source", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--window-half", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N skeleton pixels")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = args.source or cfg["data"].get("source", "data/processed_scene_v1")
    base = args.base or cfg["data"].get("base", "data/processed_scene_v2")
    sk_cfg = cfg.get("skeleton", {})
    dilation = int(sk_cfg.get("safety_dilation_cells", 1))
    out_dir = os.path.join(base, "skeletons")

    report = {"window_half": args.window_half, "safety_dilation": dilation,
              "mazes": {}}
    for name in args.mazes:
        occ = np.load(os.path.join(source, "maps", name + ".npy"))
        graph = load_graph_npz(os.path.join(out_dir, name + ".npz"))
        h, w = occ.shape
        shape4 = np.zeros((h, w, 4), dtype=np.float32)
        valid = np.zeros((h, w), dtype=bool)
        ys, xs = np.nonzero(graph.skeleton)
        order = np.lexsort((xs, ys))
        ys, xs = ys[order], xs[order]
        if args.limit:
            ys, xs = ys[:args.limit], xs[:args.limit]

        t0 = time.time()
        n_fail = 0
        areas = []
        for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
            center = graph.pixel_to_scene(np.array([x, y], dtype=float))
            s4, P, safe = fixed_center_shape4(
                occ, center, window_half=args.window_half,
                safety_dilation=dilation)
            if s4 is None or not np.isfinite(s4).all():
                n_fail += 1
                continue
            shape4[y, x] = s4.astype(np.float32)
            valid[y, x] = True
            areas.append(float(np.pi * np.exp(s4[0] + s4[1])))
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print("  [%s] %d/%d  %.0fs (%.0f ms/pixel)"
                      % (name, i + 1, len(ys), el, el / (i + 1) * 1000),
                      flush=True)

        np.save(os.path.join(out_dir, name + "_shape4.npy"), shape4)
        np.save(os.path.join(out_dir, name + "_shape_valid.npy"), valid)
        if not args.no_plot:
            plot_shapes(occ, graph.skeleton, shape4, valid,
                        os.path.join(out_dir, name + "_shape_overlay.png"),
                        "%s: fixed-centre IRIS labels" % name)
        areas = np.asarray(areas) if areas else np.zeros(1)
        report["mazes"][name] = {
            "skeleton_pixels": int(len(ys)),
            "labelled": int(valid.sum()),
            "failed": int(n_fail),
            "safe_rate": float(valid.sum()) / max(len(ys), 1),
            "area_mean": float(areas.mean()),
            "area_median": float(np.median(areas)),
            "area_min": float(areas.min()),
            "area_max": float(areas.max()),
            "seconds": float(time.time() - t0),
        }
        print("[%s] labelled %d / %d (%.1f%%) in %.0fs"
              % (name, int(valid.sum()), len(ys),
                 100.0 * valid.sum() / max(len(ys), 1), time.time() - t0),
              flush=True)

    with open(os.path.join(out_dir, "shape_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
