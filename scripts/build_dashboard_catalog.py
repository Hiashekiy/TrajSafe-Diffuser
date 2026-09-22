"""build_dashboard_catalog.py - catalog for the Diffusion Lens CARLA dashboard.

Writes ``diffusion-dashboard/lib/dashboard-catalog-carla.json`` with, for every
selected sample of a split:

  * the occupancy map as run-length ``wallRuns`` in SVG pixel space
    (``[x, y_svg, width]`` per obstacle row, exactly the format the frontend
    already draws);
  * start/goal condition and the 128-point GT curve.

Each sample gets its OWN map entry because the CARLA occupancy is a local crop
that changes with the ego pose.

    python scripts/build_dashboard_catalog.py --split test --num 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

RES = 256


def wall_runs(occupancy: np.ndarray):
    """Canonical occupancy [256,256] -> [[x, y_svg, width], ...]."""
    occ = np.asarray(occupancy) > 0.5
    runs = []
    for row in range(occ.shape[0]):
        y_svg = RES - 1 - row                     # scene y=-1 is the bottom row
        line = occ[row]
        x = 0
        while x < line.shape[0]:
            if not line[x]:
                x += 1
                continue
            start = x
            while x < line.shape[0] and line[x]:
                x += 1
            runs.append([int(start), int(y_svg), int(x - start)])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/carla_processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=24)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--include-worst", type=int, default=0,
                    help="also include the N worst samples of "
                         "outputs/bspline_carla/preview_test*/preview_test.json")
    ap.add_argument("--out", default="diffusion-dashboard/lib/"
                                     "dashboard-catalog-carla.json")
    args = ap.parse_args()

    processed = os.path.abspath(args.processed)
    split_dir = os.path.join(processed, args.split)
    conditions = np.load(os.path.join(split_dir, "conditions.npy"))
    curve_gt = np.load(os.path.join(split_dir, "curve_gt.npy"))
    control_gt = np.load(os.path.join(split_dir, "control_gt.npy"))
    occupancy = np.load(os.path.join(split_dir, "occupancy.npy"), mmap_mode="r")
    n = len(conditions)

    idxs = list(range(args.offset, min(args.offset + args.num, n)))
    if args.num < (n - args.offset):
        idxs = np.linspace(args.offset, n - 1, args.num).astype(int).tolist()
    extra = []
    if args.include_worst:
        for cand in sorted([p for p in os.listdir(os.path.join(ROOT, "outputs",
                                                              "bspline_carla"))
                            if p.startswith("preview_")], reverse=True):
            p = os.path.join(ROOT, "outputs", "bspline_carla", cand,
                             "preview_%s.json" % args.split)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    picked = json.load(f).get("picked", [])
                extra = [int(v) for v in picked[-args.include_worst:]]
                break
    for v in extra:
        if v not in idxs:
            idxs.append(int(v))

    samples, maps = [], {}
    for i in idxs:
        key = "%s_%04d" % (args.split, int(i))
        maze = "carla_%04d" % int(i)
        maps[maze] = {"resolution": RES,
                      "wallRuns": wall_runs(np.asarray(occupancy[i]))}
        samples.append({
            "key": key,
            "maze": maze,
            "datasetId": int(i),
            "condition": np.round(conditions[i], 5).tolist(),
            "groundTruth": {"P": np.round(curve_gt[i], 5).tolist(),
                            "control": np.round(control_gt[i], 5).tolist()},
        })
        if (len(samples)) % 8 == 0:
            print("[catalog] %d/%d" % (len(samples), len(idxs)), flush=True)

    # Derive the run directory from --processed so one script serves every
    # dataset, instead of hard-coding the 80 m run:
    #   data/carla_processed        -> outputs/bspline_carla
    #   data/carla_processed_160k8  -> outputs/bspline_carla_160k8
    base = os.path.basename(os.path.normpath(args.processed))
    suffix = base[len("carla_processed"):] \
        if base.startswith("carla_processed") else ""
    run_dir = os.path.join(ROOT, "outputs", "bspline_carla" + suffix)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    summary = os.path.join(run_dir, "training_summary.json")
    epochs = {}
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            d = json.load(f)
        epochs = {"best_task": d.get("best_task_epoch"),
                  "best": d.get("best_epoch"),
                  "latest": d.get("epochs_done")}
    ckpts = {}
    for name in ("best_task", "latest", "best", "best_run1"):
        p = os.path.join(ckpt_dir, name + ".pt")
        if not os.path.exists(p):
            continue
        # the checkpoint file itself is the source of truth; the training
        # summary may describe a different (or unfinished) run
        epoch = None
        try:
            import torch
            epoch = torch.load(p, map_location="cpu",
                               weights_only=False).get("epoch")
        except Exception:
            epoch = epochs.get(name)
        ckpts[name] = "%s.pt (epoch %s)" % (name, epoch)

    catalog = {
        "provenance": {
            "dataset": "%s/%s" % (args.processed, args.split),
            "checkpoints": ckpts,
            "sceneUnits": "[-1,1]^2 (160 m local crop, CARLA local frame)",
            "horizon": 128,
            "timesteps": 16,
            "diffusionState": "32 cubic B-spline controls -> 128-point curve",
            "engine": "trajsafe-controlspace-32",
        },
        "maps": maps,
        "samples": samples,
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, separators=(",", ":"))
    print("wrote %s (%d samples, %.1f MB)"
          % (out, len(samples), os.path.getsize(out) / 1e6), flush=True)


if __name__ == "__main__":
    main()
