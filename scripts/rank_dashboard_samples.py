"""rank_dashboard_samples.py - measure each dashboard sample and sort the catalog.

Runs the real engine (same code path as the panel) for every sample of
``diffusion-dashboard/lib/dashboard-catalog-carla.json``, measures the final
curve RMSE against the dataset GT, writes it into the catalog as
``sample["quality"]`` and re-orders the samples best -> worst so the panel does
not open on its hardest case.

    python scripts/rank_dashboard_samples.py --ckpt best_task
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASH = os.path.join(ROOT, "diffusion-dashboard")
sys.path.insert(0, ROOT)
sys.path.insert(0, DASH)

from engine_carla import Engine                                  # noqa: E402

METERS = 40.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=os.path.join(
        DASH, "lib", "dashboard-catalog-carla.json"))
    ap.add_argument("--ckpt", default="best_task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    eng = Engine(device)
    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)

    out = []
    for n, sample in enumerate(catalog["samples"], 1):
        i = int(sample["datasetId"])
        occ, _cond, gt = eng.sample_arrays("test", i)
        cond = np.asarray(sample["condition"], dtype=np.float32)
        payload = eng.generate(sample["key"], "test", i, occ, cond, args.seed,
                               model_id=args.ckpt)
        final = np.asarray(payload["x0_history"][-1], dtype=np.float64)
        rmse = float(np.linalg.norm(final - gt, axis=1).mean()) * METERS
        m = payload["topology"]["metrics"]
        sample["quality"] = {
            "curve_rmse_m": round(rmse, 4),
            "collision": bool(m["traj_collision"]),
            "selected_idx": int(payload["topology"]["selected_idx"]),
            "selected_ndtw": round(float(m["selected_ndtw"]), 5),
        }
        print("[rank] %2d/%d %s  rmse=%.3f m  collision=%s"
              % (n, len(catalog["samples"]), sample["key"], rmse,
                 m["traj_collision"]), flush=True)
        out.append(sample)

    out.sort(key=lambda s: s["quality"]["curve_rmse_m"])
    catalog["samples"] = out
    catalog["provenance"]["ranked_by"] = (
        "%s seed=%d: final curve RMSE vs dataset GT (best first)"
        % (args.ckpt, args.seed))
    with open(args.catalog, "w", encoding="utf-8") as f:
        json.dump(catalog, f, separators=(",", ":"))
    print("sorted %d samples, best=%.3f m  worst=%.3f m -> %s"
          % (len(out), out[0]["quality"]["curve_rmse_m"],
             out[-1]["quality"]["curve_rmse_m"], args.catalog), flush=True)


if __name__ == "__main__":
    main()
