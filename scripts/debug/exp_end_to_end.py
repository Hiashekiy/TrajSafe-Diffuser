"""End-to-end paired A/B: legacy search config vs the optimised one.

Process-to-process drift on this box is ~2x (the GPU forward alone measured 51 ms
and 113 ms on identical model code in consecutive runs), so a "before" run and an
"after" run are NOT comparable.  This harness runs BOTH configs inside one
process, on the same batch and the same seed, and reports paired differences --
which also makes the collision / RMSE comparison fair.

    python scripts/debug/exp_end_to_end.py --limit 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import make_loader          # noqa: E402
from src.diffusion.alm_local import collision_mask                   # noqa: E402
from src.diffusion.sampler_v5 import _dense_resample, sample_v5      # noqa: E402
from src.diffusion.schedule import NoiseSchedule                     # noqa: E402
from src.models.factory import build_model                           # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402

LEGACY = {"focal_w": 1.2, "heuristic": "octile", "secondary": "guidance"}
CHAMPION = {"focal_w": 1.2, "heuristic": "geodesic", "secondary": "guidance_h"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q128.yaml")
    ap.add_argument("--ckpt", default="outputs/models/v5_c32q128/ckpt/final_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    loader, _ = make_loader(
        args.split, cfg["data"]["processed_root"], batch_size=1, shuffle=False,
        num_workers=0, limit=args.limit, require_labels=False,
        num_controls=num_controls(cfg), num_safety_queries=num_safety_queries(cfg))
    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                             "squaredcos_cap_v2")).to(args.device)
    model = build_model(cfg).to(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()
    dense_n = int((cfg.get("alm") or {}).get("dense_verify_points", 512))

    def variant(over):
        c = dict(cfg)
        gs = dict(cfg.get("grid_search") or {})
        gs.update(over)
        c["grid_search"] = gs
        return c

    variants = {"legacy": variant(LEGACY), "new": variant(CHAMPION)}
    rows = []
    for bi, batch in enumerate(loader):
        sid = int(batch["sample_id"][0])
        rec = {"sample": sid}
        for name, c in variants.items():
            out = sample_v5(model, sched, batch, c, device=args.device,
                            steps=args.steps, seed=0)
            st = (out["trace"][0]["search"] or [{}])[0]
            curve = out["final"][0].detach().cpu().numpy()
            occ = batch["occupancy"][0:1]
            hit, _ = collision_mask(occ, _dense_resample(curve, dense_n))
            gt = batch["pos"][0].numpy()
            cond = batch["cond"][0].numpy()
            rec[name] = {"total_ms": out["total_ms"], "search_ms": st.get("search_ms", 0.0),
                         "grid_ms": out["grid_ms"], "expanded": st.get("expanded", -1),
                         "coll_frac": float(hit.mean()),
                         "rmse": float(np.sqrt(((curve - gt) ** 2).sum(-1).mean())),
                         "endpoint_err": float(np.linalg.norm(curve[0] - cond[0])
                                               + np.linalg.norm(curve[-1] - cond[1])),
                         "src": (out.get("final_source") or ["model"])[0]}
        rows.append(rec)
        if (bi + 1) % 10 == 0:
            print("[e2e] %d samples" % (bi + 1), flush=True)

    def col(name, key):
        return np.asarray([r[name][key] for r in rows], float)

    print("\n=== %d samples, paired (same batch, same seed) ===" % len(rows))
    for key in ("total_ms", "search_ms", "grid_ms", "expanded", "rmse"):
        a, b = col("legacy", key), col("new", key)
        print("  %-10s legacy p50 %8.2f  new p50 %8.2f  paired delta %+8.2f  "
              "(new faster/smaller on %d/%d)"
              % (key, np.median(a), np.median(b), np.median(b) - np.median(a),
                 int((b < a).sum()), len(rows)))
    for name in ("legacy", "new"):
        c = col(name, "coll_frac")
        print("  %-7s collisions: samples>0 %d/%d  mean frac %.5f  max %.5f"
              % (name, int((c > 0).sum()), len(rows), c.mean(), c.max()))
    print("  speedup (median total): %.2fx   (sum total): %.2fx"
          % (np.median(col("legacy", "total_ms")) / np.median(col("new", "total_ms")),
             col("legacy", "total_ms").sum() / col("new", "total_ms").sum()))
    print("  RMSE mean: legacy %.4f -> new %.4f (%+.1f%%), max abs diff %.4f"
          % (col("legacy", "rmse").mean(), col("new", "rmse").mean(),
             100 * (col("new", "rmse").mean() / col("legacy", "rmse").mean() - 1),
             np.abs(col("new", "rmse") - col("legacy", "rmse")).max()))
    print("  endpoint_err max diff %.3e" % np.abs(
        col("new", "endpoint_err") - col("legacy", "endpoint_err")).max())
    with open(os.path.join(ROOT, ".tmp", "exp_end_to_end.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
