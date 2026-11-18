"""Single-sample (batch=1) latency of the full 4-step pipeline.

Reports the honest per-sample wall clock with a stage breakdown.  batch=1 is
mandatory for a latency claim: the evaluation script's per-sample numbers are
batch totals divided by B, which understates a single frame (and the original
report already warned about the 2164 ms == batch-of-8 mix-up).

    python scripts/debug/bench_single_sample.py --limit 120 --warmup 10
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
from src.diffusion.sampler_v5 import sample_v5                       # noqa: E402
from src.diffusion.schedule import NoiseSchedule                     # noqa: E402
from src.models.factory import build_model                           # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q128.yaml")
    ap.add_argument("--ckpt", default="outputs/models/v5_c32q128/ckpt/final_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--legacy", action="store_true",
                    help="old search config (w 1.2, octile h, guidance ordering)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.legacy:
        cfg.setdefault("grid_search", {}).update(
            {"focal_w": 1.2, "heuristic": "octile", "secondary": "guidance",
             "h_tie_tol_cells": 0.0})
    loader, _ = make_loader(
        args.split, cfg["data"]["processed_root"], batch_size=1, shuffle=False,
        num_workers=0, limit=args.limit + args.warmup, require_labels=False,
        num_controls=num_controls(cfg), num_safety_queries=num_safety_queries(cfg))
    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                             "squaredcos_cap_v2")).to(args.device)
    model = build_model(cfg).to(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()

    rows = []
    for bi, batch in enumerate(loader):
        out = sample_v5(model, sched, batch, cfg, device=args.device,
                        steps=args.steps, seed=0)
        if bi < args.warmup:
            continue
        st = (out["trace"][0]["search"] or [{}])[0]
        net = sum(float(s["net_ms"]) for s in out["trace"])
        alm_steps = sum(1 for s in out["trace"] if (s.get("alm") or {}).get("alm"))
        rows.append({"sample": int(batch["sample_id"][0]),
                     "total": float(out["total_ms"]),
                     "grid": float(out["grid_ms"]),
                     "search": float(st.get("search_ms") or 0.0),
                     "net": net,
                     "alm_steps": alm_steps,
                     "expanded": st.get("expanded", -1),
                     "geo_ms": st.get("geo_ms", 0.0),
                     "center_ms": None})

    def c(k):
        return np.asarray([r[k] for r in rows], float)

    tag = "LEGACY search" if args.legacy else "current config"
    n = len(rows)
    print("\n=== batch=1, steps=%d, %s, %d samples (after %d warmup) ==="
          % (args.steps, tag, n, args.warmup))
    tot = c("total")
    print("  total      p50 %7.1f  mean %7.1f  p90 %7.1f  p99 %7.1f  max %7.1f  ms"
          % (np.median(tot), tot.mean(), pct(tot, 90), pct(tot, 99), tot.max()))
    # sample_v5 starts its total_ms AFTER _build_safe_grids, so the per-sample
    # end-to-end cost is grid_ms + total_ms.  Keep the accounting explicit.
    parts = {"net(4 steps)": c("net"), "search": c("search")}
    parts["pipeline(other)"] = tot - parts["net(4 steps)"] - parts["search"]
    parts["grid(safe grid+nbr)"] = c("grid")
    for k, v in parts.items():
        print("  %-22s p50 %7.1f  mean %7.1f  p90 %7.1f"
              % (k, np.median(v), v.mean(), pct(v, 90)))
    print("  end-to-end per sample  p50 %7.1f  mean %7.1f  ms  (total + grid)"
          % (np.median(tot + c("grid")), (tot + c("grid")).mean()))
    print("  expanded p50 %.0f max %.0f | samples with ALM: %d/%d | geo field p50 %.2f ms"
          % (np.median(c("expanded")), c("expanded").max(),
             int((c("alm_steps") > 0).sum()), n, np.median(c("geo_ms"))))
    # drift check inside the run (this box slows down after ~60 samples)
    if n >= 60:
        a, b = tot[:30], tot[-30:]
        print("  drift check: first 30 p50 %.1f ms / last 30 p50 %.1f ms  (x%.2f)"
              % (np.median(a), np.median(b), np.median(b) / max(np.median(a), 1e-9)))
    with open(os.path.join(ROOT, ".tmp", "bench_single_sample%s.json"
                          % ("_legacy" if args.legacy else "")), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
