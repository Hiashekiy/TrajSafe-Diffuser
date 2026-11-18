"""Paired (same-sample, same-grid) comparison of search configurations.

Process-level noise (CPU frequency, background load) makes separate runs
incomparable -- ``grid_ms`` drifted 1.8x between runs although it does not depend
on the config at all.  This harness therefore builds the grid ONCE per sample,
keeps the step-1 prediction fixed, and then runs each config on exactly the same
inputs, reporting per-sample paired differences.

    python scripts/debug/exp_search_configs.py --limit 80
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
from src.diffusion.schedule import NoiseSchedule                     # noqa: E402
from src.models.factory import build_model                           # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402


CONFIGS = {
    "w1.2_octile": {"focal_w": 1.2, "heuristic": "octile"},
    "w1.2_geo": {"focal_w": 1.2, "heuristic": "geodesic"},
    "w1.0_octile": {"focal_w": 1.0, "heuristic": "octile"},
    "w1.0_geo": {"focal_w": 1.0, "heuristic": "geodesic"},
    "w1.0_geo_gh": {"focal_w": 1.0, "heuristic": "geodesic",
                    "secondary": "guidance_h"},
    "w1.2_geo_gh": {"focal_w": 1.2, "heuristic": "geodesic",
                    "secondary": "guidance_h"},
    "w1.4_geo_gh": {"focal_w": 1.4, "heuristic": "geodesic",
                    "secondary": "guidance_h"},
    "w1.6_geo_gh": {"focal_w": 1.6, "heuristic": "geodesic",
                    "secondary": "guidance_h"},
    "gh_tol1": {"focal_w": 1.2, "heuristic": "geodesic",
                "secondary": "guidance_h", "h_tie_tol_cells": 1.0},
    "gh_tol2": {"focal_w": 1.2, "heuristic": "geodesic",
                "secondary": "guidance_h", "h_tie_tol_cells": 2.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q128.yaml")
    ap.add_argument("--ckpt", default="outputs/models/v5_c32q128/ckpt/final_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="comma separated config names")
    args = ap.parse_args()

    from src.diffusion.sampler_v5 import _build_safe_grids, _search_once

    cfg = load_config(args.config)
    names = ([k for k in CONFIGS if k in args.only.split(",")]
             if args.only else list(CONFIGS))
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

    rows = []
    for bi, batch in enumerate(loader):
        sid = int(batch["sample_id"][0])
        cond = batch["cond"].to(args.device)
        occ = batch["occupancy"].to(args.device)
        heading = batch.get("heading")
        hv = batch.get("heading_valid")
        heading = heading.to(args.device) if heading is not None else None
        hv = hv.to(args.device) if hv is not None else None
        grids, spec, _ = _build_safe_grids(occ, cfg)
        T = int(sched.num_timesteps)
        q = torch.randn(1, int(model.num_controls), 2, device=args.device)
        q[:, 0] = cond[:, 0]
        q[:, -1] = cond[:, -1]
        tb = torch.full((1,), T - 1, device=args.device, dtype=torch.long)
        ab = sched.sqrt_alphas_cumprod[T - 1].expand(1).to(args.device)
        with torch.no_grad():
            out = model(q, occ, cond, tb, ab, active_path=None, feedback_delta=None,
                        feedback_valid=None, heading=heading, heading_valid=hv)
        curve = out["final"].detach()
        rec = {"sample": sid}
        for name in names:
            c = dict(cfg)
            gs = dict(cfg.get("grid_search") or {})
            gs.update(CONFIGS[name])
            c["grid_search"] = gs
            path, st = _search_once(grids, spec, cond, curve, c, 0)
            rec[name] = {"ms": st.get("search_ms", np.nan),
                         "expanded": st.get("expanded", -1),
                         "status": st.get("status"),
                         "len_m": st.get("path_len_m", np.nan),
                         "clear": st.get("clearance_after", np.nan),
                         "geo_ms": st.get("geo_ms", 0.0)}
        rows.append(rec)
        print("[cfg] %d  " % sid + "  ".join(
            "%s: %6.1fms e=%4d" % (n, rec[n]["ms"], rec[n]["expanded"])
            for n in names), flush=True)

    def col(name, key):
        return np.asarray([r[name][key] for r in rows], float)

    print("\n=== %d samples, paired ===" % len(rows))
    base = names[0]
    for name in names:
        ms = col(name, "ms")
        ok = np.isfinite(ms)
        d = ms[ok] - col(base, "ms")[ok]
        print("%-12s total %7.2fs  ms p50 %6.1f p90 %7.1f p99 %8.1f max %8.1f | "
              "expanded p50 %5.0f max %6.0f | vs %s: %+7.2fs"
              % (name, ms[ok].sum() / 1000, np.median(ms[ok]),
                 np.percentile(ms[ok], 90), np.percentile(ms[ok], 99), ms[ok].max(),
                 np.median(col(name, "expanded")), col(name, "expanded").max(),
                 base, d.sum() / 1000))
    print()
    for name in names:
        n_ok = int(np.sum([r[name]["status"] == "SUCCESS" for r in rows]))
        print("%-12s SUCCESS %d/%d  path_len p50 %.1f  clearance_after p50 %.2f"
              % (name, n_ok, len(rows), np.median(col(name, "len_m")),
                 np.median(col(name, "clear"))))
    # paired win counts vs the first config
    for name in names[1:]:
        a, b = col(base, "ms"), col(name, "ms")
        print("%-12s faster than %s on %d/%d samples"
              % (name, base, int((b < a).sum()), len(rows)))
    with open(os.path.join(ROOT, ".tmp", "exp_search_configs.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
