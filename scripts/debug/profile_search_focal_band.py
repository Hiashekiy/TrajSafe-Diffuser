"""Deep dive on a heavy-tail search sample: is the FOCAL band inflating?

For one sample id it records, per expansion, the FOCAL band size, the band's
minimum f0 (= the live f_min), the f0 of the node the guidance actually picked,
and how that compares with plain A* (f0 = min) and with the final path cost.

    python scripts/debug/profile_search_focal_band.py --id 71008
"""

from __future__ import annotations

import argparse
import importlib
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q128.yaml")
    ap.add_argument("--ckpt", default="outputs/models/v5_c32q128/ckpt/final_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    fa = importlib.import_module("src.grid_search.focal_astar")
    rec = {"band": [], "fmin": [], "chosen_f0": [], "h_d": [], "pick": []}
    orig_tiebreak = fa._tiebreak_pick

    def tiebreak(cand_idx, h_d, f0, clearance, secondary):
        pick = orig_tiebreak(cand_idx, h_d, f0, clearance, secondary)
        pos = int(np.flatnonzero(cand_idx == pick)[0])
        rec["band"].append(len(cand_idx))
        rec["fmin"].append(float(f0.min()))
        rec["chosen_f0"].append(float(f0[pos]))
        rec["h_d"].append(float(h_d[pos]) if h_d is not None else np.nan)
        rec["pick"].append(int(pick))
        return pick

    fa._tiebreak_pick = tiebreak

    from src.diffusion.sampler_v5 import _build_safe_grids, _search_once, ddim_step_v5

    cfg = load_config(args.config)
    device = args.device
    loader, ds = make_loader(
        args.split, cfg["data"]["processed_root"], batch_size=1, shuffle=False,
        num_workers=0, limit=None, require_labels=False,
        num_controls=num_controls(cfg), num_safety_queries=num_safety_queries(cfg))
    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                             "squaredcos_cap_v2")).to(device)
    model = build_model(cfg).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval()

    batch = None
    for b in loader:
        if int(b["sample_id"][0]) == args.id:
            batch = b
            break
    if batch is None:
        raise SystemExit("sample %d not in split" % args.id)

    cond = batch["cond"].to(device)
    occ = batch["occupancy"].to(device)
    heading = batch.get("heading")
    hv = batch.get("heading_valid")
    heading = heading.to(device) if heading is not None else None
    hv = hv.to(device) if hv is not None else None
    grids, spec, grid_ms = _build_safe_grids(occ, cfg)
    T = int(sched.num_timesteps)
    q = torch.randn(1, int(model.num_controls), 2, device=device)
    q[:, 0] = cond[:, 0]
    q[:, -1] = cond[:, -1]
    t = T - 1
    tb = torch.full((1,), t, device=device, dtype=torch.long)
    ab = sched.sqrt_alphas_cumprod[t].expand(1).to(device)
    with torch.no_grad():
        out = model(q, occ, cond, tb, ab, active_path=None, feedback_delta=None,
                    feedback_valid=None, heading=heading, heading_valid=hv)
    curve = out["final"]

    path, st = _search_once(grids, spec, cond, curve, cfg, 0)
    band = np.asarray(rec["band"], float)
    fmin = np.asarray(rec["fmin"], float)
    chosen = np.asarray(rec["chosen_f0"], float)
    pick = np.asarray(rec["pick"])
    expanded = len(band)
    goal_idx = int(spec.cell_to_index(spec.world_to_cell(
        cond[0, 1].detach().cpu().numpy().astype(np.float64))))
    free_frac = float(grids[0][0].free.mean())

    print("sample %d  expanded=%d  grid=%dx%d (%.1f%% free cells)  status=%s"
          % (args.id, expanded, spec.height, spec.width, 100 * free_frac, st["status"]))
    print("search_ms=%.1f  path_len_m=%.1f  clearance %.2f -> %.2f"
          % (st["search_ms"], st.get("path_len_m", -1),
             st.get("clearance_before") or -1, st.get("clearance_after") or -1))
    print("\n--- FOCAL band over the expansion sequence (percentiles) ---")
    for name, v in (("band size", band), ("f_min (m)", fmin),
                    ("chosen f0 (m)", chosen), ("slack = chosen - f_min", chosen - fmin)):
        print("  %-22s p5=%.2f p50=%.2f p95=%.2f max=%.2f"
              % (name, *np.percentile(v, [5, 50, 95]), v.max()))
    print("\n  f_min growth: start %.1f -> final %.1f m"
          % (fmin[0], fmin[-1]))
    print("  expansions that picked a node with f0 > 1.01*f_min : %.1f%%"
          % (100 * float((chosen > 1.01 * fmin).mean())))
    print("  expansions that picked a node with f0 > 1.10*f_min : %.1f%%"
          % (100 * float((chosen > 1.10 * fmin).mean())))
    print("  fraction of grid expanded: %.1f%%"
          % (100.0 * expanded / (spec.height * spec.width)))
    print("  unique nodes expanded: %d / %d free cells (%.1f%%)"
          % (len(np.unique(pick)), int(grids[0][0].free.sum()),
             100.0 * len(np.unique(pick)) / max(int(grids[0][0].free.sum()), 1)))
    # where in the sequence did the band blow up?
    print("\n--- band size vs expansion index ---")
    for frac in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
        i = min(int(frac * (expanded - 1)), expanded - 1)
        print("  i=%6d (%3.0f%%)  band=%6d  f_min=%8.2f  chosen_f0=%8.2f  slack=%7.2f"
              % (i, 100 * frac, band[i], fmin[i], chosen[i], chosen[i] - fmin[i]))
    np.savez(os.path.join(ROOT, ".tmp", "focal_band_%d.npz" % args.id),
             band=band, fmin=fmin, chosen=chosen, pick=pick)


if __name__ == "__main__":
    main()
