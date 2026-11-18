"""A/B: does the dynamic FOCAL threshold cause the expansion explosion?

The production loop sets ``thr = w * f_min`` where ``f_min`` is the LIVE minimum
f0 of the open set.  Because the guidance rule deliberately expands nodes with
``f0 > f_min`` (that is what FOCAL is for), ``f_min`` creeps upward, the
threshold creeps upward with it, and more nodes are admitted -- a positive
feedback loop.  This script measures it against a STATIC threshold
``w * h(start)`` (a lower bound on the optimal cost, so the same w-bound holds).

Both variants are otherwise the production code (same tie-break, same guidance,
same neighbour table); "astar" is the plain w=1 reference.

    python scripts/debug/exp_focal_threshold.py --limit 60
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import make_loader          # noqa: E402
from src.diffusion.schedule import NoiseSchedule                     # noqa: E402
from src.models.factory import build_model                           # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402

fa = importlib.import_module("src.grid_search.focal_astar")
dp = importlib.import_module("src.grid_search.diffusion_path")
lg = importlib.import_module("src.grid_search.lazy_guidance")


def focal_variant(safe, start_idx, goal_idx, cfg, neighbors, mode="dynamic"):
    """Copy of focal_search's loop with a switchable FOCAL threshold."""
    spec = safe.spec
    n = spec.n_cells
    start_idx, goal_idx = int(start_idx), int(goal_idx)
    secondary = cfg.secondary
    guidance = cfg.guidance if bool(getattr(cfg.guidance, "valid", False)) else None
    if mode == "astar":
        secondary, guidance = "f0_index", None
    goal_rc = spec.index_to_rc(np.array([goal_idx]))[0]
    width = int(spec.width)
    g = np.full(n, np.inf)
    f0 = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int32)
    closed = np.zeros(n, dtype=bool)
    in_open = np.zeros(n, dtype=bool)
    clearance_flat = np.ascontiguousarray(safe.clearance_m.reshape(-1))
    nbr_idx, move_cost = neighbors.idx, neighbors.cost
    rr, cc = np.divmod(np.arange(n, dtype=np.int64), width)
    h_tab = fa.octile_heuristic(rr, cc, goal_rc, float(spec.cell_m))
    static_thr = float(cfg.w) * float(h_tab[start_idx])   # lower bound on w*C*
    g[start_idx] = 0.0
    f0[start_idx] = h_tab[start_idx]
    in_open[start_idx] = True
    open_nodes = np.empty(n, dtype=np.int32)
    open_nodes[0] = start_idx
    open_n, expanded = 1, 0

    def _compact():
        nonlocal open_n
        alive = open_nodes[:open_n]
        alive = alive[in_open[alive]]
        open_n = len(alive)
        open_nodes[:open_n] = alive

    t0 = time.perf_counter()
    while open_n:
        live = open_nodes[:open_n]
        mask = in_open[live]
        if not mask.any():
            break
        f_live = np.where(mask, f0[live], np.inf)
        f_min = float(f_live.min())
        if mode == "static":
            thr = static_thr
        elif mode == "tie1":
            thr = f_min
        else:
            thr = float(cfg.w) * f_min
        cand = live[mask & (f_live <= thr + 1e-12)]
        if len(cand) == 0:
            break
        if mode == "astar":
            cur = int(cand[np.argmin(f0[cand])])
        else:
            h_d = guidance.score(cand, g[cand], cfg.alpha_d, cfg.alpha_p,
                                 cfg.sigma_m, cfg.L_ref_m)
            cur = fa._tiebreak_pick(cand, h_d, f0[cand], clearance_flat[cand],
                                    secondary)
        in_open[cur] = False
        closed[cur] = True
        expanded += 1
        if cur == goal_idx:
            elapsed = (time.perf_counter() - t0) * 1000.0
            path = fa._reconstruct(parent, goal_idx)
            return {"expanded": expanded, "ms": elapsed, "f_min_end": f_min,
                    "cost_m": fa._path_cost(spec, path), "status": "SUCCESS"}
        base = g[cur]
        nbs = nbr_idx[cur]
        for k in range(8):
            nb = int(nbs[k])
            if nb < 0:
                continue
            tentative = base + float(move_cost[k])
            if tentative + 1e-12 >= g[nb]:
                continue
            g[nb] = tentative
            f0[nb] = tentative + float(h_tab[nb])
            parent[nb] = cur
            closed[nb] = False
            if not in_open[nb]:
                in_open[nb] = True
                if open_n >= len(open_nodes):
                    _compact()
                open_nodes[open_n] = nb
                open_n += 1
        if open_n > 1024 and open_n > 2 * int(in_open[open_nodes[:open_n]].sum()):
            _compact()
    return {"expanded": expanded, "ms": (time.perf_counter() - t0) * 1000.0,
            "cost_m": np.inf, "status": "NO_PATH", "f_min_end": f_min}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q128.yaml")
    ap.add_argument("--ckpt", default="outputs/models/v5_c32q128/ckpt/final_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-ids", default=None)
    args = ap.parse_args()

    from src.diffusion.sampler_v5 import _build_safe_grids

    cfg = load_config(args.config)
    gcfg = dict(cfg.get("grid_search") or {})
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
    only = set(int(x) for x in args.only_ids.split(",")) if args.only_ids else None

    rows = []
    for bi, batch in enumerate(loader):
        if only is not None and int(batch["sample_id"][0]) not in only:
            continue
        if bi % args.stride:
            continue
        sid = int(batch["sample_id"][0])
        cond = batch["cond"].to(args.device)
        occ = batch["occupancy"].to(args.device)
        heading = batch.get("heading")
        hv = batch.get("heading_valid")
        heading = heading.to(args.device) if heading is not None else None
        hv = hv.to(args.device) if hv is not None else None
        grids, spec, _ = _build_safe_grids(occ, cfg)
        safe, nbr = grids[0]
        T = int(sched.num_timesteps)
        q = torch.randn(1, int(model.num_controls), 2, device=args.device)
        q[:, 0] = cond[:, 0]
        q[:, -1] = cond[:, -1]
        tb = torch.full((1,), T - 1, device=args.device, dtype=torch.long)
        ab = sched.sqrt_alphas_cumprod[T - 1].expand(1).to(args.device)
        with torch.no_grad():
            out = model(q, occ, cond, tb, ab, active_path=None, feedback_delta=None,
                        feedback_valid=None, heading=heading, heading_valid=hv)
        curve = out["final"][0].detach().cpu().numpy().astype(np.float64)
        ref = dp.prepare_reference_path(curve, spec,
                                        start_world=cond[0, 0].cpu().numpy().astype(np.float64),
                                        goal_world=cond[0, 1].cpu().numpy().astype(np.float64))
        field = lg.LazyGuidance(ref, spec, J=int(gcfg.get("J", 3)))
        s_idx = int(spec.cell_to_index(spec.world_to_cell(
            cond[0, 0].cpu().numpy().astype(np.float64))))
        g_idx = int(spec.cell_to_index(spec.world_to_cell(
            cond[0, 1].cpu().numpy().astype(np.float64))))
        base = dict(heuristic="octile", secondary="guidance", guidance=field,
                    alpha_d=1.0, alpha_p=float(gcfg.get("alpha_p", 1.0)))
        res = {}
        for mode in ("dynamic", "static", "tie1", "astar"):
            # fresh field cache per variant so nobody benefits from the other's
            f = lg.LazyGuidance(ref, spec, J=int(gcfg.get("J", 3)))
            c = fa.SearchConfig(w=float(gcfg.get("focal_w", 1.2)), **{**base, "guidance": f})
            res[mode] = focal_variant(safe, s_idx, g_idx, c, nbr, mode)
        rows.append({"sample": sid, **{f"{m}_{k}": v for m, r in res.items()
                                       for k, v in r.items()}})
        print("[ab] %d  dyn exp=%6d %7.1fms cost=%6.1f | tie1 exp=%5d %6.1fms cost=%6.1f"
              " | astar exp=%5d %6.1fms cost=%6.1f"
              % (sid, res["dynamic"]["expanded"], res["dynamic"]["ms"], res["dynamic"]["cost_m"],
                 res["tie1"]["expanded"], res["tie1"]["ms"], res["tie1"]["cost_m"],
                 res["astar"]["expanded"], res["astar"]["ms"], res["astar"]["cost_m"]),
              flush=True)

    def c(k):
        return np.asarray([r[k] for r in rows], float)

    print("\n=== %d samples ===" % len(rows))
    for m in ("dynamic", "static", "tie1", "astar"):
        e, t, cm = c(m + "_expanded"), c(m + "_ms"), c(m + "_cost_m")
        print("%-8s expanded p50=%7.0f p90=%8.0f max=%7.0f | ms p50=%7.1f p90=%8.1f "
              "total=%8.2fs | cost p50=%6.1f mean=%6.1f"
              % (m, np.median(e), np.percentile(e, 90), e.max(), np.median(t),
                 np.percentile(t, 90), t.sum() / 1000.0, np.median(cm), cm.mean()))
    opt = c("astar_cost_m")
    for m in ("dynamic", "static", "tie1"):
        ratio = c(m + "_cost_m") / opt
        print("%-8s cost / astar-cost: p50=%.3f p95=%.3f max=%.3f  (over 1.2x: %d)"
              % (m, np.median(ratio), np.percentile(ratio, 95), ratio.max(),
                 int((ratio > 1.2 + 1e-9).sum())))
    print("speedup tie1  vs dynamic: %.2fx on this subset"
          % (c("dynamic_ms").sum() / max(c("tie1_ms").sum(), 1e-9)))
    print("speedup astar vs dynamic: %.2fx on this subset"
          % (c("dynamic_ms").sum() / max(c("astar_ms").sum(), 1e-9)))
    with open(os.path.join(ROOT, ".tmp", "exp_focal_threshold.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
