"""A/B: octile heuristic vs an exact geodesic (MCP) heuristic for the goal.

The octile heuristic is ~20 % loose on these maps (paths have to wind around
obstacles), so the f0 == C* plateau is huge and A* expands thousands of cells.
``skimage.graph.MCP_Geometric`` gives the exact 8-connected geodesic distance to
the goal on the same cost grid (~3 ms for 256x256); it is admissible AND
consistent for our move costs, so it can replace ``h`` without changing the
guarantees.

    python scripts/debug/exp_geodesic_heuristic.py --only-ids 71008,1004
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


def geodesic_field(free, cell_m, goal_rc):
    """Exact 8-connected geodesic distance (metres) from the goal, or None."""
    from skimage.graph import MCP_Geometric
    costs = np.where(free, float(cell_m), np.inf)
    mcp = MCP_Geometric(costs, fully_connected=True)
    field, _ = mcp.find_costs([tuple(int(v) for v in goal_rc)])
    return field


def run(safe, s_idx, g_idx, field_obj, cfg, neighbors, h_tab):
    n = safe.spec.n_cells
    spec = safe.spec
    width = int(spec.width)
    g = np.full(n, np.inf)
    f0 = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int32)
    in_open = np.zeros(n, dtype=bool)
    clearance_flat = np.ascontiguousarray(safe.clearance_m.reshape(-1))
    nbr_idx, move_cost = neighbors.idx, neighbors.cost
    g[s_idx] = 0.0
    f0[s_idx] = h_tab[s_idx]
    in_open[s_idx] = True
    open_nodes = np.empty(n, dtype=np.int32)
    open_nodes[0] = s_idx
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
        cand = live[mask & (f_live <= f_min + 1e-12)]
        h_d = field_obj.score(cand, g[cand], cfg.alpha_d, cfg.alpha_p,
                              cfg.sigma_m, cfg.L_ref_m)
        cur = fa._tiebreak_pick(cand, h_d, f0[cand], clearance_flat[cand], "guidance")
        in_open[cur] = False
        expanded += 1
        if cur == g_idx:
            path = fa._reconstruct(parent, g_idx)
            return {"expanded": expanded, "ms": (time.perf_counter() - t0) * 1000.0,
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
            if not in_open[nb]:
                in_open[nb] = True
                if open_n >= len(open_nodes):
                    _compact()
                open_nodes[open_n] = nb
                open_n += 1
        if open_n > 1024 and open_n > 2 * int(in_open[open_nodes[:open_n]].sum()):
            _compact()
    return {"expanded": expanded, "ms": (time.perf_counter() - t0) * 1000.0,
            "cost_m": np.inf, "status": "NO_PATH"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q128.yaml")
    ap.add_argument("--ckpt", default="outputs/models/v5_c32q128/ckpt/final_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--only-ids", default=None)
    ap.add_argument("--device", default="cuda")
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
        sid = int(batch["sample_id"][0])
        if only is not None and sid not in only:
            continue
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
        ref = dp.prepare_reference_path(
            curve, spec, start_world=cond[0, 0].cpu().numpy().astype(np.float64),
            goal_world=cond[0, 1].cpu().numpy().astype(np.float64))
        s_idx = int(spec.cell_to_index(spec.world_to_cell(
            cond[0, 0].cpu().numpy().astype(np.float64))))
        g_idx = int(spec.cell_to_index(spec.world_to_cell(
            cond[0, 1].cpu().numpy().astype(np.float64))))
        goal_rc = spec.index_to_rc(np.array([g_idx]))[0]
        n = spec.n_cells
        rr, cc = np.divmod(np.arange(n, dtype=np.int64), int(spec.width))
        h_octile = fa.octile_heuristic(rr, cc, goal_rc, float(spec.cell_m))
        t0 = time.perf_counter()
        field = geodesic_field(safe.free, spec.cell_m, goal_rc)
        mcp_ms = (time.perf_counter() - t0) * 1000.0
        field = np.ascontiguousarray(field.reshape(-1), dtype=np.float64)
        field = np.where(np.isfinite(field), field, np.inf)
        res = {}
        for name, h in (("octile", h_octile), ("geodesic", field)):
            f = lg.LazyGuidance(ref, spec, J=int(gcfg.get("J", 3)))
            c = fa.SearchConfig(w=1.0, secondary="guidance", guidance=f,
                                alpha_d=1.0, alpha_p=float(gcfg.get("alpha_p", 1.0)))
            res[name] = run(safe, s_idx, g_idx, f, c, nbr, h)
        rows.append({"sample": sid, "mcp_ms": mcp_ms,
                     **{f"{m}_{k}": v for m, r in res.items() for k, v in r.items()}})
        print("[geo] %d  mcp=%.1fms | octile exp=%6d %7.1fms cost=%6.1f | geodesic exp=%5d "
              "%6.1fms cost=%6.1f"
              % (sid, mcp_ms, res["octile"]["expanded"], res["octile"]["ms"],
                 res["octile"]["cost_m"], res["geodesic"]["expanded"],
                 res["geodesic"]["ms"], res["geodesic"]["cost_m"]), flush=True)

    def c(k):
        return np.asarray([r[k] for r in rows], float)

    print("\n=== %d samples ===" % len(rows))
    print("MCP field build: mean %.2f ms" % c("mcp_ms").mean())
    for m in ("octile", "geodesic"):
        e, t, cm = c(m + "_expanded"), c(m + "_ms"), c(m + "_cost_m")
        print("%-9s expanded p50=%7.0f max=%7.0f | loop ms p50=%7.1f total=%7.2fs | "
              "cost p50=%6.1f" % (m, np.median(e), e.max(), np.median(t), t.sum() / 1000,
                                  np.median(cm)))
    same = np.allclose(c("octile_cost_m"), c("geodesic_cost_m"))
    print("path cost identical in every sample: %s" % same)
    with open(os.path.join(ROOT, ".tmp", "exp_geodesic.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
