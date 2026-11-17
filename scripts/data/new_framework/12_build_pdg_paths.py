#!/usr/bin/env python
"""12_build_pdg_paths.py - offline training paths from a teacher x0 + PDG-Focal A*.

Bootstrap step 2-4 of plan 搂9.2, for the C=32 / Q=64 root:

  1. run the shipped v4 teacher (C=48, ``outputs/models/oneshot_k4p_c48``) with
     PLAIN DDIM (``guidance="none"``, all timesteps, ``return_trace=True``) and
     take the LAST trace entry's ``final`` curve as the teacher ``x0``;
  2. build the SAFE grid of that sample with the REAL robot inflation
     (``robot_radius_m=0.8`` + ``safety_margin_m=0.3`` -> 1.76 cells) and
     ``already_inflated_cells=0``: the k4p map is ERODED (obstacles shrunk by 4
     cells, ``05_erode_occupancy.py``), which is NOT a safety inflation, so the
     robot radius must not be cancelled (plan 搂4.3);
  3. PDG-Focal A* (w=1.2) on that grid with the x0 curve as the soft reference;
  4. independent ``validate_path`` re-check; on success the path centres are
     converted to scene coordinates and resampled to 128 arc-length-uniform
     points (endpoints pinned to the continuous start/goal, like every other
     geometry array of this cache);
  5. on NO_PATH / INVALID_START / INVALID_GOAL / validation failure a STRAIGHT
     start->goal line is stored instead and ``pdg_path_valid = 0``.

Output (new root, per split):
    pdg_path_points.npy  [N,128,2] float32
    pdg_path_valid.npy   [N]       bool
    pdg_path_status.npy  [N]       int8   (see STATUS_NAMES; 0 = search success)
  + pdg_stats.json (per split: success rate, expansions, timings, reasons)

Resumable: partial arrays are flushed to ``<split>/_pdg_partial.npz`` every
``--save-every`` samples.

    # self-check, then the REQUIRED order: test -> val -> train
    python scripts/data/new_framework/12_build_pdg_paths.py --limit 8
    python scripts/data/new_framework/12_build_pdg_paths.py --splits test
    python scripts/data/new_framework/12_build_pdg_paths.py --splits val
    python scripts/data/new_framework/12_build_pdg_paths.py --splits train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.diffusion.sampler import sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.grid_search import (GridSpec, baselines, build_neighbor_table,  # noqa: E402
                             build_progress_guidance, build_safe_grid,
                             prepare_reference_path, validate_path)
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_ROOT = "data/processed/160m-k4p-c32q64"
DEFAULT_CONFIG = "configs/config_160k4p_c48_oneshot.yaml"
DEFAULT_CKPT = "outputs/models/oneshot_k4p_c48/ckpt/best_task.pt"
RES = 256
H_POINTS = 128
PATH_POINTS = 128
ROBOT_RADIUS_M = 0.8
SAFETY_MARGIN_M = 0.3
ALREADY_INFLATED_CELLS = 0.0        # plan 搂4.3: erosion is NOT inflation
W_FOCAL = 1.2
J_GUIDE = 3
STATUS_NAMES = {0: "SUCCESS", 1: "NO_PATH", 2: "INVALID_START",
                3: "INVALID_GOAL", 4: "DEGENERATE_PATH", 5: "EXCEPTION",
                6: "VALIDATION_FAILED"}
CAVEATS = [
    "the k4p occupancy has its obstacles ERODED by 4 cells (0.625 m/cell); "
    "that widens the free channels and is NOT a safety inflation, so the robot "
    "radius (0.8 m) + margin (0.3 m) is applied on top (already_inflated_cells=0)",
    "the eroded map has already lost obstacle detail: the true safe boundary "
    "cannot be recovered exactly, and the ellipses/paths are optimistic with "
    "respect to the original map",
    "teacher x0 comes from the C=48 v4 checkpoint; it is used only as the soft "
    "reference of the grid search, never as a label",
]


def _resolve(path_like) -> str:
    p = os.path.expanduser(str(path_like))
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _load_split(root: str, split: str):
    d = os.path.join(root, split)
    out = {}
    for name, key in (("conditions.npy", "cond"),
                      ("candidate_mask.npy", "mask"),
                      ("candidate_geometry_lengths.npy", "glen"),
                      ("candidate_geometry_offsets.npy", "goff")):
        out[key] = np.load(os.path.join(d, name))
    for name, key in (("occupancy.npy", "occ"),
                      ("candidate_xy.npy", "cxy"),
                      ("candidate_geometry.npy", "geom")):
        out[key] = np.load(os.path.join(d, name), mmap_mode="r")
    return out, d


def _dense_geometry(src, i, cell):
    """Same construction as ``CarlaSplineDataset._dense_geometry`` (scene coords)."""
    M = int(src["mask"].shape[1])
    glen = np.asarray(src["glen"][i], dtype=np.int64).copy()
    G = max(int(glen.max(initial=0)) if glen.size else 0, 2)
    geom = np.zeros((M, G, 2), np.float32)
    cond = np.asarray(src["cond"][i], np.float32).reshape(2, 2)
    for m in range(M):
        n = int(glen[m])
        if n <= 0:
            glen[m] = 0
            continue
        lo, hi = int(src["goff"][i, m]), int(src["goff"][i, m + 1])
        px = np.asarray(src["geom"][lo:hi], dtype=np.float32)
        n = min(n, len(px))
        glen[m] = n
        geom[m, :n] = (px[:n] + 0.5) * cell - 1.0
        if n >= 2:
            geom[m, 0] = cond[0]
            geom[m, n - 1] = cond[1]
    return geom, glen


def _resample(points: np.ndarray, n: int, start, goal) -> np.ndarray | None:
    """Arc-length resample to ``n`` points, endpoints pinned to start/goal."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2 or not np.isfinite(pts).all():
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-9:
        return None
    s = np.linspace(0.0, cum[-1], int(n))
    out = np.stack([np.interp(s, cum, pts[:, k]) for k in range(2)], axis=-1)
    out[0] = np.asarray(start, np.float64)
    out[-1] = np.asarray(goal, np.float64)
    return out.astype(np.float32)


def _straight(start, goal, n) -> np.ndarray:
    a = np.asarray(start, np.float64).reshape(2)
    b = np.asarray(goal, np.float64).reshape(2)
    t = np.linspace(0.0, 1.0, int(n))[:, None]
    return (a[None, :] + t * (b - a)[None, :]).astype(np.float32)


def _x0_free_fraction(occ_free: np.ndarray, x0: np.ndarray) -> float:
    rc = np.rint((np.asarray(x0, np.float64) + 1.0) / 2.0 * RES - 0.5).astype(np.int64)
    rc = np.clip(rc, 0, RES - 1)
    return float(occ_free[rc[:, 1], rc[:, 0]].mean())


def _solve_one(occ, start, goal, x0, spec):
    """One sample: safe grid -> PDG path -> 128 path points + diagnostics."""
    t0 = time.perf_counter()
    safe = build_safe_grid(occ, spec, robot_radius_m=ROBOT_RADIUS_M,
                           safety_margin_m=SAFETY_MARGIN_M,
                           already_inflated_cells=ALREADY_INFLATED_CELLS)
    nbr = build_neighbor_table(safe.free, spec.cell_m)
    t1 = time.perf_counter()
    ref = prepare_reference_path(np.asarray(x0, np.float64), spec,
                                 np.asarray(start, np.float64),
                                 np.asarray(goal, np.float64))
    field = build_progress_guidance(ref, spec, J=J_GUIDE)
    t2 = time.perf_counter()
    s_idx = int(spec.cell_to_index(spec.world_to_cell(np.asarray(start, np.float64))))
    g_idx = int(spec.cell_to_index(spec.world_to_cell(np.asarray(goal, np.float64))))
    res = baselines.pdg_focal_astar(safe, s_idx, g_idx, field, w=W_FOCAL,
                                   neighbors=nbr)
    t3 = time.perf_counter()
    diag = dict(prep_ms=(t1 - t0) * 1e3, guide_ms=(t2 - t1) * 1e3,
                search_ms=(t3 - t2) * 1e3, expanded=int(res.stats.get(
                    "expanded_nodes", -1)), cost_m=float(res.cost_m),
                x0_free_frac=_x0_free_fraction(safe.free, x0),
                ref_valid=bool(ref.valid), ref_reason=ref.reason or "",
                guide_valid=bool(getattr(field, "valid", False)),
                start_free=bool(safe.is_free_index(np.array([s_idx]))[0]),
                goal_free=bool(safe.is_free_index(np.array([g_idx]))[0]),
                search_cost_m=float(res.stats.get("search_cost_m", np.nan)))
    if res.status != "SUCCESS" or res.path is None or len(res.path) < 2:
        code = {"SUCCESS": 4, "NO_PATH": 1, "INVALID_START": 2,
                "INVALID_GOAL": 3}.get(str(res.status), 5)
        diag["status"] = code
        diag["detail"] = str(res.status)
        diag["snap_m"] = [float("nan"), float("nan")]
        return None, diag
    chk = validate_path(safe, res.path, s_idx, g_idx, res.cost_m)
    if not chk.get("valid", False):
        diag["status"] = 6
        diag["detail"] = ";".join(chk.get("errors") or ["invalid"])
        diag["snap_m"] = [float("nan"), float("nan")]
        return None, diag
    rc = spec.index_to_rc(res.path)
    cells = rc.astype(np.float64)[:, ::-1]                  # (col, row)
    pts = spec.cell_to_world(cells)
    diag["snap_m"] = [float(np.linalg.norm(pts[0] - start) * spec.scene_to_meter
                            / 1.0),
                      float(np.linalg.norm(pts[-1] - goal) * spec.scene_to_meter
                            / 1.0)]
    out = _resample(pts, PATH_POINTS, start, goal)
    if out is None:
        diag["status"] = 4
        diag["detail"] = "degenerate_resample"
        return None, diag
    diag["status"] = 0
    diag["detail"] = ""
    return out, diag


def _partial_paths(split_dir: str, limit=None) -> str:
    name = ("_pdg_partial.npz" if limit is None
            else "_pdg_partial_limit%d.npz" % int(limit))
    return os.path.join(split_dir, name)


def _out_dir(split_dir: str, limit) -> str:
    """Official arrays only for a FULL run; ``--limit`` writes to a sidecar."""
    if limit is None:
        return split_dir
    d = os.path.join(split_dir, "_selfcheck_partial")
    os.makedirs(d, exist_ok=True)
    return d


def _save_partial(split_dir, state, done, n_done, limit=None):
    np.savez_compressed(
        _partial_paths(split_dir, limit),
        points=state["points"], valid=state["valid"], status=state["status"],
        expanded=state["expanded"], search_ms=state["search_ms"],
        prep_ms=state["prep_ms"], guide_ms=state["guide_ms"],
        snap=state["snap"], x0_free=state["x0_free"],
        cost_m=state["cost_m"], done=done, n_done=np.int64(n_done))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--splits", nargs="*", default=["test", "val", "train"],
                    help="REQUIRED order: test -> val -> train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stepstxt", dest="steps", type=int, default=None,
                    help="DDIM steps (default: all, i.e. the full 16-step pass)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--force", action="store_true",
                    help="ignore the resume checkpoint and recompute")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = _resolve(args.root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = GridSpec.from_resolution(RES, RES, RES,
                                    scene_to_meter=float(
                                        (cfg.get("data") or {}).get(
                                            "scene_to_meter", 80.0)))
    scene_cell = 2.0 / RES
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model, ckpt, _ = load_model(cfg, args.ckpt, arch="auto", device=device)
    model.eval()
    print("[pdg] root=%s device=%s cell=%.3f m inflation=%.2f cells "
          "(r=%.2f+%.2f m, already_inflated=%.0f) T=%d teacher C=%d"
          % (root, device, spec.cell_m, (ROBOT_RADIUS_M + SAFETY_MARGIN_M)
             / spec.cell_m, ROBOT_RADIUS_M, SAFETY_MARGIN_M,
             ALREADY_INFLATED_CELLS, schedule.num_timesteps,
             int(model.num_controls)), flush=True)

    stats_all = {}
    stats_path = os.path.join(root, "pdg_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, encoding="utf-8") as fh:
                stats_all = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print("[pdg] WARNING: unreadable %s (%s) -> starting a fresh report"
                  % (stats_path, exc), flush=True)
            stats_all = {}
    report = {"root": root, "config": args.config, "ckpt": args.ckpt,
              "scene_to_meter": spec.scene_to_meter, "cell_m": spec.cell_m,
              "robot_radius_m": ROBOT_RADIUS_M,
              "safety_margin_m": SAFETY_MARGIN_M,
              "already_inflated_cells": ALREADY_INFLATED_CELLS,
              "inflation_cells": (ROBOT_RADIUS_M + SAFETY_MARGIN_M) / spec.cell_m,
              "w_focal": W_FOCAL, "j_guide": J_GUIDE,
              "guidance": "none", "steps": args.steps, "batch": args.batch,
              "seed": args.seed, "caveats": CAVEATS}
    if isinstance(stats_all, dict) and "per_split" in stats_all:
        report["per_split"] = stats_all["per_split"]
    else:
        report["per_split"] = {}

    for split in args.splits:
        src, split_dir = _load_split(root, split)
        n_full = int(len(src["cond"]))
        n = n_full if args.limit is None else min(int(args.limit), n_full)
        state = {
            "points": np.zeros((n, PATH_POINTS, 2), np.float32),
            "valid": np.zeros(n, bool),
            "status": np.zeros(n, np.int8),
            "expanded": np.full(n, -1, np.int32),
            "search_ms": np.zeros(n, np.float32),
            "prep_ms": np.zeros(n, np.float32),
            "guide_ms": np.zeros(n, np.float32),
            "snap": np.full((n, 2), np.nan, np.float32),
            "x0_free": np.full(n, np.nan, np.float32),
            "cost_m": np.full(n, np.nan, np.float32),
        }
        done = np.zeros(n, bool)
        part = _partial_paths(split_dir, args.limit)
        if os.path.exists(part) and not args.force:
            with np.load(part) as z:
                if int(z["points"].shape[0]) == n:
                    for k in ("points", "valid", "status", "expanded",
                              "search_ms", "prep_ms", "guide_ms", "snap",
                              "x0_free", "cost_m"):
                        state[k] = np.asarray(z[k])
                    done = np.asarray(z["done"], dtype=bool)
                    print("[pdg] %-5s resumed %d/%d samples from %s"
                          % (split, int(done.sum()), n, os.path.basename(part)),
                          flush=True)
                else:
                    print("[pdg] %-5s partial shape mismatch -> recompute" % split)
                    done = np.zeros(n, bool)
        todo = [i for i in range(n) if not done[i]]
        print("[pdg] %-5s n=%d todo=%d" % (split, n, len(todo)), flush=True)
        t_split = time.time()
        n_done = int(done.sum())
        chunk_start = 0
        cell = scene_cell
        fails, tally = {}, {"ref_invalid": 0, "start_blocked": 0,
                            "goal_blocked": 0}
        while chunk_start < len(todo):
            chunk = todo[chunk_start:chunk_start + int(args.batch)]
            cond_b = np.stack([np.asarray(src["cond"][i], np.float32).reshape(2, 2)
                               for i in chunk])
            occ_b = np.stack([np.asarray(src["occ"][i], np.float32)[None]
                              for i in chunk])
            geom_b, glen_b, cxy_b, mask_b = [], [], [], []
            for i in chunk:
                g, gl = _dense_geometry(src, i, cell)
                geom_b.append(g)
                glen_b.append(gl)
                cxy_b.append(np.asarray(src["cxy"][i], np.float32))
                mask_b.append(np.asarray(src["mask"][i]).astype(bool))
            G = max(g.shape[1] for g in geom_b)
            geom_b = np.stack([np.pad(g, ((0, 0), (0, G - g.shape[1]), (0, 0)))
                               for g in geom_b]).astype(np.float32)
            t_samp = time.perf_counter()
            out = sample(model, schedule,
                         torch.as_tensor(cond_b, device=device),
                         torch.as_tensor(occ_b, device=device),
                         torch.as_tensor(np.stack(cxy_b), device=device),
                         torch.as_tensor(np.stack(mask_b), device=device),
                         torch.as_tensor(geom_b, device=device),
                         torch.as_tensor(np.stack(glen_b), device=device),
                         device=device, steps=args.steps, seed=args.seed,
                         seed_offset=int(chunk[0]), return_trace=True,
                         guidance="none", alm_config=cfg.get("alm") or {},
                         corridor_config=cfg.get("corridor") or {})
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            samp_ms = (time.perf_counter() - t_samp) * 1000.0 / len(chunk)
            x0 = out["trace"][-1]["final"].cpu().numpy()        # [B,128,2]
            for k, i in enumerate(chunk):
                start = np.asarray(src["cond"][i], np.float64)[0]
                goal = np.asarray(src["cond"][i], np.float64)[1]
                try:
                    pts, diag = _solve_one(np.asarray(src["occ"][i]),
                                           start, goal,
                                           np.asarray(x0[k], np.float64), spec)
                except Exception as exc:                        # noqa: BLE001
                    pts, diag = None, {"status": 5, "detail": repr(exc),
                                       "prep_ms": 0.0, "guide_ms": 0.0,
                                       "search_ms": 0.0, "expanded": -1,
                                       "cost_m": np.nan, "x0_free_frac": np.nan,
                                       "ref_valid": False, "ref_reason": "",
                                       "guide_valid": False, "start_free": False,
                                       "goal_free": False, "snap_m": [np.nan] * 2}
                state["status"][i] = int(diag["status"])
                state["expanded"][i] = int(diag.get("expanded", -1))
                state["search_ms"][i] = float(diag.get("search_ms", 0.0))
                state["prep_ms"][i] = float(diag.get("prep_ms", 0.0))
                state["guide_ms"][i] = float(diag.get("guide_ms", 0.0))
                state["x0_free"][i] = float(diag.get("x0_free_frac", np.nan))
                state["cost_m"][i] = float(diag.get("cost_m", np.nan))
                state["snap"][i] = np.asarray(diag.get("snap_m", [np.nan, np.nan]),
                                              np.float32)
                tally["ref_invalid"] += int(not diag.get("ref_valid", True))
                tally["start_blocked"] += int(not diag.get("start_free", True))
                tally["goal_blocked"] += int(not diag.get("goal_free", True))
                if pts is None:
                    state["points"][i] = _straight(start, goal, PATH_POINTS)
                    state["valid"][i] = False
                    fails[str(diag.get("detail", "") or "unknown")] = \
                        fails.get(str(diag.get("detail", "") or "unknown"), 0) + 1
                else:
                    state["points"][i] = pts
                    state["valid"][i] = True
                done[i] = True
                n_done += 1
            chunk_start += len(chunk)
            if n_done % int(args.save_every) < len(chunk) or chunk_start >= len(todo):
                _save_partial(split_dir, state, done, n_done, args.limit)
            el = time.time() - t_split
            print("[pdg] %-5s %d/%d  %.1f ms/sample (x0 %.0f ms, search %.0f ms)"
                  "  valid=%.3f  %.0fs"
                  % (split, n_done, n, el * 1000.0 / max(n_done, 1), samp_ms,
                     float(np.nanmean(state["search_ms"][:chunk[0] + len(chunk)])),
                     float(state["valid"][:chunk[0] + len(chunk)].mean()), el),
                  flush=True)

        _save_partial(split_dir, state, done, n_done, args.limit)
        out_dir = _out_dir(split_dir, args.limit)
        # HARD CONTRACT: an official array must describe the WHOLE split
        if args.limit is None and state["points"].shape[0] != n_full:
            raise SystemExit("[error] %s: refusing to write %d rows over a "
                             "%d-row split"
                             % (split, state["points"].shape[0], n_full))
        np.save(os.path.join(out_dir, "pdg_path_points.npy"), state["points"])
        np.save(os.path.join(out_dir, "pdg_path_valid.npy"), state["valid"])
        np.save(os.path.join(out_dir, "pdg_path_status.npy"), state["status"])
        if args.limit is not None:
            print("[pdg] %-5s --limit %d: official arrays NOT touched; wrote the "
                  "self-check copies to %s/_selfcheck_partial"
                  % (split, args.limit, split), flush=True)
        reasons = {}
        for i in range(n):
            code = int(state["status"][i])
            if code != 0:
                key = STATUS_NAMES.get(code, "UNKNOWN")
                reasons[key] = reasons.get(key, 0) + 1
        ok = state["valid"]
        exp = state["expanded"][ok]
        sms = state["search_ms"]
        stats = dict(
            n=int(n), n_full=int(n_full), success=int(ok.sum()),
            success_rate=float(ok.mean()), failure_reasons=reasons,
            expanded_mean=float(exp.mean()) if exp.size else None,
            expanded_median=float(np.median(exp)) if exp.size else None,
            expanded_p95=float(np.percentile(exp, 95)) if exp.size else None,
            search_ms_mean=float(sms.mean()), search_ms_median=float(np.median(sms)),
            prep_ms_mean=float(state["prep_ms"].mean()),
            guide_ms_mean=float(state["guide_ms"].mean()),
            x0_free_frac_mean=float(np.nanmean(state["x0_free"])),
            path_len_m_mean=float(np.nanmean(state["cost_m"])),
            endpoint_snap_m_max=float(np.nanmax(state["snap"])) if np.isfinite(
                state["snap"]).any() else None,
            per_sample_ms=(state["prep_ms"].mean() + state["guide_ms"].mean()
                           + state["search_ms"].mean()),
            search_seconds=float(sms.sum() / 1000.0),
            x0_reference_invalid=tally["ref_invalid"],
            start_not_free=tally["start_blocked"],
            goal_not_free=tally["goal_blocked"],
            failure_details=dict(sorted(fails.items(),
                                        key=lambda kv: -kv[1])[:10]),
            wall_seconds=float(time.time() - t_split), limited=bool(args.limit))
        # a --limit self-check must not overwrite the official per-split entry
        report["per_split"][split if args.limit is None
                            else "%s__limit%d" % (split, args.limit)] = stats
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
        print("[pdg] %-5s DONE n=%d success=%.4f expanded_mean=%s "
              "search_ms=%.1f wall=%.0fs"
              % (split, n, stats["success_rate"],
                 "n/a" if stats["expanded_mean"] is None
                 else "%.1f" % stats["expanded_mean"],
                 stats["search_ms_mean"], stats["wall_seconds"]), flush=True)

    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    print("\nDONE pdg paths -> %s  (report: pdg_stats.json)" % root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
