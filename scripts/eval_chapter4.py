#!/usr/bin/env python
"""第4章统一评测入口（paper/TrajSafe-Diffuser第4章实验执行指南.md 4.1.3/4.2/4.3/4.4）.

一次调用 = 一个「方法 × 一个 seed × 一个 split」，写出

    <out>.json      聚合结果 + 逐 step 过程序列（4.4 的曲线直接来自这里）
    <out>.md        人读摘要（主表一行 + 逐步表）
    <out>.jsonl     逐样本记录（指南「结果文件建议」的字段表）

方法由 (checkpoint, guidance, feedback, final_project) 四元组唯一确定：

    --guidance none          -> Diffusion（标准条件扩散，不做任何安全处理）
    --guidance penalty       -> Penalty-guided Diffusion
    --guidance projection    -> Projected Diffusion（顺序凸化 + L2 投影）
    --guidance primal_dual   -> Primal-Dual Diffusion
    --guidance alm_sdf       -> SDF-ALM-guided Diffusion
    --guidance alm           -> 本文完整方法（走廊 ALM；--feedback / --final-project 控制两个开关）

公平性（指南 4.1.4）：同一 ``--seed`` 下，初始噪声由
``sample(seed=seed, seed_offset=全局样本下标)`` 决定，因此**同一个样本在所有方法里
拿到完全相同的一份噪声**，chunk 划分、512 点碰撞协议、SDF 与 d_safe 也完全一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import (CarlaSplineDataset,  # noqa: E402
                                               make_collate)
from src.diffusion.sampler import (_dense_decode, _membership_rate,  # noqa: E402
                                   _point_violation, sample)
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.sdf_field import build_occupancy_sdf  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

#: guide 4.1.3 §4 (constraint_tol)
FEASIBLE_EPS = 1e-3
#: guide 4.1.3 §2 -- the reported metrics live on the 512-point dense decode
DENSE_POINTS = 512
#: guide 4.1.4 -- warm-up batches excluded from the timing statistics
TIMING_WARMUP_CHUNKS = 3
#: guide 4.1.4 -- the same 3 reverse steps are warm-up for every method
DEFAULT_WARMUP = 3


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _cast(value: str):
    low = value.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _apply_overrides(cfg: dict, items, label: str) -> None:
    for item in items or []:
        if "=" not in item:
            raise SystemExit("[ch4] --%s wants KEY=VALUE, got %r" % (label, item))
        key, _, value = item.partition("=")
        cfg[key.strip()] = _cast(value)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha1(handle.read()).hexdigest()[:16]
    except OSError:
        return "missing"


def _ms_key(key: str) -> str:
    """``network_s`` -> ``network_ms`` (the accumulator is in seconds)."""
    stem = key[:-2] if key.endswith("_s") else key
    return stem + "_ms"


def _tolist(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [float(v) for v in value.detach().cpu().reshape(-1)]
    return [float(value)]


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _std(values):
    values = [v for v in values if v is not None]
    return float(np.std(values)) if values else None


def _pct(values, q):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def _write_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# metrics (guide 4.1.3)
# ---------------------------------------------------------------------------


def collision_free(occ_row: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``sampler._free_mask`` semantics, re-stated here for the report.

    free  <=>  bilinear occupancy <= 0.5  AND  the point is inside [-1,1]^2.
    The out-of-map test is part of the criterion (guide 4.1.3 §1).
    """
    grid = points.reshape(1, 1, -1, 2)
    value = F.grid_sample(occ_row[None].to(points.dtype), grid, mode="bilinear",
                          padding_mode="border", align_corners=False)
    return (value[0, 0, 0] <= 0.5) & (points.abs() <= 1.0).all(dim=-1)


def offline_corridor_metrics(curve: torch.Tensor, batch: dict):
    """Model-independent check against the OFFLINE GT corridor (128 points)."""
    cell_a = batch["alm_cell_a"].to(curve.device).float()
    cell_b = batch["alm_cell_b"].to(curve.device).float()
    cell_valid = batch["alm_cell_valid"].to(curve.device).bool()
    alm_valid = batch["alm_valid"].to(curve.device).bool()
    nrm = cell_a.norm(dim=-1).clamp_min(1e-9)
    signed = (torch.einsum("bcfk,bhk->bhcf", cell_a, curve)
              - cell_b[:, None]) / nrm[:, None]
    worst = signed.amax(dim=-1).masked_fill(~cell_valid[:, None, :], float("inf"))
    violation = torch.nan_to_num(worst.amin(dim=-1), nan=0.0, posinf=0.0,
                                 neginf=0.0)
    per_max = violation.max(dim=1).values
    per_member = (violation <= 0).float().mean(dim=1)
    per_max = torch.where(alm_valid, per_max, torch.zeros_like(per_max))
    per_member = torch.where(alm_valid, per_member, torch.ones_like(per_member))
    return per_max, per_member


def smoothness_m(points: torch.Tensor, scene_to_meter: float) -> torch.Tensor:
    """Guide 4.1.3 §8: mean second difference of the dense trajectory, in m."""
    if points.shape[1] < 3:
        return points.new_zeros(points.shape[0])
    d2 = points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]
    return d2.norm(dim=-1).mean(dim=1) * float(scene_to_meter)


def evaluate_controls(codec, q: torch.Tensor, occ: torch.Tensor, cond: torch.Tensor,
                      curve_gt: torch.Tensor, sdf, pack, batch: dict,
                      scene_to_meter: float) -> dict:
    """Every reported metric for one batch of control polygons.

    The 4.2 columns come from the OCCUPANCY SDF (``with_boundary=False``), which
    is the field the guide's formulas name; the combined field (which also sees
    the crop) is reported beside it so both conventions are visible.
    """
    q = q.detach()
    p512 = _dense_decode(codec, q, DENSE_POINTS)
    p128 = codec.decode_controls(q)
    stats = sdf.stats(p512, d_safe=0.0, with_boundary=False)
    full = sdf.stats(p512, d_safe=0.0, with_boundary=True)
    free = torch.stack([collision_free(occ[b], p512[b])
                        for b in range(p512.shape[0])])
    rmse = ((p128 - curve_gt).norm(dim=-1).pow(2).mean(dim=1).sqrt()
            * float(scene_to_meter))
    off_max, off_member = offline_corridor_metrics(p128, batch)
    out = {
        "sdf_max_residual_m": sdf.to_metres(stats["max_residual"]),
        "sdf_violation_m": sdf.to_metres(stats["violation"]),
        "sdf_point_mean_violation_m": sdf.to_metres(stats["mean_violation"]),
        "sdf_feasible": stats["feasible"],
        "sdf_min_distance_m": sdf.to_metres(stats["min_distance"]),
        "sdf_safe_point_rate": stats["safe_point_rate"],
        "sdf_full_violation_m": sdf.to_metres(full["violation"]),
        "collision": ~free.all(dim=1),
        "free_rate": free.to(torch.float32).mean(dim=1),
        "map_excursion": (p512.abs() > 1.0).any(dim=-1).any(dim=-1),
        "rmse_m": rmse,
        "smoothness_m": smoothness_m(p128, scene_to_meter),
        "offline_corridor_max_violation": off_max,
        "offline_corridor_membership_rate": off_member,
        "endpoint_error": ((p512[:, 0] - cond[:, 0]).norm(dim=-1)
                           + (p512[:, -1] - cond[:, 1]).norm(dim=-1)),
    }
    if pack is not None:
        params = torch.linspace(0.0, 1.0, DENSE_POINTS, device=q.device)
        out["online_corridor_max_violation"] = torch.tensor(
            [float(_point_violation(pack, p512[b], params, b))
             for b in range(p512.shape[0])], device=q.device)
        out["online_corridor_membership_rate"] = torch.tensor(
            [_membership_rate(pack, params, p512[b], b)
             for b in range(p512.shape[0])], device=q.device)
    else:
        out["online_corridor_max_violation"] = None
        out["online_corridor_membership_rate"] = None
    return out


# ---------------------------------------------------------------------------
# per-step aggregation (guide 4.4.1 / 4.4.2)
# ---------------------------------------------------------------------------

#: name -> candidate sources, first present one wins.  A source is either a
#: plain trace key or ``("alm_stats", key)`` for a nested corridor-ALM quantity.
#: The nested entries are REQUIRED: the corridor ALM stores its correction-after
#: value only inside ``alm_stats``, so a top-level-only lookup silently produced
#: an all-null "correction after" curve for the paper's own method (the SDF
#: baselines worked, which made the omission easy to miss).
STEP_SERIES = {
    "raw_violation": ("raw_violation", "guard_violation_before",
                      ("alm_stats", "max_violation_before")),
    "safe_violation": (("alm_stats", "max_violation_after"),
                       "guard_violation_after"),
    "correction": ("alm_correction", "guard_correction",
                   ("alm_stats", "max_curve_correction_scene")),
    "smooth_raw": ("smooth_raw", "guard_smoothness_before"),
    "smooth_safe": ("smooth_safe", "guard_smoothness_after"),
}


def _step_values(step: dict, keys, n_expected: int):
    for key in keys:
        if key is None:
            continue
        if isinstance(key, tuple):
            container = step.get(key[0])
            value = container.get(key[1]) if isinstance(container, dict) else None
        else:
            value = step.get(key)
        if value is None:
            continue
        value = torch.as_tensor(value).reshape(-1).float()
        if value.numel() == n_expected:
            return value
    return None


def _step_batch(step: dict):
    """The batch size of ONE trace step (the last chunk of a split is ragged)."""
    for key in ("q0_raw", "q0_safe", "guided", "raw_violation",
                "guard_violation_before"):
        value = step.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.dim() == 0:
                return 1
            return int(value.shape[0])
    return None


def _step_active(step: dict, n_samples: int) -> torch.Tensor:
    """Which samples were actually corrected at this reverse step."""
    guided = step.get("guided")
    active = (torch.as_tensor(guided).reshape(-1).bool()
              if guided is not None and torch.as_tensor(guided).numel() == n_samples
              else torch.zeros(n_samples, dtype=torch.bool))
    for key in ("guard_violation_after", "guard_violation_before"):
        value = step.get(key)
        if value is not None:
            value = torch.as_tensor(value).reshape(-1)
            if value.numel() == n_samples:
                active = active | torch.isfinite(value)
    return active


def aggregate_steps(traces, n_samples: int | None = None) -> dict:
    """Mean / IQR per EXECUTED reverse step (index 1..len), the 4.4 curves.

    ``n_samples`` is only a hint; the authoritative batch size is taken from
    EACH STEP, because the final chunk of a split is ragged (420 = 13x32 + 4).
    Using one campaign-wide mask silently dropped that chunk and then crashed on
    the corridor-ALM diagnostics.
    """
    names = list(STEP_SERIES) + ["feasible_rate", "lambda_max", "inner_steps"]
    series = {name: {"mean": [], "p25": [], "p75": [], "n": []} for name in names}
    series["active_n"] = []
    max_len = max((len(t) for t in traces), default=0)
    for index in range(max_len):
        steps = [t[index] for t in traces if len(t) > index]
        if not steps:
            continue
        active_total = 0
        chunks = {name: [] for name in STEP_SERIES}
        lams, inners, feasible = [], [], []
        for step in steps:
            n_step = _step_batch(step)
            if not n_step:
                continue
            active = _step_active(step, n_step)
            if not bool(active.any()):
                continue
            active_total += int(active.sum())
            for name, keys in STEP_SERIES.items():
                value = _step_values(step, keys, n_step)
                if value is not None:
                    chunks[name].append(value[active])
            alm = step.get("alm_stats")
            if alm is not None:
                for key, sink in (("lambda_max", lams),
                                  ("inner_steps_used", inners)):
                    value = torch.as_tensor(alm[key]).reshape(-1).float()
                    if value.numel() == n_step:
                        sink.append(value[active])
                    elif value.numel():
                        # a scalar diagnostic, not per-sample: keep it once
                        sink.append(value.max().reshape(1))
                after = torch.as_tensor(alm["max_violation_after"]).reshape(-1).float()
                if after.numel() == n_step:
                    feasible.append((after[active] <= FEASIBLE_EPS)
                                    .to(torch.float32).mean().item())
            guard = step.get("guard_violation_after")
            if guard is not None:
                guard = torch.as_tensor(guard).reshape(-1).float()
                if guard.numel() == n_step:
                    feasible.append((guard[active] <= FEASIBLE_EPS)
                                    .to(torch.float32).mean().item())
        series["active_n"].append(active_total)
        for name in STEP_SERIES:
            if not chunks[name]:
                series[name]["mean"].append(None)
                series[name]["p25"].append(None)
                series[name]["p75"].append(None)
                series[name]["n"].append(0)
                continue
            merged = torch.cat(chunks[name])
            series[name]["mean"].append(float(merged.mean()))
            series[name]["p25"].append(float(merged.quantile(0.25)))
            series[name]["p75"].append(float(merged.quantile(0.75)))
            series[name]["n"].append(int(merged.numel()))
        series["lambda_max"]["mean"].append(
            float(torch.cat(lams).mean()) if lams else None)
        series["lambda_max"]["p25"].append(None)
        series["lambda_max"]["p75"].append(None)
        series["lambda_max"]["n"].append(int(sum(v.numel() for v in lams)))
        series["inner_steps"]["mean"].append(
            float(torch.cat(inners).mean()) if inners else None)
        series["inner_steps"]["p25"].append(None)
        series["inner_steps"]["p75"].append(None)
        series["inner_steps"]["n"].append(int(sum(v.numel() for v in inners)))
        series["feasible_rate"]["mean"].append(_mean(feasible))
        series["feasible_rate"]["p25"].append(None)
        series["feasible_rate"]["p75"].append(None)
        series["feasible_rate"]["n"].append(len(feasible))
    series["step_index"] = list(range(1, max_len + 1))
    return series


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_160k4p_c48_oneshot.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--method", required=True, help="name used in the tables")
    ap.add_argument("--guidance", default="alm",
                    choices=["none", "penalty", "projection", "primal_dual",
                             "alm_sdf", "alm"])
    ap.add_argument("--feedback", default=None, choices=["on", "off"],
                    help="history safety feedback (paper method only)")
    ap.add_argument("--final-project", default=None, choices=["on", "off"],
                    help="terminal L1 corridor projection (paper method only)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--alm-set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--sdf-set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep-assert-shapes", action="store_true",
                    help="keep the per-forward shape/finiteness asserts "
                         "(they cost ~18% of the wall clock at inference)")
    ap.add_argument("--allow-partial-load", action="store_true",
                    help="accept a checkpoint whose keys do not match the "
                         "configuration (default: refuse)")
    args = ap.parse_args()

    device = torch.device(args.device
                          or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_config(args.config)
    if args.processed:
        cfg.setdefault("data", {})["processed_root"] = args.processed
    processed = cfg["data"]["processed_root"]
    scene_to_meter = float(cfg["data"].get("scene_to_meter", 80.0))

    alm_cfg = dict(cfg.get("alm") or {})
    corridor_cfg = dict(cfg.get("corridor") or {})
    _apply_overrides(alm_cfg, args.alm_set, "alm-set")
    if args.feedback is not None:
        # guide 4.3.1: force feedback_valid = 0 while KEEPING FeedbackEncoder
        # and its weights in the graph (never unload the module).
        alm_cfg["force_feedback_valid_zero"] = (args.feedback == "off")
    if args.final_project is not None:
        alm_cfg["final_project"] = (args.final_project == "on")
    sdf_cfg: dict = {}
    _apply_overrides(sdf_cfg, args.sdf_set, "sdf-set")
    alm_cfg["guidance"] = args.guidance
    warmup = int(alm_cfg.get("warmup_reverse_steps", DEFAULT_WARMUP))

    limit = None if args.samples in (0, None) else int(args.samples)
    dataset = CarlaSplineDataset(args.split, processed, geometry_points=1280,
                                 limit=limit)
    collate = make_collate(dataset)
    n_total = len(dataset)
    dcfg = cfg["diffusion"]
    schedule = NoiseSchedule(
        int(dcfg.get("timesteps", 16)),
        beta_schedule=dcfg.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=float(dcfg.get("beta_start", 1e-4)),
        beta_end=float(dcfg.get("beta_end", 0.02)))
    model, _, load_report = load_model(cfg, args.ckpt, device=device,
                                       verbose=False)
    if not args.keep_assert_shapes and getattr(model, "assert_shapes", False):
        # ``model.assert_shapes: true`` re-checks every tensor shape/finiteness
        # on EVERY forward, and each check forces a CUDA synchronisation:
        # measured 0.50 s of a 2.8 s run (18 %).  It is a development-time
        # invariant check, not part of the method -> off for the campaign run.
        model.assert_shapes = False
        print("[ch4] model.assert_shapes disabled for inference "
              "(pass --keep-assert-shapes to keep the dev-time checks)",
              flush=True)
    # A checkpoint silently loaded into a DIFFERENT architecture (e.g. the
    # D-C48 config with an A-C48 checkpoint: 2-stream vs 3-stream fusion) leaves
    # most weights at their random init and still "loads" fine, which would
    # quietly produce garbage numbers.  Refuse it unless explicitly allowed.
    if load_report is not None and not args.allow_partial_load:
        missing = list(load_report.get("missing") or [])
        adapted = list(load_report.get("adapted") or [])
        unexpected = list(load_report.get("unexpected") or [])
        if missing or adapted or unexpected:
            raise SystemExit(
                "[ch4] checkpoint/config MISMATCH for %s | config=%s | "
                "missing=%d %s | shape-adapted=%d %s | unexpected=%d %s | "
                "(pass --allow-partial-load only if this is intended)"
                % (args.ckpt, args.config, len(missing), missing[:6],
                   len(adapted), adapted[:4], len(unexpected), unexpected[:6]))

    slug = "%s__%s__s%d" % (args.method, args.guidance, args.seed)
    out_path = args.out or os.path.join(ROOT, "outputs", "ch4", slug + ".json")
    md_path = args.md or out_path[:-5] + ".md"
    rec_path = args.records or out_path[:-5] + ".jsonl"
    for path in (out_path, md_path, rec_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(rec_path):
        os.remove(rec_path)

    experiment_id = "%s+%s%s" % (args.method, args.guidance, args.tag)
    header = {
        "experiment_id": experiment_id,
        "method": args.method,
        "guidance": args.guidance,
        "checkpoint": args.ckpt,
        "checkpoint_hash": _file_hash(args.ckpt),
        "model_dir": os.path.basename(os.path.dirname(args.ckpt)),
        "config": args.config,
        "git_commit": _git_commit(),
        "split": args.split,
        "n_samples": n_total,
        "chunk": args.chunk,
        "steps": args.steps,
        "seed": args.seed,
        "warmup_reverse_steps": warmup,
        "feedback": (args.feedback if args.guidance == "alm" else None),
        "final_project": (args.final_project if args.guidance == "alm" else None),
        "scene_to_meter": scene_to_meter,
        "dense_points": DENSE_POINTS,
        "feasible_eps": FEASIBLE_EPS,
        "alm_overrides": args.alm_set,
        "sdf_overrides": args.sdf_set,
        "config_snapshot": {k: cfg[k] for k in ("data", "model", "bspline",
                                                "diffusion", "alm", "corridor",
                                                "loss", "train")
                            if k in cfg},
        "device": str(device),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("[ch4] %s | guidance=%s | %s n=%d chunk=%d steps=%d seed=%d warmup=%d"
          % (args.method, args.guidance, args.split, n_total, args.chunk,
             args.steps, args.seed, warmup), flush=True)

    num_cols = ("collision", "free_rate", "map_excursion", "rmse_m", "smoothness_m",
                "sdf_max_residual_m", "sdf_violation_m",
                "sdf_point_mean_violation_m", "sdf_feasible", "sdf_min_distance_m",
                "sdf_safe_point_rate", "sdf_full_violation_m",
                "offline_corridor_max_violation",
                "offline_corridor_membership_rate", "endpoint_error")
    acc = {k: [] for k in num_cols}
    acc_bool = {"collision": [], "sdf_feasible": [], "map_excursion": []}
    online_max, online_member = [], []
    per_traj_ms, traces = [], []
    status_counter: Counter = Counter()
    failing_samples = 0
    timing = {"network_s": 0.0, "corridor_s": 0.0, "guard_s": 0.0,
              "final_projection_s": 0.0, "sdf_build_s": 0.0, "wall_s": 0.0}
    gt_acc = {k: [] for k in ("collision", "rmse_m", "smoothness_m",
                              "sdf_max_residual_m", "sdf_violation_m",
                              "sdf_min_distance_m", "sdf_safe_point_rate")}
    n_chunks = 0
    timed_chunks = 0
    started = time.time()

    for start in range(0, n_total, args.chunk):
        rows = list(range(start, min(start + args.chunk, n_total)))
        batch = collate([dataset[i] for i in rows])
        occ = batch["occupancy"].to(device)
        cond = batch["cond"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall0 = time.perf_counter()
        out = sample(model, schedule, cond, occ,
                     batch["candidate_xy"].to(device),
                     batch["candidate_mask"].to(device),
                     batch["candidate_geometry"].to(device),
                     batch["candidate_geometry_lengths"].to(device),
                     device=str(device), steps=args.steps, seed=args.seed,
                     seed_offset=start, return_trace=True,
                     alm_config=alm_cfg, corridor_config=corridor_cfg,
                     guidance=args.guidance, sdf_config=sdf_cfg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - wall0

        sdf = build_occupancy_sdf(occ, scene_to_meter=scene_to_meter,
                                  map_margin=float(sdf_cfg.get("map_margin",
                                                               alm_cfg.get("map_margin", 0.0))))
        metrics = evaluate_controls(model.bspline, out["control"], occ, cond,
                                    batch["pos"].to(device), sdf, out.get("pack"),
                                    batch, scene_to_meter)
        gt = evaluate_controls(model.bspline, batch["control_gt"].to(device), occ,
                               cond, batch["pos"].to(device), sdf, None, batch,
                               scene_to_meter)
        for key in gt_acc:
            gt_acc[key].extend(_tolist(gt[key]))
        for key in num_cols:
            acc[key].extend(_tolist(metrics[key]))
        for key in acc_bool:
            acc_bool[key].extend([bool(v) for v in
                                  torch.as_tensor(metrics[key]).reshape(-1)])
        if metrics["online_corridor_max_violation"] is not None:
            online_max.extend(_tolist(metrics["online_corridor_max_violation"]))
            online_member.extend(_tolist(metrics["online_corridor_membership_rate"]))

        n_chunks += 1
        if n_chunks > TIMING_WARMUP_CHUNKS:
            per_traj_ms.append(wall * 1000.0 / len(rows))
            timed_chunks += 1
        for key in timing:
            timing[key] += float(out.get("timing", {}).get(key, 0.0))
        timing["wall_s"] += wall
        traces.append(out.get("trace") or [])   # 4.4 process curves

        frames = out.get("guard_frames") or []
        for frame in frames:
            status_counter.update(frame["status"])
        bad = 0
        for b in range(len(rows)):
            statuses = [frame["status"][b] for frame in frames]
            if any(s not in ("penalty_ok", "ok", "primal_dual_ok", "alm_sdf_ok",
                             "not_needed", "skipped") for s in statuses):
                bad += 1
        failing_samples += bad

        steps_done = torch.as_tensor(out.get("sdf_guided_steps",
                                             torch.zeros(len(rows)))).reshape(-1)
        for local, index in enumerate(rows):
            record = {
                "experiment_id": experiment_id,
                "model_id": header["model_dir"],
                "checkpoint_hash": header["checkpoint_hash"],
                "git_commit": header["git_commit"],
                "seed": args.seed,
                "sample_id": int(batch["sample_id"][local]),
                "index": index,
                "split": args.split,
                "method": args.method,
                "guidance": args.guidance,
                "collision": bool(metrics["collision"][local]),
                "max_constraint_residual": float(metrics["sdf_max_residual_m"][local]),
                "nonnegative_violation": float(metrics["sdf_violation_m"][local]),
                "feasible": bool(metrics["sdf_feasible"][local]),
                "sdf_min_distance_m": float(metrics["sdf_min_distance_m"][local]),
                "sdf_max_residual_m": float(metrics["sdf_max_residual_m"][local]),
                "sdf_safe_point_rate": float(metrics["sdf_safe_point_rate"][local]),
                "offline_corridor_membership_rate": float(
                    metrics["offline_corridor_membership_rate"][local]),
                "rmse_m": float(metrics["rmse_m"][local]),
                "smoothness_m": float(metrics["smoothness_m"][local]),
                "runtime_ms": (per_traj_ms[-1] if per_traj_ms else None),
                "guided_steps": int(steps_done[local]) if steps_done.numel() else 0,
                "solver_status": (";".join(sorted({f["status"][local]
                                                   for f in frames}))
                                  if frames else None),
                "corridor_guided": bool(out["guided"][local]),
            }
            _write_jsonl(rec_path, record)

        print("  [%s] %d/%d collisions=%d rmse=%.2fm"
              % (args.method, rows[-1] + 1, n_total,
                 sum(acc_bool["collision"]), _mean(acc["rmse_m"]) or -1.0),
              flush=True)
        del out, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = len(acc["collision"])
    collisions = int(sum(acc_bool["collision"]))
    feasible_rate = float(np.mean(acc_bool["sdf_feasible"])) if n else None
    aggregate = {
        "n": n,
        "collisions": collisions,
        "collision_rate": (collisions / n) if n else None,
        "free_rate": _mean(acc["free_rate"]),
        "map_excursion_rate": (float(np.mean(acc_bool["map_excursion"])) if n else None),
        "sdf_feasible_rate": feasible_rate,
        "sdf_mean_violation_m": _mean(acc["sdf_violation_m"]),
        "sdf_p95_violation_m": _pct(acc["sdf_violation_m"], 95),
        "sdf_max_violation_m": max(acc["sdf_violation_m"]) if n else None,
        "sdf_point_mean_violation_m": _mean(acc["sdf_point_mean_violation_m"]),
        "sdf_max_residual_m": _mean(acc["sdf_max_residual_m"]),
        "sdf_min_distance_m": _mean(acc["sdf_min_distance_m"]),
        "sdf_safe_point_rate": _mean(acc["sdf_safe_point_rate"]),
        "sdf_full_violation_m": _mean(acc["sdf_full_violation_m"]),
        "offline_corridor_max_violation": _mean(acc["offline_corridor_max_violation"]),
        "offline_corridor_membership_rate": _mean(acc["offline_corridor_membership_rate"]),
        "online_corridor_max_violation": _mean(online_max),
        "online_corridor_membership_rate": _mean(online_member),
        "rmse_m": _mean(acc["rmse_m"]),
        "smoothness_m": _mean(acc["smoothness_m"]),
        "endpoint_error": _mean(acc["endpoint_error"]),
        "runtime_ms_mean": _mean(per_traj_ms),
        "runtime_ms_std": _std(per_traj_ms),
        "timed_chunks": timed_chunks,
        # NOTE: every value here is MILLISECONDS PER TRAJECTORY (the raw
        # accumulator below is in seconds for the whole run).  The keys end in
        # ``_ms`` on purpose: an earlier ``network_s``-style naming made a
        # correct 95.2 ms/traj read as "95 seconds" during review.
        "timing_ms_per_trajectory": {
            _ms_key(key): (1000.0 * value / n if n else None)
            for key, value in timing.items()},
        "solver_status": dict(status_counter),
        "solver_failure_samples": failing_samples,
        "wall_seconds": time.time() - started,
        "gt_reference": {
            "collision_rate": (float(np.mean(gt_acc["collision"]))
                               if gt_acc["collision"] else None),
            "rmse_m": _mean(gt_acc["rmse_m"]),
            "smoothness_m": _mean(gt_acc["smoothness_m"]),
            "sdf_max_residual_m": _mean(gt_acc["sdf_max_residual_m"]),
            "sdf_violation_m": _mean(gt_acc["sdf_violation_m"]),
            "sdf_min_distance_m": _mean(gt_acc["sdf_min_distance_m"]),
            "sdf_safe_point_rate": _mean(gt_acc["sdf_safe_point_rate"]),
        },
    }
    payload = dict(header)
    payload["aggregate"] = aggregate
    payload["per_step"] = aggregate_steps(traces, args.chunk)
    payload["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    print("[ch4] %s | coll=%d/%d (%.2f%%) | sdf feas=%.2f%% | mean viol=%.4f m | "
          "rmse=%.2f m | smooth=%.3f m | %.1f ms/traj"
          % (args.method, collisions, n, 100.0 * (collisions / n if n else 0),
             100.0 * (feasible_rate or 0.0), aggregate["sdf_mean_violation_m"] or 0.0,
             aggregate["rmse_m"] or 0.0, aggregate["smoothness_m"] or 0.0,
             aggregate["runtime_ms_mean"] or 0.0), flush=True)
    print("[ch4] wrote %s" % out_path, flush=True)
    print("[ch4] wrote %s" % md_path, flush=True)
    print("[ch4] wrote %s (%d records)" % (rec_path, n), flush=True)
    return 0


def render_markdown(payload: dict) -> str:
    agg = payload["aggregate"]
    lines = [
        "# %s" % payload["experiment_id"],
        "",
        "- checkpoint: `%s` (sha1 %s)" % (payload["checkpoint"],
                                          payload["checkpoint_hash"]),
        "- config: `%s` | git %s" % (payload["config"], payload["git_commit"]),
        "- protocol: split=%s n=%d chunk=%d steps=%d seed=%d warmup=%d "
        "dense_points=%d" % (payload["split"], payload["n_samples"],
                             payload["chunk"], payload["steps"],
                             payload["seed"], payload["warmup_reverse_steps"],
                             payload["dense_points"]),
        "- guidance=%s feedback=%s final_project=%s"
        % (payload["guidance"], payload["feedback"], payload["final_project"]),
        "",
        "| collisions | coll rate | SDF feas | SDF mean viol (m) | SDF P95 viol (m) | "
        "safe-point | min dist (m) | RMSE (m) | smooth (m) | ms/traj |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| %d/%d | %.2f%% | %.2f%% | %.4f | %.4f | %.2f%% | %.3f | %.2f | %.3f | %.1f |"
        % (agg["collisions"], agg["n"],
           100.0 * (agg["collision_rate"] or 0.0),
           100.0 * (agg["sdf_feasible_rate"] or 0.0),
           agg["sdf_mean_violation_m"] or 0.0,
           agg["sdf_p95_violation_m"] or 0.0,
           100.0 * (agg["sdf_safe_point_rate"] or 0.0),
           agg["sdf_min_distance_m"] or 0.0,
           agg["rmse_m"] or 0.0,
           agg["smoothness_m"] or 0.0,
           agg["runtime_ms_mean"] or 0.0),
        "",
        "GT reference: coll %.2f%% | RMSE %.2f m | smooth %.3f m | min dist %.3f m"
        % (100.0 * (agg["gt_reference"]["collision_rate"] or 0.0),
           agg["gt_reference"]["rmse_m"] or 0.0,
           agg["gt_reference"]["smoothness_m"] or 0.0,
           agg["gt_reference"]["sdf_min_distance_m"] or 0.0),
        "",
        "Timing per trajectory (ms): %s"
        % ", ".join("%s=%.1f" % (k, v or 0.0)
                    for k, v in agg["timing_ms_per_trajectory"].items()),
        "Solver status: %s | failure samples: %d"
        % (agg["solver_status"], agg["solver_failure_samples"]),
        "",
    ]
    step = payload["per_step"]
    if step.get("step_index"):
        lines += ["## 逐步过程（4.4）", "",
                  "| step | active n | raw viol | safe viol | correction | "
                  "smooth raw | smooth safe | feasible | lambda max | inner |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

        def cell(name, key, fmt="%.5f"):
            values = step.get(name, {}).get(key) or []
            return [fmt % v if v is not None else "-" for v in values]

        raw = cell("raw_violation", "mean")
        safe = cell("safe_violation", "mean")
        corr = cell("correction", "mean")
        sraw = cell("smooth_raw", "mean")
        ssafe = cell("smooth_safe", "mean")
        feas = cell("feasible_rate", "mean", "%.3f")
        lam = cell("lambda_max", "mean", "%.4f")
        inner = cell("inner_steps", "mean", "%.2f")
        for i, index in enumerate(step["step_index"]):
            lines.append("| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
                         % (index, step["active_n"][i], raw[i], safe[i], corr[i],
                            sraw[i], ssafe[i], feas[i], lam[i], inner[i]))
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
