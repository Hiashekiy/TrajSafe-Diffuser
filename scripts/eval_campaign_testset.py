"""eval_campaign_testset.py - same-protocol FULL-SPLIT evaluation of checkpoints.

Every ``--model NAME=ckpt[::config]`` is rebuilt, run through the FULL inference
loop (DDIM + frozen safety corridor + per-step B-spline ALM) on the SAME split,
the SAME samples, the SAME seed and the SAME chunk boundaries, and compared on:

  * final validation - collision rate, dense constraint violation, corridor
    membership rate, endpoint error (``dense_validation``);
  * the model-INDEPENDENT check: the final curve against the OFFLINE GT corridor
    (``alm_cell_*.npy``), identical for every model;
  * the per-reverse-step series of report section 18 - raw constraint violation,
    ALM correction magnitude and decoded-curve smoothness before/after the ALM;
  * the feedback history and the decoded-curve RMSE against the GT curve (m).

The whole test split (420 samples) does NOT fit in one batch, so it is evaluated
in chunks (``--chunk``, default 32).  ``sample()`` re-seeds per call, which means
every chunk replays the SAME noise draw - and since all models share the chunk
boundaries, each sample gets the identical starting noise for every model: the
comparison is PAIRED (much lower variance than independent draws).  Aggregates
are accumulated with correct weights: rates over samples, RMSE from the summed
squared error, per-step series from summed guided values.

``--ablation A`` switches the corridor + ALM off (raw diffusion) and answers
"can this model be deployed without the ALM?".

Usage:

    python scripts/eval_campaign_testset.py --config configs/config_160k8p.yaml \
        --split test --samples 0 --chunk 32 --steps 16 --seed 0 \
        --model A_oneshot=outputs/campaign_a_oneshot/ckpt/best_task.pt \
        --model B2_feedback=outputs/campaign_b2_feedback/ckpt/best_task.pt \
        --model "B1_base=outputs/campaign_b1_base/ckpt/best_task.pt::configs/config_160k8p_s1.yaml" \
        --model REF_160k8=outputs/bspline_carla_160k8/ckpt/best_task.pt \
        --out outputs/campaign_testset_eval.json \
        --md  outputs/campaign_testset_eval.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.diffusion.sampler import ablation_configs, sample  # noqa: E402
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,  # noqa: E402
                                               make_collate)

STEP_KEYS = ("raw_violation", "alm_correction", "smooth_raw", "smooth_safe")


def parse_model(spec: str, default_config: str):
    """``NAME=ckpt[::config]`` -> (name, ckpt, config)."""
    if "=" not in spec:
        raise SystemExit("[eval] --model must be NAME=ckpt[::config], got %r"
                         % spec)
    name, rest = spec.split("=", 1)
    ckpt, _, config = rest.partition("::")
    return name.strip(), ckpt.strip(), (config.strip() or default_config)


def offline_corridor_metrics(p: torch.Tensor, batch: dict):
    """Model-INDEPENDENT check: the final curve against the OFFLINE GT corridor.

    The sampler builds the corridor from each model's OWN predicted ellipses, so
    ``final_max_constraint_violation`` is measured against a different corridor
    per model (the deployed protocol, but not a common yardstick).  The offline
    cache (``alm_cell_a/b/valid``, built from the GT route) is IDENTICAL for
    every model.  Returns per-sample ``(max_violation, point_membership)`` lists
    (``None`` where the sample has no offline corridor).
    """
    cell_a = batch["alm_cell_a"].detach().float()
    cell_b = batch["alm_cell_b"].detach().float()
    cell_valid = batch["alm_cell_valid"].detach().bool()
    alm_ok = batch["alm_valid"].detach().bool()
    b = int(p.shape[0])
    worst_out = [None] * b
    member_out = [None] * b
    if not bool(alm_ok.any()):
        return worst_out, member_out
    nrm = cell_a.norm(dim=-1).clamp_min(1e-9)                    # [B,C,F]
    signed = (torch.einsum("bcfk,bhk->bhcf", cell_a, p)
              - cell_b[:, None]) / nrm[:, None]                  # [B,H,C,F]
    worst = signed.amax(dim=-1)                                  # [B,H,C]
    worst = worst.masked_fill(~cell_valid[:, None, :], float("inf"))
    violation = torch.nan_to_num(worst.amin(dim=-1), nan=0.0, posinf=0.0,
                                 neginf=0.0)                      # [B,H]
    per_max = violation.max(dim=1).values
    per_member = (violation <= 0).float().mean(dim=1)
    for i in range(b):
        if bool(alm_ok[i]):
            worst_out[i] = float(per_max[i])
            member_out[i] = float(per_member[i])
    return worst_out, member_out


def guided_series(trace, key):
    """Per-step mean over the GUIDED samples of one trace key."""
    out = []
    for step in trace:
        value = step.get(key)
        guided = step.get("guided")
        if value is None or guided is None:
            out.append(None)
            continue
        mask = torch.as_tensor(guided).bool()
        if not bool(mask.any()):
            out.append(None)
            continue
        v = torch.as_tensor(value).reshape(-1).float()
        out.append(float(v[mask].mean()))
    return out


def _cast(value: str):
    """CLI override value -> int / float / bool / str."""
    low = str(value).strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return str(value)


def evaluate_chunk(config, ckpt, batch, schedule, device, steps, seed,
                   ablation=None, alm_overrides=None):
    """Run ONE chunk; return per-sample raw numbers (aggregation is separate)."""
    cfg = load_config(config)
    alm_cfg, corridor_cfg = cfg.get("alm"), cfg.get("corridor")
    if ablation:
        alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg,
                                                 ablation)
    if alm_overrides:
        alm_cfg = dict(alm_cfg or {})
        for item in alm_overrides:
            key, _, value = str(item).partition("=")
            if not key:
                raise SystemExit("[eval] --alm-set must be KEY=VALUE")
            alm_cfg[key] = _cast(value)
    model, _, _ = load_model(cfg, ckpt, device=device, verbose=False)
    out = sample(model, schedule, batch["cond"], batch["occupancy"],
                 batch["candidate_xy"], batch["candidate_mask"],
                 batch["candidate_geometry"],
                 batch["candidate_geometry_lengths"], device=device,
                 steps=steps, seed=seed, return_trace=True,
                 alm_config=alm_cfg, corridor_config=corridor_cfg)
    p = out["p"].detach().cpu()
    gt = batch["pos"].detach().cpu()
    err = torch.linalg.norm(p - gt, dim=-1)                      # [B,H]
    off_max, off_member = offline_corridor_metrics(p, batch)
    fv = out["final_validation"]
    return {
        "n": int(p.shape[0]),
        "collision": [bool(v.get("final_collision")) for v in fv],
        "free_rate": [float(v.get("final_free_rate")) for v in fv],
        "max_violation": [v.get("final_max_constraint_violation") for v in fv],
        "membership": [v.get("final_corridor_membership_rate") for v in fv],
        "feasible": [v.get("final_constraint_feasible") for v in fv],
        "endpoint_error": [float(v.get("endpoint_error")) for v in fv],
        "se_sum": float(err.pow(2).sum()), "se_count": int(err.numel()),
        "off_max": off_max, "off_member": off_member,
        "fb_history": dict(out["feedback"]["history"]),
        "guided": [bool(v) for v in out["guided"].detach().cpu().tolist()],
        "alm_status": Counter(out["alm_status"]),
        "fb_valid": out["feedback"]["valid"].detach().cpu().tolist(),
        "series": {k: guided_series(out["trace"], k) for k in STEP_KEYS},
        "elapsed_s": None,
    }


def acc_new():
    return {
        "n": 0, "collision": [], "free_rate": [], "max_violation": [],
        "membership": [], "feasible": [], "endpoint_error": [],
        "off_max": [], "off_member": [], "guided": [], "fb_valid": [],
        "se_sum": 0.0, "se_count": 0,
        "fb_history": Counter(), "alm_status": Counter(),
        "series_sum": {k: [] for k in STEP_KEYS},
        "series_cnt": {k: [] for k in STEP_KEYS},
    }


def acc_add(acc, raw):
    acc["n"] += raw["n"]
    for key in ("collision", "free_rate", "max_violation", "membership",
                "feasible", "endpoint_error", "off_max", "off_member",
                "guided", "fb_valid"):
        acc[key].extend(raw[key])
    acc["se_sum"] += raw["se_sum"]
    acc["se_count"] += raw["se_count"]
    acc["fb_history"].update(raw["fb_history"])
    acc["alm_status"].update(raw["alm_status"])
    for k in STEP_KEYS:
        sums, cnts = acc["series_sum"][k], acc["series_cnt"][k]
        for i, value in enumerate(raw["series"][k]):
            while len(sums) <= i:
                sums.append(0.0)
                cnts.append(0)
            if value is not None:
                sums[i] += value
                cnts[i] += 1
    return acc


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def acc_finalize(acc, scene_to_meter):
    res = {
        "samples": acc["n"],
        "collisions": int(sum(1 for v in acc["collision"] if v)),
        "collision_rate": _mean(acc["collision"]),
        "free_rate": _mean(acc["free_rate"]),
        "curve_rmse_m": (float(np.sqrt(acc["se_sum"] / max(acc["se_count"], 1)))
                         * scene_to_meter),
        "final_max_constraint_violation": _mean(acc["max_violation"]),
        "corridor_membership_rate": _mean(acc["membership"]),
        "constraint_feasible_rate": _mean(acc["feasible"]),
        "endpoint_error": _mean(acc["endpoint_error"]),
        "offline_corridor_max_violation": _mean(acc["off_max"]),
        "offline_corridor_membership_rate": _mean(acc["off_member"]),
        "guided_rate": _mean(acc["guided"]),
        "feedback_valid_rate": _mean(acc["fb_valid"]),
        "feedback_history": dict(acc["fb_history"]),
        "alm_status": dict(acc["alm_status"]),
    }
    for k in STEP_KEYS:
        sums, cnts = acc["series_sum"][k], acc["series_cnt"][k]
        res["per_step_" + k] = [None if cnts[i] == 0 else sums[i] / cnts[i]
                                for i in range(len(sums))]
        res["per_step_" + k + "_n"] = list(cnts)
    for key in ("collision", "max_violation", "membership", "off_max",
                "off_member", "free_rate", "endpoint_error"):
        res["per_sample_" + key] = acc[key]
    raw = [(i, v) for i, v in enumerate(res["per_step_raw_violation"])
           if v is not None]
    corr = [(i, v) for i, v in enumerate(res["per_step_alm_correction"])
            if v is not None]
    res["first_guided_raw_violation"] = raw[0][1] if raw else None
    res["last_guided_raw_violation"] = raw[-1][1] if raw else None
    res["mean_guided_raw_violation"] = _mean([v for _, v in raw])
    res["first_guided_alm_correction"] = corr[0][1] if corr else None
    res["last_guided_alm_correction"] = corr[-1][1] if corr else None
    res["mean_guided_alm_correction"] = _mean([v for _, v in corr])
    return res


def _fmt(x, nd=4):
    return "-" if x is None else ("%.*f" % (nd, float(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_160k8p.yaml")
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=ckpt[::config]; repeat for several models")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=0,
                    help="0 = the WHOLE split (default)")
    ap.add_argument("--chunk", type=int, default=32,
                    help="samples per sampler call (memory; all models share "
                         "the chunk boundaries so the noise draw is paired)")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ablation", default=None, choices=["A", "B", "C", "D"],
                    help="A = raw diffusion (no corridor, no ALM)")
    ap.add_argument("--alm-set", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override one alm.* value (repeatable), e.g. "
                         "warmup_reverse_steps=1.  Warm-up is counted in "
                         "EXECUTED reverse forwards, so a short --steps schedule "
                         "must lower it too or the ALM barely runs.")
    ap.add_argument("--out", default="outputs/campaign_testset_eval.json")
    ap.add_argument("--md", default="outputs/campaign_testset_eval.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scene_to_meter = float(data_cfg.get("scene_to_meter", 40.0))
    ds = CarlaSplineDataset(args.split, data_cfg.get("processed_root"),
                            geometry_points=1280,
                            limit=(int(args.samples) if args.samples else None),
                            num_controls=num_controls(cfg),
                            num_safety_queries=num_safety_queries(cfg))
    n, chunk = len(ds), max(1, int(args.chunk))
    print("[eval] split=%s samples=%d chunk=%d steps=%d seed=%s ablation=%s "
          "device=%s" % (args.split, n, chunk, args.steps, args.seed,
                         args.ablation, device), flush=True)
    schedule = NoiseSchedule(
        cfg["diffusion"]["timesteps"],
        beta_schedule=cfg["diffusion"].get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=cfg["diffusion"].get("beta_start", 1e-4),
        beta_end=cfg["diffusion"].get("beta_end", 0.02)).to(device)

    specs = []
    for spec in args.model:
        name, ckpt, mcfg = parse_model(spec, args.config)
        path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
        if not os.path.exists(path):
            print("[eval] SKIP %s: %s not found" % (name, ckpt), flush=True)
            continue
        specs.append((name, ckpt, mcfg))

    collate = make_collate(ds)
    results = []
    for name, ckpt, mcfg in specs:
        acc = acc_new()
        for lo in range(0, n, chunk):
            sub = collate([ds[i] for i in range(lo, min(lo + chunk, n))])
            acc_add(acc, evaluate_chunk(mcfg, ckpt, sub, schedule, device,
                                        args.steps, args.seed,
                                        ablation=args.ablation,
                                        alm_overrides=args.alm_set))
            print("  [%s] %d/%d samples, collisions=%d"
                  % (name, acc["n"], n,
                     sum(1 for v in acc["collision"] if v)), flush=True)
        res = acc_finalize(acc, scene_to_meter)
        res.update({"name": name, "ckpt": ckpt, "config": mcfg,
                    "ablation": args.ablation, "split": args.split,
                    "steps": args.steps, "seed": args.seed,
                    "alm_overrides": list(args.alm_set)})
        results.append(res)
        print("[eval] %-12s n=%d coll=%.4f (%d) rmse=%.2fm | OFFv=%s OFFm=%s "
              "| viol=%s member=%s | guided=%.2f fb_valid=%.3f"
              % (name, res["samples"], res["collision_rate"] or 0.0,
                 res["collisions"], res["curve_rmse_m"],
                 _fmt(res["offline_corridor_max_violation"]),
                 _fmt(res["offline_corridor_membership_rate"]),
                 _fmt(res["final_max_constraint_violation"]),
                 _fmt(res["corridor_membership_rate"]),
                 res["guided_rate"] or 0.0,
                 res["feedback_valid_rate"] or 0.0), flush=True)

    out_path = args.out if os.path.isabs(args.out) \
        else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"split": args.split, "samples": n, "chunk": chunk,
                   "steps": args.steps, "seed": args.seed,
                   "ablation": args.ablation, "config": args.config,
                   "results": results}, fh, indent=2, ensure_ascii=False)
    print("[eval] written %s" % out_path)

    md_path = args.md if os.path.isabs(args.md) else os.path.join(ROOT, args.md)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    tag = ("（ALM 关 / ablation %s）" % args.ablation) if args.ablation \
        else "（ALM 开）"
    lines = ["# 160k8p campaign: 同口径 DDIM + ALM 采样评估 %s" % tag, "",
             "* split `%s`，**%d 样本（全量）**，chunk %d，%d reverse steps，"
             "seed %d（各模型配对同一噪声）"
             % (args.split, n, chunk, args.steps, args.seed), "",
             "| model | collisions | coll rate | rmse_m | OFFLINE viol | "
             "OFFLINE member | sampler viol | sampler member | fb_valid | "
             "guided |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append("| %s | %d/%d | %.4f | %.2f | %s | %s | %s | %s | %.3f "
                     "| %.2f |"
                     % (r["name"], r["collisions"], r["samples"],
                        r["collision_rate"], r["curve_rmse_m"],
                        _fmt(r["offline_corridor_max_violation"]),
                        _fmt(r["offline_corridor_membership_rate"]),
                        _fmt(r["final_max_constraint_violation"]),
                        _fmt(r["corridor_membership_rate"]),
                        r["feedback_valid_rate"] or float("nan"),
                        r["guided_rate"] or 0.0))
    lines += ["", "## 逐 step 曲线（guided step 均值）", ""]
    for r in results:
        lines += ["### %s" % r["name"], "",
                  "| reverse step idx | raw violation | ALM correction | "
                  "curve smooth raw | curve smooth safe | n |",
                  "|---|---|---|---|---|---|"]
        for i in range(len(r["per_step_raw_violation"])):
            if (r["per_step_raw_violation"][i] is None
                    and r["per_step_alm_correction"][i] is None):
                continue
            lines.append("| %d | %s | %s | %s | %s | %d |"
                         % (i, _fmt(r["per_step_raw_violation"][i]),
                            _fmt(r["per_step_alm_correction"][i]),
                            _fmt(r["per_step_smooth_raw"][i], 5),
                            _fmt(r["per_step_smooth_safe"][i], 5),
                            r["per_step_raw_violation_n"][i]))
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[eval] written %s" % md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
