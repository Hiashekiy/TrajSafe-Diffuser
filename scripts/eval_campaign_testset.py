"""eval_campaign_testset.py - same-protocol 16-step ALM evaluation of checkpoints.

This is the experiment that can actually answer "did the historical safety
feedback help?": every ``--model NAME=ckpt[::config]`` is rebuilt, run through
the FULL inference loop (DDIM + frozen safety corridor + per-step B-spline ALM)
on the SAME split, the SAME samples and the SAME seed, and the results are
compared on:

  * final validation  - collision rate, dense constraint violation, corridor
    membership rate, endpoint error (``dense_validation``);
  * the per-reverse-step series of report section 18 - raw constraint violation,
    ALM correction magnitude, decoded-curve smoothness before/after the ALM;
  * the feedback history (accepted / already_safe / rejected) and whether the
    gated feedback branch was ever active;
  * the decoded-curve RMSE against the GT curve, in metres.

Nothing is trained here, so it is safe to run while the GPU is busy (it just
gets slower); use ``--device cpu`` for a plumbing smoke test.

Usage:

    python scripts/eval_campaign_testset.py --config configs/config_160k8p.yaml \
        --split test --samples 16 --steps 16 --seed 0 \
        --model A=outputs/campaign_a_oneshot/ckpt/best_task.pt \
        --model B=outputs/campaign_b2_feedback/ckpt/best_task.pt \
        --model B1=outputs/campaign_b1_base/ckpt/best_task.pt::configs/config_160k8p_s1.yaml \
        --out outputs/campaign_testset_eval.json

``::config`` (optional) overrides the YAML for that model - needed for the
feedback-OFF stages, so their sampler runs exactly the plain-diffusion path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.diffusion.sampler import sample  # noqa: E402
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,  # noqa: E402
                                               make_collate)


def parse_model(spec: str, default_config: str):
    """``NAME=ckpt[::config]`` -> (name, ckpt, config)."""
    if "=" not in spec:
        raise SystemExit("[eval] --model must be NAME=ckpt[::config], got %r"
                         % spec)
    name, rest = spec.split("=", 1)
    ckpt, _, config = rest.partition("::")
    return name.strip(), ckpt.strip(), (config.strip() or default_config)


def mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def series(trace, key):
    """Mean over the GUIDED samples of a per-step trace key."""
    out = []
    for step in trace:
        value = step.get(key)
        if value is None:
            out.append(None)
            continue
        guided = step.get("guided")
        value = torch.as_tensor(value).reshape(-1).float()
        if guided is not None and bool(torch.as_tensor(guided).any()):
            out.append(float(value[torch.as_tensor(guided).bool()].mean()))
        else:
            out.append(None)
    return out


def offline_corridor_metrics(p: torch.Tensor, batch: dict):
    """Model-INDEPENDENT check: the final curve against the OFFLINE GT corridor.

    The sampler builds the corridor from each model's OWN predicted ellipses, so
    ``final_max_constraint_violation`` is measured against a different corridor
    per model (that is the deployed protocol, but not a common yardstick).  The
    offline cache (``alm_cell_a/b/valid``, built from the GT route by
    ``scripts/data/carla_full/04_build_alm_constraints.py``) is IDENTICAL for
    every model, so this is the apples-to-apples number.

    ``violation(p) = min_i max_f (A_if . p - b_if)`` - the same half-space
    semantics as ``losses.alm_corridor_loss``; positive = outside the corridor.
    Returns ``(mean_max_violation, mean_point_membership)`` over the samples
    whose offline corridor closed.
    """
    cell_a = batch["alm_cell_a"].detach().float()
    cell_b = batch["alm_cell_b"].detach().float()
    cell_valid = batch["alm_cell_valid"].detach().bool()
    alm_ok = batch["alm_valid"].detach().bool()
    if not bool(alm_ok.any()):
        return None, None
    nrm = cell_a.norm(dim=-1).clamp_min(1e-9)                    # [B,C,F]
    signed = (torch.einsum("bcfk,bhk->bhcf", cell_a, p)
              - cell_b[:, None]) / nrm[:, None]                  # [B,H,C,F]
    worst = signed.amax(dim=-1)                                  # [B,H,C]
    worst = worst.masked_fill(~cell_valid[:, None, :], float("inf"))
    violation = worst.amin(dim=-1)                               # [B,H]
    violation = torch.nan_to_num(violation, nan=0.0, posinf=0.0, neginf=0.0)
    per_max = violation.max(dim=1).values[alm_ok]
    per_member = (violation <= 0).float().mean(dim=1)[alm_ok]
    return float(per_max.mean()), float(per_member.mean())


def evaluate(name: str, config: str, ckpt: str, batch, schedule, device,
             steps: int, seed: int, scene_to_meter: float,
             ablation: str | None = None) -> dict:
    from src.diffusion.sampler import ablation_configs
    cfg = load_config(config)
    alm_cfg, corridor_cfg = cfg.get("alm"), cfg.get("corridor")
    if ablation:
        # A = raw diffusion (no corridor, no ALM): the model's OWN prediction is
        # what is returned, so this isolates "how good is the network alone".
        alm_cfg, corridor_cfg = ablation_configs(alm_cfg, corridor_cfg,
                                                 ablation)
    model, _, report = load_model(cfg, ckpt, device=device, verbose=False)
    out = sample(model, schedule, batch["cond"], batch["occupancy"],
                 batch["candidate_xy"], batch["candidate_mask"],
                 batch["candidate_geometry"],
                 batch["candidate_geometry_lengths"], device=device,
                 steps=steps, seed=seed, return_trace=True,
                 alm_config=alm_cfg, corridor_config=corridor_cfg)
    trace = out["trace"]
    p = out["p"].detach().cpu()
    gt = batch["pos"].detach().cpu()
    err = torch.linalg.norm(p - gt, dim=-1)
    fv = out["final_validation"]
    guided = out["guided"].detach().cpu()
    _off_max, _off_member = offline_corridor_metrics(p, batch)
    res = {
        "name": name, "ckpt": ckpt, "config": config,
        "ablation": ablation,
        "feedback_enabled": bool(out["feedback"]["enabled"]),
        "arch_from_ckpt": report.get("arch") if report else None,
        "guided_rate": float(guided.float().mean()),
        "alm_status": sorted(set(out["alm_status"])),
        "curve_rmse_m": float(err.pow(2).mean().sqrt()) * scene_to_meter,
        # model-INDEPENDENT (offline GT corridor, identical for every model)
        "offline_corridor_max_violation": _off_max,
        "offline_corridor_membership_rate": _off_member,
        "curve_max_err_m": float(err.max()) * scene_to_meter,
        "final_collision_rate": mean([v.get("final_collision") for v in fv]),
        "final_free_rate": mean([v.get("final_free_rate") for v in fv]),
        "final_max_constraint_violation": mean(
            [v.get("final_max_constraint_violation") for v in fv]),
        "final_constraint_feasible_rate": mean(
            [v.get("final_constraint_feasible") for v in fv]),
        "final_corridor_membership_rate": mean(
            [v.get("final_corridor_membership_rate") for v in fv]),
        "endpoint_error": mean([v.get("endpoint_error") for v in fv]),
        "feedback_history": out["feedback"]["history"],
        "feedback_valid_rate": float(
            out["feedback"]["valid"].float().mean()),
        "per_step_raw_violation": series(trace, "raw_violation"),
        "per_step_alm_correction": series(trace, "alm_correction"),
        "per_step_smooth_raw": series(trace, "curve_smoothness_raw"),
        "per_step_smooth_safe": series(trace, "curve_smoothness_safe"),
        "per_step_guided": [bool(torch.as_tensor(s["guided"]).any())
                            for s in trace],
    }
    # first guided step / last guided step: the "does the ALM still need to
    # intervene late in the reverse process?" question of report section 18
    raw = [(i, v) for i, v in enumerate(res["per_step_raw_violation"])
           if v is not None]
    corr = [(i, v) for i, v in enumerate(res["per_step_alm_correction"])
            if v is not None]
    res["first_guided_raw_violation"] = raw[0][1] if raw else None
    res["last_guided_raw_violation"] = raw[-1][1] if raw else None
    res["first_guided_alm_correction"] = corr[0][1] if corr else None
    res["last_guided_alm_correction"] = corr[-1][1] if corr else None
    res["mean_guided_raw_violation"] = mean([v for _, v in raw])
    res["mean_guided_alm_correction"] = mean([v for _, v in corr])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_160k8p.yaml")
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=ckpt[::config]; repeat for several models")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ablation", default=None, choices=["A", "B", "C", "D"],
                    help="A = raw diffusion (ALM off); default = the YAML")
    ap.add_argument("--out", default="outputs/campaign_testset_eval.json")
    ap.add_argument("--md", default="outputs/campaign_testset_eval.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = data_cfg.get("processed_root")
    scene_to_meter = float(data_cfg.get("scene_to_meter", 40.0))
    ds = CarlaSplineDataset(args.split, root, geometry_points=1280,
                            limit=int(args.samples),
                            num_controls=num_controls(cfg),
                            num_safety_queries=num_safety_queries(cfg))
    batch = make_collate(ds)([ds[i] for i in range(len(ds))])
    schedule = NoiseSchedule(
        cfg["diffusion"]["timesteps"],
        beta_schedule=cfg["diffusion"].get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=cfg["diffusion"].get("beta_start", 1e-4),
        beta_end=cfg["diffusion"].get("beta_end", 0.02)).to(device)
    print("[eval] split=%s samples=%d steps=%d device=%s"
          % (args.split, len(ds), args.steps, device), flush=True)

    results = []
    for spec in args.model:
        name, ckpt, config = parse_model(spec, args.config)
        if not os.path.exists(ckpt if os.path.isabs(ckpt)
                              else os.path.join(ROOT, ckpt)):
            print("[eval] SKIP %s: %s not found" % (name, ckpt), flush=True)
            continue
        res = evaluate(name, config, ckpt, batch, schedule, device,
                       args.steps, args.seed, scene_to_meter,
                       ablation=args.ablation)
        results.append(res)
        # NOTE: 0.0 is falsy, so never use ``x or default`` here
        print("[eval] %-11s coll=%s viol=%s member=%s rmse=%.2fm "
              "| OFFv=%s OFFm=%s | fb_valid=%.3f raw=%s->%s corr=%s->%s"
              % (res["name"], _fmt(res["final_collision_rate"]),
                 _fmt(res["final_max_constraint_violation"]),
                 _fmt(res["final_corridor_membership_rate"]),
                 res["curve_rmse_m"],
                 _fmt(res["offline_corridor_max_violation"]),
                 _fmt(res["offline_corridor_membership_rate"]),
                 res["feedback_valid_rate"],
                 _fmt(res["first_guided_raw_violation"]),
                 _fmt(res["last_guided_raw_violation"]),
                 _fmt(res["first_guided_alm_correction"]),
                 _fmt(res["last_guided_alm_correction"])), flush=True)

    out_path = args.out if os.path.isabs(args.out) \
        else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"split": args.split, "samples": len(ds),
                   "steps": args.steps, "seed": args.seed,
                   "ablation": args.ablation,
                   "results": results}, fh, indent=2, ensure_ascii=False)
    print("[eval] written %s" % out_path)

    md_path = args.md if os.path.isabs(args.md) else os.path.join(ROOT, args.md)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    lines = ["# 160k8p campaign: 同口径 DDIM + ALM 采样评估", "",
             "* split `%s`, %d samples, %d reverse steps, seed %d"
             % (args.split, len(ds), args.steps, args.seed), "",
             "| model | coll | sampler viol | sampler member | rmse_m | "
             "OFFLINE viol | OFFLINE member | fb_valid | raw viol 1st→last | "
             "ALM corr 1st→last | fb history |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append("| %s | %s | %s | %s | %.2f | %s | %s | %.3f | %s→%s | "
                     "%s→%s | %s |"
                     % (r["name"], _fmt(r["final_collision_rate"]),
                        _fmt(r["final_max_constraint_violation"]),
                        _fmt(r["final_corridor_membership_rate"]),
                        r["curve_rmse_m"],
                        _fmt(r["offline_corridor_max_violation"]),
                        _fmt(r["offline_corridor_membership_rate"]),
                        r["feedback_valid_rate"],
                        _fmt(r["first_guided_raw_violation"]),
                        _fmt(r["last_guided_raw_violation"]),
                        _fmt(r["first_guided_alm_correction"]),
                        _fmt(r["last_guided_alm_correction"]),
                        json.dumps(r["feedback_history"],
                                   ensure_ascii=False)))
    lines += ["", "## 逐 step 曲线（guided step，均值 / 有效样本）", ""]
    for r in results:
        lines += ["### %s" % r["name"], "",
                  "| reverse step idx | raw violation | ALM correction | "
                  "curve smooth raw | curve smooth safe |",
                  "|---|---|---|---|---|"]
        for i in range(len(r["per_step_raw_violation"])):
            if r["per_step_raw_violation"][i] is None \
                    and r["per_step_alm_correction"][i] is None:
                continue
            lines.append("| %d | %s | %s | %s | %s |"
                         % (i, _fmt(r["per_step_raw_violation"][i]),
                            _fmt(r["per_step_alm_correction"][i]),
                            _fmt(r["per_step_smooth_raw"][i], 5),
                            _fmt(r["per_step_smooth_safe"][i], 5)))
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[eval] written %s" % md_path)
    return 0


def _fmt(x, nd=4):
    return "-" if x is None else ("%.*f" % (nd, x))


if __name__ == "__main__":
    raise SystemExit(main())
