#!/usr/bin/env python
"""实验 3：ALM 迭代预算与安全性-计算效率权衡（实验/docs/实验3_ALM迭代预算与安全效率权衡_实施方案.md）。

只改变 ALM 的**最大内迭代次数**，其余条件全部固定为加速版推理配置
（生成器 A、4 步 DDIM、预热 2、凸区域共享 stride=8、历史反馈开启、无末端投影）。

阶段
----
``p0``      静态核查 + 4 样本差分探针：预算真正控制的循环、Default 取值、早停、容差
``p1``      验证集 60 任务上六组配置预实验（确认预算真实生效）
``run``     测试集 420 × 种子 0/1/2：质量/安全（chunk=32）
``latency`` 测试集 420 × 种子 0，chunk=1：在线端到端延迟
``table``   表 1/表 2、配对 bootstrap、核心图

已核对的实现事实
----------------
* 预算作用点 ``src/diffusion/sampler.py``：每次 reverse step 构造 per-sample
  ``budget``（首次触发 = ``activation_inner_steps``，之后 = ``inner_steps``）并作为
  ``bspline_alm_correct(max_steps=budget)`` 传入。故**预算上限 ≠ 实际迭代次数**。
* 早停 ``src/diffusion/bspline_alm.py``：``remaining = (it+1 < max_steps) & (v_max_now > tol)``，
  全部样本"用完预算 或 满足 tol"时跳出。
* Default 的真实取值：``activation_inner_steps=8``、``inner_steps=10``（配置文件读出）。
* 连续约束满足率**独立复算**：对最终控制点调用 ``constraint_state(q_final, pack)`` 得到
  每个 Bézier 控制点的半空间取值，阈值用 ``alm.constraint_tol``（1e-3 scene = 0.08 m），
  不使用求解器内部的 feasible 标志。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import metrics as M         # noqa: E402
from src.diffusion.bspline_alm import (_violation_stats,  # noqa: E402
                                       constraint_state)
from src.utils.config import load_config  # noqa: E402

OUT = os.path.join(C.OUT_ROOT, "exp3")
RAW = os.path.join(OUT, "raw")
LAT = os.path.join(OUT, "raw_latency")
P1 = os.path.join(OUT, "raw_p1")

#: 六组预算配置；``activation`` = 首次触发的那次调用，``inner`` = 之后的每次调用。
CONFIGS = {
    "I2": dict(activation=2, inner=2, label="I2"),
    "I4": dict(activation=4, inner=4, label="I4"),
    "I6": dict(activation=6, inner=6, label="I6"),
    "I8": dict(activation=8, inner=8, label="I8"),
    "I10": dict(activation=10, inner=10, label="I10"),
    "Default": dict(activation=8, inner=10, label="Default"),
}
ORDER = ["I2", "I4", "I6", "I8", "I10", "Default"]
#: 加速版推理配置（对应实验 1 表 2 的"组合加速"档）
BASE = dict(steps=4, region_stride=8, feedback_on=True, final_project=False)
#: 计时预热：与 ``common.run_split`` / ``eval_chapter4`` 相同的前 3 个 chunk 丢弃
WARMUP_CHUNKS = 3


def _alm_overrides(cfg: dict) -> dict:
    return dict(warmup_reverse_steps=3,      # 4 步下生效值 = 2
                activation_inner_steps=cfg["activation"],
                inner_steps=cfg["inner"],
                force_feedback_valid_zero=not BASE["feedback_on"],
                safe_prediction_to_ddim=True,
                final_project=BASE["final_project"])


def _corridor_overrides() -> dict:
    return dict(region_stride=BASE["region_stride"])


def _tolerance() -> float:
    return float((load_config(C.CONFIG_A).get("alm") or {})
                 .get("constraint_tol", 1e-3))


# ---------------------------------------------------------------------------
# per-call logging + independent feasibility check
# ---------------------------------------------------------------------------


class CallLogger:
    """Accumulates the per-ALM-call log and the final-trajectory feasibility."""

    def __init__(self, tol_scene: float):
        self.tol = float(tol_scene)
        self.calls: list = []        # one row per ALM call (chunk scoped)
        self.samples: list = []      # one row per sample per ALM call
        self.feasibility: dict = {}  # global index -> dict

    def hook(self, payload):
        trace = payload.get("trace") or []
        out = payload["out"]
        pack = out.get("pack")
        ids = payload["batch"]["sample_id"].detach().cpu().numpy()
        rows_index = payload["rows_index"]
        guided = out["guided"].detach().cpu().numpy()
        alm_status = [str(s) for s in (out.get("alm_status") or [])]
        for fwd, step in enumerate(trace, start=1):
            st = step.get("alm_stats")
            if not step.get("alm_active") or st is None:
                continue
            used = st["inner_steps_used"].detach().cpu().numpy()
            budget = step.get("alm_budget")
            budget = (budget.numpy() if budget is not None
                      else np.zeros(len(rows_index), dtype=np.int64))
            v_before = st["max_violation_before"].detach().cpu().numpy()
            v_after = st["max_violation_after"].detach().cpu().numpy()
            corr = st["max_curve_correction_scene"].detach().cpu().numpy()
            secs = step.get("alm_seconds")
            active = budget > 0
            n_active = int(active.sum())
            self.calls.append(dict(
                forward_index=fwd, diffusion_t=int(step.get("t", -1)),
                chunk_start=int(payload.get("start", -1)),
                n_active=n_active,
                budget_max=int(budget.max()) if n_active else 0,
                iterations_mean=float(used[active].mean()) if n_active else None,
                alm_seconds_batch=secs,
                alm_ms_per_active=(None if (not secs or not n_active)
                                   else 1000.0 * float(secs) / n_active)))
            for local, index in enumerate(rows_index):
                b = int(budget[local]) if local < len(budget) else 0
                if b <= 0:
                    continue
                u = int(used[local])
                # NOTE: ``inner_steps_used`` is a per-sample counter of the
                # BATCHED loop, so a sample that converged early can still show
                # a count > 1 (the batch kept iterating for others).  The stop
                # reason is therefore decided by the violation, not the counter.
                if v_after[local] <= self.tol:
                    reason = "tol_reached"
                elif u >= b:
                    reason = "budget_exhausted"
                else:
                    reason = "batch_continued"
                self.samples.append(dict(
                    sample_id=int(ids[local]), index=int(index),
                    chunk_start=int(payload.get("start", -1)),
                    forward_index=fwd, diffusion_t=int(step.get("t", -1)),
                    budget=b, iterations_used=u, stop_reason=reason,
                    violation_before_scene=float(v_before[local]),
                    violation_after_scene=float(v_after[local]),
                    correction_max_scene=float(corr[local]),
                    alm_seconds=secs))
        # independent continuous-constraint recheck on the FINAL control polygon
        q_final = out["control"].detach()
        v_max = v_rate = None
        if pack is not None:
            _, g, mask = constraint_state(q_final, pack)
            v_max_t, _, v_rate_t = _violation_stats(g, mask)
            v_max = v_max_t.detach().cpu().numpy()
            v_rate = v_rate_t.detach().cpu().numpy()
        for local, index in enumerate(rows_index):
            self.feasibility[int(index)] = dict(
                constraint_max_violation_scene=(None if v_max is None
                                                else float(v_max[local])),
                constraint_feasible_rate=(None if v_rate is None
                                          else float(v_rate[local])),
                constraint_feasible=(None if v_max is None
                                     else bool(v_max[local] <= self.tol)),
                alm_status=(alm_status[local] if local < len(alm_status) else ""),
                corridor_guided=bool(guided[local]))


def _run_config(tag, seed, args, *, limit, split, chunk, log: CallLogger,
                dump_dir=None):
    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=split,
                          processed=args.processed, limit=limit)
    res = C.run_split(ctx, method="budget_%s" % tag, guidance="alm",
                      alm_overrides=_alm_overrides(CONFIGS[tag]),
                      corridor_overrides=_corridor_overrides(),
                      steps=BASE["steps"], seed=seed, chunk=chunk,
                      chunk_hook=log.hook)
    for row in res.rows:
        info = log.feasibility.get(int(row["index"]), {})
        row.update(info)
    del ctx
    if dump_dir:
        C.ensure_dir(dump_dir)
        stem = os.path.join(dump_dir, "%s_seed%d" % (tag, seed))
        C.save_csv(stem + ".csv", res.rows)
        C.save_arrays(stem + ".npz", control=res.controls,
                      reference=res.reference, endpoints=res.endpoints)
        C.save_json(stem + ".json", dict(header=res.header, timing=res.timing,
                                         budget=CONFIGS[tag], base=BASE))
    return res


# ---------------------------------------------------------------------------
# P0
# ---------------------------------------------------------------------------


def _stage_p0(args) -> None:
    cfg = load_config(C.CONFIG_A)
    alm = cfg.get("alm") or {}
    cor = cfg.get("corridor") or {}
    payload = dict(
        experiment="ALM iteration budget vs safety/efficiency",
        generator="A (feedback-trained)", checkpoint=C.CKPT_A,
        checkpoint_hash=C.file_hash(C.CKPT_A), config=C.CONFIG_A,
        git_commit=C.git_commit(), python=sys.executable,
        torch=torch.__version__,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        ddim_steps=BASE["steps"],
        warmup_configured=int(alm.get("warmup_reverse_steps", 3)),
        warmup_effective=int(min(int(alm.get("warmup_reverse_steps", 3)),
                                 BASE["steps"] - 2)),
        region_stride=BASE["region_stride"],
        feedback_enabled=BASE["feedback_on"],
        final_project=BASE["final_project"],
        default_budget=dict(
            activation_inner_steps=int(alm.get("activation_inner_steps", 8)),
            inner_steps=int(alm.get("inner_steps", 10))),
        constraint_tol_scene=_tolerance(),
        constraint_tol_m=_tolerance() * float(cfg["data"].get(
            "scene_to_meter", 80.0)),
        rho=float(alm.get("rho", 5.0)),
        step_size=float(alm.get("step_size", 0.05)),
        max_curve_step_scene=float(alm.get("max_curve_step_scene", 0.2)),
        corridor_region=cor.get("region"), corridor_bridge=cor.get("bridge"),
        budget_site=("src/diffusion/sampler.py: per-sample `budget` tensor -> "
                     "bspline_alm_correct(max_steps=budget)"),
        early_stop=("src/diffusion/bspline_alm.py: remaining = (it+1 < max_steps)"
                    " & (v_max_now > tol); break when no sample remains"),
        feasibility_recheck=("constraint_state(q_final, pack) + _violation_stats; "
                             "threshold = alm.constraint_tol (independent of the "
                             "solver flag)"),
        timing_scope=("bspline_alm_correct only: residual + analytic gradient + "
                      "Cholesky step + dual update; excludes network forward, SDF "
                      "build and corridor construction"),
        configurations={k: dict(activation_inner_steps=v["activation"],
                                inner_steps=v["inner"])
                        for k, v in CONFIGS.items()},
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    probe = {}
    for tag in ("I2", "Default"):
        log = CallLogger(_tolerance())
        _run_config(tag, 0, args, limit=4, split=args.split, chunk=4, log=log)
        used = [r["iterations_used"] for r in log.samples]
        secs = [c["alm_seconds_batch"] for c in log.calls if c["alm_seconds_batch"]]
        probe[tag] = dict(calls=len(log.calls),
                          iterations_mean=float(np.mean(used)) if used else None,
                          iterations_max=int(max(used)) if used else None,
                          seconds_per_call=float(np.mean(secs)) if secs else None)
    payload["probe_4samples"] = probe
    payload["probe_budget_effective"] = bool(
        probe["I2"]["iterations_max"] and probe["Default"]["iterations_max"]
        and probe["I2"]["iterations_max"] < probe["Default"]["iterations_max"])
    C.save_json(os.path.join(OUT, "alm_budget_config.json"), payload)
    print("[exp3] P0 written: probe=%s effective=%s"
          % (json.dumps(probe), payload["probe_budget_effective"]), flush=True)


# ---------------------------------------------------------------------------
# P1 (validation pre-experiment)
# ---------------------------------------------------------------------------


def _stage_p1(args) -> None:
    C.ensure_dir(P1)
    rows, calls = [], []
    for tag in ORDER:
        log = CallLogger(_tolerance())
        _run_config(tag, 0, args, limit=args.p1_limit, split="val", chunk=args.chunk,
                    log=log, dump_dir=P1)
        for r in log.samples:
            r.update(config=tag, seed=0, split="val")
        for c in log.calls:
            c.update(config=tag, seed=0)
        rows += log.samples
        calls += log.calls
    C.save_csv(os.path.join(OUT, "p1_alm_call_log.csv"), rows)
    C.save_csv(os.path.join(OUT, "p1_alm_call_timing.csv"), calls)
    table = []
    for tag in ORDER:
        used = [r["iterations_used"] for r in rows if r["config"] == tag]
        secs = [c["alm_seconds_batch"] for c in calls
                if c["config"] == tag and c["alm_seconds_batch"]]
        at_cap = [r for r in rows if r["config"] == tag
                  and r["stop_reason"] == "budget_exhausted"]
        table.append(dict(config=tag, n_calls=len(used),
                          iterations_mean=(float(np.mean(used)) if used else None),
                          iterations_p95=(float(np.percentile(used, 95))
                                          if used else None),
                          at_budget_pct=(100.0 * len(at_cap) / len(used)
                                         if used else None),
                          alm_ms_per_call=(1000.0 * float(np.mean(secs))
                                           if secs else None)))
    C.save_csv(os.path.join(OUT, "p1_summary.csv"), table)
    C.save_json(os.path.join(OUT, "p1_summary.json"),
                dict(split="val", tasks=args.p1_limit, table=table))
    print("[exp3] P1 done (%d val tasks): %s" % (args.p1_limit, table), flush=True)


# ---------------------------------------------------------------------------
# P2 (full test set)
# ---------------------------------------------------------------------------


def _stage_run(args) -> None:
    C.ensure_dir(RAW)
    for tag in args.configs:
        for seed in (int(s) for s in args.seeds):
            stem = os.path.join(RAW, "%s_seed%d.npz" % (tag, seed))
            if os.path.exists(stem) and not args.force:
                print("[exp3] skip %s_seed%d" % (tag, seed), flush=True)
                continue
            log = CallLogger(_tolerance())
            _run_config(tag, seed, args, limit=args.limit or None,
                        split=args.split, chunk=args.chunk, log=log,
                        dump_dir=RAW)
            for r in log.samples:
                r.update(config=tag, seed=seed)
            for c in log.calls:
                c.update(config=tag, seed=seed)
            C.save_csv(os.path.join(RAW, "calls_%s_seed%d.csv" % (tag, seed)),
                       log.samples)
            C.save_csv(os.path.join(RAW, "timing_%s_seed%d.csv" % (tag, seed)),
                       log.calls)
    print("[exp3] full runs done", flush=True)


def _stage_latency(args) -> None:
    """End-to-end inference time WITHOUT trace instrumentation, seed 0.

    Two protocols are measured for the same checkpoint/config, exactly as the
    guide asks: ``batch = 32`` (per-trajectory amortised, comparable with 实验 1
    表 2) and ``batch = 1`` (single-sample online latency).  The per-step trace
    used for the ALM diagnostics costs ~3.7x the sampling wall clock, so it is
    switched OFF here -- otherwise the numbers measure the logger.
    """
    C.ensure_dir(LAT)
    for tag in args.configs:
        for bsz, name in ((args.chunk, "b32"), (1, "b1")):
            stem = os.path.join(LAT, "%s_%s_seed0" % (tag, name))
            if os.path.exists(stem + ".csv") and not args.force:
                print("[exp3] skip timing %s %s" % (tag, name), flush=True)
                continue
            ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                                  processed=args.processed,
                                  limit=args.limit or None)
            res = C.run_split(ctx, method="timing_%s_%s" % (tag, name),
                              guidance="alm",
                              alm_overrides=_alm_overrides(CONFIGS[tag]),
                              corridor_overrides=_corridor_overrides(),
                              steps=BASE["steps"], seed=0, chunk=bsz,
                              return_trace=False, progress=False)
            C.save_csv(stem + ".csv", res.rows,
                       ["method", "seed", "sample_id", "collision",
                        "sdf_min_distance_m", "rmse_m", "inference_ms"])
            C.save_json(stem + ".json",
                        dict(protocol=dict(batch=bsz, n=len(res.rows),
                                           seed=0, steps=BASE["steps"],
                                           return_trace=False,
                                           warmup_chunks=WARMUP_CHUNKS),
                             mean_ms=res.timing.get("mean_per_traj_ms"),
                             budget=CONFIGS[tag]))
            del ctx
    print("[exp3] timing runs done", flush=True)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _load_group(directory, pattern="%s_seed%d.npz"):
    runs = {}
    for tag in ORDER:
        stems = sorted(f[:-4] for f in
                       glob.glob(os.path.join(directory, "%s_seed*.npz" % tag)))
        if not stems:
            continue
        rows, controls, refs, ends = [], [], [], []
        for stem in stems:
            seed = int(os.path.basename(stem).split("_seed")[1])
            part = C.load_csv(stem + ".csv")
            for row in part:
                row["config"] = tag
                row["seed"] = int(row["seed"])
                row["sample_id"] = int(row["sample_id"])
                row["collision"] = str(row["collision"]).lower() == "true"
                row["sdf_min_distance_m"] = float(row["sdf_min_distance_m"])
                row["rmse_m"] = float(row["rmse_m"])
                for key in ("inference_ms", "constraint_max_violation_scene",
                            "constraint_feasible_rate"):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                if row.get("constraint_feasible") in ("True", "False"):
                    row["constraint_feasible"] = row["constraint_feasible"] == "True"
                if row.get("constraint_max_violation_scene") not in (None, ""):
                    row["constraint_feasible"] = (
                        float(row["constraint_max_violation_scene"]) <= _tolerance())
                else:
                    row["constraint_feasible"] = None
            data = np.load(stem + ".npz")
            rows += part
            controls.append(data["control"])
            refs.append(data["reference"])
            ends.append(data["endpoints"])
        runs[tag] = dict(rows=rows, control=np.concatenate(controls),
                         reference=np.concatenate(refs),
                         endpoints=np.concatenate(ends))
    return runs


def _coerce_call_row(row):
    """Numeric coercion + stop-reason re-derivation for one ALM log row."""
    for key in ("chunk_start", "forward_index"):
        if row.get(key) not in (None, ""):
            row[key] = int(row[key])
    for key in ("alm_seconds_batch", "violation_before_scene",
                "violation_after_scene", "correction_max_scene",
                "iterations_mean", "alm_ms_per_active"):
        if row.get(key) not in (None, ""):
            row[key] = float(row[key])
    if row.get("iterations_used") not in (None, ""):
        row["iterations_used"] = int(row["iterations_used"])
        if row.get("budget") not in (None, ""):
            row["budget"] = int(row["budget"])
            v_after = row.get("violation_after_scene")
            if isinstance(v_after, float):
                if v_after <= _tolerance():
                    row["stop_reason"] = "tol_reached"
                elif row["iterations_used"] >= row["budget"]:
                    row["stop_reason"] = "budget_exhausted"
                else:
                    row["stop_reason"] = "batch_continued"
    return row


def _load_rows_csv(path):
    """Load one ALM log CSV whose rows already carry config/seed columns."""
    return [_coerce_call_row(dict(row)) for row in C.load_csv(path)]


def _load_calls(directory, prefix="calls_"):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, prefix + "*.csv"))):
        base = os.path.basename(path)[len(prefix):].rsplit(".", 1)[0]
        if "_seed" not in base:
            continue
        tag, seed_s = base.split("_seed")
        for row in C.load_csv(path):
            row["config"] = tag
            row["seed"] = int(seed_s)
            rows.append(_coerce_call_row(row))
    return rows

def _bootstrap_ci(a, b, n_boot=2000, seed=0):
    """Paired bootstrap over tasks for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return None
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return dict(mean=float(d.mean()), lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)), n=int(d.size))


def _paired_by_task(rows, key):
    """Per (sample_id) mean over seeds, so the bootstrap resamples tasks."""
    acc = {}
    for row in rows:
        v = row.get(key)
        if v is None or v == "":
            continue
        acc.setdefault(int(row["sample_id"]), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _stage_timing(args) -> None:
    """INTERLEAVED timing: every chunk/sample runs all six budgets back-to-back.

    实验 3 §3.2 step 6 recommends interleaving the configurations so a drifting
    device load (this machine shares the GPU with other processes) hits all of
    them equally.  The per-step trace is OFF: it costs ~3.7x the sampling wall
    clock, so timing it would measure the logger.
    """
    C.ensure_dir(LAT)
    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                          processed=args.processed, limit=args.limit or None)
    collate = C.collate_fn(ctx)
    dev, model = ctx.device, ctx.model
    n_total = ctx.n_total
    rows = []
    plans = [("b32", int(args.chunk), n_total),
             ("b1", 1, min(n_total, int(args.b1_tasks)))]
    for name, bsz, n_use in plans:
        n_chunks = 0
        for start in range(0, n_use, bsz):
            idx = list(range(start, min(start + bsz, n_use)))
            batch = collate([ctx.dataset[i] for i in idx])
            tensors = (batch["cond"].to(dev), batch["occupancy"].to(dev),
                       batch["candidate_xy"].to(dev),
                       batch["candidate_mask"].to(dev),
                       batch["candidate_geometry"].to(dev),
                       batch["candidate_geometry_lengths"].to(dev))
            n_chunks += 1
            warm = n_chunks <= WARMUP_CHUNKS
            for tag in ORDER:
                alm = dict(ctx.alm_cfg)
                alm.update(_alm_overrides(CONFIGS[tag]))
                alm["guidance"] = "alm"
                corridor = dict(ctx.corridor_cfg)
                corridor.update(_corridor_overrides())
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                C.sample(model, ctx.schedule, *tensors, device=str(dev),
                         steps=BASE["steps"], seed=0, seed_offset=start,
                         return_trace=False, alm_config=alm,
                         corridor_config=corridor, guidance="alm")
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                if not warm:
                    rows.append(dict(protocol=name, config=tag,
                                     chunk_start=start, n=len(idx),
                                     ms_per_traj=1000.0 * wall / len(idx)))
            if n_chunks % 4 == 0:
                print("[exp3] timing %s %d/%d" % (name, idx[-1] + 1, n_use),
                      flush=True)
    C.save_csv(os.path.join(OUT, "timing_interleaved.csv"), rows)
    summary = {}
    for name, _, _ in plans:
        summary[name] = {}
        for tag in ORDER:
            vals = np.asarray([r["ms_per_traj"] for r in rows
                               if r["protocol"] == name and r["config"] == tag],
                              dtype=np.float64)
            if vals.size:
                summary[name][tag] = dict(
                    mean=float(vals.mean()), median=float(np.median(vals)),
                    p95=float(np.percentile(vals, 95)), n=int(vals.size))
    C.save_json(os.path.join(OUT, "timing_interleaved.json"),
                dict(protocol=dict(steps=BASE["steps"],
                                   region_stride=BASE["region_stride"],
                                   return_trace=False,
                                   warmup_chunks=WARMUP_CHUNKS,
                                   interleaved=True,
                                   b1_tasks=int(args.b1_tasks)),
                     summary=summary,
                     note=("configs run back-to-back on the same batch so device "
                           "load drifting hits all of them equally")))
    print("[exp3] interleaved timing done: %s" % json.dumps(summary), flush=True)


def _stage_p1table(args) -> None:
    """Re-derive the P1 summary from the saved CSV (no re-running)."""
    path = os.path.join(OUT, "p1_alm_call_log.csv")
    if not os.path.exists(path):
        raise SystemExit("[exp3] run --stage p1 first")
    rows = _load_rows_csv(path)
    timing = _load_rows_csv(os.path.join(OUT, "p1_alm_call_timing.csv"))
    table = []
    for tag in ORDER:
        mine = [r for r in rows if r["config"] == tag]
        used = [float(r["iterations_used"]) for r in mine]
        at_cap = sum(1 for r in mine if r["stop_reason"] == "budget_exhausted")
        at_tol = sum(1 for r in mine if r["stop_reason"] == "tol_reached")
        secs = [float(c["alm_seconds_batch"]) for c in timing
                if c["config"] == tag and c.get("alm_seconds_batch") is not None]
        table.append(dict(
            config=tag, n_calls=len(mine),
            iterations_mean=(float(np.mean(used)) if used else None),
            iterations_p95=(float(np.percentile(used, 95)) if used else None),
            at_budget_pct=(100.0 * at_cap / len(mine) if mine else None),
            tol_reached_pct=(100.0 * at_tol / len(mine) if mine else None),
            violation_before_mean=(float(np.mean(
                [r["violation_before_scene"] for r in mine])) if mine else None),
            violation_after_mean=(float(np.mean(
                [r["violation_after_scene"] for r in mine])) if mine else None),
            alm_ms_per_call_cold=(1000.0 * float(np.mean(secs)) if secs else None)))
    C.save_csv(os.path.join(OUT, "p1_summary.csv"), table)
    C.save_json(os.path.join(OUT, "p1_summary.json"),
                dict(split="val", tasks=args.p1_limit, table=table,
                     note=("60 validation tasks, one seed; timings include cold "
                           "CUDA start-up (only 2 chunks) and are therefore not "
                           "the reported cost")))
    print("[exp3] P1 summary recomputed: %s" % table, flush=True)


def _stage_table(args) -> None:
    runs = _load_group(RAW)
    if not runs:
        raise SystemExit("[exp3] no runs in %s" % RAW)
    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                          processed=args.processed, limit=1)
    codec, sm = ctx.codec, ctx.scene_to_meter
    tol = _tolerance()
    calls = _load_calls(RAW, "calls_")          # per-sample per-call rows
    call_timing = _load_calls(RAW, "timing_")   # per-call wall clock (batch scoped)
    latency = {}
    for tag in ORDER:
        for name in ("b32", "b1"):
            path = os.path.join(LAT, "%s_%s_seed0.csv" % (tag, name))
            if not os.path.exists(path):
                continue
            vals = [float(r["inference_ms"]) for r in C.load_csv(path)
                    if r.get("inference_ms") not in (None, "")]
            if vals:
                latency[(tag, name)] = np.asarray(vals)
    # interleaved measurement (all budgets back-to-back on the same batch):
    # this is the reported timing because the device load drifts during a run
    inter = {}
    inter_path = os.path.join(OUT, "timing_interleaved.json")
    if os.path.exists(inter_path):
        with open(inter_path, encoding="utf-8") as handle:
            inter = json.load(handle).get("summary") or {}
    per_task, table1, table2 = [], [], []
    for tag in ORDER:
        if tag not in runs:
            print("[exp3] WARNING no run for %s" % tag, flush=True)
            continue
        pack = runs[tag]
        M.enrich(pack["rows"], pack["control"], pack["reference"],
                 pack["endpoints"], codec, sm, step_m=args.geometry_step)
        rows = pack["rows"]
        per_task += rows
        n = len(rows)
        coll = sum(1 for r in rows if r["collision"])
        feas = [r for r in rows if r.get("constraint_feasible") is not None]
        feas_ok = sum(1 for r in feas if r["constraint_feasible"])
        stats = M.summarise(rows, ("clearance_m", "detour_ratio", "nrmse_pct",
                                   "kappa_max_inv_m", "kappa2_mean_inv_m2"))
        table1.append(dict(
            config=tag, constraint_feasible_pct=(100.0 * feas_ok / len(feas)
                                                if feas else None),
            constraint_checked=len(feas),
            collision_pct=100.0 * coll / n, collisions=coll, n=n,
            clearance_m=stats["clearance_m"]["mean"],
            clearance_n=stats["clearance_m"]["valid_n"],
            detour_ratio=stats["detour_ratio"]["mean"],
            nrmse_pct=stats["nrmse_pct"]["mean"],
            kappa_max_inv_m=stats["kappa_max_inv_m"]["mean"],
            kappa2_mean_inv_m2=stats["kappa2_mean_inv_m2"]["mean"]))
        mine = [c for c in calls if c["config"] == tag]
        used = [float(r["iterations_used"]) for r in mine]
        at_cap = sum(1 for r in mine if r["stop_reason"] == "budget_exhausted")
        # per-call wall clock: only post-warm-up chunks, identical for every
        # configuration (cold CUDA kernels otherwise dominate the first calls)
        warm = WARMUP_CHUNKS * int(args.chunk)
        secs = [float(c["alm_seconds_batch"]) for c in call_timing
                if c["config"] == tag
                and c.get("alm_seconds_batch") is not None
                and c["chunk_start"] >= warm]
        lat32 = latency.get((tag, "b32"))
        lat1 = latency.get((tag, "b1"))
        ib32 = (inter.get("b32") or {}).get(tag) or {}
        ib1 = (inter.get("b1") or {}).get(tag) or {}
        table2.append(dict(
            config=tag, n_calls=len(mine),
            iterations_mean=(float(np.mean(used)) if used else None),
            iterations_p95=(float(np.percentile(used, 95)) if used else None),
            at_budget_pct=(100.0 * at_cap / len(mine) if mine else None),
            alm_ms_per_call=(1000.0 * float(np.mean(secs)) if secs else None),
            alm_ms_per_call_p95=(1000.0 * float(np.percentile(secs, 95))
                                 if secs else None),
            alm_calls_timed=len(secs),
            # primary timing: interleaved across budgets (load-drift neutral)
            e2e_ms_batch32=ib32.get("mean"),
            e2e_ms_batch32_median=ib32.get("median"),
            e2e_ms_batch32_p95=ib32.get("p95"),
            e2e_ms_batch1=ib1.get("mean"),
            e2e_ms_batch1_median=ib1.get("median"),
            e2e_ms_batch1_p95=ib1.get("p95"),
            e2e_ms_batch1_n=ib1.get("n", 0),
            # secondary: same protocol but one configuration per process
            e2e_ms_batch32_seq=(float(lat32.mean()) if lat32 is not None
                                else None),
            e2e_ms_batch1_seq=(float(lat1.mean()) if lat1 is not None else None)))
    columns1 = ["config", "constraint_feasible_pct", "constraint_checked",
                "collision_pct", "collisions", "n", "clearance_m", "clearance_n",
                "detour_ratio", "nrmse_pct", "kappa_max_inv_m",
                "kappa2_mean_inv_m2"]
    columns2 = ["config", "n_calls", "iterations_mean", "iterations_p95",
                "at_budget_pct", "alm_ms_per_call", "alm_ms_per_call_p95",
                "alm_calls_timed", "e2e_ms_batch32", "e2e_ms_batch32_median",
                "e2e_ms_batch32_p95", "e2e_ms_batch1", "e2e_ms_batch1_median",
                "e2e_ms_batch1_p95", "e2e_ms_batch1_n",
                "e2e_ms_batch32_seq", "e2e_ms_batch1_seq"]
    C.save_csv(os.path.join(OUT, "table1_budget_safety.csv"), table1, columns1)
    C.save_csv(os.path.join(OUT, "table2_budget_cost.csv"), table2, columns2)
    # combined per-configuration summary (实验3 §8: alm_budget_summary.csv)
    by_tag2 = {r["config"]: r for r in table2}
    merged = [dict(r, **{k: v for k, v in by_tag2.get(r["config"], {}).items()
                         if k != "config"}) for r in table1]
    C.save_csv(os.path.join(OUT, "alm_budget_summary.csv"), merged,
               columns1 + [c for c in columns2 if c != "config"])
    C.save_csv(os.path.join(OUT, "alm_trajectory_metrics.csv"), per_task,
               ["config", "seed", "sample_id", "collision", "clearance_m",
                "sdf_min_distance_m", "detour_ratio", "nrmse_pct", "rmse_m",
                "kappa_max_inv_m", "kappa2_mean_inv_m2", "constraint_feasible",
                "constraint_max_violation_scene", "constraint_feasible_rate",
                "corridor_guided", "alm_status", "activation_step",
                "inference_ms"])
    calls_all = calls
    C.save_csv(os.path.join(OUT, "alm_call_log.csv"), calls_all)
    # paired bootstrap vs Default
    base_rows = runs.get("Default", {}).get("rows", [])
    boot = {}
    for tag in ORDER:
        if tag not in runs or tag == "Default" or not base_rows:
            continue
        entry = {}
        for key in ("detour_ratio", "nrmse_pct", "clearance_m",
                    "constraint_feasible", "collision"):
            a = _paired_by_task(runs[tag]["rows"], key)
            b = _paired_by_task(base_rows, key)
            common_keys = sorted(set(a) & set(b))
            if not common_keys:
                continue
            entry[key] = _bootstrap_ci([a[k] for k in common_keys],
                                       [b[k] for k in common_keys],
                                       n_boot=args.bootstrap)
        boot[tag] = entry
    payload = dict(table1=table1, table2=table2, bootstrap_vs_default=boot,
                   config=BASE, budgets={k: dict(activation_inner_steps=v["activation"],
                                                 inner_steps=v["inner"])
                                         for k, v in CONFIGS.items()},
                   tolerance=dict(scene=tol, m=tol * sm),
                   geometry_step_m=args.geometry_step,
                   baseline_table1="实验 1 (same metric definitions)")
    C.save_json(os.path.join(OUT, "table3_summary.json"), payload)
    _markdown(os.path.join(OUT, "table1_budget_safety.md"), table1, columns1)
    _markdown(os.path.join(OUT, "table2_budget_cost.md"), table2, columns2)
    _figure(table1, table2, os.path.join(OUT, "alm_budget_safety_runtime"))
    print("[exp3] tables + figure written", flush=True)


def _markdown(path, rows, columns) -> None:
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, bool):
            return "Y" if v else "N"
        if isinstance(v, float):
            return "%.4g" % v
        return str(v)
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")
    C.ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _figure(table1, table2, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "legend.frameon": False, "figure.dpi": 300})
    budgets = [2, 4, 6, 8, 10]
    tags = ["I2", "I4", "I6", "I8", "I10"]
    t1 = {r["config"]: r for r in table1}
    t2 = {r["config"]: r for r in table2}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    # (a) budget vs continuous constraint satisfaction (+ collision rate)
    feas = [t1.get(t, {}).get("constraint_feasible_pct") for t in tags]
    coll = [t1.get(t, {}).get("collision_pct") for t in tags]
    ax = axes[0]
    ax.plot(budgets, feas, marker="o", color="#1f4e79",
            label="Continuous constraint satisfied")
    ax.plot(budgets, coll, marker="s", linestyle="--", color="#B22222",
            label="Collision rate")
    if "Default" in t1:
        ax.scatter([11.5], [t1["Default"]["constraint_feasible_pct"]], marker="*",
                   s=90, color="#1f4e79", zorder=5)
        ax.scatter([11.5], [t1["Default"]["collision_pct"]], marker="*", s=90,
                   color="#B22222", zorder=5)
        ax.annotate("Default\n(8/10)", (11.5, 50), ha="center", fontsize=7,
                    color="0.3")
    ax.set_xlabel("Maximum ALM inner iterations")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Safety vs ALM budget", fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_xticks(range(2, 11, 2))
    ax.set_xlim(1, 12.5)
    ax.legend(loc="best", fontsize=7.5)
    # (b) budget vs end-to-end latency
    ax = axes[1]
    # per-chunk values of the interleaved b1 measurement -> mean +- 95% CI band
    raw = {}
    path = os.path.join(OUT, "timing_interleaved.csv")
    if os.path.exists(path):
        for row in C.load_csv(path):
            if row.get("protocol") != "b1":
                continue
            raw.setdefault(row["config"], []).append(float(row["ms_per_traj"]))
    lat = [t2.get(t, {}).get("e2e_ms_batch1") for t in tags]
    lat_p95 = [t2.get(t, {}).get("e2e_ms_batch1_p95") for t in tags]
    ci = []
    for t in tags:
        vals = np.asarray(raw.get(t, []), dtype=np.float64)
        ci.append(1.96 * float(vals.std(ddof=1) / np.sqrt(vals.size))
                  if vals.size > 1 else 0.0)
    lo = [None if (m is None or c is None) else m - c for m, c in zip(lat, ci)]
    hi = [None if (m is None or c is None) else m + c for m, c in zip(lat, ci)]
    if all(v is not None for v in lo):
        ax.fill_between(budgets, lo, hi, color="#2E7D32", alpha=0.18, linewidth=0,
                        label="Mean 95% CI (batch = 1)")
    ax.plot(budgets, lat, marker="o", color="#2E7D32", label="Mean (batch = 1)")
    ax.plot(budgets, lat_p95, marker="^", linestyle=":", color="#6A1B9A",
            label="P95 (batch = 1)")
    if "Default" in t2:
        ax.scatter([11.5], [t2["Default"]["e2e_ms_batch1"]], marker="*", s=90,
                   color="#2E7D32", zorder=5)
        ax.annotate("Default\n(8/10)", (11.5, t2["Default"]["e2e_ms_batch1"]),
                    ha="center", va="bottom", fontsize=7, color="0.3")
    ax.set_xlabel("Maximum ALM inner iterations")
    ax.set_ylabel("End-to-end latency (ms/trajectory)")
    ax.set_title("(b) Inference cost vs ALM budget", fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_xticks(range(2, 11, 2))
    ax.set_xlim(1, 12.5)
    ax.legend(loc="best", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (stem, ext), bbox_inches="tight")
    plt.close(fig)


def _stage_figure(args) -> None:
    """Re-plot from the saved tables only (no metric recomputation)."""
    t1 = C.load_csv(os.path.join(OUT, "table1_budget_safety.csv"))
    t2 = C.load_csv(os.path.join(OUT, "table2_budget_cost.csv"))
    for row in t1:
        for key in ("constraint_feasible_pct", "collision_pct"):
            row[key] = float(row[key]) if row.get(key) not in (None, "") else None
    for row in t2:
        for key in ("e2e_ms_batch1", "e2e_ms_batch1_p95"):
            row[key] = float(row[key]) if row.get(key) not in (None, "") else None
    _figure(t1, t2, os.path.join(OUT, "alm_budget_safety_runtime"))
    print("[exp3] figure redrawn", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["p0", "p1", "p1table", "run", "latency", "timing",
                             "table", "figure"])
    ap.add_argument("--configs", nargs="*", default=ORDER, choices=ORDER)
    ap.add_argument("--seeds", nargs="*", default=["0", "1", "2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--p1-limit", type=int, default=60)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--geometry-step", type=float, default=M.GEOMETRY_STEP_M)
    ap.add_argument("--b1-tasks", type=int, default=120)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"p0": _stage_p0, "p1": _stage_p1, "p1table": _stage_p1table,
     "run": _stage_run, "latency": _stage_latency,
     "timing": _stage_timing, "table": _stage_table,
     "figure": _stage_figure}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
