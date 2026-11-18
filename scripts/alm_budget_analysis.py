#!/usr/bin/env python
"""ALM 迭代预算对安全约束修正效果的影响（独立实验）。

设计
----
阶段 A：用论文正式配置运行一次基准采样，在**每次** `bspline_alm_correct()` 调用前捕获
        真实输入：`q_ref`、冻结约束包 `pack`、进入时的乘子 `lam_in`（原实现给 `None`
        就保留 `None`），以及场景/种子/扩散步/原始预算等元数据。
阶段 B：不再运行扩散网络。对每份缓存输入，用**完全相同**的 `q_ref / pack / lam_in /
        ALM 配置`，分别以 I1、I2、I4、I6、I8、I10 调用同一个 `bspline_alm_correct()`，
        只改最大迭代次数。**从不调用 `bspline_hard_project()`**，也不评价最终碰撞率或耗时。

指标
----
* 指标一（主）：整次调用的**所有有效约束**满足 `max_violation_after <= constraint_tol`
  记为可行。不使用代码里的 `constraint_feasible_rate`（那是单项约束满足比例）。
* 指标二：`max_violation_after` 的均值与 P95（场景单位，另存米制）。
* 指标三：`mean_curve_correction_m`（修正前后解码轨迹点的平均位移，米）。
* 辅助：实际迭代次数、有效调用数、基准已满足比例、无效约束包/排除数。

采集方式：运行期包装 `src.diffusion.sampler.bspline_alm_correct`（`sample()` 通过模块全局
名调用它），因此 **sampler.py 不需要改动**，正式采样默认行为与返回格式不变。

统计：per_call → 按 (sample_id, seed) 聚合 → 再按 seed 聚合成 per_scene → 以 420 个
测试场景为重采样单位做配对 Bootstrap。

用法
----
    python scripts/alm_budget_analysis.py --config configs/paper_modified.yaml \\
        --ckpt <A/best_task.pt> --processed <dataset> --split test --seeds 0,1,2 \\
        --budgets 1,2,4,6,8,10 --out outputs/alm_budget_analysis --rebuild-cache
    # 复用缓存（只重跑预算扫描与统计）
    python scripts/alm_budget_analysis.py ... --reuse-cache
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _extra in (ROOT, os.path.join(ROOT, "scripts")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from src.datasets.carla_spline_dataset import (CarlaSplineDataset,  # noqa: E402
                                               make_collate)
import src.diffusion.sampler as sampler_module  # noqa: E402
from src.diffusion.sampler import sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.bspline_constraints import BSplineConstraintPack  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_BUDGETS = (1, 2, 4, 6, 8, 10)
#: 论文主表推理协议（run_20261009）：16 步 DDIM、前 3 次前向预热、走廊 stride 用配置默认(1)
PROTOCOL = dict(steps=16, chunk=32, feedback_on=True, safe_to_ddim=True,
                final_project=False, region_stride=None)


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def git_rev(*flags: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", *flags], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json)
    os.replace(path + ".tmp", path)


def _json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value))


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return "%.10g" % value
    return value


def write_csv(path, rows, columns=None):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = columns or list(rows[0].keys())
    with open(path + ".tmp", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in columns})
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------
# stage A: capture the exact ALM call inputs
# ---------------------------------------------------------------------------


def _pack_to_cpu(pack):
    if pack.extraction.device.type == "cpu":
        return pack
    return BSplineConstraintPack(
        extraction=pack.extraction.detach().cpu().clone(),
        piece_A=pack.piece_A.detach().cpu().clone(),
        piece_b=pack.piece_b.detach().cpu().clone(),
        face_mask=pack.face_mask.detach().cpu().clone(),
        piece_mask=pack.piece_mask.detach().cpu().clone(),
        piece_region_id=pack.piece_region_id.detach().cpu().clone(),
        intervals=pack.intervals.detach().cpu().clone(),
        num_pieces=pack.num_pieces.detach().cpu().clone())


class AlmInputRecorder:
    """Wrap ``sampler.bspline_alm_correct`` + ``model.forward_all``.

    ``sample()`` resolves the ALM function as a module global, so replacing the
    module attribute intercepts every real call with EXACTLY the arguments the
    sampler passed -- including ``lam`` as given (``None`` stays ``None``) and the
    per-sample ``max_steps``.  ``forward_all`` is wrapped only to record the exact
    diffusion step ``t``.  Nothing inside the sampler is modified.
    """

    def __init__(self, tol_scene: float):
        self.tol = float(tol_scene)
        self.calls: list = []
        self.packs: dict = {}
        self.codec_basis = None
        self._orig = None
        self._forward_orig = None
        self._t = None
        self._ctx = None
        self._call_index: dict = {}

    def patch(self):
        self._orig = sampler_module.bspline_alm_correct

        def wrapped(q_ref, pack, codec, lam=None, config=None, inner_steps=3,
                    max_steps=None, collect_stats=True):
            if self.codec_basis is None and codec is not None:
                self.codec_basis = codec.basis.detach().cpu().clone()
            self._record(q_ref, pack, lam, max_steps)
            return self._orig(q_ref, pack, codec, lam, config, inner_steps,
                              max_steps, collect_stats)

        sampler_module.bspline_alm_correct = wrapped

    def unpatch(self):
        if self._orig is not None:
            sampler_module.bspline_alm_correct = self._orig
            self._orig = None

    def patch_model(self, model):
        self._forward_orig = model.forward_all

        def forward_wrapped(*args, **kwargs):
            try:
                self._t = int(torch.as_tensor(args[3]).reshape(-1)[0].item())
            except Exception:
                self._t = None
            return self._forward_orig(*args, **kwargs)

        model.forward_all = forward_wrapped

    def unpatch_model(self, model):
        if self._forward_orig is not None:
            model.forward_all = self._forward_orig
            self._forward_orig = None

    def begin_chunk(self, seed: int, chunk_start: int, sample_ids):
        self._ctx = dict(seed=int(seed), chunk_start=int(chunk_start),
                         sample_ids=[int(s) for s in sample_ids])

    def _record(self, q_ref, pack, lam, max_steps):
        if self._ctx is None or max_steps is None or pack is None:
            return
        key = "%d:%d" % (self._ctx["seed"], self._ctx["chunk_start"])
        if key not in self.packs:
            self.packs[key] = _pack_to_cpu(pack)
        budget = torch.as_tensor(max_steps).reshape(-1).cpu()
        q = q_ref.detach().cpu()
        lam_cpu = None if lam is None else lam.detach().cpu()
        for row, sample_id in enumerate(self._ctx["sample_ids"]):
            if row >= q.shape[0]:
                break
            traj = (self._ctx["seed"], int(sample_id))
            self._call_index[traj] = self._call_index.get(traj, 0) + 1
            self.calls.append(dict(
                seed=int(self._ctx["seed"]), sample_id=int(sample_id),
                chunk_start=int(self._ctx["chunk_start"]), packed_row=int(row),
                pack_key=key, alm_call_index=int(self._call_index[traj]),
                diffusion_t=(None if self._t is None else int(self._t)),
                original_budget=(int(budget[row]) if row < budget.numel()
                                 else 0),
                q_ref=q[row].clone(),
                lam_in=(None if lam_cpu is None else lam_cpu[row].clone())))


def build_cache(args):
    cfg = load_config(args.config)
    if args.processed:
        cfg.setdefault("data", {})["processed_root"] = args.processed
    processed = cfg["data"]["processed_root"]
    scene_to_meter = float(cfg["data"].get("scene_to_meter", 80.0))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                         else "cpu"))
    alm_cfg = dict(cfg.get("alm") or {})
    corridor_cfg = dict(cfg.get("corridor") or {})
    if PROTOCOL["region_stride"]:
        corridor_cfg["region_stride"] = PROTOCOL["region_stride"]
    alm_cfg["force_feedback_valid_zero"] = not PROTOCOL["feedback_on"]
    alm_cfg["safe_prediction_to_ddim"] = PROTOCOL["safe_to_ddim"]
    alm_cfg["final_project"] = PROTOCOL["final_project"]
    alm_cfg["guidance"] = "alm"
    tol = float(alm_cfg.get("constraint_tol", 1e-3))

    dataset = CarlaSplineDataset(args.split, processed, geometry_points=1280,
                                 limit=args.limit or None)
    collate = make_collate(dataset)
    dcfg = cfg["diffusion"]
    schedule = NoiseSchedule(int(PROTOCOL["steps"]),
                             beta_schedule=dcfg.get("beta_schedule",
                                                    "squaredcos_cap_v2"),
                             beta_start=float(dcfg.get("beta_start", 1e-4)),
                             beta_end=float(dcfg.get("beta_end", 0.02)))
    model, _, report = load_model(cfg, args.ckpt, device=device, verbose=False)
    bad = {k: (report or {}).get(k) for k in ("missing", "adapted", "unexpected")
           if report and report.get(k)}
    if bad:
        raise SystemExit("[cache] checkpoint/config mismatch: %s"
                         % {k: v[:4] for k, v in bad.items()})
    if getattr(model, "assert_shapes", False):
        model.assert_shapes = False
    model.eval()

    rec = AlmInputRecorder(tol)
    rec.patch()
    rec.patch_model(model)
    chunk = int(PROTOCOL["chunk"])
    n_total = len(dataset)
    t0 = time.time()
    try:
        with torch.no_grad():
            for start in range(0, n_total, chunk):
                idx = list(range(start, min(start + chunk, n_total)))
                batch = collate([dataset[i] for i in idx])
                occ = batch["occupancy"].to(device)
                cond = batch["cond"].to(device)
                ids = [int(s) for s in batch["sample_id"].tolist()]
                for seed in args.seed_list:
                    rec.begin_chunk(seed, start, ids)
                    sample(model, schedule, cond, occ,
                           batch["candidate_xy"].to(device),
                           batch["candidate_mask"].to(device),
                           batch["candidate_geometry"].to(device),
                           batch["candidate_geometry_lengths"].to(device),
                           device=str(device), steps=PROTOCOL["steps"],
                           seed=int(seed), seed_offset=start,
                           return_trace=False, alm_config=alm_cfg,
                           corridor_config=corridor_cfg, guidance="alm")
                print("[cache] %d/%d (%.0fs)"
                      % (min(start + chunk, n_total), n_total,
                         time.time() - t0), flush=True)
    finally:
        rec.unpatch_model(model)
        rec.unpatch()

    pack_ids = {}
    for call in rec.calls:
        pack_ids.setdefault(call["pack_key"], len(pack_ids))
    calls = []
    for call in rec.calls:
        item = dict(call)
        item["pack_id"] = pack_ids[item.pop("pack_key")]
        calls.append(item)
    payload = dict(calls=calls,
                   packs={str(i): rec.packs[k] for k, i in pack_ids.items()},
                   codec_basis=rec.codec_basis)
    manifest = dict(
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        branch=git_rev("--abbrev-ref", "HEAD"), commit=git_rev("HEAD"),
        config=os.path.abspath(args.config),
        checkpoint=os.path.abspath(args.ckpt),
        checkpoint_hash=file_hash(args.ckpt),
        processed_dataset=os.path.abspath(processed),
        split=args.split, seeds=list(args.seed_list), n_samples=n_total,
        n_calls=len(calls), n_packs=len(payload["packs"]),
        protocol=dict(PROTOCOL), n_ddim_steps=int(PROTOCOL["steps"]),
        chunk=chunk, constraint_tol_scene=tol,
        constraint_tol_m=tol * scene_to_meter, scene_to_meter=scene_to_meter,
        alm_settings={k: alm_cfg.get(k) for k in
                      ("rho", "step_size", "proximity_weight",
                       "correction_smooth_weight", "correction_smooth_kernel",
                       "correction_smooth_sigma", "max_curve_step_scene",
                       "activation_inner_steps", "inner_steps",
                       "inherit_dual_across_reverse_steps", "constraint_tol",
                       "warmup_reverse_steps", "final_project")},
        device=str(device), python=sys.executable, torch=torch.__version__,
        elapsed_s=round(time.time() - t0, 1),
        capture=("runtime wrapper around src.diffusion.sampler."
                 "bspline_alm_correct; sampler.py unmodified"))
    return payload, manifest


# ---------------------------------------------------------------------------
# stage B: replay with different budgets
# ---------------------------------------------------------------------------


class _CodecShim:
    """Only ``codec.basis`` is used by ``bspline_alm_correct``."""

    def __init__(self, basis: torch.Tensor):
        self.basis = basis


def _run_alm(codec, pack_batch, row, q_ref, lam_in, alm_cfg, budgets,
             device):
    n_b = len(budgets)
    pack = pack_batch.index_select(torch.tensor([row] * n_b)).to(device)
    q = q_ref.to(device).unsqueeze(0).repeat(n_b, 1, 1)
    lam = (None if lam_in is None
           else lam_in.to(device).unsqueeze(0).repeat(n_b, 1, 1, 1))
    max_steps = torch.tensor(budgets, dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, stats = sampler_module.bspline_alm_correct(
            q, pack, codec, lam, alm_cfg,
            inner_steps=int(max(budgets)), max_steps=max_steps)
    return dict(
        v_before=stats["max_violation_before"].detach().cpu().numpy(),
        v_after=stats["max_violation_after"].detach().cpu().numpy(),
        corr_m=stats["mean_curve_correction_m"].detach().cpu().numpy(),
        used=stats["inner_steps_used"].detach().cpu().numpy(),
        rate=stats["constraint_feasible_rate"].detach().cpu().numpy())


def parity_check(payload, alm_cfg, budgets, tol, n_calls=8, device="cpu"):
    """Batched-over-budgets vs one-budget-at-a-time must agree exactly."""
    codec = _CodecShim(payload["codec_basis"])
    dev = torch.device(device)
    diffs = []
    calls = payload["calls"][:n_calls]
    for call in calls:
        pack_batch = payload["packs"][str(call["pack_id"])]
        row = int(call["packed_row"])
        if int(pack_batch.num_pieces[row].item()) <= 0:
            continue
        batched = _run_alm(codec, pack_batch, row, call["q_ref"],
                           call["lam_in"], alm_cfg, budgets, dev)
        for i, budget in enumerate(budgets):
            single = _run_alm(codec, pack_batch, row, call["q_ref"],
                              call["lam_in"], alm_cfg, [budget], dev)
            diffs.append(abs(float(batched["v_after"][i])
                             - float(single["v_after"][0])))
            diffs.append(abs(float(batched["corr_m"][i])
                             - float(single["corr_m"][0])))
            diffs.append(abs(float(batched["used"][i])
                             - float(single["used"][0])))
    return dict(n_replays=len(diffs) // 3,
                max_abs_diff=(max(diffs) if diffs else 0.0),
                ok=bool(diffs and max(diffs) <= 1e-5))


def replay(payload, manifest, alm_cfg, budgets, device="cpu",
           use_batching=True):
    codec = _CodecShim(payload["codec_basis"])
    dev = torch.device(device)
    tol = float(manifest["constraint_tol_scene"])
    s2m = float(manifest["scene_to_meter"])
    rows = []
    t0 = time.time()
    calls = payload["calls"]
    for n, call in enumerate(calls, start=1):
        pack_batch = payload["packs"][str(call["pack_id"])]
        row = int(call["packed_row"])
        num_pieces = int(pack_batch.num_pieces[row].item())
        face_valid = int(pack_batch.face_mask[row].sum().item())
        pack_valid = num_pieces > 0 and face_valid > 0
        base = dict(seed=call["seed"], sample_id=call["sample_id"],
                    alm_call_index=call["alm_call_index"],
                    diffusion_t=call["diffusion_t"],
                    original_budget=call["original_budget"],
                    num_pieces=num_pieces, face_count=face_valid,
                    pack_valid=pack_valid,
                    is_first_activation=bool(call["alm_call_index"] == 1),
                    lam_in_is_none=(call["lam_in"] is None))
        if not pack_valid or int(call["original_budget"]) <= 0:
            for budget in budgets:
                rows.append(dict(base, budget=int(budget),
                                 status=("excluded_no_pack" if not pack_valid
                                         else "excluded_not_guided"),
                                 violation_before_scene=None,
                                 violation_after_scene=None, feasible=None,
                                 iterations_used=None,
                                 mean_curve_correction_m=None))
            continue
        if use_batching:
            out = _run_alm(codec, pack_batch, row, call["q_ref"],
                           call["lam_in"], alm_cfg, budgets, dev)
            idx = list(range(len(budgets)))
        else:
            out = {k: [] for k in ("v_before", "v_after", "corr_m", "used",
                                   "rate")}
            for budget in budgets:
                single = _run_alm(codec, pack_batch, row, call["q_ref"],
                                  call["lam_in"], alm_cfg, [budget], dev)
                for key in out:
                    out[key].append(float(single[key][0]))
            idx = list(range(len(budgets)))
        for i, budget in zip(idx, budgets):
            vb, va = float(out["v_before"][i]), float(out["v_after"][i])
            rows.append(dict(
                base, budget=int(budget), status="ok",
                violation_before_scene=vb, violation_after_scene=va,
                violation_before_m=vb * s2m, violation_after_m=va * s2m,
                feasible=bool(va <= tol),
                iterations_used=int(out["used"][i]),
                mean_curve_correction_m=float(out["corr_m"][i]),
                single_constraint_rate=float(out["rate"][i])))
        if n % 1000 == 0:
            print("[replay] %d/%d calls (%.0fs)" % (n, len(calls),
                                                    time.time() - t0),
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(values, n_boot, seed=0):
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return dict(mean=float(vals.mean()), lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)), n=int(vals.size))


def aggregate(call_rows, budgets, tol, n_boot):
    """per_call -> per_trajectory -> per_scene -> summary."""
    ok = [r for r in call_rows if r["status"] == "ok"]
    by_traj = {}
    for r in ok:
        by_traj.setdefault((r["seed"], r["sample_id"]), []).append(r)
    per_traj = []
    for (seed, sample_id), calls in sorted(by_traj.items()):
        for budget in budgets:
            part = [c for c in calls if c["budget"] == budget]
            if not part:
                continue
            per_traj.append(dict(
                seed=seed, sample_id=sample_id, budget=budget,
                n_calls=len(part),
                feasible_fraction=float(np.mean([c["feasible"] for c in part])),
                violation_after_mean_scene=float(np.mean(
                    [c["violation_after_scene"] for c in part])),
                violation_after_p95_scene=float(np.percentile(
                    [c["violation_after_scene"] for c in part], 95)),
                correction_mean_m=float(np.mean(
                    [c["mean_curve_correction_m"] for c in part])),
                iterations_mean=float(np.mean(
                    [c["iterations_used"] for c in part]))))
    by_scene = {}
    for row in per_traj:
        by_scene.setdefault(row["sample_id"], {}).setdefault(
            row["budget"], []).append(row)
    per_scene = []
    for sample_id, per_budget in sorted(by_scene.items()):
        for budget, part in sorted(per_budget.items()):
            per_scene.append(dict(
                sample_id=sample_id, budget=budget, n_seeds=len(part),
                feasible_fraction=float(np.mean(
                    [p["feasible_fraction"] for p in part])),
                violation_after_mean_scene=float(np.mean(
                    [p["violation_after_mean_scene"] for p in part])),
                violation_after_p95_scene=float(np.mean(
                    [p["violation_after_p95_scene"] for p in part])),
                correction_mean_m=float(np.mean(
                    [p["correction_mean_m"] for p in part])),
                iterations_mean=float(np.mean(
                    [p["iterations_mean"] for p in part]))))
    summary = []
    for budget in budgets:
        part = [r for r in per_scene if r["budget"] == budget]
        if not part:
            continue
        calls_b = [c for c in ok if c["budget"] == budget]
        summary.append(dict(
            budget=int(budget), n_scenes=len(part), n_calls=len(calls_b),
            n_calls_feasible=sum(1 for c in calls_b if c["feasible"]),
            feasible_rate_pct=100.0 * float(np.mean(
                [r["feasible_fraction"] for r in part])),
            feasible_ci=bootstrap_ci(
                [r["feasible_fraction"] * 100.0 for r in part], n_boot),
            global_call_feasible_pct=(100.0 * sum(1 for c in calls_b
                                                  if c["feasible"])
                                      / max(len(calls_b), 1)),
            violation_after_mean_scene=float(np.mean(
                [r["violation_after_mean_scene"] for r in part])),
            violation_ci_scene=bootstrap_ci(
                [r["violation_after_mean_scene"] for r in part], n_boot),
            violation_after_p95_scene=float(np.mean(
                [r["violation_after_p95_scene"] for r in part])),
            violation_p95_ci_scene=bootstrap_ci(
                [r["violation_after_p95_scene"] for r in part], n_boot),
            correction_mean_m=float(np.mean(
                [r["correction_mean_m"] for r in part])),
            correction_ci_m=bootstrap_ci(
                [r["correction_mean_m"] for r in part], n_boot),
            iterations_mean=float(np.mean(
                [r["iterations_mean"] for r in part]))))
    return per_traj, per_scene, summary


def flatten_summary(summary):
    rows = []
    for s in summary:
        rows.append(dict(
            budget=s["budget"], n_scenes=s["n_scenes"], n_calls=s["n_calls"],
            n_calls_feasible=s["n_calls_feasible"],
            feasible_rate_pct=s["feasible_rate_pct"],
            feasible_ci_lo=s["feasible_ci"]["lo"],
            feasible_ci_hi=s["feasible_ci"]["hi"],
            global_call_feasible_pct=s["global_call_feasible_pct"],
            violation_after_mean_scene=s["violation_after_mean_scene"],
            violation_ci_lo_scene=s["violation_ci_scene"]["lo"],
            violation_ci_hi_scene=s["violation_ci_scene"]["hi"],
            violation_after_p95_scene=s["violation_after_p95_scene"],
            violation_p95_ci_lo_scene=s["violation_p95_ci_scene"]["lo"],
            violation_p95_ci_hi_scene=s["violation_p95_ci_scene"]["hi"],
            correction_mean_m=s["correction_mean_m"],
            correction_ci_lo_m=s["correction_ci_m"]["lo"],
            correction_ci_hi_m=s["correction_ci_m"]["hi"],
            iterations_mean=s["iterations_mean"]))
    return rows


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def make_figure(summary, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.linewidth": 0.8,
                         "lines.linewidth": 1.4, "legend.frameon": False,
                         "figure.dpi": 300})
    b = [s["budget"] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    ax.fill_between(b, [s["feasible_ci"]["lo"] for s in summary],
                    [s["feasible_ci"]["hi"] for s in summary],
                    color="#1f4e79", alpha=0.18, linewidth=0)
    ax.plot(b, [s["feasible_rate_pct"] for s in summary], marker="o",
            color="#1f4e79")
    ax.set_xlabel("Maximum ALM inner iterations")
    ax.set_ylabel("Constraint-feasible calls (%)")
    ax.set_title("(a) Bézier constraint feasibility", fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xticks(b)
    ax.set_xticklabels([str(x) for x in b])
    ax = axes[1]
    ax.fill_between(b, [s["violation_ci_scene"]["lo"] for s in summary],
                    [s["violation_ci_scene"]["hi"] for s in summary],
                    color="#2E7D32", alpha=0.18, linewidth=0)
    ax.plot(b, [s["violation_after_mean_scene"] for s in summary], marker="o",
            color="#2E7D32", label="Mean (95% CI)")
    ax.plot(b, [s["violation_after_p95_scene"] for s in summary], marker="^",
            linestyle=":", color="#6A1B9A", label="P95")
    ax.set_xlabel("Maximum ALM inner iterations")
    ax.set_ylabel("Max constraint violation after ALM (scene units)")
    ax.set_title("(b) Residual violation", fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xticks(b)
    ax.set_xticklabels([str(x) for x in b])
    ax.legend(loc="best", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (stem, ext), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budgets", default="1,2,4,6,8,10")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs",
                                                  "alm_budget_analysis"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--allow-commit-drift", action="store_true",
                    help="reuse a cache built on another commit; only valid for "
                         "the replay/statistics stage, since ALM replay never "
                         "touches the network")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--replay-device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--no-parity-check", action="store_true")
    args = ap.parse_args()
    args.seed_list = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    args.budget_list = [int(x) for x in
                        str(args.budgets).replace(",", " ").split()]
    out = os.path.abspath(args.out)
    cache_dir = os.path.abspath(args.cache_dir or os.path.join(out, "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    inputs_path = os.path.join(cache_dir, "alm_inputs.pt")
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")

    if args.reuse_cache and not args.rebuild_cache:
        if not (os.path.exists(inputs_path) and os.path.exists(manifest_path)):
            raise SystemExit("[cache] --reuse-cache but no cache found")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        # 方案要求缓存与 checkpoint、配置、数据集和代码版本一致才能复用。
        # 配置按**内容**校验（ALM 超参 + 容差），而不是按文件路径，这样同一个
        # 缓存在不同 checkout（例如 v3 worktree vs 当前分支）上仍可做只读回放。
        cur_alm = {k: (load_config(args.config).get("alm") or {}).get(k)
                   for k in manifest["alm_settings"]}
        checks = dict(
            checkpoint_hash=file_hash(args.ckpt), split=args.split,
            seeds=list(args.seed_list),
            processed_dataset=(os.path.abspath(args.processed)
                               if args.processed
                               else manifest.get("processed_dataset")),
            alm_settings=cur_alm)
        drift = {k: (manifest.get(k), v) for k, v in checks.items()
                 if manifest.get(k) != v}
        if drift:
            raise SystemExit("[cache] refusing to reuse stale cache "
                             "(checkpoint/config/data/budgets differ): %s "
                             "(use --rebuild-cache)" % drift)
        if manifest.get("commit") != git_rev("HEAD"):
            note = ("commit %s -> %s" % (str(manifest.get("commit"))[:10],
                                         git_rev("HEAD")[:10]))
            if args.allow_commit_drift:
                print("[cache] WARNING commit drift allowed explicitly (%s); "
                      "only the ALM replay is re-run, the cached inputs are "
                      "unaffected" % note, flush=True)
            else:
                raise SystemExit("[cache] cache was built on another commit "
                                 "(%s); pass --allow-commit-drift to replay it "
                                 "anyway (the replay never touches the network)"
                                 % note)
        payload = torch.load(inputs_path, map_location="cpu",
                             weights_only=False)
        print("[cache] reused %d calls" % len(payload["calls"]), flush=True)
    else:
        payload, manifest = build_cache(args)
        torch.save(payload, inputs_path)
        write_json(manifest_path, manifest)
        print("[cache] wrote %d calls / %d packs (%.1f MB)"
              % (len(payload["calls"]), len(payload["packs"]),
                 os.path.getsize(inputs_path) / 2 ** 20), flush=True)

    budgets = list(args.budget_list)
    alm_cfg = dict(manifest["alm_settings"])
    alm_cfg["scene_to_meter"] = float(manifest["scene_to_meter"])
    tol = float(manifest["constraint_tol_scene"])

    parity = None
    use_batching = True
    if not args.no_parity_check:
        parity = parity_check(payload, alm_cfg, budgets, tol)
        print("[parity] %s" % parity, flush=True)
        if not parity["ok"]:
            use_batching = False
            print("[parity] batching deviates -> falling back to one-budget-"
                  "at-a-time replay", flush=True)

    call_rows = replay(payload, manifest, alm_cfg, budgets,
                       device=args.replay_device, use_batching=use_batching)
    write_csv(os.path.join(out, "per_call_all.csv"), call_rows)

    main_rows = [r for r in call_rows if r["status"] == "ok"
                 and r["violation_before_scene"] > tol]
    per_traj, per_scene, summary = aggregate(main_rows, budgets, tol,
                                             args.bootstrap)
    _, _, summary_all = aggregate([r for r in call_rows if r["status"] == "ok"],
                                  budgets, tol, args.bootstrap)
    write_csv(os.path.join(out, "per_call.csv"), main_rows)
    write_csv(os.path.join(out, "per_trajectory.csv"), per_traj)
    write_csv(os.path.join(out, "per_scene.csv"), per_scene)
    summary_rows = flatten_summary(summary)
    write_csv(os.path.join(out, "summary.csv"), summary_rows)
    write_csv(os.path.join(out, "summary_all_inputs.csv"),
              flatten_summary(summary_all))

    # ---- exclusion / coverage bookkeeping (over unique calls)
    uniq = {}
    for r in call_rows:
        key = (r["seed"], r["sample_id"], r["alm_call_index"])
        uniq.setdefault(key, r)
    n_cached = len(uniq)
    n_ok = sum(1 for r in uniq.values() if r["status"] == "ok")
    n_no_pack = sum(1 for r in uniq.values()
                    if r["status"] == "excluded_no_pack")
    n_unguided = sum(1 for r in uniq.values()
                     if r["status"] == "excluded_not_guided")
    n_infeasible = sum(1 for r in uniq.values() if r["status"] == "ok"
                       and r["violation_before_scene"] > tol)
    stats_block = dict(
        n_cached_calls=n_cached, n_valid_calls=n_ok,
        n_excluded_no_pack=n_no_pack, n_excluded_not_guided=n_unguided,
        n_calls_baseline_infeasible=n_infeasible,
        baseline_satisfied_pct=(100.0 * (n_ok - n_infeasible) / n_ok
                                if n_ok else None),
        n_scenes_total=len({k[1] for k in uniq}),
        n_scenes_main=len({r["sample_id"] for r in per_scene}),
        n_trajectories_main=len({(r["seed"], r["sample_id"]) for r in per_traj}),
        n_lam_none=sum(1 for r in uniq.values() if r.get("lam_in_is_none")),
        replay_mode=("batched_over_budgets" if use_batching
                     else "one_budget_at_a_time"),
        parity=parity)
    write_json(os.path.join(out, "run_config.json"), dict(
        manifest=manifest, budgets=budgets, bootstrap=args.bootstrap,
        analysis_set="main (baseline-infeasible inputs, fixed across budgets)",
        main_set_rule="violation_before_scene > constraint_tol",
        feasibility_rule=("max_violation_after <= constraint_tol over ALL valid "
                          "Bézier constraints (not constraint_feasible_rate)"),
        unit_note=("violations kept in scene units; metres = scene * "
                   "scene_to_meter from the config (verified)"),
        stats=stats_block))

    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)
    if summary:
        make_figure(summary, os.path.join(figs, "alm_budget_analysis"))

    lines = ["# ALM 迭代预算对安全约束修正效果的影响", "", "## 设置", "",
             "- checkpoint：`%s`（sha256 `%s`）" % (manifest["checkpoint"],
                                                   manifest["checkpoint_hash"]),
             "- 代码：branch `%s` @ `%s`" % (manifest["branch"],
                                             str(manifest["commit"])[:10]),
             "- 数据：`%s`；split=%s；样本 %d；种子 %s" % (
                 manifest["processed_dataset"], manifest["split"],
                 manifest["n_samples"], manifest["seeds"]),
             "- 基准采样：%d 步 DDIM；预热 %s；chunk=%d；缓存由运行期包装器采集"
             "（sampler.py 未改）" % (manifest["n_ddim_steps"],
                                     manifest["alm_settings"].get(
                                         "warmup_reverse_steps"),
                                     manifest["chunk"]),
             "- ALM 超参：rho=%s step_size=%s proxim=%s smooth_w=%s "
             "max_curve_step=%s" % (
                 manifest["alm_settings"].get("rho"),
                 manifest["alm_settings"].get("step_size"),
                 manifest["alm_settings"].get("proximity_weight"),
                 manifest["alm_settings"].get("correction_smooth_weight"),
                 manifest["alm_settings"].get("max_curve_step_scene")),
             "- 容差：`constraint_tol = %.3g` scene = %.4g m" % (
                 manifest["constraint_tol_scene"],
                 manifest["constraint_tol_m"]),
             "- 缓存调用：**%d**；有效 **%d**；其中基准输入已违反 **%d**"
             "（已满足 %.1f%%）" % (
                 stats_block["n_cached_calls"], stats_block["n_valid_calls"],
                 stats_block["n_calls_baseline_infeasible"],
                 stats_block["baseline_satisfied_pct"] or 0.0),
             "- 排除：约束包为空 **%d**；未被引导(预算 0) **%d**" % (
                 stats_block["n_excluded_no_pack"],
                 stats_block["n_excluded_not_guided"]),
             "- 主分析集：基准输入违反约束的调用；有效场景 **%d** / %d，"
             "有效轨迹 **%d**" % (stats_block["n_scenes_main"],
                                 stats_block["n_scenes_total"],
                                 stats_block["n_trajectories_main"]),
             "- 回放模式：`%s`；批量等价性校验：%s" % (
                 stats_block["replay_mode"],
                 ("未执行" if parity is None else
                  "最大偏差 %.2e（%s）" % (parity["max_abs_diff"],
                                          "通过" if parity["ok"] else "未通过"))),
             "", "## 结果（主分析集，按场景配对 Bootstrap 95% CI）", "",
             "| 最大迭代 | 可行率 (%) | 95% CI | 最大违反量 (scene) | 95% CI | "
             "P95 | 平均修正幅度 (m) | 实际迭代 | 调用数 |",
             "|---|---:|---|---:|---|---:|---:|---:|---:|"]
    for s in summary_rows:
        lines.append("| %d | %.2f | [%.2f, %.2f] | %.5f | [%.5f, %.5f] | "
                     "%.5f | %.5f | %.2f | %d |" % (
                         s["budget"], s["feasible_rate_pct"],
                         s["feasible_ci_lo"], s["feasible_ci_hi"],
                         s["violation_after_mean_scene"],
                         s["violation_ci_lo_scene"], s["violation_ci_hi_scene"],
                         s["violation_after_p95_scene"],
                         s["correction_mean_m"], s["iterations_mean"],
                         s["n_calls"]))
    lines += ["", "全部输入（含基准已满足的调用）见 `summary_all_inputs.csv`。",
              "单位说明：违反量以场景单位记录；米制换算仅在 `scene_to_meter` "
              "核对后使用（本表为场景单位）。", ""]
    if len(summary_rows) >= 2:
        f, l = summary_rows[0], summary_rows[-1]
        lines += ["## 趋势", "",
                  "- 可行率：I%d = %.2f%% → I%d = %.2f%%（%+.2f 个百分点）" % (
                      f["budget"], f["feasible_rate_pct"], l["budget"],
                      l["feasible_rate_pct"],
                      l["feasible_rate_pct"] - f["feasible_rate_pct"]),
                  "- 最大违反量：I%d = %.5f → I%d = %.5f" % (
                      f["budget"], f["violation_after_mean_scene"], l["budget"],
                      l["violation_after_mean_scene"]),
                  "- 实际迭代：I%d = %.2f（上限 %d）→ I%d = %.2f（上限 %d）" % (
                      f["budget"], f["iterations_mean"], f["budget"], l["budget"],
                      l["iterations_mean"], l["budget"]), ""]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("[done] %s" % out, flush=True)
    for s in summary_rows:
        print("  I%-3d feasible=%6.2f%% [%.2f,%.2f] viol=%.5f corr=%.5f "
              "iters=%.2f n=%d" % (
                  s["budget"], s["feasible_rate_pct"], s["feasible_ci_lo"],
                  s["feasible_ci_hi"], s["violation_after_mean_scene"],
                  s["correction_mean_m"], s["iterations_mean"],
                  s["n_calls"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
