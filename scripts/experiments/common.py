"""Shared harness for 实验 1/2/3.

The only job of this module is to run ONE (method, seed, split) evaluation and
return per-task rows plus the raw B-spline control polygons, so that

* every experiment sees exactly the same inference code path
  (``src.diffusion.sampler.sample`` through ``scripts/eval_chapter4.py``'s
  helpers -- imported, never re-implemented), and
* the new metrics of 实验 1 are computed ONCE, offline, from the saved control
  polygons, instead of being re-derived inside each evaluation script.

Defaults are the historical campaign defaults, so nothing here changes the
behaviour of the existing scripts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from eval_chapter4 import DENSE_POINTS, evaluate_controls  # noqa: E402
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,  # noqa: E402
                                               make_collate)
from src.diffusion.sampler import effective_warmup_steps, sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.sdf_field import build_occupancy_sdf  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

def _first_existing(*candidates: str) -> str:
    """First candidate path that exists (the repo tree was reorganised once)."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


#: canonical campaign artefacts (实验 1 §2 / §5.1).  The repository tree has
#: been reorganised more than once, so both known locations are tried.
CAMPAIGN = _first_existing(
    os.path.join(_ROOT, "outputs", "paper_modified",
                 "run_20261009_no_projection"),
    os.path.join(_ROOT, "outputs", "paper_ral",
                 "run_20261009_no_projection"))
CKPT_D = os.path.join(CAMPAIGN, "D", "ckpt", "best_task.pt")
CKPT_S = os.path.join(CAMPAIGN, "S", "ckpt", "best_task.pt")
CKPT_A = os.path.join(CAMPAIGN, "A", "ckpt", "best_task.pt")
CONFIG_D = os.path.join(_ROOT, "configs", "paper_modified_d.yaml")
CONFIG_S = os.path.join(_ROOT, "configs", "paper_modified_s.yaml")
CONFIG_A = os.path.join(_ROOT, "configs", "paper_modified.yaml")
#: the three SDF-guidance hyper-parameters chosen on the VALIDATION set by
#: ``scripts/ch4_sweep_val.py`` (never on the test split)
SWEEP_DIR = os.path.join(CAMPAIGN, "sweeps")
#: experiment outputs -- prefer the organised ``论文整合/实验/results`` tree
OUT_ROOT = _first_existing(
    os.path.join(_ROOT, "论文整合", "实验", "results"),
    os.path.join(_ROOT, "实验", "results"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sweep_overrides(mode: str) -> dict:
    """Validation-selected SDF guidance parameters for ``penalty``/``...``."""
    path = os.path.join(SWEEP_DIR, "%s.json" % mode)
    if not os.path.exists(path):
        raise SystemExit("missing validation sweep %s" % path)
    with open(path, encoding="utf-8") as handle:
        winner = json.load(handle).get("winner") or {}
    return dict(winner)


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


@dataclass
class Context:
    cfg: dict
    model: torch.nn.Module
    schedule: NoiseSchedule
    dataset: CarlaSplineDataset
    device: torch.device
    scene_to_meter: float
    config_path: str
    ckpt_path: str
    alm_cfg: dict
    corridor_cfg: dict
    sdf_cfg: dict
    release_notes: dict = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.dataset)

    @property
    def codec(self):
        return self.model.bspline


def build_context(config_path: str, ckpt_path: str, *, split: str = "test",
                  device: str | None = None, processed: str | None = None,
                  limit: int | None = None, keep_assert_shapes: bool = False,
                  allow_partial_load: bool = False,
                  geometry_points: int = 1280) -> Context:
    """Load config + checkpoint + dataset exactly like ``eval_chapter4``."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    cfg = load_config(config_path)
    if processed:
        cfg.setdefault("data", {})["processed_root"] = processed
    scene_to_meter = float(cfg["data"].get("scene_to_meter", 80.0))
    dataset = CarlaSplineDataset(split, cfg["data"]["processed_root"],
                                 geometry_points=geometry_points, limit=limit)
    dcfg = cfg["diffusion"]
    schedule = NoiseSchedule(
        int(dcfg.get("timesteps", 16)),
        beta_schedule=dcfg.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=float(dcfg.get("beta_start", 1e-4)),
        beta_end=float(dcfg.get("beta_end", 0.02)))
    model, _, report = load_model(cfg, ckpt_path, device=dev, verbose=False)
    if not keep_assert_shapes and getattr(model, "assert_shapes", False):
        model.assert_shapes = False
    if report is not None and not allow_partial_load:
        bad = {k: report.get(k) for k in ("missing", "adapted", "unexpected")
               if report.get(k)}
        if bad:
            raise SystemExit("checkpoint/config mismatch for %s: %s"
                             % (ckpt_path, {k: v[:5] for k, v in bad.items()}))
    model.eval()
    return Context(cfg=cfg, model=model, schedule=schedule, dataset=dataset,
                   device=dev, scene_to_meter=scene_to_meter,
                   config_path=config_path, ckpt_path=ckpt_path,
                   alm_cfg=dict(cfg.get("alm") or {}),
                   corridor_cfg=dict(cfg.get("corridor") or {}),
                   sdf_cfg={})


def collate_fn(ctx: Context):
    return make_collate(ctx.dataset)


# ---------------------------------------------------------------------------
# one evaluation run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    rows: list
    controls: np.ndarray          # [N,C,2] float32, scene units
    reference: np.ndarray         # [N,H,2] float32, curve_gt (scene units)
    endpoints: np.ndarray         # [N,2,2] float32, [start, goal]
    header: dict
    timing: dict

    def column(self, name):
        return np.asarray([r.get(name) for r in self.rows], dtype=object)


def run_split(ctx: Context, *, method: str, guidance: str,
              alm_overrides: dict | None = None,
              corridor_overrides: dict | None = None,
              sdf_overrides: dict | None = None,
              steps: int = 16, seed: int = 0, chunk: int = 32,
              return_trace: bool = True,
              capture_corridors: bool = False,
              chunk_hook=None, progress: bool = True) -> RunResult:
    """Run one ``(method, guidance, seed)`` over the loaded split.

    ``chunk_hook(payload)`` is called after every chunk with the raw tensors
    (trace, corridors, batch, occupancy) -- that is how 实验 2/3 collect their
    process-level and geometry-level logs without touching the sampler.

    ``return_trace=False`` disables the per-step trace bookkeeping. The trace
    clones ~15 tensors and decodes three curves per reverse step, which costs
    about 3.7x the wall clock of the sampling itself; use it for any timing
    measurement (the historical campaign always timed with it off).
    """
    alm_cfg = dict(ctx.alm_cfg)
    alm_cfg.update(alm_overrides or {})
    alm_cfg["guidance"] = guidance
    corridor_cfg = dict(ctx.corridor_cfg)
    corridor_cfg.update(corridor_overrides or {})
    sdf_cfg = dict(ctx.sdf_cfg)
    sdf_cfg.update(sdf_overrides or {})
    warmup = effective_warmup_steps(
        int(alm_cfg.get("warmup_reverse_steps", 3)), steps)

    collate = collate_fn(ctx)
    model, dev, sm = ctx.model, ctx.device, ctx.scene_to_meter
    n_total = ctx.n_total
    rows, controls, reference, endpoints = [], [], [], []
    timing = {"wall_s": 0.0, "per_traj_ms": []}
    n_chunks = 0
    header = {
        "method": method, "guidance": guidance, "seed": seed,
        "split": getattr(ctx.dataset, "split", "?"),
        "steps": steps, "warmup_reverse_steps": warmup, "chunk": chunk,
        "n_samples": n_total,
        "config": ctx.config_path, "checkpoint": ctx.ckpt_path,
        "checkpoint_hash": file_hash(ctx.ckpt_path),
        "git_commit": git_commit(), "scene_to_meter": sm,
        "alm_overrides": {k: v for k, v in (alm_overrides or {}).items()},
        "corridor_overrides": {k: v for k, v in (corridor_overrides or {}).items()},
        "sdf_overrides": {k: v for k, v in (sdf_overrides or {}).items()},
        "dense_points": DENSE_POINTS,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for start in range(0, n_total, chunk):
        idx = list(range(start, min(start + chunk, n_total)))
        batch = collate([ctx.dataset[i] for i in idx])
        occ = batch["occupancy"].to(dev)
        cond = batch["cond"].to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sample(model, ctx.schedule, cond, occ,
                     batch["candidate_xy"].to(dev),
                     batch["candidate_mask"].to(dev),
                     batch["candidate_geometry"].to(dev),
                     batch["candidate_geometry_lengths"].to(dev),
                     device=str(dev), steps=steps, seed=seed,
                     seed_offset=start, return_trace=return_trace,
                     alm_config=alm_cfg, corridor_config=corridor_cfg,
                     guidance=guidance, sdf_config=sdf_cfg)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        timing["wall_s"] += wall
        n_chunks += 1
        if n_chunks > 3:
            timing["per_traj_ms"].append(wall * 1000.0 / len(idx))
        # per-trajectory inference cost of THIS chunk (None during warm-up).
        # With ``chunk=1`` this is the single-sample online latency.
        chunk_ms = (wall * 1000.0 / len(idx)) if n_chunks > 3 else None

        sdf = build_occupancy_sdf(
            occ, scene_to_meter=sm,
            map_margin=float(sdf_cfg.get("map_margin",
                                         alm_cfg.get("map_margin", 0.0))))
        metrics = evaluate_controls(model.bspline, out["control"], occ, cond,
                                   batch["pos"].to(dev), sdf, out.get("pack"),
                                   batch, sm)
        q = out["control"].detach().cpu().numpy().astype(np.float64)
        guided = out["guided"].detach().cpu().numpy().astype(bool)
        alm_status = [str(s) for s in (out.get("alm_status") or [""] * len(idx))]
        act_raw = out.get("activation_step")
        act_step = (act_raw.detach().cpu().numpy() if act_raw is not None
                    else np.full(len(idx), -1))
        corridors = out.get("corridors") if capture_corridors else None

        for local, index in enumerate(idx):
            row = {
                "method": method, "guidance": guidance, "seed": seed,
                "index": int(index),
                "sample_id": int(batch["sample_id"][local]),
                "collision": bool(metrics["collision"][local]),
                "sdf_min_distance_m": float(metrics["sdf_min_distance_m"][local]),
                "sdf_violation_m": float(metrics["sdf_violation_m"][local]),
                "sdf_feasible": bool(metrics["sdf_feasible"][local]),
                "rmse_m": float(metrics["rmse_m"][local]),
                "smoothness_m_legacy": float(metrics["smoothness_m"][local]),
                "corridor_guided": bool(guided[local]),
                "activation_step": int(act_step[local]),
                "alm_status": alm_status[local] if local < len(alm_status) else "",
                "endpoint_error_m": float(metrics["endpoint_error"][local]) * sm,
                "map_excursion": bool(metrics["map_excursion"][local]),
                "inference_ms": chunk_ms,
            }
            if capture_corridors:
                corr = (corridors[local] if corridors is not None else None)
                row["corridor_valid"] = bool(corr["valid"]) if corr else False
                row["corridor_failure"] = (corr["failure_reason"]
                                           if corr else "no_corridor")
                row["base_cell_count"] = (int(corr["base_cell_count"])
                                          if corr else 0)
                row["bridge_cell_count"] = (int(corr["bridge_cell_count"])
                                            if corr else 0)
                row["corridor_cells"] = corr
            rows.append(row)

        controls.append(q.astype(np.float32))
        reference.append(batch["pos"].cpu().numpy().astype(np.float32))
        endpoints.append(batch["cond"].cpu().numpy().astype(np.float32))
        if chunk_hook is not None:
            chunk_hook(dict(rows_index=[int(i) for i in idx], batch=batch,
                            occ=occ, cond=cond, out=out, trace=out.get("trace"),
                            metrics=metrics, sdf=sdf, start=start))
        if progress:
            print("[run] %s/%s s%d %d/%d" % (method, guidance, seed,
                                             min(start + chunk, n_total),
                                             n_total), flush=True)

    timing["mean_per_traj_ms"] = (float(np.mean(timing["per_traj_ms"]))
                                  if timing["per_traj_ms"] else None)
    header["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return RunResult(rows=rows,
                     controls=np.concatenate(controls, axis=0),
                     reference=np.concatenate(reference, axis=0),
                     endpoints=np.concatenate(endpoints, axis=0),
                     header=header, timing=timing)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json)
    os.replace(tmp, path)


def _json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value))


def save_csv(path: str, rows, columns=None) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if not rows:
        return
    columns = columns or [k for k in rows[0] if not isinstance(
        rows[0][k], (dict, list))]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in columns})
    os.replace(tmp, path)


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.10g" % value
    return value


def load_csv(path: str) -> list:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_arrays(path: str, **arrays) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.savez_compressed(path, **arrays)
