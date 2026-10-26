#!/usr/bin/env python
"""【已被取代】原实验 3：椭圆几何先验有效性（实验/docs/实验3_生成式安全约束有效性实验_实施方案.md）。

实验 3 已按 实验/docs/实验3_ALM迭代预算与安全效率权衡_实施方案.md 更换为 ALM 迭代预算研究，
本脚本仅作为历史实现保留，结果目录为 实验/results/exp3_superseded_ellipse_prior/）。

Stages, in the order the guide mandates:

``--stage priors``  freeze ``r0, a0, b0`` from the VALIDATION split (never test)
``--stage check``   prove that the three priors really change the convex regions
                    (§2.2: do this BEFORE any large run)
``--stage stageA``  local corridor comparison on IDENTICAL snapshots
                    (same map, same skeleton, same network state)
``--stage stageB``  full 16-step DDIM planning with each prior
``--stage table``   the single result table + the three-panel figure

Nothing else in the pipeline is touched: the network, skeleton selection, ALM,
DDIM updates and history feedback are common to all three configurations; only
``corridor.ellipse_prior`` differs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import metrics as M         # noqa: E402
from src.diffusion.sampler import _DEFAULT_REGION_KEYS, _region_builder  # noqa: E402
from src.geometry.ellipse_prior import apply_ellipse_prior  # noqa: E402
from src.geometry.safety_corridor import (build_safety_corridor,  # noqa: E402
                                          polygon_area)

OUT = os.path.join(C.OUT_ROOT, "exp3_superseded_ellipse_prior")
RAW = os.path.join(OUT, "raw")
PRIOR_FILE = os.path.join(OUT, "fixed_prior.json")
MODES = ("learned", "fixed_circle", "fixed_ellipse")
#: 16-step standard inference; ALM is first active on reverse forward 4
SNAPSHOT_FORWARD = 4


def _prior_block(prior: dict, mode: str) -> dict:
    if mode == "learned":
        return {}
    return dict(mode=mode, radius_scene=prior.get("radius_scene"),
                semi_major_scene=prior.get("semi_major_scene"),
                semi_minor_scene=prior.get("semi_minor_scene"))


def _corridor_cfg(ctx, prior: dict, mode: str) -> dict:
    cfg = dict(ctx.corridor_cfg)
    block = _prior_block(prior, mode)
    if block:
        cfg["ellipse_prior"] = block
    return cfg


def _builder_cfg(ctx) -> dict:
    bcfg = {k: ctx.alm_cfg[k] for k in _DEFAULT_REGION_KEYS
            if k in ctx.alm_cfg}
    bcfg.update(ctx.corridor_cfg.get("region") or {})
    return bcfg


# ---------------------------------------------------------------------------
# stage: freeze the fixed priors on the validation split
# ---------------------------------------------------------------------------


def _stage_priors(args) -> None:
    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split="val",
                          processed=args.processed, limit=args.limit or None)
    collect = {"a": [], "b": [], "n": 0}

    def hook(payload):
        trace = payload.get("trace") or []
        step = next((s for i, s in enumerate(trace, start=1)
                     if i == SNAPSHOT_FORWARD and s.get("alm_active")), None)
        if step is None:
            return
        a = step["ellipse_a"].detach().cpu().numpy().reshape(-1)
        b = step["ellipse_b"].detach().cpu().numpy().reshape(-1)
        keep = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        collect["a"].append(a[keep])
        collect["b"].append(b[keep])
        collect["n"] += int(payload["batch"]["sample_id"].numel())

    C.run_split(ctx, method="priors_val", guidance="alm",
                alm_overrides=dict(force_feedback_valid_zero=False,
                                   safe_prediction_to_ddim=True,
                                   final_project=False),
                steps=args.steps, seed=0, chunk=args.chunk, chunk_hook=hook)
    a = np.concatenate(collect["a"]) if collect["a"] else np.array([np.nan])
    b = np.concatenate(collect["b"]) if collect["b"] else np.array([np.nan])
    a0, b0 = float(np.median(a)), float(np.median(b))
    a0, b0 = max(a0, b0), min(a0, b0)
    radius = float(np.sqrt(a0 * b0))
    sm = ctx.scene_to_meter
    payload = dict(
        split="val", model_checkpoint=C.CKPT_A,
        checkpoint_hash=C.file_hash(C.CKPT_A),
        config=C.CONFIG_A, snapshot_forward=SNAPSHOT_FORWARD,
        seeds_used=[0], n_samples=int(collect["n"]),
        n_ellipses=int(a.size),
        radius_scene=radius, semi_major_scene=a0, semi_minor_scene=b0,
        scene_to_meter=sm,
        radius_m=radius * sm, semi_major_m=a0 * sm, semi_minor_m=b0 * sm,
        rule="a0=median(a), b0=median(b), a0>=b0, r0=sqrt(a0*b0)",
        git_commit=C.git_commit(), frozen_at=__import__("time").strftime(
            "%Y-%m-%d %H:%M:%S"))
    C.save_json(PRIOR_FILE, payload)
    print("[exp3] frozen priors: r0=%.6f (%.3f m), a0=%.6f, b0=%.6f scene"
          % (radius, radius * sm, a0, b0), flush=True)


def _load_prior() -> dict:
    if not os.path.exists(PRIOR_FILE):
        raise SystemExit("[exp3] run --stage priors first (%s missing)"
                         % PRIOR_FILE)
    with open(PRIOR_FILE, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# stage: geometry sanity check (guide §2.2 / §10.3)
# ---------------------------------------------------------------------------


def _capture_snapshot(ctx, args, limit: int | None = None):
    """Collect (centers, shape4, selected_idx) at the first ALM-active forward."""
    snaps: dict = {}

    def hook(payload):
        trace = payload.get("trace") or []
        step = trace[SNAPSHOT_FORWARD - 1] if len(trace) >= SNAPSHOT_FORWARD else None
        if step is None or not step.get("alm_active"):
            step = next((s for s in trace if s.get("alm_active")), None)
        if step is None:
            return
        centers = step["ellipse_center"].detach().cpu().numpy()
        shape4 = step["ellipse_shape4"].detach().cpu().numpy()
        sel = step["selected_idx"].detach().cpu().numpy()
        for local, index in enumerate(payload["rows_index"]):
            snaps[int(index)] = dict(
                sample_id=int(payload["batch"]["sample_id"][local]),
                centers=centers[local].astype(np.float64),
                shape4=shape4[local].astype(np.float64),
                selected_idx=int(sel[local]))

    res = C.run_split(ctx, method="snapshot", guidance="alm",
                      alm_overrides=dict(force_feedback_valid_zero=False,
                                         safe_prediction_to_ddim=True,
                                         final_project=False),
                      corridor_overrides={}, steps=args.steps, seed=args.seed,
                      chunk=args.chunk, chunk_hook=hook, progress=False)
    return snaps, res


def _rebuild(ctx, snap: dict, prior: dict, mode: str):
    """Rebuild one corridor for one snapshot with one prior (same inputs)."""
    item = ctx.dataset[snap["sample_id_index"]]
    # dataset item is [1,H,W]; the region builder wants [B,1,H,W]
    occ = item["occupancy"][None].float()
    geometry = item["candidate_geometry"].float()
    lengths = item["candidate_geometry_lengths"]
    sel = snap["selected_idx"]
    gamma = geometry[sel]
    glen = int(lengths[sel].item())
    if snap.get("sample_id", -1) < 0:            # pragma: no cover
        raise RuntimeError("snapshot has no sample id")
    shape4 = apply_ellipse_prior(
        snap["shape4"], snap["centers"], gamma.numpy(), glen,
        np.linspace(0.0, 1.0, snap["centers"].shape[0]),
        mode=mode, radius=prior["radius_scene"],
        semi_major=prior["semi_major_scene"],
        semi_minor=prior["semi_minor_scene"])
    builder = _region_builder(occ, _builder_cfg(ctx))
    corridor = build_safety_corridor(builder, snap["centers"], shape4,
                                     np.linspace(0.0, 1.0, shape4.shape[0]),
                                     gamma=gamma, gamma_lengths=max(glen, 1),
                                     config=_corridor_cfg(ctx, prior, mode))
    base = [c for c in corridor.cells if c.source == "network"]
    areas = [polygon_area(c.polygon) for c in base if c.polygon is not None]
    return dict(valid=bool(corridor.valid),
                failure_reason=corridor.failure_reason,
                base_cells=len(base), bridge_cells=int(corridor.bridge_cell_count),
                area_mean_scene=float(np.mean(areas)) if areas else None,
                area_total_scene=float(np.sum(areas)) if areas else None,
                cells=[dict(source=c.source, anchor_s=float(c.anchor_s),
                            polygon=(np.round(c.polygon, 6).tolist()
                                     if c.polygon is not None else []))
                       for c in corridor.cells])


def _stage_check(args) -> None:
    prior = _load_prior()
    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                          processed=args.processed, limit=args.limit or 4)
    snaps, res = _capture_snapshot(ctx, args)
    report = []
    for index, snap in sorted(snaps.items()):
        snap = dict(snap)
        snap["sample_id_index"] = index
        row = dict(index=index, sample_id=snap["sample_id"],
                   selected_idx=snap["selected_idx"])
        for mode in MODES:
            info = _rebuild(ctx, snap, prior, mode)
            row[mode] = {k: info[k] for k in ("valid", "base_cells",
                                              "bridge_cells", "area_mean_scene")}
        report.append(row)
    C.save_json(os.path.join(OUT, "check_geometry.json"), report)
    for row in report:
        print("[exp3] sample %s: %s" % (row["sample_id"], " | ".join(
            "%s valid=%s base=%d bridge=%d area=%.5f" %
            (m, row[m]["valid"], row[m]["base_cells"], row[m]["bridge_cells"],
             row[m]["area_mean_scene"] or -1) for m in MODES)), flush=True)
    changed = any(len({json.dumps(row[m]) for m in MODES}) > 1 for row in report)
    print("[exp3] priors %s the convex geometry"
          % ("CHANGE" if changed else "do NOT change"), flush=True)


# ---------------------------------------------------------------------------
# stage A: corridor quality on identical snapshots
# ---------------------------------------------------------------------------


def _stage_a(args) -> None:
    prior = _load_prior()
    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                          processed=args.processed, limit=args.limit or None)
    cache = os.path.join(OUT, "snapshots.npz")
    if os.path.exists(cache) and not args.force:
        data = np.load(cache, allow_pickle=True)
        snaps = {int(k): v for k, v in zip(data["index"], data["payload"])}
        print("[exp3] loaded %d cached snapshots" % len(snaps), flush=True)
    else:
        snaps, _ = _capture_snapshot(ctx, args)
        C.save_arrays(cache, index=np.array(sorted(snaps)),
                      payload=np.array([dict(snaps[i], sample_id_index=i)
                                        for i in sorted(snaps)], dtype=object))
        print("[exp3] captured %d snapshots" % len(snaps), flush=True)
    rows = []
    for index, snap in sorted(snaps.items()):
        snap = dict(snap)
        snap.setdefault("sample_id_index", index)
        base = dict(index=int(index), sample_id=int(snap["sample_id"]),
                    selected_idx=int(snap["selected_idx"]))
        for mode in MODES:
            info = _rebuild(ctx, snap, prior, mode)
            row = dict(base, prior_mode=mode, corridor_success=info["valid"],
                       failure_reason=info["failure_reason"],
                       base_region_count=info["base_cells"],
                       bridge_region_count=info["bridge_cells"],
                       mean_base_area_scene=info["area_mean_scene"],
                       mean_base_area_m2=(None if info["area_mean_scene"] is None
                                          else info["area_mean_scene"]
                                          * ctx.scene_to_meter ** 2))
            rows.append(row)
    C.save_csv(os.path.join(OUT, "stage_a_per_snapshot.csv"), rows)
    table = _stage_a_table(rows)
    C.save_csv(os.path.join(OUT, "stage_a_table.csv"), table)
    C.save_json(os.path.join(OUT, "stage_a_summary.json"),
                dict(table=table, n_snapshots=len(snaps),
                     prior=prior, builder_cfg=_builder_cfg(ctx)))
    print("[exp3] stage A done on %d snapshots" % len(snaps), flush=True)


def _stage_a_table(rows):
    table = []
    for mode in MODES:
        part = [r for r in rows if r["prior_mode"] == mode]
        ok = [r for r in part if r["corridor_success"]]
        areas = [r["mean_base_area_m2"] for r in ok
                 if r["mean_base_area_m2"] is not None]
        table.append(dict(
            prior_mode=mode, n=len(part), success=len(ok),
            success_rate_pct=(100.0 * len(ok) / len(part)) if part else None,
            mean_base_area_m2=float(np.mean(areas)) if areas else None,
            mean_bridge_regions=(float(np.mean([r["bridge_region_count"]
                                                for r in ok])) if ok else None),
            failure_reasons=dict(_count(r["failure_reason"] for r in part
                                        if not r["corridor_success"]))))
    return table


def _count(items):
    out = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


# ---------------------------------------------------------------------------
# stage B: full planning with each prior
# ---------------------------------------------------------------------------


def _stage_b(args) -> None:
    prior = _load_prior()
    seeds = [int(s) for s in args.seeds]
    for mode in args.modes:
        for seed in seeds:
            stem = os.path.join(RAW, "stageB_%s_seed%d" % (mode, seed))
            if os.path.exists(stem + ".npz") and not args.force:
                print("[exp3] skip %s_seed%d" % (mode, seed), flush=True)
                continue
            ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                                  processed=args.processed,
                                  limit=args.limit or None)
            res = C.run_split(
                ctx, method="stageB_%s" % mode, guidance="alm",
                alm_overrides=dict(force_feedback_valid_zero=False,
                                   safe_prediction_to_ddim=True,
                                   final_project=False),
                corridor_overrides=_corridor_cfg(ctx, prior, mode),
                steps=args.steps, seed=seed, chunk=args.chunk,
                capture_corridors=True)
            C.save_csv(stem + ".csv", res.rows)
            C.save_arrays(stem + ".npz", control=res.controls,
                          reference=res.reference, endpoints=res.endpoints)
            C.save_json(stem + ".json", dict(header=res.header,
                                             timing=res.timing,
                                             prior=prior, mode=mode))
            del ctx


def _stage_table(args) -> None:
    prior = _load_prior()
    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                          processed=args.processed, limit=1)
    codec, sm = ctx.codec, ctx.scene_to_meter
    stage_a = {}
    path_a = os.path.join(OUT, "stage_a_table.csv")
    if os.path.exists(path_a):
        stage_a = {r["prior_mode"]: r for r in C.load_csv(path_a)}
    table, per_task = [], []
    for mode in MODES:
        rows, controls, refs, ends = [], [], [], []
        for path in sorted(glob.glob(os.path.join(RAW,
                                                  "stageB_%s_seed*.npz" % mode))):
            stem = path[:-4]
            part = C.load_csv(stem + ".csv")
            for row in part:
                row["collision"] = str(row["collision"]).lower() == "true"
                row["sdf_min_distance_m"] = float(row["sdf_min_distance_m"])
                row["rmse_m"] = float(row["rmse_m"])
                row["seed"] = int(row["seed"])
                row["sample_id"] = int(row["sample_id"])
                row["prior_mode"] = mode
            data = np.load(path)
            M.enrich(part, data["control"], data["reference"],
                     data["endpoints"], codec, sm,
                     step_m=args.geometry_step)
            rows.extend(part)
            controls.append(data["control"])
            refs.append(data["reference"])
            ends.append(data["endpoints"])
        if not rows:
            print("[exp3] WARNING no stage B run for %s" % mode, flush=True)
            continue
        per_task.extend(rows)
        n = len(rows)
        coll = sum(1 for r in rows if r["collision"])
        # detour: preferred reporting口径 = collision-free, valid trajectories
        safe = [r for r in rows if not r["collision"] and r["detour_ratio"]]
        a = stage_a.get(mode, {})
        table.append(dict(
            prior_mode=mode,
            corridor_success_pct=(float(a["success_rate_pct"])
                                  if a.get("success_rate_pct") else None),
            mean_base_area_m2=(float(a["mean_base_area_m2"])
                               if a.get("mean_base_area_m2") else None),
            mean_bridge_regions=(float(a["mean_bridge_regions"])
                                 if a.get("mean_bridge_regions") else None),
            generated=n, generated_of=1260,
            collision_pct=100.0 * coll / n,
            collisions=coll,
            valid_outputs=sum(1 for r in rows
                              if str(r.get("corridor_guided")).lower() == "true"),
            detour_ratio=float(np.mean([r["detour_ratio"] for r in safe]))
            if safe else None,
            detour_n=len(safe),
            nrmse_pct=float(np.mean([r["nrmse_pct"] for r in rows
                                     if r["nrmse_pct"] is not None]))))
    columns = ["prior_mode", "corridor_success_pct", "mean_base_area_m2",
               "mean_bridge_regions", "generated", "generated_of",
               "collision_pct", "collisions", "valid_outputs", "detour_ratio",
               "detour_n", "nrmse_pct"]
    C.save_csv(os.path.join(OUT, "table3_geometry.csv"), table, columns)
    C.save_csv(os.path.join(OUT, "per_task_stage_b.csv"), per_task,
               ["prior_mode", "seed", "sample_id", "collision", "detour_ratio",
                "nrmse_pct", "sdf_min_distance_m", "corridor_guided",
                "corridor_valid", "corridor_failure", "base_cell_count",
                "bridge_cell_count", "activation_step", "alm_status"])
    C.save_json(os.path.join(OUT, "table3_summary.json"),
                dict(table=table, prior=prior))
    _markdown(os.path.join(OUT, "table3_geometry.md"), table, columns)
    _figure(prior, args)
    print("[exp3] table + figure written", flush=True)


def _markdown(path, rows, columns) -> None:
    def fmt(v):
        if v is None:
            return "N/A"
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


# ---------------------------------------------------------------------------
# figure: three panels on ONE map / task / skeleton
# ---------------------------------------------------------------------------


def _ellipse_outline(shape4_row, center, n=48):
    """Sample the prior ellipse ``[log a, log b, cos 2t, sin 2t]`` as a polygon."""
    log_a, log_b, c2, s2 = shape4_row
    a = float(np.exp(np.clip(log_a, -20.0, 5.0)))
    b = float(np.exp(np.clip(log_b, -20.0, 5.0)))
    theta = 0.5 * float(np.arctan2(s2, c2))
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return (np.asarray(center, dtype=np.float64)
            + a * np.cos(t)[:, None] * u + b * np.sin(t)[:, None] * v)


def _figure(prior: dict, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                          processed=args.processed, limit=None)
    cache = os.path.join(OUT, "snapshots.npz")
    if os.path.exists(cache) and not args.force:
        data = np.load(cache, allow_pickle=True)
        snaps = {int(k): dict(v, sample_id_index=int(k))
                 for k, v in zip(data["index"], data["payload"])}
    else:
        snaps, _ = _capture_snapshot(ctx, args)
        snaps = {i: dict(s, sample_id_index=i) for i, s in snaps.items()}
    if not snaps:
        print("[exp3] no snapshot for the figure", flush=True)
        return
    # representative sample: the largest relative spread of the base-region
    # area among the three priors, all corridors valid (guide §8.2)
    chosen, best = None, -1.0
    for index, snap in sorted(snaps.items()):
        infos = {mode: _rebuild(ctx, snap, prior, mode) for mode in MODES}
        if not all(infos[m]["valid"] for m in MODES):
            continue
        areas = [infos[m]["area_mean_scene"] or 0.0 for m in MODES]
        rel = (max(areas) - min(areas)) / max(float(np.mean(areas)), 1e-9)
        if rel > best:
            best, chosen = rel, (index, snap, infos)
    if chosen is None:
        print("[exp3] no sample with three valid corridors", flush=True)
        return
    index, snap, infos = chosen
    item = ctx.dataset[index]
    occ = item["occupancy"][0].numpy()
    geometry = item["candidate_geometry"].float()
    sel = snap["selected_idx"]
    glen = int(item["candidate_geometry_lengths"][sel].item())
    skeleton = geometry[sel][:glen].numpy()
    anchors = np.linspace(0.0, 1.0, snap["centers"].shape[0])

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 7, "axes.linewidth": 0.6,
                         "figure.dpi": 300})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    titles = {"fixed_circle": "(a) Fixed Circle",
              "fixed_ellipse": "(b) Fixed Ellipse",
              "learned": "(c) Learned Ellipse"}
    colors = {"fixed_circle": "#2E7D32", "fixed_ellipse": "#EF6C00",
              "learned": "#1f4e79"}
    for ax, mode in zip(axes, ("fixed_circle", "fixed_ellipse", "learned")):
        ax.imshow(occ, cmap="Greys", origin="lower", extent=[-1, 1, -1, 1],
                  alpha=0.45, interpolation="nearest")
        info = infos[mode]
        for cellinfo in info["cells"]:
            poly = cellinfo["polygon"]
            if not poly:
                continue
            is_bridge = cellinfo["source"] == "bridge"
            ax.add_patch(MplPolygon(
                poly, closed=True, fill=not is_bridge,
                facecolor=colors[mode], edgecolor=colors[mode],
                alpha=0.10 if not is_bridge else 0.0, linewidth=0.30,
                linestyle="--" if is_bridge else "-", zorder=2))
        shape4 = apply_ellipse_prior(
            snap["shape4"], snap["centers"], skeleton, glen, anchors,
            mode=mode, radius=prior["radius_scene"],
            semi_major=prior["semi_major_scene"],
            semi_minor=prior["semi_minor_scene"])
        for i in range(0, len(anchors), 16):
            outline = _ellipse_outline(shape4[i], snap["centers"][i])
            ax.plot(outline[:, 0], outline[:, 1], color=colors[mode],
                    linewidth=0.55, alpha=0.95, zorder=3)
        ax.plot(skeleton[:, 0], skeleton[:, 1], color="0.15", linewidth=1.0,
                zorder=4)
        ax.set_title("%s  (%d base, %d bridge, %.0f m$^2$)" % (
            titles[mode], info["base_cells"], info["bridge_cells"],
            (info["area_mean_scene"] or 0.0) * ctx.scene_to_meter ** 2),
            fontsize=7)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal")
        ax.set_xlabel("x (scene)")
    axes[0].set_ylabel("y (scene)")
    fig.tight_layout(pad=0.3)
    stem = os.path.join(OUT, "fig3_geometry_priors")
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (stem, ext), bbox_inches="tight")
    plt.close(fig)
    C.save_json(os.path.join(OUT, "figure_source.json"),
                dict(index=int(index), sample_id=int(snap["sample_id"]),
                     selected_idx=int(sel), prior=prior,
                     area_rel_spread=float(best),
                     note=("prior ellipses drawn every 16th anchor; polygons are "
                           "the deterministic convex regions"),
                     summary={m: {k: infos[m][k] for k in
                                  ("valid", "base_cells", "bridge_cells",
                                   "area_mean_scene")} for m in MODES}))
    print("[exp3] figure: sample %s (base-area spread %.2f%%)"
          % (snap["sample_id"], 100.0 * best), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["priors", "check", "stageA", "stageB", "table",
                             "figure"])
    ap.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    ap.add_argument("--seeds", nargs="*", default=["0", "1", "2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--geometry-step", type=float, default=M.GEOMETRY_STEP_M,
                    help="arc-length spacing (m) used to evaluate curvature")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"priors": _stage_priors, "check": _stage_check, "stageA": _stage_a,
     "stageB": _stage_b, "table": _stage_table,
     "figure": lambda a: _figure(_load_prior(), a)}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
