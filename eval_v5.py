"""Evaluation of the new single-path framework (v5) with the 4-step sampler.

    python eval_v5.py --config configs/config_v5_c32q64.yaml \
        --ckpt outputs/models/v5_c32q64/ckpt/best_task.pt --split test --limit 420

Reports (per sample -> CSV, plus an aggregate JSON):
  * search: status / 回退 / 扩展节点 / 搜索耗时 / 引导场耗时 / 路径长度
  * every denoise step: raw-x0 collision fraction, whether ALM ran and passed,
    whether active_path was updated, whether the feedback was valid
  * final: collision rate on the raw map AND on the r_robot-inflated safe grid,
    curve RMSE vs GT, endpoint error, start/goal heading error, length
  * timing: grid build, per-step network, ALM, corridor, total
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import make_collate, make_loader
from src.diffusion.alm_local import collision_mask
from src.diffusion.sampler_v5 import sample_v5
from src.diffusion.schedule import NoiseSchedule
from src.models.factory import build_model
from src.utils.config import load_config, num_controls, num_safety_queries


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def heading_cos(curve, heading, which=0):
    """cos between the curve's endpoint tangent and the target heading."""
    if which == 0:
        d = curve[1] - curve[0]
    else:
        d = curve[-1] - curve[-2]
    n = float(np.linalg.norm(d))
    t = heading[which]
    tn = float(np.linalg.norm(t))
    if n < 1e-9 or tn < 1e-9:
        return float("nan")
    return float(np.dot(d, t) / (n * tn))


def figure(batch, out, i, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occ = _np(batch["occupancy"][i, 0])
    cond = _np(batch["cond"][i])
    gt = _np(batch["pos"][i])
    trace = out["trace"]
    n = len(trace)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    axes = axes.ravel()

    def base(ax, title):
        ax.imshow(occ, origin="lower", cmap="gray_r", alpha=0.5,
                  interpolation="nearest")
        ax.plot(*((gt + 1) / 2 * 256 - 0.5).T, color="#2ca02c", lw=1.1,
                label="GT")
        ax.scatter(*(((cond + 1) / 2 * 256 - 0.5).T), marker="*", s=150,
                   c=["#17a34a", "#e31a1c"], zorder=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)

    def curve_px(c):
        return (np.asarray(c) + 1) / 2 * 256 - 0.5

    base(axes[0], "occupancy + GT")
    if trace[0].get("search"):
        base(axes[1], "step1 x0 (red=collision) + A* path")
        c0 = curve_px(trace[0]["raw_curve"][i])
        axes[1].plot(*c0.T, color="#ff7f0e", lw=1.0)
        st = trace[0]["search"][i]
        hit, _ = collision_mask(torch.from_numpy(occ)[None], trace[0]["raw_curve"][i])
        axes[1].scatter(*c0[hit].T, s=4, c="#d62728")
        if st.get("path") is not None:
            pp = curve_px(np.asarray(st["path"]))
            axes[1].plot(*pp.T, color="#1f77b4", lw=2.2, label="searched path")
        axes[1].set_title("step1 x0 + search: %s (exp %s)  path clearance "
                          "%.1f->%.1f m" % (st.get("status"), st.get("expanded"),
                                            st.get("clearance_before") or -1,
                                            st.get("clearance_after") or -1),
                          fontsize=8)
    for k in range(1, n):
        ax = axes[1 + k]
        base(ax, "step%d raw x0  t=%d" % (trace[k]["step"], trace[k]["t"]))
        c = curve_px(trace[k]["raw_curve"][i])
        axes[1 + k].plot(*c.T, color="#ff7f0e", lw=1.1)
        hit, _ = collision_mask(torch.from_numpy(occ)[None], trace[k]["raw_curve"][i])
        axes[1 + k].scatter(*c[hit].T, s=4, c="#d62728")
        alm = trace[k].get("alm") or {}
        # safety ellipses
        ell = trace[k].get("ellipse") or {}
        if "center" in ell and "a" in ell:
            cen = ell["center"][i]
            aa, bb, th = ell["a"][i], ell["b"][i], ell["theta"][i]
            cc = curve_px(cen)
            axes[1 + k].scatter(*cc.T, s=2, c="#17becf", alpha=0.7)
            tt = np.linspace(0, 2 * np.pi, 40)
            for q in range(0, len(cen), 4):
                ex = aa[q] * np.cos(tt)
                ey = bb[q] * np.sin(tt)
                ct, stt = np.cos(th[q]), np.sin(th[q])
                axes[1 + k].plot((cen[q, 0] + ct * ex - stt * ey + 1) / 2 * 256 - 0.5,
                                 (cen[q, 1] + stt * ex + ct * ey + 1) / 2 * 256 - 0.5,
                                 color="#17becf", lw=0.6, alpha=0.6)
        # convex corridor cells
        cells_all = alm.get("cells") or []
        cells_i = cells_all[i] if i < len(cells_all) else []
        for poly in cells_i:
            pp = curve_px(np.asarray(poly))
            axes[1 + k].fill(*pp.T, color="#4c72b0", alpha=0.06, lw=0.0)
            axes[1 + k].plot(*pp.T, color="#4c72b0", lw=0.35, alpha=0.3)
        # ALM-corrected curve
        sc = trace[k].get("safe_curve")
        if sc is not None:
            sc = np.asarray(sc)
            sc = sc[i] if sc.ndim == 3 else sc
            pcs = curve_px(sc)
            axes[1 + k].plot(*pcs.T, color="#9467bd", lw=2.0, ls="--",
                             label="ALM corrected")
        axes[1 + k].set_title(
            "step%d t=%d  coll %.3f -> alm=%s ok=%s  cells %d"
            % (trace[k]["step"], trace[k]["t"], float(hit.mean()),
               alm.get("alm"), alm.get("alm_ok"),
               len(cells_i)), fontsize=8)
    ax = axes[-1]
    base(ax, "final curve")
    c = curve_px(_np(out["final"][i]))
    ax.plot(*c.T, color="#1f77b4", lw=1.6)
    hit, _ = collision_mask(torch.from_numpy(occ)[None], out["final"][i])
    ax.scatter(*c[hit].T, s=5, c="#d62728")
    ax.set_title("final (coll %.3f)" % float(hit.mean()), fontsize=8)
    for ax in axes[n + 1:]:
        ax.axis("off")
    fig.suptitle("v5 4-step sample %d   search=%s   steps=%d"
                 % (int(batch["sample_id"][i]), [s.get("status") for s in trace[0].get("search", [])][0]
                    if trace[0].get("search") else "-", n), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v5_c32q64.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=420)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--figures", type=int, default=4)
    ap.add_argument("--no-heading", action="store_true")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    ds_loader, ds = make_loader(
        args.split, cfg["data"]["processed_root"], batch_size=args.batch,
        shuffle=False, num_workers=0, limit=args.limit, require_labels=False,
        num_controls=num_controls(cfg), num_safety_queries=num_safety_queries(cfg))
    sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                          beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                             "squaredcos_cap_v2")).to(device)
    model = build_model(cfg).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print("[eval] ckpt=%s epoch=%s missing=%d unexpected=%d"
          % (args.ckpt, ck.get("epoch"), len(missing), len(unexpected)),
          flush=True)

    rows, n_fig = [], 0
    curves = []
    for bi, batch in enumerate(ds_loader):
        out = sample_v5(model, sched, batch, cfg, device=device,
                        steps=args.steps, seed=args.seed + bi,
                        use_heading=not args.no_heading,
                        use_feedback=not args.no_feedback)
        final = out["final"]
        for i in range(final.shape[0]):
            occ_i = batch["occupancy"][i:i + 1]
            c = final[i].detach().cpu().numpy()
            hit_raw, oob = collision_mask(occ_i, c)
            sid = int(batch["sample_id"][i])
            # ---- metrics -------------------------------------------------
            gt = _np(batch["pos"][i])
            cond = _np(batch["cond"][i])
            heading = _np(batch["heading"][i]) if "heading" in batch else None
            hvalid = _np(batch["heading_valid"][i]) if "heading_valid" in batch else None
            rmse = float(np.sqrt(((c - gt) ** 2).sum(-1).mean()))
            step_trace = []
            for st in out["trace"]:
                alm = st.get("alm") or {}
                step_trace.append({
                    "step": st["step"], "t": st["t"], "net_ms": st["net_ms"],
                    "has_path": st["has_path"], "has_safety": st["has_safety"],
                    "coll_frac": float(st["collision"][i]["frac"]),
                    "n_intervals": len(st["collision"][i]["intervals"]),
                    "alm_ran": bool(alm.get("alm", False)),
                    "alm_ok": bool(alm.get("alm_ok", False)),
                    "alm_reason": alm.get("reason", ""),
                    "feedback_valid": st["feedback_valid"],
                })
            search = (out["trace"][0].get("search") or [{}] * final.shape[0])[i]
            rows.append({
                "sample": sid,
                "search_status": search.get("status"),
                "search_fallback": search.get("fallback"),
                "guide_valid": search.get("guide_valid"),
                "expanded": search.get("expanded"),
                "search_ms": round(float(search.get("search_ms", 0.0)), 2),
                "path_len_m": search.get("path_len_m"),
                "grid_ms": round(float(out["grid_ms"]) / max(len(batch["cond"]), 1), 2),
                # sampler timings are per BATCH -> divide to get per-sample numbers
        "total_ms": round(float(out["total_ms"]) / max(len(batch["cond"]), 1), 2),
                "steps": out["steps"],
                "final_coll_frac": float(hit_raw.mean()),
        "final_source": ((out.get("final_source") or ["model"])[i]
                         if isinstance(out.get("final_source"), list)
                         else out.get("final_source", "model")),
                "final_oob": float(oob.mean()),
                "curve_rmse": rmse,
                "endpoint_err": float(np.linalg.norm(c[0] - cond[0])
                                      + np.linalg.norm(c[-1] - cond[1])),
                "heading_cos_start": heading_cos(c, heading, 0) if heading is not None else None,
                "heading_cos_goal": heading_cos(c, heading, 1) if heading is not None else None,
                "heading_valid_start": bool(hvalid[0]) if hvalid is not None else None,
                "heading_valid_goal": bool(hvalid[1]) if hvalid is not None else None,
                "length_m": float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() * 80.0),
                "alm_ran_steps": sum(1 for s in step_trace if s["alm_ran"]),
                "alm_ok_steps": sum(1 for s in step_trace if s["alm_ok"]),
                "path_updates": sum(1 for s in step_trace if s["alm_ok"]),
                "step_trace": json.dumps(step_trace),
            })
            curves.append(c.astype(np.float32))
            if n_fig < args.figures:
                figure(batch, out, i,
                       os.path.join(args.out, "figures", "v5_%s_%03d.png"
                                    % (args.split, sid)), cfg)
                n_fig += 1
        if (bi + 1) % 5 == 0:
            print("[eval] %d/%d batches" % (bi + 1, len(ds_loader)), flush=True)

    csv_path = os.path.join(args.out, "eval_%s.csv" % args.split)
    keys = [k for k in rows[0] if k != "step_trace"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["step_trace"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    np.savez_compressed(os.path.join(args.out, "curves_%s.npz" % args.split),
                        final=np.stack(curves),
                        samples=np.asarray([r["sample"] for r in rows]))

    def stat(key, fn=np.mean):
        v = [r[key] for r in rows if r[key] is not None
             and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(fn(v)) if v else None

    from collections import Counter
    src_cnt = Counter(r.get("final_source", "model") for r in rows)
    summary = {
        "split": args.split, "n": len(rows), "ckpt": args.ckpt,
        "final_source_counts": dict(src_cnt),
        "use_heading": not args.no_heading,
        "use_feedback": not args.no_feedback,
        "steps": rows[0]["steps"] if rows else None,
        "search_success_rate": float(np.mean([r["search_status"] == "SUCCESS"
                                              for r in rows])),
        "search_fallback_rate": float(np.mean([r["search_fallback"] not in (None, "")
                                               for r in rows])),
        "expanded_median": stat("expanded", np.median),
        "search_ms_median": stat("search_ms", np.median),
        "grid_ms_median": stat("grid_ms", np.median),
        "total_ms_median": stat("total_ms", np.median),
        "final_collision_rate_mean": stat("final_coll_frac"),
        "final_collision_any_rate": float(np.mean([r["final_coll_frac"] > 0
                                                    for r in rows])),
        "curve_rmse_mean": stat("curve_rmse"),
        "endpoint_err_mean": stat("endpoint_err"),
        "heading_cos_start_mean": stat("heading_cos_start"),
        "heading_cos_goal_mean": stat("heading_cos_goal"),
        "alm_ran_rate": float(np.mean([r["alm_ran_steps"] > 0 for r in rows])),
        "alm_ok_rate": float(np.mean([r["alm_ok_steps"] > 0 for r in rows
                                      if r["alm_ran_steps"] > 0])) if any(
            r["alm_ran_steps"] > 0 for r in rows) else None,
        "path_update_rate": float(np.mean([r["path_updates"] > 0 for r in rows])),
        "length_m_median": stat("length_m", np.median),
    }
    with open(os.path.join(args.out, "summary_%s.json" % args.split), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("[eval] ->", args.out, flush=True)


if __name__ == "__main__":
    main()
