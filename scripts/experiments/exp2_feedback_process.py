#!/usr/bin/env python
"""实验 2 第二部分：历史安全反馈的作用（同权重、只切换反馈输入）.

Both configurations load the SAME generator A checkpoint and keep the SAME ALM,
convex corridor and safe DDIM update.  The only switch is
``alm.force_feedback_valid_zero`` (v_hist = 0 vs the real history), which zeroes
the gated injection while leaving the FeedbackEncoder weights untouched.

Per reverse forward we log (实验 2 §4.5):

    sample_id, seed, feedback_enabled, forward_index, diffusion_t,
    pre_violation_m, correction_m, alm_valid, output_valid

``pre_violation_m``  = max_i [d_req - d(tau_raw(u_i))]_+ on the SAME SDF criterion
                       used by 实验 1 (d_req = 0, SDF already on inflated
                       obstacles), NOT the corridor residual of the ALM pack.
``correction_m``     = RMS_i || tau_ALM(u_i) - tau_raw(u_i) ||, same u_i grid,
                       N_p = 512.

Note: this is a read-only log around the existing inference; the sampler is not
modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import metrics as M         # noqa: E402
from src.diffusion.sampler import _dense_decode  # noqa: E402

OUT = os.path.join(C.OUT_ROOT, "exp2")
RAW = os.path.join(OUT, "raw_feedback")
#: guide §4.1: warm-up forwards 1-3 carry no ALM; the curves start at forward 4
FIRST_FORWARD = 4
#: guide §4.6: the paired statistics use forwards 5-16 (4's correction cannot
#: have influenced the network yet)
PAIR_FROM = 5

CONFIGS = {
    "without_feedback": dict(label="Without Feedback", feedback=False),
    "with_feedback": dict(label="With Feedback", feedback=True),
}


def _hook_factory(store, ctx, feedback_on):
    codec, sm = ctx.codec, ctx.scene_to_meter
    dev = ctx.device

    def hook(payload):
        trace = payload.get("trace") or []
        sdf, rows_index = payload["sdf"], payload["rows_index"]
        for fwd, step in enumerate(trace, start=1):
            q_raw = step.get("q0_raw")
            q_safe = step.get("q0_safe")
            if q_raw is None or q_safe is None:
                continue
            p_raw = _dense_decode(codec, q_raw.to(dev), C.DENSE_POINTS)
            p_safe = _dense_decode(codec, q_safe.to(dev), C.DENSE_POINTS)
            stats = sdf.stats(p_raw, d_safe=0.0, with_boundary=False)
            viol = sdf.to_metres(stats["violation"]).detach().cpu().numpy()
            min_d = sdf.to_metres(
                stats["min_distance"]).detach().cpu().numpy()
            corr = ((p_safe - p_raw).norm(dim=-1).pow(2).mean(dim=1).sqrt()
                    * sm).detach().cpu().numpy()
            guided = step.get("guided")
            guided = (guided.detach().cpu().numpy() if guided is not None
                      else np.zeros(len(rows_index), dtype=bool))
            alm = step.get("alm_stats")
            feasible_after = np.full(len(rows_index), np.nan)
            if alm is not None and "max_violation_after" in alm:
                after = alm["max_violation_after"].detach().cpu().numpy()
                if after.shape[0] == len(rows_index):
                    feasible_after = after.astype(np.float64)
            for local, index in enumerate(rows_index):
                store.append({
                    "sample_id": int(payload["batch"]["sample_id"][local]),
                    "index": int(index),
                    "forward_index": int(fwd),
                    "diffusion_t": int(step.get("t", -1)),
                    "feedback_enabled": bool(feedback_on),
                    "alm_active": bool(step.get("alm_active", False)),
                    "guided": bool(guided[local]) if local < len(guided) else False,
                    "pre_violation_m": float(viol[local]),
                    "min_signed_distance_m": float(min_d[local]),
                    "correction_m": float(corr[local]),
                    "alm_valid": (bool(feasible_after[local] <= 1e-3)
                                  if np.isfinite(feasible_after[local]) else False),
                    "output_valid": True,
                })
    return hook


def _run(args) -> None:
    seeds = [int(s) for s in args.seeds]
    for tag in args.configs:
        feedback_on = tag == "with_feedback"
        alm = dict(force_feedback_valid_zero=not feedback_on,
                   safe_prediction_to_ddim=True, final_project=False)
        for seed in seeds:
            stem = os.path.join(RAW, "%s_seed%d" % (tag, seed))
            if os.path.exists(stem + ".csv") and not args.force:
                print("[exp2b] skip %s_seed%d" % (tag, seed), flush=True)
                continue
            store: list = []
            ctx = C.build_context(C.CONFIG_A, C.CKPT_A, split=args.split,
                                  processed=args.processed,
                                  limit=args.limit or None)
            res = C.run_split(ctx, method=tag, guidance="alm",
                              alm_overrides=alm, steps=args.steps, seed=seed,
                              chunk=args.chunk,
                              chunk_hook=_hook_factory(store, ctx, feedback_on))
            C.save_csv(stem + "_process.csv", store)
            C.save_csv(stem + ".csv", res.rows)
            C.save_arrays(stem + ".npz", control=res.controls,
                          reference=res.reference, endpoints=res.endpoints)
            C.save_json(stem + ".json", dict(header=res.header,
                                             timing=res.timing,
                                             forward_first=FIRST_FORWARD,
                                             pair_from=PAIR_FROM))
            del ctx


def _load_process(tag):
    """All per-forward records of one configuration (seed taken from the stem)."""
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(RAW, "%s_seed*_process.csv" % tag))):
        seed = int(os.path.basename(path).split("_seed")[1].split("_")[0])
        for row in C.load_csv(path):
            row["seed"] = seed
            row["forward_index"] = int(row["forward_index"])
            for key in ("pre_violation_m", "correction_m",
                        "min_signed_distance_m"):
                row[key] = float(row[key])
            out.append(row)
    return out


def _curve(rows, keys):
    by_step = {}
    for row in rows:
        by_step.setdefault(row["forward_index"], []).append(row)
    steps = sorted(by_step)
    out = {"forward_index": steps}
    for key in keys:
        out[key] = [float(np.mean([r[key] for r in by_step[s]])) if by_step[s]
                    else None for s in steps]
    out["n"] = [len(by_step[s]) for s in steps]
    return out


def _paired(rows_with, rows_without, key):
    def per_task(rows):
        acc = {}
        for row in rows:
            if row["forward_index"] < PAIR_FROM:
                continue
            acc.setdefault((row["sample_id"], row["seed"]), []).append(row[key])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    a = per_task(rows_with)
    b = per_task(rows_without)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    diff = np.asarray([a[k] - b[k] for k in common], dtype=np.float64)
    out = {"key": key, "n_pairs": len(common),
           "mean_with": float(np.mean([a[k] for k in common])),
           "mean_without": float(np.mean([b[k] for k in common])),
           "mean_diff": float(diff.mean()),
           "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else None,
           "forward_from": PAIR_FROM}
    try:
        from scipy import stats
        if len(diff) > 2:
            out["wilcoxon_p"] = float(stats.wilcoxon(diff).pvalue)
            out["paired_t_p"] = float(stats.ttest_rel(
                [a[k] for k in common], [b[k] for k in common]).pvalue)
    except Exception as exc:                                   # pragma: no cover
        out["stats_error"] = str(exc)
    return out


def _summarise(args) -> None:
    keys = ("pre_violation_m", "correction_m")
    payload = {}
    data = {}
    for tag in ("without_feedback", "with_feedback"):
        rows = _load_process(tag)
        if not rows:
            print("[exp2b] WARNING no process log for %s" % tag, flush=True)
            continue
        curve = _curve(rows, keys)
        data[tag] = curve
        payload[tag] = dict(
            label=CONFIGS[tag]["label"], n_records=len(rows),
            forwards=sorted({r["forward_index"] for r in rows}),
            curve=curve)
    if len(data) == 2:
        payload["paired"] = {
            key: _paired([r for r in _load_process("with_feedback")],
                         [r for r in _load_process("without_feedback")], key)
            for key in keys}
    C.save_json(os.path.join(OUT, "feedback_process_summary.json"), payload)
    _write_curve_csv(data)
    _plot(data, os.path.join(OUT, "fig_feedback_process"))
    print("[exp2b] process curves written", flush=True)


def _write_curve_csv(data) -> None:
    rows = []
    tags = [t for t in ("without_feedback", "with_feedback") if t in data]
    if not tags:
        return
    steps = data[tags[0]]["forward_index"]
    for i, step in enumerate(steps):
        row = {"forward_index": step,
               "diffusion_t": data[tags[0]].get("diffusion_t", [None] * len(steps))[i]}
        for tag in tags:
            curve = data[tag]
            row["%s_pre_violation_m" % tag] = curve["pre_violation_m"][i]
            row["%s_correction_m" % tag] = curve["correction_m"][i]
            row["%s_n" % tag] = curve["n"][i]
        rows.append(row)
    C.save_csv(os.path.join(OUT, "feedback_process_curves.csv"), rows)


def _plot(data, stem: str) -> None:
    if len(data) < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "legend.frameon": False, "figure.dpi": 300,
    })
    styles = {
        "without_feedback": dict(label="Without Feedback", color="#B22222",
                                 marker="s", linestyle="--"),
        "with_feedback": dict(label="With Feedback", color="#1f4e79",
                              marker="o", linestyle="-"),
    }
    specs = (("pre_violation_m", "(a) Pre-correction safety violation",
              "Mean violation before ALM  (m)"),
             ("correction_m", "(b) ALM correction magnitude",
              "Mean correction  (m)"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (key, title, ylabel) in zip(axes, specs):
        for tag in ("without_feedback", "with_feedback"):
            if tag not in data:
                continue
            curve = data[tag]
            keep = [(x, y) for x, y in zip(curve["forward_index"], curve[key])
                    if x >= FIRST_FORWARD and y is not None]
            if not keep:
                continue
            xs, ys = zip(*keep)
            ax.plot(xs, ys, **styles[tag])
        ax.set_xlabel("Reverse step (forward index)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.set_xlim(FIRST_FORWARD - 0.5, 16.5)
        ax.set_xticks(range(FIRST_FORWARD, 17, 2))
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (stem, ext), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "summarise"], required=True)
    ap.add_argument("--configs", nargs="*",
                    default=["without_feedback", "with_feedback"],
                    choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="*", default=["0", "1", "2"])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    _run(args) if args.stage == "run" else _summarise(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
