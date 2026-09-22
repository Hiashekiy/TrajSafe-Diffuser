"""compare_campaign.py - side-by-side report of the A/B training campaign.

Reads ``outputs/stage_chain_status.json`` (written by
:mod:`scripts.stage_chain`) plus every stage's ``training_summary.json`` and
prints / writes a markdown table with:

  * epochs actually completed vs requested, wall-clock, exit code;
  * the ``best_task`` (task = curve_rmse_m + 80 * collision_rate) and
    ``best`` (val total) epochs of each stage;
  * the validation curve summary: minimum task, minimum rmse, minimum
    collision, topology accuracy;
  * the historical-safety-feedback diagnostics of the last epoch and the
    average over the run (``fb_valid_rate`` / ``fb_raw_violation`` /
    ``fb_mean_violation`` / ``fb_correction`` / ``Lfbsafe`` / ``Lcurve``) - only
    the feedback-trained stages have these.

Usage:

    python scripts/compare_campaign.py
    python scripts/compare_campaign.py --status outputs/stage_chain_status.json \
        --md outputs/campaign_compare.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

FB_KEYS = ["fb_valid_rate", "fb_raw_violation", "fb_mean_violation",
           "fb_correction", "fb_delta_norm", "Lfbsafe", "Lcurve",
           "fb_topo_match"]


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_stage(entry: dict) -> dict:
    """One status entry + the stage's training history."""
    out = dict(entry)
    summary = _abs(os.path.join(entry["out_dir"], "training_summary.json"))
    history, meta = [], {}
    if os.path.exists(summary):
        with open(summary, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        history = data.get("history") or []
        meta = {k: data.get(k) for k in
                ("epochs_requested", "best_val_total", "best_epoch",
                 "best_task_score", "best_task_epoch", "elapsed_seconds",
                 "processed_root", "lr", "batch_size")}
    out["history"] = history
    out["meta"] = meta
    rows = []
    for e in history:
        v = e.get("val") or {}
        t = e.get("train") or {}
        rmse = v.get("curve_rmse_m")
        coll = v.get("collision_rate")
        rows.append({
            "epoch": e.get("epoch"),
            "task": (None if rmse is None or coll is None
                     else float(rmse) + 80.0 * float(coll)),
            "rmse": rmse, "coll": coll,
            "topo": v.get("pred_topo_best_rate"),
            "val": v.get("total"),
            **{k: t.get(k) for k in FB_KEYS},
        })
    out["rows"] = rows
    return out


def summarise(stage: dict) -> dict:
    rows = [r for r in stage["rows"] if r["task"] is not None]
    fb_rows = [r for r in rows if r.get("fb_valid_rate") is not None]
    def _best(key, fn=min):
        vals = [r for r in rows if r.get(key) is not None]
        return fn(vals, key=lambda r: r[key]) if vals else None
    res = {"name": stage["name"], "epochs_done": len(stage["rows"]),
           "exit_code": stage.get("exit_code"),
           "best_epoch": stage.get("best_epoch"),
           "meta": stage["meta"]}
    bt = _best("task")
    res.update({"best_task_epoch": None if bt is None else bt["epoch"],
                "best_task": None if bt is None else bt["task"],
                "best_rmse": None if bt is None else bt["rmse"],
                "best_coll": None if bt is None else bt["coll"]})
    br = _best("rmse")
    res["min_rmse_epoch"] = None if br is None else br["epoch"]
    bc = _best("coll")
    res["min_coll"] = None if bc is None else bc["coll"]
    res["min_coll_epoch"] = None if bc is None else bc["epoch"]
    bt2 = _best("topo", fn=max)
    res["best_topo"] = None if bt2 is None else bt2["topo"]
    if fb_rows:
        last = fb_rows[-1]
        res["fb_last"] = {k: last.get(k) for k in FB_KEYS}
        res["fb_mean"] = {k: sum(r[k] for r in fb_rows if r.get(k) is not None)
                          / max(1, sum(1 for r in fb_rows
                                       if r.get(k) is not None))
                          for k in FB_KEYS}
        res["fb_valid_first"] = fb_rows[0].get("fb_valid_rate")
        res["fb_valid_last"] = last.get("fb_valid_rate")
    return res


def _fmt(x, nd=3):
    return "-" if x is None else ("%.*f" % (nd, x))


def table(summaries: list) -> str:
    head = ("| stage | epochs | exit | best_task(ep) | task | rmse_m | coll | "
            "best_topo | fb_valid(first→last) | Lfbsafe | Lcurve |")
    sep = "|---" * 11 + "|"
    lines = [head, sep]
    for s in summaries:
        fb = s.get("fb_last") or {}
        lines.append("| %s | %d | %s | %s(%s) | %s | %s | %s | %s | %s | %s | %s |"
                     % (s["name"], s["epochs_done"], s["exit_code"],
                        _fmt(s["best_task"], 2), s["best_task_epoch"],
                        _fmt(s["best_task"], 2), _fmt(s["best_rmse"], 2),
                        _fmt(s["best_coll"], 4), _fmt(s["best_topo"], 3),
                        "%s→%s" % (_fmt(s.get("fb_valid_first")),
                                   _fmt(s.get("fb_valid_last"))),
                        _fmt(fb.get("Lfbsafe")), _fmt(fb.get("Lcurve"))))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="outputs/stage_chain_status.json")
    ap.add_argument("--md", default="outputs/campaign_compare.md")
    args = ap.parse_args()

    if not os.path.exists(_abs(args.status)):
        print("[compare] %s not found: the campaign has not finished a stage yet"
              % args.status)
        return 1
    with open(_abs(args.status), "r", encoding="utf-8") as fh:
        status = json.load(fh)

    summaries = [summarise(load_stage(e)) for e in status.get("stages", [])]
    if not summaries:
        print("[compare] no stage recorded yet")
        return 1

    text = table(summaries)
    print(text)
    report = ["# 160k8p campaign 对比（A_oneshot vs B1_base + B2_feedback）", "",
              text, ""]
    for s in summaries:
        m = s["meta"]
        report += ["## %s" % s["name"], "",
                   "* epochs done: **%d** (requested %s), exit code %s"
                   % (s["epochs_done"], m.get("epochs_requested"),
                      s["exit_code"]),
                   "* data: `%s`" % m.get("processed_root"),
                   "* lr %s, batch %s" % (m.get("lr"), m.get("batch_size")),
                   "* best_task epoch %s (score %s); val-total best epoch %s"
                   % (s["best_task_epoch"], _fmt(s["best_task"], 3),
                      m.get("best_epoch")),
                   "* min rmse %.3f m @ epoch %s; min collision %s @ epoch %s; "
                   "best topo %s" % (s["best_rmse"] or float("nan"),
                                     s["min_rmse_epoch"],
                                     _fmt(s["min_coll"], 4),
                                     s["min_coll_epoch"],
                                     _fmt(s["best_topo"])),
                   ""]
        if s.get("fb_mean"):
            report += ["| fb metric | last epoch | run mean |", "|---|---|---|"]
            for k in FB_KEYS:
                report.append("| %s | %s | %s |"
                              % (k, _fmt(s["fb_last"].get(k)),
                                 _fmt(s["fb_mean"].get(k))))
            report.append("")
    md = _abs(args.md)
    os.makedirs(os.path.dirname(md) or ".", exist_ok=True)
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n[compare] written %s" % md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
