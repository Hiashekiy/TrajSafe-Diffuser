"""finalize_night.py - after training: validate, evaluate, preview, report.

    python scripts/finalize_night.py --out outputs/bspline_carla

Steps
  1. copy the cleaning/preprocessing reports into the output directory;
  2. run the full processed-cache validator (03_validate_processed.py);
  3. run evaluate.py on the test split with the best checkpoint;
  4. run sample.py to produce qualitative previews and copy the first one to
     ``sample_preview.png``;
  5. write ``NIGHT_RUN_REPORT.md`` (scripts/night_report.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def run(cmd, log_path, timeout=None):
    print("[run] %s" % " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write("\n" + "=" * 78 + "\n[cmd] %s\n" % " ".join(cmd))
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/carla_processed")
    ap.add_argument("--out", default="outputs/bspline_carla")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--skip-train-check", action="store_true")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    proc_root = os.path.abspath(args.processed)
    os.makedirs(out, exist_ok=True)
    py = sys.executable
    # The val-total minimum (best.pt) is dominated by the topology CE, which
    # overfits early, so the task-metric checkpoint (best_task.pt) is preferred
    # when it exists; latest.pt is evaluated as well.
    candidates = [args.ckpt] if args.ckpt else [
        os.path.join(out, "ckpt", "best_task.pt"),
        os.path.join(out, "ckpt", "best.pt"),
        os.path.join(out, "ckpt", "latest.pt")]
    ckpt = next((c for c in candidates if c and os.path.exists(c)), candidates[-1])
    print("[ckpt] using %s" % ckpt, flush=True)

    # 1. copy reports -------------------------------------------------------
    for name in ("cleaning_report.json", "clean_manifest.jsonl",
                 "preprocess_candidates_report.json",
                 "preprocess_labels_report.json", "preprocess_report.json"):
        src = os.path.join(proc_root, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, name))
            print("[copy] %s" % name)

    # 2. validator ----------------------------------------------------------
    run([py, os.path.join(ROOT, "scripts", "data", "carla",
                          "03_validate_processed.py"),
         "--processed", proc_root, "--config", args.config,
         "--allow-warnings"],
        os.path.join(out, "validate.log"))

    # 3. evaluation ---------------------------------------------------------
    if os.path.exists(ckpt):
        run([py, os.path.join(ROOT, "evaluate.py"), "--config", args.config,
             "--ckpt", ckpt, "--split", "test", "--num-batches", "2",
             "--runs", "2", "--out", os.path.join(out, "eval_test.json")],
            os.path.join(out, "eval.log"))
        latest_ckpt = os.path.join(out, "ckpt", "latest.pt")
        if os.path.exists(latest_ckpt) and os.path.abspath(latest_ckpt) != os.path.abspath(ckpt):
            run([py, os.path.join(ROOT, "evaluate.py"), "--config", args.config,
                 "--ckpt", latest_ckpt, "--split", "test", "--num-batches", "2",
                 "--runs", "2", "--out", os.path.join(out, "eval_test_latest.json")],
                os.path.join(out, "eval.log"))
        # 4. previews
        run([py, os.path.join(ROOT, "sample.py"), "--config", args.config,
             "--ckpt", ckpt, "--split", "test", "--num", "3", "--seed", "0",
             "--out", out], os.path.join(out, "sample.log"))
        pngs = sorted(glob.glob(os.path.join(out, "samples_test_*_*.png")))
        if pngs:
            shutil.copy2(pngs[0], os.path.join(out, "sample_preview.png"))
            print("[copy] %s -> sample_preview.png" % os.path.basename(pngs[0]))
        pngs2 = sorted(glob.glob(os.path.join(out, "trace_test_*_*.png")))
        if pngs2:
            shutil.copy2(pngs2[0], os.path.join(out, "sample_trace.png"))
    else:
        print("[warn] no checkpoint at %s" % ckpt, flush=True)

    # 5. night report -------------------------------------------------------
    run([py, os.path.join(ROOT, "scripts", "night_report.py"),
         "--out", out, "--processed", proc_root],
        os.path.join(out, "night_report.log"))

    summary = {}
    sp = os.path.join(out, "training_summary.json")
    if os.path.exists(sp):
        summary = json.load(open(sp, encoding="utf-8"))
    ev = {}
    ep = os.path.join(out, "eval_test.json")
    if os.path.exists(ep):
        try:
            ev = json.load(open(ep, encoding="utf-8"))
        except Exception:
            ev = {}
    ev_latest = {}
    ep2 = os.path.join(out, "eval_test_latest.json")
    if os.path.exists(ep2):
        try:
            ev_latest = json.load(open(ep2, encoding="utf-8"))
        except Exception:
            ev_latest = {}
    keys = ("traj_collision", "goal_dist_m", "curve_rmse_m", "ctrl_rmse_m",
            "ellipse_collision", "sel_best_rate", "pred_topo_best_rate",
            "recall")
    print(json.dumps({
        "epochs_done": summary.get("epochs_done"),
        "best_epoch": summary.get("best_epoch"),
        "best_val_total": summary.get("best_val_total"),
        "best_ckpt": summary.get("best_ckpt"),
        "latest_ckpt": summary.get("latest_ckpt"),
        "ckpt_evaluated": ckpt,
        "eval": {k: ev.get(k) for k in keys},
        "eval_latest": {k: ev_latest.get(k) for k in keys},
        "report": os.path.join(out, "NIGHT_RUN_REPORT.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
