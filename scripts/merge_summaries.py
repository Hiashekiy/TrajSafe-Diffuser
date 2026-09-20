"""merge_summaries.py - merge the run-1 summary with the continuation summary.

    python scripts/merge_summaries.py --out-dir outputs/bspline_carla

Produces a single ``training_summary.json`` whose ``history`` covers BOTH runs
(epochs deduplicated, later run wins) and keeps the run-1 best-checkpoint
information under the ``run1`` key.
"""
from __future__ import annotations

import argparse
import json
import os

KEYS = ("epochs_done", "best_epoch", "best_val_total", "best_ckpt",
        "best_task_ckpt", "best_task_epoch", "best_task_score", "finished_at")


def _load(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/bspline_carla")
    ap.add_argument("--run1", default=None)
    args = ap.parse_args()
    out = args.out_dir
    run1_path = args.run1 or os.path.join(out, "training_summary_run1.json")
    cur_path = os.path.join(out, "training_summary.json")
    run1 = _load(run1_path)
    cur = _load(cur_path)
    hist = {}
    for entry in list(run1.get("history", [])) + list(cur.get("history", [])):
        try:
            hist[int(entry["epoch"])] = entry
        except Exception:
            continue
    merged = dict(cur)
    merged["history"] = [hist[k] for k in sorted(hist)]
    merged["epochs_done"] = max(int(run1.get("epochs_done") or 0),
                                int(cur.get("epochs_done") or 0))
    merged["run1"] = {k: run1.get(k) for k in KEYS}
    merged["resumed_from"] = "ckpt/latest.pt @ epoch 272 (run1 -> run2)"
    with open(cur_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print("merged history entries: %d | epochs_done: %s | best_val: %s "
          "(run1 %s) | best_task: %s (run1 %s)"
          % (len(merged["history"]), merged["epochs_done"],
             merged.get("best_val_total"), run1.get("best_val_total"),
             merged.get("best_task_score"), run1.get("best_task_score")))


if __name__ == "__main__":
    main()
