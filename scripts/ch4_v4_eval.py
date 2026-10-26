#!/usr/bin/env python
"""Run the chapter-4 "Full" protocol row for BOTH checkpoints we must compare.

The point is a SINGLE-VARIABLE comparison under one code version and one data
root (``data/processed/160m-k4p-c48``):

    Full (A-C48+ALM+FB+Proj)     v3  A-C48      outputs/models/oneshot_k4p_c48/ckpt/best_task.pt
    Full (v4-map16+ALM+FB+Proj)  v4  best_task  outputs/models/v4_map16_c48/ckpt/best_task.pt

Both rows use exactly the switches ``ch4_campaign.py --stage main`` uses for its
"Full" row (``--guidance alm --feedback on --final-project on``), the same 16
reverse steps, the same warm-up, the same frozen ALM config and the same test
split (420 samples), so the archived table and this table are the same protocol.

Everything is delegated to ``scripts/eval_chapter4.py`` through
``ch4_campaign.run`` so the JSON/MD/JSONL layout, resume-on-exists and failure
bookkeeping are identical to the official campaign.

    python scripts/ch4_v4_eval.py --seeds 0 1 2
    python scripts/ch4_v4_eval.py --seeds 0 --samples 8 --out-dir outputs/chapter4_v4/smoke
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ch4_campaign import FAILURES, run          # noqa: E402

A_CFG = "configs/config_160k4p_c48_oneshot.yaml"
A_CKPT = "outputs/models/oneshot_k4p_c48/ckpt/best_task.pt"
V4_CFG = "configs/config_160k4p_c48_v4_map16.yaml"
V4_CKPT = "outputs/models/v4_map16_c48/ckpt/best_task.pt"
L1_CFG = "configs/config_160k4p_c48_v4_map16_l1.yaml"
L1_CKPT = "outputs/models/v4_map16_c48_l1/ckpt/best_task.pt"

# the exact switch set of the campaign's "Full" row
FULL_EXTRA = [["--feedback", "on"], ["--final-project", "on"]]

#: (tag, method name, config, checkpoint) -- the tag is what --only matches
ROWS = [
    ("v3", "Full (A-C48+ALM+FB+Proj)", A_CFG, A_CKPT),
    ("v4", "Full (v4-map16+ALM+FB+Proj)", V4_CFG, V4_CKPT),
    ("l1", "Full (v4-map16-L1+ALM+FB+Proj)", L1_CFG, L1_CKPT),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out-dir", default="outputs/chapter4_v4/runs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, nargs="+",
                    help="limit to rows by tag (v3 / v4 / l1)")
    args = ap.parse_args()

    for tag, name, config, ckpt in ROWS:
        if args.only and tag not in args.only:
            continue
        if not os.path.exists(os.path.join(ROOT, ckpt)):
            print("[v4-eval] MISSING checkpoint %s -- skipping %s"
                  % (ckpt, name), flush=True)
            FAILURES.append("%s (missing ckpt)" % name)
            continue
        run(name, config, ckpt, "alm", args.seeds, args.split, args.samples,
            args.chunk, args.steps, [list(x) for x in FULL_EXTRA],
            args.out_dir, force=args.force)

    if FAILURES:
        print("[v4-eval] finished WITH FAILURES:", flush=True)
        for item in FAILURES:
            print("   FAILED %s" % item, flush=True)
        return 1
    print("[v4-eval] finished -> %s" % args.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
