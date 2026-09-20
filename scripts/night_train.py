"""night_train.py - restart-safe supervisor for the overnight CARLA training run.

The GPU is shared with other processes on this machine (CARLA, the desktop), so
a transient CUDA failure ("unable to find an engine", "unknown error", OOM)
must not end the night:

  * the whole training process is restarted automatically;
  * after a crash it resumes from ``<ckpt-dir>/latest.pt``;
  * a global wall-clock deadline (``--max-hours``) is enforced across restarts
    by passing the REMAINING hours to every attempt;
  * every child writes to ``<out-dir>/train.log`` (append).

    python scripts/night_train.py --config configs/config.yaml --max-hours 5.5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="total wall-clock budget across all restarts")
    ap.add_argument("--deadline", default=None,
                    help="absolute stop time 'YYYY-mm-dd HH:MM[:SS]'; survives "
                         "supervisor restarts (preferred over --max-hours)")
    ap.add_argument("--max-restarts", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ckpt-dir", default="outputs/bspline_carla/ckpt")
    ap.add_argument("--out-dir", default="outputs/bspline_carla")
    ap.add_argument("--preview-png", default=None)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from on the FIRST attempt too")
    ap.add_argument("--init-best-val", type=float, default=None)
    ap.add_argument("--init-best-task", type=float, default=None)
    ap.add_argument("--restart-sleep", type=float, default=30.0)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    latest = os.path.join(args.ckpt_dir, "latest.pt")
    t0 = time.time()
    attempts = 0

    while attempts <= args.max_restarts:
        attempts += 1
        remaining = None
        if args.deadline:
            deadline = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M")
                                   ) if len(args.deadline) <= 16 else time.mktime(
                time.strptime(args.deadline, "%Y-%m-%d %H:%M:%S"))
            remaining = (deadline - time.time()) / 3600.0
            if remaining <= 0.03:
                print("[supervisor] deadline %s reached" % args.deadline,
                      flush=True)
                break
        if args.max_hours:
            by_hours = float(args.max_hours) - (time.time() - t0) / 3600.0
            remaining = by_hours if remaining is None else min(remaining, by_hours)
            if remaining <= 0.03:
                print("[supervisor] wall-clock budget exhausted", flush=True)
                break
        cmd = [sys.executable, os.path.join(ROOT, "train.py"),
               "--config", args.config,
               "--ckpt-dir", args.ckpt_dir,
               "--out-dir", args.out_dir]
        if args.epochs is not None:
            cmd += ["--epochs", str(int(args.epochs))]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(int(args.batch_size))]
        if args.accum is not None:
            cmd += ["--accum", str(int(args.accum))]
        if args.lr is not None:
            cmd += ["--lr", str(float(args.lr))]
        if args.preview_png:
            cmd += ["--preview-png", args.preview_png]
        if remaining is not None:
            cmd += ["--max-hours", "%.4f" % remaining]
        if attempts > 1 and os.path.exists(latest):
            cmd += ["--resume", latest]
        elif args.resume:
            cmd += ["--resume", args.resume]
        if args.init_best_val is not None:
            cmd += ["--init-best-val", str(float(args.init_best_val))]
        if args.init_best_task is not None:
            cmd += ["--init-best-task", str(float(args.init_best_task))]
        cmd += list(args.extra)

        with open(log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write("\n" + "=" * 78 + "\n")
            log.write("[supervisor] attempt %d  %s\n"
                      % (attempts, time.strftime("%Y-%m-%d %H:%M:%S")))
            log.write("[supervisor] %s\n" % " ".join(cmd))
            log.flush()
            print("[supervisor] attempt %d -> %s" % (attempts, " ".join(cmd)),
                  flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT)
            code = proc.wait()
            log.write("[supervisor] exit code %d\n" % code)
            log.flush()
        if code == 0:
            print("[supervisor] training finished cleanly", flush=True)
            break
        print("[supervisor] child exited with %d; restarting in %.0fs"
              % (code, args.restart_sleep), flush=True)
        time.sleep(max(1.0, float(args.restart_sleep)))

    print("[supervisor] done after %d attempt(s)" % attempts, flush=True)


if __name__ == "__main__":
    main()
