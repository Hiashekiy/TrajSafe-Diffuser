"""stage_chain.py - run several supervised training stages back to back.

The overnight comparison campaign needs several training runs that must follow
each other on the SAME GPU (a 12 GB card cannot host two of them):

    arm A  one-shot : feedback enabled from scratch
    arm B  two-stage: base (feedback off)  ->  fine-tune (feedback on)

Every stage is launched through :mod:`scripts.night_train`, so a transient CUDA
failure restarts that stage from its ``latest.pt`` and a per-stage wall-clock
budget (``max_hours``) is enforced by ``train.py`` itself: a stage that runs out
of time exits CLEANLY at the end of an epoch, leaving ``best_task.pt`` /
``latest.pt`` behind.  That is what makes an overnight plan safe: the budgets
bound the whole chain, and every stage still delivers its best-so-far model.

Plan file (JSON list).  Per stage:

    name                  label used in the status file and by
                          ``resume_best_task_of``
    config                YAML for train.py
    ckpt_dir / out_dir    where the stage writes
    epochs                int, or "+N" = resume_epoch + 1 + N
    max_hours             wall-clock budget of THIS stage
    batch, accum, lr      optional, forwarded to train.py (defaults from YAML)
    resume                optional explicit checkpoint path
    resume_best_task_of   optional stage name -> <its out_dir>/ckpt/best_task.pt
    init_best_task        optional float, seeds the stage's best_task tracking

Usage:

    python scripts/stage_chain.py --plan configs/campaign_160k8p_compare.json
    python scripts/stage_chain.py --plan ... --dry-run
    python scripts/stage_chain.py --plan ... --only B2_feedback

The status file (``--status``, default ``outputs/stage_chain_status.json``) is
rewritten after every stage, so a later run can see what already finished
(``--skip-done`` skips the stages marked ``exit_code == 0``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DEFAULT_STATUS = os.path.join("outputs", "stage_chain_status.json")


def checkpoint_epoch(path: str) -> int:
    """Epoch stored inside a checkpoint (``-1`` when it cannot be read)."""
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and data.get("epoch") is not None:
        return int(data["epoch"])
    return -1


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_status(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return {"stages": []}


def save_status(path: str, status: dict) -> None:
    os.makedirs(os.path.dirname(_abs(path)) or ".", exist_ok=True)
    tmp = _abs(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, _abs(path))


def resolve_resume(spec: dict, done: dict) -> str | None:
    if spec.get("resume"):
        return _abs(spec["resume"])
    other = spec.get("resume_best_task_of")
    if not other:
        return None
    ref = done.get(other)
    if not ref:
        raise SystemExit("[chain] stage %r needs %r, which has not run yet"
                         % (spec.get("name"), other))
    path = os.path.join(_abs(ref["out_dir"]), "ckpt", "best_task.pt")
    if not os.path.exists(path):
        raise SystemExit("[chain] %s not found for stage %r"
                         % (path, spec.get("name")))
    return path


def resolve_epochs(spec: dict, resume: str | None) -> int:
    epochs = spec.get("epochs", 200)
    if isinstance(epochs, str) and epochs.strip().startswith("+"):
        extra = int(epochs.strip()[1:])
        base = checkpoint_epoch(resume) if resume else -1
        return base + 1 + extra
    return int(epochs)


def stage_argv(spec: dict, epochs: int, resume: str | None) -> list:
    argv = [sys.executable, os.path.join(ROOT, "scripts", "night_train.py"),
            "--config", spec["config"],
            "--ckpt-dir", spec["ckpt_dir"], "--out-dir", spec["out_dir"],
            "--epochs", str(int(epochs)),
            "--max-hours", "%.4f" % float(spec["max_hours"])]
    for key, flag in (("batch", "--batch-size"), ("accum", "--accum")):
        if spec.get(key) is not None:
            argv += [flag, str(int(spec[key]))]
    if spec.get("lr") is not None:
        argv += ["--lr", "%.6g" % float(spec["lr"])]
    if resume:
        argv += ["--resume", resume]
    if spec.get("init_best_task") is not None:
        argv += ["--init-best-task", "%.6f" % float(spec["init_best_task"])]
    argv += list(spec.get("extra") or [])
    return argv


def run_stage(spec: dict, status: dict, status_path: str) -> dict:
    name = spec["name"]
    done = {s["name"]: s for s in status["stages"] if s.get("exit_code") == 0}
    resume = resolve_resume(spec, done)
    epochs = resolve_epochs(spec, resume)
    argv = stage_argv(spec, epochs, resume)
    print("[chain] %s: epochs=%d resume=%s" % (name, epochs, resume), flush=True)
    print("[chain]   %s" % " ".join(argv), flush=True)
    os.makedirs(_abs(spec["out_dir"]), exist_ok=True)
    log_path = os.path.join(_abs(spec["out_dir"]), "chain.log")
    entry = {"name": name, "config": spec["config"],
             "out_dir": spec["out_dir"], "ckpt_dir": spec["ckpt_dir"],
             "epochs": epochs, "resume": resume,
             "max_hours": float(spec["max_hours"]),
             "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "exit_code": None, "best_epoch": None, "finished": None}
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write("\n" + "=" * 78 + "\n")
        log.write("[chain] %s started %s\n" % (name, entry["started"]))
        log.write("[chain] %s\n" % " ".join(argv))
        log.flush()
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT)
        entry["pid"] = proc.pid
        code = proc.wait()
    entry["exit_code"] = code
    entry["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    best = os.path.join(_abs(spec["ckpt_dir"]), "best_task.pt")
    entry["best_epoch"] = checkpoint_epoch(best) if os.path.exists(best) else None
    entry["summary"] = os.path.join(_abs(spec["out_dir"]),
                                    "training_summary.json")
    status["stages"] = [s for s in status["stages"] if s.get("name") != name]
    status["stages"].append(entry)
    save_status(status_path, status)
    print("[chain] %s exited %d, best_task epoch %s"
          % (name, code, entry["best_epoch"]), flush=True)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--status", default=DEFAULT_STATUS)
    ap.add_argument("--only", default=None,
                    help="run a single stage by name (ignores the rest)")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip stages already recorded with exit_code 0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(_abs(args.plan), "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    if isinstance(plan, dict):
        plan = plan["stages"]
    status = load_status(args.status)
    done_names = {s["name"] for s in status["stages"] if s.get("exit_code") == 0}

    print("[chain] plan %s: %d stage(s), %s"
          % (args.plan, len(plan),
             "%.2f h total" % sum(float(s["max_hours"]) for s in plan)),
          flush=True)
    t0 = time.time()
    for spec in plan:
        name = spec["name"]
        if args.only and name != args.only:
            continue
        if args.skip_done and name in done_names:
            print("[chain] skip %s (already done)" % name, flush=True)
            continue
        if args.dry_run:
            done = {s["name"]: s for s in status["stages"]}
            try:
                resume = resolve_resume(spec, done)
                epochs = resolve_epochs(spec, resume)
                print("[chain] DRY %s -> %s" % (name, " ".join(
                    stage_argv(spec, epochs, resume))), flush=True)
            except SystemExit as exc:
                print("[chain] DRY %s -> %s" % (name, exc), flush=True)
            continue
        run_stage(spec, status, args.status)
    print("[chain] done in %.2f h" % ((time.time() - t0) / 3600.0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
