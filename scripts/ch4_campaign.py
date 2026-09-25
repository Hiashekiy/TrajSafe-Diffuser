#!/usr/bin/env python
"""第4章正式实验的编排器（指南「实际执行顺序」第 10-13 步）.

按阶段调用 ``scripts/eval_chapter4.py``，每个「方法 × seed」写一个独立 JSON，
已存在就跳过（可断点续跑）。四个外部处理器的超参数从
``outputs/ch4/sweep_<mode>.json`` 的 ``winner`` 读取，即验证集上冻结的那一组。

阶段::

    main        4.2 主表：Diffusion + 4 个 SDF 基线 (D-C48) + 本文完整方法 (A-C48)
                × seed 0/1/2 × test 420
    ablation_a  4.3.1 推理期消融 A0..A3（同一 A-C48 checkpoint）
    ablation_t  4.3.2 训练期 2x2 消融 T0..T3（S-C48 与 A-C48，全部关闭末端投影）
    smoke       每方法 16 样本，用于跑通管线

用法::

    python scripts/ch4_campaign.py --stage smoke
    python scripts/ch4_campaign.py --stage main --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

#: every run that failed, reported at the end (see run())
FAILURES: list = []

D_CKPT = "outputs/oneshot_k4p_c48_d/ckpt/best_task.pt"
D_CFG = "configs/config_160k4p_c48_d_std.yaml"
A_CKPT = "outputs/oneshot_k4p_c48/ckpt/best_task.pt"
A_CFG = "configs/config_160k4p_c48_oneshot.yaml"
S_CKPT = "outputs/oneshot_k4p_c48_s/ckpt/best_task.pt"
S_CFG = "configs/config_160k4p_c48_s_nofb.yaml"


def frozen_params(mode: str, required: bool = True) -> dict:
    """The validation-selected hyper-parameters of one SDF processor."""
    path = os.path.join(ROOT, "outputs", "ch4", "sweep_%s.json" % mode)
    if not os.path.exists(path):
        if required:
            raise SystemExit(
                "[campaign] missing %s -- run scripts/ch4_sweep_val.py "
                "--guidance %s first" % (path, mode))
        # smoke only: fall back to the first grid point so the pipeline can be
        # exercised before the validation sweep has run
        from ch4_sweep_val import GRIDS      # same directory
        fallback = {key: values[0] for key, values in GRIDS[mode].items()}
        print("[campaign] WARNING: no frozen params for %s, using %s"
              % (mode, json.dumps(fallback)), flush=True)
        return fallback
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    params = payload["winner"]
    print("[campaign] frozen %s = %s" % (mode, json.dumps(params)), flush=True)
    return params


def run(name: str, config: str, ckpt: str, guidance: str, seeds, split: str,
        samples: int, chunk: int, steps: int, extra, out_dir: str,
        records: bool = True, force: bool = False) -> list:
    made = []
    for seed in seeds:
        out_json = os.path.join(ROOT, out_dir, "%s__s%d.json" % (name, seed))
        if os.path.exists(out_json) and not force:
            print("[campaign] skip %s seed=%d (exists)" % (name, seed), flush=True)
            made.append(out_json)
            continue
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "eval_chapter4.py"),
               "--config", config, "--ckpt", ckpt, "--method", name,
               "--guidance", guidance, "--split", split,
               "--samples", str(samples), "--chunk", str(chunk),
               "--steps", str(steps), "--seed", str(seed), "--device", "cuda",
               "--out", out_json, "--md", out_json[:-5] + ".md",
               "--records", out_json[:-5] + ".jsonl"]
        for item in extra:
            cmd += item if isinstance(item, list) else [item]
        print("[campaign] RUN %s seed=%d guidance=%s" % (name, seed, guidance),
              flush=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            # Do NOT abort the whole campaign: one broken method used to take the
            # remaining methods (and every later seed) down with it, which cost a
            # complete re-run.  Record it and keep going; the exit code at the
            # end still reports the failure loudly.
            print("[campaign] FAILED %s seed=%d (rc=%d) -- continuing"
                  % (name, seed, proc.returncode), flush=True)
            FAILURES.append("%s seed=%d (rc=%d)" % (name, seed, proc.returncode))
            continue
        print("[campaign] done %s seed=%d in %.0fs"
              % (name, seed, time.time() - started), flush=True)
        made.append(out_json)
    return made


def pairs(params: dict):
    return [["--sdf-set", "%s=%s" % (k, v)] for k, v in sorted(params.items())]


def main() -> int:
    global D_CKPT, A_CKPT, S_CKPT, D_CFG, A_CFG, S_CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "main", "ablation_a", "ablation_t", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out-dir", default="outputs/ch4/runs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--d-ckpt", default=D_CKPT,
                    help="override the D-C48 checkpoint (default %s)" % D_CKPT)
    ap.add_argument("--a-ckpt", default=A_CKPT, help="override the A-C48 checkpoint")
    ap.add_argument("--s-ckpt", default=S_CKPT, help="override the S-C48 checkpoint")
    ap.add_argument("--d-config", default=D_CFG, help="override the D-C48 config")
    ap.add_argument("--a-config", default=A_CFG, help="override the A-C48 config")
    ap.add_argument("--s-config", default=S_CFG, help="override the S-C48 config")
    args = ap.parse_args()
    D_CKPT, A_CKPT, S_CKPT = args.d_ckpt, args.a_ckpt, args.s_ckpt
    D_CFG, A_CFG, S_CFG = args.d_config, args.a_config, args.s_config

    stages = (["main", "ablation_a", "ablation_t"] if args.stage == "all"
              else [args.stage])
    if args.stage == "smoke":
        args.samples = 16
        args.chunk = 16
        args.out_dir = "outputs/ch4/smoke"
        args.seeds = [0]
        stages = ["main"]          # the smoke stage runs the main table, tiny

    for stage in stages:
        if stage == "main":
            seeds = args.seeds
            strict = args.stage != "smoke"
            frozen = {mode: frozen_params(mode, required=strict) for mode in
                      ("penalty", "projection", "primal_dual", "alm_sdf")}
            run("Diffusion", D_CFG, D_CKPT, "none", seeds, args.split,
                args.samples, args.chunk, args.steps, [], args.out_dir,
                force=args.force)
            run("Penalty-guided", D_CFG, D_CKPT, "penalty", seeds, args.split,
                args.samples, args.chunk, args.steps, pairs(frozen["penalty"]),
                args.out_dir, force=args.force)
            run("Projected", D_CFG, D_CKPT, "projection", seeds, args.split,
                args.samples, args.chunk, args.steps, pairs(frozen["projection"]),
                args.out_dir, force=args.force)
            run("Primal-Dual", D_CFG, D_CKPT, "primal_dual", seeds, args.split,
                args.samples, args.chunk, args.steps, pairs(frozen["primal_dual"]),
                args.out_dir, force=args.force)
            run("SDF-ALM", D_CFG, D_CKPT, "alm_sdf", seeds, args.split,
                args.samples, args.chunk, args.steps, pairs(frozen["alm_sdf"]),
                args.out_dir, force=args.force)
            run("Full (A-C48+ALM+FB+Proj)", A_CFG, A_CKPT, "alm", seeds,
                args.split, args.samples, args.chunk, args.steps,
                [["--feedback", "on"], ["--final-project", "on"]],
                args.out_dir, force=args.force)
        elif stage == "ablation_a":
            # guide 4.3.1 -- same A-C48 checkpoint, only the inference switches
            table = {
                "A0": ("none", []),
                "A1": ("alm", [["--feedback", "off"], ["--final-project", "off"]]),
                "A2": ("alm", [["--feedback", "on"], ["--final-project", "off"]]),
                "A3": ("alm", [["--feedback", "on"], ["--final-project", "on"]]),
            }
            for tag, (guidance, extra) in table.items():
                run(tag, A_CFG, A_CKPT, guidance, args.seeds, args.split,
                    args.samples, args.chunk, args.steps, extra, args.out_dir,
                    force=args.force)
        elif stage == "ablation_t":
            # guide 4.3.2 -- 2x2 over (training feedback) x (inference ALM),
            # terminal projection OFF everywhere so it cannot mask the effect
            table = [
                ("T0_S-C48_noalm", S_CFG, S_CKPT, "none", []),
                ("T1_S-C48_alm", S_CFG, S_CKPT, "alm",
                 [["--feedback", "off"], ["--final-project", "off"]]),
                ("T2_A-C48_noalm", A_CFG, A_CKPT, "none", []),
                ("T3_A-C48_alm", A_CFG, A_CKPT, "alm",
                 [["--feedback", "on"], ["--final-project", "off"]]),
            ]
            for tag, config, ckpt, guidance, extra in table:
                run(tag, config, ckpt, guidance, args.seeds, args.split,
                    args.samples, args.chunk, args.steps, extra, args.out_dir,
                    force=args.force)
    if FAILURES:
        print("[campaign] stage(s) %s finished WITH FAILURES:" % ",".join(stages),
              flush=True)
        for item in FAILURES:
            print("   FAILED %s" % item, flush=True)
        return 1
    print("[campaign] stage(s) %s finished" % ",".join(stages), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
