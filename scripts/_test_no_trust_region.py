#!/usr/bin/env python
"""_test_no_trust_region.py - what the ALM does with the trust region removed.

``_assert_common_shapes`` (planner.py) refuses a forward whose decoded curves do
not share their endpoints, and the 420-sample eval trips it ~160 samples in once
``alm.max_curve_step_scene`` is off.  The endpoint of a clamped B-spline IS the
first/last control, and the boundary decoder re-imposes them as

    Q*_0 = Q~_0 + 1.0 * (S - Q~_0)

which in float32 loses everything once ``Q~`` blows up.  This script runs the
SAME protocol with that one assert downgraded to a log line, so we get both the
collision count and the magnitude of the blow-up (how large the control polygon
gets before the next forward).

    python scripts/_test_no_trust_region.py
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = "D:/ProjectDirectory/Neural-IRISDiffuser"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.models.trajsafe.planner import TrajSafePlanner  # noqa: E402

_ORIG = TrajSafePlanner._assert_common_shapes
STATE = {"trips": 0, "worst": 0.0, "worst_mag": 0.0, "logged": 0}


def _endpoint_errors(out):
    ic = out["input_curve"]
    e = {}
    for k in ("input_curve", "coarse", "final"):
        p = out[k]
        e[k] = (float((p[:, 0] - ic[:, 0]).abs().max()),
                float((p[:, -1] - ic[:, -1]).abs().max()))
    return e


def patched(self, out, B):
    try:
        _ORIG(self, out, B)
    except AssertionError:
        e = _endpoint_errors(out)
        worst = max(max(v) for v in e.values())
        if worst < 1e-5:
            raise                      # a DIFFERENT assert failed: do not hide it
        mag = float(out["input_control"].abs().max())
        STATE["trips"] += 1
        STATE["worst"] = max(STATE["worst"], worst)
        STATE["worst_mag"] = max(STATE["worst_mag"], mag)
        if STATE["logged"] < 6:
            STATE["logged"] += 1
            print("  [assert] endpoint mismatch %s  max|input_control| = %.3e"
                  % ({k: (round(a, 3), round(b, 3)) for k, (a, b) in e.items()}, mag),
                  flush=True)


TrajSafePlanner._assert_common_shapes = patched

import eval_campaign_testset as E  # noqa: E402

if __name__ == "__main__":
    sys.argv = [
        "eval_campaign_testset.py",
        "--config", "configs/config_160k4p_c48_oneshot.yaml",
        "--split", "test", "--samples", "0", "--chunk", "32",
        "--steps", "16", "--seed", "0",
        "--model", "K4P_c48_oneshot=outputs/oneshot_k4p_c48/ckpt/best_task.pt"
                   "::configs/config_160k4p_c48_oneshot.yaml",
        "--out", "outputs/eval_k4p_c48_alm_notrust.json",
        "--md", "outputs/eval_k4p_c48_alm_notrust.md",
    ]
    code = E.main()
    print()
    print("=== trust-region OFF summary ===")
    print("  endpoint asserts tripped : %d" % STATE["trips"])
    print("  worst endpoint mismatch  : %.3e" % STATE["worst"])
    print("  worst max|control|       : %.3e" % STATE["worst_mag"])
    raise SystemExit(code)
