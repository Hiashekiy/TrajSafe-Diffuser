"""Regression tests for the chapter-4 per-step aggregation.

The bug these pin down (it killed the paper's own method AFTER all 420 samples
had been processed): ``aggregate_steps`` built ONE active mask of
``args.chunk`` samples and reused it for every chunk, but the last chunk of a
split is ragged (420 = 13 x 32 + 4).  Consequences were two-sided:

* the corridor-ALM diagnostics indexed a 4-row ``lambda_max`` with a 32-row
  mask -> ``IndexError``;
* the external-SDF series silently DROPPED the final chunk (a 4-element series
  can never match a 32-element mask), so 4/420 samples were missing from every
  4.4 process curve without any warning.

The aggregation must size its mask from each STEP's own batch.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_chapter4 import (FEASIBLE_EPS, _step_batch,  # noqa: E402
                          aggregate_steps)


#: the sampler leaves the first ``warmup_reverse_steps`` (3) reverse steps
#: untouched: ``guided`` is all-False and BOTH the ALM stats and the guard stats
#: are None, because no processor ran at all.  The fixture must reproduce that or
#: it tests a trace that cannot exist.
WARMUP = 3


def _step(batch: int, step_index: int, guided_from: int = 4,
          alm: bool = True) -> dict:
    """One synthetic trace step for a chunk of ``batch`` samples."""
    if step_index < WARMUP:
        return {
            "t": 16 - step_index, "s": 15 - step_index,
            "guided": torch.zeros(batch, dtype=torch.bool),
            "q0_raw": torch.zeros(batch, 48, 2),
            "q0_safe": torch.zeros(batch, 48, 2),
            "raw_violation": None, "alm_correction": None,
            "smooth_raw": torch.full((batch,), 0.01),
            "smooth_safe": torch.full((batch,), 0.01),
            "guard_violation_before": None, "guard_violation_after": None,
            "guard_smoothness_before": None, "guard_smoothness_after": None,
            "guard_correction": None, "alm_stats": None,
        }
    guided = torch.zeros(batch, dtype=torch.bool)
    guided[guided_from:] = True
    # The corridor-ALM path never fills the guard_* keys (those belong to the
    # four external SDF baselines), so a corridor trace has them all None.
    guard = (None if alm else {
        "guard_violation_before": torch.full((batch,), 0.5),
        "guard_violation_after": torch.full((batch,), 0.1),
        "guard_smoothness_before": torch.full((batch,), 0.01),
        "guard_smoothness_after": torch.full((batch,), 0.02),
        "guard_correction": torch.full((batch,), 0.25),
    })
    step = {
        "t": 16 - step_index,
        "s": 15 - step_index,
        "guided": guided,
        "q0_raw": torch.zeros(batch, 48, 2),
        "q0_safe": torch.zeros(batch, 48, 2),
        "raw_violation": None if not alm else torch.full((batch,), 0.5),
        "alm_correction": None if not alm else torch.full((batch,), 0.25),
        "smooth_raw": torch.full((batch,), 0.01),
        "smooth_safe": torch.full((batch,), 0.02),
        "guard_violation_before": None, "guard_violation_after": None,
        "guard_smoothness_before": None, "guard_smoothness_after": None,
        "guard_correction": None,
    }
    if guard:
        step.update(guard)
    if alm:
        step["alm_stats"] = {
            # per-sample (batch) quantities
            "max_violation_before": torch.full((batch,), 0.5),
            "max_violation_after": torch.full((batch,), 0.1),
            "max_curve_correction_scene": torch.full((batch,), 0.25),
            "inner_steps_used": torch.full((batch,), 6.0),
            # per-sample: lam.flatten(1).max(dim=1).values -> [B]
            "lambda_max": torch.full((batch,), 0.75),
        }
    else:
        step["alm_stats"] = None
    return step


def _traces():
    """13 full chunks of 32 + one ragged chunk of 4 (exactly 420 samples)."""
    full = [[_step(32, i) for i in range(16)] for _ in range(13)]
    ragged = [[_step(4, i) for i in range(16)]]
    return full + ragged, 32


def test_ragged_final_chunk_does_not_crash_and_is_included():
    traces, chunk = _traces()
    series = aggregate_steps(traces, chunk)          # must not raise
    assert series["step_index"] == list(range(1, 17))
    # warm-up: the first 3 executed steps guide nothing
    assert series["active_n"][:3] == [0, 0, 0]
    # steps 4..16 guide 28/32 in each full chunk; the ragged chunk has all 4
    # samples outside the guided range (`guided_from=4` on a 4-sample chunk)
    expected = 13 * 28 + 0
    assert series["active_n"][3] == expected
    assert series["active_n"][-1] == expected
    # every series covers EVERY active sample -- including the ragged chunk when
    # it is active (see the next test); here it contributes 0, not an error
    for name in ("raw_violation", "safe_violation", "correction",
                 "smooth_raw", "smooth_safe", "inner_steps"):
        assert series[name]["n"][3] == expected, name


def test_ragged_chunk_is_counted_when_it_is_also_guided():
    """A ragged chunk whose samples ARE guided must contribute to the curves."""
    traces, chunk = _traces()
    for step in traces[-1]:                            # guide the last 4 samples
        step["guided"] = torch.ones(4, dtype=torch.bool)
    series = aggregate_steps(traces, chunk)
    assert series["active_n"][3] == 13 * 28 + 4
    assert series["raw_violation"]["n"][3] == 13 * 28 + 4
    assert series["lambda_max"]["n"][3] == 13 * 28 + 4


def test_lambda_max_is_per_sample():
    """``lambda_max`` is ``lam.flatten(1).max(1)`` -> one value PER SAMPLE.

    (The ragged chunk's 4 values were what the old 32-long mask indexed, which
    is the crash this file exists to prevent.)
    """
    traces, chunk = _traces()
    series = aggregate_steps(traces, chunk)
    assert series["lambda_max"]["mean"][0] is None      # warm-up
    assert series["lambda_max"]["mean"][3] == pytest.approx(0.75)
    assert series["lambda_max"]["n"][3] == 13 * 28      # one per active sample


def test_step_batch_falls_back_to_any_present_key():
    assert _step_batch({"q0_raw": torch.zeros(7, 48, 2)}) == 7
    assert _step_batch({"guided": torch.zeros(3, dtype=torch.bool)}) == 3
    assert _step_batch({}) is None


def test_sdf_only_traces_use_the_guard_series():
    """The four external baselines have no alm_stats: the guard keys must win."""
    traces = [[_step(8, i, alm=False) for i in range(16)]]
    series = aggregate_steps(traces, 8)
    assert series["raw_violation"]["mean"][3] == pytest.approx(0.5)       # from guard_before
    assert series["safe_violation"]["mean"][3] == pytest.approx(0.1)      # from guard_after
    assert series["correction"]["mean"][3] == pytest.approx(0.25)
    assert series["lambda_max"]["mean"][3] is None
    # 0.1 m > FEASIBLE_EPS, so nothing counts as feasible here
    assert FEASIBLE_EPS < 0.1
    assert series["feasible_rate"]["mean"][3] == 0.0


def test_corridor_traces_prefer_the_applied_alm_values():
    """When both are present the ALM stats (the applied correction) are used."""
    traces, chunk = _traces()
    series = aggregate_steps(traces, chunk)
    # alm_stats.max_violation_after = 0.1, guard_violation_after = 0.1 -> equal
    assert series["safe_violation"]["mean"][3] == pytest.approx(0.1)
    # correction: alm_correction (0.25) is preferred over guard_correction
    assert series["correction"]["mean"][3] == pytest.approx(0.25)
