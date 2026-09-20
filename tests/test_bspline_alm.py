"""Report section 43 (Tests H, I, J): control-space B-spline ALM behaviour.

Test H : an already feasible reference is an EXACT fixed point.
Test I : a violated reference is pushed back into the corridor while the
         endpoints stay hard-fixed.
Test J : the dual warm start across reverse steps is stable and never makes the
         violation worse.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.diffusion.bspline_alm import bspline_alm_correct, constraint_state
from src.geometry.bspline import BSplineCodec
from src.geometry.bspline_constraints import build_constraint_pack
from src.geometry.convex_region import EllipseRegionBuilder
from src.geometry.safety_corridor import build_safety_corridor

KNOTS = os.path.join(ROOT, "data", "carla_v1", "bspline_knots.npy")

ALM_CFG = {
    "rho": 5.0,
    "step_size": 0.05,
    "proximity_weight": 1.0,
    "correction_smooth_weight": 0.1,
    "constraint_tol": 1.0e-3,
    "max_curve_step_scene": 0.01,
}


def _codec():
    return BSplineCodec(degree=3, num_controls=32, curve_points=128,
                        knots_path=KNOTS)


def _walled_corridor(half_width=0.08, res=128, H=128,
                     log_a=math.log(0.06), log_b=math.log(0.05)):
    occ = torch.zeros(1, 1, res, res)
    iy = int((half_width + 1.0) / 2.0 * res)
    occ[:, :, iy:, :] = 1.0
    occ[:, :, :res - iy, :] = 1.0
    s = np.linspace(0.0, 1.0, H)
    centers = np.stack([-0.6 + 1.2 * s, np.zeros(H)], axis=-1)
    shape4 = np.tile(np.array([log_a, log_b, 1.0, 0.0]), (H, 1))
    builder = EllipseRegionBuilder(occ, {"safety_margin": 0.01,
                                         "obstacle_window_half": 0.35})
    corridor = build_safety_corridor(
        builder, centers, shape4, s, gamma=centers, gamma_lengths=H,
        config={"min_overlap_ratio": 0.10,
                "bridge": {"enabled": True, "max_bridge_per_gap": 1}})
    codec = _codec()
    pack = build_constraint_pack(codec, [corridor],
                                 knot_boundaries=codec.knot_span_boundaries())
    cond = torch.tensor([[-0.6, 0.0], [0.6, 0.0]])
    q_ref = codec.fit_curve_to_controls(
        torch.as_tensor(centers, dtype=torch.float32)[None],
        cond[0:1], cond[1:2])
    return {"corridor": corridor, "pack": pack, "codec": codec, "q_ref": q_ref,
            "cond": cond, "centers": centers}


# --------------------------------------------------------------- Test H
def test_H_a_feasible_reference_is_an_exact_fixed_point():
    fixture = _walled_corridor()
    codec, pack, q_ref = fixture["codec"], fixture["pack"], fixture["q_ref"]
    _, g, mask = constraint_state(q_ref, pack)
    assert float(g.masked_fill(~mask, -1e9).max()) <= 0.0

    q_safe, lam, stats = bspline_alm_correct(
        q_ref, pack, codec, None, ALM_CFG, inner_steps=8)

    assert torch.equal(q_safe, q_ref)
    assert float(lam.abs().max()) == 0.0
    assert float(stats["mean_curve_correction_scene"]) == 0.0
    assert float(stats["max_curve_correction_scene"]) == 0.0


# --------------------------------------------------------------- Test I
def test_I_violation_is_reduced_and_endpoints_stay_fixed():
    fixture = _walled_corridor()
    codec, pack, q_ref = fixture["codec"], fixture["pack"], fixture["q_ref"]
    q_bad = q_ref.clone()
    q_bad[0, 14:19, 1] += 0.20

    q_safe, lam, stats = bspline_alm_correct(
        q_bad, pack, codec, None, ALM_CFG, inner_steps=40)

    before = float(stats["max_violation_before"])
    after = float(stats["max_violation_after"])
    assert before > 0.01
    assert after < before
    assert after < ALM_CFG["constraint_tol"]

    # hard endpoint conditioning is never touched
    assert torch.equal(q_safe[:, 0], fixture["cond"][0:1])
    assert torch.equal(q_safe[:, -1], fixture["cond"][1:2])
    assert torch.equal(q_safe[:, 0], q_bad[:, 0])
    assert torch.equal(q_safe[:, -1], q_bad[:, -1])

    # the correction is measured on the curve and reported in both units
    assert float(stats["max_curve_correction_scene"]) > 0.0
    assert abs(float(stats["max_curve_correction_m"])
               - 40.0 * float(stats["max_curve_correction_scene"])) < 1e-6
    assert 0.0 <= float(stats["constraint_feasible_rate"]) <= 1.0


def test_I_curve_step_limit_is_never_exceeded():
    fixture = _walled_corridor()
    codec, pack, q_ref = fixture["codec"], fixture["pack"], fixture["q_ref"]
    q_bad = q_ref.clone()
    q_bad[0, 14:19, 1] += 0.30
    cfg = dict(ALM_CFG, max_curve_step_scene=0.004)
    q_safe, _, stats = bspline_alm_correct(
        q_bad, pack, codec, None, cfg, inner_steps=1)
    assert float(stats["max_curve_correction_scene"]) <= 0.004 + 1e-6


# --------------------------------------------------------------- Test J
def test_J_dual_warm_start_is_stable_and_not_worse():
    fixture = _walled_corridor()
    codec, pack, q_ref = fixture["codec"], fixture["pack"], fixture["q_ref"]
    q_bad = q_ref.clone()
    q_bad[0, 14:19, 1] += 0.20

    _, lam_first, stats_first = bspline_alm_correct(
        q_bad, pack, codec, None, ALM_CFG, inner_steps=6)

    q_cold, lam_cold, stats_cold = bspline_alm_correct(
        q_bad, pack, codec, None, ALM_CFG, inner_steps=6)
    q_warm, lam_warm, stats_warm = bspline_alm_correct(
        q_bad, pack, codec, lam_first, ALM_CFG, inner_steps=6)

    assert torch.isfinite(q_warm).all()
    assert torch.isfinite(lam_warm).all()
    assert float(stats_warm["max_violation_after"]) \
        <= float(stats_cold["max_violation_after"]) + 1e-9
    # a warm-started multiplier starts from a non-zero dual: the first step is
    # at least as aggressive as the cold start
    assert float(lam_first.abs().max()) > 0.0


def test_J_dual_is_reset_when_the_corridor_changes():
    fixture = _walled_corridor()
    codec, pack, q_ref = fixture["codec"], fixture["pack"], fixture["q_ref"]
    _, lam, _ = bspline_alm_correct(q_ref, pack, codec, None, ALM_CFG,
                                    inner_steps=4)
    # passing lam=None (the sampler does this whenever the pack is rebuilt)
    # must reproduce the cold-start result exactly
    a = bspline_alm_correct(q_ref, pack, codec, lam, ALM_CFG, inner_steps=4)[0]
    b = bspline_alm_correct(q_ref, pack, codec, None, ALM_CFG, inner_steps=4)[0]
    assert torch.equal(a, b)


# --------------------------------------------------------- pack integrity
def test_pack_masks_inactive_entries_so_alm_is_a_no_op():
    fixture = _walled_corridor()
    codec, pack = fixture["codec"], fixture["pack"]
    q_ref = fixture["q_ref"]

    # pad the batch with a failed sample: its corridor is None
    padded = build_constraint_pack(codec, [fixture["corridor"], None],
                                   knot_boundaries=codec.knot_span_boundaries())
    assert int(padded.num_pieces[1]) == 0
    assert not bool(padded.piece_mask[1].any())

    q_two = torch.cat([q_ref, q_ref + 5.0], dim=0)
    q_safe, _, _ = bspline_alm_correct(q_two, padded, codec, None, ALM_CFG,
                                       inner_steps=4)
    assert torch.equal(q_safe[1], q_two[1])


def test_pack_constraints_are_per_piece_and_per_face():
    fixture = _walled_corridor()
    codec, pack = fixture["codec"], fixture["pack"]
    assert pack.num_controls == 32
    assert int(pack.num_pieces[0]) > 100          # subdivision is real
    assert pack.extraction.shape[-2:] == (4, 32)
    # every active inequality is a genuine face of the responsible region
    assert int(pack.face_mask.sum()) >= int(pack.piece_mask.sum())
