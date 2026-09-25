"""Contract tests for the four external SDF constraint processors (§4.2).

The properties that matter for the paper's main table:

1. every processor must actually REDUCE the SDF violation of a colliding
   reference (otherwise the "safety benefit" column is meaningless);
2. endpoints stay pinned (the start/goal condition is a hard constraint for all
   of them, in every inner iteration);
3. rows that are not ``active`` come back BIT-IDENTICAL (warm-up steps and
   samples without a candidate must not be touched);
4. the Projected method reports ``projection_failed`` instead of inventing a
   feasible answer when the LP/QP has no solution;
5. non-finite iterates never leak: they are reverted and flagged.
"""

from __future__ import annotations

import pytest
import torch

from src.diffusion.sdf_guidance import (GUIDANCE_MODES, apply_sdf_guidance,
                                        curve_roughness, sdf_alm_correct,
                                        sdf_penalty_correct,
                                        sdf_primal_dual_correct,
                                        sdf_projection_correct)
from src.geometry.bspline import BSplineCodec
from src.geometry.sdf_field import build_occupancy_sdf

RES = 256
CELL = 2.0 / RES
C = 48


def _codec():
    return BSplineCodec(degree=3, num_controls=C, curve_points=128, knots="auto")


def _scene(row, col):
    return (-1.0 + (col + 0.5) * CELL, -1.0 + (row + 0.5) * CELL)


def _colliding_setup():
    """A straight control polygon that runs through a blob, plus a free one."""
    occ = torch.zeros(2, 1, RES, RES)
    occ[0, 0, 110:150, 110:150] = 1.0                 # blob in front of sample 0
    sdf = build_occupancy_sdf(occ, scene_to_meter=80.0, map_margin=0.0)
    start = torch.tensor([[-0.9, 0.01], [-0.9, -0.9]])
    goal = torch.tensor([[0.9, 0.01], [0.9, -0.9]])
    t = torch.linspace(0.0, 1.0, C)[None, :, None]
    q_ref = start[:, None, :] * (1 - t) + goal[:, None, :] * t
    cond = torch.stack([start, goal], dim=1)          # [B,2,2]
    return sdf, q_ref, cond


def _max_violation(codec, sdf, q):
    p = codec.decode_controls(q)
    value = sdf.query(p, with_boundary=True, with_grad=False)
    return (0.0 - value).max(dim=1).values


def test_reference_really_collides():
    """Guard the fixture itself: sample 0 violates, sample 1 does not."""
    sdf, q_ref, cond = _colliding_setup()
    codec = _codec()
    violation = _max_violation(codec, sdf, q_ref)
    assert float(violation[0]) > 0.05                 # > 4 m into the blob
    assert float(violation[1]) < 0.0                  # the free straight line


@pytest.mark.parametrize("mode", ["penalty", "projection", "primal_dual", "alm_sdf"])
def test_every_processor_reduces_the_violation(mode):
    sdf, q_ref, cond = _colliding_setup()
    codec = _codec()
    cfg = {"inner_steps": 10, "step_size": 0.05, "rho": 5.0, "dual_step": 1.0,
           "penalty_weight": 5.0, "scene_to_meter": 80.0}
    out = apply_sdf_guidance(mode, q_ref, sdf, codec, cond, lam=None, config=cfg)
    q_safe, _, stats = out
    before = _max_violation(codec, sdf, q_ref)
    after = _max_violation(codec, sdf, q_safe)
    assert float(after[0]) < float(before[0]), (mode, before.tolist(), after.tolist())
    assert float(stats["violation_after"][0]) <= float(stats["violation_before"][0]) + 1e-9
    assert float(stats["correction_m"][0]) > 0.0
    # the reference polygon moved, the free one did not get worse
    assert not torch.allclose(q_safe[0], q_ref[0])
    assert float(after[1]) <= float(before[1]) + 1e-6


@pytest.mark.parametrize("mode", ["penalty", "projection", "primal_dual", "alm_sdf"])
def test_endpoints_are_pinned_and_inactive_rows_untouched(mode):
    sdf, q_ref, cond = _colliding_setup()
    codec = _codec()
    active = torch.tensor([True, False])
    cfg = {"inner_steps": 5, "scene_to_meter": 80.0}
    out = apply_sdf_guidance(mode, q_ref, sdf, codec, cond, lam=None, config=cfg,
                             active=active)
    q_safe = out[0]
    for b in range(2):
        assert torch.equal(q_safe[b, 0], cond[b, 0])
        assert torch.equal(q_safe[b, -1], cond[b, 1])
    assert torch.equal(q_safe[1], q_ref[1])           # untouched row
    assert stats_status(out[2], 1) == "skipped"


def stats_status(stats, index):
    return stats["status"][index]


def test_projection_reports_failure_instead_of_faking_feasibility():
    """Endpoints INSIDE an obstacle make the projection infeasible."""
    occ = torch.zeros(1, 1, RES, RES)
    occ[0, 0, 60:200, 60:200] = 1.0                   # start/goal are inside
    sdf = build_occupancy_sdf(occ, scene_to_meter=80.0)
    codec = _codec()
    start = torch.tensor([[0.0, 0.0]])
    goal = torch.tensor([[0.2, 0.2]])
    t = torch.linspace(0.0, 1.0, C)[None, :, None]
    q_ref = start[:, None, :] * (1 - t) + goal[:, None, :] * t
    cond = torch.stack([start, goal], dim=1)
    q_safe, _, stats = sdf_projection_correct(
        q_ref, sdf, codec, cond, {"inner_steps": 3, "scene_to_meter": 80.0})
    # no fake feasible answer: the raw polygon is kept and the failure is named
    assert torch.allclose(q_safe, q_ref)
    assert stats["status"][0].startswith("projection_failed")


def test_primal_dual_and_alm_raise_their_multipliers():
    sdf, q_ref, cond = _colliding_setup()
    codec = _codec()
    cfg = {"inner_steps": 5, "step_size": 0.05, "rho": 5.0, "dual_step": 1.0,
           "scene_to_meter": 80.0}
    _, lam_pd, stats_pd = sdf_primal_dual_correct(q_ref, sdf, codec, cond, None, cfg)
    _, lam_alm, stats_alm = sdf_alm_correct(q_ref, sdf, codec, cond, None, cfg)
    assert lam_pd.shape == (2, 128)
    assert float(lam_pd[0].max()) > 0.0
    assert float(lam_alm[0].max()) > 0.0
    assert float(stats_pd["lambda_max"][0]) > 0.0
    assert float(stats_alm["lambda_max"][0]) > 0.0
    # the dual is only carried for the active row
    assert float(lam_pd[1].max()) == 0.0


def test_inherited_dual_is_reused_and_reset_on_demand():
    sdf, q_ref, cond = _colliding_setup()
    codec = _codec()
    cfg = {"inner_steps": 3, "rho": 5.0, "step_size": 0.05, "scene_to_meter": 80.0}
    _, lam, _ = sdf_alm_correct(q_ref, sdf, codec, cond, None, cfg)
    _, warm, _ = sdf_alm_correct(q_ref, sdf, codec, cond, lam, cfg)
    assert float(warm.max()) >= float(lam.max())      # warm start grows, not resets
    cfg_off = dict(cfg, inherit_dual_across_reverse_steps=False)
    _, cold, _ = sdf_alm_correct(q_ref, sdf, codec, cond, lam, cfg_off)
    assert float(cold.max()) <= float(lam.max()) + 1e-9


def test_nonfinite_iterates_are_reverted_and_flagged():
    """A hostile step size must produce a flagged no-op, never NaN controls."""
    sdf, q_ref, cond = _colliding_setup()
    codec = _codec()
    cfg = {"inner_steps": 2, "step_size": 1e9, "penalty_weight": 1e9,
           "scene_to_meter": 80.0}
    q_safe, _, stats = sdf_penalty_correct(q_ref, sdf, codec, cond, cfg)
    assert torch.isfinite(q_safe).all()
    assert torch.allclose(q_safe, q_ref)              # reverted
    # the guard fires either on non-finite iterates or on a worse result; both
    # are acceptable, silently keeping a diverged polygon is not
    assert stats["status"][0] in ("diverged", "reverted_worse")


def test_curve_roughness_matches_the_sampler_definition():
    straight = torch.zeros(1, 5, 2)
    assert float(curve_roughness(straight)[0]) == 0.0
    bendy = straight.clone()
    bendy[0, 2, 1] = 1.0
    assert float(curve_roughness(bendy)[0]) > 0.0
    assert curve_roughness(torch.zeros(1, 2, 2)).shape == (1,)


def test_unknown_mode_is_rejected():
    sdf, q_ref, cond = _colliding_setup()
    with pytest.raises(ValueError):
        apply_sdf_guidance("magic", q_ref, sdf, _codec(), cond)
    assert "penalty" in GUIDANCE_MODES and "alm" in GUIDANCE_MODES


def test_gradient_step_direction():
    """Pin the SIGN of the shared gradient helper (a silent sign flip turns
    every processor into gradient ascent, which the monotone guard then reverts
    to a no-op: the parameters stop mattering and the whole sweep grid returns
    identical numbers -- measured, and exactly what this test prevents).

    ``_grad_q(grad_p, relu(g))`` must INCREASE ``d_occ``: adding it reduces the
    violation, subtracting it does not.
    """
    from src.diffusion.sdf_guidance import _constraint_rows, _grad_q, _violation_of

    sdf, q_ref, cond = _colliding_setup()
    codec = _codec()
    base, _, _ = _violation_of(codec, q_ref, sdf, 0.0)
    assert float(base[0]) > 0.05                       # sample 0 really violates
    _, _, grad_p, g = _constraint_rows(codec, q_ref, sdf, 0.0)
    step = _grad_q(codec, grad_p, torch.relu(g))
    plus, _, _ = _violation_of(codec, q_ref + 0.05 * step, sdf, 0.0)
    minus, _, _ = _violation_of(codec, q_ref - 0.05 * step, sdf, 0.0)
    assert float(plus[0]) < float(base[0]), (plus.tolist(), base.tolist())
    assert float(minus[0]) > float(base[0]), (minus.tolist(), base.tolist())
