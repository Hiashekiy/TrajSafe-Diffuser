"""Historical safety feedback (上一轮 ALM 的安全修正作为下一步的条件).

Contract pinned down here:

  * ``FeedbackEncoder`` / ``FeedbackFusion`` shapes and the gate semantics:
    ``valid = 0`` is the EXACT identity, ``valid = 1`` moves the features;
  * a zero-initialised model is bit-identical with and without feedback, so an
    old checkpoint keeps its exact behaviour and the sampler's warm-up phase is
    untouched;
  * ``build_constraint_pack_from_regions`` turns the OFFLINE ALM cache
    (``alm_cell_a/b/valid``) into the very same exact Bezier constraint pack the
    inference ALM projects onto, including the per-cell mask semantics;
  * ``feedback_step`` implements design-note section 4 (already-safe / corrected
    / failed-never-overwrites);
  * ``L_fbsafe`` is zero inside the corridor and grows outside it;
    ``L_curve`` is zero for a straight decoded curve and grows when it wiggles.
"""
from __future__ import annotations

import numpy as np
import torch

from helpers import tiny_batch, tiny_model

from src.diffusion.bspline_alm import constraint_state
from src.diffusion.sampler import feedback_step
from src.geometry.bspline import BSplineCodec
from src.geometry.bspline_constraints import (build_constraint_pack_from_regions,
                                              build_constraint_pack)
from src.losses.losses import (curve_smoothness_loss, feedback_safety_loss,
                               pack_max_violation, pack_violation)
from src.models.trajsafe.feedback import FeedbackEncoder, FeedbackFusion


# --------------------------------------------------------------- primitives
def test_feedback_fusion_is_an_exact_identity_when_invalid():
    fused = FeedbackFusion(8, gate_bias=1.0)          # a NON-zero gate bias
    h = torch.randn(2, 5, 8)
    f = torch.randn(2, 5, 8)
    assert torch.equal(fused(h, f, torch.zeros(2)), h)
    mixed = fused(h, f, torch.tensor([1.0, 0.0]))
    assert torch.equal(mixed[1], h[1])                # the invalid row is frozen
    assert not torch.allclose(mixed[0], h[0])         # the valid row moves
    # [B,1] and [B,C,1] validity are both accepted
    assert torch.equal(fused(h, f, torch.zeros(2, 1)), h)
    assert torch.equal(fused(h, f, torch.zeros(2, 5, 1)), h)


def test_feedback_encoder_zero_init_is_exactly_zero():
    enc = FeedbackEncoder(8, hidden=4, zero_init=True)
    assert torch.equal(enc(torch.randn(2, 5, 5)), torch.zeros(2, 5, 8))
    enc2 = FeedbackEncoder(8, hidden=4, zero_init=False)
    assert not torch.allclose(enc2(torch.randn(2, 5, 5)), torch.zeros(2, 5, 8))


def _forward(model, b, select_index=None, **kwargs):
    """``select_index`` defaults to the cached expert m* (as in training)."""
    if select_index is None:
        select_index = b["topology_best"]
    return model.forward_all(
        b["pos"], b["occ"], b["cond"], b["t"], b["ab"],
        b["candidate_xy"], b["candidate_mask"], b["candidate_geometry"],
        b["candidate_geometry_lengths"], select_index=select_index, **kwargs)


def test_zero_init_model_is_bit_identical_with_and_without_feedback():
    b = tiny_batch(B=2, H=8, M=3, L=8)
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8, d_model=32,
                       feedback={"enabled": True, "hidden": 16})
    assert model.feedback_enabled
    model.eval()
    with torch.no_grad():
        base = _forward(model, b)
        fed = _forward(model, b,
                       feedback_control=torch.randn(2, 8, 2),
                       feedback_delta=torch.randn(2, 8, 2),
                       feedback_valid=torch.ones(2, dtype=torch.bool))
    assert torch.equal(base["control"], fed["control"])
    assert fed["H_fb"].shape == (2, 8, 32)
    assert torch.equal(fed["H_fb"], torch.zeros(2, 8, 32))
    assert fed["feedback_valid"].shape == (2,)


def test_trained_feedback_condition_changes_the_prediction_and_gets_gradient():
    b = tiny_batch(B=2, H=8, M=3, L=8)
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8, d_model=32,
                       feedback={"enabled": True, "hidden": 16,
                                 "zero_init": False})
    model.train()
    fb = dict(feedback_control=torch.randn(2, 8, 2),
              feedback_delta=0.1 * torch.randn(2, 8, 2),
              feedback_valid=torch.ones(2, dtype=torch.bool))
    out = _forward(model, b, **fb)
    out["control"].pow(2).mean().backward()
    grads = [p.grad for p in model.feedback_encoder.parameters()
             if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0

    model.eval()
    with torch.no_grad():
        without = _forward(model, b)
        with_fb = _forward(model, b, **fb)
    assert not torch.allclose(without["control"], with_fb["control"])


def test_feedback_shape_validation():
    b = tiny_batch(B=2, H=8, M=3, L=8)
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8, d_model=32,
                       feedback={"enabled": True, "hidden": 16})
    try:
        _forward(model, b, feedback_control=torch.zeros(2, 5, 2),
                 feedback_valid=torch.ones(2, dtype=torch.bool))
    except ValueError as exc:
        assert "feedback_control" in str(exc)
    else:                                                   # pragma: no cover
        raise AssertionError("a wrong control shape must be refused")


# ------------------------------------------------- constraint pack (offline)
def _tube_regions(half=0.10, cells=4, span=0.6, padded_face=False):
    """Region table of a straight tube around ``y = 0`` (the offline layout).

    ``A [R,F,2]`` unit normals, ``b [R,F]``, ``mask [R]`` per CELL, exactly the
    ``alm_cell_a`` / ``alm_cell_b`` / ``alm_cell_valid`` contract.  Padded faces
    carry ``A = 0`` / ``b = +inf``.  Every cell is the box
    ``|x| <= span, |y| <= half`` so the piece->region assignment cannot change
    the constraint set (only the machinery is exercised).
    """
    faces = 4 + (1 if padded_face else 0)
    A = np.zeros((cells, faces, 2), np.float64)
    b = np.zeros((cells, faces), np.float64)
    mask = np.ones(cells, bool)
    for i in range(cells):
        A[i, :4] = [[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]]
        b[i, :4] = [half, half, span, span]
    if padded_face:
        b[:, 4] = np.inf
    anchors = np.linspace(0.0, 1.0, cells)
    return (torch.as_tensor(A, dtype=torch.float32),
            torch.as_tensor(b, dtype=torch.float32),
            torch.as_tensor(mask), torch.as_tensor(anchors,
                                                   dtype=torch.float32))


def _straight_controls(codec, start=(-0.5, 0.0), goal=(0.5, 0.0), offset=(0.0, 0.0)):
    u = torch.linspace(0.0, 1.0, codec.num_controls)[:, None]
    q = (torch.tensor(start)[None] * (1.0 - u)
         + torch.tensor(goal)[None] * u)
    return q[None] + torch.tensor(offset)[None, None]


def test_offline_region_table_becomes_the_exact_inference_pack():
    codec = BSplineCodec(degree=3, num_controls=8, curve_points=32,
                         knots_path="auto", endpoint_constrained=False)
    A, b, mask, anchors = _tube_regions(half=0.10, padded_face=True)
    pack = build_constraint_pack_from_regions(
        codec, A[None], b[None], mask[None], anchors=anchors)
    assert pack.num_pieces[0] > 0
    assert int(pack.num_controls) == 8

    inside = _straight_controls(codec)
    outside = _straight_controls(codec, offset=(0.0, 0.15))
    assert float(pack_max_violation(inside, pack)) == 0.0
    assert abs(float(pack_max_violation(outside, pack)) - 0.05) < 1e-4

    # IDENTICAL to the ALM's own constraint view
    _, g, m = constraint_state(outside, pack)
    assert torch.allclose((torch.relu(g) * m).flatten(1).amax(dim=1),
                          pack_max_violation(outside, pack), atol=1e-6)

    # the padded face can never bind
    finite = pack.piece_b[pack.face_mask]
    assert bool(torch.isfinite(finite).all())


def test_invalid_corridor_row_is_fully_masked_and_masked_out_of_the_loss():
    codec = BSplineCodec(degree=3, num_controls=8, curve_points=32,
                         knots_path="auto", endpoint_constrained=False)
    A, b, mask, anchors = _tube_regions(half=0.10)
    A = torch.stack([A, A], dim=0)          # [B=2, R, F, 2]
    b = torch.stack([b, b], dim=0)
    mask = torch.stack([mask, torch.zeros_like(mask)])
    pack = build_constraint_pack_from_regions(
        codec, A, b, mask, anchors=anchors,
        sample_valid=torch.tensor([True, True]))
    assert int(pack.num_pieces[0]) > 0
    assert int(pack.num_pieces[1]) == 0          # no valid cell -> no constraint
    q = torch.cat([_straight_controls(codec),
                   _straight_controls(codec, offset=(0.0, 0.5))])
    assert float(pack_max_violation(q, pack)[1]) == 0.0
    loss = feedback_safety_loss(q, pack, sample_mask=pack.num_pieces > 0)
    assert float(loss) == 0.0                    # the only violating row is masked


def test_pack_matches_the_corridor_builder_layout():
    """``region_table``-style faces and the offline layout produce one pack."""
    codec = BSplineCodec(degree=3, num_controls=8, curve_points=32,
                         knots_path="auto", endpoint_constrained=False)
    A, b, mask, anchors = _tube_regions(half=0.10)

    class _Cell:
        def __init__(self, a, bb):
            self.A, self.b = a, bb

    class _Corridor:
        valid = True
        bridge_cell_count = 0
        overlap_ratio = None
        failure_reason = None

        def __init__(self):
            self.cells = [_Cell(A[i].numpy().astype(np.float64),
                                b[i].numpy().astype(np.float64))
                          for i in range(A.shape[0])]

        @property
        def num_cells(self):
            return len(self.cells)

        def anchors(self):
            return anchors.numpy()

    offline = build_constraint_pack_from_regions(codec, A[None], b[None],
                                                 mask[None], anchors=anchors)
    corridor = build_constraint_pack(codec, [_Corridor()])
    assert int(offline.num_pieces[0]) == int(corridor.num_pieces[0])
    assert torch.allclose(offline.extraction, corridor.extraction, atol=1e-6)
    assert torch.allclose(offline.piece_A, corridor.piece_A, atol=1e-6)
    assert torch.allclose(offline.piece_b, corridor.piece_b, atol=1e-5)
    assert torch.equal(offline.face_mask, corridor.face_mask)


# --------------------------------------------------------- feedback state
def test_feedback_step_implements_section_4():
    raw = torch.zeros(1, 3, 2)
    safe = torch.ones(1, 3, 2)
    zero = torch.zeros(1, 3, 2)
    empty_v = torch.zeros(1, dtype=torch.bool)
    guided = torch.ones(1, dtype=torch.bool)
    has_pack = torch.ones(1, dtype=torch.bool)

    # (1) warm-up: not guided -> no feedback at all
    c, d, v, st = feedback_step(zero, zero, empty_v,
                                torch.zeros(1, dtype=torch.bool),
                                has_pack, torch.zeros(1), torch.zeros(1),
                                raw, safe, 1e-3)
    assert not bool(v[0]) and st["accepted"] == 0

    # (2) the ALM had to correct a violation -> Q0_safe / Delta
    c, d, v, st = feedback_step(c, d, v, guided, has_pack,
                                torch.tensor([0.2]), torch.tensor([0.0]),
                                raw, safe, 1e-3)
    assert bool(v[0]) and st["accepted"] == 1 and st["already_safe"] == 0
    assert torch.equal(c, safe) and torch.equal(d, safe - raw)

    # (3) the raw prediction was already safe -> delta = 0, no correction
    c2, d2, v2, st = feedback_step(zero, zero, empty_v, guided, has_pack,
                                   torch.tensor([0.0]), torch.tensor([0.0]),
                                   raw, raw, 1e-3)
    assert bool(v2[0]) and st["already_safe"] == 1
    assert torch.equal(c2, raw) and torch.equal(d2, torch.zeros_like(d2))

    # (4) a FAILING ALM must never overwrite the last verified feedback
    c3, d3, v3, st = feedback_step(
        c, d, v, guided, has_pack, torch.tensor([0.2]), torch.tensor([0.5]),
        raw, torch.full_like(safe, 2.0), 1e-3)
    assert bool(v3[0]) and st["rejected"] == 1
    assert torch.equal(c3, c) and torch.equal(d3, d)

    # (5) no corridor at all -> nothing is trusted
    c4, d4, v4, _ = feedback_step(empty_v.new_zeros(1, 3, 2),
                                  empty_v.new_zeros(1, 3, 2), empty_v, guided,
                                  torch.zeros(1, dtype=torch.bool),
                                  torch.zeros(1), torch.zeros(1), raw, safe,
                                  1e-3)
    assert not bool(v4[0])


# ------------------------------------------------------------------- losses
def test_feedback_safety_loss_zero_inside_and_gradient_outside():
    codec = BSplineCodec(degree=3, num_controls=8, curve_points=32,
                         knots_path="auto", endpoint_constrained=False)
    A, b, mask, anchors = _tube_regions(half=0.10)
    pack = build_constraint_pack_from_regions(codec, A[None], b[None],
                                              mask[None], anchors=anchors)
    inside = _straight_controls(codec).requires_grad_(True)
    inside_loss = feedback_safety_loss(inside, pack)
    assert float(inside_loss.detach()) == 0.0
    inside_loss.backward()
    # relu() at 0 has zero derivative: a feasible polygon gets no gradient
    assert torch.equal(inside.grad, torch.zeros_like(inside.grad))

    outside = _straight_controls(codec, offset=(0.0, 0.2)).requires_grad_(True)
    loss = feedback_safety_loss(outside, pack)
    assert float(loss.detach()) > 0.1
    loss.backward()
    assert bool(torch.isfinite(outside.grad).all())
    assert float(outside.grad.abs().sum()) > 0.0


def test_curve_smoothness_loss_straight_is_zero_and_wiggle_is_penalised():
    codec = BSplineCodec(degree=3, num_controls=8, curve_points=64,
                         knots_path="auto", endpoint_constrained=False)
    straight = _straight_controls(codec)
    floor = float(curve_smoothness_loss(straight, straight, codec.basis))
    # a straight clamped-uniform B-spline is not an exactly affine
    # parameterisation, so the floor is small but not exactly 0
    assert floor < 0.05
    kink = torch.sign(torch.sin(torch.arange(codec.num_controls).float()
                                * np.pi))[None, :, None]
    wiggle = straight + 0.02 * kink * torch.tensor([0.0, 1.0])
    assert float(curve_smoothness_loss(wiggle, straight,
                                       codec.basis)) > 1.2 * floor
    # a straight translation of an already smooth curve stays free
    shifted = straight + torch.tensor([0.0, 0.3])
    assert abs(float(curve_smoothness_loss(shifted, straight, codec.basis))
               - floor) < 1e-6


# ------------------------------------------------- the training-time rollout
def _scene_cells(batch, half=0.03, cells=4):
    """A narrow tube around each sample's own start->goal line (scene units)."""
    cond = batch["cond"]
    B = cond.shape[0]
    start, goal = cond[:, 0], cond[:, 1]
    d = goal - start
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    n = torch.stack([-d[:, 1], d[:, 0]], dim=-1)
    A = torch.stack([n, -n, d, -d], dim=1)                      # [B,4,2]
    b = torch.stack([(n * start).sum(-1) + half, (-n * start).sum(-1) + half,
                     (d * goal).sum(-1) + 0.5,
                     (-d * start).sum(-1) + 0.5], dim=1)        # [B,4]
    A = A[:, None].expand(B, cells, 4, 2).contiguous()
    b = b[:, None].expand(B, cells, 4).contiguous()
    mask = torch.ones(B, cells, dtype=torch.bool)
    return A, b, mask


def _rollout_batch(batch, model, alm_a, alm_b, alm_mask, alm_valid):
    batch = dict(batch)
    batch["control_gt"] = model.traj_to_control(batch["pos"], batch["cond"])
    # tiny_batch uses the short alias of the occupancy tensor
    batch["occupancy"] = batch["occ"]
    batch["alm_cell_a"] = alm_a
    batch["alm_cell_b"] = alm_b
    batch["alm_cell_valid"] = alm_mask
    batch["alm_valid"] = alm_valid
    return batch


def test_training_two_step_rollout_runs_backprops_and_fills_the_second_step():
    from train import LOSS_KEYS, batch_losses

    from src.diffusion.schedule import NoiseSchedule

    b = tiny_batch(B=2, H=8, M=3, L=8)
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8, d_model=32,
                       feedback={"enabled": True, "hidden": 16,
                                 "zero_init": False})
    model.train()
    A, bb, mask = _scene_cells(b)
    batch = _rollout_batch(b, model, A, bb, mask,
                           torch.ones(2, dtype=torch.bool))
    schedule = NoiseSchedule(8)
    lcfg = {"ellipse_safe_res": 16}
    alm_cfg = {"inner_steps": 2, "max_curve_step_scene": 0.05,
               "constraint_tol": 1e-3}
    raw, weights, total, out, stats = batch_losses(
        batch, model, schedule, lcfg, "cpu", alm_cfg=alm_cfg,
        fb_cfg={"rollout": True, "drop_prob": 0.0})
    assert set(LOSS_KEYS) <= set(raw)
    assert len(stats["loss_parts"]) == 2          # step 1 and step 2 graphs
    assert torch.isfinite(total).all()
    for part in stats["loss_parts"]:
        (part / 2).backward()
    grads = [p.grad for p in model.feedback_encoder.parameters()
             if p.grad is not None]
    assert grads, "the rollout never reached the feedback encoder"

    # a corridor that contains everything -> the ALM is a no-op and the network
    # is told "you were already safe" (delta = 0), i.e. feedback_valid = 1
    wide_A = torch.zeros(2, 4, 4, 2)
    wide_A[..., 0, 1] = 1.0
    wide_A[..., 1, 1] = -1.0
    wide_A[..., 2, 0] = 1.0
    wide_A[..., 3, 0] = -1.0
    wide_b = torch.full((2, 4, 4), 1.5)
    wide_mask = torch.ones(2, 4, dtype=torch.bool)
    batch2 = _rollout_batch(b, model, wide_A, wide_b, wide_mask,
                            torch.ones(2, dtype=torch.bool))
    raw2, _, total2, _, stats2 = batch_losses(
        batch2, model, schedule, lcfg, "cpu", alm_cfg=alm_cfg,
        fb_cfg={"rollout": True, "drop_prob": 0.0})
    assert stats2["fb_valid_rate"] == 1.0
    assert stats2["fb_delta_norm"] == 0.0         # already safe -> no correction
    assert float(raw2["Lfbsafe"].detach()) == 0.0
    assert torch.isfinite(total2).all()

    # warm-up simulation: the whole batch is forced back to "no history"
    raw3, _, _, _, stats3 = batch_losses(
        batch2, model, schedule, lcfg, "cpu", alm_cfg=alm_cfg,
        fb_cfg={"rollout": True, "simulate_warmup": True})
    assert stats3["fb_valid_rate"] == 0.0
    assert torch.isfinite(raw3["Lfbsafe"]).all()


def test_rollout_timesteps_never_use_the_last_step_for_the_two_step_rollout():
    from train import rollout_pair, rollout_timesteps

    t = rollout_timesteps(512, 16, True, device="cpu")
    assert int(t.min()) >= 1 and int(t.max()) <= 15
    every = rollout_timesteps(512, 16, False, device="cpu")
    assert int(every.min()) >= 0 and int(every.max()) <= 15
    tt, ss = rollout_pair(torch.tensor([0, 1, 15]), 16)
    assert tt.tolist() == [0, 1, 15]
    assert ss.tolist() == [0, 0, 14]              # the real next reverse step


# ------------------------------------------------------------ empty pack
def _empty_pack(codec, B=1):
    """A pack whose corridors ALL failed: no piece, no face at all."""
    A, b, mask, anchors = _tube_regions(half=0.10)
    A = A[None].expand(B, *A.shape).contiguous()
    b = b[None].expand(B, *b.shape).contiguous()
    mask = mask[None].expand(B, mask.shape[0]).contiguous()
    return build_constraint_pack_from_regions(
        codec, A, b, mask, anchors=anchors,
        sample_valid=torch.zeros(B, dtype=torch.bool))


def test_empty_pack_is_a_differentiable_zero_and_never_raises():
    codec = BSplineCodec(degree=3, num_controls=8, curve_points=32,
                         knots_path="auto", endpoint_constrained=False)
    pack = _empty_pack(codec, B=1)                 # batch_size = 1
    assert int(pack.num_pieces[0]) == 0
    q = _straight_controls(codec, offset=(0.0, 0.5)).requires_grad_(True)
    assert float(pack_max_violation(q, pack).detach()) == 0.0
    assert float(pack_violation(q, pack, reduction="mean").detach()) == 0.0
    loss = feedback_safety_loss(q, pack)
    assert float(loss.detach()) == 0.0
    loss.backward()                                # graph-connected zero
    assert torch.equal(q.grad, torch.zeros_like(q.grad))
    # the same for a batch whose corridors all failed
    pack2 = _empty_pack(codec, B=3)
    q2 = _straight_controls(codec).expand(3, -1, -1).contiguous()
    assert pack_max_violation(q2, pack2).shape == (3,)
    assert float(pack_max_violation(q2, pack2).abs().max()) == 0.0


def test_select_index_minus_one_mixes_argmax_with_explicit_indices():
    """``-1`` = "argmax(pi) for THIS row" (the sampler's mixed-batch routing)."""
    b = tiny_batch(B=2, H=8, M=3, L=8)
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8, d_model=32)
    model.eval()
    args = (b["pos"], b["occ"], b["cond"], b["t"], b["ab"], b["candidate_xy"],
            b["candidate_mask"], b["candidate_geometry"],
            b["candidate_geometry_lengths"])
    with torch.no_grad():
        auto = model.forward_all(*args, select_index=None)    # argmax for all
        mixed = model.forward_all(*args, select_index=torch.tensor([-1, 0]))
        forced = model.forward_all(*args, select_index=torch.tensor([0, 0]))
    assert torch.equal(mixed["selected_idx"][0], auto["selected_idx"][0])
    assert int(mixed["selected_idx"][1]) == 0
    # the RESOLVED index really drives the forward pass
    assert torch.equal(mixed["control"][0], auto["control"][0])
    assert torch.equal(mixed["control"][1], forced["control"][1])


def test_alm_on_an_empty_pack_is_a_noop_with_finite_stats():
    from src.diffusion.bspline_alm import bspline_alm_correct

    codec = BSplineCodec(degree=3, num_controls=8, curve_points=32,
                         knots_path="auto", endpoint_constrained=False)
    pack = _empty_pack(codec, B=2)
    q = _straight_controls(codec, offset=(0.0, 0.4))
    q = q.expand(2, -1, -1).contiguous()
    out, lam, stats = bspline_alm_correct(q, pack, codec, None,
                                          {"step_size": 0.05}, inner_steps=3)
    assert torch.equal(out, q)                     # nothing to project onto
    assert torch.isfinite(lam).all()
    for key in ("max_violation_before", "max_violation_after",
                "mean_positive_violation_after", "lambda_max"):
        assert torch.isfinite(stats[key]).all(), key
    assert float(stats["max_violation_after"].abs().max()) == 0.0


def test_training_rollout_survives_a_batch_without_any_corridor():
    """batch_size = 1, ``alm_valid = False``: the whole training step must run."""
    from train import LOSS_KEYS, batch_losses

    from src.diffusion.schedule import NoiseSchedule

    b = tiny_batch(B=1, H=8, M=3, L=8)
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8, d_model=32,
                       feedback={"enabled": True, "hidden": 16,
                                 "zero_init": False})
    model.train()
    A, bb, mask = _scene_cells(b)
    batch = _rollout_batch(b, model, A, bb, mask,
                           torch.zeros(1, dtype=torch.bool))   # corridor failed
    schedule = NoiseSchedule(8)
    raw, _, total, _, stats = batch_losses(
        batch, model, schedule, {"ellipse_safe_res": 16}, "cpu",
        alm_cfg={"inner_steps": 2, "max_curve_step_scene": 0.05},
        fb_cfg={"rollout": True, "drop_prob": 0.0})
    assert set(LOSS_KEYS) <= set(raw)
    assert stats["fb_valid_rate"] == 0.0
    assert stats["fb_mean_violation"] == 0.0
    assert float(raw["Lfbsafe"].detach()) == 0.0
    assert float(raw["Lcurve"].detach()) >= 0.0
    assert torch.isfinite(total).all()
    for part in stats["loss_parts"]:
        (part / 2).backward()                      # empty pack must not explode


def test_corridor_fit_reports_whether_a_skeleton_is_inside_the_corridor():
    from train import corridor_fit

    codec = BSplineCodec(degree=3, num_controls=8, curve_points=32,
                         knots_path="auto", endpoint_constrained=False)
    A, b, mask, _ = _tube_regions(half=0.10)
    inside = _straight_controls(codec)[0]
    outside = _straight_controls(codec, offset=(0.0, 0.4))[0]
    assert corridor_fit(inside, A, b, mask) == 1.0
    assert corridor_fit(outside, A, b, mask) == 0.0
    # a half-inside polyline (one end shifted) lands strictly in between
    mixed = inside.clone()
    mixed[len(mixed) // 2:] += torch.tensor([0.0, 0.4])
    half = corridor_fit(mixed, A, b, mask, stride=1)
    assert 0.0 < half < 1.0
    # no corridor at all is reported as 0, never as a crash
    empty = _empty_pack(codec, B=1)
    assert corridor_fit(inside, empty.piece_A[0], empty.piece_b[0],
                        empty.face_mask[0].any(dim=-1)) == 0.0


def test_second_step_topology_mode_is_configurable_and_logged():
    from train import batch_losses

    from src.diffusion.schedule import NoiseSchedule

    b = tiny_batch(B=2, H=8, M=3, L=8)
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8, d_model=32,
                       feedback={"enabled": True, "hidden": 16,
                                 "zero_init": False})
    model.train()
    A, bb, mask = _scene_cells(b)
    batch = _rollout_batch(b, model, A, bb, mask,
                           torch.ones(2, dtype=torch.bool))
    schedule = NoiseSchedule(8)
    lcfg = {"ellipse_safe_res": 16}
    alm_cfg = {"inner_steps": 2, "max_curve_step_scene": 0.05}
    _, _, _, _, expert = batch_losses(
        batch, model, schedule, lcfg, "cpu", alm_cfg=alm_cfg,
        fb_cfg={"rollout": True, "drop_prob": 0.0, "topology": "expert"})
    assert expert["fb_topo_match"] == 1.0          # expert routing IS m*
    assert "fb_topo_corridor_fit" not in expert    # not paid for on the expert path
    _, _, _, _, pred = batch_losses(
        batch, model, schedule, lcfg, "cpu", alm_cfg=alm_cfg,
        fb_cfg={"rollout": True, "drop_prob": 0.0, "topology": "pi"})
    assert 0.0 <= pred["fb_topo_match"] <= 1.0     # how often pi picks m*
    # the compatibility probe of review item 3 runs in the pi mode
    assert 0.0 <= pred["fb_topo_corridor_fit"] <= 1.0
    try:
        batch_losses(batch, model, schedule, lcfg, "cpu", alm_cfg=alm_cfg,
                     fb_cfg={"rollout": True, "topology": "nonsense"})
    except ValueError as exc:
        assert "topology" in str(exc)
    else:                                                   # pragma: no cover
        raise AssertionError("an unknown topology mode must be refused")


# ------------------------------------------------- full reverse loop (real)
def test_full_reverse_loop_with_the_real_planner_feeds_feedback_back(monkeypatch):
    """Real planner + real ALM/pack + real feedback cache over 4 reverse steps.

    ``build_safety_corridor`` is replaced by a trivially satisfied box so the
    test does not depend on a random tiny model's ellipses closing a real
    corridor.  Every constraint holds, which pins the "already safe -> delta = 0"
    branch, and the loop must still run the real ALM/DDIM/pack machinery.
    """
    import src.diffusion.sampler as sampler_mod
    from src.diffusion.sampler import sample as ddim_sample
    from src.diffusion.schedule import NoiseSchedule

    class _Cell:
        def __init__(self, A, b):
            self.A, self.b = A, b

        @property
        def face_count(self):
            return len(self.A)

    class _Corridor:
        valid = True
        failure_reason = None
        overlap_ratio = [0.9, 0.9, 0.9]
        bridge_cell_count = 0

        def __init__(self, cells=4, bound=1e6):
            A = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
            b = np.array([bound, bound, bound, bound])
            self.cells = [_Cell(A, b) for _ in range(cells)]

        @property
        def num_cells(self):
            return len(self.cells)

        def anchors(self):
            return np.linspace(0.0, 1.0, len(self.cells))

        def to_dict(self):
            return {"valid": True, "num_cells": self.num_cells}

    monkeypatch.setattr(sampler_mod, "build_safety_corridor",
                        lambda *a, **k: _Corridor())
    b = tiny_batch(B=2, H=8, M=3, L=8)
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8, d_model=32,
                       feedback={"enabled": True, "hidden": 16,
                                 "zero_init": False})
    schedule = NoiseSchedule(8)
    out = ddim_sample(
        model, schedule, b["cond"], b["occ"], b["candidate_xy"],
        b["candidate_mask"], b["candidate_geometry"],
        b["candidate_geometry_lengths"], device="cpu", steps=4, seed=0,
        return_trace=True,
        alm_config={"enabled": True, "mode": "guided_bspline",
                    "warmup_reverse_steps": 1, "max_activation_delay_steps": 1,
                    "activation_inner_steps": 2, "inner_steps": 2,
                    "max_curve_step_scene": 0.02},
        corridor_config={"topology_trials": 1})
    trace = out["trace"]
    assert out["feedback"]["enabled"] is True
    assert bool(out["guided"].all())
    assert out["alm_status"] == ["guided", "guided"]
    assert len(trace) == 4
    guided = [s for s in trace if s["alm_active"]]
    assert len(guided) == 3
    # the box is trivially satisfied -> "already safe" for every guided row
    assert out["feedback"]["history"]["already_safe"] == 2 * len(guided)
    assert out["feedback"]["history"]["rejected"] == 0
    assert bool(out["feedback"]["valid"].all())
    assert all(float(s["feedback_delta"].abs().max()) == 0.0 for s in trace[1:])
    # warm-up sees no history, the last reverse step does
    assert not bool(trace[0]["feedback_valid_in"].any())
    assert bool(trace[-1]["feedback_valid_in"].all())
    # each guided step reports the raw violation / ALM correction / smoothness
    for step in guided:
        assert step["raw_violation"] is not None
        assert step["alm_correction"] is not None
        assert float(step["alm_correction"].abs().max()) == 0.0
        assert step["curve_smoothness_raw"].shape == (2,)
    assert out["p"].shape == (2, 8, 2)
    assert bool(torch.isfinite(out["p"]).all())
