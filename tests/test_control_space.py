"""Contract tests for the control-space refactor.

The diffusion state is the C-control B-spline polygon and ``128`` survives only
as the number of Skeleton / ellipse "safety" queries.  These tests pin down:

  * C is config-driven (8 / 12 / 64), never hard-coded to 32;
  * the trajectory stream has exactly C tokens, the safety stream exactly Q;
  * the parameter-free ``BoundaryDecoder`` makes the endpoints exact and is
    invisible to ``state_dict``;
  * ``SafetyControlFusion`` starts as the identity (zero-init output);
  * the two forward chains dispatch on ``model.control_space``;
  * checkpoint detection / flexible loading across the refactor;
  * the new 7 exported losses and their exact zero points;
  * every parameter receives a gradient from the full 8-term loss.
"""
from __future__ import annotations

import pytest
import torch

from helpers import tiny_batch, tiny_model

from src.losses import losses
from src.models.trajsafe import TrajSafePlanner
from src.models.trajsafe.boundary import BoundaryDecoder
from src.models.trajsafe.fusion import SafetyControlFusion
from src.utils.checkpoint import (ARCH_CONTROL_SPACE, ARCH_LEGACY_CURVE,
                                  detect_architecture, load_state_dict_flexible)
from src.utils.config import curve_points as cfg_curve_points
from src.utils.config import num_controls as cfg_num_controls
from src.utils.config import num_safety_queries as cfg_num_safety_queries


def _forward(model, b, select_index=None):
    return model.forward_all(
        b["pos"], b["occ"], b["cond"], b["t"], b["ab"],
        b["candidate_xy"], b["candidate_mask"], b["candidate_geometry"],
        b["candidate_geometry_lengths"], select_index=select_index)


# ----------------------------------------------------------- config-driven C
def test_control_count_is_config_driven_for_8_12_and_64():
    for C in (8, 12, 64):
        Q = 8
        model = tiny_model(num_controls=C, curve_points=8,
                           num_safety_queries=Q, rel_bias_len=C, d_model=32)
        # the codec and the network share the ONE control count
        assert model.num_controls == C
        assert model.bspline.num_controls == C
        assert model.bspline.basis.shape == (model.curve_points, C)
        assert model.traj_encoder.horizon == model.curve_points
        assert model.rel_bias_len >= max(C, Q, model.curve_points)

        b = tiny_batch(B=2, H=C, M=3, L=Q)
        out = _forward(model, b)
        assert out["H_traj"].shape == (2, C, 32)
        assert out["R"].shape == (2, 3, C, 32)
        assert out["R_use"].shape == (2, C, 32)
        assert out["H_path"].shape == (2, C, 32)
        assert out["control"].shape == (2, C, 2)
        assert out["q_raw_final"].shape == (2, C, 2)
        assert out["H_safety"].shape == (2, Q, 32)
        assert out["A_safety"].shape == (2, C, 32)
        assert out["ellipse"]["shape4"].shape == (2, Q, 4)
        assert out["final"].shape == (2, model.curve_points, 2)

    # model.num_controls and bspline.num_controls MUST agree
    with pytest.raises(ValueError):
        TrajSafePlanner({"num_controls": 8, "d_model": 16, "num_heads": 4},
                        None, {"num_controls": 12, "knots": "auto"})

    # the config helpers are the single source of truth
    cfg = {"bspline": {"num_controls": 12, "curve_points": 16},
           "model": {"num_controls": 12, "horizon": 16,
                     "num_safety_queries": 16}}
    assert cfg_num_controls(cfg) == 12
    assert cfg_curve_points(cfg) == 16
    assert cfg_num_safety_queries(cfg) == 16
    with pytest.raises(ValueError):
        cfg_num_controls({"bspline": {"num_controls": 8},
                          "model": {"num_controls": 12}})


def test_c8_model_has_exactly_eight_control_tokens():
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8)
    b = tiny_batch(B=2, H=8, L=8, M=3)
    out = _forward(model, b)
    assert out["input_control"].shape[1] == 8
    assert out["H_traj"].shape[1] == 8
    assert out["control"].shape[1] == 8
    assert out["final"].shape == (2, model.curve_points, 2)
    assert model.horizon == 8
    # a 32-token state is now an error, not the silent default
    bad = dict(b)
    bad["pos"] = torch.randn(2, 32, 2)
    with pytest.raises(ValueError):
        _forward(model, bad)


def test_safety_query_count_is_decoupled_from_the_curve_density():
    """Q (Skeleton / ellipse queries) is NOT the decoded curve density."""
    model = tiny_model(num_controls=8, curve_points=16, num_safety_queries=4,
                       rel_bias_len=16)
    b = tiny_batch(B=2, H=8, L=4, M=3)
    out = _forward(model, b)
    assert model.horizon == 16
    assert model.bspline.curve_points == 16
    assert out["H_traj"].shape[1] == model.num_controls == 8
    assert out["H_safety"].shape[1] == model.num_safety_queries == 4
    assert out["ellipse"]["progress"].shape == (2, 4)
    assert out["ellipse"]["shape4"].shape == (2, 4, 4)
    assert out["final"].shape == (2, 16, 2)
    assert out["raw_curve"].shape == (2, 16, 2)


# ------------------------------------------------------------ boundary decoder
def test_boundary_decoder_endpoint_exactness_and_profile():
    bd = BoundaryDecoder()
    assert list(bd.parameters()) == []
    assert "profile" not in bd.state_dict()

    for C in (2, 3, 4, 5, 8, 12, 32):
        torch.manual_seed(C)
        q_raw = torch.randn(3, C, 2)
        cond = torch.randn(3, 2, 2)
        q = bd(q_raw, cond)
        assert q.shape == (3, C, 2)
        # q_0 = start and q_{C-1} = goal for ANY raw prediction
        assert torch.allclose(q[:, 0], cond[:, 0], atol=1e-6)
        assert torch.allclose(q[:, -1], cond[:, 1], atol=1e-6)
        ws, wg = bd.weights(C)
        assert ws.shape == (C,) and wg.shape == (C,)
        assert float(bd.weights(C)[0][0]) == 1.0
        assert float(bd.weights(C)[1][-1]) == 1.0
        assert float(ws.sum()) > 0.0 and float(wg.sum()) > 0.0

    # C=2 degenerates to plain endpoint hard-conditioning
    ws2, wg2 = bd.weights(2)
    assert torch.allclose(ws2, torch.tensor([1.0, 0.0]))
    assert torch.allclose(wg2, torch.tensor([0.0, 1.0]))
    # C=4 uses the full profile mirrored at both ends
    ws4, wg4 = bd.weights(4)
    assert torch.allclose(ws4, torch.tensor([1.0, 0.75, 0.0, 0.0]))
    assert torch.allclose(wg4, torch.tensor([0.0, 0.0, 0.75, 1.0]))


def test_raw_controls_keep_an_endpoint_error_that_the_decoder_fixes():
    torch.manual_seed(3)
    model = tiny_model()
    b = tiny_batch(B=2, H=model.num_controls)
    out = _forward(model, b)
    for raw_key, fixed_key in (("q_raw_final", "control"),
                               ("q_coarse_raw", "q_coarse")):
        raw_err0 = (out[raw_key][:, 0] - b["cond"][:, 0]).abs().max().detach()
        raw_err1 = (out[raw_key][:, -1] - b["cond"][:, 1]).abs().max().detach()
        assert float(raw_err0) > 1e-4 and float(raw_err1) > 1e-4
        assert torch.allclose(out[fixed_key][:, 0], b["cond"][:, 0], atol=1e-6)
        assert torch.allclose(out[fixed_key][:, -1], b["cond"][:, 1], atol=1e-6)


# ------------------------------------------------------------------- fusion
def test_safety_cross_attention_starts_as_the_identity():
    fusion = SafetyControlFusion(16, 4)
    assert torch.count_nonzero(fusion.out.weight) == 0
    assert torch.count_nonzero(fusion.out.bias) == 0
    h = torch.randn(2, 5, 16)
    mem = torch.randn(2, 7, 16)
    h_t = torch.randn(2, 16)
    out = fusion(h, mem, h_t)
    assert out.shape == h.shape
    assert torch.allclose(out, h, atol=1e-6)


# ------------------------------------------------------------- chain dispatch
def test_control_space_flag_dispatches_the_two_chains():
    b = tiny_batch(B=2, H=8, L=8, M=3)

    ctrl = tiny_model(control_space=True, num_controls=8, curve_points=8,
                      num_safety_queries=8)
    out = _forward(ctrl, b)
    assert ctrl.control_space is True
    assert out["H_traj"].shape[1] == ctrl.num_controls == 8
    assert out["R"].shape[2] == 8
    assert out["control"].shape == (2, 8, 2)
    assert out["q_raw_final"].shape == (2, 8, 2)
    assert out["ellipse"]["shape4"].shape[1] == ctrl.num_safety_queries

    legacy = tiny_model(control_space=False, num_controls=8, curve_points=8,
                        num_safety_queries=8)
    out2 = _forward(legacy, b)
    assert legacy.control_space is False
    assert out2["H_traj"].shape[1] == legacy.curve_points == 8
    assert out2["R"].shape[2] == legacy.curve_points
    # the legacy chain fits the raw curve-space output back onto the controls,
    # so q_raw_final IS the corrected control polygon
    assert out2["q_raw_final"].shape == (2, legacy.num_controls, 2)
    assert torch.equal(out2["q_raw_final"], out2["control"])
    assert out2["final"].shape == (2, legacy.curve_points, 2)


# ---------------------------------------------------------------- checkpoints
def test_checkpoint_detection_and_flexible_loading():
    model = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8)
    with torch.no_grad():
        model.traj_backbone[0].sa.b_horizon.copy_(
            torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4))
    state = dict(model.state_dict())
    assert detect_architecture(state) == ARCH_CONTROL_SPACE

    legacy_state = {k: v for k, v in state.items()
                    if not k.startswith(("safety_query_head.",
                                         "safety_cross_attention."))}
    assert detect_architecture(legacy_state) == ARCH_LEGACY_CURVE
    assert detect_architecture({"model_state": legacy_state}) == ARCH_LEGACY_CURVE

    # (a) a differently sized relative-bias table is adapted (first rows kept)
    big = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                     rel_bias_len=16)
    report = load_state_dict_flexible(big, state, verbose=False)
    assert report["arch"] == ARCH_CONTROL_SPACE
    assert any("b_horizon" in entry for entry in report["adapted"])
    loaded = big.traj_backbone[0].sa.b_horizon
    assert loaded.shape == (16, 4)
    assert torch.allclose(loaded[:8],
                          model.traj_backbone[0].sa.b_horizon)
    assert torch.count_nonzero(loaded[8:]) == 0

    # (b) a legacy state dict never raises: the new modules stay fresh
    fresh = tiny_model(num_controls=8, curve_points=8, num_safety_queries=8,
                       rel_bias_len=8)
    report2 = load_state_dict_flexible(fresh, legacy_state, verbose=False)
    assert report2["arch"] == ARCH_LEGACY_CURVE
    assert report2["adapted"] == []
    for key in report2["missing"]:
        assert key.startswith(("safety_query_head.", "safety_cross_attention."))

    # (c) BoundaryDecoder needs no checkpoint entry at all
    assert not any(k.startswith("boundary_decoder.") for k in fresh.state_dict())
    assert "boundary_decoder.profile" not in fresh.state_dict()


# -------------------------------------------------------------------- losses
def test_control_losses_have_the_expected_zero_points():
    # L_smooth ~ 0 on an affine control polygon, > 0 on a jagged one
    t = torch.linspace(0.0, 1.0, 16)[None, :, None]
    straight = torch.cat([t, 2.0 * t + 1.0], dim=-1).expand(3, 16, 2).clone()
    assert float(losses.control_smoothness_loss(straight, straight)) < 1e-6
    jagged = straight.clone()
    jagged[:, 8] += 0.5
    assert float(losses.control_smoothness_loss(jagged, straight)) > 0.0

    # L_boundary ~ 0 exactly on the GT local polygon translated to S/G
    bd = BoundaryDecoder()
    torch.manual_seed(0)
    q_gt = torch.randn(3, 8, 2)
    cond = torch.randn(3, 2, 2)
    target = bd.boundary_targets(q_gt, cond)
    ws, wg = bd.weights(8)
    assert float(losses.boundary_control_loss(
        target, q_gt, cond, ws, wg)) < 1e-6
    perturbed = target.clone()
    perturbed[:, 2] += 1.0
    assert float(losses.boundary_control_loss(
        perturbed, q_gt, cond, ws, wg)) > 0.0


def test_loss_exports_are_exactly_the_new_terms():
    expected = {"control_x0_loss", "control_smoothness_loss",
                "boundary_control_loss", "topology_ce", "ellipse_shape_loss",
                "ellipse_iou_loss", "ellipse_safety_loss"}
    import src.losses as losses_pkg
    assert set(losses_pkg.__all__) == expected
    namespace = {}
    exec("from src.losses import *", namespace)
    assert {k for k in namespace if not k.startswith("__")} == expected
    assert not hasattr(losses_pkg, "trajectory_x0_loss")
    assert "trajectory_x0_loss" not in namespace
    with pytest.raises(ImportError):
        exec("from src.losses import trajectory_x0_loss", {})


# ----------------------------------------------------------------- gradients
def test_every_parameter_receives_a_gradient_from_the_eight_term_loss():
    torch.manual_seed(0)
    model = tiny_model()
    b = tiny_batch(B=2, H=model.num_controls, L=model.num_safety_queries)
    out = _forward(model, b)
    q0 = b["pos"]                                    # [B,C,2] GT controls
    cond = b["cond"]
    ell = out["ellipse"]
    has_cand = b["has_candidate"]
    ws, wg = model.boundary_decoder.weights(q0.shape[1])

    l_safe, _, _ = losses.ellipse_safety_loss(
        ell["center"], ell["a"], ell["b"], ell["theta"], b["occ"],
        raster_res=32, sample_mask=has_cand)
    raw = {
        "Lctrl": losses.control_x0_loss(out["q_raw_final"], q0),
        "Lcoarse": losses.control_x0_loss(out["q_coarse_raw"], q0),
        "Lsmooth": losses.control_smoothness_loss(out["q_raw_final"], q0),
        "Lboundary": losses.boundary_control_loss(
            out["q_raw_final"], q0, cond, ws, wg),
        "Ltopo": losses.topology_ce(out["topo"]["pi"], b["topology_best"],
                                    has_cand),
        "Lshape": losses.ellipse_shape_loss(
            ell["shape4"], b["ellipse_shape4_gt"], b["shape_valid"], has_cand),
        "Liou": losses.ellipse_iou_loss(
            ell["center"], ell["a"], ell["b"], ell["theta"], b["ellipse_mask"],
            b["shape_valid"], has_cand, raster_res=32),
        "Lsafe": l_safe,
    }
    for name, value in raw.items():
        assert torch.isfinite(value).all(), name

    model.zero_grad()
    sum(raw.values()).backward()
    missing = [name for name, p in model.named_parameters()
               if p.grad is None]
    assert missing == []
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), name
