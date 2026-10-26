"""D-C48 / S-C48 model-level contracts (guide 一.1 and 一.2).

The paper claims D-C48 is a STANDARD conditional diffusion that never receives
ellipse safety regions, convex corridors, historical safety feedback or any
constraint state.  That claim is only worth making if it is mechanically true,
so it is asserted here:

* the D-C48 config really removes the safety-region modules (they are not built
  at all -- not built-and-zeroed);
* D-C48's forward signature admits exactly (occupancy, endpoints, noisy
  controls, timestep) plus the map-derived topology/candidate geometry;
* the ellipse / safety entries of a D-C48 forward are zero placeholders;
* the paper model (safety features ON) is untouched: the default 3-stream
  ``FusionMLP`` keeps the exact parameter shapes every existing checkpoint uses;
* the S-C48 config differs from A-C48 by the feedback rollout flag ONLY.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.models.trajsafe.planner import TrajSafePlanner
from src.utils.config import load_config, num_controls

A_CFG = "configs/config_160k4p_c48_oneshot.yaml"
D_CFG = "configs/config_160k4p_c48_d_std.yaml"
S_CFG = "configs/config_160k4p_c48_s_nofb.yaml"


def _build(cfg_path):
    cfg = load_config(cfg_path)
    model = TrajSafePlanner(cfg["model"], cfg.get("ellipse_label"), cfg["bspline"])
    return cfg, model


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


# --------------------------------------------------------------- paper model


def test_a_c48_keeps_every_safety_module_and_the_3_stream_fusion():
    cfg, model = _build(A_CFG)
    assert model.safety_features_enabled is True
    assert model.feedback_enabled is True
    for name in ("safety_query_head", "safety_cross_attention", "ellipse_geometry",
                 "ellipse_shape_head", "feedback_encoder", "feedback_fusion"):
        assert getattr(model, name) is not None, name
    assert model.fusion_mlp.num_streams == 3
    assert model.fusion_mlp.net[1].in_features == 3 * model.d_model


def test_default_and_s_c48_configs_are_the_paper_architecture():
    """S-C48 differs from A-C48 by the training rollout flag ONLY."""
    a = load_config(A_CFG)
    s = load_config(S_CFG)
    assert s["train"]["feedback"]["rollout"] is False
    assert a["train"]["feedback"]["rollout"] is True
    for key in ("model", "bspline", "loss", "alm", "corridor"):
        assert s[key] == a[key], key
    assert num_controls(s) == 48
    assert s["data"]["processed_root"] == a["data"]["processed_root"]


# -------------------------------------------------------------------- D-C48


def test_dc48_has_no_safety_region_and_no_feedback_modules():
    cfg, model = _build(D_CFG)
    assert model.safety_features_enabled is False
    assert model.feedback_enabled is False
    for name in ("safety_query_head", "safety_cross_attention", "ellipse_geometry",
                 "ellipse_shape_head", "feedback_encoder", "feedback_fusion"):
        assert getattr(model, name) is None, name
    assert model.fusion_mlp.num_streams == 2
    assert model.fusion_mlp.net[1].in_features == 2 * model.d_model
    # the safety branch really is gone from the parameter count
    _, a_model = _build(A_CFG)
    assert _count_params(model) < _count_params(a_model)


def test_dc48_config_disables_every_safety_loss():
    cfg = load_config(D_CFG)
    loss = cfg["loss"]
    for key in ("lambda_shape", "lambda_iou", "lambda_safe",
                "lambda_curve_smooth"):
        assert float(loss[key]) == 0.0, key
    assert cfg["train"]["feedback"]["rollout"] is False
    assert cfg["model"]["safety_features"]["enabled"] is False
    # the trajectory supervision is untouched
    for key in ("lambda_control", "lambda_coarse", "lambda_smooth",
                "lambda_boundary", "lambda_topology"):
        assert float(loss[key]) > 0.0, key


def test_dc48_forward_signature_admits_no_safety_information():
    """The allowed inputs are occupancy, endpoints, noisy controls, timestep."""
    cfg, model = _build(D_CFG)
    params = set(inspect.signature(model.forward_all).parameters)
    forbidden = {"ellipse", "ellipse_shape4", "corridor", "alm_cell_a",
                 "alm_cell_b", "alm_cell_valid", "violation", "lam", "lambda",
                 "multiplier", "safety", "guidance"}
    assert not (params & forbidden), params & forbidden
    # the topology/candidate geometry it DOES take is map+endpoint derived
    for allowed in ("q_t", "occ", "cond", "t", "candidate_xy", "candidate_mask",
                    "geometry", "geometry_lengths"):
        assert allowed in params, allowed


def test_dc48_ignores_the_shared_feedback_api_arguments():
    """``forward_all`` keeps the feedback_* parameters for API compatibility.

    For D-C48 they must be provably INERT: the modules are not built, so handing
    it a safety trace cannot change a single output bit.  This is the mechanical
    version of "D-C48 does not use historical safety feedback".
    """
    cfg, model = _build(D_CFG)
    model.eval()
    B, C, M, L, G = 1, num_controls(cfg), 4, 128, 1280
    torch.manual_seed(3)
    q_t = torch.randn(B, C, 2)
    cond = torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]])
    occ = (torch.rand(B, 1, 256, 256) > 0.85).float()
    candidate_xy = torch.rand(B, M, L, 2) * 2 - 1
    mask = torch.ones(B, M, dtype=torch.bool)
    geometry = torch.rand(B, M, G, 2) * 2 - 1
    lengths = torch.full((B, M), L, dtype=torch.long)
    t = torch.zeros(B, dtype=torch.long)
    ab = torch.ones(B)
    with torch.no_grad():
        plain = model.forward_all(q_t, occ, cond, t, ab, candidate_xy, mask,
                                  geometry, lengths)
        fed = model.forward_all(
            q_t, occ, cond, t, ab, candidate_xy, mask, geometry, lengths,
            feedback_delta=torch.rand(B, C, 2) * 4 - 2,
            feedback_valid=torch.ones(B, dtype=torch.bool))
    for key in ("control", "q_raw_final", "final", "q_coarse", "H_clean"):
        assert torch.equal(plain[key], fed[key]), key


def test_dc48_forward_emits_zero_safety_placeholders_and_a_valid_curve():
    cfg, model = _build(D_CFG)
    model.eval()
    B, C, M, L, G = 1, num_controls(cfg), 4, 128, 1280
    torch.manual_seed(0)
    q_t = torch.randn(B, C, 2)
    start = torch.tensor([[-0.8, -0.8]])
    goal = torch.tensor([[0.8, 0.8]])
    cond = torch.stack([start, goal], dim=1)
    occ = (torch.rand(B, 1, 256, 256) > 0.8).float()
    candidate_xy = torch.rand(B, M, L, 2) * 2 - 1
    mask = torch.ones(B, M, dtype=torch.bool)
    geometry = torch.rand(B, M, G, 2) * 2 - 1
    lengths = torch.full((B, M), L, dtype=torch.long)
    with torch.no_grad():
        out = model.forward_all(q_t, occ, cond, torch.zeros(B, dtype=torch.long),
                                torch.ones(B), candidate_xy, mask, geometry, lengths)
    assert torch.isfinite(out["control"]).all()
    assert torch.isfinite(out["final"]).all()
    # endpoints are the hard condition (clamped B-spline)
    assert torch.allclose(out["control"][:, 0], start, atol=1e-5)
    assert torch.allclose(out["control"][:, -1], goal, atol=1e-5)
    # safety placeholders: present, correctly shaped, and carrying no information
    for key in ("a", "b", "theta"):
        assert float(out["ellipse"][key].abs().max()) == 0.0, key
    assert float(out["ellipse"]["shape4"].abs().max()) == 0.0
    assert float(out["ellipse"]["H_ell"].abs().max()) == 0.0
    assert float(out["H_safety"].abs().max()) == 0.0
    assert float(out["A_safety"].abs().max()) == 0.0
    assert out["ellipse"]["center"].shape == (B, model.num_safety_queries, 2)
    assert out["H_fb"] is None and out["feedback_valid"] is None


def test_dc48_reuses_the_same_backbone_as_the_paper_model():
    """Backbone parity (guide 一.2): only the safety branch differs."""
    _, d_model = _build(D_CFG)
    _, a_model = _build(A_CFG)
    assert d_model.num_controls == a_model.num_controls == 48
    assert d_model.curve_points == a_model.curve_points == 128
    assert d_model.num_safety_queries == a_model.num_safety_queries == 128
    assert d_model.d_model == a_model.d_model
    assert d_model.traj_blocks == a_model.traj_blocks
    assert len(d_model.traj_backbone) == len(a_model.traj_backbone)
    assert len(d_model.final_denoiser.blocks) == len(a_model.final_denoiser.blocks)
    # the shared modules keep identical parameter shapes
    for name in ("traj_encoder", "head_p", "skeleton_encoder", "match_block",
                 "topology_head", "path_feature_head", "scene_cnn"):
        d_shapes = {k: tuple(v.shape) for k, v in
                    getattr(d_model, name).state_dict().items()}
        a_shapes = {k: tuple(v.shape) for k, v in
                    getattr(a_model, name).state_dict().items()}
        assert d_shapes == a_shapes, name


def test_fusion_mlp_default_keeps_checkpoint_compatible_shapes():
    from src.models.trajsafe.fusion import FusionMLP
    three = FusionMLP(128)
    two = FusionMLP(128, num_streams=2)
    assert three.net[0].normalized_shape == (384,)
    assert two.net[0].normalized_shape == (256,)
    h = torch.randn(2, 5, 128)
    assert three(h, h, h).shape == (2, 5, 128)
    assert two(h, h).shape == (2, 5, 128)
    with pytest.raises(ValueError):
        three(h, h)                       # the paper model needs the ellipse stream
    with pytest.raises(ValueError):
        FusionMLP(128, num_streams=4)


# ------------------------------------------------- architecture resolution
def test_architecture_detection_does_not_depend_on_the_safety_branch():
    """D-C48 has no safety modules, so the KEY HEURISTIC calls it "legacy".

    ``resolve_architecture`` must therefore prefer the checkpoint's own stored
    config.  Without that, a D-C48 checkpoint is dispatched to the legacy
    curve-token chain, which calls the (absent) ``ellipse_geometry`` and dies --
    this actually happened for every validation-sweep candidate.
    """
    from src.utils.checkpoint import (ARCH_CONTROL_SPACE, detect_architecture,
                                      resolve_architecture)

    # no safety keys at all (exactly D-C48's state dict)
    d_state = {"traj_backbone.0.norm.weight": torch.zeros(1),
               "fusion_mlp.net.1.weight": torch.zeros(1)}
    assert detect_architecture(d_state) != ARCH_CONTROL_SPACE     # the heuristic
    # 1) the checkpoint's own config wins
    resolved = resolve_architecture(
        {"control_space": True}, d_state, "auto",
        checkpoint_cfg={"model": {"control_space": True}})
    assert resolved["control_space"] is True
    # 2) without it, the heuristic is still used (old checkpoints)
    assert resolve_architecture({}, d_state, "auto")["control_space"] is False
    # 3) a safety-branch state is detected as control-space either way
    a_state = {"safety_query_head.net.0.weight": torch.zeros(1)}
    assert detect_architecture(a_state) == ARCH_CONTROL_SPACE
    assert resolve_architecture({}, a_state, "auto")["control_space"] is True
    # 4) an explicit arch always wins
    assert resolve_architecture(
        {}, d_state, "control",
        checkpoint_cfg={"model": {"control_space": False}})["control_space"] is True
    # 5) the stored config must beat a WRONG caller default
    assert resolve_architecture(
        {"control_space": False}, d_state, "auto",
        checkpoint_cfg={"model": {"control_space": True}})["control_space"] is True
