"""Contract tests for the report-faithful TrajSafe-Diffuser.

The tests assert the structural contracts that the implementation report makes
explicit:

  * only the trajectory is a diffusion state;
  * every denoising block gets the timestep through AdaLN;
  * coarse and final share the SAME Head_P;
  * there is exactly ONE trajectory-skeleton MatchBlock;
  * progress is structurally monotone with s_0 = 0, s_{H-1} = 1;
  * the ellipse centre always comes from the selected Skeleton Curve;
  * the ellipse head has no centre output;
  * a >= b > 0 and (cos 2t, sin 2t) is a unit vector, including the (0, 0)
    fallback with finite gradients;
  * invalid candidates never receive probability mass;
  * ShapeValid=False entries do not enter shape / IoU supervision;
  * L_center reaches the Progress Head, L_shape / L_safe reach the Ellipse Head;
  * all key tensor shapes match report section 27.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
import torch

from v3_utils import tiny_batch, tiny_model

from src.diffusion import sampler_v3
from src.diffusion.schedule import NoiseSchedule
from src.geometry.ellipse_shape import raw_to_shape4, shape4_to_abtheta
from src.geometry.skeleton_paths import nearest_arclength
from src.losses import v3_losses
from src.models.skeleton_v3 import MatchBlock, SkeletonPlannerV3


def _forward(model, b, select_index=None):
    return model.forward_all(
        b["pos"], b["occ"], b["cond"], b["t"], b["ab"],
        b["candidate_xy"], b["candidate_mask"], b["candidate_geometry"],
        b["candidate_geometry_lengths"], select_index=select_index)


# --------------------------------------------------------------------- shapes
def test_forward_shapes_and_structural_constraints():
    model = tiny_model(horizon=8, d_model=32)
    b = tiny_batch(H=8)
    out = _forward(model, b)
    B, H, D = 2, 8, 32
    M, L = b["candidate_mask"].shape[1], b["candidate_xy"].shape[2]
    assert out["H_traj"].shape == (B, H, D)
    assert out["H_S"].shape == (B, M, L, D)
    assert out["R"].shape == (B, M, H, D)
    assert out["topo"]["pi"].shape == (B, M)
    assert out["H_prog"].shape == (B, H, D)
    assert out["ellipse"]["progress"].shape == (B, H)
    assert out["ellipse"]["center"].shape == (B, H, 2)
    assert out["ellipse"]["H_ell"].shape == (B, H, D)
    assert out["ellipse"]["shape4"].shape == (B, H, 4)
    assert out["F"].shape == (B, H, D)
    assert out["H_clean"].shape == (B, H, D)
    assert out["final"].shape == (B, H, 2)

    s = out["ellipse"]["progress"]
    assert torch.allclose(s[:, 0], torch.zeros(B), atol=1e-6)
    assert torch.allclose(s[:, -1], torch.ones(B), atol=1e-5)
    assert torch.all(s[:, 1:] > s[:, :-1])
    assert torch.all(out["ellipse"]["a"] >= out["ellipse"]["b"] - 1e-7)
    assert torch.all(out["ellipse"]["a"] > 0.0)
    unit = (out["ellipse"]["shape4"][..., 2] ** 2
            + out["ellipse"]["shape4"][..., 3] ** 2)
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-5)
    assert torch.isfinite(out["final"]).all()


def test_timestep_enters_every_denoising_block_through_adaln():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    (out["final"].sum() + out["coarse"].sum()
     + out["ellipse"]["shape4"].sum()).backward()
    adalns = [(n, m) for n, m in model.named_modules()
              if type(m).__name__ == "AdaLN"]
    assert len(adalns) >= model.traj_blocks + model.final_blocks + 3
    nonzero = 0
    for name, module in adalns:
        grad = module.mod.weight.grad
        assert grad is not None and torch.isfinite(grad).all(), name
        if float(grad.abs().sum()) > 0.0:
            nonzero += 1
    assert nonzero >= model.traj_blocks + model.final_blocks + 3


def test_skeleton_encoder_does_not_receive_timestep_or_handcrafted_features():
    model = tiny_model()
    enc = model.skeleton_encoder
    # after the shared SpatialPE the MLP_S input is exactly D-dimensional
    assert enc.mlp_s.net[0].in_features == model.d_model
    src = inspect.getsource(enc.forward)
    assert "h_t" not in src and "global" not in src and "candidate_features" not in src


# ------------------------------------------------------------------ shared head
def test_coarse_and_final_share_the_same_head_p():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    before = model.head_p.weight.detach().clone()
    with torch.no_grad():
        model.head_p.weight.mul_(0.0).add_(1.0)
    out2 = _forward(model, b)
    assert not torch.allclose(out["coarse"], out2["coarse"])
    assert not torch.allclose(out["final"], out2["final"])
    with torch.no_grad():
        model.head_p.weight.copy_(before)

    src = inspect.getsource(SkeletonPlannerV3)
    assert src.count("self.head_p(") == 2
    heads = [m for m in model.modules()
             if isinstance(m, torch.nn.Linear) and m.out_features == 2]
    assert len(heads) == 1 and heads[0] is model.head_p


def test_only_one_match_block_and_one_progress_path():
    model = tiny_model()
    matches = [m for m in model.modules() if isinstance(m, MatchBlock)]
    assert len(matches) == 1 and matches[0] is model.match_block
    src = inspect.getsource(SkeletonPlannerV3)
    assert "match_block" in src
    assert "second" not in src.lower()
    # no leftover interleaved fusion / type embeddings / ellipse diffusion state
    for forbidden in ("JointFusionBlock", "traj_type", "role_type", "mlp_e",
                      "head_e", "center_head", "chamfer", "candidate_lengths"):
        assert forbidden not in src, forbidden


def test_no_ellipse_diffusion_state_or_second_cross_attention():
    model = tiny_model()
    assert not hasattr(model, "ellipse")
    names = [n for n, _ in model.named_modules()]
    assert "ellipse_geometry" in names and "ellipse_shape_head" in names
    # the old ellipse head class is gone; only the report's two ellipse modules exist
    assert not hasattr(model, "mlp_e")
    src = inspect.getsource(sampler_v3.sample_v3)
    for forbidden in ("eps_e", "x0_e", "ellipse_state"):
        assert forbidden not in src, forbidden
    assert re.search(r"\be_t\b", src) is None
    assert re.search(r"\bcommit_t\b", src) is None


# ------------------------------------------------------------------- progress
def test_progress_head_is_monotone_for_any_input():
    model = tiny_model(horizon=8, d_model=32)
    h_prog, s = model.progress_head(torch.randn(3, 8, 32) * 30.0)
    assert h_prog.shape == (3, 8, 32)
    assert torch.all(s[:, 1:] > s[:, :-1])
    assert torch.allclose(s[:, 0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(s[:, -1], torch.ones(3), atol=1e-5)


def test_center_always_comes_from_the_selected_skeleton_curve():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b, select_index=torch.zeros(2, dtype=torch.long))
    s = out["ellipse"]["progress"]
    expected = model.curve_decoder(out["gamma"], out["gamma_lengths"], s)
    assert torch.allclose(out["ellipse"]["center"], expected, atol=1e-6)

    mask = out["has_candidate"]
    for b in range(out["final"].shape[0]):
        if not bool(mask[b]):
            continue
        n = int(out["gamma_lengths"][b])
        _, dist = nearest_arclength(out["ellipse"]["center"][b].detach().numpy(),
                                    out["gamma"][b, :n].detach().numpy())
        assert float(np.max(dist)) < 1e-5


# ------------------------------------------------------------------- topology
def test_invalid_candidates_get_zero_probability():
    model = tiny_model()
    b = tiny_batch(B=4, M=3)
    b["candidate_mask"][:] = False
    b["candidate_mask"][:, 1] = True
    b["candidate_geometry_lengths"][:] = 0
    b["candidate_geometry_lengths"][:, 1] = b["candidate_geometry"].shape[2]
    out = _forward(model, b)
    pi = out["topo"]["pi"]
    assert torch.allclose(pi.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert float(pi[:, 0].abs().sum()) == 0.0
    assert float(pi[:, 2].abs().sum()) == 0.0
    assert bool((out["selected_idx"] == 1).all())
    assert torch.isinf(out["topo"]["logits"][:, 0]).all()
    assert torch.isinf(out["topo"]["logits"][:, 2]).all()


def test_all_invalid_candidates_produce_no_nan_and_fall_back_to_coarse():
    model = tiny_model()
    b = tiny_batch(B=3, M=3)
    b["candidate_mask"][:] = False
    b["candidate_geometry_lengths"][:] = 0
    out = _forward(model, b)
    assert torch.isfinite(out["final"]).all()
    assert torch.allclose(out["final"], out["coarse"], atol=1e-7)
    assert float(out["topo"]["pi"].abs().sum()) == 0.0
    assert torch.isfinite(out["ellipse"]["center"]).all()
    assert torch.isfinite(out["ellipse"]["shape4"]).all()


# --------------------------------------------------------------- ellipse head
def test_ellipse_head_has_no_center_output():
    model = tiny_model()
    head = model.ellipse_shape_head
    assert not any(isinstance(m, torch.nn.Linear) and m.out_features == 2
                   for m in head.modules())
    src = inspect.getsource(type(head))
    assert "center" not in src
    out = head(torch.randn(2, 5, model.d_model), torch.randn(2, model.d_model))
    assert set(out.keys()) == {"raw", "shape4", "a", "b", "theta"}


def test_shape_parameterisation_is_ordered_and_unit():
    raw = torch.tensor([[2.0, 1.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0, 4.0],
                        [-5.0, 5.0, 0.0, 0.0],
                        [0.0, 0.0, 1e-9, -1e-9]])
    shape4 = raw_to_shape4(raw)
    a, b, theta = shape4_to_abtheta(shape4)
    assert torch.all(a >= b) and torch.all(b > 0)
    unit = shape4[..., 2] ** 2 + shape4[..., 3] ** 2
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-6)
    # exact zero direction must fall back to (1, 0) -> theta = 0
    assert torch.allclose(shape4[0, 2:4], torch.tensor([1.0, 0.0]))
    assert torch.allclose(theta[0], torch.tensor(0.0))
    assert torch.isfinite(shape4).all() and torch.isfinite(theta).all()


def test_zero_direction_backward_is_nan_free():
    raw = torch.zeros(2, 4, 4, requires_grad=True)
    shape4 = raw_to_shape4(raw)
    a, b, theta = shape4_to_abtheta(shape4)
    (shape4.sum() + theta.sum()).backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


# ------------------------------------------------------------------- losses
def test_shape_loss_ignores_invalid_locations():
    torch.manual_seed(0)
    shape4 = torch.randn(2, 5, 4, requires_grad=True)
    gt = shape4.detach().clone()
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[0, 0] = False
    sample = torch.ones(2, dtype=torch.bool)
    l1 = v3_losses.ellipse_shape_loss(shape4, gt, valid, sample)
    gt2 = gt.clone()
    gt2[0, 0] = 1e6
    l2 = v3_losses.ellipse_shape_loss(shape4, gt2, valid, sample)
    assert torch.allclose(l1, l2)


def test_iou_loss_ignores_invalid_locations():
    center = torch.zeros(1, 2, 2, requires_grad=True)
    a = torch.full((1, 2), 0.2, requires_grad=True)
    b = torch.full((1, 2), 0.1, requires_grad=True)
    theta = torch.zeros(1, 2, requires_grad=True)
    gt = torch.zeros(1, 2, 16, 16)
    valid = torch.tensor([[True, False]])
    sample = torch.ones(1, dtype=torch.bool)
    l1 = v3_losses.ellipse_iou_loss(center, a, b, theta, gt, valid, sample,
                                    raster_res=16)
    gt2 = gt.clone()
    gt2[0, 1] = 1.0
    l2 = v3_losses.ellipse_iou_loss(center, a, b, theta, gt2, valid, sample,
                                    raster_res=16)
    assert torch.allclose(l1, l2)


def test_center_loss_gives_gradient_to_progress_head():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    model.zero_grad()
    loss = v3_losses.ellipse_center_loss(out["ellipse"]["center"],
                                         b["ellipse_center_gt"],
                                         b["has_candidate"])
    loss.backward()
    g = model.progress_head.mlp_prog[1].weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0


def test_shape_loss_gives_gradient_to_ellipse_head():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    model.zero_grad()
    loss = v3_losses.ellipse_shape_loss(out["ellipse"]["shape4"],
                                        b["ellipse_shape4_gt"],
                                        b["shape_valid"], b["has_candidate"])
    loss.backward()
    g = model.ellipse_shape_head.mlp[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0


def test_safety_loss_gives_gradient_to_ellipse_shape():
    model = tiny_model()
    b = tiny_batch()
    # a partial obstacle makes the unsafe fraction depend on a/b/theta
    b["occ"][:, :, b["occ"].shape[-2] // 2:, :] = 1.0
    out = _forward(model, b)
    ell = out["ellipse"]
    model.zero_grad()
    loss, mean, cvar = v3_losses.ellipse_safety_loss(
        ell["center"], ell["a"], ell["b"], ell["theta"], b["occ"],
        sample_mask=b["has_candidate"])
    loss.backward()
    grads = [p.grad for p in model.ellipse_shape_head.parameters()
             if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0.0 for g in grads)
    assert float(mean) >= 0.0 and float(cvar) >= 0.0


def test_trajectory_losses_are_finite():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    l_traj = v3_losses.trajectory_x0_loss(out["final"], b["pos"])
    l_coarse = v3_losses.trajectory_x0_loss(out["coarse"], b["pos"])
    l_smooth = v3_losses.trajectory_smoothness_loss(out["final"], b["pos"])
    assert torch.isfinite(l_traj) and torch.isfinite(l_coarse)
    assert torch.isfinite(l_smooth)


# ------------------------------------------------------------------ sampler
def test_sampler_only_updates_the_trajectory_and_rescores_every_step():
    model = tiny_model()
    b = tiny_batch(B=1, M=2)
    calls = {"n": 0}
    orig = model.topology_head.forward

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    model.topology_head.forward = spy
    out = sampler_v3.sample_v3(
        model, NoiseSchedule(16), b["cond"], b["occ"], b["candidate_xy"],
        b["candidate_mask"], b["candidate_geometry"],
        b["candidate_geometry_lengths"], device="cpu", seed=0,
        return_trace=True)
    assert calls["n"] == len(out["trace"]) == 16
    assert out["p"].shape == (1, model.horizon, 2)
    assert torch.isfinite(out["p"]).all()
    assert torch.allclose(out["p"], out["trace"][-1]["final"], atol=1e-6)
    assert "ellipse_center" in out and out["ellipse_center"].shape == (1, 8, 2)


def test_subsampled_schedule_keeps_the_clean_transition():
    model = tiny_model()
    b = tiny_batch(B=1, M=2)
    out = sampler_v3.sample_v3(
        model, NoiseSchedule(16), b["cond"], b["occ"], b["candidate_xy"],
        b["candidate_mask"], b["candidate_geometry"],
        b["candidate_geometry_lengths"], device="cpu", steps=4, seed=0,
        return_trace=True)
    trace = out["trace"]
    assert len(trace) == 4
    assert all(step["s"] >= 0 for step in trace[:-1])
    assert trace[-1]["t"] == 0 and trace[-1]["s"] == -1
    assert torch.allclose(out["p"], trace[-1]["final"], atol=1e-6)


def test_sampler_source_has_no_commit_or_ellipse_state():
    src = inspect.getsource(sampler_v3.sample_v3)
    for forbidden in ("committed", "selected_path", "selected_feat",
                      "eps_e", "x0_e", "ellipse_state"):
        assert forbidden not in src, forbidden
    assert re.search(r"\be_t\b", src) is None
    assert re.search(r"\bcommit_t\b", src) is None
