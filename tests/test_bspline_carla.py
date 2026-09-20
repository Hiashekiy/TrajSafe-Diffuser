"""Tests for the CARLA + 32-control B-spline control-space pipeline.

Covers the mandatory checks of the night-run spec:
  1. B-spline decode [B,32,2] -> [B,128,2]
  2. hard control endpoints == curve endpoints
  3. curve -> control -> curve on real CARLA samples (RMSE / max error)
  4. fixed progress s_i = i/127
  5. ellipse center [B,128,2] / shape4 [B,128,4]
  6. the real diffusion state is [B,32,2]
  7. one full forward + backward with finite gradients
  8. CARLA y-axis: raw row 0 is y_local=+40 m, canonical map is flipped
  9. no learned progress / no alignment loss remains in the main chain
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.geometry.bspline import (BSplineCodec, bspline_basis_matrix,
                                  numpy_fit_curve_to_controls)
from src.models.trajsafe import TrajSafePlanner
from src.utils.config import load_config

KNOTS = os.path.join(ROOT, "data", "carla_v1", "bspline_knots.npy")
SCENE_TO_METER = 40.0
FORBIDDEN = ["progress_gt", "ellipse_center_gt", "lambda_align",
             "center_alignment_loss", "Head_prog", "ProgressHead"]


def _codec():
    return BSplineCodec(degree=3, num_controls=32, curve_points=128,
                        knots_path=KNOTS)


def _carla_samples(n=8):
    files = sorted(glob.glob(os.path.join(
        ROOT, "data", "carla_v1", "samples", "train", "*.npz")))[:n]
    out = []
    for f in files:
        with np.load(f) as z:
            out.append((z["trajectory_128"].astype(np.float64),
                        z["start"].astype(np.float64),
                        z["goal"].astype(np.float64),
                        z["occupancy"]))
    return out


def test_1_decode_shape():
    codec = _codec()
    q = torch.randn(4, 32, 2)
    p = codec.decode_controls(q)
    assert tuple(p.shape) == (4, 128, 2)
    assert tuple(codec.basis.shape) == (128, 32)
    assert tuple(codec.interior_pinv.shape) == (30, 128)


def test_2_hard_endpoints():
    codec = _codec()
    cond = torch.tensor([[[-0.75, 0.0], [0.4, 0.3]]], dtype=torch.float32)
    q = torch.randn(1, 32, 2)
    q = BSplineCodec.hard_control_endpoints(q, cond)
    p = codec.decode_controls(q)
    assert torch.allclose(p[0, 0], cond[0, 0], atol=1e-6)
    assert torch.allclose(p[0, -1], cond[0, 1], atol=1e-6)


def test_3_curve_to_control_roundtrip_real_data():
    codec = _codec()
    samples = _carla_samples(8)
    assert samples, "no CARLA samples found"
    rmses, maxes = [], []
    for traj, start, goal in [(s[0], s[1], s[2]) for s in samples]:
        q = numpy_fit_curve_to_controls(np.load(KNOTS), 32, 3, traj, start,
                                        goal, 128)
        assert np.allclose(q[0], start, atol=1e-9)
        assert np.allclose(q[-1], goal, atol=1e-9)
        rec = codec.decode_controls(
            torch.as_tensor(q, dtype=torch.float32)[None])[0].detach().numpy()
        err = np.linalg.norm(rec - traj, axis=1)
        rmses.append(float(np.sqrt((err ** 2).mean())))
        maxes.append(float(err.max()))
    rmses = np.asarray(rmses)
    maxes = np.asarray(maxes)
    print("fit rmse m p50/p95/max: %.5f %.5f %.5f"
          % (np.percentile(rmses, 50) * SCENE_TO_METER,
             np.percentile(rmses, 95) * SCENE_TO_METER,
             rmses.max() * SCENE_TO_METER))
    print("fit max  m p50/p95/max: %.5f %.5f %.5f"
          % (np.percentile(maxes, 50) * SCENE_TO_METER,
             np.percentile(maxes, 95) * SCENE_TO_METER,
             maxes.max() * SCENE_TO_METER))
    # the endpoint constraint forces the last control onto the route goal, so
    # the fit error is dominated by the goal-vs-executed-endpoint gap
    assert rmses.max() * SCENE_TO_METER < 1.0
    assert maxes.max() * SCENE_TO_METER < 5.0


def test_4_fixed_progress():
    cfg = load_config("configs/config.yaml")
    model = TrajSafePlanner(cfg["model"], cfg.get("ellipse_label"),
                            cfg.get("bspline"))
    s = model.fixed_progress
    assert s.shape == (128,)
    assert float(s[0]) == 0.0 and float(s[-1]) == 1.0
    assert torch.allclose(s[1:] - s[:-1], torch.full((127,), 1.0 / 127.0),
                          atol=1e-7)
    assert not hasattr(model, "progress_head")


# --------------------------------------------------------------- full forward
def _synthetic_batch(cfg, B=2, M=4):
    torch.manual_seed(0)
    H = int(cfg["model"]["horizon"])
    cond = torch.stack([
        torch.stack([torch.full((2,), -0.75), torch.zeros(2)]),
        torch.stack([torch.tensor([0.3, -0.1]), torch.tensor([0.45, 0.2])]),
    ])[:B]
    q0 = torch.randn(B, 32, 2) * 0.1
    q0[:, 0] = cond[:, 0]
    q0[:, -1] = cond[:, 1]
    curve = _codec().decode_controls(q0)
    occ = torch.zeros(B, 1, 256, 256)
    occ[:, :, 0:8, :] = 1.0                      # obstacle band
    occ[:, :, 120:136, 0:100] = 1.0
    cand_xy = torch.rand(B, M, H, 2) * 0.4 - 0.2
    cand_xy[:, :, 0] = cond[:, 0, None, :].expand(B, M, 2)
    cand_xy[:, :, -1] = cond[:, 1, None, :].expand(B, M, 2)
    mask = torch.ones(B, M, dtype=torch.bool)
    geo = torch.rand(B, M, 1280, 2) * 0.4 - 0.2
    geo[:, :, :, 0] = torch.linspace(-0.75, 0.4, 1280)[None, None]
    geo[:, :, :, 1] = torch.linspace(0.0, 0.3, 1280)[None, None]
    glen = torch.full((B, M), 1280, dtype=torch.long)
    return {
        "control_gt": q0, "pos": curve, "cond": cond, "occupancy": occ,
        "candidate_xy": cand_xy, "candidate_mask": mask,
        "candidate_geometry": geo, "candidate_geometry_lengths": glen,
        "topology_best": torch.zeros(B, dtype=torch.long),
        "has_candidate": mask.any(dim=1),
        "ellipse_shape4_gt": torch.tensor(
            [[[np.log(0.2), np.log(0.1), 1.0, 0.0]] * H] * B),
        "shape_valid": torch.ones(B, H, dtype=torch.bool),
    }


def test_5_6_7_forward_shapes_backward():
    cfg = load_config("configs/config.yaml")
    mcfg = dict(cfg["model"])
    mcfg["assert_shapes"] = True
    model = TrajSafePlanner(mcfg, cfg.get("ellipse_label"), cfg.get("bspline"))
    model.train()
    batch = _synthetic_batch(cfg, B=2)
    q_t = batch["control_gt"] + 0.3 * torch.randn_like(batch["control_gt"])
    assert tuple(q_t.shape) == (2, 32, 2)                  # test 6
    ab = torch.linspace(1.0, 0.2, 16)
    out = model.forward_all(q_t, batch["occupancy"], batch["cond"],
                            torch.tensor([3, 12]), ab[torch.tensor([3, 12])],
                            batch["candidate_xy"], batch["candidate_mask"],
                            batch["candidate_geometry"],
                            batch["candidate_geometry_lengths"],
                            select_index=batch["topology_best"])
    assert tuple(out["control"].shape) == (2, 32, 2)
    assert tuple(out["final"].shape) == (2, 128, 2)
    assert tuple(out["ellipse"]["center"].shape) == (2, 128, 2)   # test 5
    assert tuple(out["ellipse"]["shape4"].shape) == (2, 128, 4)
    assert torch.allclose(out["ellipse"]["progress"][0], model.fixed_progress)

    from src.losses.losses import (boundary_control_loss, control_smoothness_loss,
                                   control_x0_loss, topology_ce)
    ws, wg = model.boundary_decoder.weights(batch["control_gt"].shape[1])
    l = (control_x0_loss(out["q_raw_final"], batch["control_gt"])
         + control_x0_loss(out["q_coarse_raw"], batch["control_gt"])
         + control_smoothness_loss(out["q_raw_final"], batch["control_gt"])
         + boundary_control_loss(out["q_raw_final"], batch["control_gt"],
                                 batch["cond"], ws, wg)
         + topology_ce(out["topo"]["pi"], batch["topology_best"],
                       batch["has_candidate"])
         + out["ellipse"]["shape4"].pow(2).mean())
    l.backward()
    checked = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), "non-finite grad in %s" % name
        checked += 1
    assert checked > 50
    for name in ("traj_encoder", "traj_backbone", "head_p",
                 "skeleton_encoder", "match_block", "topology_head",
                 "path_feature_head", "safety_query_head",
                 "safety_cross_attention", "ellipse_geometry",
                 "ellipse_shape_head", "fusion_mlp",
                 "final_denoiser"):
        mod = model
        for part in name.split("."):
            mod = getattr(mod, part)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in mod.parameters()), "no gradient for %s" % name
    # the fixed boundary decoder is parameter-free and never in the checkpoint
    assert list(model.boundary_decoder.parameters()) == []
    assert "boundary_decoder.profile" not in model.state_dict()


def test_8_carla_y_axis():
    samples = _carla_samples(4)
    assert samples
    cell = 2.0 / 256.0
    for traj, start, goal, raw_occ in samples:
        raw = np.asarray(raw_occ)
        canon = np.flipud(raw).copy()
        # raw row 0 <=> y_local = +40 m <=> scene y = +1
        assert np.array_equal(canon[0], raw[-1])
        assert np.array_equal(canon[-1], raw[0])
        # the SAME scene point maps to row iy in canon and row 255-iy in raw
        iy = int(np.floor((start[1] + 1.0) / cell))
        ix = int(np.floor((start[0] + 1.0) / cell))
        assert 0 <= iy < 256 and 0 <= ix < 256
        assert canon[iy, ix] == 0, "start pixel is not free in canon"
        assert raw[255 - iy, ix] == 0, "start pixel is not free in raw"
        # the GT curve stays inside the canonical crop
        iy = np.floor((traj[:, 1] + 1.0) / cell).astype(int)
        ix = np.floor((traj[:, 0] + 1.0) / cell).astype(int)
        assert iy.min() >= 0 and iy.max() < 256
        assert ix.min() >= 0 and ix.max() < 256


def test_9_no_legacy_progress_in_main_chain():
    files = ["train.py", "sample.py", "evaluate.py",
             "src/models/trajsafe/planner.py", "src/models/trajsafe/heads.py",
             "src/models/trajsafe/fusion.py", "src/models/trajsafe/ellipse.py",
             "src/losses/losses.py", "src/diffusion/sampler.py",
             "src/datasets/carla_spline_dataset.py"]
    for rel in files:
        path = os.path.join(ROOT, rel)
        text = open(path, encoding="utf-8").read()
        for token in FORBIDDEN:
            assert token not in text, "%s still contains %s" % (rel, token)


def test_10_basis_matches_numpy():
    codec = _codec()
    knots = np.load(KNOTS)
    B = bspline_basis_matrix(knots, 32, 3, np.linspace(0.0, 1.0, 128))
    assert np.allclose(B, codec.basis.numpy(), atol=1e-12)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(B[0], np.eye(32)[0], atol=1e-12)
    assert np.allclose(B[-1], np.eye(32)[-1], atol=1e-12)


def test_11_alias_hard_endpoints_is_control_space():
    cfg = load_config("configs/config.yaml")
    model = TrajSafePlanner(cfg["model"], cfg.get("ellipse_label"),
                            cfg.get("bspline"))
    cond = torch.tensor([[[-0.75, 0.0], [0.25, 0.1]]])
    q = torch.randn(1, 32, 2)
    q = model.hard_endpoints(q, cond)
    assert tuple(q.shape) == (1, 32, 2)
    p = model.decode_controls(q)
    assert torch.allclose(p[0, 0], cond[0, 0], atol=1e-6)
    assert torch.allclose(p[0, -1], cond[0, 1], atol=1e-6)
