"""V2 ellipse tests: fixed-centre IRIS labels and centre immovability."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from v2_utils import tiny_model

from src.geometry import ellipse_raster
from src.geometry.fixed_center_iris import (P_to_shape4, ellipse_is_inside_free,
                                            fixed_center_shape4, shape4_to_P,
                                            solve_fixed_center_iris)
from src.losses import v2_losses
from src.models.skeleton.ellipse_shape_head import raw_to_shape4
from src.models.skeleton.path_ops import abtheta_to_shape4, shape4_to_abtheta


def _corridor_occ(size=64, y0=24, y1=40):
    occ = np.ones((size, size), dtype=np.float32)
    occ[y0:y1, :] = 0.0
    return occ


def _ellipse_points(P, c, n=256):
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    unit = np.vstack([np.cos(ang), np.sin(ang)])
    return (np.asarray(P) @ unit).T + np.asarray(c)


def test_fixed_center_iris_does_not_move_center():
    """The label is a shape at a FIXED centre; the centre is never optimised."""
    occ = _corridor_occ()
    c = np.array([0.0, 0.05])
    s4, P, safe = fixed_center_shape4(occ, c)
    assert s4 is not None and s4.shape == (4,)      # no centre in the output
    assert safe
    assert ellipse_is_inside_free(P, c, occ)

    pts = _ellipse_points(P, c)
    assert np.allclose(pts.mean(axis=0), c, atol=1e-9)   # centred at c

    # every obstacle corner point must stay outside the ellipse
    from src.geometry.fixed_center_iris import scene_obstacle_points
    obs = scene_obstacle_points(occ, c)
    q = np.einsum("ij,jk,ik->i", obs - c, np.linalg.inv(P.T @ P), obs - c)
    assert q.min() >= 1.0 - 1e-3

    # asking for a different centre moves the ellipse, not the other way round
    c2 = c + np.array([0.1, 0.0])
    s4b, Pb, safe_b = fixed_center_shape4(occ, c2)
    assert safe_b
    assert np.allclose(_ellipse_points(Pb, c2).mean(axis=0), c2, atol=1e-9)
    assert not np.allclose(s4, s4b)
    assert ellipse_is_inside_free(Pb, c2, occ)


def test_solve_fixed_center_iris_uses_center_as_a_constant():
    """||P A_j|| + A_j^T c <= b_j with c constant: the ellipse fits the walls."""
    occ = _corridor_occ()
    c = np.array([0.0, 0.0])
    from src.geometry.fixed_center_iris import scene_obstacle_points
    obs = scene_obstacle_points(occ, c)
    P = solve_fixed_center_iris(obs, c)
    s4 = P_to_shape4(P)
    a, b = float(np.exp(s4[0])), float(np.exp(s4[1]))
    # the corridor is 16 cells tall = 0.5 scene units -> minor axis ~0.25,
    # while the major axis is limited only by the scene box
    assert 0.15 < b < 0.30
    assert 0.5 < a <= 1.0 + 1e-6


def test_ellipse_losses_take_center_and_shape_separately():
    """No loss may rebuild the centre from a prediction (V1 did)."""
    for fn in (v2_losses.ellipse_mask_losses, v2_losses.ellipse_shape_loss):
        body = inspect.getsource(fn)
        assert "e_pred" not in body
        assert "p_pred" not in body
        assert "shape4[..., :2]" not in body
        assert "center +" not in body
    sig = inspect.signature(v2_losses.ellipse_mask_losses)
    assert list(sig.parameters)[:2] == ["center", "shape4"]


def test_ellipse_losses_can_move_center_but_only_as_an_input():
    """The centre is an input: the loss sees it, the shape head cannot move it."""
    center = torch.zeros(1, 4, 2, requires_grad=True)
    shape4 = torch.zeros(1, 4, 4, requires_grad=True)
    occ = torch.zeros(1, 1, 64, 64)
    gt = torch.zeros(1, 4, 64, 64, dtype=torch.uint8)
    iou, safe, _, _ = v2_losses.ellipse_mask_losses(center, shape4, occ, gt)
    (iou + safe).backward()
    assert center.grad is not None and float(center.grad.abs().sum()) > 0
    assert shape4.grad is not None and float(shape4.grad.abs().sum()) > 0


def test_shape_head_cannot_move_the_center():
    """d(centre) / d(shape head params) == 0: the head has no centre output."""
    model = tiny_model(horizon=8)
    base = model.encode_trajectory(
        torch.randn(1, 8, 2), torch.zeros(1, 1, 64, 64), torch.zeros(1, 2, 2),
        torch.full((1,), 5), torch.full((1,), 0.5))
    path = torch.randn(1, 12, 2) * 0.3
    out = model.refine_with_path(base, path, torch.randn(1, 12, 32))
    # the shape head SHARES the spatial PE with the backbone, so only the
    # head's own layers are inspected here
    own = [(n, p) for n, p in model.shape.named_parameters()
           if not n.startswith("spatial_pe")]
    grads = torch.autograd.grad(out["ellipse_center"].sum(),
                                [p for _, p in own], allow_unused=True)
    assert all(g is None for g in grads), "the shape head moved the centre"
    # and the centre is exactly gamma_m(s), independent of the shape head
    from src.models.skeleton.path_ops import gather_path_points
    assert torch.allclose(out["ellipse_center"],
                          gather_path_points(path, out["progress"]))


def test_shape_codec_roundtrip():
    """[log a, log b, cos 2t, sin 2t] <-> (a, b, theta) with a >= b."""
    for a, b, th in [(0.3, 0.1, 0.4), (0.1, 0.3, -1.2), (0.2, 0.2, 0.0)]:
        s4 = abtheta_to_shape4(torch.tensor([a]), torch.tensor([b]),
                               torch.tensor([th]))
        aa, bb, tt = shape4_to_abtheta(s4)
        assert float(aa[0]) >= float(bb[0])
        assert float(aa[0]) == pytest.approx(max(a, b), rel=1e-6)
        assert float(bb[0]) == pytest.approx(min(a, b), rel=1e-6)
        assert abs(float(s4[0, 2]) ** 2 + float(s4[0, 3]) ** 2 - 1.0) < 1e-6


def test_shape_head_output_is_normalised_and_ordered():
    raw = torch.tensor([[[-1.0, 0.5, 3.0, 4.0], [0.2, 0.9, 0.0, 0.0]]])
    s4 = raw_to_shape4(raw)
    assert torch.all(s4[..., 0] >= s4[..., 1])
    assert torch.allclose((s4[..., 2:] ** 2).sum(-1), torch.ones(1, 2), atol=1e-6)
    assert float(s4[0, 0, 2]) == pytest.approx(0.6)
    assert float(s4[0, 0, 3]) == pytest.approx(0.8)
    # the degenerate direction falls back to (1, 0), never to a zero vector
    assert float(s4[0, 1, 2]) == 1.0 and float(s4[0, 1, 3]) == 0.0


def test_ellipse_mask_losses_zero_for_a_perfect_prediction():
    occ = torch.zeros(1, 1, 64, 64)
    center = torch.zeros(1, 3, 2)
    shape4 = torch.tensor([[[np.log(0.1), np.log(0.08), 1.0, 0.0]]]).expand(1, 3, 4)
    a, b, theta = shape4_to_abtheta(shape4)
    mask = ellipse_raster.ellipse_soft_mask(center, a, b, theta, 64, 10.0)
    gt = (mask * 255).round().to(torch.uint8)
    iou, safe, safe_mean, _ = v2_losses.ellipse_mask_losses(center, shape4, occ, gt)
    # the GT mask is uint8-quantised, so a perfect prediction is not bit exact,
    # and the soft-area normalisation carries the usual raster discretisation
    # error (~2% at 64x64 for a 3-cell radius ellipse); see V1's identical term
    assert float(iou) < 5e-3
    assert float(safe) < 0.05
    # a fully occupied map must be reported as clearly unsafe
    occ_blocked = torch.ones(1, 1, 64, 64)
    _, safe_blocked, _, _ = v2_losses.ellipse_mask_losses(
        center, shape4, occ_blocked, gt)
    assert float(safe_blocked) > 0.9


def test_ellipse_mask_losses_penalise_obstacle_overlap():
    occ = torch.zeros(1, 1, 64, 64)
    occ[:, :, :, 32:] = 1.0                        # right half is obstacle
    center = torch.zeros(1, 2, 2)
    small = torch.tensor([[[np.log(0.05), np.log(0.05), 1.0, 0.0]]]).expand(1, 2, 4)
    big = torch.tensor([[[np.log(0.5), np.log(0.5), 1.0, 0.0]]]).expand(1, 2, 4)
    gt = torch.zeros(1, 2, 64, 64, dtype=torch.uint8)
    _, safe_small, _, _ = v2_losses.ellipse_mask_losses(center, small, occ, gt)
    _, safe_big, _, _ = v2_losses.ellipse_mask_losses(center, big, occ, gt)
    assert float(safe_small) < float(safe_big)


def test_smoothness_loss_matches_v1():
    """L_smooth must be numerically identical to the frozen V1 implementation."""
    import train as v1_train
    torch.manual_seed(0)
    p_pred = torch.randn(4, 32, 2)
    p_gt = torch.randn(4, 32, 2)
    a = v2_losses.trajectory_smoothness_loss(p_pred, p_gt)
    b = v1_train.trajectory_smoothness_loss(p_pred, p_gt)
    assert float(a) == pytest.approx(float(b), rel=1e-9)
