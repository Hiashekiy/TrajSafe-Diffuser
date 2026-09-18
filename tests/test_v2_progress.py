"""V2 progress head / ellipse-centre tests (spec section 37)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from v2_utils import MAPS_DIR, SKELETON_DIR, tiny_model

from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.skeleton_paths import CandidateConfig, generate_candidates
from src.models.skeleton.path_ops import gather_path_points, path_arclength

RING_START = (-0.52, -0.45)
RING_GOAL = (0.52, 0.42)


def _ring_map(size=64, outer=(10, 54), inner=(20, 44)):
    occ = np.ones((size, size), dtype=np.float32)
    occ[outer[0]:outer[1], outer[0]:outer[1]] = 0.0
    occ[inner[0]:inner[1], inner[0]:inner[1]] = 1.0
    return occ


@pytest.fixture(scope="module")
def ring_candidates():
    graph = build_skeleton_graph(_ring_map(), safety_dilation_cells=1)
    cfg = CandidateConfig(num_candidates=2, candidate_points=32)
    cands = generate_candidates(graph, RING_START, RING_GOAL, cfg)
    return graph, cands


def test_progress_starts_zero_ends_one():
    model = tiny_model(horizon=16)
    traj = torch.randn(2, 16, 32)
    path = torch.randn(2, 12, 32)
    s, fused = model.progress(traj, path)
    assert s.shape == (2, 16)
    assert torch.allclose(s[:, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(s[:, -1], torch.ones(2), atol=1e-5)
    assert fused.shape == (2, 16, 32)


def test_progress_strictly_monotonic():
    model = tiny_model(horizon=32)
    s, _ = model.progress(torch.randn(4, 32, 32), torch.randn(4, 10, 32))
    d = s[:, 1:] - s[:, :-1]
    assert torch.all(d > 0)
    assert torch.all(s >= 0.0) and torch.all(s <= 1.0 + 1e-6)


def test_progress_is_bounded_between_zero_and_one():
    """softplus gaps + normalisation keep s in [0,1] for any input magnitude."""
    model = tiny_model(horizon=8)
    s, _ = model.progress(torch.randn(2, 8, 32) * 50.0,
                          torch.randn(2, 5, 32) * 50.0)
    assert torch.all(s >= 0.0) and torch.all(s <= 1.0)
    assert torch.allclose(s[:, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(s[:, -1], torch.ones(2), atol=1e-5)


def test_gather_path_points_lies_on_the_polyline():
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]]])
    s = torch.linspace(0.0, 1.0, 33)[None]
    pts = gather_path_points(coords, s)
    assert pts.shape == (1, 33, 2)
    on_h = torch.isclose(pts[..., 1], torch.zeros_like(pts[..., 1]), atol=1e-6)
    on_v = torch.isclose(pts[..., 0], torch.ones_like(pts[..., 0]), atol=1e-6)
    assert torch.all(on_h | on_v)
    assert torch.allclose(pts[0, 0], coords[0, 0])
    assert torch.allclose(pts[0, -1], coords[0, -1])
    # project back onto the polyline: zero distance and perfectly uniform arc
    # length (the chord between two samples is shorter across the corner, so the
    # chord length itself is NOT the arc length)
    from src.geometry.skeleton_paths import nearest_arclength
    s_back, dist = nearest_arclength(pts[0].numpy(), coords[0].numpy())
    assert np.all(dist < 1e-6)
    assert np.allclose(s_back, np.linspace(0.0, 1.0, 33), atol=1e-6)


def test_gather_path_points_is_differentiable_in_s():
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]]])
    s = torch.tensor([[0.1, 0.5, 0.9]], requires_grad=True)
    pts = gather_path_points(coords, s)
    pts.sum().backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0


def test_centers_always_on_selected_path(ring_candidates):
    """c_i = gamma_m(s_i) is a point of the committed polyline for ANY s."""
    graph, cands = ring_candidates
    model = tiny_model(horizon=8)
    k = int(cands.valid_index()[0])
    coords = torch.from_numpy(cands.coords[k])[None]           # [1,L,2]
    s = torch.rand(1, 8)
    pts = gather_path_points(coords, s)[0].detach().numpy()
    poly = cands.coords[k]
    from src.geometry.skeleton_paths import nearest_arclength
    _, dist = nearest_arclength(pts, poly)
    assert np.all(dist < 1e-6), "an ellipse centre left the selected path"


def test_centers_always_free(ring_candidates):
    """Every ellipse centre produced by the head is inside free space."""
    graph, cands = ring_candidates
    model = tiny_model(horizon=8)
    base = model.encode_trajectory(
        torch.randn(1, 8, 2), torch.zeros(1, 1, 64, 64), torch.zeros(1, 2, 2),
        torch.full((1,), 5), torch.full((1,), 0.5))
    for k in cands.valid_index():
        path = torch.from_numpy(cands.coords[k])[None]
        out = model.refine_with_path(base, path,
                                     torch.randn(1, path.shape[1], 32))
        center = out["ellipse_center"][0].detach().numpy()
        px = np.rint(graph.scene_to_pixel(center)).astype(int)
        assert graph.free[px[:, 1], px[:, 0]].all()


def test_refine_outputs_shapes_and_endpoints(ring_candidates):
    graph, cands = ring_candidates
    model = tiny_model(horizon=8)
    base = model.encode_trajectory(
        torch.randn(2, 8, 2), torch.zeros(2, 1, 64, 64), torch.zeros(2, 2, 2),
        torch.full((2,), 5), torch.full((2,), 0.5))
    path = torch.from_numpy(cands.coords[0])[None].expand(2, -1, -1).contiguous()
    out = model.refine_with_path(base, path, torch.randn(2, path.shape[1], 32))
    assert out["x0_p"].shape == (2, 8, 2)
    assert out["progress"].shape == (2, 8)
    assert out["ellipse_center"].shape == (2, 8, 2)
    assert out["ellipse_shape4"].shape == (2, 8, 4)
    assert torch.allclose(out["ellipse_center"][:, 0],
                          path[:, 0], atol=1e-5)
    assert torch.allclose(out["ellipse_center"][:, -1],
                          path[:, -1], atol=1e-5)
