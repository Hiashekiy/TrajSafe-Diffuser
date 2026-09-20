"""Report section 41 (Tests B, C, D): overlap ratio and the gap bridge."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.geometry.convex_region import EllipseRegionBuilder
from src.geometry.safety_corridor import (build_bridge_region,
                                          build_safety_corridor,
                                          convex_polygon_intersection,
                                          overlap_ratio, polygon_area,
                                          progress_alignment_stats)

MIN_OVERLAP = 0.10


def _square(cx, cy, half):
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]],
                    dtype=np.float64)


def _builder(res=128):
    return EllipseRegionBuilder(torch.zeros(1, 1, res, res),
                                {"safety_margin": 0.01,
                                 "obstacle_window_half": 0.35})


def _tiny_shape4(h=2, log_a=np.log(0.005), log_b=np.log(0.005)):
    out = np.zeros((h, 4))
    out[:, 0] = log_a
    out[:, 1] = log_b
    out[:, 2] = 1.0
    return out


# --------------------------------------------------------------- Test B
def test_B_polygon_area_of_a_unit_square():
    assert abs(polygon_area(_square(0.0, 0.0, 0.5)) - 1.0) < 1e-12
    assert polygon_area(np.zeros((0, 2))) == 0.0


def test_B_identical_polygons_have_ratio_one():
    a = _square(0.0, 0.0, 0.5)
    assert abs(overlap_ratio(a, a.copy()) - 1.0) < 1e-9


def test_B_disjoint_polygons_have_ratio_zero():
    a = _square(-1.0, 0.0, 0.3)
    b = _square(1.0, 0.0, 0.3)
    assert overlap_ratio(a, b) == 0.0
    assert len(convex_polygon_intersection(a, b)) == 0


def test_B_partial_overlap_matches_hand_computation():
    a = _square(0.0, 0.0, 0.5)          # area 1
    b = _square(0.5, 0.0, 0.5)          # area 1, overlap area 0.5
    assert abs(polygon_area(convex_polygon_intersection(a, b)) - 0.5) < 1e-9
    assert abs(overlap_ratio(a, b) - 0.5) < 1e-9


def test_B_containment_gives_ratio_one():
    outer = _square(0.0, 0.0, 1.0)
    inner = _square(0.1, 0.0, 0.2)
    assert abs(overlap_ratio(outer, inner) - 1.0) < 1e-9
    assert abs(overlap_ratio(inner, outer) - 1.0) < 1e-9


# --------------------------------------------------------------- Test C
def test_C_bridge_is_a_skeleton_point_seeded_region():
    builder = _builder()
    centers = np.array([[-0.5, 0.0], [0.5, 0.0]])
    gamma = np.array([[-0.5, 0.0], [0.0, -0.15], [0.5, 0.0]])
    corridor = build_safety_corridor(
        builder, centers, _tiny_shape4(2), np.array([0.0, 1.0]),
        gamma=gamma, gamma_lengths=len(gamma),
        config={"min_overlap_ratio": MIN_OVERLAP,
                "bridge": {"enabled": True, "max_bridge_per_gap": 1}})

    assert corridor.valid, corridor.failure_reason
    assert corridor.base_cell_count == 2
    assert corridor.bridge_cell_count == 1
    assert [c.source for c in corridor.cells] == ["network", "bridge",
                                                  "network"]

    bridge = corridor.cells[1]
    # the seed is Gamma((s0+s1)/2), NOT the euclidean midpoint
    assert np.allclose(bridge.center, [0.0, -0.15], atol=1e-6)
    assert np.linalg.norm(bridge.center - np.array([0.0, 0.0])) > 0.1

    assert len(corridor.overlap_ratio) == 2
    assert min(corridor.overlap_ratio) >= MIN_OVERLAP
    assert corridor.bridge_gaps and corridor.bridge_gaps[0]["gap"] == 0


def test_C_bridge_region_uses_the_isotropic_metric():
    builder = _builder()
    region = build_bridge_region(builder, np.array([0.1, -0.2]))
    assert region["valid"]
    assert region["center_inside"]
    # identity metric -> Euclidean point-seeded region, still a bounded box
    assert region["polygon"] is not None

    A, b, mask, valid, diag = builder.build_from_metric(
        torch.tensor([[[0.1, -0.2]]]),
        torch.eye(2)[None, None].expand(1, 1, 2, 2).contiguous(),
        return_diagnostics=True)
    assert bool(valid[0, 0])
    assert np.allclose(
        A[0, 0][mask[0, 0]].numpy(), region["A"], atol=1e-6)


def test_C_no_bridge_when_the_base_overlap_is_sufficient():
    builder = _builder()
    centers = np.array([[-0.1, 0.0], [0.1, 0.0]])
    gamma = np.array([[-0.1, 0.0], [0.1, 0.0]])
    corridor = build_safety_corridor(
        builder, centers, _tiny_shape4(2), np.array([0.0, 1.0]),
        gamma=gamma, gamma_lengths=len(gamma),
        config={"min_overlap_ratio": MIN_OVERLAP,
                "bridge": {"enabled": True, "max_bridge_per_gap": 1}})
    assert corridor.valid
    assert corridor.bridge_cell_count == 0
    assert len(corridor.cells) == 2
    assert min(corridor.overlap_ratio) >= MIN_OVERLAP


# --------------------------------------------------------------- Test D
def test_D_unclosable_gap_fails_without_recursive_bridging():
    builder = _builder()
    centers = np.array([[-0.5, 0.0], [0.5, 0.0]])
    # the transition seed is far above the two base cells: even a point-seeded
    # region there cannot overlap both of them.
    gamma = np.array([[-0.5, 0.0], [0.0, 0.75], [0.5, 0.0]])
    corridor = build_safety_corridor(
        builder, centers, _tiny_shape4(2), np.array([0.0, 1.0]),
        gamma=gamma, gamma_lengths=len(gamma),
        config={"min_overlap_ratio": MIN_OVERLAP,
                "bridge": {"enabled": True, "max_bridge_per_gap": 1}})

    assert not corridor.valid
    assert corridor.failure_reason is not None
    assert corridor.failure_reason.startswith("bridge_failed")
    # V1 inserts at most one bridge per gap and never recurses
    assert corridor.bridge_cell_count == 0
    assert corridor.cells == []


def test_D_bridge_disabled_reports_the_overlap_failure():
    builder = _builder()
    centers = np.array([[-0.5, 0.0], [0.5, 0.0]])
    gamma = np.array([[-0.5, 0.0], [0.0, -0.15], [0.5, 0.0]])
    corridor = build_safety_corridor(
        builder, centers, _tiny_shape4(2), np.array([0.0, 1.0]),
        gamma=gamma, gamma_lengths=len(gamma),
        config={"min_overlap_ratio": MIN_OVERLAP,
                "bridge": {"enabled": False, "max_bridge_per_gap": 1}})
    assert not corridor.valid
    assert corridor.failure_reason == "overlap_below_threshold:0"


def test_D_invalid_base_region_aborts_the_corridor():
    builder = _builder()
    centers = np.array([[0.0, 0.0]])
    # NaN shape produces a non-finite metric -> invalid base region
    shape4 = _tiny_shape4(1)
    shape4[0, 0] = np.nan
    corridor = build_safety_corridor(
        builder, centers, shape4, np.array([0.0]),
        gamma=np.array([[0.0, 0.0]]), gamma_lengths=1,
        config={"min_overlap_ratio": MIN_OVERLAP,
                "bridge": {"enabled": True, "max_bridge_per_gap": 1}})
    assert not corridor.valid
    assert corridor.failure_reason is not None


# --------------------------------------------------- V1 progress diagnostics
def test_progress_alignment_stats_reports_scene_and_meters():
    p = np.zeros((4, 2))
    c = np.zeros((4, 2))
    c[:, 0] = 0.01
    stats = progress_alignment_stats(p, c)
    assert pytest.approx(stats["progress_alignment_rmse"], rel=1e-6) == 0.01
    assert pytest.approx(stats["progress_alignment_rmse_m"], rel=1e-6) == 0.4
