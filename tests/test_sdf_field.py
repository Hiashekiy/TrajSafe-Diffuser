"""Contract tests for ``src/geometry/sdf_field.py`` (第4章 4.1.2).

These pin down the three things the four 4.2 baselines depend on and that are
easy to get silently wrong:

1. ORIENTATION / ISO-LEVEL -- ``d_occ < 0`` must coincide with the repo's own
   collision test (``occ`` bilinearly ``> 0.5``), otherwise the optimiser would
   be solving a different problem from the one that is scored.
2. GRADIENT -- the analytic direction field must match finite differences of
   the interpolated value field, and must NOT vanish deep inside a large
   obstacle (that is the whole reason the direction field exists).
3. MAP BOUNDARY -- ``d_box`` positive inside / negative outside, and the
   combined field must flag a point outside ``[-1,1]^2`` as a violation even
   when the occupancy there is free.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.geometry.sdf_field import OCCUPIED_LEVEL, OccupancySDF, build_occupancy_sdf

RES = 256
CELL = 2.0 / RES


def _grid(obstacles):
    """``obstacles``: list of (row_slice, col_slice) -> occ [1,1,RES,RES]."""
    occ = torch.zeros(1, 1, RES, RES)
    for rs, cs in obstacles:
        occ[0, 0, rs, cs] = 1.0
    return occ


def _blob_grid():
    """A big obstacle blob plus a thin wall: exercises interior + adjacency."""
    return _grid([(slice(40, 200), slice(60, 180)),      # 160x120 blob
                  (slice(210, 214), slice(20, 240))])    # thin wall


def test_scene_to_cell_mapping_matches_the_repo_convention():
    """Cell centre (i,j) <-> scene ((j+0.5)*cell-1, (i+0.5)*cell-1)."""
    sdf = build_occupancy_sdf(_blob_grid())
    # the grid_sample convention: point -> pixel index (x+1)*RES/2 - 0.5
    for col, row in ((0, 0), (10, 200), (255, 255), (128, 63)):
        x = -1.0 + (col + 0.5) * CELL
        y = -1.0 + (row + 0.5) * CELL
        assert (x + 1.0) * RES / 2.0 - 0.5 == pytest.approx(float(col))
        assert (y + 1.0) * RES / 2.0 - 0.5 == pytest.approx(float(row))
        assert sdf.width == sdf.height == RES
        assert sdf.cell == pytest.approx(CELL)


def test_sign_agrees_with_the_occupancy_iso_level():
    """``d_occ < 0`` <=> bilinear occupancy > 0.5, on a dense random sample."""
    occ = _blob_grid()
    sdf = build_occupancy_sdf(occ, scene_to_meter=80.0)
    torch.manual_seed(0)
    pts = torch.rand(1, 20000, 2) * 2.0 - 1.0
    value = sdf.query(pts, with_grad=False)
    bilinear = torch.nn.functional.grid_sample(
        occ, pts.unsqueeze(1), mode="bilinear", padding_mode="border",
        align_corners=False)[:, 0, 0, :]
    assert torch.equal(value <= 0.0, bilinear > OCCUPIED_LEVEL)


def test_adjacent_free_cell_has_half_cell_clearance():
    """The obstacle surface sits half a cell from a free cell's centre."""
    sdf = build_occupancy_sdf(_grid([(slice(100, 120), slice(100, 120))]))
    row, col = 99, 110                       # free cell just above the blob
    x = -1.0 + (col + 0.5) * CELL
    y = -1.0 + (row + 0.5) * CELL
    value = sdf.query(torch.tensor([[[x, y]]]), with_grad=False)
    assert float(value[0, 0]) == pytest.approx(0.5 * CELL, rel=1e-6)
    row_c, col_c = 110, 110                  # centre of the blob
    xc = -1.0 + (col_c + 0.5) * CELL
    yc = -1.0 + (row_c + 0.5) * CELL
    inside = sdf.query(torch.tensor([[[xc, yc]]]), with_grad=False)
    assert float(inside[0, 0]) < 0.0


def test_gradient_points_away_from_the_obstacle_and_inside_it_towards_free():
    """The readable specification of the direction field."""
    sdf = build_occupancy_sdf(_grid([(slice(100, 140), slice(100, 140))]))
    # a free point to the LEFT of the blob -> gradient points further left (-x)
    left = torch.tensor([[[-1.0 + 90.5 * CELL, -1.0 + 120.5 * CELL]]])
    assert float(sdf.query(left, with_grad=False)[0, 0]) > 0.0
    _, grad = sdf.query(left)
    assert float(grad[0, 0, 0]) == pytest.approx(-1.0, abs=1e-5)
    assert float(grad[0, 0, 1]) == pytest.approx(0.0, abs=1e-5)
    # a point INSIDE the blob -> gradient points towards the nearest free cell
    inside = torch.tensor([[[-1.0 + 110.5 * CELL, -1.0 + 110.5 * CELL]]])
    assert float(sdf.query(inside, with_grad=False)[0, 0]) < 0.0
    _, grad_in = sdf.query(inside)
    assert float(grad_in.norm()) == pytest.approx(1.0, abs=1e-5)
    # ascending the field must raise the SDF (the property the optimisers use)
    step = 0.02 * grad_in
    assert float(sdf.query(inside + step, with_grad=False)[0, 0]) > \
        float(sdf.query(inside, with_grad=False)[0, 0])


def test_gradient_matches_finite_differences_where_the_field_is_smooth():
    """Analytic direction == FD direction in the mean, and ascent never fails.

    On the medial axis -- and on cells where the nearest-obstacle assignment
    switches inside the interpolation stencil -- the FD of the bilinear
    interpolant is not the gradient of the distance function, so ``min`` is not
    asserted.  What IS asserted: the two agree almost everywhere, and moving
    along the analytic gradient never decreases the SDF.
    """
    sdf = build_occupancy_sdf(_blob_grid())
    torch.manual_seed(1)
    pts = torch.rand(1, 4000, 2) * 1.6 - 0.8
    value, grad = sdf.query(pts)
    eps = 1e-4
    fd = torch.zeros_like(pts)
    for axis in (0, 1):
        step = torch.zeros_like(pts)
        step[..., axis] = eps
        plus = sdf.query(pts + step, with_grad=False)
        minus = sdf.query(pts - step, with_grad=False)
        fd[..., axis] = (plus - minus) / (2.0 * eps)
    cos = torch.nn.functional.cosine_similarity(grad[0], fd[0], dim=-1)
    # ~1 % of random points sit on the medial axis / on an assignment switch,
    # where the FD of the interpolant is meaningless; everywhere else the two
    # agree to <1 degree.  A sign error would give mean cos = -1, so this is a
    # sharp contract even with the relaxed tail thresholds.
    assert float(cos.mean()) > 0.98
    assert float((cos > 0.9).to(torch.float32).mean()) > 0.95
    # ascent: the direction field always points towards larger d_occ
    moved = sdf.query(pts + 0.01 * grad, with_grad=False)
    assert float((moved >= value - 1e-6).to(torch.float32).mean()) > 0.99
    assert float(grad.norm(dim=-1).min()) == pytest.approx(1.0, abs=1e-5)


def test_gradient_does_not_vanish_inside_a_large_obstacle():
    """The exact distance-function direction, unlike the interpolated value."""
    sdf = build_occupancy_sdf(_grid([(slice(20, 240), slice(20, 240))]))
    centre = torch.tensor([[[-1.0 + 130.5 * CELL, -1.0 + 130.5 * CELL]]])
    assert float(sdf.query(centre, with_grad=False)[0, 0]) < -0.5   # deep inside
    _, grad = sdf.query(centre)
    assert float(grad.norm()) == pytest.approx(1.0, abs=1e-5)


def test_box_field_and_combined_field_flag_leaving_the_map():
    sdf = build_occupancy_sdf(_blob_grid(), map_margin=0.0)
    inside = torch.tensor([[[0.0, 0.0]]])
    outside = torch.tensor([[[1.4, 0.0]]])
    assert float(sdf.box_field(inside)[0, 0]) == pytest.approx(1.0)
    assert float(sdf.box_field(outside)[0, 0]) == pytest.approx(-0.4)
    # occupancy is free out there, so only the combined field sees the problem
    assert float(sdf.query(outside, with_grad=False)[0, 0]) > 0.0
    assert float(sdf.query(outside, with_boundary=True, with_grad=False)[0, 0]) < 0.0
    # and the combined gradient points back into the map there
    _, grad = sdf.query(outside, with_boundary=True)
    assert float(grad[0, 0, 0]) < 0.0


def test_stats_follow_the_guide_formulas():
    """r = max(d_safe - d_occ), V = relu(r), D_min = min(d_occ), safe ratio."""
    sdf = build_occupancy_sdf(_blob_grid(), scene_to_meter=80.0)
    pts = torch.tensor([[[-0.9, -0.9], [0.9, 0.9]]])       # both free
    stats = sdf.stats(pts, d_safe=0.0)
    value = sdf.query(pts, with_grad=False)
    assert float(stats["max_residual"][0]) == pytest.approx(float(-value.min()))
    assert float(stats["min_distance"][0]) == pytest.approx(float(value.min()))
    assert float(stats["safe_point_rate"][0]) == pytest.approx(1.0)
    assert bool(stats["feasible"][0])
    # a colliding curve: residual > 0, violation = residual, not feasible
    bad = torch.tensor([[[-1.0 + 110.5 * CELL, -1.0 + 110.5 * CELL]]])
    bad_stats = sdf.stats(bad, d_safe=0.0)
    assert float(bad_stats["max_residual"][0]) > 0.0
    assert float(bad_stats["violation"][0]) == pytest.approx(
        float(bad_stats["max_residual"][0]))
    assert not bool(bad_stats["feasible"][0])
    assert float(bad_stats["safe_point_rate"][0]) == 0.0


def test_to_metres_uses_the_configured_scale():
    sdf = build_occupancy_sdf(_blob_grid(), scene_to_meter=80.0)
    assert sdf.cell_m == pytest.approx(0.625)
    value = torch.tensor([0.5])
    assert float(sdf.to_metres(value)[0]) == pytest.approx(40.0)
    assert sdf.scene_to_meter == 80.0
    assert repr(sdf).startswith("OccupancySDF(")


def test_batch_dimension_is_checked():
    sdf = build_occupancy_sdf(_blob_grid())
    with pytest.raises(ValueError):
        sdf.query(torch.zeros(2, 8, 2))
    with pytest.raises(ValueError):
        OccupancySDF(torch.zeros(1, RES, RES))
