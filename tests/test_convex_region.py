import math

import torch

from src.geometry.convex_region import EllipseRegionBuilder, _halfspaces_are_bounded
from src.geometry.convex_region import halfspaces_to_vertices


def test_dense_local_border_points_produce_a_bounded_region():
    occupancy = torch.zeros(1, 1, 32, 32)
    center = torch.zeros(1, 1, 2)
    shape4 = torch.tensor(
        [[[math.log(0.12), math.log(0.07), 1.0, 0.0]]])
    builder = EllipseRegionBuilder(occupancy, {
        "safety_margin": 0.01,
        "obstacle_window_half": 0.35,
    })

    border = builder._local_border_points(torch.zeros(1, 2))[0]
    assert len(border) > 4
    assert torch.all(
        torch.isclose(border.abs().amax(dim=-1), torch.tensor(0.35), atol=1e-6)
    )

    # ABSOLUTE centre API: the builder never sees a trajectory any more.
    A, b, mask, valid = builder.build_from_ellipse(center, shape4)
    polygon = halfspaces_to_vertices(
        A[0, 0, mask[0, 0]].numpy(),
        b[0, 0, mask[0, 0]].numpy(),
        interior_point=torch.zeros(2).numpy(),
    )
    assert valid.item()

    assert polygon is not None
    assert len(polygon) >= 3
    assert abs(polygon).max() <= 0.35 + 1e-5


def test_three_faces_can_form_a_bounded_triangle():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    A = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)[None, None]
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    assert _halfspaces_are_bounded(A, mask).item()


def test_three_faces_in_a_semicircle_are_unbounded():
    angles = torch.tensor([0.0, math.pi / 4.0, math.pi])
    A = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)[None, None]
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    assert not _halfspaces_are_bounded(A, mask).item()


def test_center_outside_halfspaces_is_diagnostic_only():
    """``center_inside`` is REPORTED but no longer part of ``valid``.

    It used to invalidate the cell, and because ONE rejected base cell voids the
    WHOLE corridor (``invalid_base_region:<i>``), a route that hugs an obstacle
    (its ellipse centre closer to the wall than ``safety_margin``, so the cut
    face lands BEHIND that centre) produced an empty pack and an ALM that never
    ran.  Measured on the C=48 k=4 run, test_0342: 9 of 128 cells rejected, the
    corridor empty.  The flag is still computed and reported; it just does not
    invalidate the region any more.
    """
    occupancy = torch.zeros(1, 1, 32, 32)
    occupancy[0, 0, :, 16] = 1.0
    theta = 1.2
    center = torch.tensor([[[0.025, 0.0]]])
    shape4 = torch.tensor([[[
        math.log(0.3), math.log(0.03),
        math.cos(2.0 * theta), math.sin(2.0 * theta),
    ]]]).to(torch.float32)
    builder = EllipseRegionBuilder(occupancy, {
        "safety_margin": 0.02,
        "obstacle_window_half": 0.35,
    })

    A, b, mask, valid, diagnostics = builder.build_from_ellipse(
        center, shape4, return_diagnostics=True)

    assert not diagnostics["center_inside"].item()   # still reported ...
    assert valid.item()                              # ... but no longer fatal



def test_filtered_obstacle_square_must_be_fully_separated():
    # Centre .49 is beyond x<=.48, but the occupied square crosses it.
    builder = EllipseRegionBuilder(torch.zeros(1, 1, 64, 64),
                                   {'safety_margin': .02})
    points = torch.tensor([[[.5, 0.], [.49, .2]]])
    A, b, mask = builder._greedy_faces(
        torch.zeros(1, 2), torch.eye(2)[None], points,
        torch.ones(1, 2, dtype=torch.bool), point_half_extent=(1/64, 1/64))
    assert mask.sum() == 2
    min_projection = torch.einsum('sfd,snd->snf', A, points) - A.abs().sum(-1)[:, None]/64
    separated = (min_projection - b[:, None]).masked_fill(~mask[:, None], -torch.inf)
    assert (separated.amax(-1) >= -1e-6).all()


def test_selected_square_is_separated_even_with_zero_margin():
    builder = EllipseRegionBuilder(torch.zeros(1, 1, 32, 64),
                                   {'safety_margin': 0.})
    points = torch.tensor([[[.5, .2]]])
    A, b, mask = builder._greedy_faces(
        torch.zeros(1, 2), torch.eye(2)[None], points,
        torch.ones(1, 1, dtype=torch.bool), point_half_extent=(1/64, 1/32))
    min_projection = torch.einsum('sfd,snd->snf', A, points)
    min_projection -= (A.abs()*torch.tensor([1/64,1/32])).sum(-1)[:,None]
    assert ((min_projection-b[:,None])[mask[:,None]] >= -1e-6).all()
