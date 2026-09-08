import torch

from src.geometry.ellipse_center_repair import (
    EllipseCenterRepair,
    reencode_ellipse_centers,
)
from src.geometry.polytope_projection import project_point_to_polytope_2d


class _BoxBuilder:
    """Small deterministic builder for testing repair state transitions."""

    max_faces = 4

    def __init__(self):
        self.maps = torch.zeros(1, 1, 8, 8)

    def points_are_free(self, points):
        return ((points[..., 0] <= 0.500001) &
                (points.abs() <= 1.0).all(dim=-1))

    def build_from_centers(self, centers, ellipse):
        batch, horizon = centers.shape[:2]
        face = centers.new_tensor(((1.0, 0.0), (-1.0, 0.0),
                                   (0.0, 1.0), (0.0, -1.0)))
        A = face.view(1, 1, 4, 2).expand(batch, horizon, -1, -1).clone()
        b = centers.new_full((batch, horizon, 4), 0.5)
        mask = torch.ones(batch, horizon, 4, dtype=torch.bool)
        valid = ellipse[..., 2] < 5.0
        return A, b, mask, valid


class _CenteredBoxBuilder(_BoxBuilder):
    """Every raw center has a valid local box, even outside the prior box."""

    def points_are_free(self, points):
        return (points.abs() <= 1.0).all(dim=-1)

    def build_from_centers(self, centers, ellipse):
        batch, horizon = centers.shape[:2]
        face = centers.new_tensor(((1.0, 0.0), (-1.0, 0.0),
                                   (0.0, 1.0), (0.0, -1.0)))
        A = face.view(1, 1, 4, 2).expand(batch, horizon, -1, -1).clone()
        half = 0.25
        b = torch.stack((centers[..., 0] + half,
                         -centers[..., 0] + half,
                         centers[..., 1] + half,
                         -centers[..., 1] + half), dim=-1)
        mask = torch.ones(batch, horizon, 4, dtype=torch.bool)
        valid = torch.ones(batch, horizon, dtype=torch.bool)
        return A, b, mask, valid


def _inputs(center_x, invalid_index=None):
    center_x = torch.tensor(center_x, dtype=torch.float32)
    trajectory = torch.zeros(1, len(center_x), 2)
    ellipse = torch.zeros(1, len(center_x), 6)
    ellipse[0, :, 0] = center_x
    ellipse[..., 4] = 1.0
    if invalid_index is not None:
        ellipse[0, invalid_index, 2] = 10.0
    return trajectory, ellipse


def test_projection_returns_exact_closest_point_and_satisfies_polytope():
    point = torch.tensor([[0.8, 0.2]])
    A = torch.tensor([[[1.0, 0.0], [-1.0, 0.0],
                       [0.0, 1.0], [0.0, -1.0]]])
    b = torch.full((1, 4), 0.5)
    projected, success = project_point_to_polytope_2d(
        point, A, b, torch.ones_like(b, dtype=torch.bool))

    assert success.item()
    assert torch.allclose(projected, torch.tensor([[0.5, 0.2]]))
    assert ((A * projected[:, None]).sum(dim=-1) <= b + 1e-6).all()


def test_first_center_is_anchored_and_good_centers_are_unchanged():
    trajectory, ellipse = _inputs([0.3, 0.2, 0.4])
    start = torch.tensor([[-0.25, 0.1]])
    result = EllipseCenterRepair(_BoxBuilder())(trajectory, ellipse, start)

    assert torch.equal(result.centers[:, 0], start)
    assert torch.equal(result.centers[0, 1:], ellipse[0, 1:, :2])


def test_wall_center_is_projected_and_multiple_bad_centers_continue():
    trajectory, ellipse = _inputs([0.0, 0.8, 0.9, 0.7])
    result = EllipseCenterRepair(_BoxBuilder())(
        trajectory, ellipse, torch.tensor([[0.0, 0.0]]))

    assert torch.allclose(result.centers[0, 1:, 0], torch.full((3,), 0.5))
    violation = ((result.A * result.centers[:, :, None]).sum(dim=-1) - result.b)
    assert (violation.masked_fill(~result.face_mask, -torch.inf)[result.valid]
            <= 1e-6).all()


def test_free_valid_raw_center_outside_propagation_region_is_still_projected():
    trajectory, ellipse = _inputs([0.0, 0.8])
    builder = _CenteredBoxBuilder()
    raw_A, raw_b, raw_mask, raw_valid = builder.build_from_centers(
        ellipse[..., :2], ellipse)
    assert builder.points_are_free(ellipse[..., :2]).all()
    assert raw_valid.all()

    result = EllipseCenterRepair(builder)(
        trajectory, ellipse, torch.tensor([[0.0, 0.0]]))

    assert not torch.equal(result.centers[0, 1], ellipse[0, 1, :2])
    assert torch.allclose(result.centers[0, 1], torch.tensor([0.25, 0.0]))
    assert result.valid[0, 1]
    assert result.stats["adjacent_region_overlap_rate"] == 1
    # The raw region really was valid; it was not used to bypass propagation.
    assert ((raw_A[:, 1] * ellipse[:, 1, None, :2]).sum(dim=-1)
            <= raw_b[:, 1] + 1e-6).all()


def test_failed_rebuild_stays_invalid_but_propagation_region_is_reused():
    trajectory, ellipse = _inputs([0.0, 0.8, 0.9], invalid_index=1)
    result = EllipseCenterRepair(_BoxBuilder())(
        trajectory, ellipse, torch.tensor([[0.0, 0.0]]))

    assert not result.valid[0, 1]
    assert not result.face_mask[0, 1].any()
    assert result.valid[0, 2]
    assert torch.allclose(result.centers[0, 2], torch.tensor([0.5, 0.0]))
    assert result.stats["propagation_reuse_rate"] > 0


def test_first_region_failure_uses_start_cell_only_for_propagation():
    trajectory, ellipse = _inputs([0.0, 0.8], invalid_index=0)
    result = EllipseCenterRepair(_BoxBuilder())(
        trajectory, ellipse, torch.tensor([[0.0, 0.0]]))

    assert not result.valid[0, 0]
    assert not result.face_mask[0, 0].any()
    assert result.propagation_valid.item()
    assert result.valid[0, 1]


def test_reencoding_after_alm_preserves_physical_centers_and_shape():
    ellipse = torch.randn(2, 5, 6)
    centers = torch.randn(2, 5, 2)
    guided_trajectory = torch.randn(2, 5, 2)
    encoded = reencode_ellipse_centers(ellipse, centers, guided_trajectory)

    assert torch.allclose(guided_trajectory + encoded[..., :2], centers)
    assert torch.equal(encoded[..., 2:], ellipse[..., 2:])
