import torch

from src.diffusion.alm_guidance import alm_correct
from src.geometry.convex_region import EllipseRegionBuilder


def test_alm_reduces_box_violation_and_preserves_endpoints():
    p = torch.tensor([[[0.0, 0.0], [1.2, 0.0], [1.2, 0.0], [0.0, 0.0]]])
    A = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0],
                        [0.0, 1.0], [0.0, -1.0]]] * 3])
    b = torch.full((1, 3, 4), 0.8)
    mask = torch.ones((1, 3, 4), dtype=torch.bool)
    valid = torch.ones((1, 3), dtype=torch.bool)

    corrected, _, stats = alm_correct(
        p, A, b, mask, valid, torch.zeros(1, 3), rho=1.0,
        step_size=0.1, inner_steps=5, collect_stats=True,
    )

    assert stats["violation_after"] < stats["violation_before"]
    assert torch.equal(corrected[:, 0], p[:, 0])
    assert torch.equal(corrected[:, -1], p[:, -1])


def test_feasible_trajectory_is_an_exact_fixed_point():
    p = torch.tensor([[[0.0, 0.0], [0.2, 0.1], [0.4, 0.1], [0.6, 0.0]]])
    A = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0],
                        [0.0, 1.0], [0.0, -1.0]]] * 3])
    b = torch.full((1, 3, 4), 0.9)
    mask = torch.ones((1, 3, 4), dtype=torch.bool)
    valid = torch.ones((1, 3), dtype=torch.bool)

    corrected, lam, stats = alm_correct(
        p, A, b, mask, valid, torch.zeros(1, 3), rho=5.0,
        step_size=0.03, inner_steps=4, proximity_weight=1.0,
        correction_smooth_weight=4.0, collect_stats=True,
    )

    assert torch.equal(corrected, p)
    assert torch.equal(lam, torch.zeros_like(lam))
    assert stats["raw_max_positive_rate"] == 0
    assert stats["mean_correction"] == 0


def test_physical_safety_gate_prevents_conservative_region_from_moving_point():
    p = torch.tensor([[[0.0, 0.0], [0.7, 0.0], [0.8, 0.0]]])
    A = torch.tensor([[[[1.0, 0.0]]] * 2])
    b = torch.full((1, 2, 1), 0.5)
    mask = torch.ones((1, 2, 1), dtype=torch.bool)
    valid = torch.ones((1, 2), dtype=torch.bool)
    physically_unsafe = torch.zeros((1, 2), dtype=torch.bool)

    corrected, _, _ = alm_correct(
        p, A, b, mask, valid, torch.zeros(1, 2), rho=5.0,
        enforce_mask=physically_unsafe, inner_steps=4,
    )
    assert torch.equal(corrected, p)


def test_corridor_builder_batches_and_normalises_faces():
    occ = torch.zeros(2, 1, 32, 32)
    occ[:, :, 12:20, 12:20] = 1.0
    # absolute centres on a straight line (no p0 / delta-centre semantics)
    center = torch.zeros(2, 8, 2)
    center[:, :, 0] = torch.linspace(-0.8, 0.8, 8)
    shape4 = torch.zeros(2, 8, 4)
    shape4[..., 0:2] = -2.0
    shape4[..., 2] = 1.0

    A, b, mask, valid = EllipseRegionBuilder(
        occ, {"corridor_chunk_size": 4},
    ).build_from_ellipse(center, shape4)

    assert A.shape[:2] == (2, 8)
    assert A.shape[2] == b.shape[2]
    assert A.shape[-1] == 2
    assert mask.shape == b.shape
    assert valid.all()
    assert mask.sum(dim=-1).min() >= 4
    assert torch.allclose(A.norm(dim=-1)[mask], torch.ones_like(b[mask]), atol=1e-6)

    # There is no fixed face budget: filtering always consumes every real and
    # local-border obstacle point.
    assert valid.all()


def test_region_is_keyed_by_its_absolute_center():
    """The region follows the ABSOLUTE centre that was passed in.

    The old contract ("move p0 and compensate the delta and the region stays")
    is gone: there is no trajectory argument at all any more.
    """
    occ = torch.zeros(1, 1, 32, 32)
    occ[:, :, 12:20, 12:20] = 1.0
    c1 = torch.tensor([[[-0.8, -0.5], [-0.5, -0.5], [-0.2, -0.5]]])
    c2 = c1.clone()
    c2[:, 1, 0] += 0.2
    shape4 = torch.zeros(1, 3, 4)
    shape4[..., 0:2] = -2.0
    shape4[..., 2] = 1.0
    builder = EllipseRegionBuilder(occ)

    A1, b1, m1, _ = builder.build_from_ellipse(c1, shape4)
    A2, b2, m2, _ = builder.build_from_ellipse(c2, shape4)

    # Moving the centre DOES move the region (the local map moved with it).
    assert not torch.allclose(A1[:, 1], A2[:, 1])
    # ... and calling it with a trajectory is no longer possible at all.
    try:
        builder(c1, shape4)
    except NotImplementedError:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("the delta-centre entry point must be gone")
