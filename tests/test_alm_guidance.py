import torch

from src.diffusion.alm_guidance import alm_correct
from src.geometry.convex_corridor import EllipseRegionBuilder


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
    p = torch.zeros(2, 8, 2)
    p[:, :, 0] = torch.linspace(-0.8, 0.8, 8)
    e = torch.zeros(2, 8, 6)
    e[..., 2:4] = -2.0
    e[..., 4] = 1.0

    A, b, mask, valid = EllipseRegionBuilder(
        occ, {"max_faces": 12, "corridor_chunk_size": 4},
    )(p, e)

    assert A.shape == (2, 8, 12, 2)
    assert b.shape == (2, 8, 12)
    assert mask.shape == (2, 8, 12)
    assert valid.all()
    assert mask.sum(dim=-1).min() >= 4
    assert torch.allclose(A.norm(dim=-1)[mask], torch.ones_like(b[mask]), atol=1e-6)

    # A deliberately smaller cap cannot represent the central square from
    # every seed. Those truncated regions must be rejected, not treated as
    # occupancy-verified corridors.
    _, _, _, capped_valid = EllipseRegionBuilder(
        occ, {"max_faces": 8, "corridor_chunk_size": 4},
    )(p, e)
    assert (~capped_valid).any()


def test_region_is_keyed_by_its_own_physical_ellipse():
    occ = torch.zeros(1, 1, 32, 32)
    occ[:, :, 12:20, 12:20] = 1.0
    p1 = torch.tensor([[[-0.8, -0.5], [-0.5, -0.5], [-0.2, -0.5]]])
    p2 = p1.clone()
    p2[:, 1, 0] += 0.2
    e1 = torch.zeros(1, 3, 6)
    e1[..., 2:4] = -2.0
    e1[..., 4] = 1.0
    e2 = e1.clone()
    e2[:, 1, 0] -= 0.2
    builder = EllipseRegionBuilder(occ, {"max_faces": 20})

    A1, b1, m1, _ = builder(p1, e1)
    A2, b2, m2, _ = builder(p2, e2)

    # Moving p_1 while compensating delta-c_1 leaves its physical ellipse
    # unchanged, so its corresponding convex region must also be unchanged.
    assert torch.equal(m1[:, 1], m2[:, 1])
    assert torch.allclose(A1[:, 1], A2[:, 1])
    assert torch.allclose(b1[:, 1], b2[:, 1])
