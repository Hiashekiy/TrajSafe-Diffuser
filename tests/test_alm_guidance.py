import torch

from src.diffusion.alm_guidance import alm_correct
from src.geometry.convex_corridor import SegmentCorridorBuilder


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


def test_corridor_builder_batches_and_normalises_faces():
    occ = torch.zeros(2, 1, 32, 32)
    occ[:, :, 12:20, 12:20] = 1.0
    p = torch.zeros(2, 8, 2)
    p[:, :, 0] = torch.linspace(-0.8, 0.8, 8)
    e = torch.zeros(2, 8, 6)
    e[..., 2:4] = -2.0
    e[..., 4] = 1.0

    A, b, mask, valid = SegmentCorridorBuilder(
        occ, {"max_faces": 12, "corridor_chunk_size": 4},
    )(p, e)

    assert A.shape == (2, 7, 12, 2)
    assert b.shape == (2, 7, 12)
    assert mask.shape == (2, 7, 12)
    assert valid.all()
    assert mask.sum(dim=-1).min() >= 4
    assert torch.allclose(A.norm(dim=-1)[mask], torch.ones_like(b[mask]), atol=1e-6)

    # A deliberately smaller cap cannot represent the central square from
    # every seed. Those truncated regions must be rejected, not treated as
    # occupancy-verified corridors.
    _, _, _, capped_valid = SegmentCorridorBuilder(
        occ, {"max_faces": 8, "corridor_chunk_size": 4},
    )(p, e)
    assert (~capped_valid).any()
