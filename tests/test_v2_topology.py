"""V2 topology selector tests (spec section 37)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from v2_utils import tiny_batch, tiny_model, straight_path


def test_topology_masked_softmax():
    """Invalid candidate slots must receive exactly zero probability."""
    model = tiny_model()
    b = tiny_batch(B=3, M=4)
    b["mask"][0] = torch.tensor([True, True, False, False])
    b["mask"][1] = torch.tensor([True, False, False, False])
    b["mask"][2] = torch.tensor([False, False, False, False])
    base = model.encode_trajectory(
        torch.randn(3, 8, 2), b["occ"], b["cond"],
        torch.full((3,), 5), torch.full((3,), 0.5))
    topo = model.score_candidates(base, b["cand"], b["mask"], b["lengths"])
    pi = topo["pi"]
    assert torch.allclose(pi.sum(dim=-1)[:2], torch.ones(2), atol=1e-6)
    assert float(pi[0, 2:].sum()) == 0.0
    assert float(pi[1, 1:].sum()) == 0.0
    # a sample with no valid candidate gets an all-zero distribution (the loss
    # masks it out and the sampler falls back to the base trajectory)
    assert float(pi[2].sum()) == 0.0
    assert torch.isneginf(topo["logits"][0, 2])
    assert not torch.isnan(pi).any()


def test_topology_loss_no_grad_to_trajectory_when_detached():
    """With detach_trajectory_feature the topology loss cannot move the backbone."""
    model = tiny_model(detach=True)
    b = tiny_batch()
    p_t = torch.randn(2, 8, 2)
    base = model.encode_trajectory(p_t, b["occ"], b["cond"],
                                   torch.full((2,), 5), torch.full((2,), 0.5))
    topo = model.score_candidates(base, b["cand"], b["mask"], b["lengths"])
    q = torch.tensor([[0.6, 0.4, 0.0], [1.0, 0.0, 0.0]])
    from src.losses.v2_losses import topology_soft_ce
    loss = topology_soft_ce(topo["pi"], q, torch.ones(2, dtype=torch.bool))
    grad_traj = torch.autograd.grad(loss, base["traj_feat"], allow_unused=True,
                                    retain_graph=True)[0]
    assert grad_traj is None, "the topology loss reached the trajectory features"
    loss.backward()
    # the selector parameters are trained ...
    assert model.selector.score[0].weight.grad is not None
    assert float(model.selector.score[0].weight.grad.abs().sum()) > 0
    # ... but nothing reaches the trajectory backbone
    for name, param in model.blocks.named_parameters():
        assert param.grad is None or float(param.grad.abs().sum()) == 0.0, name
    assert p_t.grad is None


def test_topology_loss_can_reach_backbone_when_not_detached():
    """Ablation hook: with detach=False the gradient does reach the backbone."""
    model = tiny_model(detach=False)
    b = tiny_batch()
    base = model.encode_trajectory(torch.randn(2, 8, 2), b["occ"], b["cond"],
                                   torch.full((2,), 5), torch.full((2,), 0.5))
    topo = model.score_candidates(base, b["cand"], b["mask"], b["lengths"])
    grad = torch.autograd.grad(topo["logits"][b["mask"]].sum(),
                               base["traj_feat"], allow_unused=True)[0]
    assert grad is not None and float(grad.abs().sum()) > 0


def test_chamfer_distance_matches_bruteforce():
    from src.models.skeleton.topology_selector import chamfer_mean_distance
    pred = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    path = torch.tensor([[[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]])
    d = chamfer_mean_distance(pred, path)
    assert d.shape == (1, 1)
    assert float(d[0, 0]) == 0.0
    pred2 = pred + torch.tensor([0.0, 0.5])          # perpendicular offset
    assert float(chamfer_mean_distance(pred2, path)[0, 0]) == pytest.approx(0.5)


def test_soft_target_is_used_not_one_hot():
    """A soft target with two comparable routes must not be pushed to one-hot."""
    from src.losses.v2_losses import topology_soft_ce
    pi = torch.tensor([[0.5, 0.5, 0.0]])
    q = torch.tensor([[0.52, 0.45, 0.03]])
    loss = topology_soft_ce(pi, q, torch.ones(1, dtype=torch.bool))
    expected = -float((q * torch.log(pi.clamp_min(1e-12))).sum())
    assert float(loss) == pytest.approx(expected, rel=1e-6)


def test_path_encoder_shapes():
    model = tiny_model(horizon=8)
    cand = torch.randn(2, 3, 8, 5)
    traj = torch.randn(2, 8, 32)
    tokens, pooled = model.selector.path_encoder(cand, traj)
    assert tokens.shape == (2, 3, 8, 32)
    assert pooled.shape == (2, 3, 32)


def test_selector_ignores_padded_slots_in_gradient():
    """Padded slots are multiplied out by the softmax, not merely ignored."""
    model = tiny_model()
    b = tiny_batch(B=1, M=2)
    b["mask"][0, 1] = False
    b["cand"][0, 1] = 1e3                       # absurd padded values
    base = model.encode_trajectory(torch.randn(1, 8, 2), b["occ"], b["cond"],
                                   torch.full((1,), 5), torch.full((1,), 0.5))
    topo = model.score_candidates(base, b["cand"], b["mask"], b["lengths"])
    assert float(topo["pi"][0, 1]) == 0.0
    assert float(topo["pi"][0, 0]) == 1.0
