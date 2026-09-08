import torch

from train import ellipse_center_safety_loss


def x_coordinate_sdf(resolution=32):
    coordinate = ((torch.arange(resolution, dtype=torch.float32) + 0.5)
                  * (2.0 / resolution) - 1.0)
    return coordinate[None, None, None, :].expand(1, 1, resolution, -1).clone()


def test_center_safety_loss_pushes_center_toward_free_space():
    p = torch.zeros(1, 4, 2)
    e = torch.zeros(1, 4, 6, requires_grad=True)
    with torch.no_grad():
        e[..., 0] = -0.25

    loss, mean, cvar = ellipse_center_safety_loss(
        p, e, x_coordinate_sdf(), margin=0.02,
        cvar_fraction=0.5, cvar_weight=1.0,
    )
    loss.backward()

    assert loss.item() > 0
    assert mean.item() > 0
    assert cvar.item() > 0
    assert torch.all(e.grad[..., 0] < 0)
    assert torch.allclose(e.grad[..., 1], torch.zeros_like(e.grad[..., 1]))


def test_center_safety_loss_is_zero_for_clear_in_bounds_centers():
    p = torch.zeros(1, 4, 2)
    e = torch.zeros(1, 4, 6)
    e[..., 0] = 0.5

    loss, mean, cvar = ellipse_center_safety_loss(
        p, e, x_coordinate_sdf(), margin=0.02,
        cvar_fraction=0.5, cvar_weight=1.0,
    )

    assert loss.item() == 0.0
    assert mean.item() == 0.0
    assert cvar.item() == 0.0


def test_center_safety_loss_penalizes_out_of_bounds_centers():
    p = torch.zeros(1, 4, 2)
    e = torch.zeros(1, 4, 6)
    e[..., 0] = 1.1

    loss, _, _ = ellipse_center_safety_loss(
        p, e, torch.ones(1, 1, 32, 32), margin=0.02,
        cvar_fraction=0.5, cvar_weight=1.0,
    )

    assert loss.item() > 0.0
