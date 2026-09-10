import torch

from train import ellipse_center_safety_loss
from src.models.joint import JointPlanner


def x_coordinate_sdf(resolution=32):
    coordinate = ((torch.arange(resolution, dtype=torch.float32) + 0.5)
                  * (2.0 / resolution) - 1.0)
    return coordinate[None, None, None, :].expand(
        1, 1, resolution, -1
    ).clone()


def test_center_loss_moves_offset_toward_free_space_without_trajectory_gradient():
    p = torch.zeros(1, 4, 2, requires_grad=True)
    e = torch.zeros(1, 4, 6, requires_grad=True)
    with torch.no_grad():
        e[..., 0] = -0.25

    loss = ellipse_center_safety_loss(
        p, e, x_coordinate_sdf(), margin=0.01, tau=0.02,
    )
    loss.backward()

    assert p.grad is None
    assert torch.all(e.grad[..., 0] < 0)
    assert torch.allclose(e.grad[..., 1:], torch.zeros_like(e.grad[..., 1:]))


def test_model_center_safety_path_updates_only_ellipse_head_center_rows():
    model = JointPlanner({
        "horizon": 4,
        "d_model": 8,
        "num_heads": 2,
        "joint_blocks": 1,
        "ffn_dim": 16,
        "map_res": 16,
        "global_mem_res": 4,
        "dropout": 0.0,
    })
    with torch.no_grad():
        model.head_e.weight.zero_()
        model.head_e.bias.zero_()
        model.head_e.bias[0] = -0.25

    p_t = torch.zeros(1, 4, 2)
    e_t = torch.zeros(1, 4, 6)
    occ = torch.zeros(1, 1, 16, 16)
    cond = torch.zeros(1, 2, 2)
    out = model(p_t, e_t, occ, cond, torch.zeros(1, dtype=torch.long),
                torch.ones(1))
    loss = ellipse_center_safety_loss(
        out["x0_p"], out["x0_e_center_safe"], x_coordinate_sdf(),
        margin=0.01, tau=0.02,
    )
    loss.backward()

    parameters_with_grad = {
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    }
    assert parameters_with_grad == {"head_e.weight", "head_e.bias"}
    assert model.head_e.weight.grad[:2].abs().sum() > 0
    assert torch.equal(model.head_e.weight.grad[2:],
                       torch.zeros_like(model.head_e.weight.grad[2:]))
    assert torch.equal(model.head_e.bias.grad[2:],
                       torch.zeros_like(model.head_e.bias.grad[2:]))


def test_center_loss_has_bounded_clearance_gradient_and_handles_out_of_bounds():
    p = torch.zeros(1, 2, 2)
    e = torch.zeros(1, 2, 6, requires_grad=True)
    with torch.no_grad():
        e[0, 0, 0] = -0.25
        e[0, 1, 0] = 1.1

    loss = ellipse_center_safety_loss(
        p, e, x_coordinate_sdf(), margin=0.01, tau=0.02,
    )
    loss.backward()

    assert loss.item() > 0.0
    # Both SDF and box-boundary clearance are 1-Lipschitz in this fixture.
    assert e.grad[..., :2].norm(dim=-1).max().item() <= 0.5 + 1e-6


def test_center_loss_is_negligible_for_clear_in_bounds_centers():
    p = torch.zeros(1, 4, 2)
    e = torch.zeros(1, 4, 6)
    e[..., 0] = 0.5

    loss = ellipse_center_safety_loss(
        p, e, x_coordinate_sdf(), margin=0.01, tau=0.02,
    )

    assert loss.item() < 1e-10
