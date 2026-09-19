"""V3 model / sampler / loss behaviour tests (spec section 32).

Items covered here (the geometry ones live in tests/test_v3_geometry.py):

    1  timestep enters every denoising block through AdaLN
    2  coarse and final trajectory share the SAME head_p parameters
    3  progress is strictly monotone with s_1 = 0 and s_H = 1
    4  progress gets gradient from L_align without any progress label
    5  safety loss has non-zero gradient w.r.t. a / b / theta
    6  area loss pushes the ellipse to grow
    7  safety and area back-propagate together
    8  an invalid candidate is never selected
    9  every reverse step re-scores the topology
   10  no commit_t / committed cache anywhere in the V3 path
   11  the sub-sampled DDIM last step returns x0 exactly
   12  a no-candidate row works end to end
   13  different P_t may select different skeletons (and the opposite
       assumption - "same seed must pick the same skeleton" - is NOT asserted)
"""
from __future__ import annotations

import inspect
import re

import pytest
import torch

from v3_utils import tiny_batch, tiny_model

from src.diffusion import sampler_v3
from src.diffusion.schedule import NoiseSchedule
from src.losses import v3_losses
from src.models.skeleton_v3 import SkeletonPlannerV3


def _forward(model, b, select_index=None):
    return model.forward_all(b["pos"], b["occ"], b["cond"], b["t"], b["ab"],
                             b["features"], b["mask"], b["lengths"],
                             b["geom"], b["geom_len"],
                             select_index=select_index)


# --------------------------------------------------------------------- 1
def test_timestep_enters_every_block_through_adaln():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    loss = out["final"].sum() + out["coarse"].sum()
    loss.backward()
    adalns = []
    for name, module in model.named_modules():
        if type(module).__name__ == "AdaLN":
            adalns.append((name, module))
    assert len(adalns) >= (model.traj_blocks + model.fusion_blocks + 3)
    for name, module in adalns:
        grad = module.mod.weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0.0, name


def test_time_embedding_changes_the_prediction():
    model = tiny_model()
    # AdaLN is zero-initialised (V1 convention), so h_t has no effect until its
    # modulation is non-trivial; randomise it to probe the live path.
    for module in model.modules():
        if type(module).__name__ == "AdaLN":
            torch.nn.init.normal_(module.mod.weight, std=0.1)
            torch.nn.init.normal_(module.mod.bias, std=0.1)
    b = tiny_batch()
    base_a = model.encode_trajectory(b["pos"], b["occ"], b["cond"],
                                     torch.zeros(2, dtype=torch.long), b["ab"])
    base_b = model.encode_trajectory(b["pos"], b["occ"], b["cond"],
                                     torch.full((2,), 9, dtype=torch.long), b["ab"])
    assert not torch.allclose(base_a["traj_feat"], base_b["traj_feat"])


# --------------------------------------------------------------------- 2
def test_coarse_and_final_share_the_same_head():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    before = model.head_p.weight.detach().clone()
    with torch.no_grad():
        model.head_p.weight.mul_(0.0).add_(1.0)
    out2 = _forward(model, b)
    assert not torch.allclose(out["coarse"], out2["coarse"])
    assert not torch.allclose(out["final"], out2["final"])
    with torch.no_grad():
        model.head_p.weight.copy_(before)
    assert sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear)
               and _.out_features == 2) >= 1
    assert model.final_trajectory.__func__ is model.final_trajectory.__func__
    # both decodes must go through the very same module object
    src = inspect.getsource(SkeletonPlannerV3)
    assert src.count("self.head_p(") == 2
    assert "head_p_base" not in src and "head_p_refine" not in src


# --------------------------------------------------------------------- 3
def test_progress_is_monotone_with_fixed_endpoints():
    model = tiny_model(horizon=16)
    b = tiny_batch(H=16)
    out = _forward(model, b)
    s = out["ellipse"]["progress"]
    assert torch.allclose(s[:, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(s[:, -1], torch.ones(2), atol=1e-5)
    assert torch.all(s[:, 1:] - s[:, :-1] > 0)
    assert torch.all(s >= 0.0) and torch.all(s <= 1.0 + 1e-6)


def test_progress_head_is_monotone_for_any_input():
    model = tiny_model(horizon=8)
    s, _ = model.progress(torch.randn(3, 8, 32) * 30.0,
                          torch.randn(3, 5, 32) * 30.0,
                          torch.randn(3, 32) * 30.0)
    assert torch.all(s[:, 1:] > s[:, :-1])
    assert torch.allclose(s[:, 0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(s[:, -1], torch.ones(3), atol=1e-5)


# --------------------------------------------------------------------- 4
def test_align_loss_gives_gradient_without_progress_labels():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    loss = v3_losses.align_loss(out["ellipse"]["center"], b["pos"],
                                torch.ones(2, dtype=torch.bool))
    model.zero_grad()
    loss.backward()
    g = model.progress.mlp[0].weight.grad
    assert g is not None and float(g.abs().sum()) > 0
    # the ellipse head must NOT be involved: c_i = gamma_m(s_i) does not depend
    # on it, which is exactly why L_align cannot move the shape head.
    assert model.ellipse.mlp[0].weight.grad is None


# --------------------------------------------------------------------- 5/6/7
def test_safety_loss_has_gradient_on_axes_and_angle():
    center = torch.zeros(1, 4, 2)
    a = torch.full((1, 4), 0.2, requires_grad=True)
    bb = torch.full((1, 4), 0.1, requires_grad=True)
    theta = torch.zeros(1, 4, requires_grad=True)
    occ = torch.zeros(1, 1, 64, 64)
    occ[:, :, :, 32:] = 1.0
    loss, mean, cvar = v3_losses.ellipse_safety_loss(center, a, bb, theta, occ)
    loss.backward()
    for t in (a, bb, theta):
        assert t.grad is not None and float(t.grad.abs().sum()) > 0
    assert float(mean) > 0 and float(cvar) > 0


def test_area_loss_pushes_the_ellipse_to_grow():
    a = torch.full((1, 4), 0.1, requires_grad=True)
    b = torch.full((1, 4), 0.05, requires_grad=True)
    loss = v3_losses.ellipse_area_loss(a, b, a_max=0.8, b_max=0.3)
    loss.backward()
    before = float((a * b).mean())
    with torch.no_grad():
        a += -0.05 * a.grad
        b += -0.05 * b.grad
    assert float((a * b).mean()) > before


def test_area_loss_normalisation_and_mask():
    """a_max * b_max normalisation, and exactly 0 when nothing is valid."""
    a = torch.full((2, 4), 0.8)
    b = torch.full((2, 4), 0.30)
    # a maximal ellipse is a ZERO loss (the old a_max^2 normaliser gave 0.625)
    assert float(v3_losses.ellipse_area_loss(a, b, 0.8, 0.30)) < 1e-6
    assert float(v3_losses.ellipse_area_loss(a * 0.5, b * 0.5, 0.8, 0.30)) > 0.5
    # an all-invalid batch (and a mixed one) must not invent a loss
    empty = torch.zeros(2, dtype=torch.bool)
    mixed = torch.tensor([True, False])
    small = torch.full((2, 4), 0.1)
    assert float(v3_losses.ellipse_area_loss(small, small, 0.8, 0.30, empty)) == 0.0
    only_first = float(v3_losses.ellipse_area_loss(small, small, 0.8, 0.30, mixed))
    assert abs(only_first - float(v3_losses.ellipse_area_loss(
        small[:1], small[:1], 0.8, 0.30))) < 1e-6


def test_masked_mean_normalises_by_entries_not_samples():
    """A [B] mask over a [B, H] tensor must divide by B*H, not by B.

    The bug made L_align / L_gap H times too large, which turned the ellipse
    centre alignment into ~98% of the total objective.
    """
    B, H = 4, 128
    values = torch.full((B, H), 0.5)
    mask = torch.tensor([True, True, False, False])
    assert abs(float(v3_losses._masked_mean(values, mask)) - 0.5) < 1e-6
    # 1-D stays what it always was
    assert abs(float(v3_losses._masked_mean(torch.full((B,), 0.5), mask)) - 0.5) < 1e-6
    # an all-false mask is 0, not a division blow-up
    assert float(v3_losses._masked_mean(values, torch.zeros(B, dtype=torch.bool))) == 0.0
    # align_loss with a constant offset 0.1 is 0.5 * 0.1^2 (SmoothL1, beta = 1)
    centers = torch.zeros(B, H, 2)
    gt = torch.full((B, H, 2), 0.1)
    assert abs(float(v3_losses.align_loss(centers, gt, mask)) - 0.005) < 1e-6


def test_safety_and_area_backpropagate_together():
    model = tiny_model()
    b = tiny_batch()
    out = _forward(model, b)
    ell = out["ellipse"]
    occ = b["occ"]
    safe, _, _ = v3_losses.ellipse_safety_loss(ell["center"], ell["a"], ell["b"],
                                               ell["theta"], occ)
    area = v3_losses.ellipse_area_loss(ell["a"], ell["b"], model.ellipse.a_max)
    model.zero_grad()
    (safe + 0.05 * area).backward()
    for name, p in model.ellipse.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert float(safe) >= 0.0 and float(area) >= 0.0


# --------------------------------------------------------------------- 8
def test_invalid_candidate_is_never_selected():
    model = tiny_model()
    b = tiny_batch(B=4, M=3)
    b["mask"][:] = False
    b["mask"][:, 1] = True                      # only slot 1 is valid
    out = _forward(model, b)
    pi = out["topo"]["pi"]
    assert torch.allclose(pi.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert float(pi[:, 0].abs().sum()) == 0.0
    assert float(pi[:, 2].abs().sum()) == 0.0
    assert bool((out["selected_idx"] == 1).all())
    assert torch.isinf(out["topo"]["logits"][0, 0])
    assert torch.isinf(out["topo"]["logits"][0, 2])


def test_argmax_respects_lengths_and_geometry():
    """The selector really consumes the explicit chamfer distance d_m."""
    model = tiny_model()
    b = tiny_batch(B=1, M=2, H=8, L=8, G=16)
    b["mask"][0, :] = True
    d_model = model.d_model

    class _PickByDistance(torch.nn.Module):
        def forward(self, feat):
            return -feat[..., 2 * d_model:2 * d_model + 1]

    model.selector.score = _PickByDistance()
    # isolate the selector: the coarse decode is the identity on P_t, which is
    # exactly the quantity the selector matches against the candidate features.
    model.coarse_trajectory = lambda base, cond: model.hard_endpoints(b["pos"], cond)
    for cand, expect in ((0, 0), (1, 1)):
        b["pos"] = b["features"][0, cand, :8, :2].clone()[None]
        b["pos"][:, 0] = b["cond"][:, 0]
        b["pos"][:, -1] = b["cond"][:, 1]
        out = _forward(model, b)
        assert int(out["selected_idx"][0]) == expect


# --------------------------------------------------------------------- 9/10
def test_every_reverse_step_rescores_the_topology():
    model = tiny_model()
    b = tiny_batch(B=1, M=2)
    calls = {"n": 0}
    orig = model.score_candidates

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    model.score_candidates = spy
    out = sampler_v3.sample_v3(model, NoiseSchedule(16), b["cond"], b["occ"],
                               b["features"], b["mask"], b["lengths"],
                               b["geom"], b["geom_len"], device="cpu",
                               seed=0, return_trace=True)
    assert calls["n"] == len(out["trace"]) == 16
    idx = [int(x["selected_idx"][0]) for x in out["trace"]]
    assert len(idx) == 16


def test_sampler_has_no_commit_state():
    src = inspect.getsource(sampler_v3.sample_v3)
    for forbidden in ("commit_t", "committed", "selected_path", "selected_feat",
                      "mlp_e(", "E6"):
        assert forbidden not in src, forbidden
    model_src = inspect.getsource(SkeletonPlannerV3)
    for forbidden in ("commit", "tc", "refine_blocks", "head_p_refine"):
        assert forbidden not in model_src, forbidden
    assert re.search(r"\bcommit", src) is None


def test_sampler_selection_can_change_between_steps():
    """Nothing caches the selection: the recorded pi/idx are per step."""
    model = tiny_model()
    b = tiny_batch(B=1, M=2)
    b["mask"][0, :] = True
    out = sampler_v3.sample_v3(model, NoiseSchedule(16), b["cond"], b["occ"],
                               b["features"], b["mask"], b["lengths"], b["geom"],
                               b["geom_len"], device="cpu", seed=1,
                               return_trace=True)
    assert len(out["trace"]) == 16
    for step in out["trace"]:
        assert step["pi"].shape == (1, 2)
        assert step["coarse"].shape == (1, model.horizon, 2)
        assert step["final"].shape == (1, model.horizon, 2)


# --------------------------------------------------------------------- 11
def test_subsampled_schedule_ends_with_the_clean_transition():
    """Section 21: the sub-sampled schedule must ALSO end with 0 -> -1.

    Otherwise the last scheduled index (t = 5 for steps = 4) hands its own x0 to
    DDIM, the network never runs at t = 0, and P_0 is not the clean prediction.
    """
    model = tiny_model()
    b = tiny_batch(B=2, M=2)
    assert sampler_v3.pick_times(16, 4) == [0, 5, 10, 15]
    assert sampler_v3.pick_times(16, None) is None
    for steps, want in ((None, 16), (4, 4), (2, 2), (8, 8)):
        out = sampler_v3.sample_v3(model, NoiseSchedule(16), b["cond"], b["occ"],
                                   b["features"], b["mask"], b["lengths"],
                                   b["geom"], b["geom_len"], device="cpu",
                                   steps=steps, seed=0, return_trace=True)
        trace = out["trace"]
        assert len(trace) == want, steps
        # every step but the last is a real DDIM transition (s_t >= 0) ...
        assert all(step["s"] >= 0 for step in trace[:-1]), steps
        # ... and the last one is the explicit clean transition t = 0 -> -1
        assert trace[-1]["t"] == 0 and trace[-1]["s"] == -1, steps
        assert torch.allclose(out["p"], trace[-1]["final"], atol=1e-6), steps


def test_subsampled_schedule_evaluates_the_network_at_every_listed_time():
    model = tiny_model()
    b = tiny_batch(B=1, M=2)
    seen = []
    orig = model.forward_all

    def spy(p_t, occ, cond, t, ab, *a, **k):
        seen.append(int(t[0]))
        return orig(p_t, occ, cond, t, ab, *a, **k)

    model.forward_all = spy
    sampler_v3.sample_v3(model, NoiseSchedule(16), b["cond"], b["occ"],
                         b["features"], b["mask"], b["lengths"], b["geom"],
                         b["geom_len"], device="cpu", steps=4, seed=0)
    assert seen == [15, 10, 5, 0], seen


# --------------------------------------------------------------------- 12
def test_no_candidate_row_works_end_to_end():
    model = tiny_model()
    b = tiny_batch(B=3, M=3)
    b["mask"][:] = False
    b["geom_len"][:] = 0
    out = _forward(model, b)
    assert torch.isfinite(out["final"]).all()
    assert torch.isfinite(out["coarse"]).all()
    assert torch.isfinite(out["ellipse"]["center"]).all()
    assert torch.isfinite(out["ellipse"]["a"]).all()
    # ... and it really degenerates to the plain trajectory diffusion
    assert torch.allclose(out["final"], out["coarse"], atol=1e-7)
    has = b["mask"].any(dim=1)
    safe, _, _ = v3_losses.ellipse_safety_loss(
        out["ellipse"]["center"], out["ellipse"]["a"], out["ellipse"]["b"],
        out["ellipse"]["theta"], b["occ"], sample_mask=has)
    topo = v3_losses.topology_ce(out["topo"]["pi"], torch.zeros(3, dtype=torch.long), has)
    area = v3_losses.ellipse_area_loss(out["ellipse"]["a"], out["ellipse"]["b"],
                                       model.ellipse.a_max, model.ellipse.b_max,
                                       has)
    assert float(safe) == 0.0 and float(topo) == 0.0 and float(area) == 0.0
    assert torch.isfinite(safe) and torch.isfinite(topo) and torch.isfinite(area)
    # the sampler must survive the same rows
    res = sampler_v3.sample_v3(model, NoiseSchedule(16), b["cond"], b["occ"],
                               b["features"], b["mask"], b["lengths"], b["geom"],
                               b["geom_len"], device="cpu", seed=0)
    assert torch.isfinite(res["p"]).all()
    assert not bool(res["has_candidate"].any())


def test_no_candidate_bypass_is_per_row():
    """A mixed batch: only the candidate-free rows fall back to the coarse x0."""
    model = tiny_model()
    b = tiny_batch(B=3, M=3)
    b["mask"][:] = False
    b["mask"][0, 0] = True
    b["geom_len"][:] = 0
    b["geom_len"][0, 0] = 16
    out = _forward(model, b)
    assert torch.allclose(out["final"][1:], out["coarse"][1:], atol=1e-7)
    assert not torch.allclose(out["final"][0], out["coarse"][0])


# --------------------------------------------------------------------- 13
def test_different_pt_can_select_different_skeletons():
    model = tiny_model()
    b = tiny_batch(B=1, M=2, H=8, L=8, G=16)
    b["mask"][0, :] = True
    d_model = model.d_model

    class _PickByDistance(torch.nn.Module):
        def forward(self, feat):
            return -feat[..., 2 * d_model:2 * d_model + 1]

    model.selector.score = _PickByDistance()
    # isolate the selector: the coarse decode is the identity on P_t, which is
    # precisely the quantity the selector is supposed to match against.
    model.coarse_trajectory = lambda base, cond: model.hard_endpoints(b["pos"], cond)
    picks = []
    for cand in (0, 1):
        b["pos"] = b["features"][0, cand, :8, :2].clone()[None]
        b["pos"][:, 0] = b["cond"][:, 0]
        b["pos"][:, -1] = b["cond"][:, -1]
        out = _forward(model, b)
        picks.append(int(out["selected_idx"][0]))
    assert picks == [0, 1], picks


def test_lengths_feature_is_used():
    """L_m is an input of the score MLP (it cannot be dropped silently)."""
    model = tiny_model()
    assert model.selector.score[0].in_features == 3 * model.d_model + 2

def test_ellipse_axes_are_bounded_and_ordered():
    """b_min <= b <= a <= a_max must hold for ANY head output (section 16)."""
    model = tiny_model(b_min=0.02, b_max=0.30, a_max=0.80)
    b = tiny_batch()
    out = _forward(model, b)
    ell = out["ellipse"]
    assert float(ell["b"].min()) >= 0.02 - 1e-6
    assert float(ell["b"].max()) <= 0.30 + 1e-6
    assert float(ell["a"].max()) <= 0.80 + 1e-6
    assert bool((ell["a"] >= ell["b"] - 1e-6).all())

    class _Const(torch.nn.Module):
        def __init__(self, value, zero_dir=False):
            super().__init__()
            self.value = float(value)
            self.zero_dir = bool(zero_dir)

        def forward(self, x):
            out = torch.full((*x.shape[:-1], 4), self.value, device=x.device,
                             dtype=x.dtype)
            if self.zero_dir:
                out[..., 2:4] = 0.0        # exact (u, v) = (0, 0)
            return out

    for value in (-80.0, 80.0):
        model.ellipse.mlp = _Const(value)
        ell = _forward(model, b)["ellipse"]
        assert float(ell["b"].min()) >= 0.02 - 1e-6, value
        assert float(ell["b"].max()) <= 0.30 + 1e-6, value
        assert float(ell["a"].min()) >= 0.02 - 1e-6, value
        assert float(ell["a"].max()) <= 0.80 + 1e-6, value
        assert bool((ell["a"] >= ell["b"] - 1e-6).all()), value
        # the direction must stay a unit vector even for a degenerate raw output
        unit = (ell["shape4"][..., 2] ** 2 + ell["shape4"][..., 3] ** 2)
        assert torch.allclose(unit, torch.ones_like(unit), atol=1e-6)


@pytest.mark.parametrize("value,fallback", [(0.0, True), (-0.0, True),
                                            (1e-7, True), (1e-4, False)])
def test_ellipse_degenerate_direction_is_finite_and_unit(value, fallback):
    """raw (u, v) = (0, 0) exactly: no NaN gradient, unit direction, theta = 0."""
    model = tiny_model()
    b = tiny_batch()

    class _ZeroDir(torch.nn.Module):
        def forward(self, x):
            out = torch.full((*x.shape[:-1], 4), float(value), device=x.device,
                             dtype=x.dtype)
            out[..., 2:4] = float(value)
            return out

    model.ellipse.mlp = _ZeroDir()
    out = _forward(model, b)
    ell = out["ellipse"]
    assert torch.isfinite(ell["a"]).all() and torch.isfinite(ell["b"]).all()
    assert torch.isfinite(ell["theta"]).all()
    unit = ell["shape4"][..., 2] ** 2 + ell["shape4"][..., 3] ** 2
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-6)
    if fallback:
        # the zero direction is replaced by the constant (1, 0)
        assert torch.allclose(ell["shape4"][..., 2], torch.ones_like(unit),
                              atol=1e-6)
        assert torch.allclose(ell["shape4"][..., 3], torch.zeros_like(unit),
                              atol=1e-6)
        assert torch.allclose(ell["theta"], torch.zeros_like(unit), atol=1e-6)
    # ... and the backward pass through the direction must not produce NaN
    model.zero_grad()
    loss = ell["theta"].sum() + ell["shape4"].sum() + out["final"].sum()
    loss.backward()
    grads = [p.grad for p in model.ellipse.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_ellipse_tokens_use_both_type_embeddings():
    """Trajectory and ellipse streams must not share one type embedding."""
    model = tiny_model()
    b = tiny_batch()
    with torch.no_grad():
        model.traj_type.weight[0].fill_(0.0)
        model.traj_type.weight[1].fill_(0.0)
    base = model.encode_trajectory(b["pos"], b["occ"], b["cond"], b["t"], b["ab"])
    coarse = model.coarse_trajectory(base, b["cond"])
    topo = model.score_candidates(base, coarse, b["features"], b["mask"],
                                  b["lengths"])
    ar = torch.arange(b["mask"].shape[0])
    idx = topo["pi"].argmax(dim=-1)
    ell0 = model.build_ellipses(base, coarse, topo["path_feat"][ar, idx],
                                b["geom"][ar, idx], b["geom_len"][ar, idx],
                                b["ab"])
    with torch.no_grad():
        model.traj_type.weight[1].add_(1.0)
    ell1 = model.build_ellipses(base, coarse, topo["path_feat"][ar, idx],
                                b["geom"][ar, idx], b["geom_len"][ar, idx],
                                b["ab"])
    assert not torch.allclose(ell0["tokens"], ell1["tokens"])
    assert torch.allclose(ell1["tokens"] - ell0["tokens"],
                          torch.ones_like(ell0["tokens"]), atol=1e-6)

