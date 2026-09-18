"""V2 sampler tests (spec section 37): commit-once, no ellipse state, modality."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
import torch

from v2_utils import tiny_batch, tiny_model

from src.diffusion import sampler_v2
from src.diffusion.schedule import NoiseSchedule
from src.geometry.skeleton_paths import nearest_arclength


def _schedule():
    return NoiseSchedule(16)


def _run(model, b, **kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("seed", 0)
    return sampler_v2.sample_v2(model, _schedule(), b["cond"], b["occ"], b["cand"],
                                b["mask"], b["lengths"], **kw)


def test_topology_commits_only_once():
    """score_candidates is called exactly once, and only after t <= commit_t."""
    model = tiny_model(horizon=8)
    b = tiny_batch(B=2, M=3, L=8)
    calls = {"n": 0}
    orig = model.score_candidates

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    model.score_candidates = spy
    out = _run(model, b, commit_t=7)
    assert calls["n"] == 1, "the topology was re-scored during the reverse pass"
    assert int(out["committed_at"][0]) == 7
    assert int(out["committed_at"][1]) == 7


def test_subsampled_sampler_still_commits():
    """--steps can skip level 7; the rule is the first t <= commit_t."""
    model = tiny_model(horizon=8)
    b = tiny_batch(B=1, M=3, L=8)
    times = sampler_v2.pick_times(16, 4)
    assert times == [0, 5, 10, 15]
    calls = {"n": 0}
    orig = model.score_candidates

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    model.score_candidates = spy
    out = _run(model, b, steps=4, commit_t=7)
    assert calls["n"] == 1
    assert int(out["committed_at"][0]) == 5
    assert int(out["selected_idx"][0]) in (0, 1)


def test_full_run_commits_exactly_at_commit_t():
    model = tiny_model(horizon=8)
    b = tiny_batch(B=1, M=2, L=8)
    out = _run(model, b, commit_t=11)
    assert int(out["committed_at"][0]) == 11


def test_selected_topology_never_changes_after_commit():
    model = tiny_model(horizon=8)
    b = tiny_batch(B=2, M=3, L=8)
    seen = []
    orig = model.refine_with_path

    def spy(base, selected_path, selected_feat):
        seen.append(selected_path.detach().clone())
        return orig(base, selected_path, selected_feat)

    model.refine_with_path = spy
    _run(model, b, commit_t=7)
    assert len(seen) >= 2
    first = seen[0]
    for later in seen[1:]:
        assert torch.equal(first, later), "the committed topology changed"


def test_sampler_has_no_ellipse_diffusion_state():
    """Structural check: V2 diffuses the trajectory only."""
    src = inspect.getsource(sampler_v2.sample_v2)
    assert "torch.randn(B, H, 2" in src
    for forbidden in ("torch.randn(B, H, 6", "x0_e", "eps_e", "mlp_e("):
        assert forbidden not in src, forbidden
    # no standalone ellipse state variable either
    assert re.search(r"e_t", src) is None
    assert re.search(r"e_ts*=", src) is None


def test_sampler_endpoints_are_exact():
    model = tiny_model(horizon=8)
    b = tiny_batch(B=2, M=2, L=8)
    out = _run(model, b)
    assert torch.allclose(out["p"][:, 0], b["cond"][:, 0], atol=1e-6)
    assert torch.allclose(out["p"][:, -1], b["cond"][:, 1], atol=1e-6)


def test_sampler_centers_lie_on_the_selected_candidate():
    """Never a soft average of candidates: the geometry picks one real path."""
    model = tiny_model(horizon=8)
    b = tiny_batch(B=2, M=3, L=8)
    out = _run(model, b, selection="sample")
    cand = b["cand"].numpy()
    center = out["ellipse_center"].numpy()
    for i in range(2):
        idx = int(out["selected_idx"][i])
        assert bool(b["mask"][i, idx])
        poly = cand[i, idx, :, :2]
        _, dist = nearest_arclength(center[i], poly)
        assert np.all(dist < 1e-5)


def test_multimodal_sampling_varies_with_the_seed():
    model = tiny_model(horizon=8)
    b = tiny_batch(B=1, M=3, L=8)
    b["mask"][0, :] = True
    picks = set()
    for s in range(8):
        out = _run(model, b, seed=s, selection="sample")
        picks.add(int(out["selected_idx"][0]))
        assert bool(b["mask"][0, out["selected_idx"][0]])
    assert len(picks) > 1, "categorical topology sampling collapsed"


def test_argmax_selection_is_deterministic():
    model = tiny_model(horizon=8)
    b = tiny_batch(B=1, M=3, L=8)
    b["mask"][0, :] = True
    a = _run(model, b, seed=1, selection="argmax")
    c = _run(model, b, seed=1, selection="argmax")
    assert int(a["selected_idx"][0]) == int(c["selected_idx"][0])
    assert torch.allclose(a["p"], c["p"], atol=1e-6)
    # a different seed changes x_T (and therefore the continuous trajectory),
    # but argmax selection itself is deterministic
    d = _run(model, b, seed=2, selection="argmax")
    assert int(d["selected_idx"][0]) == int(a["selected_idx"][0])


def test_rows_without_candidates_do_not_crash():
    model = tiny_model(horizon=8)
    b = tiny_batch(B=2, M=2, L=8)
    b["mask"][0, :] = False
    out = _run(model, b)
    assert not bool(out["has_candidate"][0])
    assert bool(out["has_candidate"][1])
    assert int(out["committed_at"][0]) == -1
    assert int(out["committed_at"][1]) == 7
    assert torch.isfinite(out["p"]).all()
    assert torch.isfinite(out["ellipse_center"]).all()


def test_progress_is_re_predicted_at_every_refine_step():
    """s_i may change per reverse step (section 34) but stays valid."""
    model = tiny_model(horizon=8)
    b = tiny_batch(B=1, M=2, L=8)
    seen = []
    orig = model.refine_with_path

    def spy(base, selected_path, selected_feat):
        out = orig(base, selected_path, selected_feat)
        seen.append(out["progress"].detach().clone())
        return out

    model.refine_with_path = spy
    out = _run(model, b, commit_t=7)
    assert len(seen) >= 2
    for s in seen:
        assert torch.all(s[:, 1:] >= s[:, :-1] - 1e-7)
        assert torch.allclose(s[:, 0], torch.zeros(1), atol=1e-6)
        assert torch.allclose(s[:, -1], torch.ones(1), atol=1e-5)
    assert not all(torch.equal(seen[0], s) for s in seen[1:])
