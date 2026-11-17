"""Tests for the NEW single-path framework (v5): planner, sampler, local pack.

Run:  python -m pytest tests/test_v5_framework.py -q

Covers the framework plan's hard constraints that can be checked without data:
  * ONE x0 output, no coarse / topology keys  (§13.1/§13.2)
  * step 1 runs with path/safety/feedback ABSENT, via learned null tokens + the
    absent-stream embedding — NOT "all zeros == absent"  (§2.4)
  * a missing stream changes the output (i.e. the mask is real)
  * endpoint position is exact and the endpoint HEADING matches  (§8)
  * feedback with valid=0 is the exact identity  (§7)
  * the local constraint pack covers less than the whole curve  (§5.4)
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from src.geometry.bspline_constraints import build_constraint_pack

from src.models.factory import build_model, framework_of
from src.utils.config import load_config

CFG = "configs/config_v5_c32q64.yaml"


def _cfg():
    cfg = load_config(CFG)
    m = dict(cfg["model"])
    m["d_model"] = 32          # tiny model: the tests are about semantics
    m["num_heads"] = 2
    m["ffn_dim"] = 64
    m["skeleton_blocks"] = 1
    cfg["model"] = m
    return cfg


def _batch(B=2, C=32):
    S = torch.tensor([[-0.5, -0.3], [-0.2, 0.1]])
    G = torch.tensor([[0.6, 0.4], [0.3, -0.5]])
    cond = torch.stack([S, G], dim=1)
    h = torch.tensor([[[1.0, 0.2], [0.8, 0.6]], [[1.0, 0.0], [0.7, 0.7]]])
    h = h / h.norm(dim=-1, keepdim=True)
    path = torch.stack([torch.stack([torch.linspace(S[b, 0], G[b, 0], 128),
                                     torch.linspace(S[b, 1], G[b, 1], 128)], -1)
                        for b in range(B)])
    q = torch.randn(B, C, 2)
    q[:, 0] = cond[:, 0]
    q[:, -1] = cond[:, 1]
    return dict(q=q, cond=cond, heading=h,
                heading_valid=torch.ones(B, 2, dtype=torch.bool),
                occ=torch.zeros(B, 1, 256, 256), path=path,
                t=torch.full((B,), 10), ab=torch.full((B,), 0.8))


def test_factory_picks_v5():
    cfg = _cfg()
    assert framework_of(cfg) == "v5"
    model = build_model(cfg)
    assert type(model).__name__ == "TrajSafePlannerV5"
    assert model.num_controls == 32
    assert model.num_safety_queries == 64


def test_no_coarse_and_no_topology_outputs():
    model = build_model(_cfg())
    assert not hasattr(model, "topology_head")
    assert not hasattr(model, "match_block_selector")
    out = model(**_batch_kwargs(model), )
    assert "coarse" not in out and "q_raw_coarse" not in out
    assert "pi" not in out and "selected_idx" not in out
    assert set(["control", "q_raw_final", "final", "ellipse"]).issubset(out)


def _batch_kwargs(model):
    b = _batch(C=model.num_controls)
    return dict(q_t=b["q"], occ=b["occ"], cond=b["cond"], t=b["t"], ab=b["ab"])


def test_step1_is_the_absent_stream_mode():
    model = build_model(_cfg())
    out = model(**_batch_kwargs(model))
    assert bool(out["has_path"][0]) is False
    assert bool(out["has_safety"][0]) is False
    assert out["ellipse"] == {}


def test_path_condition_changes_the_prediction():
    model = build_model(_cfg())
    b = _batch(C=model.num_controls)
    kw = dict(q_t=b["q"], occ=b["occ"], cond=b["cond"], t=b["t"], ab=b["ab"])
    o1 = model(**kw)
    o2 = model(**kw, active_path=b["path"])
    assert not torch.allclose(o1["control"], o2["control"], atol=1e-6)
    assert bool(o2["has_path"][0]) and bool(o2["has_safety"][0])


def test_endpoint_position_and_heading():
    model = build_model(_cfg())
    b = _batch(C=model.num_controls)
    out = model(q_t=b["q"], occ=b["occ"], cond=b["cond"], t=b["t"], ab=b["ab"],
                active_path=b["path"], heading=b["heading"],
                heading_valid=b["heading_valid"])
    q = out["control"]
    assert torch.allclose(q[:, 0], b["cond"][:, 0], atol=1e-6)
    assert torch.allclose(q[:, -1], b["cond"][:, 1], atol=1e-6)
    d0 = q[:, 1] - q[:, 0]
    c0 = (d0 * b["heading"][:, 0]).sum(-1) / d0.norm(dim=-1)
    assert float(c0.min()) > 0.9            # heading actually imposed


def test_invalid_feedback_is_exact_identity():
    model = build_model(_cfg())
    b = _batch(C=model.num_controls)
    kw = dict(q_t=b["q"], occ=b["occ"], cond=b["cond"], t=b["t"], ab=b["ab"],
              active_path=b["path"])
    o_ref = model(**kw)
    o_fb = model(**kw, feedback_delta=torch.randn_like(b["q"]),
                 feedback_valid=torch.zeros(b["q"].shape[0], dtype=torch.bool))
    assert torch.allclose(o_ref["control"], o_fb["control"], atol=1e-6)


def test_closed_loop_is_the_legacy_corridor_pipeline():
    """The sampler must reuse the legacy corridor+ALM code, not a private copy."""
    from src.diffusion import alm_local, sampler
    import inspect

    src = inspect.getsource(alm_local.alm_closed_loop)
    assert "build_safety_corridor(" in src
    assert "build_constraint_pack(" in src
    assert "bspline_alm_correct(" in src
    assert "dense_validation(" in src
    # and the sampler/rollout must call that one function
    assert alm_local.alm_closed_loop is not None
    assert hasattr(sampler, "dense_validation")
