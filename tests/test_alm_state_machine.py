"""Report section 45 (steps 9): the WARMUP / TRY_ACTIVATE / GUIDED state machine.

Uses a deterministic stub planner so the test exercises the sampler contract
(activation counting, topology fallback, corridor freezing, ``Q0_safe`` entering
the DDIM update, no extra final projection) without a checkpoint.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.diffusion.sampler import sample
from src.diffusion.schedule import NoiseSchedule
from src.geometry.bspline import BSplineCodec
from src.models.trajsafe.geometry import gather_dense_path_points

KNOTS = os.path.join(ROOT, "data", "carla_v1", "bspline_knots.npy")
HORIZON = 128
GEO = 128
RES = 128

ALM_CFG = {
    "enabled": True,
    "mode": "guided_bspline",
    "warmup_reverse_steps": 3,
    "max_activation_delay_steps": 2,
    "activation_inner_steps": 6,
    "inner_steps": 3,
    "step_size": 0.05,
    "rho": 5.0,
    "proximity_weight": 1.0,
    "correction_smooth_weight": 0.1,
    "constraint_tol": 1.0e-3,
    "max_curve_step_scene": 0.01,
    "inherit_dual_across_reverse_steps": True,
    "dense_verify_points": 512,
    "safety_margin": 0.01,
    "obstacle_window_half": 0.35,
}

CORRIDOR_CFG = {
    "min_overlap_ratio": 0.10,
    "bridge": {"enabled": True, "max_bridge_per_gap": 1},
    "topology_trials": 4,
}


class StubPlanner(nn.Module):
    """Deterministic stand-in for ``TrajSafeDiffuser``.

    ``Q0`` is a shrunk noisy state plus a lateral bulge, so the raw prediction
    genuinely violates the corridor and the ALM has something to do.
    """

    def __init__(self, codec, horizon=HORIZON, bump=0.0, ellipse_log_a=None,
                 ellipse_log_b=None):
        super().__init__()
        self.bspline = codec
        self.horizon = int(horizon)
        self.num_controls = int(codec.num_controls)
        self.bump = float(bump)
        self.ellipse_log_a = (math.log(0.06) if ellipse_log_a is None
                              else float(ellipse_log_a))
        self.ellipse_log_b = (math.log(0.05) if ellipse_log_b is None
                              else float(ellipse_log_b))
        self.register_buffer("fixed_progress", torch.linspace(0.0, 1.0, horizon))

    def hard_control_endpoints(self, q, cond):
        return BSplineCodec.hard_control_endpoints(q, cond)

    def forward_all(self, q, occ, cond, t, ab, candidate_xy, candidate_mask,
                    geometry, geometry_lengths, select_index=None):
        B = q.shape[0]
        dev = q.device
        H = self.horizon
        u = torch.linspace(0.0, 1.0, self.num_controls, device=dev)[None, :, None]
        straight = cond[:, 0:1, :] * (1.0 - u) + cond[:, 1:2, :] * u
        q0 = 0.5 * q + 0.5 * straight
        if self.bump:
            q0 = q0 + self.bump * torch.sin(math.pi * u)
        q0 = self.hard_control_endpoints(q0, cond)

        M = int(candidate_mask.shape[1])
        weight = torch.arange(M, 0, -1, device=dev, dtype=torch.float32)
        logits = torch.where(candidate_mask, weight[None].expand(B, M),
                             torch.full((B, M), float("-inf"), device=dev))
        pi = torch.softmax(logits, dim=-1)
        idx = (pi.argmax(dim=-1) if select_index is None
               else select_index.to(dev).long())
        ar = torch.arange(B, device=dev)
        s = self.fixed_progress[None].expand(B, H)
        gamma = geometry[ar, idx]
        glen = geometry_lengths[ar, idx].clamp(min=1)
        center = gather_dense_path_points(gamma, glen, s)
        su = self.fixed_progress[None, :, None]
        straight_curve = (cond[:, 0:1, :] * (1.0 - su)
                          + cond[:, 1:2, :] * su)
        center = torch.where(candidate_mask.any(dim=-1)[:, None, None], center,
                             straight_curve)

        shape4 = torch.zeros(B, H, 4, device=dev)
        shape4[..., 0] = self.ellipse_log_a
        shape4[..., 1] = self.ellipse_log_b
        shape4[..., 2] = 1.0
        return {
            "input_curve": self.bspline.decode_controls(q),
            "control": q0,
            "q_coarse": straight,
            "coarse": self.bspline.decode_controls(straight),
            "final": self.bspline.decode_controls(q0),
            "selected_idx": idx,
            "topo": {"pi": pi, "logits": logits},
            "ellipse": {
                "progress": s, "center": center, "shape4": shape4,
                "a": torch.exp(shape4[..., 0]), "b": torch.exp(shape4[..., 1]),
                "theta": 0.5 * torch.atan2(shape4[..., 3], shape4[..., 2]),
            },
        }


def _codec():
    return BSplineCodec(degree=3, num_controls=32, curve_points=HORIZON,
                        knots_path=KNOTS)


def _occupancy(half_width=0.92):
    occ = torch.zeros(1, 1, RES, RES)
    iy = int((half_width + 1.0) / 2.0 * RES)
    occ[:, :, iy:, :] = 1.0
    occ[:, :, :RES - iy, :] = 1.0
    return occ


def _geometry(kind):
    """kind: 'straight' | 'zigzag'."""
    x = np.linspace(-0.5, 0.5, GEO)
    if kind == "straight":
        y = np.zeros(GEO)
    else:
        y = np.where(np.arange(GEO) % 2 == 0, 0.8, -0.8)
    return np.stack([x, y], axis=-1).astype(np.float32)


def _batch(kinds=("zigzag", "straight"), bump=0.0):
    M = len(kinds)
    geometry = np.stack([_geometry(k) for k in kinds], axis=0)[None]
    cond = torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]])
    return {
        "cond": cond,
        "occ": _occupancy(),
        "candidate_xy": torch.as_tensor(geometry[:, :, :: max(1, GEO // 128)],
                                        dtype=torch.float32),
        "candidate_mask": torch.ones(1, M, dtype=torch.bool),
        "geometry": torch.as_tensor(geometry, dtype=torch.float32),
        "geometry_lengths": torch.full((1, M), GEO, dtype=torch.long),
        "bump": bump,
    }


def _run(kinds=("zigzag", "straight"), alm=None, corridor=None, steps=None,
         seed=0, bump=0.0, T=16):
    codec = _codec()
    model = StubPlanner(codec, bump=bump)
    schedule = NoiseSchedule(T, beta_schedule="squaredcos_cap_v2")
    batch = _batch(kinds, bump)
    cfg = dict(ALM_CFG)
    cfg.update(alm or {})
    cor = dict(CORRIDOR_CFG)
    cor.update(corridor or {})
    return sample(
        model, schedule, batch["cond"], batch["occ"], batch["candidate_xy"],
        batch["candidate_mask"], batch["geometry"], batch["geometry_lengths"],
        device="cpu", steps=steps, seed=seed, return_trace=True,
        alm_config=cfg, corridor_config=cor)


# --------------------------------------------------------------- state machine
def test_alm_disabled_runs_plain_reverse_diffusion():
    out = _run(alm={"enabled": False})
    assert out["alm_status"] == ["disabled"]
    assert not bool(out["guided"].any())
    assert int(out["activation_step"][0]) == -1
    assert out["corridors"] == [None]
    assert all(not step["alm_active"] for step in out["trace"])
    assert torch.equal(out["trace"][-1]["q0_safe"], out["trace"][-1]["q0_raw"])


def test_warmup_steps_are_counted_and_activation_uses_the_frozen_topology():
    out = _run()
    assert bool(out["guided"][0])
    assert out["alm_status"] == ["guided"]
    # three warm-up forwards, activation on the fourth reverse forward
    assert int(out["activation_step"][0]) == ALM_CFG["warmup_reverse_steps"]
    # candidate 0 is the zigzag whose corridor cannot close -> fallback to m=1
    assert int(out["frozen_topology_idx"][0]) == 1
    assert out["activation_info"]["topology_fallback"][0] is True
    assert out["activation_info"]["attempts"][0] >= 2

    guided_steps = [s for s in out["trace"] if bool(s["guided"][0])]
    assert len(guided_steps) == len(out["trace"]) - ALM_CFG["warmup_reverse_steps"]
    for step in guided_steps:
        assert int(step["frozen_topology_idx"][0]) == 1
        assert int(step["selected_idx"][0]) == 1
        assert step["alm_active"]
    for step in out["trace"][:ALM_CFG["warmup_reverse_steps"]]:
        assert not bool(step["guided"][0])
        assert not step["alm_active"]


def test_corridor_is_frozen_after_activation():
    out = _run()
    corridor = out["corridors"][0]
    assert corridor is not None and corridor["valid"]
    assert corridor["bridge_cell_count"] == 0
    # the frozen corridor is reported once, not per reverse step
    assert len(out["corridors"]) == 1
    assert out["pack_summary"]["num_pieces"][0] > 100
    assert float(min(corridor["overlap_ratio"])) >= CORRIDOR_CFG["min_overlap_ratio"]


def test_alm_safe_prediction_enters_the_ddim_update():
    out = _run(bump=0.35)
    moved = [s for s in out["trace"]
             if s["alm_active"] and not torch.equal(s["q0_safe"], s["q0_raw"])]
    assert moved, "the ALM never changed the clean prediction"
    # the trajectory that is actually returned is the last SAFE prediction
    assert torch.equal(out["control"], out["trace"][-1]["q"]) 
    assert torch.equal(out["trace"][-1]["q"], out["trace"][-1]["q0_safe"])


def test_alm_reduces_the_final_constraint_violation_end_to_end():
    raw = _run(alm={"enabled": False}, bump=0.35, seed=3)
    guided = _run(bump=0.35, seed=3)
    raw_violation = raw["final_validation"][0]["final_max_constraint_violation"]
    guided_violation = guided["final_validation"][0][
        "final_max_constraint_violation"]
    assert raw_violation is None            # no corridor -> nothing to measure
    assert guided_violation is not None
    assert guided["final_validation"][0]["endpoint_error"] < 1e-5


def test_activation_failure_degrades_to_raw_diffusion_without_crashing():
    # every candidate is a pathological zigzag -> no corridor can close
    out = _run(kinds=("zigzag", "zigzag"),
               alm={"max_activation_delay_steps": 1})
    assert out["alm_status"] == ["activation_failed"]
    assert not bool(out["guided"].any())
    assert torch.isfinite(out["control"]).all()
    assert out["activation_info"]["failure_reason"][0] is not None
    # warmup + (max_activation_delay_steps + 1) attempts
    assert out["activation_info"]["attempts"][0] == 2


def test_activation_delay_gives_the_corridor_more_chances():
    out = _run(alm={"warmup_reverse_steps": 1,
                    "max_activation_delay_steps": 3})
    assert int(out["activation_step"][0]) == 1


def test_warmup_counts_reverse_forwards_not_timesteps():
    out = _run(steps=6)
    assert int(out["activation_step"][0]) == ALM_CFG["warmup_reverse_steps"]
    assert len(out["trace"]) == 6


def test_final_only_mode_is_an_ablation_not_the_default():
    out = _run(alm={"mode": "final_only"})
    assert int(out["activation_step"][0]) == len(out["trace"]) - 1
    assert sum(1 for s in out["trace"] if s["alm_active"]) == 1


# ------------------------------------------------------------- hard invariants
def test_ddim_never_applies_a_second_final_projection():
    out = _run(bump=0.2)
    # the last trace entry is the terminal 0 -> -1 transition: q == q0_safe
    assert int(out["trace"][-1]["t"]) == 0
    assert int(out["trace"][-1]["s"]) == -1
    assert torch.equal(out["trace"][-1]["q"], out["trace"][-1]["q0_safe"])
    assert torch.equal(out["control"], out["trace"][-1]["q0_safe"])


def test_legacy_waypoint_alm_is_still_refused():
    codec = _codec()
    model = StubPlanner(codec)
    schedule = NoiseSchedule(16)
    batch = _batch()
    try:
        sample(model, schedule, batch["cond"], batch["occ"],
               batch["candidate_xy"], batch["candidate_mask"],
               batch["geometry"], batch["geometry_lengths"], device="cpu",
               alm_guidance=lambda *a, **k: None)
    except RuntimeError as error:
        assert "waypoint" in str(error) or "B-spline" in str(error)
    else:                                                 # pragma: no cover
        raise AssertionError("alm_guidance must be refused")


def test_trace_carries_everything_the_dashboard_needs():
    out = _run(bump=0.2)
    step = out["trace"][-1]
    for key in ("t", "s", "q", "q0_raw", "q0_safe", "p_raw", "p_safe",
                "alm_active", "alm_stats", "frozen_topology_idx", "guided",
                "selected_idx", "pi", "ellipse_center", "ellipse_shape4"):
        assert key in step, key
    assert step["alm_stats"] is not None
    for key in ("max_violation_before", "max_violation_after",
                "mean_curve_correction_scene", "max_curve_correction_m",
                "lambda_mean", "lambda_max", "inner_steps_used",
                "constraint_feasible_rate"):
        assert key in step["alm_stats"], key
    assert out["progress_alignment"][0]["progress_alignment_rmse"] is not None
