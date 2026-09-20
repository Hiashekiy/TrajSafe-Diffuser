"""Report section 44: an OSQP/cvxpy reference solver for the SAME convex QP.

The runtime ALM is never replaced by a QP; this test only provides ground truth
so that a failure can be attributed correctly:

    QP feasible + ALM fails        -> ALM implementation / parameter bug
    QP infeasible                  -> corridor / interval assignment problem

The QP is

    min_{dQ_I}  0.5 lp ||B_I dQ_I||_F^2 + 0.5 ls ||S_I dQ_I||_F^2
    s.t.        A_j E_{l,r} (Q_ref + dQ) <= b_j

with the endpoints of ``dQ`` fixed to zero, i.e. exactly the frozen pack the ALM
optimises against.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

cp = pytest.importorskip("cvxpy")

from src.diffusion.bspline_alm import bspline_alm_correct           # noqa: E402
from src.geometry.bspline import BSplineCodec                       # noqa: E402
from src.geometry.bspline_constraints import (BSplineConstraintPack,  # noqa: E402
                                              build_constraint_pack)
from src.geometry.convex_region import EllipseRegionBuilder         # noqa: E402
from src.geometry.safety_corridor import build_safety_corridor      # noqa: E402

KNOTS = os.path.join(ROOT, "data", "carla_v1", "bspline_knots.npy")
ALM_CFG = {
    "rho": 5.0, "step_size": 0.05, "proximity_weight": 1.0,
    "correction_smooth_weight": 0.1, "constraint_tol": 1.0e-3,
    "max_curve_step_scene": 0.02,
}
LP, LS = 1.0, 0.1


def _codec():
    return BSplineCodec(degree=3, num_controls=32, curve_points=128,
                        knots_path=KNOTS)


def _walled_corridor(half_width=0.08, res=128, H=128):
    occ = torch.zeros(1, 1, res, res)
    iy = int((half_width + 1.0) / 2.0 * res)
    occ[:, :, iy:, :] = 1.0
    occ[:, :, :res - iy, :] = 1.0
    s = np.linspace(0.0, 1.0, H)
    centers = np.stack([-0.6 + 1.2 * s, np.zeros(H)], axis=-1)
    shape4 = np.tile(np.array([math.log(0.06), math.log(0.05), 1.0, 0.0]),
                     (H, 1))
    builder = EllipseRegionBuilder(occ, {"safety_margin": 0.01,
                                         "obstacle_window_half": 0.35})
    corridor = build_safety_corridor(
        builder, centers, shape4, s, gamma=centers, gamma_lengths=H,
        config={"min_overlap_ratio": 0.10,
                "bridge": {"enabled": True, "max_bridge_per_gap": 1}})
    codec = _codec()
    pack = build_constraint_pack(codec, [corridor],
                                 knot_boundaries=codec.knot_span_boundaries())
    cond = torch.tensor([[-0.6, 0.0], [0.6, 0.0]])
    q_ref = codec.fit_curve_to_controls(
        torch.as_tensor(centers, dtype=torch.float32)[None],
        cond[0:1], cond[1:2])
    return codec, pack, q_ref, cond


def _second_difference(num_points: int) -> np.ndarray:
    n = num_points - 2
    d = np.zeros((n, num_points))
    rows = np.arange(n)
    d[rows, rows] = 1.0
    d[rows, rows + 1] = -2.0
    d[rows, rows + 2] = 1.0
    return d


def _qp_from_pack(pack, codec, q_ref, sample=0):
    basis = codec.basis.double().numpy()
    C = basis.shape[1]
    b_i = basis[:, 1:-1]
    s_i = (_second_difference(basis.shape[0]) @ basis)[:, 1:-1]

    pm = pack.piece_mask[sample].numpy()
    fm = pack.face_mask[sample].numpy()
    A = pack.piece_A[sample].double().numpy()
    b = pack.piece_b[sample].double().numpy()
    E = pack.extraction[sample].double().numpy()
    q = q_ref[sample].double().numpy()

    rows, rhs = [], []
    for l in range(len(pm)):
        if not pm[l]:
            continue
        for r in range(4):
            for f in range(fm.shape[1]):
                if not fm[l, f]:
                    continue
                coef = E[l, r][:, None] * A[l, f][None, :]        # [C,2]
                gvec = coef.reshape(-1)
                rows.append(coef[1:-1].reshape(-1))
                rhs.append(b[l, f] - gvec @ q.reshape(-1))
    return (np.stack(rows), np.asarray(rhs), b_i, s_i, C)


def solve_qp(pack, codec, q_ref, sample=0):
    G, h, b_i, s_i, C = _qp_from_pack(pack, codec, q_ref, sample)
    x = cp.Variable((C - 2, 2))
    objective = cp.Minimize(0.5 * LP * cp.sum_squares(b_i @ x)
                            + 0.5 * LS * cp.sum_squares(s_i @ x))
    problem = cp.Problem(objective, [G @ cp.vec(x, order="C") <= h])
    problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8,
                  max_iter=200000, verbose=False)
    return problem.status, x.value


def _alm_violation(pack, codec, q_ref, inner_steps, cfg=None):
    q_safe, _, stats = bspline_alm_correct(
        q_ref, pack, codec, None, dict(ALM_CFG, **(cfg or {})),
        inner_steps=inner_steps)
    return float(stats["max_violation_after"]), q_safe


# ---------------------------------------------------------------------------
# 1. the corridor the sampler actually freezes must be QP-feasible
# ---------------------------------------------------------------------------


def test_qp_reference_agrees_with_the_alm_on_a_real_corridor():
    codec, pack, q_ref, _ = _walled_corridor()
    q_bad = q_ref.clone()
    q_bad[0, 14:19, 1] += 0.20

    status, x_qp = solve_qp(pack, codec, q_bad)
    assert status in ("optimal", "optimal_inaccurate"), status

    # the QP solution is genuinely feasible for the frozen pack
    if x_qp is not None:
        G, h, _, _, C = _qp_from_pack(pack, codec, q_bad)
        assert float((G @ x_qp.reshape(-1, order="C") - h).max()) < 1e-5

    after, _ = _alm_violation(pack, codec, q_bad, inner_steps=80)
    assert after < ALM_CFG["constraint_tol"], (
        "QP feasible but the ALM did not reach feasibility -> "
        "ALM implementation/parameter bug")


def test_qp_reference_reports_infeasibility_for_a_contradictory_pack():
    codec = _codec()
    B, P, F, C = 1, 1, 2, codec.num_controls
    pack = BSplineConstraintPack(
        extraction=torch.eye(C, dtype=torch.float64)[:4][None, None]
        .expand(B, P, 4, C).contiguous(),
        piece_A=torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]], dtype=torch.float64),
        piece_b=torch.tensor([[[-1.0, -1.0]]], dtype=torch.float64),
        face_mask=torch.ones(B, P, F, dtype=torch.bool),
        piece_mask=torch.ones(B, P, dtype=torch.bool),
        piece_region_id=torch.zeros(B, P, dtype=torch.long),
        intervals=torch.tensor([[[0.0, 1.0]]], dtype=torch.float64),
        num_pieces=torch.ones(B, dtype=torch.long))
    q_ref = torch.zeros(B, C, 2, dtype=torch.float64)

    status, _ = solve_qp(pack, codec, q_ref)
    assert status in ("infeasible", "infeasible_inaccurate"), status

    after, _ = _alm_violation(pack, codec, q_ref, inner_steps=60)
    assert after > ALM_CFG["constraint_tol"], (
        "an infeasible pack must be reported as a corridor problem, not "
        "silently 'solved'")
