"""The four external inference-time constraint processors of section 4.2.

All four solve the SAME physical safety requirement on the SAME signed distance
field (``src/geometry/sdf_field.py``) at the SAME 128 decoded points and with
the SAME warm-up/idempotence contract; they differ ONLY in how they handle

    g_h^{SDF}(Q) = d_safe - d_full(B_h Q) <= 0 ,      h = 1..128,  d_safe = 0

    1. ``penalty``      fixed squared penalty, plain gradient descent
    2. ``projection``   sequential convexification + L2 projection (OSQP)
    3. ``primal_dual``  primal-dual (no quadratic penalty, no ALM)
    4. ``alm_sdf``      augmented Lagrangian on the SDF

They are deliberately NOT the paper's method: no ellipse safety regions, no
generative convex corridor, no historical safety feedback, no final hard
projection.  In the main table they are grouped as "external / SDF-based" and
the full method is reported separately (guide 4.1.2).

WHY THE GRADIENTS ARE ANALYTIC
------------------------------
``sample()`` is decorated ``@torch.no_grad()`` and the SDF uses ``grid_sample``
with a hand-built direction field, so every processor derives

    d g_h / d Q_k = -grad_p[h] * B[h,k]        =>   grad_Q = -einsum('hk,bhd->bkd', B, grad_p)

in closed form.  No autograd, no ``enable_grad`` inside the sampler, and the
same formula for all four (so their per-step cost is comparable).

SHARED CONTRACT
---------------
* endpoints are re-pinned after every inner iteration (``codec.hard_control_endpoints``);
* ``active [B]`` selects the samples actually guided (warm-up steps and samples
  without a candidate are skipped, and their rows are returned bit-identical);
* non-finite results are reverted to the input and flagged (``*_diverged``),
  never propagated as NaN;
* a result that is WORSE than its input (larger max violation) is reverted --
  the same monotone guard ``bspline_hard_project`` uses;
* everything is reported in scene units and converted to metres for the log.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..geometry.bspline import BSplineCodec
from ..geometry.sdf_field import OccupancySDF

__all__ = [
    "GUIDANCE_MODES",
    "sdf_penalty_correct",
    "sdf_projection_correct",
    "sdf_primal_dual_correct",
    "sdf_alm_correct",
    "apply_sdf_guidance",
    "curve_roughness",
]

#: the guidance selector used by ``sample(alm_config={"guidance": ...})``.
#: ``none`` = plain DDIM; ``alm`` is the paper's corridor ALM (handled in
#: sampler.py); the rest are the external SDF baselines.
GUIDANCE_MODES = ("none", "penalty", "projection", "primal_dual", "alm_sdf", "alm")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def curve_roughness(points: torch.Tensor) -> torch.Tensor:
    """Mean second difference of a decoded curve -> ``[B]`` (scene units).

    Matches ``sampler._curve_roughness`` (used by the trace) so the process
    curves and the final metric agree.
    """
    if points.shape[1] < 3:
        return points.new_zeros(points.shape[0])
    d2 = points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]
    return d2.norm(dim=-1).mean(dim=1)


def _decode(codec: BSplineCodec, q: torch.Tensor) -> torch.Tensor:
    return codec.decode_controls(q)


def _constraint_rows(codec: BSplineCodec, q: torch.Tensor,
                     sdf: OccupancySDF, d_safe: float):
    """``(p, value, grad_p, g)`` for ``g_h = d_safe - d_full(p_h)``."""
    basis = codec.basis.to(dtype=q.dtype)
    p = torch.einsum("hk,bkd->bhd", basis, q)
    value, grad_p = sdf.query(p, with_boundary=True, with_grad=True)
    return p, value, grad_p, (float(d_safe) - value)


def _grad_q(codec: BSplineCodec, grad_p: torch.Tensor,
            weight: torch.Tensor) -> torch.Tensor:
    """``D_k = sum_h B[h,k] * weight_h * grad_p[h]`` -> ``[B,C,2]``.

    SIGN CONVENTION (this bit is easy to get backwards and it silently turns
    every processor into gradient ASCENT, which the monotone guard then reverts
    to a no-op -- measured): ``grad_p`` is the direction of INCREASING
    ``d_occ``, so moving a curve point ALONG ``+grad_p`` DECREASES the violation
    ``g = d_safe - d_occ``.  Therefore, with
    ``L = sum_h weight_h * g_h`` (weight >= 0),

        dL/dQ_k = -sum_h B[h,k] weight_h grad_p[h]  =  -_grad_q(grad_p, weight)

    and the DESCENT step is ``Q <- Q - eta dL/dQ = Q + eta * _grad_q(...)``.
    ``tests/test_sdf_guidance.py::test_gradient_step_direction`` pins this on a
    real occupancy grid.
    """
    basis = codec.basis.to(dtype=grad_p.dtype)
    return torch.einsum("hk,bhd->bkd", basis, weight.unsqueeze(-1) * grad_p)


def _violation_of(codec: BSplineCodec, q: torch.Tensor, sdf: OccupancySDF,
                  d_safe: float):
    """``(violation, mean_violation, residual)`` of a control polygon, scene units.

    ``violation = max_h relu(g_h)`` -- NON-NEGATIVE, exactly the convention of
    ``bspline_alm._violation_stats``, so the corridor ALM and the four SDF
    processors report the same quantity under the same name (the 4.4 process
    curves and the 4.3 "violation before correction" column compare them).
    ``residual = max_h g_h`` is the signed max constraint function of guide
    4.1.3 §2 (negative = safe margin) and is reported separately.
    """
    value = sdf.query(_decode(codec, q), with_boundary=True, with_grad=False)
    g = float(d_safe) - value
    positive = torch.relu(g)
    return positive.max(dim=1).values, positive.mean(dim=1), g.max(dim=1).values


def _empty_stats(batch: int, device, dtype) -> dict:
    z = torch.zeros(batch, device=device, dtype=dtype)
    return {
        "violation_before": z.clone(), "violation_after": z.clone(),
        "residual_before": z.clone(), "residual_after": z.clone(),
        "mean_violation_before": z.clone(), "mean_violation_after": z.clone(),
        "correction_scene": z.clone(), "roughness_before": z.clone(),
        "roughness_after": z.clone(), "lambda_max": z.clone(),
        "inner_steps_used": torch.zeros(batch, device=device, dtype=torch.long),
        "status": ["skipped"] * batch,
    }


def _finalize(codec: BSplineCodec, sdf: OccupancySDF, d_safe: float,
              q_in: torch.Tensor, q_out: torch.Tensor, active: torch.Tensor,
              stats: dict, scene_to_meter: float) -> torch.Tensor:
    """Apply the divergence / monotone guards and fill the shared stats."""
    finite = torch.isfinite(q_out).all(dim=(1, 2))
    q_out = torch.where(finite[:, None, None], q_out, q_in)
    for b in range(q_out.shape[0]):
        if not bool(active[b]):
            continue
        if not bool(finite[b]):
            stats["status"][b] = "diverged"
    before, mean_before, residual_before = _violation_of(codec, q_in, sdf, d_safe)
    after, mean_after, residual_after = _violation_of(codec, q_out, sdf, d_safe)
    worse = after > before
    q_out = torch.where(worse[:, None, None], q_in, q_out)
    after = torch.where(worse, before, after)
    mean_after = torch.where(worse, mean_before, mean_after)
    residual_after = torch.where(worse, residual_before, residual_after)
    for b in range(q_out.shape[0]):
        if bool(active[b]) and bool(worse[b]) and stats["status"][b] != "diverged":
            stats["status"][b] = "reverted_worse"
    stats["violation_before"] = before
    stats["violation_after"] = after
    stats["residual_before"] = residual_before
    stats["residual_after"] = residual_after
    stats["mean_violation_before"] = mean_before
    stats["mean_violation_after"] = mean_after
    correction = _decode(codec, q_out - q_in).norm(dim=-1)
    stats["correction_scene"] = correction.max(dim=-1).values
    stats["correction_m"] = stats["correction_scene"] * float(scene_to_meter)
    stats["roughness_before"] = curve_roughness(_decode(codec, q_in))
    stats["roughness_after"] = curve_roughness(_decode(codec, q_out))
    return q_out


# ---------------------------------------------------------------------------
# 1. Penalty-guided
# ---------------------------------------------------------------------------


@torch.no_grad()
def sdf_penalty_correct(q_ref: torch.Tensor, sdf: OccupancySDF,
                        codec: BSplineCodec, cond: torch.Tensor,
                        config: dict | None = None,
                        active: torch.Tensor | None = None):
    """``min_Q  w * sum_h relu(g_h)^2`` by fixed-step gradient descent.

    ``Q <- Q - eta * w * sum_h 2*relu(g_h) * dg_h/dQ`` (guide 4.1.2 §2).  No
    multipliers are carried between reverse steps and no adaptive penalty is
    used, so this is "collision-cost guidance from the occupancy distance
    field" and nothing else.
    """
    cfg = dict(config or {})
    w = float(cfg.get("penalty_weight", 5.0))
    eta = float(cfg.get("step_size", 0.05))
    inner = int(cfg.get("inner_steps", 5))
    d_safe = float(cfg.get("d_safe", 0.0))
    s2m = float(cfg.get("scene_to_meter", sdf.scene_to_meter))
    if active is None:
        active = torch.ones(q_ref.shape[0], dtype=torch.bool, device=q_ref.device)

    q = q_ref.clone()
    stats = _empty_stats(q_ref.shape[0], q_ref.device, q_ref.dtype)
    used = torch.zeros(q_ref.shape[0], dtype=torch.long, device=q_ref.device)
    for k in range(max(1, inner)):
        _, _, grad_p, g = _constraint_rows(codec, q, sdf, d_safe)
        positive = torch.relu(g)
        if not bool(active.any()):
            break
        # dL/dQ = -2w * _grad_q(grad_p, relu(g))  =>  descent ADDS the term
        step = 2.0 * w * _grad_q(codec, grad_p, positive)
        step = step * active[:, None, None].to(step.dtype)
        q = q + eta * step
        q = codec.hard_control_endpoints(q, cond)
        used = used + active.long()
    stats["inner_steps_used"] = used
    for b in range(q.shape[0]):
        if bool(active[b]):
            stats["status"][b] = "penalty_ok"
    q_out = _finalize(codec, sdf, d_safe, q_ref, q, active, stats, s2m)
    return q_out, None, stats


# ---------------------------------------------------------------------------
# 2. Projected (sequential convexification + L2 projection)
# ---------------------------------------------------------------------------


def _project_one(q0: np.ndarray, codec: BSplineCodec, sdf: OccupancySDF,
                 sample: int, cond_row: np.ndarray, d_safe: float,
                 eps_ridge: float, map_margin: float, osqp_cfg: dict) -> tuple:
    """One L2 projection of a single control polygon (returns ``(q, status)``)."""
    import osqp
    import scipy.sparse as sp

    C = int(codec.num_controls)
    basis = codec.basis.detach().cpu().numpy().astype(np.float64)       # [H,C]
    p0 = basis @ q0                                                     # [H,2]
    value, grad = sdf.select(sample).query(
        torch.as_tensor(p0, dtype=torch.float32, device=sdf.device)[None],
        with_boundary=True, with_grad=True)
    value = value[0].detach().cpu().numpy().astype(np.float64)
    grad = grad[0].detach().cpu().numpy().astype(np.float64)            # [H,2]
    gx, gy = grad[:, 0], grad[:, 1]

    nsdf = basis.shape[0]
    # min ||B(q - q0)||^2 + eps||q - q0||^2  ->  0.5 z'Pz + qv'z
    bt_b = basis.T @ basis                                             # [C,C]
    gram = bt_b + eps_ridge * np.eye(C)
    P = sp.block_diag([gram, gram], format="csc") * 2.0
    qv = -2.0 * np.concatenate([gram @ q0[:, 0], gram @ q0[:, 1]])

    # linearised SDF:  -grad_h' B_h q  <=  value_h - grad_h' p0_h - d_safe
    sdf_upper = value - np.einsum("hd,hd->h", grad, p0) - float(d_safe)
    sdf_rows = sp.hstack([
        sp.csr_matrix(-gx[:, None] * basis),
        sp.csr_matrix(-gy[:, None] * basis)], format="csc")
    # box: |B_h q| <= 1 - margin on both axes
    inset = 1.0 - float(map_margin)
    box_basis = sp.csr_matrix(basis)
    zeros = sp.csr_matrix((nsdf, C))
    box_rows = sp.vstack([sp.hstack([box_basis, zeros]),
                          sp.hstack([-box_basis, zeros]),
                          sp.hstack([zeros, box_basis]),
                          sp.hstack([zeros, -box_basis])], format="csc")
    # pinned endpoints: OSQP has no variable bounds, so q_0 = start and
    # q_{C-1} = goal are two equality rows per coordinate (l == u)
    pin_rows = sp.lil_matrix((4, 2 * C))
    z0 = np.concatenate([q0[:, 0], q0[:, 1]])
    pin_value = np.empty(4)
    row = 0
    for coord in (0, 1):
        for ctrl in (0, C - 1):
            k = coord * C + ctrl
            pin_rows[row, k] = 1.0
            pin_value[row] = z0[k]
            row += 1
    A = sp.vstack([sdf_rows, box_rows, pin_rows.tocsc()], format="csc")
    upper = np.concatenate([sdf_upper, np.full(4 * nsdf, inset), pin_value])
    lower = np.concatenate([np.full(sdf_rows.shape[0] + 4 * nsdf, -np.inf),
                            pin_value])

    # A fresh setup per solve is DELIBERATELY kept: sharing one OSQP object and
    # feeding it ``update(Ax=...)`` was measured to be a LOSS here (10.16 ->
    # 8.02 ms in a synthetic micro-benchmark, i.e. only 1.27x, because OSQP
    # re-factorises the KKT matrix whenever the values move), and 41 % SLOWER
    # end to end (719 -> 1014 ms/trajectory) on the real workload: a warm start
    # carried over from a different sample inflates the ADMM iteration count of
    # the hard instances, and every failure then pays for a second setup.
    # Threading does not help either -- OSQP's wrapper holds the GIL (measured
    # 7.84 vs 7.28 ms with 8 threads), so the per-sample loop stays serial.
    prob = osqp.OSQP()
    prob.setup(P=P, q=qv, A=A, l=lower, u=upper, **osqp_cfg)
    res = prob.solve()
    if res.info.status_val not in (1, 2):        # 1 = solved, 2 = solved inaccurate
        return q0, "projection_failed(%s)" % res.info.status
    z = np.asarray(res.x, dtype=np.float64)
    if not np.isfinite(z).all():
        return q0, "projection_nonfinite"
    return np.stack([z[:C], z[C:]], axis=-1), "ok"


@torch.no_grad()
def sdf_projection_correct(q_ref: torch.Tensor, sdf: OccupancySDF,
                           codec: BSplineCodec, cond: torch.Tensor,
                           config: dict | None = None,
                           active: torch.Tensor | None = None):
    """Sequential convexification + L2 minimum-deviation projection (guide §3).

    Each inner step linearises the SDF at the CURRENT polygon, builds the local
    halfspaces plus the map box, and solves

        min_Q ||B(Q - Q_raw)||_2^2   s.t.  linearised SDF, box, pinned endpoints

    with OSQP.  The paper's final L1 convex-corridor projection is a different
    object and is NOT reused here (the free space of an occupancy grid is
    non-convex, so the L1 corridor LP does not apply).  Failures keep the raw
    input and are counted, never papered over with a fake feasible result.
    """
    cfg = dict(config or {})
    inner = int(cfg.get("inner_steps", 5))
    d_safe = float(cfg.get("d_safe", 0.0))
    s2m = float(cfg.get("scene_to_meter", sdf.scene_to_meter))
    eps_ridge = float(cfg.get("projection_ridge", 1e-6))
    map_margin = float(cfg.get("map_margin", sdf.map_margin))
    osqp_cfg = {
        "eps_abs": float(cfg.get("osqp_eps_abs", 1e-7)),
        "eps_rel": float(cfg.get("osqp_eps_rel", 1e-7)),
        "max_iter": int(cfg.get("osqp_max_iter", 4000)),
        # Keep OSQP's polishing ON (the default).  Disabling it was tried to
        # silence its per-solve "Polishing not needed" line, but that line is
        # cheap while polishing is a cheap ACCURACY refinement: without it the
        # ADMM needs far more iterations to reach eps_abs=1e-7, and the guard
        # went from 719 ms to 1403 ms per trajectory on the real workload.
        "polishing": bool(cfg.get("osqp_polish", True)),
        "verbose": False,
        "warm_starting": True,
    }
    if active is None:
        active = torch.ones(q_ref.shape[0], dtype=torch.bool, device=q_ref.device)

    q = q_ref.clone()
    q_np = q_ref.detach().cpu().numpy().astype(np.float64)
    cond_np = cond.detach().cpu().numpy().astype(np.float64)
    stats = _empty_stats(q_ref.shape[0], q_ref.device, q_ref.dtype)
    solved = torch.zeros(q_ref.shape[0], dtype=torch.long, device=q_ref.device)
    for b in range(q_ref.shape[0]):
        if not bool(active[b]):
            continue
        current = q_np[b]
        status = "ok"
        steps = 0
        for _ in range(max(1, inner)):
            current, status = _project_one(current, codec, sdf, b, cond_np[b],
                                           d_safe, eps_ridge, map_margin, osqp_cfg)
            steps += 1
            if status != "ok":
                break
        if status == "ok":
            q[b] = torch.as_tensor(current, dtype=q.dtype, device=q.device)
            solved[b] = steps
        stats["status"][b] = status if status == "ok" else status
    stats["inner_steps_used"] = solved
    q_out = _finalize(codec, sdf, d_safe, q_ref, q, active, stats, s2m)
    return q_out, None, stats


# ---------------------------------------------------------------------------
# 3. Primal-Dual
# ---------------------------------------------------------------------------


@torch.no_grad()
def sdf_primal_dual_correct(q_ref: torch.Tensor, sdf: OccupancySDF,
                            codec: BSplineCodec, cond: torch.Tensor,
                            lam: Optional[torch.Tensor] = None,
                            config: dict | None = None,
                            active: torch.Tensor | None = None):
    """Alternating primal / non-negative dual updates on ``g_h^SDF`` (guide §4).

    ``Q <- Q - eta_Q sum_h lam_h dg_h/dQ`` ,  ``lam <- relu(lam + eta_lam g)``.
    NO quadratic penalty term (that is what distinguishes it from SDF-ALM) and
    no corridor / feedback / final projection.
    """
    cfg = dict(config or {})
    eta_q = float(cfg.get("step_size", 0.05))
    eta_lam = float(cfg.get("dual_step", 1.0))
    inner = int(cfg.get("inner_steps", 5))
    d_safe = float(cfg.get("d_safe", 0.0))
    s2m = float(cfg.get("scene_to_meter", sdf.scene_to_meter))
    inherit = bool(cfg.get("inherit_dual_across_reverse_steps", True))
    if active is None:
        active = torch.ones(q_ref.shape[0], dtype=torch.bool, device=q_ref.device)

    q = q_ref.clone()
    B, H = q_ref.shape[0], int(codec.curve_points)
    if lam is None or lam.shape != (B, H) or not inherit:
        lam = torch.zeros(B, H, device=q_ref.device, dtype=q_ref.dtype)
    else:
        lam = lam.clone()
    stats = _empty_stats(B, q_ref.device, q_ref.dtype)
    used = torch.zeros(B, dtype=torch.long, device=q_ref.device)
    for _ in range(max(1, inner)):
        if not bool(active.any()):
            break
        _, _, grad_p, g = _constraint_rows(codec, q, sdf, d_safe)
        weight = lam * active[:, None].to(lam.dtype)
        q = q + eta_q * _grad_q(codec, grad_p, weight)
        q = codec.hard_control_endpoints(q, cond)
        lam = torch.relu(lam + eta_lam * g * active[:, None].to(g.dtype))
        used = used + active.long()
    stats["lambda_max"] = lam.max(dim=1).values
    stats["inner_steps_used"] = used
    for b in range(B):
        if bool(active[b]):
            stats["status"][b] = "primal_dual_ok"
    q_out = _finalize(codec, sdf, d_safe, q_ref, q, active, stats, s2m)
    return q_out, lam, stats


# ---------------------------------------------------------------------------
# 4. SDF-ALM
# ---------------------------------------------------------------------------


@torch.no_grad()
def sdf_alm_correct(q_ref: torch.Tensor, sdf: OccupancySDF,
                    codec: BSplineCodec, cond: torch.Tensor,
                    lam: Optional[torch.Tensor] = None,
                    config: dict | None = None,
                    active: torch.Tensor | None = None):
    """Augmented-Lagrangian update on the SAME SDF constraints (guide §5).

        min L + sum_h lam_h g_h + rho/2 sum_h relu(g_h)^2

    Same action site and same multiplier inheritance as Primal-Dual, so the two
    differ only in the quadratic penalty — which is exactly the comparison the
    guide asks for.  Labelled "SDF-ALM" everywhere to keep it apart from the
    paper's corridor ALM.
    """
    cfg = dict(config or {})
    rho = float(cfg.get("rho", 5.0))
    eta_q = float(cfg.get("step_size", 0.05))
    inner = int(cfg.get("inner_steps", 5))
    d_safe = float(cfg.get("d_safe", 0.0))
    s2m = float(cfg.get("scene_to_meter", sdf.scene_to_meter))
    inherit = bool(cfg.get("inherit_dual_across_reverse_steps", True))
    tol = float(cfg.get("constraint_tol", 1e-3))
    if active is None:
        active = torch.ones(q_ref.shape[0], dtype=torch.bool, device=q_ref.device)

    q = q_ref.clone()
    B, H = q_ref.shape[0], int(codec.curve_points)
    if lam is None or lam.shape != (B, H) or not inherit:
        lam = torch.zeros(B, H, device=q_ref.device, dtype=q_ref.dtype)
    else:
        lam = lam.clone()
    stats = _empty_stats(B, q_ref.device, q_ref.dtype)
    used = torch.zeros(B, dtype=torch.long, device=q_ref.device)
    for _ in range(max(1, inner)):
        if not bool(active.any()):
            break
        _, _, grad_p, g = _constraint_rows(codec, q, sdf, d_safe)
        coef = (lam + rho * torch.relu(g)) * active[:, None].to(lam.dtype)
        q = q + eta_q * _grad_q(codec, grad_p, coef)
        q = codec.hard_control_endpoints(q, cond)
        lam = torch.relu(lam + rho * g * active[:, None].to(g.dtype))
        used = used + active.long()
        # early stop: the guide's "V^SDF <= eps" stopping rule, on active rows
        viol = (float(d_safe) - sdf.query(
            _decode(codec, q), with_boundary=True, with_grad=False)).max(dim=1).values
        if bool((viol[active] <= tol).all()):
            break
    stats["lambda_max"] = lam.max(dim=1).values
    stats["inner_steps_used"] = used
    for b in range(B):
        if bool(active[b]):
            stats["status"][b] = "alm_sdf_ok"
    q_out = _finalize(codec, sdf, d_safe, q_ref, q, active, stats, s2m)
    return q_out, lam, stats


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def apply_sdf_guidance(mode: str, q_ref: torch.Tensor, sdf: OccupancySDF,
                       codec: BSplineCodec, cond: torch.Tensor,
                       lam: Optional[torch.Tensor] = None,
                       config: dict | None = None,
                       active: torch.Tensor | None = None):
    """Run one of the four processors; returns ``(q_safe, lam, stats)``."""
    key = str(mode).lower()
    if key == "penalty":
        return sdf_penalty_correct(q_ref, sdf, codec, cond, config, active)
    if key == "projection":
        return sdf_projection_correct(q_ref, sdf, codec, cond, config, active)
    if key == "primal_dual":
        return sdf_primal_dual_correct(q_ref, sdf, codec, cond, lam, config, active)
    if key == "alm_sdf":
        return sdf_alm_correct(q_ref, sdf, codec, cond, lam, config, active)
    raise ValueError("unknown SDF guidance mode %r (use %s)"
                     % (mode, ", ".join(GUIDANCE_MODES)))
