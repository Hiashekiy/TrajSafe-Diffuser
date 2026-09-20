"""Control-space augmented-Lagrangian correction for the clean B-spline polygon.

The ONLY primal variable is the 32-control polygon

    Q in R^{B x 32 x 2},        Q_0 = start, Q_31 = goal

and the constraints are the FROZEN, exact, continuous B-spline/polytope
inequalities of :mod:`src.geometry.bspline_constraints`:

    g_k(Q) = A_j E_{l,r} Q - b_j <= 0

for every piece ``l``, every Bezier control ``r = 0..3`` and every valid face of
the responsible corridor region ``j``.  Because the four Bezier controls of a
piece lie in the (convex) region, the WHOLE curve piece does, which is why this
is a continuous safety certificate and not a 128-point sampling heuristic.

Objective (report section 22):

    J(Q) = lp/2 || B (Q - Q^ref) ||^2  +  ls/2 || D2 B (Q - Q^ref) ||^2

i.e. the correction is measured on the CURVE, and smoothness is applied to the
CORRECTION only.  A raw prediction that is already safe stays an exact fixed
point (Q = Q^ref) instead of being dragged around by ``||D2 B Q||^2``.

Every inner step is the same stable proximal scheme the legacy waypoint ALM
used, but now on the 30-control interior system:

    Y = dQ - eta * grad_constraint
    (I + eta * H_I) dQ_I^new = Y_I        (Cholesky solve)
    dQ <- clamp_curve_displacement(dQ_new - dQ)

The per-step correction limit is applied to the REAL curve displacement
``max_i ||B dQ_i||``, never to the control-space norm, and it is configured in
scene units (``alm.max_curve_step_scene``).

The dual ``lam`` is written in the same ordering as the pack, which is frozen
for the whole guided phase, so it can be warm-started across reverse steps.
"""

from __future__ import annotations

import torch

from ..geometry.safety_corridor import SCENE_TO_METER

__all__ = ["bspline_alm_correct", "constraint_state"]


def _second_difference_matrix(num_points: int, dtype, device) -> torch.Tensor:
    """``[H-2, H]`` second-difference operator."""
    if num_points < 3:
        return torch.zeros(0, num_points, dtype=dtype, device=device)
    n = num_points - 2
    d = torch.zeros(n, num_points, dtype=dtype, device=device)
    rows = torch.arange(n, device=device)
    d[rows, rows] = 1.0
    d[rows, rows + 1] = -2.0
    d[rows, rows + 2] = 1.0
    return d


def constraint_state(q: torch.Tensor, pack,
                     lam: torch.Tensor | None = None):
    """``beta``, ``g`` and the validity mask for a control polygon.

    ``beta [B,P,4,2]`` are the exact local Bezier controls, ``g [B,P,4,F]`` the
    signed (scene-unit) inequality values and ``mask [B,P,1,F]`` the broadcast
    validity mask.
    """
    beta = torch.einsum("bprk,bkd->bprd", pack.extraction, q)
    g = (torch.einsum("bpfd,bprd->bprf", pack.piece_A, beta)
         - pack.piece_b[:, :, None, :])
    mask = pack.piece_mask[:, :, None, None] & pack.face_mask[:, :, None, :]
    return beta, g, mask


def _violation_stats(g: torch.Tensor, mask: torch.Tensor):
    """Per-sample ``(max_violation, mean_positive_violation, feasible_rate)``."""
    valid = mask.expand_as(g)
    count = valid.sum(dim=(1, 2, 3)).clamp_min(1)
    positive = torch.relu(g) * valid
    max_violation = positive.flatten(1).max(dim=1).values
    mean_positive = positive.sum(dim=(1, 2, 3)) / count
    feasible = ((g <= 0) & valid).sum(dim=(1, 2, 3)) / count
    return max_violation, mean_positive, feasible


@torch.no_grad()
def bspline_alm_correct(
    q_ref: torch.Tensor,
    pack,
    codec,
    lam: torch.Tensor | None = None,
    config: dict | None = None,
    inner_steps: int = 3,
    max_steps: torch.Tensor | None = None,
    collect_stats: bool = True,
):
    """Project the current clean prediction into the frozen corridor.

    Parameters
    ----------
    q_ref       : [B,C,2] current network clean prediction (the proximity target,
                  re-created by the network every reverse step).
    pack        : frozen :class:`BSplineConstraintPack`.
    codec       : the fixed :class:`BSplineCodec` (basis / decode).
    lam         : [B,P,4,F] dual from the previous reverse step (warm start).
    inner_steps : default number of ALM iterations.
    max_steps   : optional [B] per-sample iteration budget.

    Returns ``(q_safe, lam, stats)``.
    """
    cfg = dict(config or {})
    rho = float(cfg.get("rho", 5.0))
    if rho <= 0:
        raise ValueError("rho must be positive")
    eta = float(cfg.get("step_size", 0.05))
    proximity = float(cfg.get("proximity_weight", 1.0))
    smooth_weight = float(cfg.get("correction_smooth_weight", 0.1))
    tol = float(cfg.get("constraint_tol", 1e-3))
    d_step = float(cfg.get("max_curve_step_scene", 0.01))
    scene_to_meter = float(cfg.get("scene_to_meter", SCENE_TO_METER))
    inner_steps = max(0, int(inner_steps))

    dtype = q_ref.dtype
    device = q_ref.device
    pack = pack.to(device=device, dtype=dtype)
    C = int(q_ref.shape[1])
    num_ctrl = int(pack.extraction.shape[-1])
    if num_ctrl != C:
        raise ValueError("pack expects %d controls, q_ref has %d"
                         % (num_ctrl, C))
    B = int(q_ref.shape[0])

    basis = codec.basis.to(dtype=dtype, device=device)             # [H,C]
    basis_i = basis[:, 1:-1]                                       # [H,C-2]
    d2 = _second_difference_matrix(basis.shape[0], dtype, device)  # [H-2,H]
    smooth_i = (d2 @ basis)[:, 1:-1]                               # [H-2,C-2]

    if proximity > 0:
        hess = proximity * (basis_i.transpose(0, 1) @ basis_i)
    else:
        hess = torch.zeros(C - 2, C - 2, dtype=dtype, device=device)
    if smooth_weight > 0:
        hess = hess + smooth_weight * (smooth_i.transpose(0, 1) @ smooth_i)
    system = torch.eye(C - 2, dtype=dtype, device=device) + eta * hess
    factor = torch.linalg.cholesky(system)

    q = q_ref.clone()
    if lam is None:
        lam = torch.zeros(pack.extraction.shape[0], pack.extraction.shape[1],
                          4, pack.piece_A.shape[2], dtype=dtype, device=device)
    else:
        lam = lam.to(dtype=dtype, device=device).clone()
    # inactive constraints keep a zero multiplier forever
    lam = lam * (pack.piece_mask[:, :, None, None]
                 & pack.face_mask[:, :, None, :])

    if max_steps is None:
        max_steps = torch.full((B,), inner_steps, dtype=torch.long,
                               device=device)
    else:
        max_steps = max_steps.to(device=device).long().clamp(min=0, max=inner_steps)
    inner_steps = int(max_steps.max().item()) if B else 0

    _, g_before, mask = constraint_state(q, pack)
    v_max_before, v_mean_before, feas_before = _violation_stats(g_before, mask)

    delta = torch.zeros_like(q)
    used = torch.zeros(B, dtype=torch.long, device=device)
    for it in range(inner_steps):
        active = (it < max_steps)
        if not bool(active.any()):
            break
        _, g, _ = constraint_state(q, pack)
        omega = torch.relu(lam + rho * g)
        omega = torch.where(mask, omega, torch.zeros_like(omega))

        # analytic constraint gradient: faces -> Bezier controls -> controls
        force_beta = torch.einsum("bprf,bpfd->bprd", omega, pack.piece_A)
        grad = torch.einsum("bprk,bprd->bkd", pack.extraction, force_beta)
        grad[:, 0] = 0.0
        grad[:, -1] = 0.0

        target = delta - eta * grad
        target[:, 0] = 0.0
        target[:, -1] = 0.0
        solved = torch.cholesky_solve(
            target[:, 1:-1].contiguous(),
            factor[None].expand(B, -1, -1))
        target = torch.cat([torch.zeros(B, 1, 2, dtype=dtype, device=device),
                            solved,
                            torch.zeros(B, 1, 2, dtype=dtype, device=device)],
                           dim=1)

        step = target - delta
        if d_step > 0:
            # limit on the REAL curve displacement, not the control-space norm
            curve_step = torch.einsum("hk,bkd->bhd", basis, step)
            d_max = curve_step.norm(dim=-1).max(dim=-1).values           # [B]
            scale = torch.where(d_max > d_step,
                                d_step / d_max.clamp_min(1e-12),
                                torch.ones_like(d_max))
            step = step * scale[:, None, None]
        step = step * active.to(dtype)[:, None, None]
        q = q + step
        q[:, 0] = q_ref[:, 0]
        q[:, -1] = q_ref[:, -1]
        delta = q - q_ref
        used = torch.where(active, torch.full_like(used, it + 1), used)

        _, g_new, _ = constraint_state(q, pack)
        lam = torch.relu(lam + rho * g_new)
        lam = torch.where(mask, lam, torch.zeros_like(lam))

        v_max_now, _, _ = _violation_stats(g_new, mask)
        remaining = (it + 1 < max_steps) & (v_max_now > tol)
        if not bool(remaining.any()):
            break

    stats = None
    if collect_stats:
        _, g_after, _ = constraint_state(q, pack)
        v_max_after, v_mean_after, feas_after = _violation_stats(g_after, mask)
        valid = mask.expand_as(g_after)
        count = valid.sum(dim=(1, 2, 3)).clamp_min(1)
        corr = (q - q_ref)
        curve_corr = torch.einsum("hk,bkd->bhd", basis, corr).norm(dim=-1)
        stats = {
            "max_violation_before": v_max_before,
            "max_violation_after": v_max_after,
            "mean_positive_violation_before": v_mean_before,
            "mean_positive_violation_after": v_mean_after,
            "constraint_feasible_rate_before": feas_before,
            "constraint_feasible_rate": feas_after,
            "active_constraint_count": valid.sum(dim=(1, 2, 3)).to(dtype),
            "mean_curve_correction_scene": curve_corr.mean(dim=-1),
            "max_curve_correction_scene": curve_corr.max(dim=-1).values,
            "mean_curve_correction_m": curve_corr.mean(dim=-1) * scene_to_meter,
            "max_curve_correction_m": curve_corr.max(dim=-1).values * scene_to_meter,
            "lambda_mean": lam.sum(dim=(1, 2, 3)) / count,
            "lambda_max": lam.flatten(1).max(dim=1).values,
            "inner_steps_used": used.to(dtype),
        }
    return q, lam, stats
