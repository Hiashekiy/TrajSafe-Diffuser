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
scene units (``alm.max_curve_step_scene``).  Do NOT remove it: it is part of the
convergence of the accumulated/proximal scheme -- with no bound the displacement
diverges and NaN-poisoned controls reach the next forward pass (measured).

The dual ``lam`` is written in the same ordering as the pack, which is frozen
for the whole guided phase, so it can be warm-started across reverse steps.
"""

from __future__ import annotations

import numpy as np
import torch

from ..geometry.safety_corridor import SCENE_TO_METER

__all__ = ["bspline_alm_correct", "constraint_state", "bspline_hard_project"]


def _smooth_along_controls(x: torch.Tensor, half_width: int,
                           sigma: float = 0.0) -> torch.Tensor:
    """Convolve a [B,C,2] displacement along the CONTROL axis C.

    A violated constraint only needs ~4 control points to move, and the
    proximal solve is essentially the identity, so without this those 4
    points are displaced IN PLACE and the curve kinks exactly there.  A
    low-pass along the control axis turns the per-iteration step into a
    smooth bell, so the neighbouring controls are dragged along with a
    decaying weight and the curve bends instead of folding.

    The endpoints are re-imposed by the caller after the update, so the
    filter is free to leak into them.
    """
    hw = int(half_width)
    if hw <= 0:
        return x
    n = 2 * hw + 1
    t = torch.arange(n, dtype=torch.float32, device=x.device) - hw
    if float(sigma or 0.0) > 0.0:
        k = torch.exp(-0.5 * (t / float(sigma)) ** 2)
    else:
        k = (hw + 1.0) - t.abs()                      # triangular
    k = (k / k.sum()).to(dtype=x.dtype)
    B, C, D = x.shape
    y = x.permute(0, 2, 1).reshape(B * D, 1, C)
    # one shared 1-channel kernel over a flattened batch (NOT depthwise:
    # the input has a single channel, so groups must stay 1)
    w = k[None, None, :].contiguous()
    out = torch.nn.functional.conv1d(y, w, padding=hw)
    return out.reshape(B, D, C).permute(0, 2, 1)


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
    """Per-sample ``(max_violation, mean_positive_violation, feasible_rate)``.

    A pack with no piece / no face (every sample of the batch failed activation)
    is a fully degenerate constraint set: ``g`` is ``[B,0,4,F]`` (or ``[B,P,4,0]``)
    and ``max`` on the empty axis would raise.  Such a batch is reported as
    "nothing violated, vacuously feasible"; ALM then returns ``q_ref`` untouched
    (the zero-size dual cannot move anything).
    """
    if g.shape[1] == 0 or g.shape[3] == 0:
        zero = g.new_zeros(g.shape[0])
        return zero, zero.clone(), torch.ones_like(zero)
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

    # Neighbour coupling ("drag the neighbours along"): a violated constraint
    # only needs ~4 control points to move, and with system = I + eta*hess
    # (eta = 0.05, smooth_weight = 0.1) the solve is essentially the identity,
    # so those 4 points get displaced IN PLACE and the curve kinks at them.
    # Smoothing the per-iteration step along the CONTROL axis makes the
    # displacement a smooth bell, so the neighbours follow with decay.
    smooth_kernel_half = int(cfg.get("correction_smooth_kernel", 0) or 0)
    smooth_sigma = float(cfg.get("correction_smooth_sigma", 0.0) or 0.0)

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
            # (temporarily disabled during the k4p_c48 investigation: with NO
            # bound at all the accumulated displacement diverges -- measured 26
            # samples of the 420-sample protocol ended up with NaN controls and
            # rmse = nan.  The cap is part of the convergence of the scheme, not
            # a safety extra: ``delta`` is an accumulator and the dual grows, so
            # each step's increment must stay bounded.)
            curve_step = torch.einsum("hk,bkd->bhd", basis, step)
            d_max = curve_step.norm(dim=-1).max(dim=-1).values           # [B]
            scale = torch.where(d_max > d_step,
                                d_step / d_max.clamp_min(1e-12),
                                torch.ones_like(d_max))
            step = step * scale[:, None, None]
        step = step * active.to(dtype)[:, None, None]
        if smooth_kernel_half > 0:
            step = _smooth_along_controls(step, smooth_kernel_half,
                                         smooth_sigma)
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
        lam_flat = lam.flatten(1)
        lam_max = (lam_flat.max(dim=1).values if lam_flat.shape[1] > 0
                   else lam.new_zeros(B))
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
            "lambda_max": lam_max,
            "inner_steps_used": used.to(dtype),
        }
    return q, lam, stats


@torch.no_grad()
def bspline_hard_project(q_ref: torch.Tensor, pack, codec,
                         config: dict | None = None,
                         tol: float = 1e-9):
    """HARD projection: put the final polygon inside the frozen constraint set.

    NOT another ALM sweep.  The guided phase runs a fixed budget and early-stops,
    so its output can still violate the pack (measured: 53/420 samples).  This
    solves the projection EXACTLY, as a linear program:

        min   sum(p) + sum(n)                       (L1 distance to q_ref)
        s.t.  M q <= v                              (every piece x Bezier x face)
              q - p + n = q_ref ,  p, n >= 0
              q_0 = start ,  q_{C-1} = goal         (endpoints pinned)

    with ``M`` the exact Bezier extraction stacked against the responsible cell
    faces: for piece ``l``, Bezier control ``r`` and face ``f`` the row is
        (A_lf . (E_lr @ q)) <= b_lf .
    L1 keeps the correction SPARSE (few control points move) instead of smearing
    it, and an LP is solved to optimality by HiGHS, so the result carries a
    feasibility certificate instead of a tolerance: after the solve the residual
    ``max(M q - v)`` is re-checked on the returned polygon.

    Samples that already satisfy the pack are left untouched; if the LP is
    infeasible (the pinned endpoints themselves violate their cells) the input is
    kept and flagged (``hard_project_feasible = False``).

    Returns ``(q_out, stats)``.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, hstack, eye as speye

    cfg = dict(config or {})
    scene_to_meter = float(cfg.get("scene_to_meter", SCENE_TO_METER))
    dtype, device = q_ref.dtype, q_ref.device
    B, C = int(q_ref.shape[0]), int(q_ref.shape[1])
    mask = pack.piece_mask[:, :, None, None] & pack.face_mask[:, :, None, :]
    _, g, _ = constraint_state(q_ref, pack)
    v_before, _, _ = _violation_stats(g, mask)

    q_out = q_ref.clone()
    basis = codec.basis.to(dtype=q_ref.dtype, device=q_ref.device)
    applied = torch.zeros(B, dtype=torch.bool, device=device)
    feasible = (v_before <= 0.0)
    status = ["not_needed"] * B

    for b in range(B):
        if bool(feasible[b]):
            continue
        P = int(pack.num_pieces[b].item())
        if P == 0:
            status[b] = "no_pack"
            continue
        E = pack.extraction[b, :P].detach().cpu().numpy().astype(np.float64)
        A = pack.piece_A[b, :P].detach().cpu().numpy().astype(np.float64)
        vb = pack.piece_b[b, :P].detach().cpu().numpy().astype(np.float64)
        fm = pack.face_mask[b, :P].detach().cpu().numpy().astype(bool)
        q0 = q_ref[b].detach().cpu().numpy().astype(np.float64)      # [C,2]

        pi, fi = np.nonzero(fm)                                     # [n]
        if len(pi) == 0:
            status[b] = "no_pack"
            continue
        # rows: (pair, r) -> one inequality per (piece, bezier control, face)
        coef = (A[pi, fi][:, None, :, None]
                * E[pi][:, :, None, :]).reshape(len(pi) * 4, 2 * C)  # [4n, 2C]
        rhs = np.repeat(vb[pi, fi], 4)                              # [4n]
        nrow = coef.shape[0]
        cols = np.tile(np.arange(2 * C), nrow)
        rows = np.repeat(np.arange(nrow), 2 * C)
        M = coo_matrix((coef.ravel(), (rows, cols)), shape=(nrow, 2 * C)).tocsr()

        qflat0 = np.concatenate([q0[:, 0], q0[:, 1]])                # [2C]
        nv = 6 * C
        A_ub = hstack([M, coo_matrix((nrow, 4 * C))]).tocsr()
        A_eq = hstack([speye(2 * C, format="csr"),
                       -speye(2 * C, format="csr"),
                       speye(2 * C, format="csr")]).tocsr()
        c = np.concatenate([np.zeros(2 * C), np.ones(4 * C)])
        bounds = [(None, None)] * (2 * C) + [(0, None)] * (4 * C)
        # endpoints are the CONDITION: pin them hard
        for ctrl in (0, C - 1):
            for coord in (0, 1):
                k = coord * C + ctrl
                bounds[k] = (qflat0[k], qflat0[k])
        res = linprog(c, A_ub=A_ub, b_ub=rhs, A_eq=A_eq, b_eq=qflat0,
                      bounds=bounds, method="highs")
        if not res.success or res.x is None:
            status[b] = "lp_infeasible(%s)" % res.status
            continue
        z = np.asarray(res.x, dtype=np.float64)
        qnew = np.stack([z[:C], z[C:2 * C]], axis=-1)                # [C,2]
        q_out[b] = torch.as_tensor(qnew, dtype=dtype, device=device)
        status[b] = "lp_optimal"
        applied[b] = True

    # ---- certificate: re-check the RETURNED polygon, and never regress -------
    _, g_new, _ = constraint_state(q_out, pack)
    v_after, _, _ = _violation_stats(g_new, mask)
    worse = v_after > v_before
    q_out = torch.where(worse[:, None, None], q_ref, q_out)
    v_final = torch.where(worse, v_before, v_after)
    applied &= ~worse
    corr = torch.einsum("hk,bkd->bhd", basis, q_out - q_ref).norm(dim=-1)
    stats = {
        "hard_project_applied": applied,
        "hard_project_status": status,
        "hard_project_violation_before": v_before,
        "hard_project_violation_after": v_after,
        "hard_project_violation": v_final,
        "hard_project_feasible": v_final <= float(tol),
        "hard_project_correction_scene": corr.max(dim=-1).values,
        "hard_project_correction_m": corr.max(dim=-1).values * scene_to_meter,
    }
    return q_out, stats
