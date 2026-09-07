"""Analytic augmented-Lagrangian correction for clean trajectory predictions.

The inequality aggregation is an exact maximum. An unnormalised log-sum-exp
adds ``tau * log(number_of_faces)`` and can turn a strictly feasible segment
into a false positive. Since inference does not use autograd, the active face
supplies an exact and inexpensive subgradient.
"""
from __future__ import annotations

import torch


def _constraint_state(p, A, b, face_mask, valid):
    """Constrain incoming segment i with the convex region of endpoint i."""
    left = (A * p[:, :-1, None]).sum(dim=-1) - b
    right = (A * p[:, 1:, None]).sum(dim=-1) - b
    violation = torch.cat((left, right), dim=-1)
    constraint_mask = torch.cat((face_mask, face_mask), dim=-1)
    masked = violation.masked_fill(~constraint_mask, -torch.inf)
    g, active_index = masked.max(dim=-1)
    g = torch.where(valid, g, torch.zeros_like(g))
    faces = A.shape[2]
    active_face = active_index.remainder(faces)
    active_normal = A.gather(
        2, active_face[..., None, None].expand(-1, -1, 1, 2),
    ).squeeze(2)
    left_active = (active_index < faces) & valid
    right_active = ~left_active & valid
    return (g, active_normal * left_active[..., None],
            active_normal * right_active[..., None])


def _second_difference(p):
    return p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]


def _correction_smoother(horizon, step_size, weight, reference):
    """Cholesky factor for the proximal correction-smoothness solve."""
    if weight <= 0 or horizon < 3:
        return None
    # Endpoints of the correction are fixed at zero. D maps the H-2 interior
    # corrections to the H-2 trajectory second differences.
    n = horizon - 2
    D = reference.new_zeros((n, n))
    row = torch.arange(n, device=reference.device)
    D[row, row] = -2.0
    if n > 1:
        D[row[1:], row[1:] - 1] = 1.0
        D[row[:-1], row[:-1] + 1] = 1.0
    system = (torch.eye(n, dtype=reference.dtype, device=reference.device) +
              float(step_size) * float(weight) * (D.T @ D))
    return torch.linalg.cholesky(system)


def alm_correct(p0, A, b, face_mask, valid, lam, rho,
                step_size=0.03, inner_steps=4,
                max_grad_norm=1.0, max_correction_per_step=0.10,
                proximity_weight=1.0, correction_smooth_weight=4.0,
                enforce_mask=None, collect_stats=False):
    """Correct ``p0`` while holding one set of segment corridors fixed.

    The primal objective combines the inequality augmented Lagrangian with a
    proximal term and smoothness of the correction field. The latter avoids
    isolated waypoint spikes without smoothing or moving a fully feasible raw
    diffusion trajectory. ``lam`` belongs only to this fixed corridor set; the
    sampler resets it whenever corridors are rebuilt.
    """
    if rho <= 0:
        raise ValueError("rho must be positive")
    p = p0.clone()
    start, goal = p0[:, :1].clone(), p0[:, -1:].clone()
    lam = lam.clone()
    enforced = valid if enforce_mask is None else (valid & enforce_mask)
    smooth_factor = _correction_smoother(
        p.shape[1], step_size, correction_smooth_weight, p)

    g_before, _, _ = _constraint_state(p, A, b, face_mask, valid)
    smoothness_before = _second_difference(p).norm(dim=-1).mean()

    for _ in range(max(0, int(inner_steps))):
        g, grad_left, grad_right = _constraint_state(p, A, b, face_mask, valid)

        # Inequality ALM after eliminating the non-negative slack. A feasible
        # constraint with a zero multiplier produces exactly zero force.
        omega = torch.relu(lam + float(rho) * g)
        omega = torch.where(enforced, omega, torch.zeros_like(omega))

        correction = p - p0
        grad = float(proximity_weight) * correction
        grad[:, :-1] += omega[..., None] * grad_left
        grad[:, 1:] += omega[..., None] * grad_right
        grad[:, 0] = 0.0
        grad[:, -1] = 0.0
        if max_grad_norm > 0:
            norm = grad.norm(dim=-1, keepdim=True)
            grad = grad * (float(max_grad_norm) / norm.clamp_min(1e-8)).clamp(max=1.0)

        # Proximal-gradient update: take the ALM/proximity step, then solve the
        # correction smoothness term exactly. This avoids the unstable explicit
        # D2^T D2 step that created alternating waypoint spikes.
        target_correction = correction - float(step_size) * grad
        target_correction[:, 0] = 0.0
        target_correction[:, -1] = 0.0
        if smooth_factor is not None:
            rhs = target_correction[:, 1:-1]
            factor = smooth_factor[None].expand(p.shape[0], -1, -1)
            target_correction[:, 1:-1] = torch.cholesky_solve(rhs, factor)
        delta = target_correction - correction
        if max_correction_per_step > 0:
            delta_norm = delta.norm(dim=-1, keepdim=True)
            delta = delta * (float(max_correction_per_step) /
                             delta_norm.clamp_min(1e-8)).clamp(max=1.0)
        p = p + delta
        p[:, :1] = start
        p[:, -1:] = goal

        g_new, _, _ = _constraint_state(p, A, b, face_mask, valid)
        lam = torch.relu(lam + float(rho) * g_new)
        lam = torch.where(enforced, lam, torch.zeros_like(lam))

    stats = None
    if collect_stats:
        g_after, _, _ = _constraint_state(p, A, b, face_mask, valid)
        correction = (p - p0).norm(dim=-1)
        valid_float = valid.float()
        valid_count = valid_float.sum().clamp_min(1.0)
        positive_before = (g_before > 0) & valid
        positive_after = (g_after > 0) & valid
        stats = {
            "corridor_valid_rate": valid_float.mean(),
            "physical_guidance_rate": enforced.float().mean(),
            "raw_max_positive_rate": positive_before.sum() / valid_count,
            "segment_endpoint_pair_inside_rate": (
                ((g_before <= 0) & valid).sum() / valid_count),
            "violation_before": (torch.relu(g_before) * valid_float).sum() / valid_count,
            "violation_after": (torch.relu(g_after) * valid_float).sum() / valid_count,
            "max_violation_after": (torch.relu(g_after) * valid_float).max(),
            "segment_feasible_rate": ((g_after <= 0) & valid).sum() / valid_count,
            "active_constraint_rate": positive_after.sum() / valid_count,
            "lambda_mean": lam.sum() / valid_count,
            "lambda_max": lam.max(),
            "mean_correction": correction.mean(),
            "max_correction": correction.max(),
            "smoothness_before": smoothness_before,
            "smoothness_after": _second_difference(p).norm(dim=-1).mean(),
        }
    return p, lam, stats
