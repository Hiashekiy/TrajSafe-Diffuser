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


def alm_correct(p0, A, b, face_mask, valid, lam, rho,
                step_size=0.03, inner_steps=4,
                max_grad_norm=1.0, max_correction_per_step=0.10,
                collision_fn=None, collect_stats=False,
                trace_callback=None):
    """Correct ``p0`` while holding one set of segment corridors fixed.

    Only the inequality augmented-Lagrangian force is applied. ``lam`` belongs
    to this fixed corridor set; the sampler resets it whenever corridors are
    rebuilt.
    """
    if rho <= 0:
        raise ValueError("rho must be positive")
    p = p0.clone()
    start, goal = p0[:, :1].clone(), p0[:, -1:].clone()
    lam = lam.clone()

    def current_collision(trajectory):
        collision = (torch.ones_like(valid) if collision_fn is None
                     else collision_fn(trajectory).bool())
        if collision.shape != valid.shape:
            raise ValueError("collision_fn must return shape [B,H-1]")
        return collision

    initial_collision = current_collision(p)
    if trace_callback is not None:
        trace_callback(0, p)

    g_before, _, _ = _constraint_state(p, A, b, face_mask, valid)

    for _ in range(max(0, int(inner_steps))):
        collision_mask = current_collision(p)
        enforced = valid & collision_mask
        lam = torch.where(enforced, lam, torch.zeros_like(lam))
        g, grad_left, grad_right = _constraint_state(p, A, b, face_mask, valid)

        # Inequality ALM after eliminating the non-negative slack. A feasible
        # constraint with a zero multiplier produces exactly zero force.
        omega = torch.relu(lam + float(rho) * g)
        omega = torch.where(enforced, omega, torch.zeros_like(omega))

        grad = torch.zeros_like(p)
        grad[:, :-1] += omega[..., None] * grad_left
        grad[:, 1:] += omega[..., None] * grad_right
        grad[:, 0] = 0.0
        grad[:, -1] = 0.0
        if max_grad_norm > 0:
            norm = grad.norm(dim=-1, keepdim=True)
            grad = grad * (float(max_grad_norm) / norm.clamp_min(1e-8)).clamp(max=1.0)

        delta = -float(step_size) * grad
        if max_correction_per_step > 0:
            delta_norm = delta.norm(dim=-1, keepdim=True)
            delta = delta * (float(max_correction_per_step) /
                             delta_norm.clamp_min(1e-8)).clamp(max=1.0)
        p = p + delta
        p[:, :1] = start
        p[:, -1:] = goal
        if trace_callback is not None:
            trace_callback(_ + 1, p)

        collision_after = current_collision(p)
        enforced_after = valid & collision_after
        g_new, _, _ = _constraint_state(p, A, b, face_mask, valid)
        lam = torch.relu(lam + float(rho) * g_new)
        lam = torch.where(enforced_after, lam, torch.zeros_like(lam))

    stats = None
    if collect_stats:
        final_collision = current_collision(p)
        g_after, _, _ = _constraint_state(p, A, b, face_mask, valid)
        correction = (p - p0).norm(dim=-1)
        valid_float = valid.float()
        valid_count = valid_float.sum().clamp_min(1.0)
        positive_before = (g_before > 0) & valid
        positive_after = (g_after > 0) & valid
        collision_count = initial_collision.float().sum().clamp_min(1.0)
        initially_enforced = valid & initial_collision
        stats = {
            "corridor_valid_rate": valid_float.mean(),
            "physical_guidance_rate": initially_enforced.float().mean(),
            "physical_collision_rate": initial_collision.float().mean(),
            "physical_collision_rate_before": initial_collision.float().mean(),
            "physical_collision_rate_after": final_collision.float().mean(),
            "new_physical_collision_rate": (
                final_collision & ~initial_collision).float().mean(),
            "resolved_physical_collision_rate": (
                initial_collision & ~final_collision).float().mean(),
            "collision_but_invalid_rate": (
                initial_collision & ~valid).float().mean(),
            "collision_covered_rate": (
                initially_enforced.float().sum() / collision_count),
            "collision_inside_region_rate": (
                final_collision & valid & (g_after <= 0)).float().mean(),
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
        }
    return p, lam, stats
