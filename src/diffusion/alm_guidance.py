"""Analytic augmented-Lagrangian correction for clean trajectory predictions."""
from __future__ import annotations

import torch


def _constraint_state(p, A, b, face_mask, valid, tau):
    """Return conservative segment constraints and their analytic gradients."""
    left = (A * p[:, :-1, None]).sum(dim=-1) - b
    right = (A * p[:, 1:, None]).sum(dim=-1) - b
    violation = torch.cat((left, right), dim=-1)
    constraint_mask = torch.cat((face_mask, face_mask), dim=-1)
    logits = (violation / tau).masked_fill(~constraint_mask, -torch.inf)
    g = tau * torch.logsumexp(logits, dim=-1)
    g = torch.where(valid, g, torch.zeros_like(g))

    weights = torch.softmax(logits, dim=-1)
    weights = torch.where(constraint_mask, weights, torch.zeros_like(weights))
    weights = torch.where(valid[..., None], weights, torch.zeros_like(weights))
    faces = A.shape[2]
    grad_left = (weights[..., :faces, None] * A).sum(dim=2)
    grad_right = (weights[..., faces:, None] * A).sum(dim=2)

    raw_max = violation.masked_fill(~constraint_mask, -torch.inf).max(dim=-1).values
    raw_max = torch.where(valid, raw_max, torch.zeros_like(raw_max))
    return g, grad_left, grad_right, raw_max


def alm_correct(p0, A, b, face_mask, valid, lam, rho,
                step_size=0.03, smooth_tau=0.03, inner_steps=2,
                max_grad_norm=1.0, max_correction_per_step=0.10,
                collect_stats=False):
    """Correct ``p0`` while holding the supplied corridors fixed.

    The inequality for segment ``k`` is a conservative smooth maximum over all
    left/right endpoint face violations.  One multiplier is retained per
    segment, so its identity remains meaningful when faces change at the next
    reverse step.  No autograd graph is created; the smooth-max gradient is
    evaluated analytically with batched tensor operations.
    """
    if smooth_tau <= 0:
        raise ValueError("smooth_tau must be positive")
    if rho <= 0:
        raise ValueError("rho must be positive")
    p = p0.clone()
    start, goal = p0[:, :1].clone(), p0[:, -1:].clone()
    lam = lam.clone()

    if collect_stats:
        _, _, _, raw_before = _constraint_state(
            p, A, b, face_mask, valid, smooth_tau)

    for _ in range(max(0, int(inner_steps))):
        g, grad_left, grad_right, _ = _constraint_state(
            p, A, b, face_mask, valid, smooth_tau)
        slack = torch.relu(-g - lam / rho)
        residual = g + slack
        omega = (lam + rho * residual).clamp_min(0.0)
        omega = torch.where(valid, omega, torch.zeros_like(omega))

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

        g_new, _, _, _ = _constraint_state(
            p, A, b, face_mask, valid, smooth_tau)
        slack_new = torch.relu(-g_new - lam / rho)
        residual_new = g_new + slack_new
        lam = (lam + rho * residual_new).clamp_min(0.0)
        lam = torch.where(valid, lam, torch.zeros_like(lam))

    stats = None
    if collect_stats:
        _, _, _, raw_after = _constraint_state(
            p, A, b, face_mask, valid, smooth_tau)
        correction = (p - p0).norm(dim=-1)
        valid_count = valid.sum().clamp_min(1)
        stats = {
            "corridor_valid_rate": valid.float().mean(),
            "violation_before": torch.relu(raw_before)[valid].mean()
                if bool(valid.any()) else p.new_zeros(()),
            "violation_after": torch.relu(raw_after)[valid].mean()
                if bool(valid.any()) else p.new_zeros(()),
            "max_violation_after": torch.relu(raw_after)[valid].max()
                if bool(valid.any()) else p.new_zeros(()),
            "segment_feasible_rate": ((raw_after <= 0) & valid).sum() / valid_count,
            "active_constraint_rate": ((raw_after > 0) & valid).sum() / valid_count,
            "mean_correction": correction.mean(),
            "max_correction": correction.max(),
        }
    return p, lam, stats
