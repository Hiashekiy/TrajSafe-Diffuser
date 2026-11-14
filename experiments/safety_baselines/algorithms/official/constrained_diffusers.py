"""Constrained Diffusers corrections — vendored from the official supplement.

Source: `.refs/NeurIPS-2025-constrained-diffusers-for-safe-planning-and-control-Supplemental-Conference/code/umaze-large/diffuser/models/diffusion.py`

    q_posterior      primal_dual branch          188-202
                     augmented_lagrangian branch 203-224
    calc_grad        autograd, no explicit J     306-322
    dual_update      lam <- clamp(lam - lr*vio, 0)   325-348   (lr = 1e-3 here;
                     umaze-small uses 2.5e-2, Hopper/swimmer/BallRunning 1e-1)
    dual_update_aug  slack, lam, rho <- rho*1.004    350-380
    _project_to_feasible_region   closed-form radial projection   382-466

Sign convention. The official code writes `g_x = sqrt(...) - 1`, so **positive is safe**
and the multipliers are clamped at zero. Our adapters use `g <= 0` feasible. Everything
below therefore takes `vio = -g_ours` and `grad = -J_ours`, which makes the updates
identical to the official lines rather than "equivalent up to a sign I had to reason
about". `tests/test_official_constrained_diffusers.py` checks the sign against the
official formulas rewritten independently.

Substituted, declared: the projection operator is the task's own nearest-point map
(official projects each violating point radially onto a hardcoded plate ellipses,
`382-466`; an occupancy map has no closed form). Nothing else differs — in particular
there is no hinge (official updates on the raw `vio`, Remark 1 is a paper claim, not
release code), no averaging over particles (official keeps one multiplier per
`(constraint, batch, timestep)`, `95-99`) and no scaling of the learning rate.
"""
import numpy as np

from experiments.safety_baselines.algorithms.kernels import row_contribution


def primal_dual(mean, vio, grad, lam, learning_rate):
    """`posterior_mean += sum_i lam_i grad_i`, then `lam <- clamp(lam - lr*vio, min=0)`.

    `lam` and `vio` are [particles, rows]; `grad` is the dense [particles, rows, D]
    tensor or a per-particle list of sparse rows (never both dense, see A5).
    """
    updated_mean = mean + row_contribution(grad, lam)
    lam_next = np.maximum(0.0, lam - learning_rate*np.asarray(vio, dtype=float))
    return updated_mean, lam_next


def augmented_lagrangian(mean, vio, grad, lam, slack, rho, growth):
    """`mean -= sum lam*grad + rho*sum(vio - slack)*grad`, then the dual/slack/rho update.

    Order matters and follows the official lines: the correction uses the multipliers and
    slack of the *previous* step (`222-223`), then `dual_update_aug` (`358-364`) moves
    `slack <- clamp(lam/rho + vio, 0)`, `lam <- lam + rho*(vio - slack)`, `rho *= 1.004`.
    Their default branch (`cbf=False`) is the one used here; the `1.004` growth lives in
    the other branch, so `rho` is constant apart from the caller's schedule.
    """
    residual = np.asarray(vio, dtype=float) - slack
    coeff = lam + rho*residual
    updated_mean = mean - row_contribution(grad, coeff)
    slack_next = np.maximum(0.0, lam/rho + vio)
    lam_next = lam + rho*(np.asarray(vio, dtype=float) - slack_next)
    return updated_mean, slack_next, lam_next, rho*growth


def projected(task, x, budget):
    """`_project_to_feasible_region`: project the sampled next state onto the safe set.

    Official form (`382-466`): every point with `vio < 0` is scaled radially onto the
    plate ellipse boundary, every other point is left untouched, no solver, one pass.
    Our map exposes the same operation as its nearest-point projection at the vertices;
    an already feasible point is returned unchanged by both.
    """
    out = np.empty_like(np.asarray(x, dtype=float))
    for i, particle in enumerate(np.asarray(x, dtype=float)):
        value, _ = task.project(particle, budget)
        out[i] = value
    return out
