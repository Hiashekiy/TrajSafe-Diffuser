"""Constrained Diffusers (NeurIPS 2025) — Projected / Primal-Dual / ALM.

The corrections live in `algorithms/official/constrained_diffusers.py`, vendored from
`.refs/NeurIPS-2025-.../code/umaze-large/diffuser/models/diffusion.py`. This file is the
official reverse chain around them (`p_mean_variance` / `q_posterior` / `p_sample`,
`diffusion.py:232-242`, `178-230`, `468-487`, and `p_sample_loop` `490-531`):

    x0     = model(x_t)                         predict_start_from_noise
    mean   = coef1_t*x0 + coef2_t*x_t           q_posterior
    mean  += correction(vio(x_t), grad(x_t))    Primal-Dual / ALM only, official 188-224
    xp1    = mean + std_t * z                   std_t = 0 at t == 1   (479)
    xp1    = project(xp1)                       Projected only         (481-482)
    ladder = range(-200, T) with i<=0 -> 1      i.e. 200 noiseless revisits of t=1

Deleted here (ours, not the paper's): the `cd_beta` level-vs-interval aggregate, the
per-particle averaging of the violation, the hinge `J*(g>0)`, and the extra steps placed
at t=0 with their own noise draw. See docs/official_vs_ours_diff.md §D.

Multipliers are per particle and per row (`dual_vars[C, B, horizon]`, `95-99`); the
`particles` config therefore acts as the official batch axis, and particle 0 is returned
(the fixed-particle protocol of `algorithms/runner.py`, no selection).
"""
import numpy as np
from scipy import sparse

from experiments.safety_baselines.algorithms.official import constrained_diffusers as official
from experiments.safety_baselines.algorithms.official.posterior import (
    is_noiseless, level_ladder, posterior_coefficients)


def _violation(task, x, scale):
    """Official `calc_grad`: g and grad g of every row, at the current x_t.

    `vio` is signed the official way (positive = safe), so `grad` is `-J` of our adapters
    and no sign fix-up is needed inside the vendored update.
    """
    rows = [task.constraints(p) for p in x]
    vio = -np.stack([r[0] for r in rows]).astype(float) * scale
    js = [r[1] for r in rows]
    if sparse.issparse(js[0]):
        grad = [-M * scale for M in js]
    else:
        grad = -np.stack([np.asarray(M, dtype=float) for M in js]) * scale
    return vio, grad


def run(task, method, rng, cfg, budget, trace):
    ab = np.asarray(task.ab, dtype=float)
    T = len(ab)
    coef1, coef2, post_var = posterior_coefficients(ab, task.betas)
    particles = int(cfg.get('particles', 1)) if method in ('Primal-Dual', 'ALM') else 1
    scale = float(cfg.get('dual_constraint_scale', 1.0))
    learning_rate = float(cfg['dual_eta'])
    rho = float(cfg['rho']); growth = float(cfg.get('rho_growth', 1.0))

    x = np.asarray(task.initial(rng, particles), dtype=float).reshape(particles, -1)
    rows = len(task.constraints(x[0])[0])
    lam = np.zeros((particles, rows))
    slack = np.zeros((particles, rows))
    ladder = level_ladder(T, cfg.get('cd_extra_steps', 0))
    repeats = len(ladder) - len(set(ladder))

    for t in ladder:
        budget.check()
        x0 = np.asarray(task.clean(x, t), dtype=float).reshape(particles, -1)
        mean = coef1[t] * x0 + coef2[t] * x
        if method in ('Primal-Dual', 'ALM'):
            vio, grad = _violation(task, x, scale)
            if method == 'Primal-Dual':
                mean, lam = official.primal_dual(mean, vio, grad, lam, learning_rate)
            else:
                mean, slack, lam, rho = official.augmented_lagrangian(
                    mean, vio, grad, lam, slack, rho, growth)
        elif method != 'Projected':
            raise ValueError('not a Constrained Diffusers method: ' + method)
        task.last_nominal = mean[0].copy()
        std = 0.0 if is_noiseless(t) else float(np.sqrt(post_var[t]))
        x = mean + std * rng.normal(size=mean.shape)
        if method == 'Projected':
            x = official.projected(task, x, budget)
        trace.append({'t': int(t), 'optimizer': method, 'particles': particles,
                      'lambda_min': float(lam.min()), 'lambda_max': float(lam.max()),
                      'rho_next': float(rho), 'slack_norm': float(np.linalg.norm(slack)),
                      'noise_std': std, 'noiseless': bool(std == 0.0),
                      'ladder_length': len(ladder), 'ladder_repeats': repeats})
    if not np.isfinite(x).all():
        raise RuntimeError('nonfinite_langevin')
    return x[0]
