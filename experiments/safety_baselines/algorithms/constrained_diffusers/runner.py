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


def _violation(task, x, scale, clip=None):
    """Official `calc_grad`: g and grad g of every row, at the current x_t.

    `vio` is signed the official way (positive = safe), so `grad` is `-J` of our adapters
    and no sign-fix is needed inside the vendored update.

    `clip` (scaled units) saturates how *deep* a violation counts. At high t the state is a
    random N(0,1) point cloud (`x = 1.9e-4*x0 + 1.0*eps`), so points fall outside the map and
    the distance field reports hundreds of metres for them; one grid cell is the depth beyond
    which "inside an obstacle" carries no extra signal, exactly as the release's
    dimensionless barrier saturates at one obstacle radius.
    """
    rows = [task.constraints(p) for p in x]
    vio = -np.stack([r[0] for r in rows]).astype(float) * scale
    if clip is not None:
        vio = np.maximum(vio, -float(clip))
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
    _prev_correction = None
    ladder = level_ladder(T, cfg.get('cd_extra_steps', 0))
    repeats = len(ladder) - len(set(ladder))
    # `cd_correct_below`: only engage the constraint in the low-noise tail. At t=255 our
    # forward process gives x = 1.9e-4*x0 + 1.0*eps, i.e. the "trajectory" is a random
    # N(0,1) point cloud over a map that is only 2 scene units wide (~60% of the points land
    # outside the free area), and the barrier reports hundreds of metres of "violation" for
    # it. Letting the multiplier integrate that ratchets it up before any geometry exists.
    # PDM (`idx > 5`) and SafeDiffuser (`t <= 10`) both ship exactly this gate; the
    # Constrained Diffusers release does not need it because its sigma_max is only a few
    # times its data range.
    gate = cfg.get('cd_correct_below')

    for t in ladder:
        budget.check()
        x0 = np.asarray(task.clean(x, t), dtype=float).reshape(particles, -1)
        mean = coef1[t] * x0 + coef2[t] * x
        corr_norm, cos_prev, lam_max = 0.0, float('nan'), 0.0
        viol_before = viol_after = float('nan')
        gated = gate is not None and t > int(gate)
        if method in ('Primal-Dual', 'ALM'):
            if not gated:
                vio, grad = _violation(task, x, scale, cfg.get('cd_violation_clip'))
                before = mean.copy()
                # does the correction actually reduce the quantity it penalises? measured on
                # the *rows* it can touch, in scene units (the task's own barrier).
                viol_before = float(np.max(-vio/scale))
                if method == 'Primal-Dual':
                    mean, lam = official.primal_dual(mean, vio, grad, lam, learning_rate)
                else:
                    mean, slack, lam, rho = official.augmented_lagrangian(
                        mean, vio, grad, lam, slack, rho, growth)
                correction = mean - before
                viol_after = float(np.max(-_violation(task, mean, scale)[0]/scale))
                corr_norm = float(np.linalg.norm(correction))
                lam_max = float(lam.max())
                # cos < 0 on consecutive steps means the push reverses direction: the signature
                # of a multiplier that stays positive while grad g flips as the point crosses
                # the boundary, i.e. a zig-zag, not a magnitude problem.
                if _prev_correction is not None:
                    denom = corr_norm*_prev_correction[0] + 1e-30
                    cos_prev = float(np.sum(correction*_prev_correction[1])/denom)
                _prev_correction = (corr_norm, correction.copy())
        elif method != 'Projected':
            raise ValueError('not a Constrained Diffusers method: ' + method)
        task.last_nominal = mean[0].copy()
        std = 0.0 if is_noiseless(t) else float(np.sqrt(post_var[t]))
        previous = x
        x = mean + std * rng.normal(size=mean.shape)
        if method == 'Projected':
            x = official.projected(task, x, budget)
        trace.append({'t': int(t), 'optimizer': method, 'particles': particles,
                      'lambda_min': float(lam.min()), 'lambda_max': lam_max,
                      'rho_next': float(rho), 'slack_norm': float(np.linalg.norm(slack)),
                      'noise_std': std, 'noiseless': bool(std == 0.0),
                      'correction_norm': corr_norm, 'correction_cos_prev': cos_prev,
                      'violation_before': viol_before, 'violation_after': viol_after,
                      'step_norm': float(np.linalg.norm(x - previous)),
                      'ladder_length': len(ladder), 'ladder_repeats': repeats})
    if not np.isfinite(x).all():
        raise RuntimeError('nonfinite_langevin')
    return x[0]
