"""SafeDiffuser (ICLR 2025) — RoS / ReS / TVS with the correction taken from the official code.

The correction itself lives in `algorithms/official/safediffuser.py`, vendored from
`.refs/SafeDiffuser/diffuser/models/diffusion.py`. This file is only the loop around it,
and the loop is the official `p_sample` / `p_sample_loop`
(`diffusion.py:1030-1097`, `1100-1131`):

    x0    = model(x_t)                                    predict_start_from_noise
    mean  = coef1_t * x0 + coef2_t * x_t                  q_posterior
    xp1   = mean + sqrt(posterior_var_t) * z              p_sample (no noise at t = 0)
    out   = CBF_QP(b(x_t), grad_b(x_t), xp1 - x_t)         the vendored correction
    x_{t-1} = x_t + out                                   position only

Deleted on 2026-11-14 (they were ours, not the paper's): the velocity-field integrator
with `cbf_dt` substeps, the discrete certificate with dt halving and level retries, the
per-point step cap, the high-noise residual gate and the `legal_gamma` floor. The paper
and its release do none of these; see docs/official_vs_ours_diff.md §C.

Model and scene are ours: `task.clean` is our safety-point denoiser and
`task.point_barrier` is our occupancy map, so the base model is untouched. Sampling is
DDPM ancestral because that is what the correction is defined against — the three
official variants all read `xp1`, the *sampled* next state, and both the ReS hard
interval and the TVS level are indexed by the noise level that produced it.
"""
import numpy as np

from experiments.safety_baselines.algorithms.official import safediffuser as official


def ddpm_coefficients(ab, betas):
    """Official `posterior_mean_coef1/2` and `posterior_variance` (diffusion.py:117-127)."""
    ab_prev = np.concatenate([[1.0], np.asarray(ab, dtype=float)[:-1]])
    ab = np.asarray(ab, dtype=float)
    betas = np.asarray(betas, dtype=float)
    var = betas * (1.0 - ab_prev) / (1.0 - ab)
    coef1 = betas * np.sqrt(ab_prev) / (1.0 - ab)
    coef2 = (1.0 - ab_prev) * np.sqrt(1.0 - betas) / (1.0 - ab)
    return coef1, coef2, var


def run(task, method, rng, cfg, budget, trace):
    if hasattr(task, 'segment_map') or not callable(getattr(task, 'point_barrier', None)):
        raise ValueError('%s is the paper per-point CBF (diffusion.py:451-956 constrains one '
                         'position per timestep); a segment-row task has no counterpart in the '
                         'official code' % method)
    ab = np.asarray(task.ab, dtype=float)
    T = len(ab)
    defaults = official.official_schedule_defaults(T)
    settings = {k: cfg.get(k, v) for k, v in defaults.items()}
    # `p_sample` ships a second, commented-out policy (diffusion.py:1071-1078):
    #   if t <= 10:  x = self.invariance_relax(x, xp1, t)     # correct
    #   else:        x = xp1                                  # leave the sample alone
    # i.e. apply the correction only in the low-noise tail. On their O(1)-scale data the
    # unconditional call is harmless; on our schedule the top levels sit at sigma ~ 5e3
    # scene units, where a hard half-space constraint on a geometric barrier is meaningless
    # and the sample explodes. Default = the rescaled 10-of-1000 threshold; set
    # `sd_apply_below: null` for the unconditional call.
    apply_below = cfg.get('sd_apply_below', settings['sd_hard_levels'])
    # One "obstacle radius" of the release's dimensionless barrier, in scene units. The
    # exact map spans `extent` over its grid, so one cell is the scene analogue of their
    # unit circle; only TVS needs it (it subtracts a level from the barrier).
    level_scale = float(cfg.get('sd_tvs_level_scene', 1.0))
    coef1, coef2, post_var = ddpm_coefficients(ab, task.betas)
    slack_weight = float(cfg.get('relax_weight', 100.0))

    x = np.asarray(task.initial(rng)[0], dtype=float)
    for t in range(T - 1, -1, -1):
        budget.check()
        x0 = np.asarray(task.clean(x, t)[0], dtype=float)
        mean = coef1[t] * x0 + coef2[t] * x
        xp1 = mean if t == 0 else mean + np.sqrt(post_var[t]) * rng.normal(size=x.shape)
        corrected = apply_below is None or t <= apply_below
        if corrected:
            b, gb = task.point_barrier(x)
            ref = (xp1 - x).reshape(-1, 2)
            if method == 'RoS-diffuser':
                u = official.ros(b, gb, ref)
                slack = 0.0
            elif method == 'ReS-diffuser':
                slack = 0.0 if t < settings['sd_hard_levels'] else slack_weight
                u = official.res(b, gb, ref, t, settings['sd_hard_levels'], slack_weight)
            elif method == 'TVS-diffuser':
                u = official.tvs(b, gb, ref, t, settings['sd_tvs_t_bias'], level_scale)
                slack = 0.0
            else:
                raise ValueError('not a SafeDiffuser variant: ' + method)
            x = x + u.reshape(-1)
            min_barrier, corr = float(np.min(b)), float(np.linalg.norm(u))
        else:
            x = xp1
            slack, min_barrier, corr = 0.0, float('nan'), 0.0
        trace.append({'t': t, 'optimizer': 'official_cbf_qp', 'corrected': bool(corrected),
                      'sd_apply_below': apply_below,
                      'slack_weight': float(slack), 'min_barrier': min_barrier,
                      'correction_norm': corr, **settings})
    return x
