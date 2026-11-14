"""SafeDiffuser correction — vendored verbatim from the official release.

Source: `.refs/SafeDiffuser/diffuser/models/diffusion.py`

    invariance        RoS, maze2d-large    lines 451-512
    invariance_relax  ReS, maze2d-large    lines 588-658
    invariance_time   TVS, maze2d-large    lines 895-956

Copied unchanged (this is the method):

    barrier        h = Lfb + k*b, k = 1                                (471-474, 620-621)
    QP             G = -grad_b, q = -(xp1 - x), Q = I                  (471-474, 501-503)
    ReS slack      sign = 100 when t >= 10 else 0, one slack column
                   per constraint row, Q = I2                             (610-616, 649-656)
    TVS level      sigma = sigmoid(t_bias - t), b = barrier - sigma,
                   Lfb = sigma*(1 - sigma), t_bias = 5                    (911-921)
    applied state  x_new = x + out, position only                        (508-509)

Substituted, declared rather than silent (see docs/official_vs_ours_diff.md):

  * `b` and `grad_b` are arguments. The official file hardcodes the two plate ellipses
    of maze2d-large (`diffusion.py:459-490`); here the same QP runs on our occupancy
    map, which is the "场景用我们的" half of the split.
  * QP backend. Official solves with `qpth.QPFunction(QPSolvers.PDIPM_BATCHED)`
    (`diffusion.py:506`); qpth is not a dependency of this repo. The problem statement
    is not changed: with identity Q, one row per point and at most one slack column it
    has the closed form

        alpha = max(0, -(h + grad_b . ref) / (||grad_b||^2 + s^2))
        u     = ref + alpha * grad_b,        r = -s * alpha

    which `tests/test_official_safediffuser.py` checks against a generic solver.

Nothing else was added: no projection gate, no discrete certificate, no substep
integration, no step cap, no retries. One QP per diffusion level, as in the paper's
`p_sample` (`diffusion.py:1030-1097`).
"""
import numpy as np

from experiments.safety_baselines.algorithms.solvers import SolverFailure


def _qp(b, gb, ref, offset, s):
    """min 0.5||u-ref||^2 + 0.5*||r||^2  s.t.  -gb.u + s*r <= offset + b.

    `s` is the official `sign` (0 for RoS/TVS and the ReS hard interval, 100 for the
    ReS relaxed interval). Exact solution of the official QP, see the module docstring.
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    gb = np.asarray(gb, dtype=float).reshape(len(b), 2)
    ref = np.asarray(ref, dtype=float).reshape(len(b), 2)
    h = offset + b if np.isscalar(offset) else np.asarray(offset, dtype=float).reshape(-1) + b
    gb_ref = np.einsum('pd,pd->p', gb, ref)
    norm = np.einsum('pd,pd->p', gb, gb) + float(s) ** 2
    alpha = np.maximum(0.0, -(h + gb_ref) / np.maximum(norm, 1e-30))
    blocked = (norm <= 1e-30) & (h + gb_ref < 0.0)
    if blocked.any():                       # the official QP is infeasible on these rows
        raise SolverFailure('cbf_infeasible_zero_gradient:rows=%d' % int(blocked.sum()))
    return ref + alpha[:, None] * gb


def ros(b, gb, ref, slack=0.0):
    """RoS-diffuser: `invariance`, no slack, no time-varying level (451-512)."""
    return _qp(b, gb, ref, 0.0, slack)


def res(b, gb, ref, t, hard_levels, slack_weight=100.0):
    """ReS-diffuser: slack only outside the hard interval (588-658).

    Official condition is `if t >= 10: sign = 100 else: sign = 0` on a 1000-step chain.
    `t` here is the level of *our* chain and `hard_levels` carries the same interval.
    """
    return _qp(b, gb, ref, 0.0, 0.0 if t < hard_levels else float(slack_weight))


def tvs(b, gb, ref, t, t_bias, level_scale=1.0):
    """TVS-diffuser: time-varying level inside the barrier (895-956).

    `level_scale` converts the level to the units of `b`. In the release `b` is
    dimensionless (`sqrt(...) - 1`, so 1 = the obstacle half-width, `diffusion.py:911`) and
    sigma in [0,1] is one obstacle radius. Our `b` is the map distance in scene units, so
    the caller passes the scene length that plays the role of "one obstacle radius";
    `level_scale=1.0` is the release's own arithmetic.
    """
    s = 1.0 / (1.0 + np.exp(-(t_bias - t)))
    return _qp(b - level_scale*s, gb, ref, level_scale*s*(1.0 - s), 0.0)


def official_schedule_defaults(num_timesteps):
    """Rescale the two literal constants of the official 1000-step chain.

    `t_bias = 5` (line 896) and `t >= 10` (line 610) are step indices of their 1000-step
    chain, i.e. the last 0.5% / 1% of the reverse process. Our chain has
    `num_timesteps` levels, so the same *fraction* is what keeps the meaning; the
    absolute values would make the TVS level vanish (sigmoid(5 - t) ~ 0 for t > 20).
    Both are configurable (`sd_tvs_t_bias`, `sd_hard_levels`) and recorded per run.
    """
    scale = float(num_timesteps) / 1000.0
    return {'sd_tvs_t_bias': 5.0 * scale, 'sd_hard_levels': max(1, int(round(10.0 * scale)))}
