"""Phase D: Constrained Diffusers — beta ladder, Gaussian stationarity, multiplier units.

D2 is the important correction: the one-step Langevin recursion
    x_{k+1} = (1 - beta/(2v)) x_k + sqrt(beta) z
has stationary variance V_inf = v / (1 - beta/(4v)) and is stable while
|1 - beta/(2v)| < 1, i.e. beta < 4v. A larger beta therefore biases the variance
(up to 4/3 for beta = v, unbounded only as beta -> 4v); it does not "diverge" for
the beta values this ladder produces. The earlier claim in this repo that the
interval-beta mode diverges was wrong and is pinned by the numbers below.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.algorithms.constrained_diffusers.runner import effective_beta, level_ladder

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def stationary_gaussian(beta, v, steps=200000, warmup=20000, seed=0):
    rng = np.random.default_rng(seed)
    c = 1.0 - beta / (2.0 * v)
    z = rng.normal(size=steps)
    x = 0.0
    xs = np.empty(steps)
    for k in range(steps):
        x = c * x + np.sqrt(beta) * z[k]
        xs[k] = x
    return float(xs[warmup:].var())


@pytest.mark.parametrize('v,beta', [(1.0, 0.05), (1.0, 0.25), (0.01, 0.002), (0.883, 0.1217)])
def test_d2_stationary_variance_matches_the_analytic_formula(v, beta):
    assert abs(1 - beta / (2 * v)) < 1, 'stability condition must hold for these values'
    v_inf = v / (1 - beta / (4 * v))
    emp = stationary_gaussian(beta, v)
    # MC tolerance from the effective sample count of this AR(1) (rate beta/(2v))
    n_eff = 180000 * (beta / (2 * v))
    tol = max(0.02, 4 * np.sqrt(2.0 / n_eff))
    assert abs(emp / v_inf - 1) < tol, (emp, v_inf, tol)
    # and it equals the direct geometric-series form
    np.testing.assert_allclose(v_inf, beta / (1 - (1 - beta / (2 * v)) ** 2), rtol=1e-12)


def test_d2_bias_is_bounded_on_the_256_level_schedule_and_the_two_beta_modes_tie():
    s = NoiseSchedule(256, 'squaredcos_cap_v2')
    ab = s.alphas_cumprod.numpy().astype(float); beta = s.betas.numpy().astype(float)
    ladder = level_ladder(256, 64, 0)
    worst_interval = max((effective_beta(ab, ladder[i], ladder[i + 1], beta[ladder[i]], 'interval')) /
                         (4 * (1 - ab[ladder[i]])) for i in range(len(ladder) - 1))
    worst_level = max(beta[t] / (4 * (1 - ab[t])) for t in range(256))
    assert worst_interval < 0.30 and worst_level < 0.30      # beta/(4v) stays <= 1/4
    # both modes give the same worst-case inflation, so "interval diverges" is wrong
    assert abs(1 / (1 - worst_interval) - 1 / (1 - worst_level)) < 0.01
    assert 1 / (1 - worst_interval) <= 4 / 3 + 1e-6


def test_d2_subsampled_ladder_would_need_substeps_to_keep_the_bias_down():
    """interval beta is larger; with cd_substeps=K the step is beta/K, bias shrinks accordingly."""
    s = NoiseSchedule(256, 'squaredcos_cap_v2')
    ab = s.alphas_cumprod.numpy().astype(float); beta = s.betas.numpy().astype(float)
    t, nxt = 4, 0
    v = 1 - ab[t]; b_int = effective_beta(ab, t, nxt, beta[t], 'interval')
    v_inf_full = v / (1 - b_int / (4 * v))
    v_inf_sub = v / (1 - (b_int / 4) / (4 * v))
    assert v_inf_full > v_inf_sub            # refinement reduces the bias
    assert v_inf_full / v <= 4 / 3 + 1e-6


@pytest.mark.parametrize('name', ['seven_native_points_paper_t256.json',
                                  'seven_native_points_vertexall_t256.json'])
def test_d1_main_configs_visit_every_trained_level(name):
    cfg = json.loads((CONFIGS / name).read_text(encoding='utf8'))
    assert cfg['cd_levels'] == cfg['schedule_facts']['timesteps'] == 256
    assert cfg['cd_beta'] == 'level'


def test_d3_multiplier_units_follow_the_squared_rule():
    """g'=s*g with lambda'=lambda/s, eta'=eta/s^2, rho'=rho/s^2 keeps PD/ALM identical."""
    from experiments.safety_baselines.algorithms.kernels import primal_dual_step, alm_step
    s = 128.0
    x = np.zeros((2, 3)); score = np.ones((2, 3)); g = np.array([[.2], [.5]]); J = np.ones((2, 1, 3))
    lam = np.full(1, .1); noise = np.zeros((2, 3)); eta = 1e-3; rho = 2.5e-4
    a, l = primal_dual_step(x, score, g, J, lam, .1, eta, noise)
    b, ll = primal_dual_step(x, score, s * g, s * J, lam / s, .1, eta / s ** 2, noise)
    np.testing.assert_allclose(a, b); np.testing.assert_allclose(l / s, ll)
    a, l, sl, rr = alm_step(x, score, g, J, lam, rho, 1.1, .1, noise)
    b, ll, ssl, r2 = alm_step(x, score, s * g, s * J, lam / s, rho / s ** 2, 1.1, .1, noise)
    np.testing.assert_allclose(a, b); np.testing.assert_allclose(l / s, ll); np.testing.assert_allclose(sl * s, ssl)
