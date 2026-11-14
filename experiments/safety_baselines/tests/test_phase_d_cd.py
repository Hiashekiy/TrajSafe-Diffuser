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
from experiments.safety_baselines.algorithms.official.posterior import level_ladder, is_noiseless

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


def test_d2_bias_is_bounded_on_the_256_level_schedule():
    s = NoiseSchedule(256, 'squaredcos_cap_v2')
    ab = s.alphas_cumprod.numpy().astype(float); beta = s.betas.numpy().astype(float)
    worst_level = max(beta[t] / (4 * (1 - ab[t])) for t in range(256))
    assert worst_level < 0.30      # beta/(4v) stays <= 1/4
    assert 1 / (1 - worst_level) <= 4 / 3 + 1e-6


def test_d1_the_official_ladder_visits_every_level_and_repeats_t1_noiselessly():
    """`reversed(range(-200, T))` with `i<=0 -> 1` (:503-510) and `nonzero_mask = (t != 1)` (:479)."""
    ladder = level_ladder(256, 200)
    assert len(ladder) == 256 + 200
    assert ladder[:3] == [255, 254, 253]               # walks DOWN from T-1
    assert ladder[-1] == 1 and ladder.count(1) == 202
    assert set(ladder) == set(range(1, 256))           # t=0 is never visited
    assert not any(is_noiseless(t) for t in ladder[:54])   # noisy while descending
    assert all(is_noiseless(t) for t in ladder[-202:])     # and silent for every revisit
    plain = level_ladder(256, 0)
    assert plain[:3] == [255, 254, 253] and plain[-2:] == [1, 1] and len(plain) == 256


@pytest.mark.parametrize('name', ['seven_native_points_paper_t256.json',
                                  'seven_native_points_vertexall_t256.json'])
def test_d1_main_configs_visit_every_trained_level(name):
    cfg = json.loads((CONFIGS / name).read_text(encoding='utf8'))
    assert cfg['cd_levels'] == cfg['schedule_facts']['timesteps'] == 256
    assert 'cd_beta' not in cfg          # only one mode left: the level the ladder visits


def test_d3_multiplier_units_follow_the_squared_rule():
    """g'=s*g with lambda'=lambda/s, lr'=lr/s^2, rho'=rho/s^2 keeps the official updates identical."""
    from experiments.safety_baselines.algorithms.official import constrained_diffusers as official
    s = 128.0
    mean = np.zeros((2, 3)); vio = np.array([[.2], [.5]]); grad = np.ones((2, 1, 3))
    lam = np.full((2, 1), .1); slack = np.full((2, 1), .05); lr = 1e-3; rho = 2.5e-4
    a, la = official.primal_dual(mean, vio, grad, lam, lr)
    b, lb = official.primal_dual(mean, s * vio, s * grad, lam / s, lr / s ** 2)
    np.testing.assert_allclose(a, b); np.testing.assert_allclose(la / s, lb)
    a, sa, la, ra = official.augmented_lagrangian(mean, vio, grad, lam, slack, rho, 1.1)
    b, sb, lb, rb = official.augmented_lagrangian(mean, s * vio, s * grad, lam / s, slack * s, rho / s ** 2, 1.1)
    np.testing.assert_allclose(a, b); np.testing.assert_allclose(sa * s, sb)
    np.testing.assert_allclose(la / s, lb); np.testing.assert_allclose(ra / s ** 2, rb)
