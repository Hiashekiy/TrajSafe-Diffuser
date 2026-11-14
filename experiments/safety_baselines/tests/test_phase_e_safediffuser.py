"""The SafeDiffuser correction as the official code writes it.

Replaces the previous Phase-E tests, which pinned behaviour we had invented (velocity
integrator with `cbf_dt` substeps, discrete certificate, dt halving, level retries, step
cap, high-noise gate). The paper's release does none of that: one QP per diffusion level
on the sampled next state (`diffusion.py:1030-1097`), with
`h = Lfb + k*b`, `G = -grad_b`, `q = -(xp1 - x)`, `Q = I` (451-506) and the ReS slack
weighted by `sign = 100` outside the hard interval (610-616).

What is checked here:
* the closed-form solve is the exact solution of the official QP (against a generic solver);
* ReS is relaxed by a factor ~1e4 outside the hard interval and exact inside it;
* TVS tightens the level monotonically as t -> 0;
* the loop is one QP per level, DDPM-noised, and leaves the pinned endpoints alone;
* an infeasible row is reported, not swallowed.
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from experiments.safety_baselines.algorithms.official import safediffuser as official
from experiments.safety_baselines.algorithms.official.safediffuser import official_schedule_defaults
from experiments.safety_baselines.algorithms.safediffuser.runner import ddpm_coefficients, run as safe_run
from experiments.safety_baselines.algorithms.solvers import Budget, SolverFailure


def solve_official_qp(b, gb, ref, offset, s):
    """min 0.5||u-ref||^2 + 0.5*r^2  s.t.  -gb.u + s*r <= offset + b, by SLSQP."""
    n = len(b)
    out = []
    for i in range(n):
        def obj(z):
            u, r = z[:2], z[2]
            return .5*np.sum((u-ref[i])**2) + .5*r*r
        cons = [{'type': 'ineq', 'fun': lambda z: (offset + b[i]) + gb[i] @ z[:2] - s*z[2]}]
        res = minimize(obj, np.r_[ref[i], 0.], method='SLSQP', constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-14})
        out.append(res.x[:2])
    return np.array(out)


@pytest.mark.parametrize('s', [0.0, 100.0])
def test_the_closed_form_is_the_exact_official_qp(s):
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = rng.normal(size=6) * .3
        gb = rng.normal(size=(6, 2))
        ref = rng.normal(size=(6, 2))
        offset = 0.0
        np.testing.assert_allclose(official._qp(b, gb, ref, offset, s),
                                   solve_official_qp(b, gb, ref, offset, s), atol=1e-6)


def test_the_correction_satisfies_the_linearised_barrier():
    rng = np.random.default_rng(1)
    b = rng.normal(size=8) * .3; gb = rng.normal(size=(8, 2)); ref = rng.normal(size=(8, 2))
    u = official.ros(b, gb, ref)
    assert np.all(b + np.einsum('pd,pd->p', gb, u) >= -1e-12)


def test_res_is_soft_outside_the_hard_interval_and_exact_inside():
    gb = np.array([[0., 1.]]); b = np.array([0.1]); ref = np.array([[0., -1.]])   # wants to dive in
    hard = official.res(b, gb, ref, t=0, hard_levels=3)
    soft = official.res(b, gb, ref, t=3, hard_levels=3)
    # hard: the linearised barrier is met; soft: the slack buys ~1e4 less correction
    assert b[0] + gb[0] @ hard[0] >= -1e-12
    assert np.linalg.norm(soft[0] - ref[0]) < 1e-3 * max(1., np.linalg.norm(ref[0]))
    assert np.linalg.norm(hard[0] - ref[0]) > np.linalg.norm(soft[0] - ref[0])


def test_tvs_level_tightens_as_noise_goes_to_zero():
    """b_eff = b - sigma(t) shrinks as t -> 0, so the same QP has to move further out."""
    gb = np.array([[1., 0.]]); b = np.array([0.5]); ref = np.array([[0., 0.]])
    achieved = []
    for t in (200, 60, 10, 0):
        u = official.tvs(b, gb, ref, t, t_bias=1.28)
        achieved.append(float(b[0] + gb[0] @ u[0]))          # clearance after the step
    assert all(m2 >= m1 - 1e-12 for m1, m2 in zip(achieved, achieved[1:]))
    assert achieved[-1] > achieved[0] + 0.05                 # the level really does bite at t=0


def test_official_schedule_defaults_keep_the_fraction_of_their_1000_step_chain():
    d = official_schedule_defaults(256)
    assert d['sd_hard_levels'] == 3 and abs(d['sd_tvs_t_bias'] - 1.28) < 1e-12
    d = official_schedule_defaults(1000)
    assert d['sd_hard_levels'] == 10 and d['sd_tvs_t_bias'] == 5.0      # the literal values


class PointTask:
    """One free waypoint between pinned endpoints; barrier is a single circle."""

    def __init__(self, T=8, start=(0.6, 0.), target=(0.3, 0.)):
        self.ab = np.linspace(.9, .05, T); self.betas = np.full(T, .3)
        self.start = np.array(start); self.target = np.array(target); self.calls = 0

    def initial(self, rng, particles=1, ve=False): return self.start[None].copy()
    def clean(self, x, t, ve=False): return self.target[None].copy()

    def point_barrier(self, x):
        self.calls += 1
        p = np.asarray(x, float).reshape(1, 2)
        d = p - np.array([[0., 0.]]); n = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        return n - .2, d / n[:, None]


def test_ddpm_coefficients_match_the_official_posterior():
    ab = np.array([.9, .5]); betas = np.array([.1, .4])
    c1, c2, var = ddpm_coefficients(ab, betas)
    ab_prev = np.concatenate([[1.], ab[:-1]])
    np.testing.assert_allclose(var, betas*(1 - ab_prev)/(1 - ab))
    # independent rewrite of diffusion.py:124-127 (posterior_mean_coef1/2)
    np.testing.assert_allclose(c1, np.sqrt(ab_prev)*betas/(1 - ab))
    np.testing.assert_allclose(c2, np.sqrt(1 - betas)*(1 - ab_prev)/(1 - ab))


@pytest.mark.parametrize('method', ['RoS-diffuser', 'ReS-diffuser', 'TVS-diffuser'])
def test_one_qp_per_level_when_the_correction_is_unconditional(method):
    task = PointTask(); trace = []
    x = safe_run(task, method, np.random.default_rng(0), {'sd_apply_below': None}, Budget(10), trace)
    assert len(trace) == len(task.ab) == task.calls        # one barrier query, one QP per level
    assert np.isfinite(x).all() and x.shape == (2,)
    assert all(t['optimizer'] == 'official_cbf_qp' and t['corrected'] for t in trace)


def test_by_default_the_correction_runs_only_in_the_low_noise_tail():
    """`p_sample` ships both policies; the tail-only one (diffusion.py:1071-1078) is the
    default because our sigmas are O(1e3) scene units at the top levels."""
    task = PointTask(); trace = []
    safe_run(task, 'RoS-diffuser', np.random.default_rng(0), {}, Budget(10), trace)
    below = trace[0]['sd_apply_below'] == 1                 # rescaled 10-of-1000 on T=8
    assert below
    expected = [t <= 1 for t in range(len(task.ab)-1, -1, -1)]
    assert [r['corrected'] for r in trace] == expected
    assert task.calls == sum(expected)


def test_the_loop_is_ddpm_noised_not_deterministic():
    finals = []
    for seed in (0, 1):
        trace = []
        finals.append(safe_run(PointTask(), 'RoS-diffuser', np.random.default_rng(seed),
                               {'sd_apply_below': None}, Budget(10), trace))
    assert finals[0].tolist() != finals[1].tolist()          # different draws of z


def test_endpoints_are_never_corrected():
    class WithEndpoints(PointTask):
        def point_barrier(self, x):
            b, gb = super().point_barrier(x)
            return b, gb                                    # only free points are exposed

    task = WithEndpoints(); trace = []
    safe_run(task, 'RoS-diffuser', np.random.default_rng(0), {'sd_apply_below': None}, Budget(10), trace)
    assert task.calls == len(task.ab)                       # nothing else touched the state


def test_an_infeasible_row_is_reported_not_swallowed():
    # already inside the obstacle (b<0) with a zero gradient: no step can satisfy the row
    with pytest.raises(SolverFailure, match='cbf_infeasible'):
        official.ros(np.array([-1e-3]), np.zeros((1, 2)), np.zeros((1, 2)))


def test_a_segment_task_has_no_official_counterpart_and_is_refused():
    class Segment(PointTask):
        point_barrier = None
    with pytest.raises(ValueError, match='segment-row'):
        safe_run(Segment(), 'RoS-diffuser', np.random.default_rng(0), {}, Budget(5), [])
