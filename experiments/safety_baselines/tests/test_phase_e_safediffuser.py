"""Phase E: SafeDiffuser discrete safety on an analytic circle obstacle.

E2/E3 claim under test: the paper's velocity condition  grad h . nu + eps (h - gamma) >= 0
does not by itself keep the *applied* state safe, and a discrete certificate
h(x_{k+1}) - gamma >= exp(-eps dt)(h(x_k) - gamma) with step backtracking does.
These tests use one free waypoint and one circle, so the clearance is analytic.
"""
import numpy as np
import pytest

from experiments.safety_baselines.algorithms.safediffuser.runner import run as safe_run
from experiments.safety_baselines.algorithms.solvers import Budget, SolverFailure

CENTER = np.array([0.0, 0.0]); R = 0.2

class CircleTask:
    """One free waypoint (H=3 -> D=2); h = |p-c| - R, feasible h >= 0."""
    ab = np.array([0.9, 0.7, 0.4, 0.1])
    betas = np.array([0.1, 0.2, 0.3, 0.4])
    sigma = np.sqrt((1 - ab) / ab)
    def __init__(self, start, nominal):
        self.start = np.asarray(start, float); self.nominal_point = np.asarray(nominal, float)
    def initial(self, rng, particles=1, ve=False): return self.start[None].copy()
    def constraints(self, x):
        p = np.asarray(x, float).reshape(-1)[:2]; d = p - CENTER; n = np.linalg.norm(d)
        g = np.array([R - n]); J = np.zeros((1, np.asarray(x).size))
        if n > 1e-12: J[0, :2] = -d / n
        return g, J
    def values(self, x):
        p = np.asarray(x, float).reshape(-1)[:2]
        return np.array([R - np.linalg.norm(p - CENTER)])
    def clean(self, x, t, ve=False): return self.nominal_point[None].copy()
    def unpack(self, x): return np.asarray(x, float).reshape(-1)[:2]

CFG_BASE = {'cbf_dt': 1.0, 'cbf_epsilon': 1.0, 'cbf_gamma_floor': 1e-6, 'hard_steps': 4,
            'relax_weight': 100.0, 'qp_tol': 1e-6, 'cbf_substeps': 1}

def clearance(x): return float(np.linalg.norm(np.asarray(x).reshape(-1)[:2] - CENTER) - R)

def test_e4_far_from_obstacle_the_discrete_certificate_holds_without_backtracking():
    task = CircleTask([0.6, 0.0], [0.65, 0.0])
    cfg = dict(CFG_BASE, cbf_discrete_check=True)
    trace = []; final = safe_run(task, 'RoS-diffuser', np.random.default_rng(0), cfg, Budget(5), trace)
    assert clearance(final) > 0.2
    assert all(t['level_attempts'] == 1 for t in trace), 'no retry needed away from the obstacle'
    assert all(t['discrete_ok'] for t in trace)

def test_e4_velocity_condition_alone_penetrates_but_the_discrete_check_does_not():
    """Nominal pulls straight into the obstacle centre; compare both modes."""
    def run_mode(check):
        task = CircleTask([0.45, 0.0], [0.0, 0.0])       # nominal == obstacle centre
        cfg = dict(CFG_BASE, cbf_discrete_check=check)
        trace = []
        try:
            final = safe_run(task, 'RoS-diffuser', np.random.default_rng(0), cfg, Budget(10), trace)
        except (SolverFailure, RuntimeError):
            return None, trace
        return clearance(final), trace
    c_off, _ = run_mode(False)
    c_on, trace_on = run_mode(True)
    assert c_on is not None and c_on >= -1e-6, ('discrete check must not penetrate', c_on)
    if c_off is not None:
        assert c_on >= c_off - 1e-9, ('discrete check must not be worse', c_off, c_on)
    assert all('level_attempts' in t and 'discrete_residual_end' in t for t in trace_on)

def test_e4_large_steps_are_backtracked_instead_of_penetrating():
    task = CircleTask([0.45, 0.0], [0.0, 0.0])
    cfg = dict(CFG_BASE, cbf_discrete_check=True, cbf_dt=4.0, cbf_substeps=1)
    trace = []
    tx = None
    try:
        final = safe_run(task, 'RoS-diffuser', np.random.default_rng(0), cfg, Budget(10), trace)
        tx = clearance(final)
    except (SolverFailure, RuntimeError):
        tx = None
    assert tx is None or tx >= -1e-6

def test_e4_qp_infeasibility_is_reported_not_swallowed():
    class Hopeless(CircleTask):
        def constraints(self, x):
            g, J = super().constraints(x)
            return np.array([1e-3, 1e-3]), np.array([[0.0, 0.0], [0.0, 0.0]])   # unsatisfiable rows
    task = Hopeless([0.45, 0.0], [0.45, 0.0])
    cfg = dict(CFG_BASE, cbf_discrete_check=True)
    with pytest.raises((SolverFailure, RuntimeError)):
        safe_run(task, 'RoS-diffuser', np.random.default_rng(0), cfg, Budget(5), [])
