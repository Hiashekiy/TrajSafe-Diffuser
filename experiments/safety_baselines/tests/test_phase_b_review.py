"""Regression tests for the 2026-11-14 Phase B-Review (S-C48 control-space port).

Pins the four fixes so they cannot silently regress:
  * warm start is actually forwarded to the projector (and only when supported);
  * the NFE counter counts real forward passes instead of reading the config;
  * the soft CBF QP uses rho = 1/w^2, i.e. the SAME convention as cbf_velocity, so the
    two solvers agree on the relaxation scale (the earlier report compared them wrongly);
  * OSQP non-convergence is raised, never returned as a solution.
"""
import numpy as np
import pytest
from scipy import sparse

from experiments.safety_baselines.algorithms.solvers import call_project, cbf_velocity, Budget
from experiments.safety_baselines.algorithms.runner import attach_nfe_counter


class Recorder:
    def __init__(self, accept_start=True, accept_partial=True):
        self.calls = []
        if accept_start and accept_partial:
            self.project = self._full
        elif accept_start:
            self.project = self._start_only
        else:
            self.project = self._bare
    def _full(self, target, budget, start=None, allow_partial=False):
        self.calls.append({'start': None if start is None else np.asarray(start).copy(),
                           'allow_partial': allow_partial})
        return np.asarray(target, dtype=float), {'status': 'identity'}
    def _start_only(self, target, budget, start=None):
        self.calls.append({'start': None if start is None else np.asarray(start).copy()})
        return np.asarray(target, dtype=float), {'status': 'identity'}
    def _bare(self, target, budget):
        self.calls.append({'start': 'not-supported'})
        return np.asarray(target, dtype=float), {'status': 'identity'}


def test_warm_start_is_forwarded_to_the_projector():
    task = Recorder(); warm = np.array([1., 2., 3.])
    call_project(task, np.zeros(3), Budget(5.), warm)
    np.testing.assert_array_equal(task.calls[-1]['start'], warm)
    call_project(task, np.zeros(3), Budget(5.), warm, allow_partial=True)
    assert task.calls[-1]['allow_partial'] is True


def test_warm_start_is_omitted_for_adapters_that_do_not_declare_it():
    task = Recorder(accept_start=False)
    out, info = call_project(task, np.zeros(3), Budget(5.), np.ones(3))   # must not raise
    assert task.calls[-1]['start'] == 'not-supported'
    np.testing.assert_array_equal(out, np.zeros(3))


def test_allow_partial_is_omitted_when_not_declared():
    task = Recorder(accept_partial=False)
    call_project(task, np.zeros(3), Budget(5.), None, allow_partial=True)   # must not raise
    assert 'allow_partial' not in task.calls[-1]


class _Task:
    def __init__(self): self.calls = 0
    def clean(self, x, t, ve=False):
        self.calls += 1; return np.zeros_like(np.atleast_2d(x))


def test_nfe_counter_counts_real_forward_passes():
    task = _Task(); counter = attach_nfe_counter(task)
    assert counter[0] == 0
    for t in range(16):
        task.clean(np.zeros(4), t)
    assert counter[0] == 16 == task.calls
    task.clean(np.zeros(4), 0, ve=True)
    assert counter[0] == 17


def test_soft_qp_slack_matches_the_legacy_rescaling():
    """cbf_velocity reports r = negative/w; the explicit QP reports s = negative.

    With rho = 1/w^2 both describe the same relaxation, so r*w must equal s. This is the
    claim that replaced the (wrong) "w=100 makes the slack free" reading.
    """
    rng = np.random.default_rng(0)
    n, m = 6, 5
    J = rng.normal(size=(m, n)); offset = rng.normal(size=m) * .1
    desired = rng.normal(size=n)
    for w in (1.0, 10.0, 100.0):
        v1, r1, info1 = cbf_velocity(desired, J, offset, w, Budget(30.))
        # same problem written out as an explicit soft QP
        import osqp
        rho = 1.0 / w ** 2
        A = sparse.vstack([sparse.hstack([sparse.csc_matrix(J), sparse.eye(m)]),
                           sparse.hstack([sparse.csc_matrix((m, n)), sparse.eye(m)])], format='csc')
        P = sparse.diags(np.r_[np.ones(n), np.full(m, rho)], format='csc')
        s = osqp.OSQP()
        s.setup(P=P, q=np.r_[-desired, np.zeros(m)], A=A, l=np.r_[-offset, np.zeros(m)],
                u=np.full(2 * m, np.inf), verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000)
        res = s.solve()
        v2 = np.asarray(res.x[:n]); s2 = np.asarray(res.x[n:])
        np.testing.assert_allclose(v1, v2, atol=1e-4)
        np.testing.assert_allclose(np.abs(r1) * w, s2, atol=1e-3)


def test_sc48_task_binds_the_cbf_solver_explicitly():
    """legacy (default) has NO cbf_solve attribute, so the repo's cbf_velocity runs."""
    source = open('experiments/safety_baselines/adapters/sc48_control.py', encoding='utf8').read()
    assert "if self.cbf_solver == 'osqp':" in source
    assert "self.cbf_solve = self.cbf_solve_osqp" in source
    assert "elif self.cbf_solver != 'legacy':" in source
    # non-convergence must raise, never be returned
    assert "status_val not in (1, 2)" in source
    assert "sc48_cbf_qp_not_converged" in source
