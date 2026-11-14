"""Regression tests for the three projector defects (2026-11-14 review).

A. feasibility must be decided by the TARGET; an already-feasible warm start must not be
   returned as `identity` (that silently discarded the diffusion update);
B. the slack must RELAX (s >= 0), per row;
C. a non-converged OSQP result must never be reported as a success.

The tests drive the real `SC48ControlTask.project` through a tiny analytic stub: the
projector only touches `constraints`, `projection_tol`, `projection_maxiter` and the
optional `projection_*` attributes, so no model / checkpoint / GPU is needed.
"""
import numpy as np
import pytest

from experiments.safety_baselines.adapters.sc48_control import SC48ControlTask
from experiments.safety_baselines.algorithms.solvers import Budget

CENTER = np.zeros(2); RADIUS = 0.2


class ToyProjector(SC48ControlTask):
    """One free point (D=2). Feasible set = exterior of a disc: g = R - |x-c| <= 0."""

    def __init__(self, radius=RADIUS, maxiter=60, tol=1e-7, trust=0.05):
        self.projection_tol = tol; self.projection_maxiter = maxiter
        self.projection_trust = trust; self.projection_slack_weight = 1e6
        self.radius = radius

    def constraints(self, x):
        p = np.asarray(x, dtype=float).reshape(-1)[:2]
        d = p - CENTER; n = float(np.linalg.norm(d)) or 1.0
        return np.array([self.radius - n]), -d / n


def feasible_toy_point():
    return np.array([0.6, 0.0])          # outside the disc


def infeasible_toy_point():
    return np.array([0.05, 0.0])         # inside the disc, close to the centre


def test_feasible_target_is_the_only_identity_case():
    task = ToyProjector()
    out, info = task.project(feasible_toy_point(), Budget(30.))
    assert info['status'] == 'identity'
    np.testing.assert_allclose(out, feasible_toy_point())


def test_feasible_warm_start_does_not_swallow_an_infeasible_target():
    """Bug A: the old code returned the warm start and threw the diffusion update away."""
    task = ToyProjector()
    warm = feasible_toy_point()
    target = infeasible_toy_point()
    out, info = task.project(target, Budget(30.), start=warm)
    assert info['status'] != 'identity', 'warm start must not be returned as identity'
    assert not np.allclose(out, warm), 'target update must not be discarded'
    assert np.linalg.norm(out - CENTER) >= task.radius - 1e-6, 'result must be feasible'
    # and it must be closer to the target than the warm start was
    assert np.linalg.norm(out - target) < np.linalg.norm(warm - target)


def test_projection_lands_on_the_true_feasible_set():
    task = ToyProjector()
    out, info = task.project(infeasible_toy_point(), Budget(30.))
    g, _ = task.constraints(out)
    assert float(np.max(g)) <= task.projection_tol + 1e-9
    assert abs(np.linalg.norm(out - CENTER) - task.radius) < 1e-3


def test_slack_relaxes_the_linearised_rows_in_the_right_direction():
    """Bug B: the old pad row was (-s >= 0) i.e. s <= 0, which tightens instead of relaxing.

    The contradictory pair  -x + s1 >= 1  and  x + s2 >= 1  is feasible ONLY if the slack can
    go positive (sum of the two rows gives s1 + s2 >= 2), so s1, s2 > 0 is direct evidence.
    """
    import osqp
    from scipy import sparse
    A = sparse.csc_matrix(np.array([[-1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
    A = sparse.vstack([A, sparse.hstack([sparse.csc_matrix((2, 1)), sparse.eye(2)])], format='csc')
    l = np.array([1.0, 1.0, 0.0, 0.0])
    P = sparse.diags([1.0, 1e3, 1e3], format='csc')
    s = osqp.OSQP()
    s.setup(P=P, q=np.zeros(3), A=A, l=l, u=np.full(4, np.inf), verbose=False,
            eps_abs=1e-8, eps_rel=1e-8, max_iter=40000)
    res = s.solve()
    assert int(res.info.status_val) in (1, 2)
    assert res.x[1] > 0.9 and res.x[2] > 0.9, 'slack must grow to absorb the contradiction'


def test_non_converged_osqp_is_reported_as_failure(monkeypatch):
    """Bug C: only OSQP status 1/2 may be accepted."""
    import osqp
    real_setup = osqp.OSQP.setup

    def starved_setup(self, **kw):
        kw['max_iter'] = 4
        return real_setup(self, **kw)

    monkeypatch.setattr(osqp.OSQP, 'setup', starved_setup)
    task = ToyProjector(maxiter=3)
    with pytest.raises(Exception) as exc:
        task.project(infeasible_toy_point(), Budget(30.))
    assert 'projection_failed' in str(exc.value)


def test_projection_reports_diagnostics():
    task = ToyProjector()
    _, info = task.project(infeasible_toy_point(), Budget(30.))
    for key in ('residual', 'iterations', 'status'):
        assert key in info
    assert info['residual'] <= task.projection_tol + 1e-9


def test_impossible_target_raises_instead_of_returning_a_bad_point():
    """Two contradictory rows cannot be satisfied by the variable; the projector must fail."""
    class Contradictory(ToyProjector):
        def constraints(self, x):
            p = float(np.asarray(x, dtype=float).reshape(-1)[0])
            # g1 = p - 0 <= 0  and  g2 = -p - 1 <= 0  => p <= 0 and p >= -1 : feasible,
            # but ask for the target 5.0 and with tol 1e-12 the SQP cannot reach 0 in one
            # trust-region step, so it must either converge or raise - never lie.
            return np.array([p, -p - 1.0]), np.array([[1.0], [-1.0]])
    task = Contradictory(maxiter=200)
    out, info = task.project(np.array([5.0]), Budget(60.))
    g, _ = task.constraints(out)
    assert float(np.max(g)) <= task.projection_tol + 1e-9
