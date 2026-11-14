"""A5: the Constrained Diffusers update must not materialise a dense [particles, rows, D] Jacobian.

Segment rows are 127 x 362 = 46 355 per sample. Building the dense tensor needed
~370 MB and raised `MemoryError` at the hinge mask (HANDOFF §5). The official update
only ever uses the weighted row sum sum_k w_k J_k -- the reference implementation
gets exactly that from `autograd.grad(g.sum(), state)`
(NeurIPS-2025-constrained-diffusers umaze-large diffusion.py:306-322) and never forms J --
so accepting a per-particle csr list is the faithful input, not a shortcut.

What is pinned here:
* the sparse row sum equals the dense einsum (so the fix cannot change any number);
* the hinge drops the same rows in both forms;
* `cd_run` completes on the real segment shape instead of dying on the allocation.
"""
import numpy as np
import pytest
from scipy import sparse

from experiments.safety_baselines.algorithms.kernels import row_contribution
from experiments.safety_baselines.algorithms.official import constrained_diffusers as official
from experiments.safety_baselines.algorithms.solvers import Budget
from experiments.safety_baselines.algorithms.constrained_diffusers.runner import run as cd_run

SEGMENTS, CIRCLES, POINTS = 127, 362, 128
ROWS = SEGMENTS * CIRCLES          # 45 974, the number in configs/*_segments.json
D = 2 * (POINTS - 2)               # 252 free coordinates


def test_row_contribution_equals_the_dense_einsum():
    rng = np.random.default_rng(0)
    P, K, Dd = 3, 5, 4
    dense = rng.normal(size=(P, K, Dd)); w = rng.normal(size=K)
    per_particle = [sparse.csr_matrix(dense[b]) for b in range(P)]
    np.testing.assert_allclose(row_contribution(dense, w),
                               row_contribution(per_particle, w), rtol=0, atol=1e-13)


def test_hinge_drops_exactly_the_dense_masked_rows():
    rng = np.random.default_rng(1)
    P, K, Dd = 2, 4, 3
    dense = rng.normal(size=(P, K, Dd)); w = np.abs(rng.normal(size=K))
    hinge = rng.random((P, K)) > .5          # particle-dependent, as in the runner
    per_particle = [sparse.csr_matrix(dense[b]) for b in range(P)]
    np.testing.assert_allclose(row_contribution(dense * hinge[..., None], w),
                               row_contribution(per_particle, w, hinge), atol=1e-13)


@pytest.mark.parametrize('step', ['pd', 'alm'])
def test_the_official_updates_take_a_csr_list_identically(step):
    rng = np.random.default_rng(2)
    P, K, Dd = 2, 6, 5
    mean = rng.normal(size=(P, Dd)); vio = rng.normal(size=(P, K)); slack = np.abs(rng.normal(size=(P, K)))
    dense = rng.normal(size=(P, K, Dd)); js = [sparse.csr_matrix(dense[b]) for b in range(P)]
    lam = np.abs(rng.normal(size=(P, K)))
    if step == 'pd':
        a = official.primal_dual(mean, vio, dense, lam, 1e-3)
        b = official.primal_dual(mean, vio, js, lam, 1e-3)
    else:
        a = official.augmented_lagrangian(mean, vio, dense, lam, slack, 2.5e-4, 1.01)
        b = official.augmented_lagrangian(mean, vio, js, lam, slack, 2.5e-4, 1.01)
    for u, v in zip(a, b): np.testing.assert_allclose(u, v, atol=1e-13)


class _Rows:
    """Linear constraint set of the segment shape: g = J x - b, with J stored as csr or dense."""

    def __init__(self, rows, dim, as_sparse, seed=0):
        self.D = dim; self.ab = np.array([.9, .5]); self.betas = np.array([.1, .4])
        rng = np.random.default_rng(seed)
        per_row = 4
        idx = rng.integers(0, dim, size=(rows, per_row))
        dat = rng.normal(size=(rows, per_row)) * .5
        indptr = np.arange(rows + 1) * per_row
        self.J = sparse.csr_matrix((dat.ravel(), idx.ravel(), indptr), shape=(rows, dim))
        self.b = rng.normal(size=rows) * .1                # mixed feasible / violated rows
        self.as_sparse = as_sparse

    def initial(self, rng, particles=1, ve=False): return rng.normal(size=(particles, self.D))
    def score(self, x, t, ve=False): return -np.atleast_2d(x) * .5
    def clean(self, x, t, ve=False): return np.atleast_2d(x) * .5

    def constraints(self, x):
        v = np.asarray(x, dtype=float).reshape(-1)
        g = np.asarray(self.J @ v).reshape(-1) - self.b
        return g, (self.J.copy() if self.as_sparse else self.J.toarray())


def _cfg():
    return {'particles': 2, 'cd_levels': 2, 'cd_extra_steps': 0, 'cd_beta': 'level',
            'cd_substeps': 1, 'dual_eta': 6.1e-3, 'rho': 2.5e-2, 'rho_growth': 1.01,
            'dual_constraint_scale': 1.0}


@pytest.mark.parametrize('method', ['Primal-Dual', 'ALM'])
def test_runner_gives_the_same_answer_for_csr_and_dense_rows(method):
    trace_d, trace_s = [], []
    a = cd_run(_Rows(24, 8, as_sparse=False), method, np.random.default_rng(7), _cfg(), Budget(30), trace_d)
    b = cd_run(_Rows(24, 8, as_sparse=True), method, np.random.default_rng(7), _cfg(), Budget(30), trace_s)
    # only the summation order differs (csr mat-vec vs dense einsum)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)
    assert [t['lambda_max'] for t in trace_d] == pytest.approx([t['lambda_max'] for t in trace_s], abs=1e-14)


@pytest.mark.parametrize('method', ['Primal-Dual', 'ALM'])
def test_runner_completes_on_the_real_segment_row_count(method):
    """45 974 rows x 252: the dense hinge mask was 370 MB, this must stay O(nnz)."""
    task = _Rows(ROWS, D, as_sparse=True)
    trace = []
    x = cd_run(task, method, np.random.default_rng(0), _cfg(), Budget(120), trace)
    assert x.shape == (D,) and np.isfinite(x).all()
    assert len(trace) == 2
    # the fix only works if the csr really is the constraint matrix of that shape
    assert task.J.shape == (ROWS, D) and task.J.nnz == ROWS * 4
