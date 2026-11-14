"""Score <-> x0 algebra, and the weighted row sum the official updates are built on.

The Primal-Dual / ALM / CBF-QP / PDM unit tests that used to live here were removed on
2026-11-14 together with the implementations they pinned; the corrections are now the
vendored official code and are tested in tests/test_official_*.py.
"""
import numpy as np

from experiments.safety_baselines.adapters.score_adapter import score_from_x0
from experiments.safety_baselines.algorithms.kernels import row_contribution


def test_score():
    x0 = np.array([1., 2.]); noise = np.array([.3, -.5]); a = .64
    x = np.sqrt(a)*x0 + np.sqrt(1-a)*noise
    np.testing.assert_allclose(score_from_x0(x, x0, a), -noise/np.sqrt(1-a))
    import pytest
    with pytest.raises(ValueError): score_from_x0(x, x0, 1)


def test_row_contribution_handles_one_multiplier_and_one_per_particle():
    rng = np.random.default_rng(0)
    jac = rng.normal(size=(2, 3, 4))
    w_shared = rng.normal(size=3)
    w_per_particle = rng.normal(size=(2, 3))
    np.testing.assert_allclose(row_contribution(jac, w_shared),
                               np.einsum('k,bkd->bd', w_shared, jac))
    np.testing.assert_allclose(row_contribution(jac, w_per_particle),
                               np.einsum('bk,bkd->bd', w_per_particle, jac))
