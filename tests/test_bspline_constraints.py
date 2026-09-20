"""Report section 42: exact B-spline derivative + exact Bezier extraction.

Test E : ``N'(u) Q``   vs a high-accuracy numerical derivative (interior), and
         a one-sided difference at the clamped ends.
Test F : the extracted cubic Bezier reproduces the B-spline EXACTLY on a
         non-knot-crossing interval (float64, max error < 1e-5).
Test G : four Bezier controls inside a convex polygon imply the whole curve is
         inside it (the continuous-safety certificate the ALM relies on).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.geometry.bspline import (BSplineCodec, bspline_basis_matrix,
                                  bspline_basis_derivative_matrix)
from src.geometry.bspline_constraints import (bezier_eval, bezier_extraction,
                                              exact_subdivision,
                                              responsibility_intervals)

KNOTS = os.path.join(ROOT, "data", "carla_v1", "bspline_knots.npy")


def _codec():
    return BSplineCodec(degree=3, num_controls=32, curve_points=128,
                        knots_path=KNOTS)


def test_basis_at_matches_the_registered_decode_basis():
    codec = _codec()
    params = np.linspace(0.0, 1.0, 128)
    assert np.allclose(codec.basis_at(params).numpy(), codec.basis.numpy(),
                       atol=1e-6)
    assert float(codec.basis_at([0.0, 1.0]).sum()) == 2.0


# --------------------------------------------------------------- Test E
def _curve(u, Q, knots):
    return bspline_basis_matrix(knots, 32, 3, [u])[0] @ Q


def _curve_d(u, Q, knots):
    return bspline_basis_derivative_matrix(knots, 32, 3, [u])[0] @ Q


def test_E_derivative_basis_matches_finite_difference_interior():
    knots = np.load(KNOTS).astype(np.float64)
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(32, 2))
    eps = 1e-6
    for u in (0.05, 0.12, 0.234, 0.5, 0.667, 0.81, 0.94):
        fd = (_curve(u + eps, Q, knots) - _curve(u - eps, Q, knots)) / (2 * eps)
        an = _curve_d(u, Q, knots)
        assert np.abs(fd - an).max() < 1e-4


def test_E_derivative_basis_clamped_endpoints_use_one_sided_limits():
    knots = np.load(KNOTS).astype(np.float64)
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(32, 2))
    eps = 1e-6
    for u0, sign in ((0.0, 1.0), (1.0, -1.0)):
        f0 = _curve(u0, Q, knots)
        f1 = _curve(u0 + sign * eps, Q, knots)
        f2 = _curve(u0 + sign * 2 * eps, Q, knots)
        fd = sign * (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * eps)
        an = _curve_d(u0, Q, knots)
        assert np.abs(fd - an).max() < 1e-4


def test_E_torch_basis_derivative_agrees_with_the_float64_matrix():
    codec = _codec()
    knots = np.load(KNOTS).astype(np.float64)
    params = [0.0, 0.13, 0.5, 0.87, 1.0]
    expected = bspline_basis_derivative_matrix(knots, 32, 3, params)
    got = codec.basis_derivative_at(params).numpy().astype(np.float64)
    assert np.abs(got - expected).max() < 1e-3


def test_derivative_operator_has_the_standard_coefficients():
    knots = np.load(KNOTS).astype(np.float64)
    # d/du of the clamped cubic at u=0 must follow the first 3 controls
    Q = np.zeros((32, 2))
    Q[0] = [0.0, 0.0]
    Q[1] = [1.0, 0.0]
    got = bspline_basis_derivative_matrix(knots, 32, 3, [0.0])[0] @ Q
    expected = 3.0 / (knots[4] - knots[1]) * (Q[1] - Q[0])
    assert np.abs(got - expected).max() < 1e-9


# --------------------------------------------------------------- Test F
def test_F_bezier_extraction_is_exact_on_a_knot_span():
    codec = _codec()
    knots = np.unique(np.load(KNOTS).astype(np.float64))
    rng = np.random.default_rng(2)
    Q = rng.normal(size=(32, 2))

    for k in (2, 10, 20, 27):
        ua = float(knots[k] + 0.2 * (knots[k + 1] - knots[k]))
        ub = float(knots[k] + 0.75 * (knots[k + 1] - knots[k]))
        E = bezier_extraction(codec, ua, ub, dtype=torch.float64)
        beta = (E.numpy() @ Q)
        ts = np.linspace(0.0, 1.0, 200)
        # exact B-spline curve on the same interval
        params = ua + ts * (ub - ua)
        exact = bspline_basis_matrix(np.load(KNOTS).astype(np.float64), 32, 3,
                                     params) @ Q
        approx = bezier_eval(torch.as_tensor(beta), torch.as_tensor(ts)).numpy()
        assert np.abs(exact - approx).max() < 1e-5


def test_F_bezier_extraction_endpoints_are_the_curve_points():
    codec = _codec()
    E = bezier_extraction(codec, 0.0, 1.0, dtype=torch.float64).numpy()
    assert np.abs(E[0] - np.eye(32)[0]).max() < 1e-12
    assert np.abs(E[3] - np.eye(32)[-1]).max() < 1e-12


def test_subdivision_never_crosses_a_knot_or_a_responsibility_boundary():
    codec = _codec()
    anchors = np.linspace(0.0, 1.0, 128)
    tau, order = responsibility_intervals(anchors)
    assert np.allclose(order, np.arange(128))
    assert tau[0] == 0.0 and tau[-1] == 1.0
    u = exact_subdivision(tau, codec.knot_span_boundaries())
    assert u[0] == 0.0 and u[-1] == 1.0
    assert np.all(np.diff(u) > 0)
    # every piece lies inside exactly one responsibility interval and one span
    for a, b in zip(u[:-1], u[1:]):
        mid = 0.5 * (a + b)
        cell = np.searchsorted(tau, mid, side="right") - 1
        assert 0 <= cell < 128
        assert tau[cell] - 1e-12 <= a and b <= tau[cell + 1] + 1e-12


# --------------------------------------------------------------- Test G
def _box_halfspaces(half_x, half_y):
    A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b = np.array([half_x, half_x, half_y, half_y])
    return A, b


def test_G_convex_hull_property_gives_continuous_safety():
    A, b = _box_halfspaces(0.30, 0.20)
    controls = torch.tensor([
        [-0.20, -0.10], [-0.25, 0.12], [0.22, 0.15], [0.28, -0.05],
    ], dtype=torch.float64)
    values = (controls @ torch.as_tensor(A.T, dtype=torch.float64)
              - torch.as_tensor(b, dtype=torch.float64)[None])
    assert float(values.max()) <= 0.0, "fixture must be feasible"

    ts = torch.linspace(0.0, 1.0, 1000, dtype=torch.float64)
    samples = bezier_eval(controls, ts)
    dense = (samples @ torch.as_tensor(A.T, dtype=torch.float64)
             - torch.as_tensor(b, dtype=torch.float64)[None])
    assert float(dense.max()) <= 1e-12


def test_G_a_control_outside_the_region_would_be_caught():
    A, b = _box_halfspaces(0.30, 0.20)
    controls = torch.tensor([
        [-0.20, -0.10], [-0.25, 0.12], [0.22, 0.15], [0.60, -0.05],
    ], dtype=torch.float64)
    values = (controls @ torch.as_tensor(A.T, dtype=torch.float64)
              - torch.as_tensor(b, dtype=torch.float64)[None])
    assert float(values.max()) > 0.0
    ts = torch.linspace(0.0, 1.0, 1000, dtype=torch.float64)
    samples = bezier_eval(controls, ts)
    dense = (samples @ torch.as_tensor(A.T, dtype=torch.float64)
             - torch.as_tensor(b, dtype=torch.float64)[None])
    assert float(dense.max()) > 0.0
