import numpy as np
from experiments.safety_baselines.adapters.curve_control import CurveControlTask
from experiments.safety_baselines.adapters.constraints import MapConstraint
from src.geometry.bspline import BSplineCodec

def task():
    t = CurveControlTask.__new__(CurveControlTask)
    t.C = 48; t.D = 92
    t.endpoints = np.array([[-.8, -.8], [.8, .8]])
    codec = BSplineCodec(num_controls=48, knots_path='auto')
    t.B = codec.basis_at(np.linspace(0, 1, 512)).numpy().astype(float)
    occ = np.zeros((32, 32), bool); occ[12:20, 12:20] = True
    t.map = MapConstraint(occ, extent=2., radius=0.)
    return t

def test_constraint_state_is_exactly_delivered_curve():
    t = task(); x = np.random.default_rng(9).normal(size=t.D)
    output = t.unpack(x)
    np.testing.assert_array_equal(t.points(x), t.B @ output)
    np.testing.assert_allclose(t.constraints(x)[0], -t.map.query(t.B @ output)[0])
    # Every optimized coordinate survives into the output; no residual discarded.
    delta = np.random.default_rng(11).normal(size=t.D) * .01
    np.testing.assert_allclose(t.points(x+delta)-t.points(x),
        t.B[:, 1:-1] @ delta.reshape(46, 2), atol=1e-14)
    np.testing.assert_array_equal(output[[0,-1]], t.endpoints)

def test_decoded_curve_constraint_jacobian():
    t = task(); x = np.random.default_rng(7).normal(size=t.D) * .2
    v = np.random.default_rng(8).normal(size=t.D); eps = 1e-7
    g, J = t.constraints(x)
    fd = (t.constraints(x+eps*v)[0]-t.constraints(x-eps*v)[0])/(2*eps)
    np.testing.assert_allclose(J @ v, fd, atol=1e-7)
