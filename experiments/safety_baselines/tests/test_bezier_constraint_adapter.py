import numpy as np
import pytest
import torch
from src.geometry.bspline import BSplineCodec
from src.geometry.bspline_constraints import build_constraint_pack_from_regions
from src.diffusion.bspline_alm import constraint_state
from experiments.safety_baselines.adapters.bezier_constraints import BezierControlConstraints
from experiments.safety_baselines.algorithms.solvers import Budget,SolverFailure

def fixture():
    codec=BSplineCodec(num_controls=48,knots_path='auto')
    A=torch.tensor([[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]]*2,dtype=torch.float64)
    b=torch.tensor([[.1,1.,.3,.3],[1.,.1,.3,.3]],dtype=torch.float64)
    pack=build_constraint_pack_from_regions(codec,A[None],b[None],torch.ones(1,2,dtype=torch.bool),dtype=torch.float64)
    ends=np.array([[-.8,0.],[.8,0.]])
    return codec,pack,BezierControlConstraints.from_pack(pack,ends)

def test_flattened_constraints_match_ours_alm_exactly():
    _,pack,c=fixture();x=np.random.default_rng(0).normal(size=92)
    q=c.unpack(x);_,g,mask=constraint_state(torch.tensor(q[None],dtype=torch.float64),pack)
    expected=g[mask.expand_as(g)].numpy()
    actual,J=c.constraints(x)
    np.testing.assert_allclose(actual,expected,atol=2e-15)
    v=np.random.default_rng(2).normal(size=92);eps=1e-6
    np.testing.assert_allclose((c.constraints(x+eps*v)[0]-actual)/eps,J@v,atol=1e-8)

def test_convex_projection_preserves_endpoints_and_entire_curve():
    codec,pack,c=fixture()
    q=np.linspace(c.endpoints[0],c.endpoints[1],48);q[1:-1,1]=.8
    c.feasible_start(Budget(3))
    value,info=c.project(q[1:-1].reshape(-1),Budget(3))
    out=c.unpack(value)
    assert c.constraints(value)[0].max()<=1e-7
    np.testing.assert_array_equal(out[[0,-1]],c.endpoints)
    # Dense check independent from the Bezier halfspace evaluation.
    u=np.linspace(0,1,10001);points=codec.basis_at(u).numpy().astype(float)@out
    assert np.max(np.abs(points[:,1]))<=.3+1e-7
    assert np.max(points[u<=.5,0])<=.1+1e-7
    assert np.min(points[u>=.5,0])>=-.1-1e-7
    # A feasible output is a fixed point, without fitting or another optimizer.
    again,_=c.project(value,Budget(3))
    np.testing.assert_allclose(again,value,atol=1e-6)

def test_infeasible_conditioning_is_rejected():
    _,pack,c=fixture()
    c=BezierControlConstraints.from_pack(pack,np.array([[-2.,0.],[.8,0.]]))
    with pytest.raises(SolverFailure,match='bezier_corridor_infeasible'):
        c.feasible_start(Budget(3))
