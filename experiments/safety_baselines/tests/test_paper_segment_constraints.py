import numpy as np
import pytest
from experiments.safety_baselines.adapters.paper_segment_constraints import CircleSegmentConstraints
from experiments.safety_baselines.algorithms.solvers import Budget,SolverFailure

def test_paper_constraint_detects_safe_vertices_but_crossing_segment():
    c=CircleSegmentConstraints([[-.8,0.],[.8,0.]],2,[[0.,0.]],[.2])
    assert np.linalg.norm(c.endpoints,axis=1).min()>.2
    assert c.constraints(np.empty(0))[0][0]>0
    with pytest.raises(SolverFailure,match='fixed path intersects'):c.project(np.empty(0),Budget(3))

def test_segment_jacobian_matches_finite_differences():
    c=CircleSegmentConstraints([[-.8,-.1],[.8,.2]],4,[[.1,0.],[-.2,.2]],[.1,.15])
    x=np.array([-.3,.35,.35,.4]);direction=np.array([.3,-.2,.1,.5]);eps=1e-7
    g,J=c.constraints(x)
    np.testing.assert_allclose((c.constraints(x+eps*direction)[0]-c.constraints(x-eps*direction)[0])/(2*eps),J@direction,atol=1e-8)

def test_interior_point_projects_entire_segments_not_just_vertices():
    c=CircleSegmentConstraints([[-1.,0.],[1.,0.]],3,[[0.,0.]],[.3])
    x,info=c.project(np.array([0.,.05]),Budget(15),maxiter=300)
    assert c.constraints(x)[0].max()<=1e-7
    assert abs(x[0])<1e-4
    np.testing.assert_allclose(x[1],.3/np.sqrt(1-.3**2),atol=5e-4)
    p=c.unpack(x)
    for a,b in zip(p[:-1],p[1:]):
        dense=a+np.linspace(0,1,10001)[:,None]*(b-a)
        assert np.linalg.norm(dense,axis=1).min()>=.3-1e-7
