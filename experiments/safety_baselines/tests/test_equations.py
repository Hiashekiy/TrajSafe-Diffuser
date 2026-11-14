import numpy as np
import pytest
from experiments.safety_baselines.adapters.score_adapter import score_from_x0
from experiments.safety_baselines.algorithms.kernels import *

def test_score():
    x0=np.array([1.,2.]); noise=np.array([.3,-.5]); a=.64
    x=np.sqrt(a)*x0+np.sqrt(1-a)*noise
    np.testing.assert_allclose(score_from_x0(x,x0,a),-noise/np.sqrt(1-a))
    with pytest.raises(ValueError): score_from_x0(x,x0,1)

def test_projected():
    np.testing.assert_allclose(projected_step(np.array([-2.]),np.array([1.]),2,np.array([0.]),lambda x:np.maximum(x,0)),0)

def test_pd():
    x=np.zeros((2,1)); J=np.ones((2,1,1)); g=np.array([[2.],[-1.]])
    y,lam=primal_dual_step(x,x,g,J,np.array([1.]),.2,.5,x)
    np.testing.assert_allclose(y,-.1);np.testing.assert_allclose(lam,1.25)

def test_alm():
    x=np.zeros((2,1));J=np.ones((2,1,1));g=np.full((2,1),-2.)
    y,lam,s,rho=alm_step(x,x,g,J,np.array([1.]),2,1.1,.2,x)
    np.testing.assert_allclose(s,1.5);np.testing.assert_allclose(y,0,atol=1e-12)
    np.testing.assert_allclose(lam,0);assert rho==2.2

def test_qp():
    v,r,trace=cbf_qp([-1.],[[1.]],[0.])
    np.testing.assert_allclose(v,0,atol=1e-8);assert r.size==0
    v,r,_=cbf_qp([-1.],[[1.]],[0.],1.)
    np.testing.assert_allclose([v[0],r[0]],[-.5,-.5],atol=1e-7)
    hard,_,_=cbf_qp([-1.],[[1.]],[0.],0.)
    np.testing.assert_allclose(hard,0,atol=1e-8)
    with pytest.raises(RuntimeError): cbf_qp([0.],[[0.]],[-1.])

def test_pdm_calls_and_seed():
    calls=[]
    def project(x):calls.append(1);return np.maximum(x,0)
    run=lambda:pdm(np.zeros(2),[2.,1.],3,lambda x,t:-x,project,np.random.default_rng(7))
    x,trace=run();assert len(calls)==6; assert len(trace)==6
    assert trace[3]['gamma']==.125
    np.testing.assert_array_equal(x,run()[0])
