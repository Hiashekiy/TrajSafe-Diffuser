import numpy as np
import torch
import pytest
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.adapters.native_points import NativePointTask
from experiments.safety_baselines.adapters.point_evaluator import evaluate_points
from experiments.safety_baselines.algorithms.solvers import Budget,cbf_velocity

class ZeroPointModel(torch.nn.Module):
    num_points=128
    def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
    def encode_scene(self,occ):return torch.zeros(1,1,1)
    def forward_from_memory(self,p,mem,cond,t):return {'trajectory':torch.zeros_like(p)}

def task():return NativePointTask(ZeroPointModel(),NoiseSchedule(4),np.zeros((16,16),bool),np.array([[-.8,0.],[.8,0.]]))

def test_native_state_is_delivered_output_and_exact_projection():
    t=task();x=np.random.default_rng(1).normal(size=t.D)*4
    out,info=t.project(x,Budget(3));p=t.unpack(out)
    np.testing.assert_array_equal(p,t.points(out))
    np.testing.assert_array_equal(p[1:-1].reshape(-1),out)
    np.testing.assert_array_equal(p[[0,-1]],t.endpoints)
    np.testing.assert_allclose(p[1:-1],np.clip(x.reshape(-1,2),-1,1),atol=1e-7)
    assert t.constraints(out)[0].max()<=1e-5

@pytest.mark.parametrize('w',[0.,100.])
def test_native_qp_is_same_paper_objective(w):
    t=task();x=t.initial(np.random.default_rng(3))[0];g,J=t.constraints(x);J=-J;b=-g-.001
    nominal=np.random.default_rng(4).normal(size=t.D)
    v,r,_=t.cbf_solve(nominal,J,b,w,Budget(3),1e-5)
    ref,slack,_=cbf_velocity(nominal,J,b,w,Budget(3))
    np.testing.assert_allclose(v,ref,atol=1e-5)
    np.testing.assert_allclose(r,slack,atol=1e-5)

def test_point_score_vp_ve_normalization():
    t=task();x=t.initial(np.random.default_rng(5))[0]
    np.testing.assert_allclose(t.score(x,2),-x[None]/(1-t.ab[2]))
    np.testing.assert_allclose(t.score(x,2,ve=True),-x[None]/t.sigma[2]**2)

def test_evaluator_catches_segment_crossing_with_safe_vertices():
    occ=np.zeros((32,32),bool);occ[12:20,12:20]=True
    p=np.array([[-.8,0.],[.8,0.]])
    result=evaluate_points(p,occ,p,p)
    assert not result['vertex_collision'] and result['collision'] and result['segment_only_collision']
