import numpy as np
import torch
from src.geometry.bspline import BSplineCodec
from experiments.safety_baselines.adapters.evaluator import evaluate_curve
from experiments.safety_baselines.adapters.paper_task import PaperTask

def test_independent_collision_and_endpoint_failure():
 c=BSplineCodec(num_controls=8,knots_path='auto');q=np.stack([np.linspace(-.8,.8,8),np.zeros(8)],axis=1);cond=q[[0,-1]];gt=c.decode_controls(torch.tensor(q,dtype=torch.float32)[None])[0].numpy()
 occ=np.zeros((32,32));occ[15:18,15:18]=1
 r=evaluate_curve(c,q,occ,cond,gt);assert r['collision'] and not r['planning_success'];assert r['evaluation_points']>512
 assert r['clearance_success_m'] is None
 r=evaluate_curve(c,q,np.zeros((32,32)),cond,gt);assert r['planning_success'] and r['rmse_m']<1e-5
 bad=cond.copy();bad[0]+=.1
 r=evaluate_curve(c,q,np.zeros((32,32)),bad,gt);assert not r['arrived'] and not r['planning_success']

def test_vp_ve_score_transformation():
 class Dummy:
  ab=np.array([.36]);sigma=np.sqrt((1-ab)/ab)
  def clean(self,x,t,ve=False):return np.ones_like(np.atleast_2d(x))*.2
 d=Dummy();y=np.array([[.7,-.5]])
 vp=PaperTask.score(d,y*np.sqrt(.36),0,ve=False)
 ve=PaperTask.score(d,y,0,ve=True)
 np.testing.assert_allclose(ve,np.sqrt(.36)*vp)
