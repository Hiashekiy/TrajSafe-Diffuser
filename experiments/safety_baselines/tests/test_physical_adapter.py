import numpy as np
import pytest
from src.geometry.bspline import BSplineCodec,numpy_fit_curve_to_controls
from experiments.safety_baselines.adapters.physical_trajectory import PhysicalTrajectoryTask
from experiments.safety_baselines.adapters.exact_map import ExactMapConstraint
from experiments.safety_baselines.adapters.paper_task import PaperTask
from experiments.safety_baselines.algorithms.solvers import Budget,cbf_velocity

def make_task():
 c=BSplineCodec(num_controls=48,knots_path='auto');B=c.basis_at(np.linspace(0,1,128)).numpy().astype(float)
 t=PhysicalTrajectoryTask.__new__(PhysicalTrajectoryTask);t.C=48;t.D=252;t.qD=92;t.B=B
 u,s,v=np.linalg.svd(B[1:-1,1:-1],full_matrices=True);t.N=np.zeros((128,80));t.N[1:-1]=u[:,46:];t.S=np.c_[B[:,1:-1],t.N];t.inverse=np.linalg.inv(t.S[1:-1]);t.coordinate_scale=1/np.linalg.svd(t.S[1:-1],compute_uv=False).min()
 t.endpoints=np.array([[-.7,-.3],[.7,.3]]);t.offset=B[:,0,None]*t.endpoints[0]+B[:,-1,None]*t.endpoints[1]
 t.map=ExactMapConstraint(np.zeros((16,16)),extent=2.)
 return t,c

def test_prior_marginal_and_endpoint_fixed_ls_output():
 t,c=make_task();r=np.random.default_rng(7);y=r.normal(size=t.D)*.2;x=t.from_latent(y)
 np.testing.assert_allclose(t.latent(x)[0],y,atol=1e-12)
 p=t.points(x);q=t.unpack(x)
 fit=numpy_fit_curve_to_controls(c.knots.numpy(),48,3,p,t.endpoints[0],t.endpoints[1])
 np.testing.assert_allclose(fit,q,atol=2e-6)
 np.testing.assert_allclose(q[1:-1].reshape(-1),y[:92],atol=1e-12)
 np.testing.assert_allclose(t.B[:,1:-1].T@t.N,0,atol=1e-12)
 mineig=np.linalg.eigvalsh(t.coordinate_scale**2*(t.S[1:-1]@t.S[1:-1].T)).min();assert abs(mineig-1)<1e-10

@pytest.mark.parametrize('ve',[False,True])
def test_transformed_score_matches_gaussian_log_density(monkeypatch,ve):
 t,c=make_task();a=.36;t.ab=np.array([a]);t.sigma=np.sqrt((1-t.ab)/t.ab);tau=.25
 def clean(self,x,level,ve=False):
  x=np.atleast_2d(x)*(np.sqrt(a) if ve else 1.)
  return np.sqrt(a)*tau/(a*tau+1-a)*x
 monkeypatch.setattr(PaperTask,'clean',clean)
 r=np.random.default_rng(5);x=t.from_latent(r.normal(size=t.D)*.1)
 def logp(xx):
  y=t.latent(xx)[0];v=(tau+(1-a)/a if ve else a*tau+1-a);zv=(1-a)/a if ve else 1-a
  return -.5*(np.sum(y[:92]**2)/v+np.sum(y[92:]**2)/zv)
 score=t.score(x,0,ve)[0]
 for k in [0,1,17,91,92,107,150,251]:
  d=np.zeros(t.D);d[k]=1e-6
  np.testing.assert_allclose((logp(x+d)-logp(x-d))/2e-6,score[k],atol=1e-7,rtol=1e-6)

def test_exact_union_distance_and_gradient():
 occ=np.zeros((16,16),bool);occ[6:10,6:10]=True;f=ExactMapConstraint(occ,extent=2.)
 p=np.array([[.04,.03],[.43,.19],[-1.3,-1.2]])
 h,J=f.query(p)
 for k in range(2):
  d=np.zeros_like(p);d[:,k]=1e-7
  np.testing.assert_allclose((f.query(p+d)[0]-f.query(p-d)[0])/2e-7,J[:,k],atol=1e-6)
 assert h[0]<0 and h[1]>0 and h[2]<0

def test_global_nearest_free_square_projection():
 t,c=make_task();r=np.random.default_rng(10);target=r.normal(size=t.D)*5
 p=t.points(target);x,info=t.project(target,Budget(3));projected=t.points(x)
 ideal=np.clip(p[1:-1],-1,1)
 assert np.linalg.norm(projected[1:-1]-ideal)<=np.sqrt(126)*1e-7+1e-10
 assert t.constraints(x)[0].max()<=1e-5

@pytest.mark.parametrize('w',[0.,100.])
def test_closed_form_cbf_qp_matches_independent_osqp(w):
 t,c=make_task();r=np.random.default_rng(8);x=t.from_latent(r.normal(size=t.D));g,J=t.constraints(x);J=-J;offset=-g-.001
 v=r.normal(size=t.D)
 a,sl,info=t.cbf_solve(v,J,offset,w,Budget(3),1e-5)
 b,slb,_=cbf_velocity(v,J,offset,w,Budget(3),1e-5)
 np.testing.assert_allclose(a,b,atol=1e-4);np.testing.assert_allclose(sl,slb,atol=1e-4)
 assert info['cbf_residual']>=-1e-5
