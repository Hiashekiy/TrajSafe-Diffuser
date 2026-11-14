import numpy as np
import pytest
from experiments.safety_baselines.algorithms.solvers import Budget,nearest_projection,cbf_velocity,SolverFailure
from experiments.safety_baselines.algorithms.safediffuser.runner import legal_gamma,tvs_schedule,run as safe_run
from experiments.safety_baselines.algorithms.constrained_diffusers.runner import run as cd_run
from experiments.safety_baselines.algorithms.pdm.runner import run as pdm_run
from experiments.safety_baselines.algorithms.runner import derived_seed
class Toy:
    D=2;B=np.zeros((1,2));ab=np.array([.9,.7,.4,.1]);betas=np.array([.1,.2,.3,.4]);sigma=np.sqrt((1-ab)/ab)
    def constraints(self,x):return np.array([-x[0]]),np.array([[-1.,0.]])
    def clean(self,x,t,ve=False):return np.ones_like(np.atleast_2d(x))*.5
    def score(self,x,t,ve=False):return -(np.atleast_2d(x)-.5)/(self.sigma[t]**2 if ve else 1-self.ab[t])
CFG={'cbf_dt':1.,'cbf_epsilon':1.,'cbf_gamma_floor':.001,'hard_steps':2,'relax_weight':100.,'qp_tol':1e-5,'particles':4,'rho':.1,'rho_growth':1.1,'dual_eta':.1,'projection_maxiter':60,'projection_tol_scene':1e-5,'pdm_inner':3}

def test_gamma_and_tvs_conditions():
 h=np.array([-2.,3.]);g=legal_gamma(h,4,1,.001)
 assert (g+1e-12>=np.abs(g-h)*np.exp(-4)).all()
 a,d=tvs_schedule(h,4,4,1);assert (a<=h).all()
 assert np.array_equal(tvs_schedule(h,0,4,1)[0],[0.,0.]);assert d[0]>0

def test_projection_nearest_and_failure():
 fun=lambda x:np.array([-x[0]]);jac=lambda x:np.array([[-1.,0.]])
 x,info=nearest_projection(np.array([-2.,3.]),fun,jac,None,Budget(3))
 np.testing.assert_allclose(x,[0,3],atol=1e-7);assert info['stationarity']<1e-5
 with pytest.raises(SolverFailure):nearest_projection(np.zeros(2),lambda x:np.ones(1),lambda x:np.zeros((1,2)),None,Budget(3))

def test_nonlinear_circle_projection():
 fun=lambda x:np.array([1-np.dot(x,x)]);jac=lambda x:-2*x[None]
 x,info=nearest_projection(np.array([.5,0.]),fun,jac,np.array([1.,0.]),Budget(3))
 np.testing.assert_allclose(x,[1,0],atol=1e-5)

def test_qp_kkt_and_terminal_hard_slack():
 v,r,i=cbf_velocity(np.array([-1.,2.]),np.array([[1.,0.]]),np.zeros(1),0,Budget(3))
 np.testing.assert_allclose(v,[0,2],atol=1e-5);assert r.size==0
 v,r,i=cbf_velocity(np.array([-1.,2.]),np.array([[1.,0.]]),np.zeros(1),1,Budget(3))
 np.testing.assert_allclose([v[0],r[0]],[-.5,-.5],atol=1e-5)

def test_relaxed_qp_elimination_at_paper_weight():
 # Analytic scalar solution with w=100, the original badly scaled QP case.
 w=100.;v,r,info=cbf_velocity(np.array([-2.]),np.array([[1.]]),np.array([0.]),w,Budget(3))
 np.testing.assert_allclose(v,[-2*w*w/(1+w*w)],atol=1e-8)
 np.testing.assert_allclose(r,[-2*w/(1+w*w)],atol=1e-8)
 assert info['cbf_residual']>=-1e-10 and info['kkt_dual']<1e-7

def test_qp_row_scaling_preserves_original_problem():
 J=np.diag([1e-4,1e4]);offset=np.array([0.,0.])
 v,_,info=cbf_velocity(np.array([-1.,-2.]),J,offset,0,Budget(3))
 np.testing.assert_allclose(v,[0,0],atol=1e-5)
 assert info['cbf_residual']>=-1e-5

def test_coupled_qp_infeasibility_is_independently_certified():
 # One shared variable cannot simultaneously move left and right at unit speed.
 with pytest.raises(SolverFailure,match='qp_infeasible_certified'):
  cbf_velocity(np.zeros(1),np.array([[1.],[-1.]]),np.array([-1.,-1.]),0,Budget(3))

@pytest.mark.parametrize('method',['RoS-diffuser','ReS-diffuser','TVS-diffuser','Projected','Primal-Dual','ALM','PDM'])
def test_actual_loops_and_determinism(method):
 def once():
  trace=[];fn=safe_run if method.endswith('diffuser') else pdm_run if method=='PDM' else cd_run
  x=fn(Toy(),method,np.random.default_rng(7),CFG,Budget(5),trace);return x,trace
 x,t=once();y,u=once();np.testing.assert_allclose(x,y);assert np.isfinite(x).all()
 assert len(t)==(12 if method=='PDM' else 4)
 if method=='ReS-diffuser':assert [s['w'] for s in t]==[100,100,0,0]
 if method=='Primal-Dual':assert all(s['lambda_min']>=0 for s in t)
 if method=='ALM':assert t[-1]['rho_next']>t[0]['rho_next']

def test_seed_independence():
 assert derived_seed(1,2,'PDM')==derived_seed(1,2,'PDM')
 assert derived_seed(1,2,'PDM')!=derived_seed(1,2,'Projected')


def test_native_pdm_and_cd_additional_budgets():
 cfg=dict(CFG,pdm_inner=100,pdm_levels=3,cd_extra_steps=4)
 tr=[];pdm_run(Toy(),'PDM',np.random.default_rng(7),cfg,Budget(10),tr)
 assert len(tr)==300 and len({s['t'] for s in tr})==3
 tr=[];cd_run(Toy(),'Primal-Dual',np.random.default_rng(7),cfg,Budget(5),tr)
 assert len(tr)==8 and sum(s['t']==0 for s in tr)==5


def test_dual_units_invariance():
 from experiments.safety_baselines.algorithms.kernels import primal_dual_step,alm_step
 x=np.zeros((2,2));score=np.ones((2,2));g=np.array([[.2],[.5]]);J=np.ones((2,1,2));lam=np.ones(1)*.1;noise=np.zeros((2,2));s=128.
 a,l=primal_dual_step(x,score,g,J,lam,.1,.2,noise)
 b,ll=primal_dual_step(x,score,s*g,s*J,lam/s,.1,.2/s**2,noise)
 np.testing.assert_allclose(a,b);np.testing.assert_allclose(l/s,ll)
 a,l,sl,rho=alm_step(x,score,g,J,lam,.3,1.1,.1,noise)
 b,ll,ssl,rr=alm_step(x,score,s*g,s*J,lam/s,.3/s**2,1.1,.1,noise)
 np.testing.assert_allclose(a,b);np.testing.assert_allclose(l/s,ll);np.testing.assert_allclose(sl*s,ssl)

def test_independent_kkt_can_verify_maxiter(monkeypatch):
 from types import SimpleNamespace
 from experiments.safety_baselines.algorithms import solvers
 monkeypatch.setattr(solvers,'minimize',lambda *a,**kw:SimpleNamespace(x=np.array([0.,3.]),success=False,status=9,message='maxiter',nit=200))
 x,info=solvers.nearest_projection(np.array([-2.,3.]),lambda x:np.array([-x[0]]),lambda x:np.array([[-1.,0.]]),None,Budget(3))
 assert info['status']=='independently_verified_local_kkt_maxiter'
 np.testing.assert_allclose(x,[0,3])
