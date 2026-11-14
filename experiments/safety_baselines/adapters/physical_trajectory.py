"""Physical point-state adapter of the SAME learned B-spline control prior.
P_free=offset+S y, y=(q,z), S=[B_interior,N], N^T B_interior=0.
Clean z=0, noisy z independent Gaussian. q marginal and network are unchanged.
Scores transform by S^{-T}; all optimizer variables are physical trajectory points.
Final Q is a fixed endpoint-constrained LS fit, with no safety fallback.
"""
import numpy as np
from experiments.safety_baselines.adapters.trajectory_state import TrajectoryStateTask
from experiments.safety_baselines.adapters.paper_task import PaperTask
from experiments.safety_baselines.algorithms.solvers import SolverFailure
class PhysicalTrajectoryTask(TrajectoryStateTask):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  # Fixed geometric unit normalization, independent of outcomes/hyperparameter tuning.
  # Ensures the minimum forward-noise covariance eigenvalue in state units is 1.
  self.coordinate_scale=1./float(np.linalg.svd(self.S[1:-1],compute_uv=False).min())
 def latent(self,x):
  x=np.atleast_2d(x).reshape(-1,126,2)
  return np.einsum('ij,bjk->bik',self.inverse,x/self.coordinate_scale-self.offset[1:-1]).reshape(len(x),-1)
 def from_latent(self,y):
  y=np.asarray(y);single=y.ndim==1;y=np.atleast_2d(y).reshape(-1,126,2)
  p=(np.einsum('ij,bjk->bik',self.S[1:-1],y)+self.offset[1:-1]).reshape(len(y),-1)
  p=p*self.coordinate_scale
  return p[0] if single else p
 def points(self,x):
  p=np.empty((128,2));p[0]=self.endpoints[0];p[-1]=self.endpoints[1];p[1:-1]=np.asarray(x).reshape(126,2)/self.coordinate_scale;return p
 def unpack(self,x):return PaperTask.unpack(self,self.latent(x)[0,:self.qD])
 def initial(self,rng,particles=1,ve=False):
  y=rng.normal(size=(particles,self.D))*(self.sigma[-1] if ve else 1.)
  return self.from_latent(y)
 def constraints(self,x):
  h,grad=self.map.query(self.points(x));J=np.zeros((128,self.D))
  ids=np.arange(126);J[ids+1,2*ids]=-grad[1:-1,0]/self.coordinate_scale;J[ids+1,2*ids+1]=-grad[1:-1,1]/self.coordinate_scale
  return -h,J
 def clean(self,x,t,ve=False):
  y=self.latent(x)
  # Call learned q denoiser only; no synthetic point state enters the network.
  q=PaperTask.clean(self,y[:,:self.qD],t,ve)
  return self.from_latent(np.concatenate([q,np.zeros_like(y[:,self.qD:])],axis=1))
 def score(self,x,t,ve=False):
  y=self.latent(x)
  q=PaperTask.clean(self,y[:,:self.qD],t,ve)
  mean=np.concatenate([q,np.zeros_like(y[:,self.qD:])],axis=1)
  sy=-(y-mean)/(self.sigma[t]**2) if ve else -(y-np.sqrt(self.ab[t])*mean)/(1-self.ab[t])
  return np.einsum('ij,bjk->bik',self.inverse.T,sy.reshape(-1,126,2)).reshape(len(y),-1)/self.coordinate_scale
 def nominal(self,x,t,clean):
  if t==0:return clean
  off=self.offset[1:-1].reshape(-1)*self.coordinate_scale
  eps=(x-off-np.sqrt(self.ab[t])*(clean-off))/np.sqrt(1-self.ab[t])
  return off+np.sqrt(self.ab[t-1])*(clean-off)+np.sqrt(1-self.ab[t-1])*eps
 def project(self,target,budget):
  budget.check()
  if self.map.radius!=0:raise SolverFailure('nonzero_radius_projection_not_implemented')
  p=self.points(target);h,grad=self.map.query(p);eps=1e-7
  if np.any(h[[0,-1]] < -1e-10):raise SolverFailure('projection_endpoint_infeasible')
  near=h[1:-1]<=eps;pp=p[1:-1].copy()
  pp[near]+=(-h[1:-1][near]+eps)[:,None]*grad[1:-1][near]
  x=pp.reshape(-1)*self.coordinate_scale;g,_=self.constraints(x)
  if np.max(g)>1e-5:raise SolverFailure('exact_projection_residual')
  # For zero radius, nearest boundary + epsilon interior is globally nearest
  # per point up to epsilon, even though the free-cell union is nonconvex.
  return x,{'status':'nearest_free_union','iterations':1,'residual':float(g.max()),'rounding_eps_scene':eps,'projection_norm_error_bound':float(np.sqrt(126)*eps*self.coordinate_scale)}
 def cbf_solve(self,nominal,J,offset,w,budget,tol):
  # Rows have disjoint point-coordinate supports: exact closed-form convex QP.
  budget.check();norm=np.sum(J*J,axis=1)+w*w
  residual=J@nominal+offset
  if np.any((norm<1e-15)&(residual < -tol)):raise SolverFailure('qp_fixed_endpoint_infeasible')
  alpha=np.maximum(-residual,0)/np.maximum(norm,1e-30)
  v=nominal+J.T@alpha;r=-w*alpha if w else np.empty(0)
  primal=J@v+offset+(w*w)*alpha
  if primal.min() < -tol:raise SolverFailure('qp_closed_form_residual')
  dual=float(np.max(np.abs(v-nominal-J.T@alpha)))
  return v,r,{'qp_status':'closed_form_exact','cbf_residual':float(primal.min()),'kkt_dual':dual,'complementarity_residual':float(np.max(np.abs(alpha*primal))),'iterations':1}

 def projection_initial(self,budget):
  return self.from_latent(super().projection_initial(budget))
 def projection_start_for(self,target,budget):return self.project(target,budget)[0]

 def max_point_speed(self,v):return float(np.linalg.norm(np.asarray(v).reshape(126,2),axis=1).max()/self.coordinate_scale)
