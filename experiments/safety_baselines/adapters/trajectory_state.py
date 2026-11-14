"""Analytic full-rank trajectory state for the SAME learned control prior.
y=(q_interior,z), z0=0, zt Gaussian; P=BQ+N z with B_interior.T@N=0.
The exact score factors into the existing learned q score and analytic z score.
Output Q is the shared endpoint-fixed least-squares fit; no safety repair.
"""
import numpy as np
from experiments.safety_baselines.adapters.paper_task import PaperTask
class TrajectoryStateTask(PaperTask):
 def __init__(self,model,schedule,occupancy,cond,points,radius=0.,field_kind='exact'):
  # Native trajectory sample count is the existing curve representation (128).
  super().__init__(model,schedule,occupancy,cond,128,radius,field_kind)
  self.qD=self.D;u,s,v=np.linalg.svd(self.B[1:-1,1:-1],full_matrices=True)
  self.N=np.zeros((128,128-self.C));self.N[1:-1]=u[:,self.C-2:]
  self.S=np.concatenate([self.B[:,1:-1],self.N],axis=1);self.D=2*self.S.shape[1]
  self.inverse=np.linalg.inv(self.S[1:-1]);self.offset=self.B[:,0,None]*self.endpoints[0]+self.B[:,-1,None]*self.endpoints[1]
 def unpack(self,x):return super().unpack(np.asarray(x)[...,:self.qD])
 def points(self,x):
  state=np.asarray(x).reshape(-1,2)
  return self.S@state+self.offset
 def constraints(self,x):
  h,grad=self.map.query(self.points(x));J=-(self.S[:,:,None]*grad[:,None,:]).reshape(128,-1)
  return -h,J
 def clean(self,x,t,ve=False):
  x=np.atleast_2d(x);q=super().clean(x[:,:self.qD],t,ve)
  return np.concatenate([q,np.zeros_like(x[:,self.qD:])],axis=1)
 def projection_initial(self,budget):
  q=super().projection_initial(budget)
  return np.r_[q,np.zeros(self.D-self.qD)]
 def projection_start_for(self,target,budget):
  # Pointwise Euclidean nearest free points provide a feasible warm start only.
  # The actual operator still solves the Euclidean y-space nearest problem.
  budget.check();p=self.points(target);h,grad=self.map.query(p);unsafe=h<0
  p[unsafe]=p[unsafe]+(-h[unsafe]+1e-7)[:,None]*grad[unsafe]
  p[0]=self.endpoints[0];p[-1]=self.endpoints[1]
  state=(self.inverse@(p[1:-1]-self.offset[1:-1])).reshape(-1)
  if np.max(self.constraints(state)[0])>1e-5:return self.projection_initial(budget)
  return state
