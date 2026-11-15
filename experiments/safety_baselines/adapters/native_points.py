"""Native waypoint state for the point checkpoint. No spline or state lifting."""
import numpy as np
import torch
from experiments.safety_baselines.adapters.exact_map import ExactMapConstraint
from experiments.safety_baselines.algorithms.solvers import SolverFailure

class NativePointTask:
    def __init__(self,model,schedule,occupancy,cond,points=128,radius=0.,field_kind='exact'):
        if field_kind!='exact':raise ValueError('native point task requires the exact map distance')
        self.model=model;self.H=model.num_points;self.D=2*(self.H-2)
        if self.H!=points:raise ValueError('checkpoint/constraint point count differs')
        device=next(model.parameters()).device
        self.occ=torch.as_tensor(occupancy,device=device,dtype=torch.float32)[None,None]
        self.cond=torch.as_tensor(cond,device=device,dtype=torch.float32)[None]
        self.endpoints=np.asarray(cond,dtype=float)
        with torch.inference_mode():self.memory=model.encode_scene(self.occ)
        self.map=ExactMapConstraint(occupancy,extent=2.,radius=radius)
        # `radius` is a margin for the *interior* of the path; the two endpoints are given by
        # the task and cannot be moved, so they are tested against the un-inflated map.
        # Without this, any margin wider than the endpoint clearance makes every projection
        # method report 'fixed endpoint is occupied' (~half the test scenes at 5 m).
        self.map_raw=ExactMapConstraint(occupancy,extent=2.,radius=0.)
        self.ab=schedule.alphas_cumprod.cpu().numpy();self.betas=schedule.betas.cpu().numpy()
        self.sigma=np.sqrt((1-self.ab)/self.ab)

    def unpack(self,x):
        x=np.asarray(x);p=np.empty((*x.shape[:-1],self.H,2))
        p[...,0,:]=self.endpoints[0];p[...,-1,:]=self.endpoints[1]
        p[...,1:-1,:]=x.reshape(*x.shape[:-1],self.H-2,2)
        return p
    def points(self,x):return self.unpack(x)
    def initial(self,rng,particles=1,ve=False):
        # Paired protocol: particle 0 is the shared trajectory noise vector, the
        # remaining particles come from a *method independent* seed so that every
        # multi-particle method (Primal-Dual, ALM) starts from the identical set.
        override=getattr(self,'initial_override',None)
        if override is not None:
            base=np.asarray(override,dtype=float).reshape(-1)
            if base.shape!=(self.D,):raise ValueError('initial_override shape mismatch')
            if particles<=1:noise=base[None]
            else:
                extra=np.random.default_rng(int(getattr(self,'particle_seed',0))).normal(size=(particles-1,self.D))
                noise=np.concatenate([base[None],extra],axis=0)
            return (noise*self.sigma[-1] if ve else noise)
        return rng.normal(size=(particles,self.D))*(self.sigma[-1] if ve else 1.)
    @torch.inference_mode()
    def clean(self,x,t,ve=False):
        x=np.atleast_2d(x);state=x*np.sqrt(self.ab[t]) if ve else x
        p=torch.as_tensor(self.unpack(state),device=self.cond.device,dtype=torch.float32)
        tt=torch.full((len(x),),t,device=p.device,dtype=torch.long)
        pred=self.model.forward_from_memory(p,self.memory.expand(len(x),-1,-1),self.cond.expand(len(x),-1,-1),tt)['trajectory']
        if not torch.isfinite(pred).all():raise RuntimeError('nonfinite point denoiser')
        return pred[:,1:-1].reshape(len(x),-1).cpu().numpy().astype(float)
    def score(self,x,t,ve=False):
        clean=self.clean(x,t,ve)
        return -(np.atleast_2d(x)-clean)/self.sigma[t]**2 if ve else -(np.atleast_2d(x)-np.sqrt(self.ab[t])*clean)/(1-self.ab[t])
    def values(self,x):
        # Barrier values only (g = -h, g<=0 feasible) without building a Jacobian.
        return -self.map.query(self.unpack(np.asarray(x,dtype=float)))[0]

    def point_barrier(self,x):
        """Scene side of the per-point CBF: clearance h and grad h for every FREE point.

        The official SafeDiffuser corrections evaluate the barrier at one position per
        timestep and need exactly this pair (`diffusion.py:459-490` reads it from two
        hardcoded plate ellipses; here it comes from our occupancy map).
        """
        h,grad=self.map.query(self.unpack(np.asarray(x,dtype=float)))
        return h[1:-1],grad[1:-1]

    def constraints(self,x):
        h,grad=self.map.query(self.unpack(x));J=np.zeros((self.H,self.D));i=np.arange(self.H-2)
        J[i+1,2*i]=-grad[1:-1,0];J[i+1,2*i+1]=-grad[1:-1,1]
        return -h,J
    def project(self,target,budget):
        budget.check()
        # A nonzero `radius` inflates the obstacle set by that margin; pushing a point until
        # h = 0 of the inflated field is exactly "leave a margin of `radius`". One Newton
        # step only closes a thin inflation, so iterate: with a 5 m margin the single-step
        # version left a residual and every projection failed (observed 2026-11-14).
        p=self.unpack(target);h,grad=self.map.query(p)
        raw_h,_=self.map_raw.query(p[[0,-1]])
        if np.min(raw_h) < -1e-10:raise SolverFailure('fixed endpoint is occupied')
        eps=1e-7;free=p[1:-1].copy()
        for _ in range(64):
            hh=self.constraints(free.reshape(-1))[0]
            if hh.max()<=1e-5:break
            budget.check()
            hh2,gg=self.map.query(self.unpack(free.reshape(-1)))
            bad=hh2[1:-1]<=eps
            if not bad.any():break
            free[bad]+=(-hh2[1:-1][bad]+eps)[:,None]*gg[1:-1][bad]
        value=free.reshape(-1);residual=float(self.constraints(value)[0].max())
        if residual>1e-5:raise SolverFailure('native point projection residual')
        return value,{'status':'nearest_free_cell_union','residual':residual,'iterations':1,'projection_error_bound_scene':float(np.sqrt(self.H-2)*eps)}
    def cbf_solve(self,nominal,J,offset,w,budget,tol):
        # Independent point supports make the PAPER QP separable. Exact solve.
        budget.check();norm=np.sum(J*J,axis=1)+w*w;rhs=J@nominal+offset
        if np.any((norm<1e-15)&(rhs<-tol)):raise SolverFailure('fixed endpoint CBF infeasible')
        alpha=np.maximum(-rhs,0)/np.maximum(norm,1e-30)
        v=nominal+J.T@alpha;r=-w*alpha if w else np.empty(0)
        residual=J@v+offset+w*w*alpha
        if residual.min()<-tol:raise SolverFailure('native point QP residual')
        return v,r,{'qp_status':'exact_separable_paper_qp','cbf_residual':float(residual.min()),'kkt_dual':float(np.max(np.abs(v-nominal-J.T@alpha))),'iterations':1}
    def max_point_speed(self,v):return float(np.linalg.norm(np.asarray(v).reshape(-1,2),axis=1).max())
