"""One scene, endpoint-free control variables, scene units throughout."""
import heapq
import numpy as np
import torch
from experiments.safety_baselines.adapters.constraints import MapConstraint
class PaperTask:
    def __init__(self,model,schedule,occupancy,cond,points,radius=0.,field_kind='bilinear'):
        self.model=model;self.schedule=schedule;self.occ=torch.as_tensor(occupancy,device='cuda',dtype=torch.float32).reshape(1,1,*occupancy.shape)
        self.cond=torch.as_tensor(cond,device='cuda',dtype=torch.float32).reshape(1,2,2)
        with torch.inference_mode():self.memory=model.encode_scene(self.occ) if hasattr(model,'encode_scene') else None
        self.endpoints=np.asarray(cond,dtype=float);self.C=model.num_controls;self.D=(self.C-2)*2
        self.B=model.bspline.basis_at(np.linspace(0,1,points)).detach().cpu().numpy().astype(float)
        if field_kind not in ('legacy_sdf','exact','bilinear'):raise ValueError('unapproved constraint field '+field_kind)
        if field_kind=='legacy_sdf':
            from experiments.safety_baselines.adapters.legacy_constraint_field import LegacyConstraintField
            self.map=LegacyConstraintField(occupancy,radius=radius)
        elif field_kind=='exact':
            from experiments.safety_baselines.adapters.exact_map import ExactMapConstraint
            self.map=ExactMapConstraint(occupancy,extent=2.,radius=radius)
        else:self.map=MapConstraint(occupancy,extent=2.,radius=radius)
        self.ab=schedule.alphas_cumprod.detach().cpu().numpy();self.betas=schedule.betas.detach().cpu().numpy()
        self.sigma=np.sqrt((1-self.ab)/self.ab)
    def unpack(self,x):
        x=np.asarray(x);q=np.zeros((*x.shape[:-1],self.C,2));q[...,1:-1,:]=x.reshape(*x.shape[:-1],self.C-2,2);q[...,0,:]=self.endpoints[0];q[...,-1,:]=self.endpoints[1];return q
    def constraints(self,x):
        q=self.unpack(x);h,grad=self.map.query(self.B@q)
        J=-(self.B[:,1:-1,None]*grad[:,None,:]).reshape(len(self.B),-1)
        return -h,J
    def projection_initial(self,budget):
        """Map-only A* warm start for a local nearest-point solver; never model input.
        No Ours skeleton/corridor or GT. This is explicitly charged to projection.
        """
        occ=self.map.occ;H,W=occ.shape
        cell=lambda p:tuple(np.floor((p+1)*np.array([W,H])/2).astype(int)[::-1])
        start,goal=map(cell,self.endpoints)
        if any(not(0<=p[0]<H and 0<=p[1]<W) or occ[p] for p in [start,goal]):raise RuntimeError('projection_endpoint_occupied')
        queue=[(0.,0.,start)];dist={start:0.};parent={}
        while queue:
            budget.check();_,cost,p=heapq.heappop(queue)
            if cost!=dist[p]:continue
            if p==goal:break
            for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                y,x=p[0]+dy,p[1]+dx
                if not(0<=y<H and 0<=x<W) or occ[y,x]:continue
                if dy and dx and (occ[p[0],x] or occ[y,p[1]]):continue
                c=cost+np.hypot(dy,dx);v=(y,x)
                if c<dist.get(v,float('inf')):
                    dist[v]=c;parent[v]=p;heapq.heappush(queue,(c+np.hypot(y-goal[0],x-goal[1]),c,v))
        if goal not in dist:raise RuntimeError('projection_no_grid_route')
        route=[goal]
        while route[-1]!=start:route.append(parent[route[-1]])
        route=np.array(route[::-1]);p=np.stack([(route[:,1]+.5)*2/W-1,(route[:,0]+.5)*2/H-1],axis=1)
        p[0]=self.endpoints[0];p[-1]=self.endpoints[1]
        arc=np.r_[0,np.linalg.norm(np.diff(p,axis=0),axis=1).cumsum()]
        if arc[-1]<=0:raise RuntimeError('degenerate_grid_route')
        samples=np.linspace(0,arc[-1],128);dense=np.stack([np.interp(samples,arc,p[:,k]) for k in [0,1]],axis=1)
        from src.geometry.bspline import numpy_fit_curve_to_controls
        codec=self.model.bspline
        q=numpy_fit_curve_to_controls(codec.knots.detach().cpu().numpy(),self.C,codec.degree,dense,self.endpoints[0],self.endpoints[1],128)
        return q[1:-1].reshape(-1)
    @torch.inference_mode()
    def clean(self,x,t,ve=False):
        x=np.atleast_2d(x);state=x*np.sqrt(self.ab[t]) if ve else x
        q=PaperTask.unpack(self,state);qt=torch.as_tensor(q,device='cuda',dtype=torch.float32)
        time=torch.full((len(x),),t,device='cuda',dtype=torch.long)
        out=(self.model.forward_from_memory(qt,self.memory.expand(len(x),-1,-1),self.cond.expand(len(x),-1,-1),time) if self.memory is not None else self.model(qt,self.occ.expand(len(x),-1,-1,-1),self.cond.expand(len(x),-1,-1),time))['control']
        value=out[:,1:-1].reshape(len(x),-1).cpu().numpy().astype(float)
        if not np.isfinite(value).all():raise RuntimeError('nonfinite_score_model')
        return value
    def score(self,x,t,ve=False):
        clean=self.clean(x,t,ve)
        return -(np.atleast_2d(x)-clean)/(self.sigma[t]**2) if ve else -(np.atleast_2d(x)-np.sqrt(self.ab[t])*clean)/(1-self.ab[t])
