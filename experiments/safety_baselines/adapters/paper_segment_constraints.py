"""PDM Appendix C.3: circle-to-entire-segment distance constraints.

Paper text (App. C.3, p.15): "For circular obstacles, this can be represented by
a minimum distance requirement, the circle radius, imposed on the nearest point
to the center falling on a line between p_n and p_n+1. These constraints are
imposed for all line segments. We adapt a similar approach for non-circular
obstacles by composing these of multiple circular constraints, hence, avoiding
over-constraining the problem."

So the obstacle set used by PDM is a union of circles and the constraint row is
`radius <= distance(center, whole segment)`. `obstacle_circles` below is the
audited occupancy-grid -> circle-union conversion: it composes the occupied-cell
union out of inscribed maximal disks, so the circle union *contains* every
occupied cell square (feasibility therefore implies zero occupancy collision)
while the added restriction on free space stays within one grid cell.
"""
import numpy as np
import osqp
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from experiments.safety_baselines.algorithms.solvers import SolverFailure

def obstacle_circles(occupancy,extent=2.,margin=0.):
    """Cover the occupied-cell union with circles (PDM App. C.3 composition).

    Radius at a cell centre is its Euclidean distance to the nearest free cell
    centre, which is >= one cell, hence >= half a cell diagonal: the circle of
    that radius contains the cell square. Every occupied cell centre is either
    an accepted centre or within the radius of an earlier accepted circle, so
    the union covers the whole obstacle footprint (verified, see audit keys).
    """
    occ=np.asarray(occupancy,dtype=bool)
    if occ.ndim!=2:raise ValueError('occupancy must be 2-D')
    H,W=occ.shape;dy=extent/H;dx=extent/W
    ys,xs=np.nonzero(occ)
    cy=-extent/2+(ys+.5)*dy;cx=-extent/2+(xs+.5)*dx
    if len(ys)==0:
        return np.zeros((0,2)),np.zeros(0),{'occupied_cells':0,'circles':0,'covered_fraction':1.,'cell_scene':float(max(dx,dy)),'free_cells_forbidden_fraction':0.}
    edt=distance_transform_edt(occ,sampling=(dy,dx))
    radius=edt[ys,xs]+.7072*max(dx,dy)   # +half cell diagonal: contains the full cell square
    order=np.argsort(-radius,kind='stable')
    covered=np.zeros(len(ys),bool);accepted=np.zeros(len(ys),bool)
    for i in order:
        if covered[i]:continue
        accepted[i]=True
        covered|=((cx-cx[i])**2+(cy-cy[i])**2)<=radius[i]**2
    centers=np.stack([cx[accepted],cy[accepted]],axis=1)
    radii=radius[accepted]+float(margin)
    # Audit: how much *free* space does the circle composition wrongly forbid?
    fy,fx=np.nonzero(~occ);fcy=-extent/2+(fy+.5)*dy;fcx=-extent/2+(fx+.5)*dx
    forbidden=np.zeros(len(fy),bool)
    for (c,r) in zip(centers,radii):forbidden|=((fcx-c[0])**2+(fcy-c[1])**2)<=r*r
    audit={'occupied_cells':int(len(ys)),'circles':int(len(centers)),
        'covered_fraction':float(covered.mean()),'radius_min_scene':float(radii.min()),
        'radius_max_scene':float(radii.max()),'radius_median_scene':float(np.median(radii)),
        'cell_scene':float(max(dx,dy)),'padding_scene':float(.7072*max(dx,dy)),
        'free_cells_forbidden_fraction':float(forbidden.mean()),
        'geometry':'inscribed maximal-disk composition + half-cell-diagonal covering pad',
        'source':'PDM NeurIPS2024 Appendix C.3 (circle union, all line segments)'}
    return centers,radii,audit

class CircleSegmentConstraints:
    def __init__(self,endpoints,num_points,centers,radii):
        self.endpoints=np.asarray(endpoints,dtype=float).reshape(2,2)
        self.num_points=int(num_points);self.D=2*(self.num_points-2)
        self.centers=np.asarray(centers,dtype=float).reshape(-1,2)
        self.radii=np.asarray(radii,dtype=float).reshape(-1)
        if len(self.centers)!=len(self.radii) or np.any(self.radii<=0):raise ValueError('invalid circles')

    def unpack(self,x):
        p=np.empty((self.num_points,2));p[[0,-1]]=self.endpoints
        p[1:-1]=np.asarray(x).reshape(self.num_points-2,2);return p

    def _terms(self,x):
        p=self.unpack(x);a=p[:-1,None,:];v=(p[1:]-p[:-1])[:,None,:]
        squared=np.sum(v*v,axis=-1)
        alpha=np.clip(np.sum((self.centers[None]-a)*v,axis=-1)/np.maximum(squared,1e-30),0.,1.)
        alpha=np.where(squared<1e-24,.5,alpha)
        delta=a+alpha[...,None]*v-self.centers[None]
        dist=np.linalg.norm(delta,axis=-1)
        return p,alpha,delta,dist

    def values(self,x):
        """g only (no sparse Jacobian); the identity/fast path of the projection."""
        _,_,_,dist=self._terms(x)
        return (self.radii[None]-dist).reshape(-1)

    def constraints(self,x):
        """g[i*M+j] = radius_j - distance(segment_i, centre_j); g<=0 is the paper set."""
        _,alpha,delta,dist=self._terms(x)
        g=(self.radii[None]-dist).reshape(-1)
        unit=delta/np.maximum(dist,1e-15)[...,None]
        da=-(1-alpha)[...,None]*unit;db=-alpha[...,None]*unit
        M=len(self.centers);seg=np.arange(self.num_points-1)
        rows=[];cols=[];values=[]
        for off,grad in ((0,da),(1,db)):
            vertex=seg+off;free=(vertex>=1)&(vertex<=self.num_points-2)
            if not free.any():continue
            idx=np.nonzero(free)[0]
            row=np.add.outer(idx*M,np.arange(M))
            for axis in range(2):
                rows.append(row.ravel());cols.append(np.repeat(2*(vertex[idx]-1)+axis,M))
                values.append(grad[idx,:,axis].ravel())
        J=sparse.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(g),self.D)) if values else sparse.csr_matrix((len(g),self.D))
        return g,J

    def residual(self,x):
        return float(self.values(x).max(initial=-np.inf))

    def near_circles(self,x,band):
        """Circles within `band` of the current polyline; a working set, not the feasible set."""
        _,_,_,dist=self._terms(x)
        return np.any(dist-self.radii[None]<band,axis=0)

    def _sqp(self,target,start,budget,maxiter,tol,penalty_weight=1e3,slack_weight=1e6,trust=0.05,band=1e-3,max_rows=256,qp_eps=1e-6):
        """Sequential least-norm quadratic programming for eq.(7) on this set.

        Each iteration solves the least-norm QP of the linearised working set
        (violated rows plus rows within `band` of the boundary) with an L2
        relaxation of the linearised constraints -- the relaxation keeps the QP
        solvable when the linearisation is contradictory between two rows that
        share a fixed endpoint -- plus a trust region, then backtracks on the
        exact L1 merit 0.5||x-target||^2 + *sum(max(g,0)). The L1 penalty is
        exact for finite weight, so the limit is eq.(7) itself rather than a
        biased stationary point; `penalty_weight` is escalated when the merit
        stalls. Terminates at a local KKT point; no global nearest certificate.
        """
        x=np.clip(np.asarray(start,dtype=float),-.9999999,.9999999);n=self.D
        feas=max(tol,1e-6)  # inner stationarity threshold; project() enforces tol
        m=np.inf;work=None
        def merit(z):
            g=(self if work is None else work).values(z)
            return .5*np.sum((z-target)**2)+penalty_weight*np.sum(np.maximum(g,0.))
        for it in range(maxiter):
            budget.check()
            keep=self.near_circles(x,band+trust)  # rows that one step can reach
            if not keep.any():keep=np.ones(len(self.radii),bool)
            work=self.subset(keep);m=merit(x)
            g,J=work.constraints(x);mx=float(work.values(x).max(initial=-np.inf))
            rows=np.nonzero(g>=-band)[0]
            if len(rows)>max_rows:  # limited working set: worst rows only
                rows=np.sort(rows[np.argsort(-g[rows])[:max_rows]])
            if len(rows)==0 and mx<=feas:
                return x,{'iterations':it,'residual':mx,'status':'segment_sqp_feasible'}
            k=len(rows)
            Av=sparse.csc_matrix(J[rows]);u=(Av@x).reshape(-1)-g[rows]
            nrm=np.maximum(np.sqrt(np.asarray(Av.multiply(Av).sum(axis=1)).reshape(-1)),1e-12)
            As=sparse.diags(1./nrm)@Av
            A=sparse.vstack([sparse.hstack([As,-sparse.eye(k,format='csc')]),
                sparse.hstack([sparse.csc_matrix((k,n)),sparse.eye(k,format='csc')])],format='csc')
            P=sparse.diags(np.r_[np.ones(n),np.full(k,slack_weight)],format='csc');q=np.r_[-target,np.zeros(k)]
            solver=osqp.OSQP()
            solver.setup(P=P,q=q,A=A,l=np.r_[np.full(k,-np.inf),np.zeros(k)],
                u=np.r_[u/nrm,np.full(k,np.inf)],verbose=False,eps_abs=qp_eps,eps_rel=qp_eps,max_iter=4000,polishing=True)
            res=solver.solve(raise_error=False)
            if res.x is None or not np.isfinite(res.x).all():raise SolverFailure('segment_sqp_qp_failed:'+str(res.info.status))
            step=np.asarray(res.x[:n],dtype=float);d=step-x;nd=float(np.linalg.norm(d))
            if nd>trust:step=x+d*(trust/nd);nd=trust
            if nd<1e-10 and mx<=feas:return x,{'iterations':it,'residual':mx,'status':'segment_sqp_feasible_stationary'}
            omega=1.;accepted=False
            for _ in range(14):
                cand=x+omega*(step-x);mc=merit(cand)
                if mc<m-1e-14:accepted=True;break
                omega*=.5
            if not accepted:
                if mx<=feas:return x,{'iterations':it,'residual':mx,'status':'segment_sqp_feasible_stationary'}
                # Finite exact penalty is stationary at a small bias; escalate it
                # (standard exact-penalty continuation) to approach eq.(7) itself.
                if penalty_weight<1e12:
                    penalty_weight*=10.;m=merit(x);continue
                raise SolverFailure(f'segment_sqp_no_descent:residual={mx:.6g}:active_rows={k}')
            x=cand;m=mc
        mx=self.residual(x)
        if mx<=feas:return x,{'iterations':maxiter,'residual':mx,'status':'segment_sqp_feasible_maxiter'}
        raise SolverFailure(f'segment_sqp_maxiter:residual={mx:.6g}')

    def finish_feasible(self,x,budget,tol,max_sweeps=8):
        """Gauss-Seidel least-norm row repairs; only used to close a small residual.

        Each violated (segment, circle) row moves its two end vertices along the
        outward normal of the closest point, which is the exact least-norm fix of
        that single linearised row. Cheap and monotone per row, not globally
        nearest; it never leaves the map box and never touches fixed endpoints.
        """
        x=np.asarray(x,dtype=float).copy();M=len(self.radii);H=self.num_points;best=np.inf
        for _ in range(max_sweeps):
            budget.check();g=self.values(x);viol=np.nonzero(g>tol)[0]
            if not len(viol):return x,0.
            here=float(g.max())
            if here>=best*(1-1e-3):return x,here   # no progress: stop early
            best=here
            for row in viol[np.argsort(-g[viol])]:
                i,j=divmod(int(row),M)
                p=self.unpack(x);v=p[i+1]-p[i];sq=float(v@v)
                al=.5 if sq<1e-24 else min(1.,max(0.,float((self.centers[j]-p[i])@v)/sq))
                dl=p[i]+al*v-self.centers[j];dist=float(np.linalg.norm(dl))
                if dist>=self.radii[j]-tol:continue
                u=dl/max(dist,1e-15);w=(1-al)**2+al**2;step=(self.radii[j]-dist+1e-12)/w
                if i>0:x[2*(i-1):2*i]+=(1-al)*step*u
                if i+1<H-1:x[2*i:2*i+2]+=al*step*u
        return x,self.residual(x)

    def subset(self,keep):
        keep=np.asarray(keep,dtype=bool)
        return CircleSegmentConstraints(self.endpoints,self.num_points,self.centers[keep],self.radii[keep])

    def _solve(self,target,budget,maxiter,tol,start=None):
        return self._sqp(target,start if start is not None else np.clip(target,-.9999999,.9999999),budget,maxiter,tol)

    def project(self,target,budget,maxiter=40,tol=1e-7,start=None,allow_partial=False,restarts=2):
        """PC(z)=argmin||x-z|| s.t. whole-polyline segment constraints (PDM eq.7).

        Identity when already feasible (App. C.4: a feasible point is returned
        unchanged). Otherwise: least-norm SQP; if the merit stalls short of the
        tolerance, exact row repair restores feasibility and the SQP is restarted
        from that feasible point, keeping the lowest-objective feasible candidate.
        The returned point is verified feasible for the *full* circle set.
        Nearest-ness is local: for a grossly infeasible target the non-convex
        projection is initialisation dependent and is never certified global.

        `allow_partial` returns the best iterate instead of raising when the
        residual cannot be closed. Used only inside the PDM inner loop, where the
        paper's own claim is that violations merely *decrease* as t -> 0; the
        final (lowest-noise) level is always required to be exact.
        """
        target=np.asarray(target,dtype=float)
        if target.shape!=(self.D,):raise ValueError('point-state dimension differs')
        if self.D==0:
            if self.values(target).max(initial=-np.inf)>tol:raise SolverFailure('fixed path intersects obstacle')
            return target.copy(),{'status':'identity_fixed_path'}
        budget.check();full=self.residual(target)
        if full<=tol:return target.copy(),{'status':'identity_feasible','residual':full,'iterations':0,'global_nearest_certified':False}
        if allow_partial:restarts=min(restarts,0)
        x=np.clip(target if start is None else np.asarray(start,dtype=float),-.9999999,.9999999)
        best=None;best_obj=np.inf;best_info=None;reason=None;partial=None
        for attempt in range(restarts+1):
            budget.check();info={}
            try:
                xx,info=self._solve(target,budget,maxiter,tol,start=x)
            except SolverFailure as exc:
                reason=str(exc)[:160];xx=x
            cand=xx;left=self.residual(cand)
            if left>tol:
                cand,left=self.finish_feasible(cand,budget,tol)
            if left>tol:
                partial=cand if partial is None else partial
                break
            obj=float(.5*np.sum((cand-target)**2))
            if obj<best_obj-1e-12:
                best,best_obj,best_info=cand,obj,{'status':info.get('status','segment_sqp_feasible' if not reason else 'segment_sqp_row_repair'),'iterations':info.get('iterations',maxiter),'reason':reason}
                x=cand
            else:
                x=cand;break
        if best is None:
            if not allow_partial:raise SolverFailure(f'segment_projection_failed:full_residual={self.residual(partial if partial is not None else x):.6g}:circles={len(self.radii)}:{reason}')
            xx=partial if partial is not None else x
            return xx,{'status':'segment_projection_partial','reason':reason,'residual':self.residual(xx),
                'iterations':maxiter,'circles':int(len(self.radii)),'global_nearest_certified':False}
        return best,{**best_info,'residual':self.residual(best),
            'initialiser':'supplied' if start is not None else 'box_clip',
            'circles':int(len(self.radii)),'global_nearest_certified':False}
