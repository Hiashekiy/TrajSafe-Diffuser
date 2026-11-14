"""Independent local nearest-point projection and convex CBF QP solvers."""
import time
import numpy as np
from scipy import sparse
from scipy.optimize import minimize,nnls,linprog
import osqp

class SolverFailure(RuntimeError):pass
class Budget:
    def __init__(self,seconds):self.end=time.perf_counter()+seconds
    def check(self):
        if time.perf_counter()>self.end:raise SolverFailure('timeout')
    def left(self):self.check();return max(.001,self.end-time.perf_counter())

def call_project(task,target,budget,warm=None,allow_partial=None):
    """task.project(target, budget, start=warm[, allow_partial=...]) with signature probing.

    Task adapters written before 2026-11-14 do not accept `start`; the warm start is an
    optional acceleration, so it is only passed when the adapter declares it (or takes
    **kwargs). Nothing is swallowed: only the keyword is conditionally omitted.
    """
    import inspect
    kw={}
    try:params=inspect.signature(task.project).parameters
    except (TypeError,ValueError):params={}
    var_kw=any(p.kind==p.VAR_KEYWORD for p in params.values())
    if warm is not None and ('start' in params or var_kw):kw['start']=warm
    if allow_partial is not None and ('allow_partial' in params or var_kw):kw['allow_partial']=allow_partial
    return task.project(target,budget,**kw)

def nearest_projection(target,fun,jac,warm,budget,maxiter=60,tol=1e-5,kkt_jac=None):
    """Local SLSQP solve of min .5||x-target||^2 s.t. g(x)<=0.
    Nonconvex nearest solution is local, not a global projection certificate.
    """
    budget.check();target=np.asarray(target);g=fun(target)
    if np.max(g)<=tol:return target.copy(),{'status':'identity','iterations':0,'residual':float(np.max(g)),'stationarity':0.}
    start=target if warm is None else np.asarray(warm)
    # A known feasible start gives an objective upper bound. This redundant
    # box cannot exclude any global nearest point and prevents SQP overflow.
    bounds=None
    if warm is not None and np.max(fun(start))<=tol:
        radius=float(np.linalg.norm(start-target))+1e-3
        bounds=list(zip(target-radius,target+radius))
    def callback(_):budget.check()
    scale=max(1.,float(np.linalg.norm(target)**2))
    result=minimize(lambda x:.5*np.sum((x-target)**2)/scale,start,jac=lambda x:(x-target)/scale,
        constraints=[{'type':'ineq','fun':lambda x:-fun(x),'jac':lambda x:-jac(x)}],
        method='SLSQP',bounds=bounds,callback=callback,options={'maxiter':maxiter,'ftol':1e-12})
    verify_jac=kkt_jac or jac
    budget.check();g=fun(result.x);J=verify_jac(result.x);active=g>=-max(tol*10,1e-4)
    stationarity=result.x-target
    if active.any():
        try:mult,_=nnls(J[active].T,-stationarity,maxiter=5000);stationarity=stationarity+J[active].T@mult
        except RuntimeError:raise SolverFailure('projection_kkt_nnls_failed')
    residual=float(np.max(g));stat=float(np.linalg.norm(stationarity,np.inf));stationarity_kind='ordinary'
    if stat>1e-3 and active.any() and residual<=tol:
        # Bilinear SDF: approximate generalized normal cone at grid boundaries.
        # No change to constraint values, objective or feasibility tolerance.
        rng=np.random.default_rng(0);blocks=[J[active]]
        for _ in range(20):
            budget.check();blocks.append(verify_jac(result.x+rng.normal(size=len(target))*1e-7)[active])
        A=np.concatenate(blocks,axis=0).T
        try:
            mult,_=nnls(A,-(result.x-target),maxiter=10000)
            stat=float(np.linalg.norm(result.x-target+A@mult,np.inf));stationarity_kind='sampled_clarke'
        except RuntimeError:pass
    if (not result.success and result.status!=9) or residual>tol or stat>1e-3 or not np.isfinite(result.x).all():
        raise SolverFailure(f'projection_failed:{result.status}:residual={residual:.4g}:kkt={stat:.4g}')
    return result.x,{'status':'local_kkt' if result.success else 'independently_verified_local_kkt_maxiter','solver_status':int(result.status),'iterations':int(result.nit),'residual':residual,'stationarity':stat,'stationarity_kind':stationarity_kind}

def cbf_velocity(nominal,J,offset,w,budget,tol=1e-5):
    budget.check();v=np.asarray(nominal);n=len(v);m=len(offset)
    if w>0:
        # Eliminate the unconstrained relaxation exactly. Given velocity z,
        # r=min((Jz+b)/w,0) is the unique minimum-norm feasible relaxation.
        # This avoids the badly conditioned [-w I] block at paper w=100.
        def objective(z):
            budget.check();negative=np.minimum(J@z+offset,0.)
            return .5*np.sum((z-v)**2)+.5*np.sum(negative**2)/(w*w),z-v+J.T@negative/(w*w)
        opt=minimize(objective,v,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-15,'gtol':tol*.01,'maxls':40})
        z=opt.x;negative=np.minimum(J@z+offset,0.);r=negative/w
        primal=float(np.min(J@z+offset-w*r));dual=float(np.linalg.norm(objective(z)[1],np.inf))
        if primal < -tol or dual>1e-3 or not np.isfinite(z).all():raise SolverFailure('qp_relaxed_residual_failed')
        return z,r,{'qp_status':'exact_relaxation_elimination','cbf_residual':primal,'kkt_dual':dual,'iterations':int(opt.nit)}
    relaxed=w>0
    A=sparse.csc_matrix(J)
    if relaxed:A=sparse.hstack([A,-w*sparse.eye(m)],format='csc')
    target=np.r_[v,np.zeros(m)] if relaxed else v
    # Row normalization preserves the feasible set and objective exactly.
    row_norm=np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).reshape(-1));fixed=row_norm<1e-14
    if np.any(fixed & (np.asarray(offset)<-tol)):raise SolverFailure('qp_fixed_condition_infeasible')
    scaling=1/np.maximum(row_norm,1e-8);scaled_A=sparse.diags(scaling)@A
    solver=osqp.OSQP();solver.setup(P=sparse.eye(len(target),format='csc'),q=-target,A=scaled_A,l=-offset*scaling,u=np.full(m,np.inf),verbose=False,eps_abs=tol*.1,eps_rel=tol*.1,max_iter=20000,time_limit=budget.left(),polishing=True)
    res=solver.solve(raise_error=False);budget.check()
    independently_verified=False
    if res.info.status_val in (2,7) and res.x is not None and res.y is not None:
        gap=A@res.x+offset;dual_variable=res.y*scaling
        primal_error=float(max(0.,-np.min(gap)))
        dual_error=float(np.linalg.norm(res.x-target+A.T@dual_variable,np.inf))
        sign_error=float(max(0.,np.max(dual_variable)))
        complementarity=float(np.max(np.abs(dual_variable*gap)))
        independently_verified=(primal_error<=tol and dual_error<=1e-3 and sign_error<=tol and complementarity<=1e-3)
    if (res.info.status_val!=1 and not independently_verified) or res.x is None:
        # Independent phase-I LP can distinguish an impossible coupled CBF
        # system from a mere OSQP convergence failure. Verify its Farkas dual.
        budget.check();use=~fixed;aa=scaled_A[use].toarray();bb=np.asarray(offset)[use]*scaling[use]
        lp=linprog(np.r_[np.zeros(len(target)),1.],A_ub=np.c_[-aa,-np.ones(len(aa))],b_ub=bb,
            bounds=[(None,None)]*len(target)+[(0,None)],method='highs',options={'time_limit':budget.left()})
        budget.check()
        if lp.success and lp.x[-1]>tol:
            multiplier=-lp.ineqlin.marginals
            normal=float(np.linalg.norm(aa.T@multiplier,np.inf));margin=float(-bb@multiplier)
            if normal<1e-6 and margin>tol and multiplier.min()>=-1e-10 and abs(multiplier.sum()-1)<1e-6:
                raise SolverFailure(f'qp_infeasible_certified:margin={margin:.6g}:dual_residual={normal:.3g}')
        raise SolverFailure('qp_'+res.info.status.replace(' ','_'))
    primal=float(np.min(A@res.x+offset));dual=float(np.linalg.norm(res.x-target+A.T@(res.y*scaling),np.inf))
    if primal < -tol or dual>1e-3:raise SolverFailure('qp_residual_failed')
    return res.x[:n],res.x[n:],{'qp_status':('independently_verified_'+res.info.status) if independently_verified else res.info.status,'cbf_residual':primal,'kkt_dual':dual,'iterations':res.info.iter}
