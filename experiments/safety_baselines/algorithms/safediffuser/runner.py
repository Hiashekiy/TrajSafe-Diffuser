"""SafeDiffuser (ICLR 2025) continuous-time CBF integrated inside the reverse chain.

Paper: eq (5)-(8) RoS, (9)-(10) ReS, (11)-(12) TVS, QP (13)/(14), theorems 3.4-3.6.
Neither the QP objective nor the barrier is changed here.

What this file adds on top of the transcription (task-book Phase E2/E3), and what
was measured on the way there:

* The paper's condition is a *velocity* condition  grad h . nu + eps (h - gamma) >= 0.
  It bounds only the first-order change, so the applied state can still penetrate.
* Per-micro-step backtracking of the step does NOT fix that here: the demanded
  velocity is dt-independent (J v >= gamma - h), so halving dt cannot repair a
  velocity deficit. Measured on 3 val cases: 532k halvings, 180 s timeout.
* Enforcing it whenever min(h) < -band freezes the trajectory (176 m penetration),
  because at high noise no first-order step can satisfy the certificate.
* Therefore the discrete certificate  h(x_{k+1}) - gamma >= exp(-eps dt)(h(x_k) - gamma)
  is evaluated ONCE PER DIFFUSION LEVEL (cheap: one extra clearance query) and, if it
  fails, the level is re-integrated with cbf_dt halved (bounded retries, never the QP).
  With `cbf_discrete_check=false` the runner is the pure velocity-condition version.

Every level records `discrete_residual_end`, `level_attempts`, `integration_steps` and
`min_h_scene`, so an unsafe delivery is visible instead of silent.
"""
import numpy as np
from experiments.safety_baselines.algorithms.solvers import cbf_velocity

def legal_gamma(h,horizon,epsilon,floor):
    """Thm 3.4: gamma >= max_k |gamma - b_k| e^{-eps N}, solved for gamma."""
    scale=np.exp(epsilon*horizon)
    return max(floor,-float(np.min(h))/(scale-1),float(np.max(h))/(scale+1))

def tvs_schedule(initial_h,j,N,dt):
    """sigma(j) = min(h0,0) j/N, sigma_dot = -min(h0,0)/(N dt)  (j decreases with tau)."""
    initial=np.minimum(initial_h,0.)
    return initial*j/N,-initial/(N*dt)

def _barrier_values(task,x):
    """h(x) only (no Jacobian): the signed clearance the barrier is defined on."""
    values=getattr(task,'values',None)
    if values is not None:return -np.asarray(values(x),dtype=float)
    return -np.asarray(task.constraints(x)[0],dtype=float)

def discrete_residual(h_next,h_now,gamma,epsilon,dt,scale=None):
    """min_k [h(x_next) - gamma - e^{-eps dt}(h(x) - gamma)]; >= 0 is the certificate."""
    r=h_next-gamma-np.exp(-epsilon*dt)*(h_now-gamma)
    sc=float(np.max(np.abs(h_now)))+abs(float(np.min(np.atleast_1d(gamma)))) if scale is None else scale
    return float(np.min(r)),float(sc)

def _integrate_level(task,method,cfg,x,nominal,h_start,h_initial,gamma,N,N0,hard_gamma,t,dt_level,eps,dt_full):
    """One level of the micro-step integrator; returns (x, trace-fields)."""
    desired=(nominal-x)/dt_full
    elapsed=0.;micro=0;worst_cbf=float('inf');max_slack=0.;info={};extra={}
    target_gamma=np.asarray([gamma]);r=np.zeros(1)
    while elapsed<dt_full-1e-12:
        g,jac=task.constraints(x);h=-g;J=-jac;w=0.
        if method=='TVS-diffuser':
            j=t+1-elapsed/dt_full
            sigma,sigma_dot=tvs_schedule(h_initial,j,N,dt_full)
            offset=-sigma_dot+eps*(h-sigma);target_gamma=sigma
            extra={'sigma_max':float(sigma.max()),'sigma_min':float(sigma.min()),'sigma_dot_max':float(sigma_dot.max())}
        else:
            if method=='ReS-diffuser':
                # Thm 3.5 uses gamma(N0, eps); the hard interval is t<=N0,
                # i.e. hard_levels = N0+1 levels (paper's ReS-diffuser-l10 -> 10).
                if t<=N0 and hard_gamma is None:hard_gamma=legal_gamma(h,N0*dt_full,eps,cfg['cbf_gamma_floor'])
                w=cfg['relax_weight'] if t>N0 else 0.
                used=gamma if hard_gamma is None else hard_gamma
            else:used=gamma
            offset=eps*(h-used);target_gamma=np.atleast_1d(used)
            extra={'gamma':float(np.min(np.atleast_1d(used))),'w':w,'hard_interval':bool(w==0),'hard_levels':int(N0+1)}
        solver=getattr(task,'cbf_solve',None) or cbf_velocity
        v,r,info=solver(desired,J,offset,w,cfg['_budget'],cfg['qp_tol'])
        step=min(dt_full-elapsed,dt_level)
        if hasattr(task,'max_point_speed') and cfg.get('cbf_max_point_step'):
            speed=task.max_point_speed(v)
            if speed>0:step=min(step,cfg['cbf_max_point_step']/speed)
        if step<1e-9:raise RuntimeError('cbf_integrator_step_underflow')
        x=x+step*v;elapsed+=step;micro+=1
        worst_cbf=min(worst_cbf,info['cbf_residual']);max_slack=max(max_slack,float(np.linalg.norm(r)))
        if micro>100000:raise RuntimeError('cbf_integrator_iteration_limit')
    return x,{'slack_norm':float(np.linalg.norm(r)),'max_slack_norm':max_slack,'integration_steps':micro,
        'minimum_cbf_residual':worst_cbf,'target_gamma':float(np.min(np.atleast_1d(target_gamma))),
        'min_h_scene':float(np.min(h)),**extra,
        **{k:v for k,v in info.items() if k in ('qp_status','cbf_residual','kkt_dual','iterations')}}

def run(task,method,rng,cfg,budget,trace):
    N=len(task.ab);rng0=getattr(task,'initial_rng',rng)
    x=task.initial(rng0)[0] if hasattr(task,'initial') else rng.normal(size=task.D)
    h_initial=-task.constraints(x)[0]
    eps=float(cfg['cbf_epsilon']);dt_full=float(cfg['cbf_dt'])
    gamma=legal_gamma(h_initial,N*dt_full,eps,cfg['cbf_gamma_floor'])
    N0=min(cfg['hard_steps']-1,N-1);hard_gamma=None
    discrete=bool(cfg.get('cbf_discrete_check',False))
    retries=int(cfg.get('cbf_level_retry',1)) if discrete else 0
    tol_rel=float(cfg.get('cbf_discrete_tol',1e-6))
    cfg['_budget']=budget
    for t in range(N-1,-1,-1):
        budget.check();clean=task.clean(x,t)[0]
        nominal=clean if t==0 else np.sqrt(task.ab[t-1])*clean+np.sqrt(1-task.ab[t-1])*(x-np.sqrt(task.ab[t])*clean)/np.sqrt(1-task.ab[t])
        if hasattr(task,'nominal'):nominal=task.nominal(x,t,clean)
        task.last_nominal=nominal.copy()
        h_start=_barrier_values(task,x)
        x0=x.copy();residual=None;attempts=0;fields=None
        for attempt in range(retries+1):
            attempts=attempt+1
            dt_level=dt_full/(2**attempt)
            x,fields=_integrate_level(task,method,cfg,x0.copy(),nominal,h_start,h_initial,gamma,N,N0,hard_gamma,t,dt_level,eps,dt_full)
            if not discrete:break
            h_end=_barrier_values(task,x)
            residual,scale=discrete_residual(h_end,h_start,np.atleast_1d(gamma),eps,dt_full)
            if residual>=-tol_rel*(1.0+scale):break
        trace.append({'t':t,'optimizer':'cbf_qp','discrete_check':discrete,'level_attempts':attempts,
            'discrete_residual_end':residual,'discrete_ok':(residual is None or residual>=0),**(fields or {})})
    return x
