import numpy as np
from experiments.safety_baselines.algorithms.solvers import cbf_velocity

def legal_gamma(h,horizon,epsilon,floor):
    scale=np.exp(epsilon*horizon)
    return max(floor,-float(np.min(h))/(scale-1),float(np.max(h))/(scale+1))

def tvs_schedule(initial_h,j,N,dt):
    initial=np.minimum(initial_h,0.)
    return initial*j/N,-initial/(N*dt)

def run(task,method,rng,cfg,budget,trace):
    N=len(task.ab);x=task.initial(rng)[0] if hasattr(task,'initial') else rng.normal(size=task.D);h_initial=-task.constraints(x)[0]
    gamma=legal_gamma(h_initial,N*cfg['cbf_dt'],cfg['cbf_epsilon'],cfg['cbf_gamma_floor'])
    N0=min(cfg['hard_steps']-1,N-1);hard_gamma=None
    for t in range(N-1,-1,-1):
        budget.check();clean=task.clean(x,t)[0]
        nominal=clean if t==0 else np.sqrt(task.ab[t-1])*clean+np.sqrt(1-task.ab[t-1])*(x-np.sqrt(task.ab[t])*clean)/np.sqrt(1-task.ab[t])
        if hasattr(task,'nominal'):nominal=task.nominal(x,t,clean)
        task.last_nominal=nominal.copy();desired=(nominal-x)/cfg['cbf_dt']
        elapsed=0.;micro=0;worst_cbf=float('inf');max_slack=0.;info={};extra={}
        while elapsed<cfg['cbf_dt']-1e-12:
            budget.check();g,jac=task.constraints(x);h=-g;J=-jac;w=0.
            if method=='TVS-diffuser':
                j=t+1-elapsed/cfg['cbf_dt']
                sigma,sigma_dot=tvs_schedule(h_initial,j,N,cfg['cbf_dt'])
                offset=-sigma_dot+cfg['cbf_epsilon']*(h-sigma)
                extra={'sigma_max':float(sigma.max()),'sigma_min':float(sigma.min()),'sigma_dot_max':float(sigma_dot.max())}
            else:
                if method=='ReS-diffuser':
                    if t<=N0 and hard_gamma is None:hard_gamma=legal_gamma(h,(N0+1)*cfg['cbf_dt'],cfg['cbf_epsilon'],cfg['cbf_gamma_floor'])
                    w=cfg['relax_weight'] if t>N0 else 0.
                    used=gamma if hard_gamma is None else hard_gamma
                else:used=gamma
                offset=cfg['cbf_epsilon']*(h-used);extra={'gamma':used,'w':w,'hard_interval':bool(w==0)}
            solver=getattr(task,'cbf_solve',None) or cbf_velocity
            v,r,info=solver(desired,J,offset,w,budget,cfg['qp_tol'])
            step=min(cfg['cbf_dt']-elapsed,cfg['cbf_dt']/cfg.get('cbf_substeps',1))
            if hasattr(task,'max_point_speed') and cfg.get('cbf_max_point_step'):
                speed=task.max_point_speed(v)
                if speed>0:step=min(step,cfg['cbf_max_point_step']/speed)
            if step<1e-9:raise RuntimeError('cbf_integrator_step_underflow')
            x=x+step*v;elapsed+=step;micro+=1
            worst_cbf=min(worst_cbf,info['cbf_residual']);max_slack=max(max_slack,float(np.linalg.norm(r)))
            if micro>100000:raise RuntimeError('cbf_integrator_iteration_limit')
        trace.append({'t':t,'optimizer':'cbf_qp','slack_norm':float(np.linalg.norm(r)),'max_slack_norm':max_slack,'integration_steps':micro,'minimum_cbf_residual':worst_cbf,**extra,**info})
    return x
