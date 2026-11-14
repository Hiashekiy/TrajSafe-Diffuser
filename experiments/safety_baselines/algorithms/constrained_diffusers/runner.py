import numpy as np
from experiments.safety_baselines.algorithms.kernels import primal_dual_step,alm_step
from experiments.safety_baselines.algorithms.solvers import nearest_projection

def run(task,method,rng,cfg,budget,trace):
    """Constrained Diffusers reverse Langevin (eq. 10/13/16) on this state space.

    One diffusion level t contributes `cd_substeps` Euler-Maruyama steps of the
    *same* Langevin SDE over unit level time: dt=1/K gives step beta_t/K for the
    score term and sqrt(beta_t/K) for the noise. dt=1 (K=1) is the paper's
    single-step form; K>1 is only a discretisation refinement and does not
    change the stationary distribution p_t the loop targets.
    """
    particles=cfg['particles'] if method in ('Primal-Dual','ALM') else 1
    substeps=max(1,int(cfg.get('cd_substeps',1)))
    x=task.initial(rng,particles) if hasattr(task,'initial') else rng.normal(size=(particles,task.D))
    lam=np.zeros(len(task.constraints(x[0])[0]));rho=cfg['rho'];warm=None
    levels=np.linspace(len(task.ab)-1,0,cfg.get('cd_levels',len(task.ab))).round().astype(int).tolist()+[0]*cfg.get('cd_extra_steps',0)
    for t in levels:
        budget.check();beta=task.betas[t]/substeps;noise_scale=np.sqrt(task.betas[t]/substeps)
        last_info={}
        for _ in range(substeps):
            budget.check();score=task.score(x,t)
            if method=='Projected':
                target=(x+beta/2*score+noise_scale*rng.normal(size=x.shape))[0]
                task.last_nominal=target.copy()
                if warm is None and not hasattr(task,'project') and hasattr(task,'projection_start_for'):warm=task.projection_start_for(target,budget)
                elif warm is None and not hasattr(task,'project') and hasattr(task,'projection_initial'):warm=task.projection_initial(budget)
                xx,info=task.project(target,budget) if hasattr(task,'project') else nearest_projection(target,lambda y:task.constraints(y)[0],lambda y:task.constraints(y)[1],warm,budget,cfg['projection_maxiter'],cfg['projection_tol_scene']);warm=xx.copy();x=xx[None];last_info=info
            else:
                task.last_nominal=(x+beta/2*score+noise_scale*rng.normal(size=x.shape))[0].copy()
                vals=[task.constraints(v) for v in x];g=np.stack([v[0] for v in vals]);J=np.stack([v[1] for v in vals])
                # Paper Remark1: E[[g]+]<=0 encodes sample safety. Hinge is a.e. differentiable.
                J=J*(g>0)[...,None];g=np.maximum(g,0)
                # Dimensionless grid-cell violation: same feasible set, explicit
                # parameter units; native ellipse constraints are dimensionless too.
                scale=cfg.get('dual_constraint_scale',1.0);J=J*scale;g=g*scale
                noise=rng.normal(size=x.shape)
                if method=='Primal-Dual':x,lam=primal_dual_step(x,score,g,J,lam,beta,cfg['dual_eta'],noise);slack=None
                else:x,lam,slack,rho=alm_step(x,score,g,J,lam,rho,cfg['rho_growth'],beta,noise)
                last_info={'mean_positive_g':float(g.mean())}
        if not np.isfinite(x).all():raise RuntimeError('nonfinite_langevin')
        trace.append({'t':t,'optimizer':method,'lambda_min':float(lam.min()),'lambda_max':float(lam.max()),
            'rho_next':rho,'slack_norm':None if method in ('Projected','Primal-Dual') else float(np.linalg.norm(slack)),
            'particles':particles,'substeps':substeps,'beta_step':float(beta),'noise_std':float(noise_scale),
            **{k:v for k,v in last_info.items() if k!='global_nearest_certified'}})
    # Fixed particle0, no GT/feasibility/quality-based selection.
    return x[0]
