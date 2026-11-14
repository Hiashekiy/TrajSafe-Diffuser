"""Constrained Diffusers (NeurIPS 2025) reverse Langevin: eq (10), (13)-(14), (16)-(18).

The paper writes one Langevin step per diffusion level t with that level's own beta:

    x_{t-1} = x_t + beta_t/2 * grad log p_t(x_t) + sqrt(beta_t) z      (eq 6)

Two implementation points that were wrong or undocumented before 2026-11-14:

* noise is drawn ONCE per update and used both for the recorded nominal and for the
  update (two independent draws made `trajectory_before` meaningless);
* the level ladder. eq (6)'s beta_t is the beta of *that* level, which is only
  correct when the ladder visits every trained level. `cd_beta="interval"` computes
  the aggregate beta of a skipped interval instead, beta_eff = 1 - alphabar_t /
  alphabar_next (the DDPM posterior over the interval), which is what a subsampled
  ladder needs. Both are implemented; the choice and the amount of subsampling are
  recorded per run so an accidentally subsampled ladder is visible in the results.
"""
import numpy as np
from experiments.safety_baselines.algorithms.kernels import primal_dual_step,alm_step
from experiments.safety_baselines.algorithms.solvers import nearest_projection

def level_ladder(num_levels,visits,extra_steps):
    ladder=np.linspace(num_levels-1,0,visits).round().astype(int).tolist()+[0]*extra_steps
    return ladder

def effective_beta(ab,t,t_next,beta_t,mode):
    """beta used for one Langevin step that covers the interval [t_next, t]."""
    if mode=='level':return float(beta_t)
    if mode=='interval':
        if t_next is None or t_next>=t:return float(beta_t)
        return float(1.-ab[t]/ab[t_next])
    raise ValueError('unknown cd_beta mode: '+mode)

def run(task,method,rng,cfg,budget,trace):
    """Constrained Diffusers reverse Langevin on this state space.

    `cd_substeps` K>1 splits each level into K Euler-Maruyama steps of the same
    Langevin SDE (dt=1/K: step beta/K for the score term and sqrt(beta/K) for the
    noise). It is a discretisation refinement and does not change the target p_t.
    """
    particles=cfg['particles'] if method in ('Primal-Dual','ALM') else 1
    substeps=max(1,int(cfg.get('cd_substeps',1)))
    beta_mode=cfg.get('cd_beta','level')
    rng0=getattr(task,'initial_rng',rng)
    x=task.initial(rng0,particles) if hasattr(task,'initial') else rng0.normal(size=(particles,task.D))
    lam=np.zeros(len(task.constraints(x[0])[0]));rho=cfg['rho'];warm=None
    ladder=level_ladder(len(task.ab),cfg.get('cd_levels',len(task.ab)),cfg.get('cd_extra_steps',0))
    ab=task.ab;subsampled=any(ladder[i]>ladder[i+1]+1 for i in range(len(ladder)-1))
    for idx,t in enumerate(ladder):
        budget.check()
        t_next=ladder[idx+1] if idx+1<len(ladder) else None
        beta_full=effective_beta(ab,t,t_next,task.betas[t],beta_mode)
        beta=beta_full/substeps;noise_scale=np.sqrt(beta_full/substeps)
        last_info={}
        for _ in range(substeps):
            budget.check();score=task.score(x,t)
            noise=rng.normal(size=x.shape)
            if method=='Projected':
                target=(x+beta/2*score+noise_scale*noise)[0]
                task.last_nominal=target.copy()
                if warm is None and not hasattr(task,'project') and hasattr(task,'projection_start_for'):warm=task.projection_start_for(target,budget)
                elif warm is None and not hasattr(task,'project') and hasattr(task,'projection_initial'):warm=task.projection_initial(budget)
                xx,info=task.project(target,budget) if hasattr(task,'project') else nearest_projection(target,lambda y:task.constraints(y)[0],lambda y:task.constraints(y)[1],warm,budget,cfg['projection_maxiter'],cfg['projection_tol_scene']);warm=xx.copy();x=xx[None];last_info=info
            else:
                task.last_nominal=(x+beta/2*score+noise_scale*noise)[0].copy()
                vals=[task.constraints(v) for v in x];g=np.stack([v[0] for v in vals]);J=np.stack([v[1] for v in vals])
                # Paper Remark1: E[[g]+]<=0 encodes sample safety. Hinge is a.e. differentiable.
                J=J*(g>0)[...,None];g=np.maximum(g,0)
                # Dimensionless grid-cell violation: same feasible set, explicit
                # parameter units. Under g'=s*g, J'=s*J the paper's updates stay
                # equivalent only with lambda'=lambda/s, eta'=eta/s^2, rho'=rho/s^2
                # (see test_dual_units_invariance and docs §4e).
                scale=cfg.get('dual_constraint_scale',1.0);J=J*scale;g=g*scale
                if method=='Primal-Dual':x,lam=primal_dual_step(x,score,g,J,lam,beta,cfg['dual_eta'],noise);slack=None
                else:x,lam,slack,rho=alm_step(x,score,g,J,lam,rho,cfg['rho_growth'],beta,noise)
                last_info={'mean_positive_g':float(g.mean())}
        if not np.isfinite(x).all():raise RuntimeError('nonfinite_langevin')
        trace.append({'t':t,'optimizer':method,'lambda_min':float(lam.min()),'lambda_max':float(lam.max()),
            'rho_next':rho,'slack_norm':None if method in ('Projected','Primal-Dual') else float(np.linalg.norm(slack)),
            'particles':particles,'substeps':substeps,'beta_step':float(beta),'beta_full':float(beta_full),
            'noise_std':float(noise_scale),'beta_mode':beta_mode,'ladder_subsampled':bool(subsampled),
            **{k:v for k,v in last_info.items() if k!='global_nearest_certified'}})
    # Fixed particle0, no GT/feasibility/quality-based selection.
    return x[0]
