import numpy as np
from experiments.safety_baselines.algorithms.solvers import nearest_projection,SolverFailure

def level_gamma(variance,t,levels,convention,ald_eps):
    """PDM step size, both documented conventions, in VE units.

    paper        : Algorithm 1 line 3, gamma_t = sigma_t^2/(2 sigma_T^2),
                   sigma_T the largest noise level of the ladder.
    official_ald : the paper's reference implementation (RAISELab-atUVA/
                   Projected-Diffusion, projected_diffusion/scoremodel.py) uses
                   step_size = eps*(sigma/sigma_min)^2 in its annealed Langevin
                   dynamics. The two agree only when sigma_T^2 = sigma_min^2/(2eps);
                   with this frozen 16-level schedule sigma_T^2/sigma_min^2 =
                   8.8e6, so the paper's form is numerically unmixable while the
                   official form mixes at rate eps*sigma_t^2/(sigma_min^2*sigma_t^2)
                   = eps/sigma_min^2 per step. Never mixed silently: the choice is
                   recorded per run and gated by the unconstrained sampler audit.
    """
    if convention=='paper':
        return float(variance[t]/(2.*variance[levels[0]]))
    if convention=='official_ald':
        return float(ald_eps*variance[t]/np.min(variance))
    raise ValueError('unknown pdm step convention: '+convention)

def run(task,method,rng,cfg,budget,trace):
    ve=cfg.get('pdm_parameterization','ve')=='ve'
    convention=cfg.get('pdm_step_convention','paper')
    ald_eps=float(cfg.get('pdm_ald_eps',2e-3))
    x=task.initial(rng,ve=ve)[0] if hasattr(task,'initial') else rng.normal(size=task.D)*(task.sigma[-1] if ve else 1.)
    levels=np.unique(np.linspace(0,len(task.ab)-1,min(cfg.get('pdm_levels',len(task.ab)),len(task.ab))).astype(int))[::-1]
    variance=task.sigma**2 if hasattr(task,'sigma') else 1-task.ab  # VE noise variance per level
    warm=None
    for t in levels:
        t=int(t)
        gamma=level_gamma(variance,t,levels,convention,ald_eps)
        last=bool(t==levels[-1])
        for i in range(cfg['pdm_inner']):
            budget.check();score=task.score(x,t,ve=ve)[0]
            # Change of variables y=x/sqrt(alpha_bar_t) maps the VE step to the VP
            # step s=gamma*alpha_bar_t exactly: both run the same SDE.
            s=gamma if ve else gamma*task.ab[t]
            target=x+s*score+np.sqrt(2.*s)*rng.normal(size=x.shape)
            task.last_nominal=target.copy()
            info={}
            if hasattr(task,'project'):
                if getattr(task,'partial_projection',False):
                    x,info=task.project(target,budget,allow_partial=not last)
                else:x,info=task.project(target,budget)
                warm=x.copy()
            else:
                if warm is None and hasattr(task,'projection_start_for'):warm=task.projection_start_for(target,budget)
                elif warm is None and hasattr(task,'projection_initial'):warm=task.projection_initial(budget)
                x,info=nearest_projection(target,lambda y:task.constraints(y)[0],lambda y:task.constraints(y)[1],warm,budget,cfg['projection_maxiter'],cfg['projection_tol_scene'])
                warm=x.copy()
            trace.append({'t':t,'inner':i,'optimizer':'projection','gamma':float(gamma),'step':float(s),
                'sigma':float(np.sqrt(variance[t])),'parameterization':'ve' if ve else 'vp',
                'step_convention':convention,'mixing_ratio':float(s/max(1-task.ab[t],1e-12)),**info})
    if getattr(task,'partial_projection',False):
        residual=float(task.constraints(x)[0].max(initial=-np.inf))
        if residual>float(getattr(task,'projection_tol',cfg.get('projection_tol_scene',1e-7))):
            raise SolverFailure(f'pdm_final_level_infeasible:residual={residual:.6g}')
    return x
