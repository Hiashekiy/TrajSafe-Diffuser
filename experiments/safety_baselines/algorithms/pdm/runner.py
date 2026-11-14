"""PDM (NeurIPS 2024) Algorithm 1 in the paper's own VE state y.

The paper defines the sampler on the *data-space* trajectory y (y = x0 + sigma_t z)
and projects it onto the task constraint set C, which lives in data coordinates:

    y <- Pi_C( y + gamma_t s_theta(y,t) + sqrt(2 gamma_t) z ),   gamma_t = sigma_t^2 / (2 sigma_T^2)

With a VP-trained model the same sample is written x = sqrt(alphabar_t) y, so in VP
coordinates the *projection set becomes time dependent* (sqrt(alphabar_t) C) and the
VE state has to be carried across levels by y = x / sqrt(alphabar_t). An earlier
version of this runner kept the state in VP coordinates but still projected onto the
unscaled C, which is neither the paper's VE algorithm nor an exact reparameterisation
of it; that variant has been removed instead of silently mixing the two. See
docs/DSH_FIX_REPORT_20261114.md §4d.
"""
import numpy as np
from experiments.safety_baselines.algorithms.solvers import nearest_projection,SolverFailure,call_project

def level_gamma(variance,t,levels,convention,ald_eps):
    """gamma_t of the paper's form  y += gamma*s + sqrt(2*gamma)*z  (VE units).

    paper        : Algorithm 1 line 3, gamma_t = sigma_t^2/(2 sigma_T^2), sigma_T the
                   largest noise level of the ladder.
    official_ald : the paper's reference implementation (RAISELab-atUVA/Projected-
                   Diffusion, projected_diffusion/scoremodel.py) uses
                       delta_t = eps*(sigma_t/sigma_min)^2
                       x += (delta_t/2) * score + sqrt(delta_t) * noise
                   so the gamma of the paper's form is gamma_t = delta_t/2. Using
                   gamma_t = delta_t doubles both the drift coefficient and the noise
                   variance; that factor-of-2 error was present until 2026-11-14.
    """
    if convention=='paper':
        return float(variance[t]/(2.*variance[levels[0]]))
    if convention=='official_ald':
        return float(ald_eps*variance[t]/(2.*np.min(variance)))
    raise ValueError('unknown pdm step convention: '+convention)

def run(task,method,rng,cfg,budget,trace):
    if str(cfg.get('pdm_parameterization','ve'))!='ve':
        raise ValueError('PDM must run in the paper VE state (pdm_parameterization="ve"); '
            'the VP-state variant needs the time-dependent projection set sqrt(alphabar_t)*C '
            'and is not implemented (see docs/DSH_FIX_REPORT_20261114.md §4d)')
    convention=cfg.get('pdm_step_convention','paper')
    ald_eps=float(cfg.get('pdm_ald_eps',2e-3))
    rng0=getattr(task,'initial_rng',rng)
    x=task.initial(rng0,ve=True)[0] if hasattr(task,'initial') else rng0.normal(size=task.D)*task.sigma[-1]
    levels=np.unique(np.linspace(0,len(task.ab)-1,min(cfg.get('pdm_levels',len(task.ab)),len(task.ab))).astype(int))[::-1]
    variance=task.sigma**2 if hasattr(task,'sigma') else 1-task.ab  # VE noise variance per level
    statuses=[];partial=0;warm=None
    for t in levels:
        t=int(t);gamma=level_gamma(variance,t,levels,convention,ald_eps)
        last=bool(t==levels[-1])
        for i in range(cfg['pdm_inner']):
            budget.check();score=task.score(x,t,ve=True)[0]
            target=x+gamma*score+np.sqrt(2.*gamma)*rng.normal(size=x.shape)
            task.last_nominal=target.copy()
            info={}
            if hasattr(task,'project'):
                # `start=` carries the previous level's feasible point: without it every
                # projection restarts from the drifted SGLD target, which is exactly the
                # regime a local SQP handles worst (2026-11-14 review).
                x,info=call_project(task,target,budget,warm,
                                    allow_partial=(not last) if getattr(task,'partial_projection',False) else None)
                warm=x.copy()
            else:
                if warm is None and hasattr(task,'projection_start_for'):warm=task.projection_start_for(target,budget)
                elif warm is None and hasattr(task,'projection_initial'):warm=task.projection_initial(budget)
                x,info=nearest_projection(target,lambda y:task.constraints(y)[0],lambda y:task.constraints(y)[1],warm,budget,cfg['projection_maxiter'],cfg['projection_tol_scene'])
                warm=x.copy()
            st=info.get('status','')
            statuses.append(st)
            if 'partial' in st:partial+=1
            score_norm=float(np.linalg.norm(score))
            trace.append({'t':t,'inner':i,'optimizer':'projection','gamma':float(gamma),
                'sigma':float(np.sqrt(variance[t])),'parameterization':'ve','step_convention':convention,
                # VE mixing rate is gamma / sigma_t^2 (NOT gamma/(1-alphabar_t), which is the
                # VP-normalised quantity and was reported by mistake before 2026-11-14).
                'mixing_ratio':float(gamma/variance[t]),
                'drift_norm':float(gamma*score_norm),'noise_std':float(np.sqrt(2.*gamma)),
                'score_norm':score_norm,'projection_status':st,'partial_so_far':partial,**info})
    tol=float(getattr(task,'projection_tol',cfg.get('projection_tol_scene',1e-7)))
    residual=float(task.constraints(x)[0].max(initial=-np.inf))
    # The delivered state must come from an *exact* projection: a partially improved
    # iterate is never reported as a feasible PDM sample.
    final_status=statuses[-1] if statuses else ''
    task.pdm_stats={'projection_calls':len(statuses),'partial_projections':partial,
        'final_status':final_status,'final_residual':residual,'tolerance':tol,
        'exact_final':('partial' not in final_status and residual<=tol)}
    if residual>tol:raise SolverFailure(f'pdm_final_level_infeasible:residual={residual:.6g}')
    return x
