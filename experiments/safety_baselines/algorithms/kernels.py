"""Paper-equation kernels; task runners must supply audited score/constraints.
No dependency on Ours safety optimizers. g <= 0; particle axis is axis 0.
"""
import numpy as np
from scipy.optimize import minimize


def projected_step(x, score, beta, noise, project):
    if beta <= 0: raise ValueError('positive beta required')
    return project(x + beta/2 * score + np.sqrt(beta)*noise)


def primal_dual_step(x, score, g, jac, lam, beta, eta, noise):
    # jac [particles, constraints, dimensions], g [particles, constraints]
    if beta <= 0 or eta <= 0: raise ValueError('positive step sizes required')
    correction = np.einsum('k,bkd->bd', lam, jac)
    return (x + beta/2*(score-correction) + np.sqrt(beta)*noise,
            np.maximum(0, lam + eta*np.mean(g, axis=0)))


def alm_step(x, score, g, jac, lam, rho, growth, beta, noise):
    if rho <= 0 or growth <= 1 or beta <= 0: raise ValueError('invalid ALM schedule')
    mean = np.mean(g, axis=0)
    slack = np.maximum(0, -mean-lam/rho)
    residual = mean + slack
    coeff = lam + rho*residual
    updated = x + beta/2*(score-np.einsum('k,bkd->bd', coeff,jac)) + np.sqrt(beta)*noise
    return updated, lam+rho*residual, slack, rho*growth


def cbf_qp(nominal, jac, offset, relaxation_weight=None, tol=1e-8):
    """min ||nu-v||² (+||r||²), jac@nu+offset-w*r>=0.
    offset=eps*(h-gamma) for RoS/ReS; -sigma_dot+alpha(h-sigma) for TVS.
    Caller validates gamma and terminal time conditions before calling.
    """
    v=np.asarray(nominal,dtype=float).reshape(-1)
    J=np.asarray(jac,dtype=float); b=np.asarray(offset,dtype=float)
    if J.shape != (b.size,v.size): raise ValueError('QP shape mismatch')
    if relaxation_weight is None:
        A=J; target=v
    else:
        w=np.broadcast_to(relaxation_weight,b.shape)
        if np.any(w<0): raise ValueError('negative relaxation weight')
        A=np.concatenate([J,-np.diag(w)],axis=1)
        target=np.r_[v,np.zeros(b.size)]
    result=minimize(lambda z: .5*np.sum((z-target)**2),target,
        jac=lambda z:z-target,method='SLSQP',
        constraints=[{'type':'ineq','fun':lambda z:A@z+b,'jac':lambda z:A}],
        options={'maxiter':200,'ftol':tol})
    residual=float(np.min(A@result.x+b))
    if not result.success or residual < -tol or not np.all(np.isfinite(result.x)):
        raise RuntimeError('qp_failed: '+str(result.message))
    return result.x[:v.size],result.x[v.size:],{'residual':residual,'iterations':result.nit}


def pdm(x, sigmas, inner_steps, score, project, rng):
    sigmas=np.asarray(sigmas,dtype=float)
    if inner_steps<1 or np.any(sigmas<=0) or np.any(np.diff(sigmas)>0):
        raise ValueError('positive decreasing sigmas and M>=1 required')
    trace=[]
    for t,sigma in enumerate(sigmas):
        gamma=sigma*sigma/(2*sigmas[0]*sigmas[0])
        for i in range(inner_steps):
            x=project(x+gamma*score(x,t)+np.sqrt(2*gamma)*rng.normal(size=x.shape))
            trace.append({'level':t,'inner':i,'gamma':float(gamma)})
    return x,trace
