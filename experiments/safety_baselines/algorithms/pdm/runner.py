"""PDM (NeurIPS 2024) — Algorithm 1 in the paper's VE state, as the official release runs it.

The loop is `algorithms/official/pdm.py`, vendored from
`.refs/Projected-Diffusion/examples/trajectories/mpd/models/diffusion_models/scoremodel.py:557-627`:

    for step in range(annealed_step):
        x = x + 0.5*delta*score(x,idx) + sqrt(delta)*z        delta = eps*(sigma/sigma_min)^2
        x = project(x)                                        only when idx is in the low-noise half
        while the projection failed: 15 more Langevin steps, project again

Deleted here (ours, not the release): the warm start that carried the previous level's
feasible point into every projection, the `allow_partial` mode that returned a
non-feasible iterate on non-final levels, and the high-noise residual gate. The release
starts every projection from the drifted target with `x0 = gt` (`solver.py:40,60`) and
retries instead of relaxing. See docs/official_vs_ours_diff.md §B.

Model and scene are ours: `task.score` is our base model in VE units and `task.project` is
our nearest-point map onto the occupancy set.
"""
import numpy as np

from experiments.safety_baselines.algorithms.official import pdm as official
from experiments.safety_baselines.algorithms.solvers import (
    SolverFailure, call_project, nearest_projection)


def level_gamma(variance, t, levels, convention, ald_eps):
    """gamma of the paper's form  y += gamma*s + sqrt(2*gamma)*z  (VE units).

    paper        : Algorithm 1 line 3, gamma_t = sigma_t^2/(2 sigma_T^2).
    official_ald : delta_t = eps*(sigma_t/sigma_min)^2, gamma_t = delta_t/2.
    """
    if convention == 'paper':
        return float(variance[t]/(2.*variance[levels[0]]))
    if convention == 'official_ald':
        return float(ald_eps*variance[t]/(2.*np.min(variance)))
    raise ValueError('unknown pdm step convention: '+convention)


def run(task, method, rng, cfg, budget, trace):
    if str(cfg.get('pdm_parameterization', 've')) != 've':
        raise ValueError('PDM must run in the paper VE state (pdm_parameterization="ve"); '
                         'the VP-state variant needs the time-dependent projection set '
                         'sqrt(alphabar_t)*C and is not implemented')
    convention = cfg.get('pdm_step_convention', 'paper')
    ald_eps = float(cfg.get('pdm_ald_eps', 2e-3))
    inner = int(cfg['pdm_inner'])
    variance = task.sigma**2 if hasattr(task, 'sigma') else 1-task.ab
    levels = np.unique(np.linspace(0, len(task.ab)-1,
                                   min(cfg.get('pdm_levels', len(task.ab)), len(task.ab))).astype(int))[::-1]
    project_from = int(cfg.get('pdm_project_from_index', official.project_index_from(levels)))

    def project(point):
        """`Projection.apply` (:49-76): nearest point of the constraint set, or a violation flag."""
        try:
            if hasattr(task, 'project'):
                value, info = call_project(task, point, budget)
            else:
                value, info = nearest_projection(
                    point, lambda y: task.constraints(y)[0], lambda y: task.constraints(y)[1],
                    None, budget, cfg['projection_maxiter'], cfg['projection_tol_scene'])
            return value, dict(info, violation=bool(info.get('violation', False)))
        except SolverFailure as exc:
            return point, {'status': 'solver_failed', 'residual': None, 'violation': True,
                           'reason': str(exc)[:120]}

    x = np.asarray(task.initial(rng, ve=True)[0], dtype=float)
    stats = []
    for idx, t in enumerate(levels):
        t = int(t)
        delta = 2.0*level_gamma(variance, t, levels, convention, ald_eps)
        project_here = idx >= project_from

        def score(point, _t=t):
            return np.asarray(task.score(point[None], _t, ve=True)[0], dtype=float)

        def on_step(kind, _t=t, _idx=idx, _delta=delta, _here=project_here):
            if kind == 'project':
                return
            trace.append({'t': _t, 'level_index': _idx, 'optimizer': 'official_pgd', 'kind': kind,
                          'gamma': float(_delta/2.), 'sigma': float(np.sqrt(variance[_t])),
                          'step_size': float(_delta), 'noise_std': float(np.sqrt(_delta)),
                          'mixing_ratio': float((_delta/2.)/variance[_t]),
                          'projected': _here})
            budget.check()

        x, info = official.project_level(
            idx, x, score, delta, rng, project, inner, project_here,
            max_retries=int(cfg.get('pdm_max_retries', 50)), on_step=on_step)
        stats.append(info)
        task.last_nominal = x.copy()

    projected = [s for s in stats if s['projected']]
    residual = float(np.max(task.constraints(x)[0]))
    tolerance = float(getattr(task, 'projection_tol', cfg.get('projection_tol_scene', 1e-7)))
    task.pdm_stats = {'projection_calls': int(sum(s['projection_calls'] for s in stats)),
                      'partial_projections': int(sum(s['retries'] for s in stats)),
                      'projected_levels': len(projected),
                      'project_from_index': project_from,
                      'final_status': projected[-1]['status'] if projected else 'endpoints_only',
                      'final_residual': residual, 'tolerance': tolerance,
                      'exact_final': bool(residual <= tolerance)}
    if residual > tolerance:
        raise SolverFailure(f'pdm_final_level_infeasible:residual={residual:.6g}')
    return x
