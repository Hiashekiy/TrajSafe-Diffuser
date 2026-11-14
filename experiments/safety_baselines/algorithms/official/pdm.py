"""PDM (Projected Diffusion Models) — vendored from the official release.

Source: `.refs/Projected-Diffusion/examples/trajectories/mpd/models/diffusion_models/scoremodel.py`

    step size      self.step_size = eps * (process / process[-1])^2        509-510
    Langevin step  x = x + 0.5*step_size*score(x,idx) + sqrt(step_size)*z   522-531
    pgd step       for step_num in range(annealed_step):                   557-627
                       x = one_langevin_step(x, idx)                       573-576
                       x, violation = projection.apply(x)                  579
                       if idx > 5:                                         581-594
                           while violation:                                586-591
                               for _ in range(15): x = langevin(x)
                               x, violation = projection.apply(x)

Two things in that listing are *not* in the paper text and are easy to miss, so they are
reproduced here explicitly because they are what the release actually does:

* `projection.apply` runs its solver only when `projection.traj_p` is true, and
  `pgd_diffusion_step` sets that flag only for `idx > 5` (`581-583`); below that index the
  same call only re-clamps the fixed endpoints (`projection.py:49-76`, `86-90`). So the
  expensive projection is applied to the **low-noise half** of the ladder only.
* when the solver reports failure the release does not accept the unprojected point: it
  takes 15 further Langevin steps and projects again, in a `while violation:` with no
  iteration cap (`586-591`). We cap it (`max_retries`) so a hopeless scene fails loudly
  instead of hanging, which is the only added guard.

The projection operator itself is the task's (`projection.py:14-78` is our nearest-point
solve of the same set, `argmin ||x-gt||^2 s.t. g(x) <= 0`).
"""
import numpy as np

from experiments.safety_baselines.algorithms.solvers import SolverFailure


def step_size(eps, process):
    """`eps * (sigma_t / sigma_min)^2` (official 509-510); the paper's gamma is delta/2."""
    process = np.asarray(process, dtype=float)
    return float(eps) * (process / process[-1]) ** 2


def langevin(x, score, delta, rng):
    """One annealing iteration of the release (522-531)."""
    return x + 0.5*delta*score(x) + np.sqrt(delta)*rng.normal(size=np.shape(x))


def project_level(level_index, x, score, delta, rng, project, annealed_step,
                  project_here, max_retries=50, retry_steps=15, on_step=None):
    """`pgd_diffusion_step` for one noise level. Returns (x, info)."""
    calls = retries = 0
    violation = False
    info = {}
    for _ in range(annealed_step):
        x = langevin(x, score, delta, rng)
        if on_step is not None: on_step('langevin')
        if not project_here:
            continue
        x, info = project(x)
        calls += 1
        if on_step is not None: on_step('project')
        violation = info.get('violation', False) or 'partial' in str(info.get('status', ''))
        while violation:                                   # official 586-591
            if retries >= max_retries:
                raise SolverFailure('pdm_projection_retry_exhausted:level=%d:retries=%d'
                                    % (level_index, retries))
            retries += 1
            for _ in range(retry_steps):
                x = langevin(x, score, delta, rng)
                if on_step is not None: on_step('retry_langevin')
            x, info = project(x)
            calls += 1
            if on_step is not None: on_step('project')
            violation = info.get('violation', False) or 'partial' in str(info.get('status', ''))
    return x, {'projection_calls': calls, 'retries': retries, 'projected': project_here,
               'status': info.get('status') if project_here else 'endpoints_only',
               'residual': info.get('residual') if project_here else None}


def project_index_from(levels, fraction=0.6):
    """The `idx > 5` threshold of a 10-level ladder, kept as a fraction of the ladder."""
    return int(round(float(fraction) * len(levels)))
