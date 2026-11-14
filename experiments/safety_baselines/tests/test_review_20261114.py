"""Regression tests for the 2026-11-14 external review findings.

Each test pins one issue that was either a provable implementation error or a
protocol/fairness defect, so a future change cannot silently reintroduce it:

1  PDM official-ALD step was gamma = delta_t instead of delta_t/2 (drift and
   noise variance both doubled).
2  PDM kept the state in VP coordinates while projecting onto the unscaled
   constraint set C; the VE-only state is now enforced.
3  A partially improved projection could in principle be reported as a feasible
   PDM sample; pdm_stats + the final-level gate now prevent that.
4  Constrained Diffusers used a level's own beta even when the ladder skipped
   trained levels; `cd_beta` now has an explicit interval mode and the amount of
   subsampling is recorded.
5  PD/ALM drew the update noise twice, so the recorded "before the correction"
   trajectory was not the trajectory that was corrected.
6  dual_eta / rho in the configs were not the s^2-equivalent of the paper values
   even though test_dual_units_invariance already encoded the correct relation.
7  The circle cover only tested cell centres (diagonally adjacent cells could
   leave a corner uncovered).
8  A segment whose closest point coincides with a circle centre had a zero
   Jacobian (no escape direction).
9  ReS used a gamma horizon of N0+1 instead of the paper's N0.
10 Initial noise was method-dependent (derived_seed included the method name),
   so the seven methods were not compared on paired samples.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.safety_baselines.adapters.native_points import NativePointTask
from experiments.safety_baselines.adapters.paper_segment_constraints import CircleSegmentConstraints, obstacle_circles
from experiments.safety_baselines.algorithms.constrained_diffusers.runner import run as cd_run
from experiments.safety_baselines.algorithms.pdm.runner import level_gamma, run as pdm_run
from experiments.safety_baselines.algorithms.runner import initial_noise, initial_seed
from experiments.safety_baselines.algorithms.pdm.runner import run as pdm_run
from experiments.safety_baselines.algorithms.solvers import Budget, SolverFailure
from src.diffusion.schedule import NoiseSchedule

CFG_DIR = Path(__file__).resolve().parents[1] / 'configs'


# 1 ---------------------------------------------------------------------------
def test_official_ald_gamma_is_half_of_delta():
    """official: delta=eps*(sigma/sigma_min)^2, x += (delta/2)s + sqrt(delta)z."""
    var = np.array([0.012, 0.04, 0.3, 2., 1490.])
    levels = np.array([4, 3, 2, 1, 0])
    for t in levels:
        eps = 2e-3
        delta = eps * var[t] / var.min()
        gamma = level_gamma(var, int(t), levels, 'official_ald', eps)
        np.testing.assert_allclose(gamma, delta / 2, rtol=1e-12)
        # the invariance test of the paper form: drift gamma*s and noise var 2*gamma
        np.testing.assert_allclose(gamma * 2, delta, rtol=1e-12)


# 2 ---------------------------------------------------------------------------
def test_pdm_rejects_a_vp_state():
    """The VP-state variant must fail loudly instead of projecting a scaled state."""
    class Toy:
        D = 1
        ab = np.array([.9, .5])
        betas = np.array([.1, .4])
        sigma = np.sqrt((1 - ab) / ab)
        def initial(self, rng, particles=1, ve=False): return rng.normal(size=(particles, self.D))
        def score(self, x, t, ve=False): return -np.atleast_2d(x)
        def constraints(self, x): return np.array([0.]), np.array([[1.]])
        def project(self, target, budget): return np.asarray(target, float), {'status': 'identity'}
    cfg = {'pdm_inner': 1, 'pdm_levels': 2, 'pdm_parameterization': 'vp',
           'pdm_step_convention': 'official_ald', 'pdm_ald_eps': 1e-3, 'projection_tol_scene': 1e-9}
    with pytest.raises(ValueError, match='VE state'):
        pdm_run(Toy(), 'PDM', np.random.default_rng(0), cfg, Budget(5), [])


# 3 ---------------------------------------------------------------------------
def test_pdm_projects_only_in_the_low_noise_half_and_retries_on_violation():
    """Official `pgd_diffusion_step` (:557-627): the solver runs for `idx > 5` only, and a
    failed projection is retried with 15 more Langevin steps instead of being accepted."""
    class Toy:
        D = 1
        ab = np.array([.9, .5])
        betas = np.array([.1, .4])
        sigma = np.sqrt((1 - ab) / ab)
        projection_tol = 1e-6
        def __init__(self):
            self.calls = 0; self.fail_first = False
        def initial(self, rng, particles=1, ve=False): return rng.normal(size=(particles, self.D))
        def score(self, x, t, ve=False): return -np.atleast_2d(x)
        def constraints(self, x): return np.array([-abs(float(np.asarray(x).reshape(-1)[0]))]), np.array([[-1.]])
        def project(self, target, budget):
            self.calls += 1
            z = np.asarray(target, float).copy(); z[...] = 0.
            if self.fail_first and self.calls == 1:
                return z, {'status': 'solver_failed', 'residual': None, 'violation': True}
            return z, {'status': 'segment_sqp_feasible', 'residual': 0.}
    cfg = {'pdm_inner': 2, 'pdm_levels': 2, 'pdm_parameterization': 've',
           'pdm_step_convention': 'official_ald', 'pdm_ald_eps': 1e-3, 'projection_tol_scene': 1e-6}
    task = Toy(); trace = []
    pdm_run(task, 'PDM', np.random.default_rng(0), cfg, Budget(5), trace)
    stats = task.pdm_stats
    assert {'projection_calls', 'partial_projections', 'final_status', 'final_residual',
            'exact_final', 'project_from_index'} <= set(stats)
    # two levels -> threshold round(0.6*2) = 1, so only the second level projects
    assert stats['project_from_index'] == 1 and stats['projected_levels'] == 1
    assert stats['projection_calls'] == 2 and stats['exact_final'] is True
    # and the retry path: one failure costs one retry and 15 extra Langevin steps
    task = Toy(); task.fail_first = True; trace = []
    pdm_run(task, 'PDM', np.random.default_rng(0), cfg, Budget(5), trace)
    assert task.pdm_stats['partial_projections'] == 1
    assert sum(1 for r in trace if r['kind'] == 'retry_langevin') == 15


# 4 ---------------------------------------------------------------------------
# Removed 2026-11-14 with the code it pinned: `cd_beta` level-vs-interval aggregation was
# our workaround for a subsampled ladder. The official chain walks every trained level and
# reuses `beta_t` of the level it visits, so there is only one mode left; the ladder shape
# (including the 200 noiseless revisits of t=1) is pinned in
# tests/test_official_constrained_diffusers.py.


# 5 ---------------------------------------------------------------------------
# Removed 2026-11-14: `test_pd_and_alm_use_one_noise_draw_for_record_and_update` pinned our
# old interleaving of `task.last_nominal` with the Langevin draw. In the official loop the
# recorded nominal is the posterior mean and the state is `mean + std_t * z`; that relation
# is pinned in tests/test_official_constrained_diffusers.py instead.


# 6 ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['seven_native_points_paper_t256.json',
                                  'seven_native_points_vertexall_t256.json',
                                  'seven_native_points_vertexall_t16.json'])
def test_config_dual_parameters_are_the_paper_values_over_scale_squared(name):
    cfg = json.loads((CFG_DIR / name).read_text(encoding='utf8'))
    s = cfg['dual_constraint_scale']
    np.testing.assert_allclose(cfg['dual_eta'] * s ** 2, 1e-3, rtol=1e-4)   # paper large eta_lambda
    np.testing.assert_allclose(cfg['rho'] * s ** 2, 2.5e-4, rtol=1e-4)      # paper large rho


def test_paper_config_is_paper_native_and_derived_constants_are_consistent():
    cfg = json.loads((CFG_DIR / 'seven_native_points_paper_t256.json').read_text(encoding='utf8'))
    assert cfg['segment_constraints_for'] == ['PDM'], 'PDM must use App.C.3 segment constraints'
    assert cfg['pdm_parameterization'] == 've'
    assert cfg['cd_levels'] == cfg['schedule_facts']['timesteps']
    assert 'cd_beta' not in cfg and cfg['sd_hard_levels'] == max(1, round(10 * cfg['cd_levels'] / 1000))
    smin2 = cfg['schedule_facts']['smin2']
    np.testing.assert_allclose(cfg['pdm_ald_eps'] / (2 * smin2), 1. / 6., rtol=1e-6)


# 7 ---------------------------------------------------------------------------
def test_circle_cover_marks_a_cell_by_its_square_not_its_centre():
    occ = np.zeros((8, 8), bool); occ[3, 3] = occ[4, 4] = occ[2, 5] = True  # diagonal neighbours
    centers, radii, audit = obstacle_circles(occ, 2.)
    assert audit['covered_fraction'] == 1. and audit['covered_corner_fraction'] == 1.
    # a centre-only test would have accepted fewer circles; assert the criterion used
    assert 'square' in audit['coverage_test']


# 8 ---------------------------------------------------------------------------
def test_segment_through_centre_has_a_usable_gradient_and_escapes():
    seg = CircleSegmentConstraints(np.array([[-.5, 0.], [.5, 0.]]), 3, [[0., 0.]], [.2])
    x = np.zeros(seg.D)
    g, J = seg.constraints(x)
    assert g.max() > 0 and np.abs(J.toarray()).max() > 0
    out, _ = seg.project(x, Budget(30), maxiter=80)
    assert seg.residual(out) <= 1e-6


# 9 ---------------------------------------------------------------------------
# Removed 2026-11-14 together with the implementation it pinned: `legal_gamma` / N0 and
# the per-level discrete certificate were ours, not SafeDiffuser's. The official
# correction has no gamma and no certificate; its ReS hard interval is now pinned by
# tests/test_phase_e_safediffuser.py against the paper's `t >= 10` rule.


# 10 --------------------------------------------------------------------------
def test_initial_noise_is_shared_by_every_method():
    a = initial_noise(1234, 0, 128)
    b = initial_noise(1234, 0, 128)
    np.testing.assert_array_equal(a, b)
    assert initial_seed(1234, 0) != initial_seed(1235, 0)
    import torch
    class TinyModel(torch.nn.Module):
        num_points = 128
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
        def encode_scene(self, occ): return torch.zeros(1, 1, 1)
        def forward_from_memory(self, p, mem, cond, t): return {'trajectory': torch.zeros_like(p)}
    sched = NoiseSchedule(4)
    occ = np.zeros((8, 8), bool)
    ends = np.array([[-.5, 0.], [.5, 0.]])
    t1 = NativePointTask(TinyModel(), sched, occ, ends, 128, 0., 'exact')
    t2 = NativePointTask(TinyModel(), sched, occ, ends, 128, 0., 'exact')
    ov = a[1:-1].reshape(-1)
    t1.initial_override = ov; t2.initial_override = ov
    r1 = t1.initial(np.random.default_rng(7), 1)
    r2 = t2.initial(np.random.default_rng(99), 1)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(r1[0], ov)
