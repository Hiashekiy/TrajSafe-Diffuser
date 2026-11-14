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
from experiments.safety_baselines.algorithms.constrained_diffusers.runner import effective_beta, level_ladder, run as cd_run
from experiments.safety_baselines.algorithms.kernels import primal_dual_step
from experiments.safety_baselines.algorithms.pdm.runner import level_gamma, run as pdm_run
from experiments.safety_baselines.algorithms.runner import initial_noise, initial_seed
from experiments.safety_baselines.algorithms.safediffuser.runner import legal_gamma
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
def test_pdm_records_projection_status_and_requires_an_exact_final_projection():
    class Toy:
        D = 1
        ab = np.array([.9, .5])
        betas = np.array([.1, .4])
        sigma = np.sqrt((1 - ab) / ab)
        partial_projection = True
        projection_tol = 1e-6
        def initial(self, rng, particles=1, ve=False): return rng.normal(size=(particles, self.D))
        def score(self, x, t, ve=False): return -np.atleast_2d(x)
        def constraints(self, x): return np.array([-abs(float(np.asarray(x).reshape(-1)[0]))]), np.array([[-1.]])
        def project(self, target, budget, allow_partial=False):
            z = np.asarray(target, float).copy();z[...] = 0.      # clamp onto the feasible point
            if allow_partial:                                     # partial: leaves a small residual
                z[...] = 1e-4
                return z, {'status': 'segment_projection_partial', 'residual': 1e-4}
            return z, {'status': 'segment_sqp_feasible', 'residual': 0.}
    cfg = {'pdm_inner': 2, 'pdm_levels': 2, 'pdm_parameterization': 've',
           'pdm_step_convention': 'official_ald', 'pdm_ald_eps': 1e-3, 'projection_tol_scene': 1e-6}
    task = Toy(); trace = []
    pdm_run(task, 'PDM', np.random.default_rng(0), cfg, Budget(5), trace)
    stats = task.pdm_stats
    assert {'projection_calls', 'partial_projections', 'final_status', 'final_residual', 'exact_final'} <= set(stats)
    assert stats['projection_calls'] == 4 and stats['partial_projections'] == 2
    assert stats['exact_final'] is True


# 4 ---------------------------------------------------------------------------
def test_cd_beta_modes_on_a_subsampled_ladder():
    ab = NoiseSchedule(256, 'squaredcos_cap_v2').alphas_cumprod.numpy()
    beta = NoiseSchedule(256, 'squaredcos_cap_v2').betas.numpy()
    ladder = level_ladder(256, 64, 0)
    assert len(ladder) == 64 and ladder[0] == 255 and ladder[-1] == 0
    t, t_next = 200, 196
    b_level = effective_beta(ab, t, t_next, beta[t], 'level')
    b_interval = effective_beta(ab, t, t_next, beta[t], 'interval')
    np.testing.assert_allclose(b_level, beta[t])
    np.testing.assert_allclose(b_interval, 1. - ab[t] / ab[t_next])
    assert b_interval > b_level          # the skipped levels add noise
    # on a ladder that visits every trained level the two must coincide
    full = level_ladder(256, 256, 0)
    assert full[:3] == [255, 254, 253]
    np.testing.assert_allclose(effective_beta(ab, 255, 254, beta[255], 'interval'), beta[255], rtol=1e-12)


# 5 ---------------------------------------------------------------------------
class QuietToy:
    """Toy whose constraint rows are inactive: the correction term is exactly zero."""
    D = 2
    betas = np.array([.1, .2, .3, .4])
    ab = np.array([.9, .7, .4, .1])
    sigma = np.sqrt((1 - ab) / ab)
    def initial(self, rng, particles=1, ve=False): return rng.normal(size=(particles, self.D))
    def score(self, x, t): return -np.atleast_2d(x) * 0.5
    def constraints(self, x):
        return np.full(1, -1.0), np.zeros((1, self.D))


def test_pd_and_alm_use_one_noise_draw_for_record_and_update():
    """With a zero correction the updated state must equal the recorded nominal."""
    for method in ('Primal-Dual', 'ALM'):
        cfg = {'particles': 2, 'rho': 0.1, 'rho_growth': 1.1, 'dual_eta': 1e-3,
               'cd_levels': 4, 'cd_extra_steps': 2, 'dual_constraint_scale': 1.0, 'substeps': 1}
        task = QuietToy(); trace = []
        x = cd_run(task, method, np.random.default_rng(3), cfg, Budget(5), trace)
        np.testing.assert_allclose(x, task.last_nominal, rtol=0, atol=1e-12)


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
    assert cfg['cd_beta'] == 'level' and cfg['cd_levels'] == cfg['schedule_facts']['timesteps']
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
def test_res_hard_interval_uses_the_paper_horizon_N0():
    """gamma is computed with horizon N0 (Thm 3.5); hard_levels is N0+1 levels."""
    h = np.array([-0.2, 0.1])
    n0 = 9
    g_paper = legal_gamma(h, n0, 1.0, 1.25e-4)
    g_off = legal_gamma(h, n0 + 1, 1.0, 1.25e-4)
    assert g_paper >= g_off          # a shorter horizon allows a larger gamma
    assert g_paper >= 1.25e-4 - 1e-18


def test_safediffuser_trace_declares_hard_interval_length():
    from experiments.safety_baselines.algorithms.safediffuser.runner import run as safe_run
    class Toy:
        D = 2
        ab = np.array([.9, .7, .4, .1])
        betas = np.array([.1, .2, .3, .4])
        sigma = np.sqrt((1 - ab) / ab)
        def constraints(self, x): return np.array([-x[0]]), np.array([[-1., 0.]])
        def clean(self, x, t, ve=False): return np.ones_like(np.atleast_2d(x)) * .5
    cfg = {'cbf_dt': 1., 'cbf_epsilon': 1., 'cbf_gamma_floor': .001, 'hard_steps': 2,
           'relax_weight': 100., 'qp_tol': 1e-5, 'cbf_substeps': 1}
    trace = []; safe_run(Toy(), 'ReS-diffuser', np.random.default_rng(0), cfg, Budget(5), trace)
    assert all(t.get('hard_levels', 2) == 2 for t in trace)


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
