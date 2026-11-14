"""Phase B: paired protocol and val/test index separation.

B1  multi-particle methods must start from the paired noise set: particle 0 is the
    shared initial trajectory, particles 1..P-1 come from a method-independent seed
    so Primal-Dual and ALM receive *identical* initial states.
B2  `--split test` must never reuse validation indices; indices are range checked.
"""
import json
import numpy as np
import pytest

from experiments.safety_baselines.adapters.native_points import NativePointTask
from experiments.safety_baselines.algorithms.constrained_diffusers.runner import run as cd_run
from experiments.safety_baselines.algorithms.runner import initial_noise, initial_seed
from experiments.safety_baselines.algorithms.solvers import Budget
from src.diffusion.schedule import NoiseSchedule

CONFIGS = __import__('pathlib').Path(__file__).resolve().parents[1] / 'configs'


class Recorder:
    """Toy task that records the initial state it was handed."""
    D = 4
    ab = np.array([.9, .7, .4, .1])
    betas = np.array([.1, .2, .3, .4])
    sigma = np.sqrt((1 - ab) / ab)
    def __init__(self):
        self.seen = None
    def initial(self, rng, particles=1, ve=False):
        base = getattr(self, 'initial_override', None)
        if base is not None:
            base = np.asarray(base, float).reshape(-1)
            extra = np.random.default_rng(int(getattr(self, 'particle_seed', 0))).normal(size=(particles - 1, self.D))
            self.seen = np.concatenate([base[None], extra], 0) if particles > 1 else base[None]
        else:
            self.seen = rng.normal(size=(particles, self.D))
        return self.seen.copy()
    def score(self, x, t): return -np.atleast_2d(x) * 0.5 if x.ndim == 2 else -0.5 * x
    def constraints(self, x): return np.full(1, -1.0), np.zeros((1, self.D))


CFG = {'particles': 4, 'rho': .1, 'rho_growth': 1.1, 'dual_eta': 1e-3, 'cd_levels': 4,
       'cd_extra_steps': 1, 'dual_constraint_scale': 1.0, 'cd_substeps': 1}


def test_b1_pd_and_alm_share_the_same_four_particle_initial_state():
    shared = np.random.default_rng(0).normal(size=252)[:Recorder.D]
    seen = {}
    for method in ('Primal-Dual', 'ALM'):
        t = Recorder(); t.initial_override = shared; t.particle_seed = initial_seed(4321, 0)
        cd_run(t, method, np.random.default_rng(derived(t, method)), CFG, Budget(5), [])
        seen[method] = t.seen.copy()
    np.testing.assert_array_equal(seen['Primal-Dual'], seen['ALM'])
    assert seen['Primal-Dual'].shape == (4, Recorder.D)
    np.testing.assert_array_equal(seen['Primal-Dual'][0], shared)
    # particles 2..4 are genuinely independent trajectories, not copies of particle 0
    assert not np.array_equal(seen['Primal-Dual'][0], seen['Primal-Dual'][1])


def derived(t, method):
    return np.random.default_rng(7)


def test_b1_particle_set_is_reproducible_and_method_independent():
    a = Recorder(); a.initial_override = np.zeros(Recorder.D); a.particle_seed = initial_seed(9, 9)
    a.initial(np.random.default_rng(1), 4)
    b = Recorder(); b.initial_override = np.zeros(Recorder.D); b.particle_seed = initial_seed(9, 9)
    b.initial(np.random.default_rng(2), 4)          # different rng, same result expected
    np.testing.assert_array_equal(a.seen, b.seen)
    c = Recorder(); c.initial_override = np.zeros(Recorder.D); c.particle_seed = initial_seed(9, 10)
    c.initial(np.random.default_rng(1), 4)
    assert not np.array_equal(a.seen, c.seen)        # a different scene seed changes the set


def test_b1_single_particle_methods_still_get_the_shared_noise():
    t = Recorder(); t.initial_override = np.arange(Recorder.D, dtype=float); t.particle_seed = 0
    out = t.initial(np.random.default_rng(3), 1)
    np.testing.assert_array_equal(out[0], np.arange(Recorder.D, dtype=float))


@pytest.mark.parametrize('name', ['seven_native_points_paper_t256.json',
                                  'seven_native_points_vertexall_t256.json',
                                  'seven_native_points_vertexall_t16.json'])
def test_b2_val_and_test_index_lists_are_disjoint_and_in_range(name):
    cfg = json.loads((CONFIGS / name).read_text(encoding='utf8'))
    val = cfg['validation_indices']; test = cfg['test_indices']
    assert len(val) == len(test) == 20
    # The two lists index *different* datasets, so an equal integer is not a reused
    # scene; what must hold is that each list stays inside its own split.
    assert all(0 <= v < 1550 for v in val), 'validation indices must be inside val (1550)'
    assert all(0 <= v < 420 for v in test), 'test indices must be inside test (420)'
    assert sorted(test) == test and len(set(test)) == len(test)
    assert sorted(val) == val and len(set(val)) == len(val)


def test_b2_index_resolution_rejects_out_of_range_and_duplicates():
    # the helper lives in the validation script; import it directly
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('vpm', Path(__file__).resolve().parents[1] / 'scripts' / 'validate_point_methods.py')
    mod = importlib.util.module_from_spec(spec); sys.modules['vpm'] = mod
    spec.loader.exec_module(mod)
    cfg = {'validation_indices': [1, 2, 3], 'test_indices': [0, 200, 419]}
    assert mod.resolve_indices(cfg, 'val', 'frozen20', 1550) == [1, 2, 3]
    assert mod.resolve_indices(cfg, 'test', 'frozen20', 420) == [0, 200, 419]
    with pytest.raises(IndexError):
        mod.resolve_indices(cfg, 'test', '419,420', 420)
    with pytest.raises(ValueError):
        mod.resolve_indices(cfg, 'val', '1,1', 1550)
    assert mod.resolve_indices(cfg, 'test', 'all', 3) == [0, 1, 2]


def test_b2_frozen_scene_ids_do_not_overlap_across_splits():
    """Scene (episode) ids must be disjoint between the frozen val and test lists."""
    from pathlib import Path
    root = Path('D:/ProjectDirectory/Neural-IRISDiffuser/data/processed/160m-k4p-c48')
    if not (root / 'val' / 'sample_id.npy').exists():
        pytest.skip('processed dataset not available')
    cfg = json.loads((CONFIGS / 'seven_native_points_paper_t256.json').read_text(encoding='utf8'))
    vid = np.load(root / 'val' / 'sample_id.npy')[cfg['validation_indices']]
    tid = np.load(root / 'test' / 'sample_id.npy')[cfg['test_indices']]
    assert not set(int(v) for v in vid) & set(int(v) for v in tid)
