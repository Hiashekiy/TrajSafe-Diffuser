"""Guidance-field unit tests (design section 4)."""

import numpy as np

from ..diffusion_path import prepare_reference_path
from ..grid_map import GridSpec
from ..progress_guidance import build_progress_guidance


def _straight_field(n=41):
    spec = GridSpec(n, n, 1.0)
    pts_cell = np.array([[20.0, 5.0], [20.0, 35.0]])
    pts_world = spec.cell_to_world(pts_cell[:, ::-1])
    ref = prepare_reference_path(pts_world, spec)
    return spec, ref, build_progress_guidance(ref, spec, J=3)


def test_distance_and_progress_are_consistent_on_a_straight_line():
    spec, ref, field = _straight_field()
    assert ref.valid and field.valid
    idx = spec.rc_to_index(np.array([20, 20]))          # on the line
    assert field.dist_m[idx, 0] < 1e-6
    assert abs(float(field.prog[idx, 0]) - 0.5) < 1e-3
    idx_off = spec.rc_to_index(np.array([25, 20]))       # 5 cells away
    assert abs(float(field.dist_m[idx_off, 0]) - 5.0) < 1e-6
    assert abs(float(field.prog[idx_off, 0]) - 0.5) < 1e-3


def test_score_formula_matches_the_design():
    spec, ref, field = _straight_field()
    idx = spec.rc_to_index(np.array([24, 20]))
    g = np.array([0.5 * field.length_m])
    h = field.score(np.array([idx]), g, alpha_d=1.0, alpha_p=1.0,
                    sigma_m=3.0, L_ref=field.length_m)[0]
    manual = min(float(field.dist_m[idx, j]) / 3.0
                 + abs(float(field.prog[idx, j]) - 0.5)
                 for j in range(field.dist_m.shape[1]))
    assert abs(h - manual) < 1e-6


def test_progress_term_is_zero_when_progress_matches():
    spec, ref, field = _straight_field()
    idx = spec.rc_to_index(np.array([20, 20]))
    u = float(field.prog[idx, 0])
    g = np.array([u * field.length_m])
    h_dist = field.score(np.array([idx]), g, alpha_d=1.0, alpha_p=0.0,
                         sigma_m=3.0, L_ref=field.length_m)[0]
    h_both = field.score(np.array([idx]), g, alpha_d=1.0, alpha_p=1.0,
                         sigma_m=3.0, L_ref=field.length_m)[0]
    assert abs(h_both - h_dist) < 1e-4


def test_all_cells_keep_candidates():
    spec, ref, field = _straight_field()
    n_j = field.dist_m.shape[1]
    assert field.dist_m.shape == (spec.n_cells, n_j)
    assert 1 <= n_j <= 3           # a single straight segment yields 1 candidate
    assert np.isfinite(field.dist_m).all()
    assert np.all(field.prog >= -1e-6) and np.all(field.prog <= 1 + 1e-6)


def test_candidates_are_sorted_by_distance():
    spec, ref, field = _straight_field()
    assert np.all(np.diff(field.dist_m, axis=1) >= -1e-6)
