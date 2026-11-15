"""T9 / T12 endpoint handling."""

import numpy as np

from .helpers import prep_scene, run


def test_t9_start_in_obstacle_is_reported():
    prep = prep_scene("T9")
    res = run(prep, "pdg")
    assert res["status"] == "INVALID_START"
    assert res["success"] == 0


def test_t9b_goal_in_obstacle_is_reported():
    prep = prep_scene("T9b")
    res = run(prep, "pdg")
    assert res["status"] == "INVALID_GOAL"


def test_t12_same_endpoints_single_point_zero_cost():
    prep = prep_scene("T12")
    res = run(prep, "pdg")
    assert res["status"] == "SUCCESS"
    assert res["path_cells"] == 1
    assert res["path_cost_m"] == 0.0
    assert res["path_valid"] == 1


def test_start_goal_are_not_silently_snapped():
    """A start on an obstacle stays there: no hidden snapping, an explicit error."""
    prep = prep_scene("T9")
    assert not bool(prep.safe.is_free_index(np.array([prep.start_idx]))[0])
    assert prep.start_idx == prep.case.spec.cell_to_index(
        prep.case.spec.world_to_cell(prep.case.start_world))
