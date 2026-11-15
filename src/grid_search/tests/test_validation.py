"""The independent path checker must actually catch broken paths."""

import numpy as np

from ..path_validation import path_cost_m, validate_path
from .helpers import prep_scene, run


def test_valid_path_passes():
    prep = prep_scene("T2")
    res = run(prep, "astar")
    check = validate_path(prep.safe, res["_result"].path, prep.start_idx,
                          prep.goal_idx, res["path_cost_m"])
    assert check["valid"], check["errors"]


def test_blocked_cell_is_caught():
    prep = prep_scene("T2")
    res = run(prep, "astar")
    path = np.asarray(res["_result"].path, dtype=np.int64).copy()
    spec = prep.case.spec
    # replace the middle vertex by a blocked wall cell
    wall = int(spec.rc_to_index(np.array([40, 40])))
    path = np.insert(path, len(path) // 2, wall)
    check = validate_path(prep.safe, path)
    assert not check["valid"]
    assert any(e.startswith("blocked_cells") or e.startswith("illegal_edges")
               for e in check["errors"])


def test_cost_mismatch_is_caught():
    prep = prep_scene("T2")
    res = run(prep, "astar")
    check = validate_path(prep.safe, res["_result"].path,
                          claimed_cost=res["path_cost_m"] + 1.0)
    assert not check["valid"]
    assert any(e.startswith("cost_mismatch") for e in check["errors"])


def test_corner_cut_is_caught():
    prep = prep_scene("T11")
    spec = prep.case.spec
    a = int(spec.rc_to_index(np.array([40, 41])))
    b = int(spec.rc_to_index(np.array([41, 40])))
    check = validate_path(prep.safe, np.array([a, b]))
    assert not check["valid"]
    assert any(e.startswith("illegal_edges") or e == "corner_cutting"
               for e in check["errors"])


def test_cost_is_recomputed_from_the_geometry():
    prep = prep_scene("T1")
    res = run(prep, "astar")
    path = res["_result"].path
    assert abs(path_cost_m(prep.safe, path) - 60.0) < 1e-9
