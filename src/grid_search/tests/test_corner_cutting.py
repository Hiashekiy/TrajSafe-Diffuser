"""T11 corner cutting: a diagonal may not slip between two touching obstacles."""

import numpy as np

from ..grid_map import GridSpec, build_safe_grid
from ..grid_neighbors import build_neighbor_table, corner_cutting_violations
from .helpers import prep_scene, run


def test_diagonal_between_blocked_sides_is_rejected():
    # 2x2 map: free / blocked / blocked / free  -> the diagonal is illegal
    occ = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    spec = GridSpec(2, 2, 1.0)
    safe = build_safe_grid(occ, spec, robot_radius_m=0.0, safety_margin_m=0.0)
    table = build_neighbor_table(safe.free, spec.cell_m)
    # move 4 is (+row, +col): (0,0) -> (1,1)
    assert int(table.idx[0, 4]) == -1
    assert int(table.idx[0, 0]) == -1        # (0,1) is blocked
    assert int(table.idx[0, 2]) == -1        # (1,0) is blocked


def test_plus_pattern_centre_has_no_legal_move():
    occ = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]], dtype=np.uint8)
    spec = GridSpec(3, 3, 1.0)
    safe = build_safe_grid(occ, spec, robot_radius_m=0.0, safety_margin_m=0.0)
    table = build_neighbor_table(safe.free, spec.cell_m)
    centre = 1 * 3 + 1
    assert int((table.idx[centre] >= 0).sum()) == 0


def test_t11_path_avoids_the_corner():
    prep = prep_scene("T11")
    res = run(prep, "pdg", w=1.2)
    assert res["status"] == "SUCCESS"
    assert res["path_valid"] == 1
    rc = prep.case.spec.index_to_rc(res["_result"].path)
    assert corner_cutting_violations(prep.safe.free, rc) == 0
    # the illegal shortcut (40,41) -> (41,40) must not appear
    pairs = set(zip(map(tuple, rc[:-1]), map(tuple, rc[1:])))
    assert ((40, 41), (41, 40)) not in pairs
    # and the detour must be longer than the illegal diagonal
    assert res["path_cost_m"] > np.sqrt(2.0) * 2
