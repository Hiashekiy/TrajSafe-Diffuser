"""T2 single wall: the straight reference crosses it, the search detours."""

from .helpers import prep_scene, run


def test_t2_detours_around_the_wall():
    prep = prep_scene("T2")
    dijk = run(prep, "dijkstra")
    astar = run(prep, "astar")
    pdg = run(prep, "pdg", w=1.2)
    for res in (dijk, astar, pdg):
        assert res["status"] == "SUCCESS"
        assert res["path_valid"] == 1
        assert res["path_cost_m"] > 60.0        # cannot go straight
    # A* must stay optimal on the same graph
    assert abs(astar["path_cost_m"] - dijk["path_cost_m"]) < 1e-6
    # PDG-Focal is bounded by w
    assert pdg["path_cost_m"] <= 1.2 * dijk["path_cost_m"] + 1e-6


def test_t2_reference_crosses_obstacles_but_search_does_not():
    prep = prep_scene("T2")
    res = run(prep, "pdg")
    assert res["reference_collision_rate"] > 0.0
    assert res["path_valid"] == 1
    # the returned path never visits a blocked cell
    from ..path_validation import validate_path
    check = validate_path(prep.safe, res["_result"].path)
    assert not [e for e in check["errors"] if e.startswith("blocked_cells")]

