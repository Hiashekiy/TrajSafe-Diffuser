"""T1 empty map: a path is found and the cost is the true grid cost."""

import numpy as np

from .helpers import prep_scene, run


def test_t1_astar_cost_equals_straight_line():
    prep = prep_scene("T1")
    res = run(prep, "astar")
    assert res["status"] == "SUCCESS"
    assert res["path_valid"] == 1
    # start (60,10) -> goal (60,70): 60 horizontal cells of 1 m
    assert abs(res["path_cost_m"] - 60.0) < 1e-6
    assert res["turn_count"] == 0


def test_t1_all_methods_agree_with_dijkstra():
    prep = prep_scene("T1")
    dijk = run(prep, "dijkstra")
    astar = run(prep, "astar")
    pdg = run(prep, "pdg", w=1.0)
    focal = run(prep, "focal_clearance", w=1.2)
    for res in (astar, pdg, focal):
        assert res["status"] == "SUCCESS"
        assert abs(res["path_cost_m"] - dijk["path_cost_m"]) < 1e-6


def test_t1_astar_expands_fewer_nodes_than_dijkstra():
    prep = prep_scene("T1")
    dijk = run(prep, "dijkstra")
    astar = run(prep, "astar")
    assert astar["expanded_nodes"] < dijk["expanded_nodes"]


def test_t1_guidance_is_built_over_the_whole_map():
    prep = prep_scene("T1")
    field = prep.guidance
    assert field.valid
    dist = field.dist_m[:, 0]
    assert np.isfinite(dist).mean() == 1.0          # no cell left uncovered
    assert np.all(np.abs(dist - dist.min()) <= prep.case.spec.n_cells ** 0.5)
