"""T3 U-shaped obstacle: the reference exits through a wall, the search does not."""

from .helpers import prep_scene, run


def test_t3_escapes_the_u_shape():
    prep = prep_scene("T3")
    res = run(prep, "pdg", w=1.2)
    assert res["status"] == "SUCCESS"
    assert res["path_valid"] == 1
    # the only way out is the bottom opening: the path must go below row 60
    rc = prep.case.spec.index_to_rc(res["_result"].path)
    assert rc[:, 0].max() > 60


def test_t3_matches_optimal_cost_bound():
    prep = prep_scene("T3")
    dijk = run(prep, "dijkstra")
    pdg = run(prep, "pdg", w=1.2)
    assert pdg["path_cost_m"] <= 1.2 * dijk["path_cost_m"] + 1e-6


def test_t4_two_channels_still_comparable():
    prep = prep_scene("T4")
    dijk = run(prep, "dijkstra")
    pdg = run(prep, "pdg", w=1.2)
    assert pdg["status"] == "SUCCESS"
    assert pdg["path_cost_m"] <= 1.2 * dijk["path_cost_m"] + 1e-6
