"""The fused guidance cost (lambda > 0) is kept as an evaluated alternative.

It is NOT the method (the report's section 7.6 explains why: adding the penalty
to g destroys the sharpness of the octile heuristic and the search degenerates
towards Dijkstra).  These tests only pin down its contract, and make sure the
default configuration is still exactly plain A*.
"""

import numpy as np

from .helpers import prep_scene, run


def _fused(prep, lam, ap=1.0):
    return run(prep, "guided_astar", w=1.0, alpha_p=ap, lam=lam)


def test_lambda_zero_is_plain_astar():
    prep = prep_scene("T2")
    a = run(prep, "astar")
    f = _fused(prep, 0.0)
    assert f["status"] == a["status"]
    assert np.array_equal(f["_result"].path, a["_result"].path)
    assert f["expanded_nodes"] == a["expanded_nodes"]
    assert abs(f["path_cost_m"] - a["path_cost_m"]) < 1e-9


def test_positive_lambda_still_returns_a_valid_path():
    prep = prep_scene("T4")
    f = _fused(prep, 0.1)
    assert f["status"] == "SUCCESS"
    assert f["path_valid"] == 1
    assert f["guide_penalty_m"] is not None
    assert f["guide_penalty_m"] >= 0.0


def test_penalty_is_paid_and_grows_with_lambda():
    prep = prep_scene("T4")
    pens = [_fused(prep, lam)["guide_penalty_m"] for lam in (0.0, 0.05, 0.2)]
    assert pens[0] == 0.0
    assert pens[1] > 0.0 and pens[2] > pens[1]


def test_fused_path_cost_is_never_below_optimal():
    prep = prep_scene("T4")
    dijk = run(prep, "dijkstra")
    for lam in (0.02, 0.1, 0.5):
        f = _fused(prep, lam)
        assert f["path_cost_m"] >= dijk["path_cost_m"] - 1e-9


def test_invalid_reference_does_not_break_the_fused_variant():
    from .helpers import case_from_scene
    from ..benchmark import prepare_case
    from ..scenes import make_scene
    scene = make_scene("T10")
    prep = prepare_case(case_from_scene(scene, ref_world=np.array([])),
                        robot_radius_m=0.0, safety_margin_m=0.0)
    f = _fused(prep, 0.2)
    assert f["status"] == "SUCCESS"
    assert f["guide_penalty_m"] == 0.0
