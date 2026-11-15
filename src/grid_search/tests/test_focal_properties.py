"""Focal-search correctness: bounds, dynamic FOCAL maintenance, determinism."""

import numpy as np
import pytest

from .helpers import naive_focal_search, prep_scene, run


@pytest.mark.parametrize("w", [1.0, 1.1, 1.2, 1.3, 1.5])
def test_cost_stays_within_the_focal_bound(w):
    for ident in ("T2", "T4", "T5", "T7"):
        prep = prep_scene(ident)
        dijk = run(prep, "dijkstra")
        pdg = run(prep, "pdg", w=w, alpha_p=1.0)
        assert pdg["status"] == "SUCCESS"
        assert pdg["path_cost_m"] <= w * dijk["path_cost_m"] + 1e-6


def test_w1_guidance_never_changes_the_optimum():
    prep = prep_scene("T4")
    dijk = run(prep, "dijkstra")
    pdg = run(prep, "pdg", w=1.0)
    assert abs(pdg["path_cost_m"] - dijk["path_cost_m"]) < 1e-6


@pytest.mark.parametrize("label,secondary", [("astar", "f0_index"),
                                             ("focal_clearance", "clearance"),
                                             ("focal_index", "index")])
def test_heap_focal_equals_naive_full_scan(label, secondary):
    """The lazy-deletion heap must implement exactly the same FOCAL rule."""
    for ident in ("T2", "T4", "T5"):
        prep = prep_scene(ident)
        fast = run(prep, label, w=1.2)
        slow = naive_focal_search(prep.safe, prep.start_idx, prep.goal_idx,
                                  w=1.2, secondary=secondary)
        assert fast["status"] == slow["status"]
        assert abs(fast["path_cost_m"] - slow["cost"]) < 1e-6
        assert fast["expanded_nodes"] == slow["expanded"]


def test_guidance_does_not_touch_anchors_or_costs():
    """Same graph, same h0: only the node ORDER inside FOCAL may differ."""
    from ..path_validation import path_cost_m
    prep = prep_scene("T4")
    for label in ("focal_clearance", "pdg"):
        res = run(prep, label, w=1.2)
        assert abs(path_cost_m(prep.safe, res["_result"].path)
                   - res["path_cost_m"]) < 1e-9


def test_runs_are_deterministic():
    prep = prep_scene("T5")
    a = run(prep, "pdg", w=1.3)
    b = run(prep, "pdg", w=1.3)
    assert np.array_equal(a["_result"].path, b["_result"].path)
    assert a["expanded_nodes"] == b["expanded_nodes"]


def test_every_expansion_is_a_free_cell():
    from ..focal_astar import SearchConfig, focal_search
    prep = prep_scene("T4")
    cfg = SearchConfig(w=1.2, secondary="guidance", guidance=prep.guidance,
                       sigma_m=3.0)
    res = focal_search(prep.safe, prep.start_idx, prep.goal_idx, cfg,
                       prep.neighbors)
    assert res.status == "SUCCESS"
    rc = prep.case.spec.index_to_rc(res.expanded)
    assert bool(prep.safe.free[rc[:, 0], rc[:, 1]].all())
    assert len(res.expanded) == len(set(res.expanded.tolist()))
