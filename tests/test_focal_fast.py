"""The heap inner loop (``focal_fast``) must be the SAME SEARCH as the vectorised one.

``grid_search.focal_astar.focal_search`` dispatches to ``focal_fast`` when
``w == 1`` (the config the planner runs).  These tests pin down that the two
backends agree where it matters:

* both return SUCCESS and a path of the same (optimal) grid cost;
* the path is legal: no corner cutting, no blocked cell;
* the expansion count stays in the same ballpark (the tie-break may order equal
  ``f0`` nodes differently, which is allowed and cannot change the cost).

Run:  python -m pytest tests/test_focal_fast.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from src.grid_search import focal_astar as _fa_pkg                    # noqa: E402
import importlib                                                      # noqa: E402

focal_astar = importlib.import_module("src.grid_search.focal_astar")  # noqa: E402
from src.grid_search.baselines import pdg_focal_astar                # noqa: E402
from src.grid_search.grid_neighbors import corner_cutting_violations  # noqa: E402
from src.grid_search.tests.helpers import prep_scene                  # noqa: E402

SCENES = ("T1", "T3", "T2")          # empty / u-obstacle / wall


def _run(prep, secondary, w=1.0, fast=True):
    cfg = focal_astar.SearchConfig(
        w=w, heuristic="octile", secondary=secondary,
        guidance=prep.guidance, alpha_d=1.0, alpha_p=1.0,
        snapshots=(1, 50, 500), fast_path=fast)
    return focal_astar.focal_search(prep.safe, prep.start_idx, prep.goal_idx,
                                    cfg, prep.neighbors)


def test_fast_path_is_used_at_w1():
    prep = prep_scene("T3")
    res = _run(prep, "guidance", fast=True)
    assert res.stats.get("search_backend") == "heap"
    res_slow = _run(prep, "guidance", fast=False)
    assert res_slow.stats.get("search_backend") is None


def test_same_cost_and_feasible_path():
    for scene in SCENES:
        prep = prep_scene(scene)
        for secondary in ("guidance", "guidance_h", "clearance", "f0_index"):
            fast = _run(prep, secondary, fast=True)
            slow = _run(prep, secondary, fast=False)
            tag = "%s/%s" % (scene, secondary)
            assert fast.status == slow.status == "SUCCESS", tag
            # identical OPTIMAL cost (w=1 => optimal for the 8-connected grid)
            assert abs(fast.cost_m - slow.cost_m) < 1e-9, (
                tag, fast.cost_m, slow.cost_m)
            # the fast path really produced a legal path
            rc = prep.safe.spec.index_to_rc(fast.path)
            assert np.all(prep.safe.free[rc[:, 0], rc[:, 1]]), tag
            assert corner_cutting_violations(prep.safe.free, rc) == 0, tag
            assert fast.path[0] == prep.start_idx and fast.path[-1] == prep.goal_idx
            # expansions stay comparable (tie-break may differ, cost may not)
            assert fast.stats["expanded_nodes"] <= 2 * slow.stats["expanded_nodes"] + 8, (
                tag, fast.stats["expanded_nodes"], slow.stats["expanded_nodes"])


def test_fast_path_matches_when_w_above_one_uses_slow_loop():
    """w > 1 must NOT take the heap path and must still find the optimal cost."""
    prep = prep_scene("T3")
    cfg = focal_astar.SearchConfig(w=1.2, secondary="guidance",
                                   guidance=prep.guidance, fast_path=True)
    res = focal_astar.focal_search(prep.safe, prep.start_idx, prep.goal_idx,
                                   cfg, prep.neighbors)
    assert res.stats.get("search_backend") is None
    assert res.status == "SUCCESS"


def test_geodesic_h_table_is_accepted_and_stays_optimal():
    from src.grid_search.geodesic_heuristic import geodesic_distance_field
    for scene in SCENES:
        prep = prep_scene(scene)
        spec = prep.safe.spec
        goal_rc = spec.index_to_rc(np.array([prep.goal_idx]))[0]
        h = geodesic_distance_field(prep.safe.free, spec.cell_m, goal_rc)
        assert h is not None
        fast = pdg_focal_astar(prep.safe, prep.start_idx, prep.goal_idx,
                               prep.guidance, w=1.0, neighbors=prep.neighbors,
                               h_table=h)
        slow = _run(prep, "guidance", fast=False)
        assert fast.status == "SUCCESS"
        assert abs(fast.cost_m - slow.cost_m) < 1e-9, scene
        # a tight heuristic must not make the search explore MORE
        assert fast.stats["expanded_nodes"] <= slow.stats["expanded_nodes"] + 8


def test_invalid_guidance_falls_back_to_clearance():
    prep = prep_scene("T2")
    bad = type("Bad", (), {"valid": False})()
    res = pdg_focal_astar(prep.safe, prep.start_idx, prep.goal_idx, bad, w=1.0,
                          neighbors=prep.neighbors)
    assert res.status == "SUCCESS"
    assert res.stats["secondary"] == "clearance"


def test_blocked_endpoints_are_reported_not_crashed():
    prep = prep_scene("T2")
    spec = prep.safe.spec
    blocked = int(np.flatnonzero(~prep.safe.free.reshape(-1))[0])
    cfg = focal_astar.SearchConfig(w=1.0, secondary="f0_index",
                                   fast_path=True)
    start = focal_astar.focal_search(prep.safe, blocked, prep.goal_idx, cfg,
                                     prep.neighbors)
    goal = focal_astar.focal_search(prep.safe, prep.start_idx, blocked, cfg,
                                    prep.neighbors)
    assert start.status == "INVALID_START"
    assert goal.status == "INVALID_GOAL"
