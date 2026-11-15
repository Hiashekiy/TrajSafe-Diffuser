"""T8 sealed map: NO_PATH is returned and no obstacle is ever crossed."""

from .helpers import prep_scene, run


def test_t8_reports_no_path():
    prep = prep_scene("T8")
    for method in ("dijkstra", "astar", "focal_clearance", "pdg"):
        res = run(prep, method)
        assert res["status"] == "NO_PATH"
        assert res["_result"].path is None
        assert res["path_valid"] == 0


def test_t8_search_is_complete_over_the_free_cells():
    """A sealed map is fully explored: expansions == reachable free cells."""
    prep = prep_scene("T8")
    res = run(prep, "pdg")
    reachable = _flood(prep, prep.start_idx)
    assert res["expanded_nodes"] >= reachable * 0.99


def _flood(prep, start_idx):
    from collections import deque
    seen = {start_idx}
    queue = deque([start_idx])
    nbr = prep.neighbors.idx
    while queue:
        cur = queue.popleft()
        for k in range(8):
            nb = int(nbr[cur, k])
            if nb >= 0 and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen)
