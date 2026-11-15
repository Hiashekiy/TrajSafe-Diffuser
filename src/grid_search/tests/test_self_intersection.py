"""T5/T6/T7: self-intersection, near-parallel legs and a backtracking reference."""

import numpy as np

from .helpers import prep_scene, run


def _probe_candidates(field, spec, rc):
    idx = int(spec.rc_to_index(np.asarray(rc, dtype=np.int64)))
    return field.dist_m[idx], field.prog[idx], field.seg_id[idx]


def test_t5_crossing_cell_keeps_distinct_progress_candidates():
    prep = prep_scene("T5")
    assert prep.guidance.valid
    d, u, seg = _probe_candidates(prep.guidance, prep.case.spec, (40, 40))
    assert len(d) >= 3
    assert np.all(np.diff(d) >= -1e-6)                    # sorted by distance
    assert len(set(seg.tolist())) >= 2                    # different segments
    assert float(u.max() - u.min()) > 0.2                 # different progress


def test_t6_parallel_legs_are_disambiguated_by_progress():
    prep = prep_scene("T6")
    d, u, seg = _probe_candidates(prep.guidance, prep.case.spec, (46, 40))
    assert float(u.max() - u.min()) > 0.2
    # the progress term must be able to pick a different candidate than distance
    node = np.array([prep.case.spec.rc_to_index(np.array([46, 40]))])
    g = np.array([0.35 * prep.ref.length_m])              # 35 % into the path
    h_dist_only = prep.guidance.score(node, g, alpha_d=1.0, alpha_p=0.0)
    h_prog = prep.guidance.score(node, g, alpha_d=1.0, alpha_p=1.0)
    assert np.isfinite(h_dist_only[0]) and np.isfinite(h_prog[0])
    assert h_prog[0] >= h_dist_only[0] - 1e-9


def test_t7_backtracking_reference_still_valid():
    prep = prep_scene("T7")
    assert prep.ref.valid
    assert prep.ref.diagnostics["n_segments"] >= 2
    res = run(prep, "pdg", w=1.2)
    assert res["status"] == "SUCCESS"
    assert res["path_valid"] == 1


def test_progress_term_changes_the_ordering():
    """The progress term must actually influence FOCAL ordering somewhere."""
    prep = prep_scene("T5")
    a = run(prep, "pdg", w=1.3, alpha_d=1.0, alpha_p=0.0)
    b = run(prep, "pdg", w=1.3, alpha_d=1.0, alpha_p=1.0)
    assert a["status"] == b["status"] == "SUCCESS"
    # either a different expansion count or a different path: not a no-op
    same_path = (a["_result"].path is not None
                 and np.array_equal(a["_result"].path, b["_result"].path))
    assert (a["expanded_nodes"] != b["expanded_nodes"]) or (not same_path)
