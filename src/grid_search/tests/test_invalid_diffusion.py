"""T10 invalid diffusion reference: degrade, never fail."""

import numpy as np

from .helpers import case_from_scene, prep_scene, run
from ..benchmark import prepare_case
from ..scenes import make_scene


def _prep_with_ref(ref_world):
    scene = make_scene("T10")
    case = case_from_scene(scene, ref_world=ref_world)
    return prepare_case(case, robot_radius_m=0.0, safety_margin_m=0.0)


def test_nan_reference_degrades_to_unguided():
    prep = _prep_with_ref(np.array([[np.nan, 0.0], [0.1, 0.1]]))
    assert prep.guidance.valid is False
    assert prep.guidance.reason
    res = run(prep, "pdg", w=1.2)
    assert res["status"] == "SUCCESS"
    assert res["path_valid"] == 1
    assert res["guidance_used"] == 0
    assert res["expanded_nodes"] == run(prep, "focal_clearance")["expanded_nodes"]


def test_empty_reference_degrades():
    prep = _prep_with_ref(np.zeros((0, 2)))
    assert prep.ref.valid is False
    res = run(prep, "pdg")
    assert res["status"] == "SUCCESS"
    assert res["guidance_used"] == 0


def test_degenerate_reference_degrades():
    start = make_scene("T10").start_world
    prep = _prep_with_ref(np.repeat(start[None], 8, axis=0))
    assert prep.ref.valid is False
    assert prep.ref.reason in ("degenerate_length", "degenerate_after_dedup")
    assert run(prep, "pdg")["status"] == "SUCCESS"


def test_out_of_bounds_reference_degrades():
    prep = _prep_with_ref(np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 5.5]]))
    assert prep.ref.valid is False
    assert prep.ref.reason == "out_of_bounds"
    assert run(prep, "pdg")["status"] == "SUCCESS"


def test_fully_colliding_reference_is_still_usable():
    """A reference that lies entirely on obstacles is a soft guide, not a veto."""
    scene = make_scene("T2")
    wall = np.array([[40.0, 36.0], [40.0, 40.0], [40.0, 44.0]])
    scene_pts = scene.spec.cell_to_world(wall[:, ::-1])
    prep = prepare_case(case_from_scene(scene, ref_world=scene_pts),
                        robot_radius_m=0.0, safety_margin_m=0.0)
    assert prep.ref.valid is True
    res = run(prep, "pdg", w=1.2)
    assert res["status"] == "SUCCESS"
    assert res["path_valid"] == 1

