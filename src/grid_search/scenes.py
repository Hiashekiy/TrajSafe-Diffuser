"""Synthetic acceptance scenes T1-T12 (design section 10).

Every scene is a real ``GridSpec`` + occupancy bitmap + start/goal + a reference
polyline that plays the role of the decoded diffusion ``x0``.  The same objects
are used by the unit tests and by the visualisation of the synthetic suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_map import GridSpec

__all__ = ["Scene", "make_scene", "scene_catalog", "SCENE_IDS"]

RES = 80
CELL_M = 1.0


@dataclass
class Scene:
    ident: str
    name: str
    occupancy: np.ndarray
    spec: GridSpec
    start_rc: tuple
    goal_rc: tuple
    ref_rc: np.ndarray = None      # [n, 2] (row, col) continuous
    notes: str = ""
    expect: str = "path"           # path | no_path | invalid_start | invalid_goal | same

    @property
    def start_world(self) -> np.ndarray:
        return self.spec.cell_to_world(np.array([self.start_rc[1],
                                                 self.start_rc[0]], float))

    @property
    def goal_world(self) -> np.ndarray:
        return self.spec.cell_to_world(np.array([self.goal_rc[1],
                                                  self.goal_rc[0]], float))

    @property
    def ref_world(self) -> np.ndarray:
        if self.ref_rc is None:
            return None
        rc = np.asarray(self.ref_rc, dtype=np.float64)
        return self.spec.cell_to_world(rc[:, ::-1])

    @property
    def start_idx(self) -> int:
        return int(self.start_rc[0]) * self.spec.width + int(self.start_rc[1])

    @property
    def goal_idx(self) -> int:
        return int(self.goal_rc[0]) * self.spec.width + int(self.goal_rc[1])


def _blank(r=RES, c=RES) -> np.ndarray:
    return np.zeros((r, c), dtype=np.uint8)


def _rect(occ, r0, r1, c0, c1, value=1):
    occ[r0:r1, c0:c1] = value
    return occ


def _scene(ident, name, occ, start_rc, goal_rc, ref_rc=None, notes="",
           expect="path", cell_m=CELL_M) -> Scene:
    h, w = occ.shape
    spec = GridSpec(h, w, cell_m, scene_to_meter=cell_m * max(h, w) / 2.0)
    return Scene(ident, name, occ, spec, tuple(start_rc), tuple(goal_rc),
                 None if ref_rc is None else np.asarray(ref_rc, dtype=np.float64),
                 notes, expect)


def t1_empty() -> Scene:
    occ = _blank()
    ref = np.array([[60, 10], [60, 70]], dtype=np.float64)
    return _scene("T1", "empty_map", occ, (60, 10), (60, 70), ref,
                  "no obstacle; straight reference")


def t2_wall() -> Scene:
    occ = _blank()
    occ[:, 40] = 1
    occ[70:, 40] = 0                    # gap at the bottom
    ref = np.array([[40, 10], [40, 70]], dtype=np.float64)   # straight through
    return _scene("T2", "single_wall", occ, (40, 10), (40, 70), ref,
                  "reference crosses the wall, gap at rows 70..79")


def t3_u_obstacle() -> Scene:
    occ = _blank()
    occ[20, 30:51] = 1                  # U top
    occ[20:61, 30] = 1                  # U left
    occ[20:61, 50] = 1                  # U right
    # opening at the bottom (row 60)
    ref = np.array([[45, 40], [10, 10]], dtype=np.float64)   # straight through top
    return _scene("T3", "u_obstacle", occ, (45, 40), (10, 10), ref,
                  "start inside a U open at the bottom; reference exits the top")


def t4_two_routes() -> Scene:
    occ = _blank()
    _rect(occ, 25, 56, 25, 56)
    ref = np.array([[8, 8], [8, 70], [40, 70]], dtype=np.float64)
    return _scene("T4", "two_channels", occ, (40, 5), (40, 75), ref,
                  "block in the middle: routes above and below; reference above")


def t5_self_intersection() -> Scene:
    occ = _blank()
    _rect(occ, 20, 61, 20, 24)
    _rect(occ, 20, 61, 56, 60)
    ref = np.array([[10, 10], [70, 70], [70, 10], [10, 70]], dtype=np.float64)
    return _scene("T5", "self_intersection", occ, (10, 10), (10, 70), ref,
                  "figure-eight reference crossing itself near (40,40)")


def t6_parallel_legs() -> Scene:
    occ = _blank()
    occ[10:70, 45] = 1
    occ[10:70, 55] = 1
    ref = np.array([[40, 10], [40, 70], [52, 70], [52, 10], [40, 10],
                    [40, 70]], dtype=np.float64)
    return _scene("T6", "parallel_legs", occ, (40, 10), (40, 70), ref,
                  "two near-parallel legs with opposite progress")


def t7_backtrack() -> Scene:
    occ = _blank()
    _rect(occ, 30, 50, 30, 50)
    ref = np.array([[20, 5], [20, 75], [20, 5], [20, 75]], dtype=np.float64)
    return _scene("T7", "reference_backtrack", occ, (20, 5), (20, 75), ref,
                  "reference runs forward and back")


def t8_no_path() -> Scene:
    occ = _blank()
    occ[:, 40] = 1
    ref = np.array([[40, 10], [40, 70]], dtype=np.float64)
    return _scene("T8", "blocked", occ, (40, 10), (40, 70), ref,
                  "full-height wall: NO_PATH, never through the wall",
                  expect="no_path")


def t9_start_in_obstacle() -> Scene:
    occ = _blank()
    _rect(occ, 36, 44, 36, 44)
    ref = np.array([[40, 40], [10, 10]], dtype=np.float64)
    return _scene("T9a", "start_in_obstacle", occ, (40, 40), (10, 10), ref,
                  "start inside the block", expect="invalid_start")


def t9b_goal_in_obstacle() -> Scene:
    occ = _blank()
    _rect(occ, 36, 44, 36, 44)
    ref = np.array([[10, 10], [40, 40]], dtype=np.float64)
    return _scene("T9b", "goal_in_obstacle", occ, (10, 10), (40, 40), ref,
                  "goal inside the block", expect="invalid_goal")


def t10_invalid_reference() -> Scene:
    occ = _blank()
    ref = np.array([[np.nan, 0.0], [10.0, 10.0]], dtype=np.float64)
    return _scene("T10", "invalid_reference", occ, (40, 10), (40, 70), ref,
                  "NaN reference must fall back to unguided Focal A*")


def t11_corner_cutting() -> Scene:
    occ = _blank()
    # two obstacles touching only at a corner (40,40) / (41,41)
    occ[35:41, 35:41] = 1
    occ[41:47, 41:47] = 1
    ref = np.array([[40, 42], [41, 40]], dtype=np.float64)
    return _scene("T11", "corner_cutting", occ, (40, 42), (41, 40), ref,
                  "diagonal contact point: must not slip through")


def t12_same_endpoints() -> Scene:
    occ = _blank()
    ref = np.array([[30, 30], [31, 31]], dtype=np.float64)
    return _scene("T12", "same_endpoints", occ, (30, 30), (30, 30), ref,
                  "start == goal -> single point, cost 0", expect="same")


_MAKERS = {
    "T1": t1_empty, "T2": t2_wall, "T3": t3_u_obstacle, "T4": t4_two_routes,
    "T5": t5_self_intersection, "T6": t6_parallel_legs, "T7": t7_backtrack,
    "T8": t8_no_path, "T9": t9_start_in_obstacle,
    "T9b": t9b_goal_in_obstacle, "T10": t10_invalid_reference,
    "T11": t11_corner_cutting, "T12": t12_same_endpoints,
}
SCENE_IDS = tuple(_MAKERS)


def make_scene(ident: str) -> Scene:
    return _MAKERS[ident]()


def scene_catalog():
    return [make_scene(k) for k in SCENE_IDS]
