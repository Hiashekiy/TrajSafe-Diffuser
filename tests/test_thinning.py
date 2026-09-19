"""Tests for the migrated thinning implementation in ``src.geometry``.

The legacy top-level ``skeleton_graph`` package was removed; the V3 online
candidate search and the offline skeleton builder now use
``src.geometry.thinning`` through ``src.geometry.skeleton_graph``.
"""
from __future__ import annotations

import numpy as np

from src.geometry.skeleton_graph import build_skeleton_graph
from src.geometry.thinning import (compute_degree, guo_hall_thinning,
                                   topology_preserved)


def test_guo_hall_thinning_preserves_topology():
    occ = np.ones((40, 40), dtype=np.float32)
    occ[8:32, 8:32] = 0.0                      # one square free room
    free255 = (occ < 0.5).astype(np.uint8) * 255
    skeleton = guo_hall_thinning(free255, backend="numpy", verbose=False)
    assert skeleton.shape == occ.shape
    assert skeleton.any()
    assert compute_degree(skeleton).max() >= 1
    ok, details = topology_preserved(free255 > 0, skeleton > 0)
    assert ok is True
    assert details["reference"] == details["candidate"]


def test_build_skeleton_graph_uses_migrated_thinning():
    occ = np.ones((64, 64), dtype=np.float32)
    occ[8:56, 8:24] = 0.0                      # left room
    occ[8:56, 40:56] = 0.0                     # right room
    occ[28:36, 8:56] = 0.0                     # corridor
    graph = build_skeleton_graph(occ, safety_dilation_cells=1,
                                 thinning_backend="numpy")
    assert graph.skeleton.any()
    assert len(graph.nodes) > 0
