"""V3 geometry tests: the four fixes required before training (spec section 8).

    1. parallel branches survive the graph expansion
    2. start/goal attach to EVERY visible anchor (super source / sink)
    3. the unsafe in-cluster fallback is gone (candidate dropped instead)
    4. gamma_m(s) runs on the dense safe cell-chain, not on the 128-point
       network feature path
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.geometry import skeleton_paths as sp  # noqa: E402
from src.geometry.skeleton_graph import (  # noqa: E402
    SkeletonBranch, SkeletonGraph, SkeletonNode, build_skeleton_graph,
)


def _handmade_graph():
    """Two junctions joined by TWO parallel branches of equal length."""
    free = np.ones((6, 6), dtype=bool)
    skeleton = np.zeros((6, 6), dtype=bool)
    for x, y in ((1, 1), (4, 1), (2, 1), (3, 1), (2, 2), (3, 2)):
        skeleton[y, x] = True
    nodes = [
        SkeletonNode(idx=0, kind="junction", pixels=[(1, 1)], center=(1.0, 1.0),
                     anchor=(1, 1)),
        SkeletonNode(idx=1, kind="junction", pixels=[(4, 1)], center=(4.0, 1.0),
                     anchor=(4, 1)),
    ]
    branches = [
        SkeletonBranch(idx=0, u=0, v=1, pixels=[(1, 1), (2, 1), (3, 1), (4, 1)],
                       length=3.0),
        SkeletonBranch(idx=1, u=0, v=1, pixels=[(1, 1), (2, 2), (3, 2), (4, 1)],
                       length=3.0),
    ]
    node_of_pixel = np.full((6, 6), -1, dtype=np.int64)
    node_of_pixel[1, 1] = 0
    node_of_pixel[1, 4] = 1
    return SkeletonGraph(res=6, free=free, skeleton=skeleton, nodes=nodes,
                         branches=branches, node_of_pixel=node_of_pixel)


def test_parallel_branches_are_not_swallowed():
    """Two branches between the same junctions must stay two topologies."""
    graph = _handmade_graph()
    g, src, sink = graph.expanded_graph([(0, [(1, 1)], 0.0)], [(1, [(4, 1)], 0.0)])
    decoded = [graph.decode_expanded_path(p)
               for p in nx.shortest_simple_paths(g, src, sink, weight="weight")]
    branch_sets = {tuple(b) for _j, b in decoded}
    assert branch_sets == {(0,), (1,)}, branch_sets
    assert len(g.nodes) == len(graph.nodes) + len(graph.branches) + 2


def test_branch_expansion_keeps_self_consistency():
    """Node paths and branch sequences of a decoded path must agree."""
    graph = _handmade_graph()
    g, src, sink = graph.expanded_graph([(0, [(1, 1)], 0.0)], [(1, [(4, 1)], 0.0)])
    for path in nx.shortest_simple_paths(g, src, sink, weight="weight"):
        junctions, branches = graph.decode_expanded_path(path)
        assert len(branches) == max(0, len(junctions) - 1)
        assert junctions[0] == 0 and junctions[-1] == 1
        for j, bid in enumerate(branches):
            br = graph.branches[bid]
            assert {br.u, br.v} == {junctions[j], junctions[j + 1]}


def _three_corridor_map(size=96):
    """Two rooms joined by three parallel corridors -> parallel branches."""
    occ = np.ones((size, size), dtype=np.float32)
    occ[8:88, 8:24] = 0.0            # left room
    occ[8:88, 72:88] = 0.0           # right room
    for y0 in (20, 44, 68):
        occ[y0:y0 + 8, 24:72] = 0.0  # three parallel corridors
    return occ


@pytest.fixture(scope="module")
def corridors_graph():
    return build_skeleton_graph(_three_corridor_map(), safety_dilation_cells=1)


def test_multi_anchor_start_goal_produces_several_topologies(corridors_graph):
    """Every visible anchor competes, so parallel corridors become candidates."""
    # the three corridors have very different lengths, so the production
    # max_length_ratio=1.5 would (correctly) drop two of them; relax it here
    # because this test is about the graph expansion, not about the filter
    # NOTE raw_k must be generous: with a super source/sink the K lightest
    # simple paths are dominated by short anchor-pair variations, so a small
    # budget never reaches the far corridor.  This is why the V3 config raises
    # raw_k from 16 to 64.
    # NOTE 32 slots on purpose: with a super source/sink the K shortest paths are
    # dominated by short anchor-pair variations (branch set {} / {5} / {8} ...),
    # so a small slot budget can fill up before the far corridor is reached.
    # The candidate space still CONTAINS all three corridors, which is what this
    # test asserts; the slot budget is a separate tuning question.
    cfg = sp.CandidateConfig(num_candidates=32, raw_k=256, max_length_ratio=4.0,
                             dedup_jaccard=0.9)
    cands = sp.generate_candidates(corridors_graph, (-0.7, 0.0), (0.7, 0.0), cfg)
    assert cands.num_valid >= 3, (cands.branch_ids, len(corridors_graph.branches))
    assert len({frozenset(b) for b in cands.branch_ids}) >= 3
    used = set()
    for b in cands.branch_ids:
        used.update(b)
    corridors = {2, 5, 8}                     # the three parallel corridors
    assert corridors.issubset(used), (corridors - used, cands.branch_ids)


def test_visible_anchors_returns_every_reachable_node(corridors_graph):
    anchors = corridors_graph.visible_anchors((-0.7, 0.0), max_candidates=16)
    assert len(anchors) >= 2
    nearest = corridors_graph.anchor_point((-0.7, 0.0), max_candidates=16)
    assert nearest is not None and anchors[0][0] == nearest[0]


def test_in_cluster_path_returns_none_instead_of_unsafe_fallback():
    """A corner-cut separated cluster pair must fail, never fall back to [a, b]."""
    free = np.ones((6, 6), dtype=bool)
    free[2, 3] = False          # blocks the diagonal step (2,2)->(3,3)
    free[3, 2] = False
    skeleton = np.zeros((6, 6), dtype=bool)
    skeleton[2, 2] = True
    skeleton[3, 3] = True
    nodes = [SkeletonNode(idx=0, kind="junction", pixels=[(2, 2), (3, 3)],
                          center=(2.5, 2.5), anchor=(2, 2))]
    node_of_pixel = np.full((6, 6), -1, dtype=np.int64)
    node_of_pixel[2, 2] = 0
    node_of_pixel[3, 3] = 0
    graph = SkeletonGraph(res=6, free=free, skeleton=skeleton, nodes=nodes,
                          branches=[], node_of_pixel=node_of_pixel)
    assert graph.in_cluster_path(0, (2, 2), (3, 3)) is None
    assert graph.in_cluster_path(0, (2, 2), (2, 2)) == [(2, 2)]


def test_candidate_geometry_is_the_dense_safe_chain(corridors_graph):
    """candidate_geometry is the dense cell chain, not the 128-point resample."""
    cands = sp.generate_candidates(corridors_graph, (-0.7, 0.0), (0.7, 0.0))
    assert len(cands.geometry) == cands.num_valid
    for k in cands.valid_index():
        dense = cands.geometry[k]
        assert len(dense) > 20
        px = np.rint(corridors_graph.scene_to_pixel(dense)).astype(int)
        assert corridors_graph.free[px[:, 1], px[:, 0]].all()
        feature = cands.coords[k]
        n = min(len(dense), len(feature))
        assert not np.allclose(dense[:n], feature[:n])


def test_gamma_uses_the_dense_geometry(corridors_graph):
    """gamma_m(s) on the dense chain stays free for a fine grid of s."""
    cands = sp.generate_candidates(corridors_graph, (-0.7, 0.0), (0.7, 0.0))
    s = np.linspace(0.0, 1.0, 501)
    for k in cands.valid_index():
        dense = cands.geometry[k]
        pts = sp.interpolate_path(dense, s)
        px = np.rint(corridors_graph.scene_to_pixel(pts)).astype(int)
        assert corridors_graph.free[px[:, 1], px[:, 0]].all(), "gamma left free space"
        _, dist = sp.nearest_arclength(pts, dense)
        assert dist.max() < 1e-5


def test_invalid_slots_carry_no_geometry(corridors_graph):
    cands = sp.generate_candidates(corridors_graph, (-0.7, 0.0), (0.7, 0.0),
                                   sp.CandidateConfig(num_candidates=8))
    assert len(cands.geometry) == cands.num_valid
    assert int(cands.mask.sum()) == len(cands.geometry)
    assert cands.geometry_lengths.shape[0] == cands.num_slots
    assert bool((cands.geometry_lengths[~cands.mask] == 0).all())
