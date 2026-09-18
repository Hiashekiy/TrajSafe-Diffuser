"""V2 geometry tests: skeleton graph correctness and safe candidate topologies.

Covers the section-37 checklist items:

    test_skeleton_nodes_are_free
    test_diagonal_no_corner_cut
    test_candidate_start_goal_connected
    test_all_candidate_segments_collision_free
    test_candidate_diversity_filter
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.geometry import skeleton_paths as sp  # noqa: E402
from src.geometry.skeleton_graph import (  # noqa: E402
    build_skeleton_graph, free_mask_from_occupancy, load_graph_npz, pixel_degrees,
    save_graph_npz, supercover_is_free, supercover_pixels,
)

MAPS_DIR = os.path.join(REPO_ROOT, "data", "processed_scene_v1", "maps")

# ring: opposite corners -> two equally short topologies.  Corridor points are
# offset slightly inside the corridor so the anchoring is not degenerate.
RING_START = (-0.52, -0.45)
RING_GOAL = (0.52, 0.42)
UMAZE_START = (-0.6, -0.55)
UMAZE_GOAL = (0.37, -0.6)
ROOMS_START = (-0.55, 0.0)
ROOMS_GOAL = (0.58, 0.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ring_map(size=64, outer=(10, 54), inner=(20, 44)):
    """Free rectangular ring -> skeleton is a cycle with two topologies."""
    occ = np.ones((size, size), dtype=np.float32)
    occ[outer[0]:outer[1], outer[0]:outer[1]] = 0.0
    occ[inner[0]:inner[1], inner[0]:inner[1]] = 1.0
    return occ


def two_room_map(size=64):
    """Two rooms joined by two disjoint corridors."""
    occ = np.ones((size, size), dtype=np.float32)
    occ[8:56, 8:20] = 0.0            # left room
    occ[8:56, 44:56] = 0.0           # right room
    occ[12:20, 20:44] = 0.0          # upper corridor
    occ[44:52, 20:44] = 0.0          # lower corridor
    return occ


def make_route(scene, branch_ids, length):
    scene = np.asarray(scene, dtype=np.float64)
    return sp.SkeletonRoute(
        pixels=scene.copy(), scene=scene, length=float(length),
        node_path=(0, 1), branch_ids=tuple(branch_ids), start_node=0, goal_node=1,
    )


@pytest.fixture(scope="module")
def ring_graph():
    return build_skeleton_graph(ring_map(), safety_dilation_cells=1)


@pytest.fixture(scope="module")
def rooms_graph():
    return build_skeleton_graph(two_room_map(), safety_dilation_cells=1)


@pytest.fixture(scope="module")
def umaze_graph():
    path = os.path.join(MAPS_DIR, "umaze.npy")
    if not os.path.exists(path):
        pytest.skip("umaze map not available")
    return build_skeleton_graph(np.load(path), safety_dilation_cells=1)


# ---------------------------------------------------------------------------
# skeleton / graph
# ---------------------------------------------------------------------------


def test_skeleton_nodes_are_free(ring_graph, rooms_graph, umaze_graph):
    """Every skeleton / node / branch pixel lies in free space."""
    for graph in (ring_graph, rooms_graph, umaze_graph):
        assert graph.skeleton.sum() > 0
        assert graph.free[graph.skeleton].all()
        for node in graph.nodes:
            for x, y in node.pixels:
                assert graph.free[y, x], "node pixel outside free space"
        for br in graph.branches:
            for x, y in br.pixels:
                assert graph.free[y, x], "branch pixel outside free space"


def test_structural_clusters_grow_only_correctly(umaze_graph, rooms_graph):
    """Structure pixels never share a node across a forbidden diagonal."""
    for graph in (umaze_graph, rooms_graph):
        for node in graph.nodes:
            for (x0, y0) in node.pixels:
                for (x1, y1) in node.pixels:
                    if (x0, y0) == (x1, y1):
                        continue
                    dx, dy = x1 - x0, y1 - y0
                    ortho = (abs(dx) + abs(dy) == 1)
                    diag = (abs(dx) == 1 and abs(dy) == 1
                            and graph.free[y0 + dy, x0] and graph.free[y0, x0 + dx])
                    assert ortho or diag or len(node.pixels) > 2


def _diagonal_chain():
    skeleton = np.zeros((6, 6), dtype=bool)
    for k in (1, 2, 3):
        skeleton[k, k] = True       # (x=k, y=k)
    return skeleton


def test_diagonal_no_corner_cut():
    """A diagonal skeleton step through an obstacle corner is forbidden."""
    free = np.ones((6, 6), dtype=bool)
    free[1, 2] = False              # (x=2, y=1) blocks (1,1) -> (2,2)
    free[2, 1] = False              # (x=1, y=2) blocks (1,1) -> (2,2)
    free[2, 3] = False              # (x=3, y=2) blocks (2,2) -> (3,3)
    free[3, 2] = False              # (x=2, y=3) blocks (2,2) -> (3,3)
    skeleton = _diagonal_chain()
    occ = (~free).astype(np.float32)
    graph = build_skeleton_graph(occ, free=free, skeleton=skeleton)
    deg = pixel_degrees(skeleton, free)
    assert deg[1, 1] == 0 and deg[2, 2] == 0 and deg[3, 3] == 0
    assert len(graph.branches) == 0
    assert len(graph.nodes) == 3
    assert graph.stats["components"] == 3


def test_diagonal_allowed_when_sides_free():
    """The same diagonal steps are legal once the side cells are free."""
    free = np.ones((6, 6), dtype=bool)
    skeleton = _diagonal_chain()
    occ = (~free).astype(np.float32)
    graph = build_skeleton_graph(occ, free=free, skeleton=skeleton)
    deg = pixel_degrees(skeleton, free)
    assert deg[1, 1] == 1 and deg[2, 2] == 2 and deg[3, 3] == 1
    assert len(graph.branches) == 1
    assert len(graph.nodes) == 2
    assert graph.stats["components"] == 1


def test_every_branch_step_is_allowed(rooms_graph, umaze_graph):
    """No branch may contain a step that the no-corner-cut rule forbids."""
    for graph in (rooms_graph, umaze_graph):
        ortho = {(0, 1), (0, -1), (1, 0), (-1, 0)}
        for br in graph.branches:
            for (x0, y0), (x1, y1) in zip(br.pixels[:-1], br.pixels[1:]):
                dx, dy = x1 - x0, y1 - y0
                assert abs(dx) <= 1 and abs(dy) <= 1 and (dx or dy)
                if (dy, dx) not in ortho:
                    assert graph.free[y0 + dy, x0] and graph.free[y0, x0 + dx], (
                        "diagonal branch step cuts a corner"
                    )


def test_supercover_pixels_cover_sampled_segment():
    """Supercover cells are a superset of every densely sampled interior cell."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        p0 = rng.uniform(0, 40, size=2)
        p1 = rng.uniform(0, 40, size=2)
        cells = set(supercover_pixels(p0, p1))
        t = np.linspace(0.0, 1.0, 4001)
        pts = p0[None] * (1 - t)[:, None] + p1[None] * t[:, None]
        idx = np.floor(pts + 0.5).astype(int)
        for i, j in idx:
            assert (int(i), int(j)) in cells


def test_supercover_detects_blocked_diagonal():
    """A blocked corner cell in a diagonal gap must fail the line check."""
    free = np.ones((6, 6), dtype=bool)
    free[3, 3] = False
    assert not supercover_is_free((2.0, 2.0), (4.0, 4.0), free)
    free[3, 3] = True
    assert supercover_is_free((2.0, 2.0), (4.0, 4.0), free)


def test_graph_npz_roundtrip(rooms_graph, tmp_path):
    path = str(tmp_path / "g.npz")
    save_graph_npz(rooms_graph, path)
    back = load_graph_npz(path)
    assert len(back.nodes) == len(rooms_graph.nodes)
    assert len(back.branches) == len(rooms_graph.branches)
    for a, b in zip(rooms_graph.branches, back.branches):
        assert a.pixels == b.pixels and a.u == b.u and a.v == b.v
    assert np.array_equal(back.free, rooms_graph.free)
    assert np.array_equal(back.skeleton, rooms_graph.skeleton)


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def test_candidate_start_goal_connected(ring_graph, umaze_graph):
    """Every valid candidate starts at the start and ends at the goal."""
    for graph, start, goal in ((ring_graph, RING_START, RING_GOAL),
                               (umaze_graph, UMAZE_START, UMAZE_GOAL)):
        cands = sp.generate_candidates(graph, start, goal)
        assert cands.num_valid >= 1
        for k in cands.valid_index():
            coords = cands.coords[k]
            assert np.linalg.norm(coords[0] - np.asarray(start)) < 0.02
            assert np.linalg.norm(coords[-1] - np.asarray(goal)) < 0.02


def test_candidate_multiple_topologies(ring_graph):
    """Opposite corners of the ring yield two equally short topologies."""
    cands = sp.generate_candidates(ring_graph, RING_START, RING_GOAL)
    assert cands.num_valid == 2
    j = sp.branch_jaccard(cands.branch_ids[0], cands.branch_ids[1])
    assert j <= 0.75
    assert set(cands.branch_ids[0]) != set(cands.branch_ids[1])
    l0, l1 = float(cands.lengths[0]), float(cands.lengths[1])
    assert abs(l0 - l1) / min(l0, l1) < 0.2


def test_all_candidate_segments_collision_free(ring_graph, rooms_graph, umaze_graph):
    """Dense resample of every candidate: all samples must land in free space."""
    cases = [
        (ring_graph, RING_START, RING_GOAL),
        (rooms_graph, ROOMS_START, ROOMS_GOAL),
        (umaze_graph, UMAZE_START, UMAZE_GOAL),
    ]
    for graph, start, goal in cases:
        cands = sp.generate_candidates(graph, start, goal)
        assert cands.num_valid >= 1
        for k in cands.valid_index():
            dense = sp.resample_polyline(cands.coords[k], 512)
            px = graph.scene_to_pixel(dense)
            ii = np.rint(px[:, 0]).astype(int)
            jj = np.rint(px[:, 1]).astype(int)
            inside = ((ii >= 0) & (ii < graph.free.shape[1])
                      & (jj >= 0) & (jj < graph.free.shape[0]))
            assert inside.all()
            assert graph.free[jj, ii].all(), "candidate crosses an obstacle"


def test_candidate_diversity_filter(monkeypatch, ring_graph):
    """Near-duplicate topologies (Jaccard > threshold) collapse to one slot."""
    base = np.linspace([-0.5, -0.4], [0.5, 0.4], 20)
    routes = [
        make_route(base, (0, 1), 1.00),
        make_route(base + 1e-4, (0, 1, 2), 1.02),        # J = 2/3 -> dropped
        make_route(base + 2e-4, (3, 4), 1.10),           # disjoint -> kept
        make_route(base + 3e-4, (0, 1, 2, 5), 1.20),     # J = 2/4 -> dropped
    ]
    monkeypatch.setattr(sp, "routes_k_shortest", lambda *a, **k: list(routes))
    cfg = sp.CandidateConfig(num_candidates=4, raw_k=8, dedup_jaccard=0.4,
                             max_length_ratio=2.0)
    cands = sp.generate_candidates(ring_graph, RING_START, RING_GOAL, cfg)
    assert cands.num_valid == 2
    assert cands.branch_ids[0] == (0, 1)
    assert cands.branch_ids[1] == (3, 4)


def test_candidate_length_filter(monkeypatch, ring_graph):
    """Routes longer than max_length_ratio * L_min are dropped."""
    base = np.linspace([-0.5, -0.4], [0.5, 0.4], 20)
    routes = [
        make_route(base, (0,), 1.00),
        make_route(base, (1,), 1.40),      # within 1.5x
        make_route(base, (2,), 1.60),      # too long
    ]
    monkeypatch.setattr(sp, "routes_k_shortest", lambda *a, **k: list(routes))
    cfg = sp.CandidateConfig(num_candidates=4, raw_k=8, dedup_jaccard=0.0,
                             max_length_ratio=1.5)
    cands = sp.generate_candidates(ring_graph, RING_START, RING_GOAL, cfg)
    assert cands.num_valid == 2


def test_candidate_padding_and_mask(umaze_graph):
    """Invalid slots stay zeroed and masked out; shapes follow the config."""
    cfg = sp.CandidateConfig(num_candidates=4, candidate_points=32)
    cands = sp.generate_candidates(umaze_graph, UMAZE_START, UMAZE_GOAL, cfg)
    assert cands.paths.shape == (4, 32, 5)
    assert cands.coords.shape == (4, 32, 2)
    assert cands.mask.shape == (4,)
    invalid = ~cands.mask
    assert not cands.paths[invalid].any()
    assert not cands.coords[invalid].any()


def test_features_have_unit_tangent_and_progress(ring_graph):
    cands = sp.generate_candidates(ring_graph, RING_START, RING_GOAL)
    k = int(cands.valid_index()[0])
    feat = cands.paths[k]
    tan = feat[:, 2:4]
    assert np.allclose(np.linalg.norm(tan, axis=1), 1.0, atol=1e-4)
    u = feat[:, 4]
    assert abs(u[0]) < 1e-6 and abs(u[-1] - 1.0) < 1e-5
    assert np.all(np.diff(u) >= -1e-6)


def test_resample_polyline_is_uniform():
    # On a straight line, uniform arc length == uniform euclidean spacing.
    line = np.array([[0.0, 0.0], [3.0, 0.0]])
    out = sp.resample_polyline(line, 101)
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(seg, seg[0], atol=1e-9)
    # Across a corner the *arc length* stays uniform (chord length shrinks).
    bent = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    out = sp.resample_polyline(bent, 101)
    assert np.allclose(out[0], bent[0]) and np.allclose(out[-1], bent[-1])
    # project every sample back onto the polyline: the recovered normalized arc
    # length must be exactly the uniform grid
    s, dist = sp.nearest_arclength(out, bent)
    assert np.allclose(dist, 0.0, atol=1e-9)
    assert np.allclose(s, np.linspace(0.0, 1.0, 101), atol=1e-9)
    on_h = np.isclose(out[:, 1], 0.0)
    on_v = np.isclose(out[:, 0], 1.0)
    assert np.all(on_h | on_v)


def test_anchor_prefers_visible_node():
    """A node hidden behind a wall must not be used as an anchor."""
    occ = np.ones((64, 64), dtype=np.float32)
    occ[10:54, 10:20] = 0.0
    occ[10:54, 44:54] = 0.0
    graph = build_skeleton_graph(occ, safety_dilation_cells=1)
    anchored = graph.anchor_point((-0.6, 0.0), max_candidates=16)
    assert anchored is not None
    node = graph.nodes[anchored[0]]
    assert node.anchor[0] < 32


def test_topology_soft_target_is_a_distribution(ring_graph):
    cands = sp.generate_candidates(ring_graph, RING_START, RING_GOAL)
    gt = sp.resample_polyline(cands.coords[int(cands.valid_index()[0])], 128)
    q = sp.soft_topology_target(gt, cands, tau=0.05)
    assert q.shape == (cands.num_slots,)
    assert abs(q.sum() - 1.0) < 1e-9
    assert q[int(cands.valid_index()[0])] > 0.5
    assert np.all(q[~cands.mask] == 0.0)


def test_progress_target_is_monotone_and_pinned(ring_graph):
    cands = sp.generate_candidates(ring_graph, RING_START, RING_GOAL)
    k = int(cands.valid_index()[0])
    gt = cands.coords[k][::4]                     # 32 GT waypoints, exact path
    s = sp.progress_target(gt, cands.coords[k], 32)
    assert s[0] == 0.0 and s[-1] == 1.0
    assert np.all(np.diff(s) >= -1e-12)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_normalized_dtw_zero_for_identical():
    a = np.linspace([0, 0], [1, 1], 50)
    assert sp.normalized_dtw(a, a) < 1e-9
    b = a.copy()
    b[:, 1] += 0.1                     # perpendicular offset, not a shift along a
    c = a.copy()
    c[:, 1] += 0.2
    d1, d2 = sp.normalized_dtw(a, b), sp.normalized_dtw(a, c)
    assert 0.02 < d1 < 0.1
    assert d2 > d1                     # monotone in the perpendicular offset


def test_isotonic_projection_removes_backtracking():
    s = np.array([0.0, 0.3, 0.2, 0.6, 0.5, 1.0])
    out = sp.isotonic_nondecreasing(s)
    assert np.all(np.diff(out) >= -1e-12)
    assert abs(out.sum() - s.sum()) < 1e-9


def test_interpolate_path_matches_geometry():
    poly = np.array([[0.0, 0.0], [1.0, 0.0]])
    pts = sp.interpolate_path(poly, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(pts, [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])


def test_free_mask_dilation_margin():
    occ = np.zeros((32, 32), dtype=np.float32)
    occ[16, 16] = 1.0
    free1 = free_mask_from_occupancy(occ, 1)
    assert not free1[15, 15] and not free1[16, 16]
    assert free1[14, 14]
    free0 = free_mask_from_occupancy(occ, 0)
    assert not free0[16, 16]
