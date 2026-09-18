"""Stage-1 pipeline tests: thinning, pruning, graph extraction, IO.

These target the parts that are easy to get subtly wrong:

* junction clusters must merge into a single node;
* a pure cycle must survive as a cycle (it has no endpoint and no junction);
* pruning must remove hairs but never touch junction-junction edges;
* every chain pixel must land on exactly one edge, polyline included;
* ``graph.json`` must rebuild the identical graph.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skeleton_graph import graph_extractor as ge  # noqa: E402
from skeleton_graph import pruning, thinning  # noqa: E402
from skeleton_graph.map_loader import FREE, OBSTACLE, load_map, map_stats  # noqa: E402

MAZE_MAP = os.path.join(
    REPO_ROOT, "data", "processed_scene_v1", "maps", "umaze.npy"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def blank(height: int = 96, width: int = 96) -> np.ndarray:
    return np.full((height, width), OBSTACLE, dtype=np.uint8)


def rect(occupancy, y0, y1, x0, x1, value=FREE):
    occupancy[y0:y1, x0:x1] = value
    return occupancy


def circle_skeleton(radius: int = 20) -> np.ndarray:
    """A one-pixel-wide 8-connected ring, built directly (no thinning)."""
    size = 2 * radius + 5
    skeleton = np.zeros((size, size), dtype=np.uint8)
    centre = size // 2
    x, y, err = radius, 0, 0
    points = set()
    while x >= y:
        for py, px in (
            (centre + y, centre + x), (centre + y, centre - x),
            (centre - y, centre + x), (centre - y, centre - x),
            (centre + x, centre + y), (centre + x, centre - y),
            (centre - x, centre + y), (centre - x, centre - y),
        ):
            points.add((py, px))
        y += 1
        if err <= 0:
            err += 2 * y + 1
        if err > 0:
            x -= 1
            err -= 2 * x + 1
    for py, px in points:
        skeleton[py, px] = 1
    return skeleton


def plus_skeleton(half: int = 15, size: int = 61) -> np.ndarray:
    """A plus-shaped skeleton: one junction pixel, four endpoints."""
    skeleton = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    skeleton[mid, mid - half : mid + half + 1] = 1
    skeleton[mid - half : mid + half + 1, mid] = 1
    return skeleton


# ---------------------------------------------------------------------------
# thinning
# ---------------------------------------------------------------------------


def test_thinning_produces_one_pixel_wide_skeleton():
    occupancy = rect(blank(64, 96), 26, 38, 4, 92)
    skeleton = thinning.guo_hall_thinning(occupancy, "numpy", verbose=False)
    diagnostics = thinning.skeleton_diagnostics(skeleton)

    assert diagnostics["pixels"] > 0
    assert diagnostics["thick_2x2_blocks"] == 0, "skeleton is not one pixel wide"


def test_thinning_preserves_topology_of_free_space():
    # A hollow box: the free space encloses exactly one hole.
    occupancy = blank(80, 80)
    rect(occupancy, 10, 70, 10, 70)
    rect(occupancy, 25, 55, 25, 55, value=OBSTACLE)

    skeleton = thinning.guo_hall_thinning(occupancy, "numpy", verbose=False)
    result = thinning.validate_thinning(occupancy > 0, skeleton)

    assert result["ok"], result
    assert result["reference"]["holes"] == 1
    assert result["candidate"]["holes"] == 1


def test_thinning_rejects_empty_free_space():
    with pytest.raises(ValueError):
        thinning.guo_hall_thinning(blank(16, 16), "numpy", verbose=False)


# ---------------------------------------------------------------------------
# pure cycles
# ---------------------------------------------------------------------------


def test_circle_skeleton_is_a_pure_cycle():
    skeleton = circle_skeleton(20)
    degree = thinning.compute_degree(skeleton, 8)
    assert set(np.unique(degree[skeleton > 0]).tolist()) == {2}


def test_pure_cycle_gets_two_auxiliary_nodes():
    graph, info = ge.skeleton_to_graph(
        circle_skeleton(20), 8, 2, "auto", verbose=False
    )
    summary = ge.graph_summary(graph)

    assert summary["node_type_counts"]["auxiliary"] == 2
    assert summary["node_type_counts"]["junction"] == 0
    assert summary["node_type_counts"]["endpoint"] == 0
    assert summary["edges"] == 2, "the two arcs of the loop must both survive"
    assert summary["cycle_rank"] == 1
    assert summary["has_cycle"]
    assert summary["self_loops"] == 0
    assert len(info["auxiliary"]["pure_cycles"]) == 1


def test_pure_cycle_single_auxiliary_node_is_a_self_loop():
    graph, _ = ge.skeleton_to_graph(
        circle_skeleton(20), 8, 1, "auto", verbose=False
    )
    summary = ge.graph_summary(graph)

    assert summary["node_type_counts"]["auxiliary"] == 1
    assert summary["edges"] == 1
    assert summary["self_loops"] == 1
    assert summary["cycle_rank"] == 1, "a self-loop still counts as a cycle"


def test_ring_shaped_free_space_keeps_its_cycle():
    """End-to-end: annulus map -> skeleton -> graph keeps the loop."""
    height = width = 101
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt((yy - 50.0) ** 2 + (xx - 50.0) ** 2)
    occupancy = np.where((radius > 22) & (radius < 34), FREE, OBSTACLE).astype(
        np.uint8
    )

    skeleton = thinning.guo_hall_thinning(occupancy, "numpy", verbose=False)
    skeleton, _ = pruning.prune_skeleton(skeleton, 10.0, verbose=False)
    graph, info = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)
    summary = ge.graph_summary(graph)
    validation = ge.validate_graph(skeleton, graph, 8, info["non_chain_pixels"])

    assert summary["cycle_rank"] >= 1
    assert validation["cycles_preserved"], validation


# ---------------------------------------------------------------------------
# junction clustering
# ---------------------------------------------------------------------------


def test_plus_skeleton_has_one_junction_and_four_endpoints():
    graph, _ = ge.skeleton_to_graph(plus_skeleton(), 8, 2, "auto", verbose=False)
    summary = ge.graph_summary(graph)

    assert summary["node_type_counts"]["junction"] == 1
    assert summary["node_type_counts"]["endpoint"] == 4
    assert summary["edges"] == 4
    assert summary["cycle_rank"] == 0


def test_adjacent_junction_pixels_merge_into_one_node():
    """A wide crossing must not become several overlapping nodes."""
    size = 61
    skeleton = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    skeleton[mid, 5 : size - 5] = 1          # horizontal bar
    skeleton[5 : size - 5, mid - 1] = 1      # two parallel vertical bars
    skeleton[5 : size - 5, mid + 1] = 1

    degree = thinning.compute_degree(skeleton, 8)
    raw_junction_pixels = int((skeleton & (degree >= 3)).sum())
    assert raw_junction_pixels >= 3, "test fixture must have a junction cluster"

    graph, info = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)
    summary = ge.graph_summary(graph)

    assert summary["node_type_counts"]["junction"] == 1, (
        "all adjacent junction pixels must collapse into a single node"
    )
    assert info["junction_cluster_sizes"][0] >= 3


def test_no_spurious_self_loops_on_a_wide_crossing():
    size = 61
    skeleton = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    skeleton[mid, 5 : size - 5] = 1
    skeleton[5 : size - 5, mid - 1] = 1
    skeleton[5 : size - 5, mid + 1] = 1

    graph, _ = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)
    assert ge.graph_summary(graph)["self_loops"] == 0


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------


def _backbone_with_hair(hair_length: int = 12):
    """Horizontal backbone (row 40, x 5..114) plus a vertical hair at x=60.

    ``hair_length`` counts the hair pixels *including* the one that sits
    directly on top of the backbone, so the traced chain is one pixel shorter:
    in an 8-connected skeleton a T-crossing always spans 2-3 junction pixels.
    """
    skeleton = np.zeros((80, 120), dtype=np.uint8)
    skeleton[40, 5:115] = 1
    skeleton[40 - hair_length : 40, 60] = 1
    return skeleton


def test_pruning_removes_a_short_hair_and_keeps_the_backbone():
    skeleton = _backbone_with_hair(12)
    pruned, info = pruning.prune_skeleton(
        skeleton, 20.0, max_prune_fraction=1.0, verbose=False
    )

    assert info["branches_removed"] == 1
    assert info["pixels_removed"] == 11, "11 chain pixels above the junction"
    assert pruned[40, 5:115].all(), "the backbone must be untouched"
    assert not pruned[28:39, 60].any(), "the hair chain must be gone"

    graph, _ = ge.skeleton_to_graph(pruned, 8, 2, "auto", verbose=False)
    summary = ge.graph_summary(graph)
    assert summary["node_type_counts"]["endpoint"] == 2, "both backbone ends survive"
    # What is left of the T-crossing has only two incident edges, so it is a
    # corner rather than a junction and must be dissolved back into one edge.
    assert summary["node_type_counts"]["junction"] == 0
    assert summary["edges"] == 1
    assert summary["connected_components"] == 1

    only_edge = next(ge.iter_edges(graph))[2]
    xs = sorted({x for x, _y in only_edge["pixels"]})
    assert xs[0] == 5 and xs[-1] == 114, "the edge spans the whole backbone"


def test_pruning_removes_short_arms_of_a_plus():
    # The plus crossing is 5 adjacent junction pixels, not one, so each traced
    # arm is 14 chain pixels rather than 15.
    skeleton = plus_skeleton(half=15, size=61)
    pruned, info = pruning.prune_skeleton(
        skeleton, 20.0, max_prune_fraction=1.0, verbose=False
    )

    assert info["branches_removed"] == 4
    assert info["pixels_removed"] == 56
    assert int(pruned.sum()) == 5, "only the junction cluster is left"

    graph, graph_info = ge.skeleton_to_graph(pruned, 8, 2, "auto", verbose=False)
    summary = ge.graph_summary(graph)
    assert graph_info["junction_cluster_sizes"] == [5]
    assert summary["node_type_counts"]["junction"] == 1
    assert summary["edges"] == 0


def test_pruning_keeps_branches_above_the_threshold():
    skeleton = _backbone_with_hair(12)
    pruned, info = pruning.prune_skeleton(skeleton, 5.0, verbose=False)

    assert info["branches_removed"] == 0
    assert np.array_equal(pruned > 0, skeleton > 0)


def test_pruning_never_removes_a_junction_junction_edge():
    """A short link between two junctions is a real corridor, not a hair."""
    skeleton = np.zeros((80, 120), dtype=np.uint8)
    skeleton[40, 5:115] = 1          # backbone
    for column in (45, 55):          # two junctions 10 px apart
        skeleton[10:40, column] = 1  # up arms
        skeleton[41:70, column] = 1  # down arms

    pruned, info = pruning.prune_skeleton(skeleton, 30.0, verbose=False)

    assert info["branches_removed"] == 4, "only the four 30 px arms may go"
    assert pruned[40, 45:56].all(), "the short junction-junction link must stay"


def test_pruning_never_removes_an_isolated_path_component():
    """A component with two endpoints and no junction is real free space."""
    skeleton = np.zeros((40, 40), dtype=np.uint8)
    skeleton[20, 5:15] = 1  # length 10, no junction anywhere

    pruned, info = pruning.prune_skeleton(skeleton, 50.0, verbose=False)

    assert info["branches_removed"] == 0
    assert info["isolated_paths_kept"] == 2
    assert np.array_equal(pruned > 0, skeleton > 0)


def test_pruning_budget_guard_stops_catastrophic_deletion():
    skeleton = plus_skeleton(half=20, size=81)
    total = int(skeleton.sum())

    pruned, info = pruning.prune_skeleton(
        skeleton, 10_000.0, max_prune_fraction=0.5, verbose=False
    )

    assert info["budget_exhausted"] is True
    assert info["pixels_removed"] <= 0.5 * total
    assert pruned.any(), "the guard must not let the skeleton disappear"


def test_corner_cluster_is_dissolved_into_a_single_edge():
    """A 90 degree bend must not become a fake junction node."""
    occupancy = blank(60, 80)
    rect(occupancy, 35, 47, 30, 70)  # horizontal arm
    rect(occupancy, 10, 47, 30, 42)  # vertical arm -> an L, i.e. one bend

    skeleton = thinning.guo_hall_thinning(occupancy, "numpy", verbose=False)
    junction_pixels = int(
        ((skeleton > 0) & (thinning.compute_degree(skeleton, 8) >= 3)).sum()
    )
    assert junction_pixels >= 1, "the bend must produce junction-ish pixels"

    graph, graph_info = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)
    summary = ge.graph_summary(graph)
    assert len(graph_info["dissolved_junctions"]) == 1
    assert summary["node_type_counts"]["junction"] == 0
    assert summary["node_type_counts"]["endpoint"] == 2
    assert summary["edges"] == 1, "the bend must become one continuous edge"

    kept_graph, _ = ge.skeleton_to_graph(
        skeleton, 8, 2, "auto", dissolve_degree2_junctions=False, verbose=False
    )
    kept = ge.graph_summary(kept_graph)
    assert kept["nodes"] > summary["nodes"], "dissolving must remove nodes"
    assert kept["cycle_rank"] == summary["cycle_rank"], "topology must not change"

    # The merged polyline must stay 8-connected and on the skeleton.
    merged = next(ge.iter_edges(graph))[2]["pixels"]
    assert merged[0][1] != merged[-1][1] and merged[0][0] != merged[-1][0]
    for (x0, y0), (x1, y1) in zip(merged, merged[1:]):
        assert max(abs(x1 - x0), abs(y1 - y0)) == 1


def test_dissolving_preserves_cycle_rank_on_maze_map():
    if not os.path.isfile(MAZE_MAP):
        pytest.skip(f"missing test map {MAZE_MAP}")

    occupancy = load_map(MAZE_MAP, verbose=False)
    skeleton = thinning.guo_hall_thinning(occupancy, "numpy", verbose=False)
    skeleton, _ = pruning.prune_skeleton(skeleton, 10.0, verbose=False)

    dissolved, _ = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)
    kept, _ = ge.skeleton_to_graph(
        skeleton, 8, 2, "auto", dissolve_degree2_junctions=False, verbose=False
    )

    assert (
        ge.graph_summary(dissolved)["cycle_rank"]
        == ge.graph_summary(kept)["cycle_rank"]
    )


# ---------------------------------------------------------------------------
# graph extraction
# ---------------------------------------------------------------------------


def test_every_chain_pixel_belongs_to_exactly_one_edge():
    skeleton = plus_skeleton()
    graph, info = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)
    validation = ge.validate_graph(skeleton, graph, 8, info["non_chain_pixels"])

    assert validation["ok"], validation
    assert validation["chain_pixels_covered_twice"] == 0


def test_edge_polylines_are_connected_and_on_the_skeleton():
    skeleton = plus_skeleton()
    graph, _ = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)

    for _u, _v, data in ge.iter_edges(graph):
        pixels = [(y, x) for x, y in data["pixels"]]
        assert len(pixels) >= 2
        assert pixels[0] in set(map(tuple, np.argwhere(skeleton > 0)))
        for (y0, x0), (y1, x1) in zip(pixels, pixels[1:]):
            assert max(abs(y1 - y0), abs(x1 - x0)) == 1, "polyline must be 8-connected"
            assert skeleton[y1, x1], "polyline must follow the skeleton"


def test_edge_length_matches_its_polyline():
    skeleton = _backbone_with_hair(12)
    graph, _ = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)

    for _u, _v, data in ge.iter_edges(graph):
        expected = ge.polyline_length([(y, x) for x, y in data["pixels"]])
        assert data["length"] == pytest.approx(expected, abs=1e-6)


def test_maze_map_graph_is_valid_end_to_end():
    if not os.path.isfile(MAZE_MAP):
        pytest.skip(f"missing test map {MAZE_MAP}")

    occupancy = load_map(MAZE_MAP, verbose=False)
    skeleton = thinning.guo_hall_thinning(occupancy, "numpy", verbose=False)
    skeleton, _ = pruning.prune_skeleton(skeleton, 10.0, verbose=False)
    graph, info = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)
    validation = ge.validate_graph(skeleton, graph, 8, info["non_chain_pixels"])
    summary = ge.graph_summary(graph)

    assert validation["ok"], validation
    assert summary["connected_components"] == 1
    assert summary["node_type_counts"]["endpoint"] >= 1


# ---------------------------------------------------------------------------
# graph.json round trip
# ---------------------------------------------------------------------------


def test_graph_json_round_trip(tmp_path):
    skeleton = _backbone_with_hair(12)
    graph, _ = ge.skeleton_to_graph(skeleton, 8, 2, "auto", verbose=False)

    path = os.path.join(tmp_path, "graph.json")
    ge.graph_to_json(graph, path, verbose=False)
    reloaded = ge.load_graph_json(path)

    ok, details = ge.graphs_equal(graph, reloaded)
    assert ok, details

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert set(payload) == {"nodes", "edges"}
    assert all(set(node) == {"id", "x", "y", "type"} for node in payload["nodes"])
    assert all(
        set(edge) == {"source", "target", "length", "pixels"}
        for edge in payload["edges"]
    )
    assert all(edge["pixels"] for edge in payload["edges"])


def test_graph_json_round_trip_preserves_a_cycle(tmp_path):
    graph, _ = ge.skeleton_to_graph(
        circle_skeleton(20), 8, 2, "auto", verbose=False
    )
    path = os.path.join(tmp_path, "graph.json")
    ge.graph_to_json(graph, path, verbose=False)
    reloaded = ge.load_graph_json(path)

    ok, details = ge.graphs_equal(graph, reloaded)
    assert ok, details
    assert ge.graph_summary(reloaded)["cycle_rank"] == 1


# ---------------------------------------------------------------------------
# map loader
# ---------------------------------------------------------------------------


def test_npy_map_uses_one_is_obstacle_convention(tmp_path):
    raw = np.zeros((16, 16), dtype=np.float32)
    raw[0, :] = 1.0
    path = os.path.join(tmp_path, "map.npy")
    np.save(path, raw)

    occupancy = load_map(path, verbose=False)

    assert occupancy[0, 0] == OBSTACLE
    assert occupancy[8, 8] == FREE


def test_image_polarity_can_be_forced(tmp_path):
    from PIL import Image

    # black = obstacle frame, white = free inside -> the natural convention
    image = np.full((24, 24), 255, dtype=np.uint8)
    image[0, :] = image[-1, :] = 0
    image[:, 0] = image[:, -1] = 0
    path = os.path.join(tmp_path, "map.png")
    Image.fromarray(image).save(path)

    natural = load_map(path, verbose=False)
    inverted = load_map(path, invert=True, verbose=False)

    assert natural[12, 12] == FREE and natural[0, 0] == OBSTACLE
    assert inverted[12, 12] == OBSTACLE and inverted[0, 0] == FREE


def test_map_stats_reports_free_area():
    occupancy = rect(blank(20, 20), 5, 15, 5, 15)
    stats = map_stats(occupancy)
    assert stats["free_pixels"] == 100
    assert stats["obstacle_pixels"] == 300
    assert stats["height"] == 20 and stats["width"] == 20
