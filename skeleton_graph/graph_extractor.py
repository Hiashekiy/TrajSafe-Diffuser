"""Skeleton -> node/edge graph.

This is where the three problems that actually matter are solved:

1. **Junction cluster merging.**  A real crossing is usually several adjacent
   ``d >= 3`` pixels (``----xxx----`` plus the arms).  Every 8-connected
   cluster of such pixels becomes exactly *one* graph node, otherwise the
   graph fills up with dozens of fake nodes sitting on top of each other.

2. **Edge tracing.**  From every node pixel the skeleton is walked through
   ``d == 2`` chain pixels until another node is reached.  The whole chain
   collapses into a single edge, and the edge keeps its full pixel polyline,
   not just its two end nodes.

3. **Cycle preservation.**  A skeleton component in which *every* pixel has
   ``d == 2`` is a pure loop and contains no endpoint and no junction.  Such a
   component would otherwise yield zero nodes and disappear.  One or two
   auxiliary nodes are injected so the loop stays a loop.

Coordinates
-----------
Internally this module indexes arrays with ``(y, x)``.  Everything that
leaves the module -- node attributes, edge polylines, ``graph.json`` -- uses
``(x, y) = (column, row)``.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from skeleton_graph.thinning import compute_degree
from skeleton_graph.topology import (
    count_holes,
    label_components,
    neighbour_offsets,
)

NODE_ENDPOINT = "endpoint"
NODE_JUNCTION = "junction"
NODE_AUXILIARY = "auxiliary"
NODE_TYPES = (NODE_ENDPOINT, NODE_JUNCTION, NODE_AUXILIARY)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def skeleton_to_graph(
    skeleton: np.ndarray,
    connectivity: int = 8,
    pure_cycle_aux_nodes: int = 2,
    graph_container: str = "auto",
    dissolve_degree2_junctions: bool = True,
    verbose: bool = True,
) -> tuple[object, dict]:
    """Convert a one-pixel skeleton into a node/edge graph.

    Returns ``(G, info)``.  ``G`` is a :class:`networkx.Graph`, or a
    :class:`networkx.MultiGraph` when the topology genuinely contains
    parallel edges (``graph_container="auto"``).
    """
    skel = np.asarray(skeleton) > 0
    if not skel.any():
        raise ValueError("skeleton is empty; nothing to convert")

    degree = compute_degree(skel, connectivity)
    endpoint_mask = skel & (degree == 1)
    junction_mask = skel & (degree >= 3)
    isolated_mask = skel & (degree == 0)

    # ---------------------------------------------------------------- nodes
    node_pixels: dict[int, list[tuple[int, int]]] = {}
    node_type: dict[int, str] = {}

    for y, x in np.argwhere(endpoint_mask):
        _add_node(node_pixels, node_type, [(int(y), int(x))], NODE_ENDPOINT)

    junction_labels, n_junction = label_components(junction_mask, connectivity)
    cluster_sizes: list[int] = []
    for label in range(1, n_junction + 1):
        pixels = [(int(y), int(x)) for y, x in np.argwhere(junction_labels == label)]
        cluster_sizes.append(len(pixels))
        _add_node(node_pixels, node_type, pixels, NODE_JUNCTION)

    aux_info = _inject_auxiliary_nodes(
        skel, endpoint_mask, junction_mask, isolated_mask, connectivity,
        pure_cycle_aux_nodes, node_pixels, node_type,
    )

    if not node_pixels:
        raise RuntimeError("no graph node could be derived from the skeleton")

    # ------------------------------------------------------------- tracing
    node_of_pixel = _build_node_of_pixel(skel.shape, node_pixels)
    absorbed = _absorb_trivial_bridges(
        skel, node_of_pixel, node_pixels, connectivity
    )
    raw_edges = _trace_edges(skel, node_pixels, node_of_pixel, connectivity)

    # Pixels that belong to a node of the *pre-dissolution* graph.  They are
    # never "chain pixels", which is what the coverage check needs to know.
    non_chain_pixels = [
        [x, y] for pixels in node_pixels.values() for y, x in pixels
    ]

    # --------------------------------------------------------------- graph
    # A degree-2 junction node is not a junction: the corridor simply turns
    # there (Guo-Hall leaves a short diagonal ladder of d >= 3 pixels at every
    # 90 degree bend).  Dissolve those before choosing the container, because
    # the merge can create parallel edges.
    final_edges, dissolved, remaining_nodes = _dissolve_degree2_junctions(
        raw_edges, node_pixels, node_type, connectivity,
        enabled=dissolve_degree2_junctions,
    )

    container = _resolve_container(final_edges, graph_container)
    graph = _build_graph(container, node_pixels, node_type, final_edges,
                         removed_nodes={d["id"] for d in dissolved})

    parallel_pairs = _parallel_pairs(final_edges)
    final_types = {
        nid: kind
        for nid, kind in node_type.items()
        if nid in remaining_nodes
    }
    info = {
        "container": container,
        "nodes": len(remaining_nodes),
        "edges": len(final_edges),
        "node_type_counts": {
            t: sum(1 for v in final_types.values() if v == t) for t in NODE_TYPES
        },
        "node_pixels": {
            str(nid): [[x, y] for y, x in node_pixels[nid]]
            for nid in sorted(remaining_nodes)
        },
        "non_chain_pixels": non_chain_pixels,
        "junction_cluster_sizes": sorted(cluster_sizes, reverse=True),
        "absorbed_bridge_pixels": absorbed,
        "dissolved_junctions": dissolved,
        "auxiliary": aux_info,
        "parallel_edge_pairs": parallel_pairs,
        "self_loops": sum(1 for u, v, _, _ in final_edges if u == v),
    }

    if verbose:
        print(
            f"[graph] container={container} nodes={info['nodes']} "
            f"({info['node_type_counts']}) edges={info['edges']} "
            f"junction_clusters={len(cluster_sizes)} "
            f"max_cluster={max(cluster_sizes) if cluster_sizes else 0}"
        )
    return graph, info


# ---------------------------------------------------------------------------
# auxiliary nodes (pure cycles / isolated pixels)
# ---------------------------------------------------------------------------


def _inject_auxiliary_nodes(
    skel: np.ndarray,
    endpoint_mask: np.ndarray,
    junction_mask: np.ndarray,
    isolated_mask: np.ndarray,
    connectivity: int,
    pure_cycle_aux_nodes: int,
    node_pixels: dict[int, list[tuple[int, int]]],
    node_type: dict[int, str],
) -> dict:
    """Add auxiliary nodes for components that would otherwise vanish."""
    component_labels, n_components = label_components(skel, connectivity)
    offsets = neighbour_offsets(connectivity)

    pure_cycles: list[dict] = []
    isolated: list[list[int]] = []

    for label in range(1, n_components + 1):
        component = component_labels == label

        # A component that already owns an endpoint or a junction is fully
        # described by those nodes and needs no auxiliary help.
        if (endpoint_mask & component).any() or (junction_mask & component).any():
            continue

        if (isolated_mask & component).any():
            # A single free-space pixel: a node with no edges.
            pixel = tuple(int(v) for v in np.argwhere(component)[0])
            _add_node(node_pixels, node_type, [pixel], NODE_AUXILIARY)
            isolated.append([pixel[1], pixel[0]])
            continue

        # Every remaining pixel has degree 2 => this component is a pure loop.
        size = int(component.sum())
        if size <= 2:
            continue

        loop = _trace_loop(component, offsets)
        if not loop:
            continue

        if pure_cycle_aux_nodes == 1:
            chosen = [loop[0]]
        else:
            # Two *separate* nodes at roughly opposite poles of the loop, so
            # the loop becomes a clean two-edge cycle instead of a self-loop.
            chosen = _dedupe_preserving_order([loop[0], loop[len(loop) // 2]])

        for pixel in chosen:
            _add_node(node_pixels, node_type, [pixel], NODE_AUXILIARY)
        pure_cycles.append(
            {
                "loop_pixels": len(loop),
                "aux_nodes": [[x, y] for y, x in chosen],
            }
        )

    return {"pure_cycles": pure_cycles, "isolated_pixels": isolated}


def _absorb_trivial_bridges(
    skel: np.ndarray,
    node_of_pixel: np.ndarray,
    node_pixels: dict[int, list[tuple[int, int]]],
    connectivity: int,
) -> int:
    """Merge unassigned chain pixels whose neighbours all belong to one node.

    A degree-2 pixel sandwiched between two pixels of the *same* junction
    cluster is part of that junction blob, not a one-pixel edge.  Left alone
    it shows up as a bogus length-2 self-loop::

        A C          A, B junction pixels of the same cluster
        . B          C chain pixel adjacent to both
                     -> walk A -> C -> B "returns" to the cluster

    Absorbing ``C`` into the cluster removes the artefact.  Real cycles are
    untouched: the first pixel of a loop leaving a junction always has a chain
    neighbour, so it is never absorbed, and a loop is only ever swallowed when
    it is a single pixel thick and one pixel long (i.e. degenerate).

    Returns the number of absorbed pixels.  ``node_of_pixel`` is modified in
    place.
    """
    offsets = neighbour_offsets(connectivity)
    padded_skel = np.pad(skel, 1, mode="constant")
    # np.pad copies, so the result must be written back into node_of_pixel
    # before returning; forgetting that silently re-creates the artefact.
    padded_nodes = np.pad(node_of_pixel, 1, mode="constant", constant_values=-1)

    absorbed = 0
    changed = True
    while changed:
        changed = False
        for py, px in np.argwhere(padded_skel & (padded_nodes < 0)):
            py, px = int(py), int(px)
            neighbour_nodes = set()
            for dy, dx in offsets:
                if padded_skel[py + dy, px + dx]:
                    neighbour_nodes.add(int(padded_nodes[py + dy, px + dx]))
            if len(neighbour_nodes) != 1:
                continue
            owner = neighbour_nodes.pop()
            if owner < 0:
                continue
            padded_nodes[py, px] = owner
            node_pixels[owner].append((py - 1, px - 1))
            absorbed += 1
            changed = True

    node_of_pixel[:] = padded_nodes[1:-1, 1:-1]
    return absorbed


def _trace_loop(component: np.ndarray, offsets) -> list[tuple[int, int]]:
    """Ordered pixel list of a simple 8-connected loop."""
    pixels = np.argwhere(component)
    if pixels.size == 0:
        return []
    start = (int(pixels[0][0]), int(pixels[0][1]))

    loop = [start]
    previous: tuple[int, int] | None = None
    current = start
    guard = int(component.sum()) + 2
    for _ in range(guard):
        y, x = current
        candidates = [
            (y + dy, x + dx)
            for dy, dx in offsets
            if component[y + dy, x + dx] and (y + dy, x + dx) != previous
        ]
        if not candidates:
            return loop
        nxt = candidates[0]
        if nxt == start:
            return loop
        loop.append(nxt)
        previous, current = current, nxt
    return loop


def _dedupe_preserving_order(items: list) -> list:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# node bookkeeping
# ---------------------------------------------------------------------------


def _add_node(
    node_pixels: dict[int, list[tuple[int, int]]],
    node_type: dict[int, str],
    pixels: list[tuple[int, int]],
    kind: str,
) -> int:
    node_id = len(node_pixels)
    node_pixels[node_id] = pixels
    node_type[node_id] = kind
    return node_id


def _build_node_of_pixel(shape, node_pixels: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    node_of_pixel = np.full(shape, -1, dtype=np.int32)
    for node_id, pixels in node_pixels.items():
        for y, x in pixels:
            node_of_pixel[y, x] = node_id
    return node_of_pixel


def node_centroid(pixels: list[tuple[int, int]]) -> tuple[int, int]:
    """Mean of the node pixels, snapped to the closest pixel of the node.

    Matches the specification: "node position may use the mean of the cluster
    pixels, finally snapped to the nearest skeleton pixel".
    """
    if len(pixels) == 1:
        return pixels[0]
    arr = np.asarray(pixels, dtype=np.float64)
    mean = arr.mean(axis=0)
    distances = ((arr - mean) ** 2).sum(axis=1)
    y, x = arr[int(np.argmin(distances))]
    return int(y), int(x)


# ---------------------------------------------------------------------------
# edge tracing
# ---------------------------------------------------------------------------


def _trace_edges(
    skel: np.ndarray,
    node_pixels: dict[int, list[tuple[int, int]]],
    node_of_pixel: np.ndarray,
    connectivity: int,
) -> list[tuple[int, int, list[tuple[int, int]], float]]:
    """Walk every edge exactly once.

    Returns ``[(u, v, path, length), ...]`` with ``path`` in ``(y, x)``.
    Because the walk is deterministic and reversible, the pixel set of a path
    is a perfect identity key: two walks covering the same pixels are the same
    edge, seen from the other end.
    """
    offsets = neighbour_offsets(connectivity)
    padded_skel = np.pad(skel, 1, mode="constant")
    padded_nodes = np.pad(node_of_pixel, 1, mode="constant", constant_values=-1)

    edges: list[tuple[int, int, list[tuple[int, int]], float]] = []
    seen: set[frozenset] = set()

    for node_id in sorted(node_pixels):
        for pixel in node_pixels[node_id]:
            y0, x0 = pixel[0] + 1, pixel[1] + 1
            for dy, dx in offsets:
                qy, qx = y0 + dy, x0 + dx
                if not padded_skel[qy, qx]:
                    continue
                if padded_nodes[qy, qx] == node_id:
                    continue  # inside this node's own cluster

                path = [(y0, x0), (qy, qx)]
                previous, (cy, cx) = (y0, x0), (qy, qx)

                while padded_nodes[cy, cx] < 0:
                    candidates = [
                        (cy + dy2, cx + dx2)
                        for dy2, dx2 in offsets
                        if padded_skel[cy + dy2, cx + dx2]
                        and (cy + dy2, cx + dx2) != previous
                    ]
                    if len(candidates) != 1:
                        # Should be unreachable while d(cur) == 2; kept as a
                        # guard so a malformed skeleton can never hang.
                        break
                    previous, (cy, cx) = (cy, cx), candidates[0]
                    path.append((cy, cx))

                target = int(padded_nodes[cy, cx])
                if target < 0:
                    continue

                key = frozenset(path)
                if key in seen:
                    continue
                seen.add(key)

                unpadded = [(y - 1, x - 1) for y, x in path]
                edges.append(
                    (node_id, target, unpadded, polyline_length(unpadded))
                )

    return edges


def polyline_length(points: list[tuple[int, int]]) -> float:
    """Euclidean length of a pixel polyline: axis step 1, diagonal ``sqrt(2)``."""
    total = 0.0
    for (y0, x0), (y1, x1) in zip(points, points[1:]):
        total += math.hypot(y1 - y0, x1 - x0)
    return total


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------


def _resolve_container(
    edges: list[tuple],
    requested: str,
) -> str:
    """Pick ``Graph`` or ``MultiGraph``.

    ``"auto"`` keeps ``networkx.Graph`` -- the container named in the
    specification -- unless the extracted topology really does contain
    parallel edges.  In that case ``MultiGraph`` is required, because a plain
    ``Graph`` would silently swallow one of the two arcs of a cycle.
    """
    if requested in ("graph", "multigraph"):
        return requested
    return "multigraph" if _parallel_pairs(edges) else "graph"


def _dissolve_degree2_junctions(
    raw_edges: list[tuple],
    node_pixels: dict[int, list[tuple[int, int]]],
    node_type: dict[int, str],
    connectivity: int,
    enabled: bool = True,
):
    """Merge away junction nodes that carry exactly two edges.

    Guo-Hall leaves a short diagonal ladder of ``d >= 3`` pixels at every
    90 degree bend of the free space.  Cluster merging correctly turns that
    ladder into a single node, but the node is not a junction: the corridor
    simply continues through it with exactly two incident edges.

    Dissolving such a node replaces it by one edge joining its two
    neighbours, with the polylines concatenated (plus the shortest in-cluster
    hop, so the merged polyline stays 8-connected and on the skeleton).
    Topology is untouched: removing a degree-2 vertex and merging its two
    edges preserves the cycle rank exactly.

    Only ``junction`` nodes are considered; auxiliary nodes must stay,
    otherwise a pure cycle would collapse into a self-loop.

    Returns ``(final_edges, dissolved, remaining_node_ids)`` where
    ``final_edges`` entries are ``(u, v, pixels_xy, length)``.
    """
    import networkx as nx

    temp = nx.MultiGraph()
    for node_id, pixels in sorted(node_pixels.items()):
        y, x = node_centroid(pixels)
        temp.add_node(node_id, x=int(x), y=int(y), type=node_type[node_id])
    for u, v, path, length in raw_edges:
        temp.add_edge(
            u, v,
            length=round(float(length), 6),
            pixels=[[int(x), int(y)] for y, x in path],
        )

    offsets_xy = [(dx, dy) for dy, dx in neighbour_offsets(connectivity)]
    cluster_of = {
        node_id: {(x, y) for y, x in pixels}
        for node_id, pixels in node_pixels.items()
    }

    dissolved: list[dict] = []
    blocked: set[int] = set()

    while enabled:
        target = None
        for node_id in temp.nodes:
            if node_id in blocked:
                continue
            if temp.nodes[node_id]["type"] != NODE_JUNCTION:
                continue
            if temp.degree(node_id) != 2:
                continue
            incident = list(temp.edges(node_id, keys=True, data=True))
            if len(incident) != 2:
                # A lone self-loop also reports degree 2.
                blocked.add(node_id)
                continue
            target = (node_id, incident)
            break
        if target is None:
            break

        node_id, incident = target
        (u1, v1, key1, data1), (u2, v2, key2, data2) = incident
        far1 = v1 if u1 == node_id else u1
        far2 = v2 if u2 == node_id else v2
        if far1 == node_id or far2 == node_id:
            blocked.add(node_id)  # self-loop attached here: leave it alone
            continue

        cluster = cluster_of[node_id]
        pixels1, end1 = _orient_in_cluster(data1["pixels"], cluster, at_end=True)
        pixels2, end2 = _orient_in_cluster(data2["pixels"], cluster, at_end=False)
        bridge = (
            None
            if pixels1 is None or pixels2 is None
            else _shortest_path_in_set(cluster, end1, end2, offsets_xy)
        )
        if bridge is None:  # pragma: no cover - defensive
            blocked.add(node_id)
            continue

        # pixels1 ends on the cluster, pixels2 starts on it, and `bridge` joins
        # the two contact pixels, so the merge never introduces a jump.
        merged: list[list[int]] = [list(p) for p in pixels1]
        merged.extend([list(p) for p in bridge[1:]])
        merged.extend([list(p) for p in pixels2[1:]])

        temp.remove_edge(u1, v1, key1)
        temp.remove_edge(u2, v2, key2)
        temp.add_edge(
            far1, far2,
            length=round(polyline_length([(y, x) for x, y in merged]), 6),
            pixels=merged,
        )
        temp.remove_node(node_id)
        cluster_of.pop(node_id, None)
        dissolved.append(
            {
                "id": int(node_id),
                "x": int(node_pixels[node_id][0][1]),
                "y": int(node_pixels[node_id][0][0]),
                "cluster_pixels": len(node_pixels[node_id]),
                "merged": [int(far1), int(far2)],
            }
        )

    final_edges = [
        (int(u), int(v), data["pixels"], float(data["length"]))
        for u, v, _key, data in temp.edges(keys=True, data=True)
    ]
    return final_edges, dissolved, set(temp.nodes)


def _orient_in_cluster(pixels, cluster: set, at_end: bool):
    """Orient a polyline so that the requested end sits on ``cluster``.

    ``at_end=True`` makes the polyline *finish* on the cluster,
    ``at_end=False`` makes it *start* on the cluster.
    """
    first = (pixels[0][0], pixels[0][1])
    last = (pixels[-1][0], pixels[-1][1])
    wanted_first, wanted_last = (first, last) if at_end else (last, first)
    if wanted_last in cluster:
        return list(pixels), wanted_last
    if wanted_first in cluster:
        return list(reversed(pixels)), wanted_first
    return None, None


def _shortest_path_in_set(pixels_set: set, start, end, offsets_xy):
    """Shortest 8-connected hop inside a node cluster (inclusive endpoints)."""
    if start == end:
        return [start]
    from collections import deque

    previous = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for dx, dy in offsets_xy:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in pixels_set and nxt not in previous:
                previous[nxt] = current
                if nxt == end:
                    path = [nxt]
                    while previous[path[-1]] is not None:
                        path.append(previous[path[-1]])
                    return list(reversed(path))
                queue.append(nxt)
    return None


def _parallel_pairs(edges: list[tuple]) -> list[list[int]]:
    seen: dict[tuple[int, int], int] = {}
    pairs: list[list[int]] = []
    for u, v, *_rest in edges:
        key = (min(u, v), max(u, v))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            pairs.append([key[0], key[1]])
    return pairs


def _build_graph(
    container: str,
    node_pixels: dict[int, list[tuple[int, int]]],
    node_type: dict[int, str],
    edges: list[tuple],
    removed_nodes: set[int] | None = None,
):
    import networkx as nx

    removed_nodes = removed_nodes or set()
    graph = nx.MultiGraph() if container == "multigraph" else nx.Graph()

    for node_id, pixels in sorted(node_pixels.items()):
        if node_id in removed_nodes:
            continue
        y, x = node_centroid(pixels)
        graph.add_node(node_id, x=int(x), y=int(y), type=node_type[node_id])

    for u, v, pixels, length in edges:
        graph.add_edge(
            u, v,
            length=round(float(length), 6),
            pixels=[[int(x), int(y)] for x, y in pixels],
        )
    return graph


def iter_edges(graph):
    """Yield ``(u, v, data)`` for ``Graph`` and ``MultiGraph`` alike."""
    if graph.is_multigraph():
        for u, v, _key, data in graph.edges(keys=True, data=True):
            yield u, v, data
    else:
        for u, v, data in graph.edges(data=True):
            yield u, v, data


# ---------------------------------------------------------------------------
# summary / validation
# ---------------------------------------------------------------------------


def graph_summary(graph) -> dict:
    """Structural summary used by the report."""
    import networkx as nx

    type_counts = {t: 0 for t in NODE_TYPES}
    for _, data in graph.nodes(data=True):
        type_counts[data.get("type", NODE_AUXILIARY)] = (
            type_counts.get(data.get("type", NODE_AUXILIARY), 0) + 1
        )

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_components = nx.number_connected_components(graph) if n_nodes else 0
    cycle_rank = n_edges - n_nodes + n_components

    self_loops = 0
    for u, v, _ in iter_edges(graph):
        if u == v:
            self_loops += 1

    total_length = sum(data.get("length", 0.0) for _, _, data in iter_edges(graph))

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "node_type_counts": type_counts,
        "connected_components": n_components,
        "cycle_rank": cycle_rank,
        "has_cycle": cycle_rank > 0,
        "self_loops": self_loops,
        "total_edge_length": round(float(total_length), 4),
        "isolated_nodes": sum(1 for n in graph.nodes if graph.degree(n) == 0),
    }


def validate_graph(
    skeleton: np.ndarray,
    graph,
    connectivity: int = 8,
    non_chain_pixels_xy=None,
) -> dict:
    """Cross-check the graph against the skeleton it came from.

    Checks performed
    ----------------
    * every skeleton pixel that is *not* part of a graph node (i.e. every
      ``d == 2`` chain pixel) is covered by exactly one edge polyline --
      nothing dropped, nothing duplicated;
    * every edge polyline lies entirely on the skeleton;
    * every edge polyline is a single 8-connected walk (no jumps);
    * edge length matches the polyline geometry;
    * the graph's cycle rank equals the number of holes in the skeleton
      (this is the formal statement of "no cycle was lost").

    ``non_chain_pixels_xy`` is ``info["non_chain_pixels"]`` from
    :func:`skeleton_to_graph`.  It lists every pixel that belongs to a node
    *before* degree-2 junctions were dissolved; without it those pixels would
    look like uncovered chain pixels.
    """
    skel = np.asarray(skeleton) > 0
    degree = compute_degree(skel, connectivity)

    node_mask = np.zeros(skel.shape, dtype=bool)
    for x, y in non_chain_pixels_xy or ():
        node_mask[int(y), int(x)] = True

    chain_mask = skel & (degree == 2) & ~node_mask
    chain_total = int(chain_mask.sum())

    coverage = np.zeros(skel.shape, dtype=np.int32)
    off_skeleton = 0
    length_mismatch = 0
    polyline_breaks = 0
    for _u, _v, data in iter_edges(graph):
        pixels = data["pixels"]
        for x, y in pixels:
            if not (0 <= y < skel.shape[0] and 0 <= x < skel.shape[1]) or not skel[y, x]:
                off_skeleton += 1
            else:
                coverage[y, x] += 1
        for (x0, y0), (x1, y1) in zip(pixels, pixels[1:]):
            if max(abs(x1 - x0), abs(y1 - y0)) != 1:
                polyline_breaks += 1
        expected = polyline_length([(y, x) for x, y in pixels])
        if abs(expected - float(data["length"])) > 1e-6:
            length_mismatch += 1

    chain_covered = int(((coverage > 0) & chain_mask).sum())
    chain_doubled = int(((coverage > 1) & chain_mask).sum())

    summary = graph_summary(graph)
    skeleton_holes = count_holes(skel)

    checks = {
        "chain_pixels_total": chain_total,
        "chain_pixels_covered": chain_covered,
        "chain_pixels_covered_twice": chain_doubled,
        "chain_coverage_complete": chain_covered == chain_total,
        "pixels_off_skeleton": off_skeleton,
        "polyline_breaks": polyline_breaks,
        "edge_length_mismatches": length_mismatch,
        "skeleton_holes": skeleton_holes,
        "graph_cycle_rank": summary["cycle_rank"],
        "cycles_preserved": skeleton_holes == summary["cycle_rank"],
    }
    checks["ok"] = (
        checks["chain_coverage_complete"]
        and off_skeleton == 0
        and polyline_breaks == 0
        and length_mismatch == 0
        and checks["cycles_preserved"]
    )
    return checks


# ---------------------------------------------------------------------------
# JSON serialisation (networkx independent)
# ---------------------------------------------------------------------------


def graph_to_json(graph, path: str, verbose: bool = True) -> dict:
    """Write ``graph.json`` in the networkx-independent interchange format."""
    payload = graph_to_dict(graph)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    if verbose:
        print(
            f"[graph.json] {path} "
            f"nodes={len(payload['nodes'])} edges={len(payload['edges'])}"
        )
    return payload


def graph_to_dict(graph) -> dict:
    """``{"nodes": [...], "edges": [...]}`` -- coordinates are ``[x, y]``."""
    nodes = [
        {
            "id": int(node_id),
            "x": int(data["x"]),
            "y": int(data["y"]),
            "type": str(data["type"]),
        }
        for node_id, data in sorted(graph.nodes(data=True))
    ]
    edges = [
        {
            "source": int(u),
            "target": int(v),
            "length": round(float(data["length"]), 6),
            "pixels": [[int(x), int(y)] for x, y in data["pixels"]],
        }
        for u, v, data in iter_edges(graph)
    ]
    return {"nodes": nodes, "edges": edges}


def load_graph_json(path: str, graph_container: str = "auto"):
    """Rebuild a networkx graph from ``graph.json``.

    The rebuilt graph is structurally identical to the one that was written,
    which is what the round-trip acceptance check verifies.
    """
    import networkx as nx

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    nodes = payload["nodes"]
    edges = payload["edges"]

    needs_multi = len({(min(e["source"], e["target"]), max(e["source"], e["target"]))
                       for e in edges}) != len(edges)
    if graph_container == "multigraph" or (graph_container == "auto" and needs_multi):
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()

    for node in nodes:
        graph.add_node(
            int(node["id"]),
            x=int(node["x"]),
            y=int(node["y"]),
            type=str(node["type"]),
        )
    for edge in edges:
        graph.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            length=round(float(edge["length"]), 6),
            pixels=[[int(x), int(y)] for x, y in edge["pixels"]],
        )
    return graph


def graphs_equal(left, right) -> tuple[bool, dict]:
    """Structural equality of two graphs (node set + edge multiset)."""
    left_nodes = sorted(
        (int(n), int(d["x"]), int(d["y"]), str(d["type"]))
        for n, d in left.nodes(data=True)
    )
    right_nodes = sorted(
        (int(n), int(d["x"]), int(d["y"]), str(d["type"]))
        for n, d in right.nodes(data=True)
    )

    def edge_key(u, v, data):
        a, b = (u, v) if u <= v else (v, u)
        return (
            int(a),
            int(b),
            round(float(data["length"]), 6),
            tuple((int(x), int(y)) for x, y in data["pixels"]),
        )

    left_edges = sorted(edge_key(u, v, d) for u, v, d in iter_edges(left))
    right_edges = sorted(edge_key(u, v, d) for u, v, d in iter_edges(right))

    details = {
        "nodes_equal": left_nodes == right_nodes,
        "edges_equal": left_edges == right_edges,
        "left_nodes": len(left_nodes),
        "right_nodes": len(right_nodes),
        "left_edges": len(left_edges),
        "right_edges": len(right_edges),
    }
    return details["nodes_equal"] and details["edges_equal"], details
