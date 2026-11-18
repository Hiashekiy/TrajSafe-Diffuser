"""Thin a DMA working copy, then call upstream sknw unchanged.

No local node merging, edge deletion, spur pruning or path tracing.
"""

import importlib.metadata
import numpy as np
import networkx as nx
import sknw
from skimage.morphology import skeletonize
from skimage.measure import euler_number, label


def digital_topology(mask):
    count = int(label(mask, connectivity=2).max())
    return dict(components=count, holes=count - int(euler_number(mask, connectivity=2)))


def build_skeleton_graph(skeleton, distance_px, resolution=0.625):
    """Return (serializable multigraph, thinned working copy).

    The input DMA array is preserved. Curves end at node-cluster boundaries,
    not necessarily at representative coordinates (sknw ``full=False``).
    Distances are supplied in pixels; exported lengths/features are meters.
    """
    skeleton = np.asarray(skeleton)
    distance_px = np.asarray(distance_px)
    if skeleton.ndim != 2 or not np.isin(skeleton, [0, 1]).all():
        raise ValueError("skeleton must be a binary 2D array")
    if (
        distance_px.shape != skeleton.shape
        or not np.isfinite(distance_px).all()
        or np.any(distance_px < 0)
    ):
        raise ValueError(
            "distance_px must be a finite nonnegative array matching skeleton"
        )
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive meters per pixel")
    raw = np.asarray(skeleton, dtype=bool)
    thin = skeletonize(np.pad(raw, 1), method="zhang")[1:-1, 1:-1].copy()
    assert not np.any(thin & ~raw)
    before, after = digital_topology(raw), digital_topology(thin)
    assert before == after, (before, after)
    graph = sknw.build_sknw(
        thin.astype(np.uint16), multi=True, iso=True, ring=True, full=False
    )
    nodes = []
    for node, data in graph.nodes(data=True):
        pts = np.asarray(data["pts"], int)
        centroid = np.asarray(data["o"], float)
        rep = pts[np.argmin(np.sum((pts - centroid) ** 2, axis=1))]
        degree = int(graph.degree(node))
        selfloop = graph.has_edge(node, node)
        kind = (
            "isolated"
            if degree == 0
            else (
                "endpoint"
                if degree == 1
                else "loop_anchor" if degree == 2 and selfloop else "junction"
            )
        )
        nodes.append(
            dict(
                node_id=int(node),
                pixel_rc=rep.tolist(),
                x=float(rep[1] * resolution),
                y=float(rep[0] * resolution),
                centroid_rc=centroid.tolist(),
                node_type=kind,
                pixel_cluster=pts.tolist(),
                degree=degree,
                clearance=float(distance_px[tuple(rep)] * resolution),
            )
        )
    edges = []
    # Upstream pts order can be opposite NetworkX's undirected edge order.
    for source, target, key, data in graph.edges(keys=True, data=True):
        pts = np.asarray(data["pts"], int).copy()
        source_set = set(map(tuple, graph.nodes[source]["pts"].tolist()))
        target_set = set(map(tuple, graph.nodes[target]["pts"].tolist()))
        if tuple(pts[0]) not in source_set:
            pts = pts[::-1].copy()
        assert tuple(pts[0]) in source_set and tuple(pts[-1]) in target_set
        assert np.all(thin[pts[:, 0], pts[:, 1]])
        steps = np.abs(np.diff(pts, axis=0))
        assert np.all((steps.max(axis=1) == 1)), "non-adjacent upstream curve step"
        xy = pts[:, ::-1] * resolution
        d = distance_px[pts[:, 0], pts[:, 1]] * resolution
        edges.append(
            dict(
                edge_id=len(edges),
                source=int(source),
                target=int(target),
                multigraph_key=int(key),
                curve_points=xy.tolist(),
                curve_pixels_rc=pts.tolist(),
                length=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                mean_clearance=float(d.mean()),
                min_clearance=float(d.min()),
                max_clearance=float(d.max()),
                mean_width=float(2 * d.mean()),
                min_width=float(2 * d.min()),
            )
        )
    represented = {tuple(p) for n in nodes for p in n["pixel_cluster"]}
    for edge in edges:
        represented.update(map(tuple, edge["curve_pixels_rc"]))
    assert represented == set(
        map(tuple, np.argwhere(thin).tolist())
    ), "unrepresented thinned pixels"
    c = nx.number_connected_components(graph)
    assert c == after["components"]
    result = dict(
        schema_version=2,
        method="scikit-image skeletonize working copy -> upstream sknw",
        parameters=dict(multi=True, iso=True, ring=True, full=False, thinning="zhang"),
        versions={
            m: importlib.metadata.version(m)
            for m in ("sknw", "scikit-image", "networkx", "numba")
        },
        resolution_m_per_pixel=resolution,
        coordinates="x=column*resolution, y=row*resolution; local origin lower",
        curve_endpoint_semantics="Endpoints belong to node pixel_cluster; not replaced by node representative.",
        nodes=nodes,
        edges=edges,
        audit=dict(
            raw_pixels=int(raw.sum()),
            thin_pixels=int(thin.sum()),
            raw_topology=before,
            thin_topology=after,
            nodes=len(nodes),
            edges=len(edges),
            components=c,
            cycle_rank=len(edges) - len(nodes) + c,
            all_thinned_pixels_represented=True,
        ),
    )
    result["audit"]["warnings"] = []
    if result["audit"]["cycle_rank"] != after["holes"]:
        result["audit"]["warnings"].append(
            "sknw cycle rank differs from digital hole count; inspect local diagonal/self-loop artifacts. No automatic pruning applied."
        )
    return result, thin


def save_arrays(path, g, raw, thin, occ, distance, nearest_background_rc=None):
    """Write graph arrays without pickle; offsets delimit variable-length data."""
    curves = [e["curve_pixels_rc"] for e in g["edges"]]
    clusters = [n["pixel_cluster"] for n in g["nodes"]]
    extra = (
        {}
        if nearest_background_rc is None
        else {"nearest_background_rc": nearest_background_rc}
    )
    np.savez_compressed(
        path,
        occupancy=occ,
        raw_skeleton=raw,
        thin_skeleton=thin,
        distance_px=distance,
        node_xy=np.asarray([[n["x"], n["y"]] for n in g["nodes"]]).reshape(-1, 2),
        node_types=np.asarray([n["node_type"] for n in g["nodes"]], dtype="U16"),
        node_degree=np.asarray([n["degree"] for n in g["nodes"]], int),
        node_cluster_rc=np.asarray(
            [p for cluster in clusters for p in cluster], int
        ).reshape(-1, 2),
        node_cluster_offsets=np.r_[0, np.cumsum([len(c) for c in clusters])],
        edge_index=np.asarray([[e["source"], e["target"]] for e in g["edges"]], int)
        .reshape(-1, 2)
        .T,
        curve_pixels_rc=np.asarray([p for curve in curves for p in curve], int).reshape(
            -1, 2
        ),
        curve_offsets=np.r_[0, np.cumsum([len(c) for c in curves])],
        edge_features=np.asarray(
            [
                [
                    e[k]
                    for k in (
                        "length",
                        "mean_clearance",
                        "min_clearance",
                        "max_clearance",
                        "mean_width",
                        "min_width",
                    )
                ]
                for e in g["edges"]
            ]
        ).reshape(-1, 6),
        edge_feature_names=np.asarray(
            [
                "length",
                "mean_clearance",
                "min_clearance",
                "max_clearance",
                "mean_width",
                "min_width",
            ]
        ),
        resolution_m_per_pixel=g["resolution_m_per_pixel"],
        **extra
    )
