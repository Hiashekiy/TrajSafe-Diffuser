"""Query-time subdivision of existing skeleton branches.

Compressed graph endpoints are topological junctions, not good attachment
points for a car already halfway down a long road. Splitting existing pixel
chains preserves the graph's geometry/connectivity while permitting local
attachments. This changes model input extraction, never its output curve.
"""
import numpy as np
from src.geometry.skeleton_graph import SkeletonGraph, SkeletonNode, SkeletonBranch


class QueryGraph(SkeletonGraph):
    def visible_anchors(self, point_scene, max_candidates=16):
        # Examine all subdivided nodes before applying the query-facing gate.
        anchors = super().visible_anchors(point_scene, len(self.nodes))
        point = np.asarray(point_scene)
        is_start = np.linalg.norm(point-self.query_start) < 1e-6
        is_goal = np.linalg.norm(point-self.query_goal) < 1e-6
        if is_start or is_goal:
            sign = 1. if is_start else -1.
            good = []
            for node, cells, length in anchors:
                delta = (self.pixel_to_scene(self.nodes[node].anchor)-point)*self.scene_scale
                forward = sign*float(delta@self.direction)
                lateral = abs(float(delta@np.array([-self.direction[1], self.direction[0]])))
                if 3.0 <= forward <= 18.0 and lateral <= max(0.6, forward*np.tan(np.deg2rad(35))):
                    good.append((node, cells, length))
            anchors = good
        return anchors[:max_candidates]


def query_graph(graph, start, goal, direction=(1., 0.), scene_scale=80., spacing_m=2.5):
    """Subdivide only existing branches. No new road or shortcut is introduced."""
    nodes = [SkeletonNode(n.idx, n.kind, list(n.pixels), n.center, n.anchor) for n in graph.nodes]
    branches = []
    node_pixels = graph.node_of_pixel.copy()
    stride = max(1, int(round(spacing_m/(graph.cell*scene_scale))))
    for br in graph.branches:
        cuts = [(0, br.u)]
        for i in range(stride, len(br.pixels)-1, stride):
            pixel = br.pixels[i]
            if node_pixels[pixel[1], pixel[0]] >= 0:
                continue
            n = len(nodes)
            nodes.append(SkeletonNode(n, 'query_auxiliary', [pixel], tuple(map(float, pixel)), pixel))
            node_pixels[pixel[1], pixel[0]] = n
            cuts.append((i, n))
        cuts.append((len(br.pixels)-1, br.v))
        for (a, u), (b, v) in zip(cuts[:-1], cuts[1:]):
            pixels = br.pixels[a:b+1]
            length = float(np.linalg.norm(np.diff(np.asarray(pixels), axis=0), axis=1).sum())
            branches.append(SkeletonBranch(len(branches), u, v, pixels, length))
    result = QueryGraph(graph.res, graph.free, graph.skeleton, nodes, branches, node_pixels, graph.stats)
    result.query_start, result.query_goal = np.asarray(start), np.asarray(goal)
    direction = np.asarray(direction, dtype=float)
    result.direction = direction/np.linalg.norm(direction)
    result.scene_scale = scene_scale
    return result
