"""Skeleton geometry step 1: occupancy -> safe free space -> skeleton -> compressed
branch graph.

This module is the *single* implementation used by both the offline
preprocessing scripts and online inference.  It deliberately contains no path
ranking and no learning: it only answers "what is the free-space skeleton, and
which skeleton nodes can a start/goal reach without crossing an obstacle".

Conventions
-----------
occupancy map
    [H, W] array, 1 = obstacle (the convention of
    data/scenes/maps/*.npy), 0 = free.  Float maps are binarised
    with > 0.5.
free mask
    bool [H, W], True = free.  Built from the occupancy map, optionally after
    binary-dilating the obstacles by safety_dilation_cells so that a skeleton
    pixel keeps a small margin from every obstacle.
skeleton
    bool [H, W], True = skeleton pixel (Guo-Hall thinning of the free mask,
    reused from the standalone skeleton_graph package).
coordinates
    (x, y) = (column, row) in *pixel centre* units.  pixel_to_scene converts to
    the canonical scene frame [-1, 1]^2 used by every model in this repository
    (x right, y up, row 0 = y = -1).

No corner cutting
-----------------
Plain 8-neighbour adjacency is wrong on a skeleton: the two pixels

    # .
    . #

are diagonally adjacent, and joining them would push the polyline through the
shared obstacle corner.  A diagonal step (x, y) -> (x + dx, y + dy) is
therefore allowed only when both orthogonal side cells (x + dx, y) and
(x, y + dy) are free.  Every emitted branch and every connector is a chain of
cells that stays inside free space.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ORTHO_OFFSETS",
    "DIAG_OFFSETS",
    "ALL_OFFSETS",
    "free_mask_from_occupancy",
    "skeletonize_free",
    "pixel_degrees",
    "allowed_neighbour_masks",
    "supercover_pixels",
    "supercover_is_free",
    "SkeletonNode",
    "SkeletonBranch",
    "SkeletonGraph",
    "build_skeleton_graph",
    "anchor_point",
    "save_graph_npz",
    "load_graph_npz",
]

#: (dy, dx) offsets, image convention (row, col).
ORTHO_OFFSETS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
DIAG_OFFSETS: Tuple[Tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ALL_OFFSETS: Tuple[Tuple[int, int], ...] = ORTHO_OFFSETS + DIAG_OFFSETS


# ---------------------------------------------------------------------------
# free space / skeleton
# ---------------------------------------------------------------------------


def free_mask_from_occupancy(occ: np.ndarray,
                             safety_dilation_cells: int = 1) -> np.ndarray:
    """occupancy (1 = obstacle) -> bool [H, W] free mask.

    safety_dilation_cells binary-dilates the obstacles with a 3x3 structure
    before complementing, so skeleton pixels keep that many cells of margin.
    """
    occ = np.asarray(occ)
    if occ.ndim != 2:
        raise ValueError("occupancy map must be 2D, got shape %s" % (occ.shape,))
    if int(safety_dilation_cells) < 0:
        raise ValueError("safety_dilation_cells must be >= 0")
    obs = (occ > 0.5) if occ.dtype.kind == "f" else (occ.astype(np.int64) > 0)
    if int(safety_dilation_cells) > 0:
        from scipy import ndimage  # local import: keeps module import cheap

        obs = ndimage.binary_dilation(
            obs, structure=np.ones((3, 3), dtype=bool),
            iterations=int(safety_dilation_cells),
        )
    free = ~obs
    if not free.any():
        raise ValueError(
            "free space is empty after dilating obstacles by "
            "%d cell(s); lower skeleton.safety_dilation_cells"
            % int(safety_dilation_cells)
        )
    return free


def skeletonize_free(free: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Guo-Hall thinning of the free mask -> bool [H, W] skeleton.

    Reuses the vendored numpy Guo-Hall implementation of the standalone
    skeleton_graph stage-1 package (cv2.ximgproc when available).
    """
    free = np.asarray(free)
    if free.dtype != bool:
        free = free > 0
    if not free.any():
        raise ValueError("free mask is empty; nothing to thin")
    occ255 = np.where(free, 255, 0).astype(np.uint8)
    try:
        from .thinning import guo_hall_thinning
    except Exception as exc:  # pragma: no cover - import environment failure
        raise ImportError(
            "skeletonize_free needs the standalone skeleton_graph package "
            "(repo root must be on sys.path): " + str(exc)
        ) from exc
    skel = guo_hall_thinning(occ255, backend=backend, verbose=False)
    return np.asarray(skel) > 0


# ---------------------------------------------------------------------------
# no-corner-cut neighbourhood
# ---------------------------------------------------------------------------


def _shift(arr: np.ndarray, dy: int, dx: int, fill=False) -> np.ndarray:
    """out[y, x] = arr[y + dy, x + dx], fill outside the array."""
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    h, w = arr.shape
    y_dst = slice(max(0, -dy), min(h, h - dy))
    y_src = slice(max(0, dy), min(h, h + dy))
    x_dst = slice(max(0, -dx), min(w, w - dx))
    x_src = slice(max(0, dx), min(w, w + dx))
    out[y_dst, x_dst] = arr[y_src, x_src]
    return out


def allowed_neighbour_masks(skeleton: np.ndarray,
                            free: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    """Per-offset bool [H, W] mask: source pixel has an *allowed* neighbour.

    Orthogonal steps are always allowed (when the neighbour is skeleton).
    Diagonal steps additionally require both orthogonal side cells to be free
    (no corner cutting).
    """
    skeleton = np.asarray(skeleton) > 0
    free = np.asarray(free) > 0
    masks: Dict[Tuple[int, int], np.ndarray] = {}
    for dy, dx in ORTHO_OFFSETS:
        masks[(dy, dx)] = skeleton & _shift(skeleton, dy, dx, False)
    for dy, dx in DIAG_OFFSETS:
        side_y = _shift(free, dy, 0, False)      # free[y + dy, x]
        side_x = _shift(free, 0, dx, False)      # free[y, x + dx]
        masks[(dy, dx)] = skeleton & _shift(skeleton, dy, dx, False) & side_y & side_x
    return masks


def pixel_degrees(skeleton: np.ndarray, free: np.ndarray) -> np.ndarray:
    """No-corner-cut 8-neighbour degree of every skeleton pixel (uint8)."""
    skeleton = np.asarray(skeleton) > 0
    masks = allowed_neighbour_masks(skeleton, free)
    deg = np.zeros(skeleton.shape, dtype=np.uint8)
    for mask in masks.values():
        deg += mask.astype(np.uint8)
    deg[~skeleton] = 0
    return deg


def _neighbours_of(x: int, y: int, skeleton: np.ndarray, free: np.ndarray
                   ) -> List[Tuple[int, int]]:
    """Allowed neighbours of skeleton pixel (x, y) as (x, y) pairs."""
    h, w = skeleton.shape
    out: List[Tuple[int, int]] = []
    for dy, dx in ORTHO_OFFSETS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
            out.append((nx, ny))
    for dy, dx in DIAG_OFFSETS:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < h and 0 <= nx < w) or not skeleton[ny, nx]:
            continue
        if free[y + dy, x] and free[y, x + dx]:
            out.append((nx, ny))
    return out


# ---------------------------------------------------------------------------
# supercover line traversal (connectors)
# ---------------------------------------------------------------------------


def supercover_pixels(p0: Sequence[float], p1: Sequence[float]
                      ) -> List[Tuple[int, int]]:
    """Cells intersected by the segment p0 -> p1 (pixel centre coords).

    Amanatides-Woo traversal in cell space with supercover corner handling: when
    the segment crosses a cell corner exactly, *both* orthogonal cells are
    emitted, so the returned chain is always 4-connected and can never squeeze
    through a diagonal gap between two obstacles.
    """
    x0, y0 = float(p0[0]) + 0.5, float(p0[1]) + 0.5
    x1, y1 = float(p1[0]) + 0.5, float(p1[1]) + 0.5
    i, j = int(math.floor(x0)), int(math.floor(y0))
    i1, j1 = int(math.floor(x1)), int(math.floor(y1))
    cells: List[Tuple[int, int]] = [(i, j)]
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return cells

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    inf = float("inf")
    if dx > 0:
        t_max_x = (i + 1 - x0) / dx
    elif dx < 0:
        t_max_x = (i - x0) / dx
    else:
        t_max_x = inf
    if dy > 0:
        t_max_y = (j + 1 - y0) / dy
    elif dy < 0:
        t_max_y = (j - y0) / dy
    else:
        t_max_y = inf
    t_delta_x = abs(1.0 / dx) if dx != 0 else inf
    t_delta_y = abs(1.0 / dy) if dy != 0 else inf

    guard = 4 * (abs(i1 - i) + abs(j1 - j) + 2) + 16
    while (i, j) != (i1, j1):
        guard -= 1
        if guard < 0:  # pragma: no cover - numerical safety net
            break
        if abs(t_max_x - t_max_y) < 1e-12:
            i += step_x
            cells.append((i, j))
            j += step_y
            cells.append((i, j))
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            i += step_x
            cells.append((i, j))
            t_max_x += t_delta_x
        else:
            j += step_y
            cells.append((i, j))
            t_max_y += t_delta_y
    return cells


def supercover_is_free(p0: Sequence[float], p1: Sequence[float],
                       free: np.ndarray) -> bool:
    """True when every cell touched by the segment is inside and free."""
    free = np.asarray(free) > 0
    h, w = free.shape
    for i, j in supercover_pixels(p0, p1):
        if not (0 <= i < w and 0 <= j < h) or not free[j, i]:
            return False
    return True


# ---------------------------------------------------------------------------
# compressed graph
# ---------------------------------------------------------------------------


@dataclass
class SkeletonNode:
    """One structure node = one 8-connected cluster of degree != 2 pixels."""

    idx: int
    kind: str                                   # junction | endpoint | auxiliary
    pixels: List[Tuple[int, int]]               # (x, y) cluster pixels
    center: Tuple[float, float]                 # centroid, pixel coords
    anchor: Tuple[int, int]                     # representative pixel
    _path_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = field(
        default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.pixels)


@dataclass
class SkeletonBranch:
    """A compressed edge between two structure nodes, with its full polyline."""

    idx: int
    u: int                                      # node id at pixels[0]
    v: int                                      # node id at pixels[-1]
    pixels: List[Tuple[int, int]]
    length: float                               # pixel-centre euclidean length

    def polyline(self, from_node: int) -> List[Tuple[int, int]]:
        if from_node == self.u:
            return list(self.pixels)
        if from_node == self.v:
            return list(reversed(self.pixels))
        raise ValueError("branch %d is not incident to node %d"
                         % (self.idx, from_node))


class SkeletonGraph:
    """Compressed skeleton graph (structure nodes + branch polylines)."""

    def __init__(self, res: int, free: np.ndarray, skeleton: np.ndarray,
                 nodes: List[SkeletonNode], branches: List[SkeletonBranch],
                 node_of_pixel: np.ndarray, stats: Optional[dict] = None):
        self.res = int(res)
        self.free = free
        self.skeleton = skeleton
        self.nodes = nodes
        self.branches = branches
        self.node_of_pixel = node_of_pixel
        self.stats = dict(stats or {})
        self._incident: Dict[int, List[int]] = {n.idx: [] for n in nodes}
        for b in branches:
            self._incident[b.u].append(b.idx)
            if b.v != b.u:
                self._incident[b.v].append(b.idx)
        self._expanded = None

    # -------------------------------------------------------------- geometry
    @property
    def cell(self) -> float:
        """Scene units per pixel (the map spans scene [-1, 1] per axis)."""
        return 2.0 / float(self.res)

    def pixel_to_scene(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return (xy + 0.5) * self.cell - 1.0

    def scene_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return (xy + 1.0) / self.cell - 0.5

    def node(self, idx: int) -> SkeletonNode:
        return self.nodes[idx]

    def incident_branches(self, idx: int) -> List[int]:
        return self._incident[idx]

    def branch_between(self, a: int, b: int) -> Optional[int]:
        """Shortest branch joining nodes a and b (parallel edges: min)."""
        best, best_len = None, float("inf")
        for bid in self._incident.get(a, ()):
            br = self.branches[bid]
            if (br.u == a and br.v == b) or (br.u == b and br.v == a):
                if br.length < best_len:
                    best, best_len = bid, br.length
        return best

    def branch_scene(self, bid: int, from_node: int) -> np.ndarray:
        pix = np.asarray(self.branches[bid].polyline(from_node), dtype=np.float64)
        return self.pixel_to_scene(pix)

    # ---------------------------------------------------------------- networkx
    def expanded_graph(self, anchors_start=None, anchors_goal=None):
        """Branch-expanded graph, optionally with a super source / sink.

        Parallel branches are PRESERVED by giving every branch its own node:

            junction -- branch-node -- junction

        so two branches joining the same junction pair become two distinct
        paths and the K-shortest search can return them as separate topologies
        (a plain nx.Graph would silently keep only the shorter one).

        anchors_start / anchors_goal are lists of (node_idx, cells, length_px)
        from visible_anchors().  When given, a super source / sink is wired to
        EVERY visible anchor with the connector length as the edge weight, so
        the start/goal attachment is chosen jointly with the route instead of
        being frozen to the single nearest node.

        Returns (graph, source_node_or_None, sink_node_or_None).
        """
        import networkx as nx

        g = nx.Graph()
        for n in self.nodes:
            g.add_node(("n", int(n.idx)))
        for br in self.branches:
            bnode = ("b", int(br.idx))
            g.add_node(bnode)
            half = 0.5 * float(br.length)
            g.add_edge(("n", int(br.u)), bnode, weight=half)
            g.add_edge(bnode, ("n", int(br.v)), weight=half)

        src = sink = None
        if anchors_start:
            src = ("super", "source")
            g.add_node(src)
            for node_idx, _cells, length in anchors_start:
                g.add_edge(src, ("n", int(node_idx)), weight=float(length))
        if anchors_goal:
            sink = ("super", "sink")
            g.add_node(sink)
            for node_idx, _cells, length in anchors_goal:
                g.add_edge(sink, ("n", int(node_idx)), weight=float(length))
        return g, src, sink

    @staticmethod
    def decode_expanded_path(path):
        """Expanded node path -> (junction node ids, branch ids)."""
        junctions: List[int] = []
        branches: List[int] = []
        for item in path:
            tag = item[0]
            if tag == "n":
                junctions.append(int(item[1]))
            elif tag == "b":
                branches.append(int(item[1]))
        return junctions, branches

    # ------------------------------------------------------------------ routes
    def in_cluster_path(self, node_idx: int, a: Tuple[int, int],
                        b: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Shortest no-corner-cut path inside one node cluster (pixels).

        Returns None when no safe path exists.  the planner forbids the former
        [a, b] fallback: joining two diagonal cluster pixels that a corner cut
        separates would emit an unsafe connector, so the whole candidate must
        be dropped instead of silently repaired.
        """
        if a == b:
            return [a]
        node = self.nodes[node_idx]
        cache_key = (a, b)
        if cache_key in node._path_cache:
            cached = node._path_cache[cache_key]
            return None if cached is None else list(cached)
        allowed = set(node.pixels)
        from collections import deque

        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {a: None}
        queue = deque([a])
        found = False
        while queue:
            cur = queue.popleft()
            if cur == b:
                found = True
                break
            for nxt in _neighbours_of(cur[0], cur[1], self.skeleton, self.free):
                if nxt in allowed and nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)
        if not found:
            node._path_cache[cache_key] = None
            return None
        path: List[Tuple[int, int]] = []
        cur2: Optional[Tuple[int, int]] = b
        while cur2 is not None:
            path.append(cur2)
            cur2 = prev[cur2]
        path.reverse()
        node._path_cache[cache_key] = list(path)
        return path

    # ------------------------------------------------------------------ anchor
    def connector(self, point_scene: Sequence[float], node_idx: int):
        """Safe connector from a scene point to one node, or None.

        Returns (cells, length_px); the length follows exactly the polyline the
        route assembler will emit (the exact start point plus the cell chain,
        the first cell being the one that contains the start point).
        """
        p = self.scene_to_pixel(point_scene)
        node = self.nodes[int(node_idx)]
        cells = supercover_pixels(p, node.anchor)
        h, w = self.free.shape
        for cx, cy in cells:
            if not (0 <= cx < w and 0 <= cy < h) or not self.free[cy, cx]:
                return None
        pts = np.vstack([p[None, :], np.asarray(cells[1:], dtype=np.float64)])             if len(cells) > 1 else p[None, :]
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())             if len(pts) > 1 else 0.0
        return cells, length

    def visible_anchors(self, point_scene: Sequence[float],
                        max_candidates: int = 16):
        """ALL visible skeleton nodes for a scene point, nearest first.

        Returns a list of (node_idx, connector_cells, connector_length_px).
        Every one of them becomes an edge of the super source/sink, so several
        start/goal attachments can compete inside the same K-shortest search.
        """
        if not self.nodes:
            return []
        p = self.scene_to_pixel(point_scene)
        anchors = np.asarray([n.anchor for n in self.nodes], dtype=np.float64)
        d = np.linalg.norm(anchors - p[None, :], axis=1)
        out = []
        for pos in np.argsort(d)[:max(1, int(max_candidates))]:
            node_idx = int(pos)
            got = self.connector(point_scene, node_idx)
            if got is not None:
                cells, length = got
                out.append((node_idx, cells, length))
        return out

    def anchor_point(self, point_scene: Sequence[float], max_candidates: int = 16
                     ) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
        """Nearest visible node (thin wrapper over visible_anchors)."""
        anchors = self.visible_anchors(point_scene, max_candidates)
        if not anchors:
            return None
        node_idx, cells, _length = anchors[0]
        return node_idx, cells


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------


def _components(skeleton: np.ndarray, free: np.ndarray):
    """Connected-component labels of the skeleton under no-corner-cut adjacency."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    h, w = skeleton.shape
    masks = allowed_neighbour_masks(skeleton, free)
    rows, cols = [], []
    for (dy, dx), mask in masks.items():
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        rows.append(ys * w + xs)
        cols.append((ys + dy) * w + (xs + dx))
    if not rows:
        # No allowed adjacency anywhere: every pixel is its own component.
        labels = np.full((h, w), -1, dtype=np.int64)
        labels[skeleton] = np.arange(int(skeleton.sum()))
        return labels, int(skeleton.sum())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(rows.size, dtype=np.int8)
    n = h * w
    mat = coo_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(mat, directed=False)
    labels = labels.reshape(h, w).astype(np.int64)
    labels[~skeleton] = -1
    return labels, int(n_comp)


def _structural_clusters(structural: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Cluster adjacent structure pixels with the no-corner-cut rule.

    Plain 8-connected clustering would merge two structure pixels that only
    touch diagonally across an obstacle corner, silently joining two regions
    that are not actually connected.  The cluster labels are contiguous
    (0 .. K-1) and -1 outside the structural mask.
    """
    labels, _ = _components(structural, free)
    out = np.full(structural.shape, -1, dtype=np.int64)
    for new_id, v in enumerate(np.unique(labels[structural]).tolist()):
        out[labels == v] = new_id
    return out


def build_skeleton_graph(occ: np.ndarray, safety_dilation_cells: int = 1,
                         thinning_backend: str = "auto",
                         pure_cycle_aux_nodes: int = 2,
                         skeleton: Optional[np.ndarray] = None,
                         free: Optional[np.ndarray] = None) -> SkeletonGraph:
    """Full pipeline: occupancy -> free mask -> skeleton -> compressed graph.

    Parameters
    ----------
    occ
        occupancy map [H, W] with 1 = obstacle.
    safety_dilation_cells
        obstacle dilation applied before thinning (keeps skeleton centres away
        from walls; 1 is the default).
    skeleton, free
        optional pre-computed stages (used by the offline cache) - when given,
        occ is only used for the resolution.
    """
    occ = np.asarray(occ)
    if occ.ndim != 2 or occ.shape[0] != occ.shape[1]:
        raise ValueError("occupancy map must be a square 2D array, got %s"
                         % (occ.shape,))
    if free is None:
        # Two masks on purpose:
        #   free_thin - obstacles dilated by safety_dilation_cells; thinning it
        #               keeps every skeleton pixel (and therefore every ellipse
        #               centre) at that margin from the walls.
        #   free      - the true (undilated) free space; this is the collision
        #               mask used by the no-corner-cut rule, the connectors and
        #               every safety check.  A start/goal that sits right next
        #               to a wall is still reachable.
        free = free_mask_from_occupancy(occ, 0)
        free_thin = free_mask_from_occupancy(occ, safety_dilation_cells)
    else:
        free = np.asarray(free) > 0
        free_thin = free
    if skeleton is None:
        skeleton = skeletonize_free(free_thin, backend=thinning_backend)
    else:
        skeleton = np.asarray(skeleton) > 0
    res = int(occ.shape[0])
    h, w = skeleton.shape

    degree = pixel_degrees(skeleton, free)
    structural = skeleton & (degree != 2)

    # ---- pure cycles: no degree-2 chain, no endpoint, no junction ----------
    labels, _ = _components(skeleton, free)
    skeleton_labels = np.unique(labels[skeleton]) if skeleton.any() else np.zeros(0, np.int64)
    aux_injected = 0
    for comp in skeleton_labels.tolist():
        sel = labels == comp
        if structural[sel].any():
            continue
        ys, xs = np.nonzero(sel)
        order = np.lexsort((xs, ys))
        picks = [0] if int(pure_cycle_aux_nodes) == 1 else [0, len(order) // 2]
        for k in picks:
            y, x = int(ys[order[k]]), int(xs[order[k]])
            structural[y, x] = True
            aux_injected += 1

    cluster_labels = _structural_clusters(structural, free)
    n_clusters = int(cluster_labels.max()) + 1
    node_of_pixel = np.full((h, w), -1, dtype=np.int64)
    nodes: List[SkeletonNode] = []
    for cid in range(n_clusters):
        ys, xs = np.nonzero(cluster_labels == cid)
        pixels = [(int(x), int(y)) for y, x in zip(ys, xs)]
        center = (float(xs.mean()), float(ys.mean()))
        d = np.hypot(xs - center[0], ys - center[1])
        ay, ax = int(ys[int(np.argmin(d))]), int(xs[int(np.argmin(d))])
        deg_max = int(degree[ys, xs].max())
        if deg_max >= 3:
            kind = "junction"
        elif deg_max == 1:
            kind = "endpoint"
        else:
            kind = "auxiliary"
        idx = len(nodes)
        nodes.append(SkeletonNode(idx=idx, kind=kind, pixels=pixels,
                                  center=center, anchor=(ax, ay)))
        for x, y in pixels:
            node_of_pixel[y, x] = idx

    # ---- branch tracing ----------------------------------------------------
    is_struct = structural
    branches: List[SkeletonBranch] = []
    seen = set()

    def node_at(p):
        return int(node_of_pixel[p[1], p[0]])

    for y, x in zip(*np.nonzero(is_struct)):
        p = (int(x), int(y))
        for q in _neighbours_of(p[0], p[1], skeleton, free):
            if is_struct[q[1], q[0]]:
                if node_at(p) == node_at(q):
                    continue                      # intra-cluster adjacency
                key = frozenset((p, q))
                if key in seen:
                    continue
                seen.add(key)
                chain = [p, q]
            else:
                chain = [p, q]
                prev, cur = p, q
                while True:
                    nxt = [n for n in _neighbours_of(cur[0], cur[1], skeleton, free)
                           if n != prev]
                    if not nxt:
                        break                     # pragma: no cover - safety
                    step = nxt[0]
                    chain.append(step)
                    if is_struct[step[1], step[0]]:
                        break
                    prev, cur = cur, step
                key = frozenset(chain)
                if key in seen:
                    continue
                seen.add(key)
            length = 0.0
            for a, b in zip(chain[:-1], chain[1:]):
                length += math.hypot(b[0] - a[0], b[1] - a[1])
            branches.append(SkeletonBranch(
                idx=len(branches), u=node_at(chain[0]), v=node_at(chain[-1]),
                pixels=chain, length=float(length)))

    # Degenerate self-loops (two structure pixels of one cluster linked by a
    # single chain pixel) are artefacts of the thinning ladder at 90 degree
    # bends; they can never be part of a start -> goal route, so drop them.
    keep = [b for b in branches if b.u != b.v]
    for i, b in enumerate(keep):
        b.idx = i

    stats = {
        "res": res,
        "free_cells": int(free.sum()),
        "thinned_free_cells": int(free_thin.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "structural_pixels": int(structural.sum()),
        "nodes": len(nodes),
        "branches": len(keep),
        "aux_nodes": int(aux_injected),
        "components": int(skeleton_labels.size),
        "node_kinds": {
            k: sum(1 for n in nodes if n.kind == k)
            for k in ("junction", "endpoint", "auxiliary")
        },
        "branch_length_px": {
            "min": float(min((b.length for b in keep), default=0.0)),
            "max": float(max((b.length for b in keep), default=0.0)),
        },
    }
    return SkeletonGraph(res=res, free=free, skeleton=skeleton, nodes=nodes,
                         branches=keep, node_of_pixel=node_of_pixel, stats=stats)


def anchor_point(graph: SkeletonGraph, point_scene: Sequence[float],
                 max_candidates: int = 16):
    """Module-level alias of SkeletonGraph.anchor_point."""
    return graph.anchor_point(point_scene, max_candidates=max_candidates)


# ---------------------------------------------------------------------------
# persistence (offline cache -> training / inference)
# ---------------------------------------------------------------------------


def save_graph_npz(graph: SkeletonGraph, path: str) -> str:
    """Serialize a SkeletonGraph to a compressed .npz (no pickle)."""
    node_offsets = np.cumsum([0] + [len(n.pixels) for n in graph.nodes])
    node_pix = np.asarray([p for n in graph.nodes for p in n.pixels],
                          dtype=np.int32).reshape(-1, 2)
    branch_offsets = np.cumsum([0] + [len(b.pixels) for b in graph.branches])
    branch_pix = np.asarray([p for b in graph.branches for p in b.pixels],
                            dtype=np.int32).reshape(-1, 2)
    np.savez_compressed(
        path,
        res=np.int32(graph.res),
        free=graph.free.astype(np.uint8),
        skeleton=graph.skeleton.astype(np.uint8),
        node_offsets=node_offsets.astype(np.int64),
        node_pix=node_pix,
        node_kind=np.asarray([n.kind for n in graph.nodes], dtype="U16"),
        node_anchor=np.asarray([n.anchor for n in graph.nodes],
                               dtype=np.int32).reshape(-1, 2),
        branch_u=np.asarray([b.u for b in graph.branches], dtype=np.int32),
        branch_v=np.asarray([b.v for b in graph.branches], dtype=np.int32),
        branch_length=np.asarray([b.length for b in graph.branches], dtype=np.float64),
        branch_offsets=branch_offsets.astype(np.int64),
        branch_pix=branch_pix,
        stats=np.asarray([json.dumps(graph.stats)], dtype="U4096"),
    )
    return path


def load_graph_npz(path: str) -> SkeletonGraph:
    """Inverse of save_graph_npz."""
    with np.load(path, allow_pickle=False) as z:
        res = int(z["res"])
        free = z["free"].astype(bool)
        skeleton = z["skeleton"].astype(bool)
        node_offsets = z["node_offsets"]
        node_pix = z["node_pix"]
        node_kind = z["node_kind"]
        node_anchor = z["node_anchor"]
        nodes: List[SkeletonNode] = []
        for i in range(len(node_kind)):
            lo, hi = int(node_offsets[i]), int(node_offsets[i + 1])
            pixels = [(int(x), int(y)) for x, y in node_pix[lo:hi]]
            ax, ay = int(node_anchor[i, 0]), int(node_anchor[i, 1])
            if pixels:
                center = (float(np.mean([p[0] for p in pixels])),
                          float(np.mean([p[1] for p in pixels])))
            else:
                center = (float(ax), float(ay))
            nodes.append(SkeletonNode(idx=i, kind=str(node_kind[i]), pixels=pixels,
                                      center=center, anchor=(ax, ay)))
        branch_offsets = z["branch_offsets"]
        branch_pix = z["branch_pix"]
        branch_u = z["branch_u"]
        branch_v = z["branch_v"]
        branch_length = z["branch_length"]
        branches: List[SkeletonBranch] = []
        for i in range(len(branch_u)):
            lo, hi = int(branch_offsets[i]), int(branch_offsets[i + 1])
            pixels = [(int(x), int(y)) for x, y in branch_pix[lo:hi]]
            branches.append(SkeletonBranch(idx=i, u=int(branch_u[i]), v=int(branch_v[i]),
                                           pixels=pixels, length=float(branch_length[i])))
        node_of_pixel = np.full(skeleton.shape, -1, dtype=np.int64)
        for n in nodes:
            for x, y in n.pixels:
                node_of_pixel[y, x] = n.idx
        stats_raw = z["stats"]
        stats = json.loads(str(stats_raw[0])) if stats_raw.size else {}
    return SkeletonGraph(res=res, free=free, skeleton=skeleton, nodes=nodes,
                         branches=branches, node_of_pixel=node_of_pixel, stats=stats)
