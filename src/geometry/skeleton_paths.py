"""V2 geometry step 2: safe candidate topologies on the compressed skeleton graph.

The graph algorithms in this module only produce *complete, connected, safe*
candidate routes between a start and a goal.  They never rank them: choosing a
route is the job of the learned topology selector.

Pipeline
--------
1. anchor the start/goal onto the nearest *visible* skeleton nodes with a
   supercover collision check (connectors are part of the route);
2. Yen K-shortest simple paths on the compressed branch graph;
3. keep routes with length <= max_length_ratio * L_min;
4. de-duplicate by branch-set Jaccard similarity (J > dedup_jaccard = same
   topology) and keep at most num_candidates;
5. arc-length resample every route to candidate_points samples and build the
   per-point features [x, y, tx, ty, u].

The very same function is used by the offline preprocessing script and by
online inference - there is deliberately only one implementation.

Ground-truth helpers
--------------------
soft_topology_target  : normalized-DTW soft distribution over candidates
progress_target       : monotone arc-length parameters of the GT waypoints on
                        the best candidate (isotonic regression)
interpolate_path      : gamma_m(s), the *only* definition of an ellipse centre
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .skeleton_graph import SkeletonGraph, supercover_pixels

__all__ = [
    "CandidateConfig",
    "SkeletonRoute",
    "CandidateSet",
    "route_between",
    "routes_k_shortest",
    "resample_polyline",
    "path_features",
    "branch_jaccard",
    "generate_candidates",
    "normalized_dtw",
    "soft_topology_target",
    "progress_target",
    "isotonic_nondecreasing",
    "interpolate_path",
    "nearest_arclength",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class CandidateConfig:
    """Every tunable of the candidate generator (config: skeleton/topology)."""

    num_candidates: int = 4
    raw_k: int = 16
    max_length_ratio: float = 1.5
    dedup_jaccard: float = 0.75
    candidate_points: int = 128
    anchor_candidates: int = 16
    tau_gt: float = 0.05

    @classmethod
    def from_dict(cls, cfg: Optional[dict], strict: bool = True) -> "CandidateConfig":
        """Build from a config dict.

        The config file keeps the generator keys and the selector/sampler keys
        (commit_t, selection, ...) under the same 'topology' section, so the
        caller can pass strict=False to ignore the non-generator keys.
        """
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(cfg) - known
        if unknown and strict:
            raise ValueError("unknown candidate config keys: %s" % sorted(unknown))
        out = cls(**{k: v for k, v in cfg.items() if k in known})
        out.validate()
        return out

    def validate(self) -> None:
        if self.num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if self.raw_k < self.num_candidates:
            raise ValueError("raw_k must be >= num_candidates")
        if self.max_length_ratio < 1.0:
            raise ValueError("max_length_ratio must be >= 1")
        if not 0.0 <= self.dedup_jaccard <= 1.0:
            raise ValueError("dedup_jaccard must be in [0, 1]")
        if self.candidate_points < 2:
            raise ValueError("candidate_points must be >= 2")
        if self.anchor_candidates < 1:
            raise ValueError("anchor_candidates must be >= 1")
        if self.tau_gt <= 0.0:
            raise ValueError("tau_gt must be > 0")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@dataclass
class SkeletonRoute:
    """One complete, collision-free start -> goal route."""

    pixels: np.ndarray                 # [L, 2] float pixel-centre coords
    scene: np.ndarray                  # [L, 2] float scene coords
    length: float                      # scene-unit arc length
    node_path: Tuple[int, ...]
    branch_ids: Tuple[int, ...]
    start_node: int
    goal_node: int


def _dedupe_consecutive(points: List[Tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(arr), dtype=bool)
    if len(arr) > 1:
        keep[1:] = np.any(np.abs(np.diff(arr, axis=0)) > 1e-9, axis=1)
    return arr[keep]


def _assemble_route(graph: SkeletonGraph, node_path: Sequence[int],
                    conn_start: Sequence[Tuple[int, int]],
                    conn_goal: Sequence[Tuple[int, int]],
                    start_pixel: Sequence[float],
                    goal_pixel: Sequence[float],
                    start_scene: Sequence[float],
                    goal_scene: Sequence[float]) -> SkeletonRoute:
    """Concatenate connectors + branch polylines + in-cluster bridges."""
    cells: List[Tuple[int, int]] = list(conn_start)
    branch_ids: List[int] = []

    if len(node_path) == 1:
        bridge = graph.in_cluster_path(node_path[0], cells[-1], conn_goal[-1])
        cells.extend(bridge[1:])
    else:
        for a, b in zip(node_path[:-1], node_path[1:]):
            bid = graph.branch_between(int(a), int(b))
            if bid is None:  # pragma: no cover - defensive
                raise ValueError("no branch between nodes %s and %s" % (a, b))
            branch_ids.append(int(bid))
            pix = graph.branches[bid].polyline(int(a))
            bridge = graph.in_cluster_path(int(a), cells[-1], pix[0])
            cells.extend(bridge[1:])
            cells.extend(pix[1:])
        bridge = graph.in_cluster_path(int(node_path[-1]), cells[-1], conn_goal[-1])
        cells.extend(bridge[1:])

    cells.extend(list(conn_goal)[-2::-1])          # anchor -> ... -> goal cell

    pixel_points = [(float(start_pixel[0]), float(start_pixel[1]))]
    pixel_points.extend((float(c[0]), float(c[1])) for c in cells)
    pixel_points.append((float(goal_pixel[0]), float(goal_pixel[1])))
    pixels = _dedupe_consecutive(pixel_points)
    scene = graph.pixel_to_scene(pixels)
    seg = np.linalg.norm(np.diff(scene, axis=0), axis=1) if len(scene) > 1 else np.zeros(1)
    return SkeletonRoute(
        pixels=pixels, scene=scene, length=float(seg.sum()),
        node_path=tuple(int(v) for v in node_path),
        branch_ids=tuple(branch_ids),
        start_node=int(node_path[0]), goal_node=int(node_path[-1]),
    )


def route_between(graph: SkeletonGraph, start_scene: Sequence[float],
                  goal_scene: Sequence[float], anchor_candidates: int = 16
                  ) -> Optional[SkeletonRoute]:
    """Shortest safe route between two scene points, or None."""
    routes = routes_k_shortest(graph, start_scene, goal_scene, k=1,
                               anchor_candidates=anchor_candidates)
    return routes[0] if routes else None


def routes_k_shortest(graph: SkeletonGraph, start_scene: Sequence[float],
                      goal_scene: Sequence[float], k: int = 16,
                      anchor_candidates: int = 16) -> List[SkeletonRoute]:
    """Yen K-shortest simple paths, assembled into full start -> goal routes."""
    import networkx as nx

    a_s = graph.anchor_point(start_scene, anchor_candidates)
    if a_s is None:
        return []
    a_g = graph.anchor_point(goal_scene, anchor_candidates)
    if a_g is None:
        return []
    v_s, conn_s = a_s
    v_g, conn_g = a_g
    start_pixel = graph.scene_to_pixel(start_scene)
    goal_pixel = graph.scene_to_pixel(goal_scene)

    node_paths: List[List[int]] = []
    if v_s == v_g:
        node_paths.append([v_s])
    else:
        g = graph.nx_graph()
        try:
            for i, p in enumerate(nx.shortest_simple_paths(g, v_s, v_g, weight="weight")):
                node_paths.append([int(v) for v in p])
                if len(node_paths) >= int(k):
                    break
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:  # pragma: no cover - defensive
            return []

    routes = []
    for np_ in node_paths:
        try:
            routes.append(_assemble_route(graph, np_, conn_s, conn_g, start_pixel,
                                          goal_pixel, start_scene, goal_scene))
        except ValueError:  # pragma: no cover - defensive
            continue
    return routes


# ---------------------------------------------------------------------------
# geometry features
# ---------------------------------------------------------------------------


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Uniform arc-length resampling of a polyline to exactly n points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if n < 2:
        raise ValueError("n must be >= 2")
    if len(points) == 0:
        return np.zeros((n, 2), dtype=np.float64)
    points = _dedupe_consecutive([tuple(p) for p in points])
    if len(points) == 1:
        return np.repeat(points, n, axis=0)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(points[:1], n, axis=0)
    target = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = np.interp(target, cum, points[:, 0])
    out[:, 1] = np.interp(target, cum, points[:, 1])
    return out


def path_features(coords: np.ndarray) -> np.ndarray:
    """[L, 2] scene polyline -> [L, 5] = [x, y, tx, ty, u]."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    n = len(coords)
    if n == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if n == 1:
        tan = np.zeros((1, 2), dtype=np.float64)
    else:
        tan = np.empty_like(coords)
        tan[1:-1] = coords[2:] - coords[:-2]
        tan[0] = coords[1] - coords[0]
        tan[-1] = coords[-1] - coords[-2]
    norm = np.linalg.norm(tan, axis=1, keepdims=True)
    tan = tan / np.maximum(norm, 1e-9)
    if n == 1:
        u = np.zeros(1, dtype=np.float64)
    else:
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        u = cum / total if total > 1e-12 else np.zeros(n)
    out = np.concatenate([coords, tan, u[:, None]], axis=1)
    return out.astype(np.float32)


def branch_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(v) for v in a), set(int(v) for v in b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 1.0


# ---------------------------------------------------------------------------
# candidate set
# ---------------------------------------------------------------------------


@dataclass
class CandidateSet:
    """Fixed-size padded candidate topology set for one OD pair."""

    paths: np.ndarray                  # [M, L, 5] float32 features
    coords: np.ndarray                 # [M, L, 2] float32 scene coordinates
    mask: np.ndarray                   # [M] bool
    lengths: np.ndarray                # [M] float32 scene arc length
    node_paths: List[Tuple[int, ...]] = field(default_factory=list)
    branch_ids: List[Tuple[int, ...]] = field(default_factory=list)
    raw_k: int = 0

    @property
    def num_slots(self) -> int:
        return int(self.mask.shape[0])

    @property
    def num_valid(self) -> int:
        return int(self.mask.sum())

    def valid_index(self) -> np.ndarray:
        return np.nonzero(self.mask)[0]

    @staticmethod
    def empty(num_candidates: int, num_points: int) -> "CandidateSet":
        return CandidateSet(
            paths=np.zeros((num_candidates, num_points, 5), dtype=np.float32),
            coords=np.zeros((num_candidates, num_points, 2), dtype=np.float32),
            mask=np.zeros(num_candidates, dtype=bool),
            lengths=np.zeros(num_candidates, dtype=np.float32),
        )


def generate_candidates(graph: SkeletonGraph, start_scene: Sequence[float],
                        goal_scene: Sequence[float],
                        cfg: Optional[CandidateConfig] = None) -> CandidateSet:
    """Full candidate generator (shared by preprocessing and inference)."""
    cfg = cfg or CandidateConfig()
    if not cfg.num_candidates:
        cfg.validate()
    n_slots, n_pts = int(cfg.num_candidates), int(cfg.candidate_points)

    raw = routes_k_shortest(graph, start_scene, goal_scene, k=int(cfg.raw_k),
                            anchor_candidates=int(cfg.anchor_candidates))
    if not raw:
        return CandidateSet.empty(n_slots, n_pts)

    raw.sort(key=lambda r: r.length)
    l_min = raw[0].length

    accepted: List[SkeletonRoute] = []
    for route in raw:
        if route.length > cfg.max_length_ratio * l_min + 1e-12:
            continue
        if any(branch_jaccard(route.branch_ids, a.branch_ids) > cfg.dedup_jaccard
               for a in accepted):
            continue
        accepted.append(route)
        if len(accepted) >= n_slots:
            break

    out = CandidateSet.empty(n_slots, n_pts)
    out.raw_k = len(raw)
    for i, route in enumerate(accepted):
        coords = resample_polyline(route.scene, n_pts)
        out.coords[i] = coords.astype(np.float32)
        out.paths[i] = path_features(coords)
        out.mask[i] = True
        out.lengths[i] = float(route.length)
        out.node_paths.append(route.node_path)
        out.branch_ids.append(route.branch_ids)
    return out


# ---------------------------------------------------------------------------
# ground-truth helpers (offline labels)
# ---------------------------------------------------------------------------


def normalized_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized DTW distance between two polylines (scene units).

    DTW(n, m) accumulated with the Euclidean ground cost, divided by
    (n + m); the value is therefore on the scale of an average point-to-point
    distance and can be compared with tau_gt directly.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    inf = float("inf")
    # Anti-diagonal (wavefront) DP: every cell of one anti-diagonal depends only
    # on the two previous ones, so each diagonal is computed with one vectorised
    # numpy expression instead of an n*m Python loop.
    acc = np.full((n + 1, m + 1), inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for k in range(1, n + m + 1):
        i_lo, i_hi = max(1, k - m), min(n, k - 1)
        if i_lo > i_hi:
            continue
        i = np.arange(i_lo, i_hi + 1)
        j = k - i
        best = np.minimum(np.minimum(acc[i - 1, j], acc[i, j - 1]),
                          acc[i - 1, j - 1])
        acc[i, j] = d[i - 1, j - 1] + best
    return float(acc[n, m]) / float(n + m)


def soft_topology_target(gt_traj: np.ndarray, candidates: CandidateSet,
                         tau: Optional[float] = None) -> np.ndarray:
    """Soft categorical target q_m = softmax(-nDTW(gt, P_m) / tau).

    A soft target avoids the one-hot supervision that would destroy the
    multimodality of the diffusion prior when two candidate topologies are
    almost equally consistent with the demonstration.
    """
    tau = float(tau if tau is not None else CandidateConfig().tau_gt)
    q = np.zeros(candidates.num_slots, dtype=np.float64)
    idx = candidates.valid_index()
    if idx.size == 0:
        return q
    d = np.array([normalized_dtw(gt_traj, candidates.coords[i]) for i in idx])
    finite = np.isfinite(d)
    if not finite.any():
        q[idx] = 1.0 / float(idx.size)
        return q
    d = d - d[finite].min()
    w = np.zeros_like(d)
    w[finite] = np.exp(-d[finite] / tau)
    total = w.sum()
    if total <= 0.0:
        q[idx] = 1.0 / float(idx.size)
    else:
        q[idx] = w / total
    return q


def isotonic_nondecreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators projection onto non-decreasing sequences."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 0:
        return y
    level = np.empty(n, dtype=np.float64)
    weight = np.empty(n, dtype=np.int64)
    j = 0
    for i in range(n):
        level[j] = y[i]
        weight[j] = 1
        j += 1
        while j > 1 and level[j - 2] > level[j - 1]:
            w = weight[j - 2] + weight[j - 1]
            level[j - 2] = (weight[j - 2] * level[j - 2]
                            + weight[j - 1] * level[j - 1]) / float(w)
            weight[j - 2] = w
            j -= 1
    out = np.empty(n, dtype=np.float64)
    k = 0
    for i in range(j):
        size = int(weight[i])
        out[k:k + size] = level[i]
        k += size
    return out


def nearest_arclength(points: np.ndarray, polyline: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Project points onto a polyline; returns (s in [0,1], distance)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    if len(poly) == 1:
        d = np.linalg.norm(points - poly[0], axis=1)
        return np.zeros(len(points)), d
    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    best_s = np.zeros(len(points))
    best_d = np.full(len(points), np.inf)
    for i in range(len(seg)):
        if seg_len[i] <= 1e-12:
            continue
        rel = points - poly[i][None, :]
        t = (rel @ seg[i]) / (seg_len[i] ** 2)
        t = np.clip(t, 0.0, 1.0)
        proj = poly[i][None, :] + t[:, None] * seg[i][None, :]
        d = np.linalg.norm(points - proj, axis=1)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_s = np.where(better, (cum[i] + t * seg_len[i]), best_s)
    if total <= 1e-12:
        return np.zeros(len(points)), best_d
    return best_s / total, best_d


def progress_target(gt_traj: np.ndarray, polyline: np.ndarray,
                    num_points: int) -> np.ndarray:
    """Monotone arc-length parameters of the GT waypoints on one candidate.

    The raw projections can move slightly backwards on a noisy demonstration,
    so they are projected onto the non-decreasing cone (isotonic regression)
    and pinned to s_0 = 0 and s_{K-1} = 1.
    """
    s_raw, _ = nearest_arclength(gt_traj, polyline)
    s = isotonic_nondecreasing(np.clip(s_raw, 0.0, 1.0))
    if len(s):
        s[0] = 0.0
        s[-1] = 1.0
        s = isotonic_nondecreasing(s)
        s[0] = 0.0
        s[-1] = 1.0
    if len(s) != int(num_points):
        src = np.linspace(0.0, 1.0, len(s)) if len(s) else np.zeros(1)
        s = np.interp(np.linspace(0.0, 1.0, int(num_points)), src, s)
    return s.astype(np.float64)


def interpolate_path(polyline: np.ndarray, s: np.ndarray) -> np.ndarray:
    """gamma_m(s): points on the polyline at normalized arc length s."""
    poly = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    s = np.asarray(s, dtype=np.float64)
    if len(poly) == 0:
        raise ValueError("empty polyline")
    if len(poly) == 1:
        return np.repeat(poly, s.shape[0], axis=0)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(poly[:1], s.shape[0], axis=0)
    target = np.clip(s, 0.0, 1.0) * total
    out = np.empty((s.shape[0], 2), dtype=np.float64)
    out[:, 0] = np.interp(target, cum, poly[:, 0])
    out[:, 1] = np.interp(target, cum, poly[:, 1])
    return out
