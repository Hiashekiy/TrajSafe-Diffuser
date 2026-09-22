"""Frozen safety corridor built from the 128 predicted ellipse regions.

Pipeline (report sections 5-9):

    128 local convex regions
        -> polygon conversion
        -> overlap ratio between consecutive cells
        -> point-seeded gap bridge where the overlap is insufficient
        -> ordered corridor

This module does geometry ONLY.  It contains no ALM, no Lagrangian, no
B-spline math; the corridor is handed to :mod:`src.geometry.bspline_constraints`
and then to :mod:`src.diffusion.bspline_alm`.

The gap bridge follows the GCOPTER ``sfc_gen.hpp`` ``convexCover()`` idea: when
two consecutive polytopes do not connect well enough, an EXTRA point-seeded
polytope is inserted at the transition point.  GCOPTER's base cells are
segment-seeded and therefore already carry a transition point ``a``; this
project's base cells are 128 independent ellipse-centred regions, so the seed is
explicitly the SELECTED dense Skeleton point at the progress midpoint

    s_b = (s_i + s_{i+1}) / 2,      c_b = Gamma(s_b)

which is an adaptation to the dense Skeleton, NOT the GCOPTER midpoint formula.
Euclidean midpoints are forbidden: on a hairpin they can cut an obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from .convex_region import EllipseRegionBuilder, halfspaces_to_vertices

__all__ = [
    "SCENE_TO_METER",
    "CorridorCell",
    "SafetyCorridor",
    "polygon_area",
    "convex_polygon_intersection",
    "overlap_ratio",
    "progress_alignment_stats",
    "build_bridge_region",
    "build_safety_corridor",
]

# The CARLA crops span 80 m mapped onto scene ``[-1, 1]``, i.e. 40 m per unit.
SCENE_TO_METER = 40.0

_AREA_EPS = 1e-12


def _to_numpy(value) -> np.ndarray:
    """Accept torch tensors on ANY device (the builder runs on the GPU)."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float64)


# ---------------------------------------------------------------------------
# polygon helpers (Sutherland-Hodgman, all convex, all CCW)
# ---------------------------------------------------------------------------


def polygon_area(polygon) -> float:
    """Absolute area of a simple polygon given as ``[V,2]`` (shoelace)."""
    if polygon is None:
        return 0.0
    p = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside(x: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """Left-of test for the CCW clip edge ``a -> b``."""
    return (b[0] - a[0]) * (x[1] - a[1]) - (b[1] - a[1]) * (x[0] - a[0]) >= -tol


def _line_intersection(p: np.ndarray, q: np.ndarray, a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
    """Intersection of segment ``p->q`` with the infinite line ``a->b``."""
    r = q - p
    s = b - a
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-15:
        return q.copy()
    t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / den
    return p + t * r


def convex_polygon_intersection(poly_a, poly_b, tol: float = 1e-12):
    """Sutherland-Hodgman clipping of convex CCW ``poly_a`` by ``poly_b``.

    Returns an ``[V,2]`` array (possibly empty).  Both inputs must be convex and
    counter-clockwise, which is exactly what :func:`halfspaces_to_vertices`
    produces.
    """
    if poly_a is None or poly_b is None:
        return np.zeros((0, 2), dtype=np.float64)
    subject = np.asarray(poly_a, dtype=np.float64).reshape(-1, 2)
    clip = np.asarray(poly_b, dtype=np.float64).reshape(-1, 2)
    if len(subject) < 3 or len(clip) < 3:
        return np.zeros((0, 2), dtype=np.float64)

    output = subject
    for i in range(len(clip)):
        if len(output) == 0:
            return output
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        source = output
        output = []
        prev = source[-1]
        prev_in = _inside(prev, a, b, tol)
        for cur in source:
            cur_in = _inside(cur, a, b, tol)
            if cur_in:
                if not prev_in:
                    output.append(_line_intersection(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(_line_intersection(prev, cur, a, b))
            prev, prev_in = cur, cur_in
        output = np.asarray(output, dtype=np.float64).reshape(-1, 2)
    return output


def overlap_ratio(poly_a, poly_b) -> float:
    """``Area(A n B) / (min(Area(A), Area(B)) + eps)`` in ``[0, 1]``."""
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= _AREA_EPS or area_b <= _AREA_EPS:
        return 0.0
    inter = polygon_area(convex_polygon_intersection(poly_a, poly_b))
    return float(inter / (min(area_a, area_b) + _AREA_EPS))


def progress_alignment_stats(raw_curve, skeleton_centers):
    """V1 ``u <-> s`` correspondence diagnostics (report section 13).

    Compares ``P_i = C(i/(H-1))`` with ``c_i = Gamma(i/(H-1))``.  Recorded ONLY; the
    mapping itself is never adapted from these numbers.
    """
    # sampler.py calls this with (None, None) when the selected candidate has
    # fewer than 2 dense points (e.g. nothing was selected and slot 0 is
    # invalid).  np.asarray(None) is a 0-d nan array, whose reshape(-1, 2)
    # raises "cannot reshape array of size 1 into shape (2)", so short-circuit.
    if raw_curve is None or skeleton_centers is None:
        return {"progress_alignment_rmse": None, "progress_alignment_max": None,
                "progress_alignment_rmse_m": None,
                "progress_alignment_max_m": None}
    p = np.asarray(raw_curve, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(skeleton_centers, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0 or len(p) != len(c):
        return {"progress_alignment_rmse": None, "progress_alignment_max": None,
                "progress_alignment_rmse_m": None,
                "progress_alignment_max_m": None}
    d = np.linalg.norm(p - c, axis=-1)
    rmse = float(np.sqrt((d ** 2).mean()))
    mx = float(d.max())
    return {"progress_alignment_rmse": rmse,
            "progress_alignment_max": mx,
            "progress_alignment_rmse_m": rmse * SCENE_TO_METER,
            "progress_alignment_max_m": mx * SCENE_TO_METER}


# ---------------------------------------------------------------------------
# corridor containers
# ---------------------------------------------------------------------------


@dataclass
class CorridorCell:
    """One ordered convex cell of the corridor.

    ``source`` is ``"network"`` for the 128 predicted-ellipse regions and
    ``"bridge"`` for the inserted point-seeded gap regions.
    """

    anchor_s: float
    center: np.ndarray                    # [2], absolute scene coordinates
    A: np.ndarray                         # [F,2]
    b: np.ndarray                         # [F]
    polygon: np.ndarray | None            # [V,2], CCW, or None
    source: str
    source_index: int | None
    valid: bool
    face_count: int = 0


@dataclass
class SafetyCorridor:
    cells: list                           # list[CorridorCell], sorted by anchor_s
    overlap_ratio: list                   # consecutive overlap ratios (len = M-1)
    base_cell_count: int
    bridge_cell_count: int
    valid: bool
    failure_reason: str | None = None
    min_overlap: float = 0.10
    bridge_gaps: list = field(default_factory=list)

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def anchors(self) -> np.ndarray:
        return np.asarray([c.anchor_s for c in self.cells], dtype=np.float64)

    def to_dict(self) -> dict:
        """Compact, JSON-friendly summary (the dashboard corridor channel)."""
        return {
            "valid": bool(self.valid),
            "failure_reason": self.failure_reason,
            "min_overlap": float(self.min_overlap),
            "base_cell_count": int(self.base_cell_count),
            "bridge_cell_count": int(self.bridge_cell_count),
            "num_cells": int(self.num_cells),
            "overlap_ratio": [float(v) for v in self.overlap_ratio],
            "bridge_gaps": list(self.bridge_gaps),
            "cells": [
                {
                    "i": int(i),
                    "source": cell.source,
                    "source_index": (-1 if cell.source_index is None
                                     else int(cell.source_index)),
                    "anchor_s": float(cell.anchor_s),
                    "center": [float(cell.center[0]), float(cell.center[1])],
                    "face_count": int(cell.face_count),
                    "valid": bool(cell.valid),
                    "polygon": ([] if cell.polygon is None
                                else np.round(np.asarray(cell.polygon,
                                                         dtype=np.float64), 5)
                                .tolist()),
                }
                for i, cell in enumerate(self.cells)
            ],
        }


# ---------------------------------------------------------------------------
# bridge
# ---------------------------------------------------------------------------


def build_bridge_region(builder: EllipseRegionBuilder, center) -> dict:
    """Point-seeded isotropic gap region at an absolute ``center`` [2].

    The bridge uses ``quadratic = I`` (no interpolated "bridge ellipse"): the
    Mahalanobis ordering degenerates to the Euclidean one, which is precisely a
    point-seeded local convex region.
    """
    c = torch.as_tensor(np.asarray(center, dtype=np.float32),
                        device=builder.maps.device,
                        dtype=builder.maps.dtype).reshape(1, 1, 2)
    quadratic = torch.eye(2, dtype=c.dtype, device=c.device).reshape(1, 1, 2, 2)
    quadratic = quadratic.expand(1, 1, 2, 2).contiguous()
    A, b, mask, valid, diag = builder.build_from_metric(
        c, quadratic, return_diagnostics=True)
    face = mask[0, 0]
    A_i = A[0, 0][face].detach().cpu().numpy().astype(np.float64)
    b_i = b[0, 0][face].detach().cpu().numpy().astype(np.float64)
    center_np = c[0, 0].detach().cpu().numpy().astype(np.float64)
    polygon = halfspaces_to_vertices(A_i, b_i, interior_point=center_np)
    ok = (bool(valid[0, 0]) and polygon is not None
          and polygon_area(polygon) > _AREA_EPS)
    return {"A": A_i, "b": b_i, "polygon": polygon, "valid": ok,
            "center": center_np, "face_count": int(face.sum()),
            "center_inside": bool(diag["center_inside"][0, 0])}


# ---------------------------------------------------------------------------
# corridor
# ---------------------------------------------------------------------------


def _default_gamma_fn():
    from ..models.trajsafe.geometry import gather_dense_path_points
    return gather_dense_path_points


def _base_cell(builder, A, b, mask, valid, centers, anchors, index):
    face = mask[index]
    A_i = A[index][face].detach().cpu().numpy().astype(np.float64)
    b_i = b[index][face].detach().cpu().numpy().astype(np.float64)
    center = np.asarray(centers[index], dtype=np.float64).reshape(2)
    polygon = halfspaces_to_vertices(A_i, b_i, interior_point=center)
    ok = bool(valid[index]) and polygon is not None \
        and polygon_area(polygon) > _AREA_EPS
    return CorridorCell(
        anchor_s=float(anchors[index]), center=center, A=A_i, b=b_i,
        polygon=polygon, source="network", source_index=int(index),
        valid=ok, face_count=int(face.sum()))


def build_safety_corridor(
    builder: EllipseRegionBuilder,
    centers,
    shape4,
    anchors,
    gamma=None,
    gamma_lengths=None,
    config: dict | None = None,
    gamma_fn: Callable | None = None,
) -> SafetyCorridor:
    """Build the ordered convex corridor from the 128 predicted ellipses.

    ``centers [H,2]`` and ``shape4 [H,4]`` are absolute; ``anchors [H]`` are the
    fixed progress values ``s_i = i/127``.  ``gamma [G,2]`` / ``gamma_lengths``
    describe the selected dense Skeleton used to place bridge seeds.

    Returns a :class:`SafetyCorridor`; ``valid == False`` with a
    ``failure_reason`` whenever the corridor cannot be closed.
    """
    cfg = dict(config or {})
    min_overlap = float(cfg.get("min_overlap_ratio", 0.10))
    bridge_cfg = dict(cfg.get("bridge") or {})
    bridge_enabled = bool(bridge_cfg.get("enabled", True))
    max_bridge = max(0, int(bridge_cfg.get("max_bridge_per_gap", 1)))
    gamma_fn = gamma_fn or _default_gamma_fn()

    centers = _to_numpy(centers).reshape(-1, 2)
    shape4 = _to_numpy(shape4).reshape(-1, 4)
    anchors = _to_numpy(anchors).reshape(-1)
    H = len(centers)
    empty = SafetyCorridor(cells=[], overlap_ratio=[], base_cell_count=0,
                           bridge_cell_count=0, valid=False,
                           min_overlap=min_overlap)
    if H == 0 or len(shape4) != H or len(anchors) != H:
        empty.failure_reason = "bad_corridor_input"
        return empty

    dev = builder.maps.device
    dt = builder.maps.dtype
    A, b, mask, valid = builder.build_from_ellipse(
        torch.as_tensor(centers, dtype=dt, device=dev)[None],
        torch.as_tensor(shape4, dtype=dt, device=dev)[None])

    base = [_base_cell(builder, A[0], b[0], mask[0], valid[0], centers,
                       anchors, i) for i in range(H)]
    for i, cell in enumerate(base):
        if not cell.valid:
            empty.failure_reason = "invalid_base_region:%d" % i
            empty.base_cell_count = H
            return empty

    def _gamma(s_value: float) -> np.ndarray:
        if gamma is None:
            raise ValueError("bridge requires the dense Skeleton Gamma")
        coords = torch.as_tensor(np.asarray(gamma, dtype=np.float32),
                                 device=dev, dtype=dt)[None]
        if torch.is_tensor(gamma_lengths):
            lengths = gamma_lengths.to(dev).long().reshape(-1)
            if lengths.numel() == 1:
                lengths = lengths.expand(1)
        else:
            lengths = torch.as_tensor(
                [int(gamma_lengths if gamma_lengths is not None
                     else len(coords[0]))], device=dev, dtype=torch.long)
        s = torch.tensor([[float(s_value)]], device=dev, dtype=dt)
        return gamma_fn(coords, lengths, s)[0, 0].detach().cpu().numpy() \
            .astype(np.float64)

    cells = [base[0]]
    ratios: list[float] = []
    bridges: list[dict] = []

    for i in range(H - 1):
        left = cells[-1]
        right = base[i + 1]
        rho = overlap_ratio(left.polygon, right.polygon)
        if rho + 1e-12 >= min_overlap:
            ratios.append(rho)
            cells.append(right)
            continue

        if not bridge_enabled or max_bridge < 1:
            empty.failure_reason = "overlap_below_threshold:%d" % i
            empty.base_cell_count = H
            return empty

        s_b = 0.5 * (float(anchors[i]) + float(anchors[i + 1]))
        try:
            c_b = _gamma(s_b)
        except Exception as error:                       # pragma: no cover
            empty.failure_reason = "bridge_gamma_failed:%d(%s)" % (i, error)
            empty.base_cell_count = H
            return empty

        bridge = build_bridge_region(builder, c_b)
        if not bridge["valid"]:
            empty.failure_reason = "bridge_region_invalid:%d" % i
            empty.base_cell_count = H
            return empty

        rho_l = overlap_ratio(left.polygon, bridge["polygon"])
        rho_r = overlap_ratio(bridge["polygon"], right.polygon)
        if rho_l + 1e-12 < min_overlap or rho_r + 1e-12 < min_overlap:
            # V1: exactly one bridge per gap, no recursive refinement.
            empty.failure_reason = "bridge_failed:%d" % i
            empty.base_cell_count = H
            return empty

        bridge_cell = CorridorCell(
            anchor_s=s_b, center=bridge["center"], A=bridge["A"],
            b=bridge["b"], polygon=bridge["polygon"], source="bridge",
            source_index=int(i), valid=True, face_count=bridge["face_count"])
        ratios.extend([rho_l, rho_r])
        cells.extend([bridge_cell, right])
        bridges.append({
            "gap": int(i),
            "anchor_s": float(s_b),
            "center": [float(c_b[0]), float(c_b[1])],
            "overlap_left": float(rho_l),
            "overlap_right": float(rho_r),
            "overlap_before": float(rho),
        })

    return SafetyCorridor(
        cells=cells, overlap_ratio=ratios, base_cell_count=H,
        bridge_cell_count=len(bridges), valid=True, failure_reason=None,
        min_overlap=min_overlap, bridge_gaps=bridges)
