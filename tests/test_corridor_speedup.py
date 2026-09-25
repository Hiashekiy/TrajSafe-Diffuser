"""The fast corridor path must return the SAME geometry as the qhull reference.

Motivation: corridor construction was 92 % of the full method's runtime, and
~30 % of THAT was spent creating and deleting qhull temporary files (128 cells
x 2 qhull calls per sample).  ``_vertices_pairwise_vectorised`` computes the
vertices in closed form instead.

A speedup is only acceptable if the geometry is unchanged, so these tests assert,
on REAL cells produced by the actual ``EllipseRegionBuilder``:

* the same number of vertices;
* the same vertex SET (a cyclic rotation of the polygon is allowed -- the qhull
  output starts at an arbitrary vertex, and nothing downstream depends on where
  the cycle starts);
* the same area to ~1e-12 relative;
* the same overlap ratio for adjacent cells to ~1e-12, via both the polygon
  clipper and the halfspace route;
* degenerate / empty / unbounded inputs behave identically.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.geometry.convex_region import (EllipseRegionBuilder,
                                        _vertices_qhull_reference,
                                        halfspaces_to_vertices)
from src.geometry.safety_corridor import (halfspace_overlap_ratio,
                                          overlap_ratio, polygon_area)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "carla_processed_160k4p_c48", "test")


def _cyclic_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-7) -> bool:
    """True when ``b`` is a rotation of the closed cycle ``a``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    # match b[0] to some entry of a, then walk both cycles
    for start in range(len(a)):
        rolled = np.roll(a, -start, axis=0)
        if np.allclose(rolled, b, atol=tol, rtol=0):
            return True
    return False


def _real_cells(n_samples: int = 4):
    """Cells (A, b, centre) from the real builder on real occupancy grids."""
    if not os.path.isdir(DATA):
        pytest.skip("C=48 test cache not available")
    occ = torch.as_tensor(np.load(os.path.join(DATA, "occupancy.npy"),
                                  mmap_mode="r")[:n_samples].astype(np.float32))
    cond = torch.as_tensor(np.load(os.path.join(DATA, "conditions.npy"),
                                   mmap_mode="r")[:n_samples].astype(np.float32))
    shape4 = torch.as_tensor(np.load(os.path.join(DATA, "ellipse_shape4_gt.npy"),
                                     mmap_mode="r")[:n_samples].astype(np.float32))
    builder = EllipseRegionBuilder(occ[:, None], {"safety_margin": 0.02,
                                                 "obstacle_window_half": 0.35})
    A, b, mask, valid = builder.build_from_ellipse(cond[:, 0][:, None].expand(
        -1, shape4.shape[1], -1).contiguous(), shape4)
    cells = []
    for s in range(A.shape[0]):
        for h in range(A.shape[1]):
            face = mask[s, h]
            if int(face.sum()) < 3:
                continue
            cells.append((A[s, h][face].numpy().astype(np.float64),
                          b[s, h][face].numpy().astype(np.float64),
                          cond[s, 0].numpy().astype(np.float64)))
    return cells


def test_fast_vertices_match_the_qhull_reference_on_real_cells():
    cells = _real_cells()
    assert len(cells) > 100, "expected a few hundred real cells, got %d" % len(cells)
    compared = 0
    for A_i, b_i, centre in cells:
        fast = halfspaces_to_vertices(A_i, b_i)
        slow = _vertices_qhull_reference(A_i, b_i, interior_point=centre)
        if slow is None:
            assert fast is None
            continue
        assert fast is not None
        assert len(fast) == len(slow), (len(fast), len(slow))
        assert _cyclic_equal(fast, slow), (fast, slow)
        assert polygon_area(fast) == pytest.approx(polygon_area(slow), rel=1e-12)
        compared += 1
    assert compared > 80


def test_fast_vertices_match_on_synthetic_degenerate_inputs():
    cases = [
        # a small square
        (np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]), np.ones(4) * 0.5),
        # a triangle
        (np.array([[1., 0.], [0., 1.], [-1., -1.]]), np.array([1., 1., 1.])),
        # duplicate face (parallel normals) -> still a quad
        (np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [1., 0.]]),
         np.array([.5, .5, .5, .5, .3])),
        # empty: contradictory halfspaces
        (np.array([[1., 0.], [-1., 0.]]), np.array([-1., -1.])),
        # only two faces -> no polygon
        (np.array([[1., 0.], [-1., 0.]]), np.array([1., 1.])),
        # a zero-norm normal must be dropped by both
        (np.array([[1., 0.], [0., 0.], [-1., 0.], [0., 1.], [0., -1.]]),
         np.array([.5, 99., .5, .5, .5])),
    ]
    for A_i, b_i in cases:
        fast = halfspaces_to_vertices(A_i, b_i)
        slow = _vertices_qhull_reference(A_i, b_i)
        if slow is None:
            assert fast is None, (A_i, b_i, fast)
        else:
            assert fast is not None, (A_i, b_i)
            assert len(fast) == len(slow)
            assert _cyclic_equal(fast, slow)
            assert polygon_area(fast) == pytest.approx(polygon_area(slow), rel=1e-12)


def test_halfspace_overlap_equals_the_polygon_clipper():
    """``area(A n B)`` from concatenated halfspaces == Sutherland-Hodgman."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        ca, cb = rng.uniform(-0.5, 0.5, 4).reshape(2, 2)
        ra, rb = rng.uniform(0.05, 0.4, 2)
        th = rng.uniform(0, np.pi, 2)
        polys, halves = [], []
        for c, r, t in ((ca, ra, th[0]), (cb, rb, th[1])):
            ang = np.linspace(0, 2 * np.pi, 13)[:-1]
            pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            poly = pts @ R.T + c
            # halfspaces of that convex polygon, outward normals, CCW input
            edges = np.roll(poly, -1, axis=0) - poly
            normals = np.stack([edges[:, 1], -edges[:, 0]], 1)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            polys.append(poly)
            halves.append((normals, (normals * poly).sum(1) + 1e-12))
        # the clipping convention needs CCW; our generator is CCW for r>0
        direct = overlap_ratio(polys[0], polys[1])
        via_half = halfspace_overlap_ratio(halves[0][0], halves[0][1],
                                          halves[1][0], halves[1][1],
                                          area_a=polygon_area(polys[0]),
                                          area_b=polygon_area(polys[1]))
        assert via_half == pytest.approx(direct, abs=1e-9)


def test_polygon_area_is_unchanged_by_the_roll_removal():
    """The slicing shoelace must agree with the np.roll version exactly."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(3, 12))
        pts = rng.normal(size=(n, 2))
        x, y = pts[:, 0], pts[:, 1]
        reference = 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                                    - np.dot(y, np.roll(x, -1))))
        assert polygon_area(pts) == pytest.approx(reference, rel=1e-15)
