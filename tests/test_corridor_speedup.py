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
                                        halfspaces_to_vertices,
                                        vertices_pairwise_batch)
from src.geometry.safety_corridor import (build_safety_corridor,
                                          halfspace_overlap_ratio,
                                          overlap_ratio, polygon_area)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "processed", "160m-k4p-c48", "test")


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


# ---------------------------------------------------------------------------
# batched polygon extraction: ragging widths, empty and contradictory sets
# ---------------------------------------------------------------------------


def _batch_reference(A_list, b_list):
    """What one ``halfspaces_to_vertices`` call per set returns."""
    return [halfspaces_to_vertices(np.asarray(a, dtype=np.float64),
                                   np.asarray(bb, dtype=np.float64))
            for a, bb in zip(A_list, b_list)]


def test_batched_vertices_match_the_per_set_loop_with_ragged_widths():
    """Padding (``A = 0, b = +inf``) must not create or destroy a vertex.

    The sets deliberately have DIFFERENT face counts, which forces the padding
    path: a padded face has ``b = +inf``, so ``b * A`` is ``inf * 0 = NaN`` for
    every pair that touches it.  Those pairs are dropped because their
    determinant is 0, but the NaN must not survive into the feasibility matmul.
    """
    A_list = [
        # 4 faces: the unit square
        [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
        # 3 faces: a triangle (one face shorter than the square)
        [[1., 0.], [0., 1.], [-1., -1.]],
        # 0 faces: the fully padded set
        np.zeros((0, 2)),
        # 5 faces: a rotated pentagon, WIDER than every other set
        [[1., 0.], [0.3, 0.95], [-0.81, 0.59], [-0.81, -0.59], [0.3, -0.95]],
        # contradictory halfspaces -> empty polytope
        [[1., 0.], [-1., 0.]],
        # a zero-norm normal, which both paths must drop
        [[1., 0.], [0., 0.], [-1., 0.], [0., 1.], [0., -1.]],
    ]
    b_list = [
        [.5, .5, .5, .5],
        [1., 1., 1.],
        np.zeros(0),
        [1., 1., 1., 1., 1.],
        [-1., -1.],
        [.5, 99., .5, .5, .5],
    ]
    got = vertices_pairwise_batch(A_list, b_list)
    want = _batch_reference(A_list, b_list)
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        if w is None:
            assert g is None, (i, g)
            continue
        assert g is not None, (i, w)
        assert len(g) == len(w), (i, len(g), len(w))
        assert _cyclic_equal(g, w), (i, g, w)
        assert polygon_area(g) == pytest.approx(polygon_area(w), rel=1e-12)


def test_batched_vertices_emit_no_numpy_warning():
    """The padded ``inf * 0`` products must be silenced, not merely outrun."""
    import warnings

    A_list = [np.array([[1., 0.], [-1., 0.], [0., 1.]]), np.zeros((0, 2)),
              np.array([[1., 0.], [-1., 0.]])]
    b_list = [np.ones(3), np.zeros(0), np.array([1., 1.])]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        vertices_pairwise_batch(A_list, b_list)


def test_batched_vertices_do_not_leak_state_across_sets():
    """40 copies of one square, one triangle and one empty set, all at once."""
    square_A = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    tri_A = np.array([[1., 0.], [0., 1.], [-1., -1.]])
    A_list = [square_A, tri_A, np.zeros((0, 2))] * 13 + [square_A]
    b_list = [np.ones(4), np.ones(3), np.zeros(0)] * 13 + [np.ones(4)]
    got = vertices_pairwise_batch(A_list, b_list)
    square = halfspaces_to_vertices(square_A, np.ones(4))
    tri = halfspaces_to_vertices(tri_A, np.ones(3))
    for i, g in enumerate(got):
        want = square if A_list[i] is square_A else (
            tri if A_list[i] is tri_A else None)
        if want is None:
            assert g is None, i
        else:
            assert g is not None and _cyclic_equal(g, want), i
    assert vertices_pairwise_batch([], []) == []


# ---------------------------------------------------------------------------
# region sharing (``region_stride``): one convex cell per k query points
# ---------------------------------------------------------------------------


def _corridor(stride: int | None, h: int = 12, spacing: float = 0.01):
    """A straight corridor of ``h`` tiny well-overlapping ellipses."""
    builder = EllipseRegionBuilder(torch.zeros(1, 1, 128, 128),
                                   {"safety_margin": 0.01,
                                    "obstacle_window_half": 0.35})
    centers = np.stack([np.linspace(0.0, spacing * (h - 1), h),
                        np.zeros(h)], axis=1)
    shape4 = np.zeros((h, 4))
    shape4[:, 0] = np.log(0.005)
    shape4[:, 1] = np.log(0.005)
    shape4[:, 2] = 1.0
    anchors = np.linspace(0.0, 1.0, h)
    gamma = centers
    config = {"min_overlap_ratio": 0.10,
              "bridge": {"enabled": True, "max_bridge_per_gap": 1}}
    if stride is not None:
        config["region_stride"] = stride
    return build_safety_corridor(builder, centers, shape4, anchors, gamma=gamma,
                                 gamma_lengths=len(gamma), config=config)


def test_region_stride_default_is_exactly_the_original_corridor():
    """No knob, ``stride = 1`` and ``stride = 0`` must all be the same corridor."""
    plain = _corridor(None)
    one = _corridor(1)
    zero = _corridor(0)
    for corridor in (plain, one, zero):
        assert corridor.valid, corridor.failure_reason
        assert corridor.base_cell_count == 12
        assert [c.source for c in corridor.cells].count("network") == 12
    assert [c.anchor_s for c in one.cells] == [c.anchor_s for c in plain.cells]
    assert [c.source for c in zero.cells] == [c.source for c in plain.cells]


def test_region_stride_shares_cells_and_always_keeps_the_last_anchor():
    """``H = 12``: stride 2 -> 7 cells, 4 -> 4, 8 -> 3 (the last anchor is kept).

    This is the regression test for the diagnostic field: the successful return
    used to report ``base_cell_count = H``, which made the knob look like a
    no-op even while it was dropping cells.
    """
    expected = {2: 7, 4: 4, 8: 3}
    base = _corridor(1)
    assert base.valid, base.failure_reason
    n_base = sum(1 for c in base.cells if c.source == "network")
    for stride, n_cells in expected.items():
        corridor = _corridor(stride)
        assert corridor.valid, corridor.failure_reason
        assert corridor.base_cell_count == n_cells, stride
        # one cell per kept anchor, plus on-demand bridges
        assert sum(1 for c in corridor.cells if c.source == "network") == n_cells
        assert sum(1 for c in corridor.cells if c.source == "bridge") \
            == corridor.bridge_cell_count
        # strictly fewer regions than the per-anchor corridor
        assert n_cells < n_base
        # the last anchor never shares its cell, so the endpoint region is intact
        assert corridor.cells[-1].source == "network"
        assert corridor.cells[-1].anchor_s == pytest.approx(1.0)
        # every bridge sits strictly between two kept anchors, seeded at their
        # midpoint (``s_b = 0.5 * (keep_anchors[i] + keep_anchors[i + 1])``)
        kept = sorted(c.anchor_s for c in corridor.cells if c.source == "network")
        for gap in corridor.bridge_gaps:
            brackets = [(k1, k2) for k1, k2 in zip(kept, kept[1:])
                        if k1 < gap["anchor_s"] < k2]
            assert brackets, (stride, gap["anchor_s"], kept)
            assert gap["anchor_s"] == pytest.approx(
                0.5 * (brackets[0][0] + brackets[0][1]))
