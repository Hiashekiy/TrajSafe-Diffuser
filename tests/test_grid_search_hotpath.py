"""Bit-identity tests for the hot-path rewrites in ``src/grid_search``.

The neighbour table and the clearance-gradient centring are on the per-sample
critical path (12.6 ms and 12.8 ms of a 111 ms inference).  Both were rewritten
for speed; these tests keep an executable copy of the ORIGINAL formulation as
the oracle so "same answer, faster" stays a testable claim rather than a hope.

Run:  python -m pytest tests/test_grid_search_hotpath.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from src.grid_search import grid_map, grid_neighbors, path_centering   # noqa: E402
from src.grid_search.grid_neighbors import (DIAGONAL, MOVES, SQRT2,   # noqa: E402
                                            build_neighbor_table)


def reference_neighbor_table(safe_free, cell_m=1.0, allow_corner_cut=False):
    """The original flat implementation, kept verbatim as the oracle."""
    free = np.ascontiguousarray(safe_free, dtype=bool)
    h, w = free.shape
    flat = free.reshape(-1)
    rows, cols = np.divmod(np.arange(h * w, dtype=np.int64), w)
    idx = np.full((h * w, 8), -1, dtype=np.int32)
    for k, (dr, dc) in enumerate(MOVES):
        r, c = rows + dr, cols + dc
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        nbr = np.full(h * w, -1, dtype=np.int64)
        rr, cc = r[ok], c[ok]
        nbr[ok] = rr * w + cc
        valid = ok.copy()
        valid[ok] = free[rr, cc]
        if DIAGONAL[k] and not allow_corner_cut:
            side_a = np.full(h * w, -1, dtype=np.int64)
            side_b = np.full(h * w, -1, dtype=np.int64)
            ok_a = (rows + dr >= 0) & (rows + dr < h) & (cols >= 0) & (cols < w)
            ok_b = (rows >= 0) & (rows < h) & (cols + dc >= 0) & (cols + dc < w)
            side_a[ok_a] = (rows[ok_a] + dr) * w + cols[ok_a]
            side_b[ok_b] = rows[ok_b] * w + (cols[ok_b] + dc)
            for side in (side_a, side_b):
                has = side >= 0
                side_free = np.zeros(h * w, dtype=bool)
                side_free[has] = flat[side[has]]
                valid &= side_free
        idx[valid, k] = nbr[valid]
    cost = np.array([SQRT2 if d else 1.0 for d in DIAGONAL],
                    dtype=np.float64) * float(cell_m)
    return grid_neighbors.NeighborTable(idx=idx, cost=cost, free=flat, width=int(w))


def _masks():
    rng = np.random.default_rng(7)
    yield "empty", np.zeros((9, 11), dtype=bool)
    yield "full", np.ones((9, 11), dtype=bool)
    yield "noise", rng.random((23, 17)) > 0.4
    yield "single_cell", np.zeros((5, 5), dtype=bool)
    one = np.zeros((5, 5), dtype=bool)
    one[2, 2] = True
    yield "one_free", one
    # a 1-wide corridor: diagonal moves must be refused (corner cutting)
    corridor = np.zeros((12, 12), dtype=bool)
    for i in range(1, 11):
        corridor[i, 5] = True
    yield "corridor", corridor


def test_neighbor_table_is_bit_identical():
    for name, mask in _masks():
        new = build_neighbor_table(mask, 0.625)
        old = reference_neighbor_table(mask, 0.625)
        assert np.array_equal(new.idx, old.idx), name
        assert np.array_equal(new.cost, old.cost), name
        assert np.array_equal(new.free, old.free), name
        assert new.width == old.width, name
        # corner-cutting must be refused on the 1-wide corridor
        if name == "corridor":
            assert (new.idx[:, 4:] == -1).all(), "diagonal through a corner"


def test_neighbor_table_allow_corner_cut_matches_oracle():
    for name, mask in _masks():
        new = build_neighbor_table(mask, 1.0, allow_corner_cut=True)
        old = reference_neighbor_table(mask, 1.0, allow_corner_cut=True)
        assert np.array_equal(new.idx, old.idx), name


def test_neighbor_table_shapes_are_flat_rows():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 1:5] = True
    tab = build_neighbor_table(mask, 1.0)
    assert tab.idx.shape == (24, 8)
    assert tab.idx[1 * 6 + 2, 0] == 1 * 6 + 3     # +x
    assert tab.idx[1 * 6 + 2, 2] == 2 * 6 + 2     # +y
    assert tab.idx[0, 2] == -1                    # -y leaves the map


# ---------------------------------------------------------------------------
# centring: the fused 3-field bilinear must reproduce the per-field version
# ---------------------------------------------------------------------------
def _reference_bilinear(field, xy):
    H, W = field.shape
    px = (xy[:, 0] + 1.0) / 2.0 * (W - 1)
    py = (xy[:, 1] + 1.0) / 2.0 * (H - 1)
    x0 = np.clip(np.floor(px).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, H - 2)
    fx = np.clip(px - x0, 0.0, 1.0)
    fy = np.clip(py - y0, 0.0, 1.0)
    f00, f10 = field[y0, x0], field[y0, x0 + 1]
    f01, f11 = field[y0 + 1, x0], field[y0 + 1, x0 + 1]
    return (f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy)
            + f01 * (1 - fx) * fy + f11 * fx * fy)


def test_fused_bilinear_is_bit_identical():
    rng = np.random.default_rng(3)
    fields = rng.random((3, 32, 40))
    xy = rng.uniform(-1.0, 1.0, size=(128, 2))
    out = path_centering._bilinear3(fields, xy)
    for k in range(3):
        assert np.array_equal(out[k], path_centering._bilinear(fields[k], xy)), k


def test_center_path_matches_original_output():
    """Same path in, same path out -- the speed rewrite must not move a point."""
    import importlib
    free = np.zeros((64, 64), dtype=bool)
    free[10:54, 10:54] = True
    free[30:34, :] = False                     # a wall across the channel
    free[30:34, 20:24] = True                  # ... with one gap
    t = np.linspace(0.05, 0.95, 128)
    path = np.stack([-1 + 2 * t, -0.7 + 1.4 * t ** 2], -1)

    new_path, new_stats = path_centering.center_path(path, free, 2.5, iters=25)

    old_mod = _original_center_path_module()
    old_path, old_stats = old_mod.center_path(path, free, 2.5, iters=25)

    assert np.array_equal(new_path, old_path)
    for k in ("clearance_before", "clearance_after", "clearance_min_before",
              "clearance_min_after", "colliding_points_before",
              "colliding_points_after"):
        assert new_stats[k] == old_stats[k], k


def test_center_path_reused_clearance_is_bit_identical():
    """Sharing the EDT from the safe-grid build must not move any point."""
    free = np.zeros((64, 64), dtype=bool)
    free[8:56, 8:56] = True
    free[28:32, :] = False
    free[28:32, 24:28] = True
    t = np.linspace(0.05, 0.95, 128)
    path = np.stack([-1 + 2 * t, -0.8 + 1.6 * t], -1)
    shared = grid_map.clearance_field(free, 2.5)
    a, sa = path_centering.center_path(path, free, 2.5, iters=15)
    b, sb = path_centering.center_path(path, free, 2.5, iters=15, clearance=shared)
    assert np.array_equal(a, b)
    assert sa == sb
    assert np.array_equal(shared, grid_map.clearance_field(free, 2.5))


def test_inflate_obstacles_returns_the_distance_it_uses():
    free = np.zeros((32, 32), dtype=bool)
    free[4:28, 4:28] = True
    free[14:18, 14:18] = False
    safe, dist = grid_map.inflate_obstacles(free, 3.0, 0.5, return_distance=True)
    assert np.array_equal(safe, grid_map.inflate_obstacles(free, 3.0, 0.5))
    assert np.array_equal(dist, grid_map.clearance_field(free, 0.5))
    # ``radius`` is in CELLS, ``dist`` in metres -> 3 cells == 1.5 m at 0.5 m/cell
    assert np.array_equal(safe, dist > 3.0 * 0.5)


def _original_center_path_module():
    import types
    from scipy.ndimage import distance_transform_edt

    def _bilinear(field, xy):
        H, W = field.shape
        px = (xy[:, 0] + 1.0) / 2.0 * (W - 1)
        py = (xy[:, 1] + 1.0) / 2.0 * (H - 1)
        x0 = np.clip(np.floor(px).astype(int), 0, W - 2)
        y0 = np.clip(np.floor(py).astype(int), 0, H - 2)
        fx = np.clip(px - x0, 0.0, 1.0)
        fy = np.clip(py - y0, 0.0, 1.0)
        f00, f10 = field[y0, x0], field[y0, x0 + 1]
        f01, f11 = field[y0 + 1, x0], field[y0 + 1, x0 + 1]
        return (f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy)
                + f01 * (1 - fx) * fy + f11 * fx * fy)

    def center_path(path, free, cell_m, *, iters=40, step_cells=1.0,
                    smooth=0.25, keep_ends=True):
        path = np.asarray(path, dtype=np.float64).copy()
        orig_path = path.copy()
        free = np.asarray(free, dtype=bool)
        H, W = free.shape
        K = len(path)
        stats = {"clearance_before": None, "clearance_after": None}
        if K < 3 or not free.any():
            return path, stats
        clear = grid_map.clearance_field(free, cell_m).astype(np.float64)
        clear[~free] = 0.0
        _ = distance_transform_edt(~free, return_indices=True)   # dead work
        scene_per_cell = 2.0 / W
        step = float(step_cells) * scene_per_cell
        gy, gx = np.gradient(clear)
        before = _bilinear(clear, path)
        stats["clearance_before"] = float(np.mean(before))
        stats["clearance_min_before"] = float(np.min(before))
        for _ in range(int(iters)):
            c = _bilinear(clear, path)
            gxv = _bilinear(gx, path)
            gyv = _bilinear(gy, path)
            norm = np.sqrt(gxv ** 2 + gyv ** 2) + 1e-12
            mag = step * np.clip(1.0 - c / max(float(np.max(c)), 1e-9), 0.15, 1.0)
            move = np.stack([gxv / norm, gyv / norm], -1) * mag[:, None]
            if keep_ends:
                move[0] = 0.0
                move[-1] = 0.0
            prev = path.copy()
            cand = path + move
            sm = cand.copy()
            sm[1:-1] = ((1 - smooth) * cand[1:-1]
                        + smooth * 0.5 * (cand[:-2] + cand[2:]))
            cand = sm
            cand = np.clip(cand, -1.0 + 1e-6, 1.0 - 1e-6)
            px = np.clip(np.rint((cand[:, 0] + 1.0) / 2.0 * W - 0.5).astype(int),
                         0, W - 1)
            py = np.clip(np.rint((cand[:, 1] + 1.0) / 2.0 * H - 0.5).astype(int),
                         0, H - 1)
            bad = ~free[py, px]
            if bad.any():
                cand[bad] = prev[bad]
            path = cand
        def _collides(q):
            px = np.clip(np.rint((q[:, 0] + 1.0) / 2.0 * W - 0.5).astype(int),
                         0, W - 1)
            py = np.clip(np.rint((q[:, 1] + 1.0) / 2.0 * H - 0.5).astype(int),
                         0, H - 1)
            return int((~free[py, px]).sum())
        n_in, n_out = _collides(np.asarray(orig_path)), _collides(path)
        stats["colliding_points_before"] = n_in
        stats["colliding_points_after"] = n_out
        if n_out > n_in:
            stats["reverted_to_raw"] = True
            after = _bilinear(clear, np.asarray(orig_path))
            stats["clearance_after"] = float(np.mean(after))
            stats["clearance_min_after"] = float(np.min(after))
            return np.asarray(orig_path).copy(), stats
        after = _bilinear(clear, path)
        stats["clearance_after"] = float(np.mean(after))
        stats["clearance_min_after"] = float(np.min(after))
        return path, stats

    return types.SimpleNamespace(center_path=center_path, _bilinear=_bilinear)
