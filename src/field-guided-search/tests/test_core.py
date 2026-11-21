"""Contract tests for the core package (stdlib unittest, no pytest needed)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from field_guided_search import (
    Grid,
    GuidanceField,
    astar,
    build_field,
    field_from_trajectory,
    guided_search,
    path_is_valid,
    propagate_heap,
    propagate_scipy,
    resample_polyline,
    trajectory_sources,
)
from field_guided_search.search import normalise_distance


def border_room(h=16, w=16, walls=()):
    occ = np.zeros((h, w), dtype=np.int8)
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    for r, c in walls:
        occ[r, c] = 1
    return occ


class GridTests(unittest.TestCase):
    def test_neighbours_respect_obstacles(self):
        grid = Grid(border_room(walls=[(5, c) for c in range(1, 10)]))
        inside = grid.idx(5, 1)
        for u in grid.neighbors(inside):
            self.assertTrue(grid.free.flat[u])

    def test_bfs_distance(self):
        grid = Grid(border_room())
        dist = grid.bfs_distance([grid.idx(2, 2)])
        self.assertEqual(dist[2, 2], 0)
        self.assertEqual(dist[2, 5], 3)
        self.assertEqual(dist[0, 0], -1)  # obstacle

    def test_unreachable_region_is_minus_one(self):
        occ = border_room(walls=[(r, 8) for r in range(1, 15)])
        grid = Grid(occ)
        dist = grid.bfs_distance([grid.idx(3, 3)])
        self.assertEqual(dist[3, 12], -1)


class SourceTests(unittest.TestCase):
    def test_resample_spans_the_polyline(self):
        pts = np.array([[2.0, 2.0], [2.0, 10.0]])
        points, progress = resample_polyline(pts, 9)
        self.assertEqual(len(points), 9)
        np.testing.assert_allclose(progress[[0, -1]], [0.0, 1.0])
        np.testing.assert_allclose(points[0], pts[0])
        np.testing.assert_allclose(points[-1], pts[-1])

    def test_drop_and_project_handle_obstacles(self):
        occ = border_room(walls=[(8, c) for c in range(2, 9)])
        grid = Grid(occ)
        traj = np.array([[8.0, 3.0], [8.0, 8.0]])          # straight through the wall
        dropped = trajectory_sources(grid, traj, num_samples=20, method="drop")
        projected = trajectory_sources(grid, traj, num_samples=20, method="project")
        self.assertGreater(dropped["num_dropped"], 0)
        self.assertEqual(projected["num_dropped"], 0)
        for src in (dropped, projected):
            pos = src["positions_rc"]
            self.assertTrue(np.all(grid.free[pos[:, 0], pos[:, 1]]))
            self.assertTrue(np.all(np.diff(src["progress"]) >= -1e-9))

    def test_dropping_keeps_a_progress_hole(self):
        occ = border_room(walls=[(8, c) for c in range(2, 9)])
        grid = Grid(occ)
        traj = np.array([[5.0, 5.0], [12.0, 5.0]])   # crosses the wall at row 8
        src = trajectory_sources(grid, traj, num_samples=20, method="drop")
        self.assertGreater(src["num_dropped"], 0)
        self.assertGreater(float(np.max(np.diff(src["progress"]))), 1.0 / 20)


class FieldTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(border_room(walls=[(8, c) for c in range(2, 12)]))
        self.traj = np.array([[3.0, 3.0], [3.0, 12.0], [12.0, 12.0]])
        self.src = trajectory_sources(self.grid, self.traj, num_samples=32)

    def test_obstacles_are_nan(self):
        field = build_field(self.grid, self.src["positions_rc"], self.src["progress"])
        self.assertTrue(np.all(np.isnan(field[~self.grid.free])))
        self.assertTrue(np.all(np.isfinite(field[self.grid.free])))

    def test_backends_agree(self):
        pos, u = self.src["positions_rc"], self.src["progress"]
        a = build_field(self.grid, pos, u, weight="linear", method="heap")
        b = build_field(self.grid, pos, u, weight="linear", method="scipy")
        ok = self.grid.free & np.isfinite(a) & np.isfinite(b)
        self.assertLess(float(np.max(np.abs(a[ok] - b[ok]))), 1e-9)

    def test_field_decays_with_distance_and_respects_weights(self):
        pos, u = self.src["positions_rc"], self.src["progress"]
        uniform = build_field(self.grid, pos, u, weight="uniform")
        self.assertAlmostEqual(float(np.nanmax(uniform)), 1.0, places=9)
        linear = build_field(self.grid, pos, u, weight="linear")
        # the start of the trajectory must not outrank its end
        self.assertLess(float(linear[3, 3]), float(linear[12, 12]))

    def test_duplicate_sources_keep_the_strongest(self):
        field = build_field(self.grid, np.array([[5, 5], [5, 5]]), np.array([0.0, 1.0]),
                            weight="linear")
        self.assertAlmostEqual(float(field[5, 5]), 1.0, places=9)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.occ = border_room(h=20, w=20, walls=[(8, c) for c in range(1, 16)])
        self.grid = Grid(self.occ)
        self.start, self.goal = (3, 3), (16, 16)

    def test_astar_is_optimal_and_valid(self):
        res = astar(self.grid, self.start, self.goal)
        self.assertEqual(res["status"], "success")
        info = path_is_valid(self.grid, res["path"], self.start, self.goal)
        self.assertTrue(info["valid"], msg=info)
        best = int(self.grid.bfs_distance([self.grid.idx(*self.start)])[self.goal])
        self.assertEqual(res["length"], best)

    def test_guided_search_beats_astar_on_expansions(self):
        traj = np.array([[3.0, 3.0], [15.0, 3.0], [15.0, 16.0], [16.0, 16.0]])
        gfield = GuidanceField(self.occ)
        gfield.build(traj, num_samples=48)
        base = astar(self.grid, self.start, self.goal)
        guided = gfield.search(self.start, self.goal)
        self.assertEqual(guided["status"], "success")
        self.assertTrue(path_is_valid(self.grid, guided["path"], self.start, self.goal)["valid"])
        self.assertLess(guided["expanded"], base["expanded"])

    def test_alpha_one_is_pure_distance_greedy(self):
        traj = np.array([[3.0, 3.0], [16.0, 16.0]])
        gfield = GuidanceField(self.occ)
        gfield.build(traj)
        res = gfield.search(self.start, self.goal, alpha=1.0)
        self.assertEqual(res["status"], "success")

    def test_unreachable_goal_is_reported(self):
        occ = border_room(h=20, w=20)
        occ[1:19, 9] = 1  # full-height wall splits the map
        grid = Grid(occ)
        self.assertFalse(grid.free[16, 16] is None)
        res = astar(grid, (3, 3), (16, 16))
        self.assertEqual(res["status"], "unreachable")
        self.assertEqual(len(res["path"]), 0)

    def test_cutoff_is_distinct_from_unreachable(self):
        res = astar(self.grid, self.start, self.goal, cutoff=5)
        self.assertEqual(res["status"], "cutoff")


class ApiTests(unittest.TestCase):
    def test_one_shot_functions_match_the_class(self):
        occ = border_room(h=20, w=20, walls=[(8, c) for c in range(1, 16)])
        traj = np.array([[3.0, 3.0], [3.0, 16.0], [16.0, 16.0]])
        grid = Grid(occ)
        field = field_from_trajectory(occ, traj, num_samples=32)
        res = guided_search(occ, (3, 3), (16, 16), field=field)
        self.assertEqual(res["status"], "success")
        self.assertTrue(path_is_valid(grid, res["path"], (3, 3), (16, 16))["valid"])

    def test_search_before_build_raises(self):
        gfield = GuidanceField(border_room())
        with self.assertRaises(RuntimeError):
            gfield.search((3, 3), (10, 10))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            GuidanceField(border_room(), weight="cubic")
        with self.assertRaises(ValueError):
            GuidanceField(border_room(), source_method="bridge")
        with self.assertRaises(ValueError):
            trajectory_sources(Grid(border_room()), np.zeros((3, 3)))

    def test_field_is_reusable_across_searches(self):
        occ = border_room(h=20, w=20, walls=[(8, c) for c in range(1, 16)])
        traj = np.array([[3.0, 3.0], [16.0, 16.0]])
        gfield = GuidanceField(occ)
        field = gfield.build(traj)
        np.testing.assert_array_equal(field, gfield.field)
        a = gfield.search((3, 3), (16, 16), alpha=0.0)
        b = gfield.search((3, 3), (16, 16), alpha=0.5)
        self.assertEqual(a["status"], "success")
        self.assertEqual(b["status"], "success")


class ReviewFixTests(unittest.TestCase):
    """Regressions for the findings of the static review."""

    def setUp(self):
        self.occ = border_room(h=16, w=16)
        self.grid = Grid(self.occ)

    def test_default_source_method_is_drop(self):
        self.assertEqual(GuidanceField(self.occ).source_method, "drop")

    def test_out_of_bounds_samples_are_dropped_not_clipped(self):
        traj = np.array([[5.0, 5.0], [40.0, 40.0]])       # second point lies outside the map
        src = trajectory_sources(self.grid, traj, num_samples=20, method="drop")
        self.assertGreater(src["num_out_of_bounds"], 0)
        self.assertLess(src["num_kept"], src["num_samples"])
        pos = src["positions_rc"]
        self.assertTrue(np.all(pos[:, 0] < self.grid.height))   # nothing pinned to the edge
        self.assertTrue(np.all(pos[:, 1] < self.grid.width))

    def test_single_cell_path_validation(self):
        same = path_is_valid(self.grid, np.array([[3, 3]]), (3, 3), (3, 3))
        self.assertTrue(same["valid"], msg=same)
        self.assertEqual(same["length"], 0)
        self.assertFalse(path_is_valid(self.grid, np.array([[3, 4]]), (3, 3), (3, 3))["valid"])
        self.assertFalse(path_is_valid(self.grid, np.array([[0, 0]]), (0, 0), (0, 0))["valid"])

    def test_one_source_is_enough(self):
        gfield = GuidanceField(self.occ)
        gfield.build(np.array([[5.0, 5.0]]))
        self.assertTrue(np.isfinite(gfield.field).any())
        self.assertEqual(gfield.search((3, 3), (12, 12))["status"], "success")

    def test_zero_sources_raise(self):
        traj = np.array([[0.0, 0.0], [0.0, 15.0]])        # runs along the top wall
        with self.assertRaises(RuntimeError):
            GuidanceField(self.occ).build(traj)

    def test_parameter_validation(self):
        cases = [{"lam": 0.0}, {"lam": -0.1}, {"weight": "cubic"},
                 {"distance": "euclid"}, {"num_samples": 0}, {"source_method": "bridge"}]
        for kwargs in cases:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                GuidanceField(self.occ, **kwargs)

    def test_obstacle_endpoint_is_rejected(self):
        gfield = GuidanceField(self.occ)
        gfield.build(np.array([[5.0, 5.0], [12.0, 12.0]]))
        with self.assertRaises(ValueError):
            gfield.search((0, 0), (12, 12))

    def test_timing_is_split(self):
        gfield = GuidanceField(self.occ)
        gfield.build(np.array([[5.0, 5.0], [12.0, 12.0]]))
        res = gfield.search((3, 3), (12, 12), alpha=0.5)
        for key in ("preprocess_ms", "search_ms", "total_ms"):
            self.assertIn(key, res)
        self.assertGreaterEqual(res["preprocess_ms"], 0.0)
        self.assertAlmostEqual(res["total_ms"], res["preprocess_ms"] + res["search_ms"], places=6)
        self.assertEqual(res["time_ms"], res["search_ms"])

    def test_distance_modes_differ(self):
        manhattan, _ = normalise_distance(self.grid, (12, 12), mode="manhattan")
        geodesic, _ = normalise_distance(self.grid, (12, 12), mode="geodesic")
        self.assertFalse(np.allclose(manhattan, geodesic))
        self.assertAlmostEqual(float(geodesic[12, 12]), 0.0)
        with self.assertRaises(ValueError):
            normalise_distance(self.grid, (12, 12), mode="euclid")


if __name__ == "__main__":
    unittest.main()
