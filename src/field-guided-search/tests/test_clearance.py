"""Tests for the clearance-aware addition to the B3 priority (plan §6.1).

The search loop, the field, the neighbour order and the first-discovery-parent rule are
untouched by this feature, so what has to be proven is:

* ``beta = 0`` costs nothing and returns exactly the old result (no extra keys, same search);
* the three risk functions behave (non-negative, monotone, hinge reaches exactly 0);
* the distance transform follows the documented padding convention;
* bad knobs / bad risk maps are rejected instead of silently clipped;
* the per-map cache is reused, and refuses to be reused across a different map;
* the search is still GBFS (priority order, never ``g``);
* nothing about obstacles, narrow passages or a wall-adjacent goal breaks;
* ``scipy`` is still optional for users who do not enable the feature.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from field_guided_search import GuidanceField, Grid  # noqa: E402
from field_guided_search import clearance as cl  # noqa: E402
from field_guided_search.search import guided_search as grid_search  # noqa: E402

OCC, FREE = 1, 0


# --------------------------------------------------------------------------- #
# maps
# --------------------------------------------------------------------------- #
def room(size=41) -> np.ndarray:
    occ = np.zeros((size, size), dtype=np.int8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = OCC
    return occ


def straight(start, goal, n=48) -> np.ndarray:
    s, g = np.asarray(start, float), np.asarray(goal, float)
    return s + np.linspace(0.0, 1.0, n)[:, None] * (g - s)


def wall_open_case():
    """Open room, one long wall: both ends of the wall are equally cheap, walls touch nothing."""
    occ = room()
    occ[12:29, 20] = OCC
    return occ, (20, 4), (20, 36)


def wall_goal_case():
    """Goal inside a one-cell-high slot: low clearance at the endpoint is unavoidable."""
    occ = room()
    occ[8:34, 10] = OCC          # wall with a single-cell gap at (20, 10)
    occ[20, 10] = FREE
    occ[5, 12:23] = OCC          # the slot's roof
    occ[7, 12:23] = OCC          # and its floor: row 6 is one cell high, goal at (6, 16)
    return occ, (20, 4), (6, 16)


def corridor_case():
    """One-cell-wide forced channel: there is no clearance to win, the search must still work."""
    occ = room()
    occ[1:41, 6] = OCC
    occ[1:41, 18] = OCC
    occ[20, 6] = FREE
    occ[20, 18] = FREE
    occ[1:20, 6:19] = OCC
    occ[21:41, 6:19] = OCC
    return occ, (20, 2), (20, 22)


# --------------------------------------------------------------------------- #
# distance transform
# --------------------------------------------------------------------------- #
class DistanceTransformTests(unittest.TestCase):
    def test_single_obstacle_gives_euclidean_distance(self):
        occ = np.zeros((9, 9), dtype=np.int8)
        occ[4, 4] = OCC
        d = cl.clearance_distance(Grid(occ))
        self.assertEqual(d[4, 4], 0.0)                     # obstacle itself
        self.assertAlmostEqual(d[4, 5], 1.0)
        self.assertAlmostEqual(d[5, 5], np.sqrt(2.0))      # diagonal is euclidean, not manhattan
        self.assertAlmostEqual(d[3, 4], 1.0)
        self.assertAlmostEqual(d[0, 0], 1.0)               # the border wins: outside = obstacle
        self.assertTrue(np.all(np.isfinite(d)))

    def test_multiple_obstacles_take_the_nearest(self):
        occ = np.zeros((25, 25), dtype=np.int8)
        occ[10, 10] = OCC
        occ[10, 16] = OCC
        d = cl.clearance_distance(Grid(occ))
        self.assertAlmostEqual(d[10, 13], 3.0)             # equidistant from both
        self.assertAlmostEqual(d[10, 7], 3.0)
        self.assertAlmostEqual(d[5, 10], 5.0)
        self.assertAlmostEqual(d[7, 7], np.hypot(3, 3))    # border is 8 away, obstacle wins
        self.assertAlmostEqual(d[20, 20], 5.0)   # border is 4+1 away

    def test_fully_free_map_is_bounded_by_the_border(self):
        d = cl.clearance_distance(Grid(np.zeros((7, 7), dtype=np.int8)))
        self.assertAlmostEqual(d[0, 0], 1.0)               # outside the map counts as obstacle
        self.assertAlmostEqual(d[3, 3], 4.0)               # 3 cells to the border + 1
        self.assertTrue(np.all(np.isfinite(d)))
        self.assertTrue(np.all(d >= 1.0))

    def test_border_cells_are_not_infinitely_clear(self):
        occ = room(9)
        d = cl.clearance_distance(Grid(occ))
        self.assertTrue(np.all(d[~Grid(occ).free] == 0.0))
        self.assertAlmostEqual(d[1, 1], 1.0)
        self.assertLessEqual(float(d.max()), 4.0)

    def test_matches_the_independent_reference_on_a_random_map(self):
        rng = np.random.default_rng(4)
        occ = (rng.random((23, 31)) < 0.18).astype(np.int8)
        grid = Grid(occ)
        from scipy.ndimage import distance_transform_edt
        padded = np.pad(grid.free, 1, mode="constant", constant_values=False)
        reference = distance_transform_edt(padded)[1:-1, 1:-1]
        reference[~grid.free] = 0.0
        np.testing.assert_allclose(cl.clearance_distance(grid), reference, atol=0)


# --------------------------------------------------------------------------- #
# risk functions
# --------------------------------------------------------------------------- #
class RiskFunctionTests(unittest.TestCase):
    D = np.linspace(0.0, 8.0, 81).reshape(9, 9)

    def test_quadratic_hinge_is_bounded_monotone_and_exactly_zero_past_the_threshold(self):
        for ds in (2.0, 3.0, 4.0):
            r = cl.clearance_risk(self.D, "quadratic_hinge", threshold=ds)
            self.assertTrue(np.all(r >= 0.0))
            self.assertTrue(np.all(r <= 1.0 + 1e-15))
            self.assertAlmostEqual(float(r.flat[0]), 1.0)          # D = 0 -> full penalty
            flat = r.ravel()
            self.assertTrue(np.all(np.diff(flat) <= 1e-15))        # non-increasing in D
            self.assertTrue(np.all(r[self.D >= ds] == 0.0))        # exactly zero, not ~0
            self.assertAlmostEqual(float(r[np.isclose(self.D, ds / 2.0)].max()), 0.25)

    def test_exponential_is_bounded_and_monotone(self):
        for sigma in (2.0, 4.0):
            r = cl.clearance_risk(self.D, "exponential", sigma=sigma)
            self.assertTrue(np.all(np.isfinite(r)))
            self.assertTrue(np.all(r > 0.0) and np.all(r <= 1.0))
            self.assertAlmostEqual(float(r.flat[0]), 1.0)
            self.assertTrue(np.all(np.diff(r.ravel()) < 0.0))
            self.assertAlmostEqual(float(r.ravel()[1]), float(np.exp(-0.1 / sigma)))

    def test_logistic_is_bounded_monotone_and_half_at_the_threshold(self):
        r = cl.clearance_risk(self.D, "logistic", threshold=3.0, tau=0.7)
        self.assertTrue(np.all(np.isfinite(r)))
        self.assertTrue(np.all(r > 0.0) and np.all(r <= 1.0))
        self.assertTrue(np.all(np.diff(r.ravel()) < 0.0))
        self.assertAlmostEqual(float(r[np.isclose(self.D, 3.0)].max()), 0.5)
        # no overflow for absurd distances
        huge = cl.clearance_risk(np.array([[1e6]]), "logistic", threshold=3.0, tau=0.1)
        self.assertEqual(float(huge[0, 0]), 0.0)

    def test_risk_is_zero_past_the_threshold_on_free_cells(self):
        occ = room()
        occ[10:20, 10:20] = OCC
        grid = Grid(occ)
        d = cl.clearance_distance(grid)
        r = cl.clearance_risk(d, "quadratic_hinge", threshold=3.0)
        free = grid.free
        self.assertTrue(np.all(r[free] >= 0.0))
        self.assertTrue(np.all(r[free & (d >= 3.0)] == 0.0))
        # obstacle cells carry R(0) = 1; they are never visited, so the value is irrelevant
        self.assertTrue(np.all(np.isfinite(r)))


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(room(11))

    def test_bad_scalars_are_rejected(self):
        for kwargs in ({"beta": -0.1}, {"beta": float("nan")}, {"beta": float("inf")},
                       {"mode": "nope"}, {"threshold": 0.0}, {"threshold": -2.0},
                       {"sigma": 0.0}, {"tau": 0.0}, {"threshold": float("inf")}):
            args = {"beta": 0.25, "mode": "quadratic_hinge", "threshold": 3.0, "sigma": 2.0,
                    "tau": 0.7}
            args.update(kwargs)
            with self.assertRaises(ValueError, msg=str(kwargs)):
                cl.validate_risk_params(**args)

    def test_valid_scalars_are_returned_normalised(self):
        out = cl.validate_risk_params(1, "exponential", 3, 2, 0.7)
        self.assertEqual(out[1], "exponential")
        self.assertIsInstance(out[0], float)
        self.assertTrue(all(isinstance(v, float) for v in out[2:]))

    def test_penalty_map_shape_must_match(self):
        with self.assertRaises(ValueError):
            cl.validate_penalty_map(np.zeros((5, 5)), self.grid)

    def test_penalty_map_must_be_finite_and_in_range(self):
        good = np.full((11, 11), 0.5)
        cl.validate_penalty_map(good, self.grid)                     # no raise
        for bad_value in (np.nan, np.inf, -0.5, 1.5):
            bad = np.full((11, 11), 0.5)
            bad[3, 3] = bad_value
            with self.assertRaises(ValueError, msg=repr(bad_value)):
                cl.validate_penalty_map(bad, self.grid)

    def test_penalty_map_must_be_real(self):
        with self.assertRaises(ValueError):
            cl.validate_penalty_map(np.full((11, 11), 0.2 + 0j), self.grid)

    def test_search_rejects_bad_arguments(self):
        occ, start, goal = wall_open_case()
        gf = GuidanceField(occ)
        gf.build(straight(start, goal))
        for kw in ({"clearance_beta": -1.0}, {"clearance_mode": "triangular"},
                   {"clearance_threshold": 0.0}, {"clearance_penalty_map": np.zeros((3, 3))}):
            with self.assertRaises(ValueError, msg=str(kw)):
                gf.search(start, goal, alpha=0.25, clearance_beta=kw.pop("clearance_beta", 0.25),
                          **kw)

    def test_distance_map_must_be_valid(self):
        for bad in (np.array([[0.0, np.nan]]), np.array([[0.0, -1.0]]),
                    np.zeros((2, 2, 2))):
            with self.assertRaises(ValueError):
                cl.clearance_risk(bad)


# --------------------------------------------------------------------------- #
# zero-cost switch and cache
# --------------------------------------------------------------------------- #
class BackwardCompatibilityTests(unittest.TestCase):
    KEYS = {"status", "path", "trace", "expanded", "generated", "length", "search_ms",
            "preprocess_ms", "total_ms", "time_ms"}

    def setUp(self):
        occ, start, goal = wall_open_case()
        self.gf = GuidanceField(occ)
        self.gf.build(straight(start, goal))
        self.start, self.goal = start, goal

    def test_beta_zero_returns_the_original_result_keys(self):
        res = self.gf.search(self.start, self.goal, alpha=0.25)
        self.assertEqual(set(res), self.KEYS)

    def test_beta_zero_ignores_the_clearance_knobs(self):
        base = self.gf.search(self.start, self.goal, alpha=0.25)
        odd = self.gf.search(self.start, self.goal, alpha=0.25, clearance_beta=0.0,
                             clearance_mode="logistic", clearance_threshold=7.0,
                             clearance_sigma=1.0, clearance_tau=3.0)
        np.testing.assert_array_equal(base["path"], odd["path"])
        self.assertEqual(base["expanded"], odd["expanded"])
        self.assertEqual(set(odd), self.KEYS)

    def test_beta_zero_never_touches_the_distance_transform(self):
        self.gf.search(self.start, self.goal, alpha=0.25)
        self.assertEqual(self.gf.clearance().edt_calls, 0)
        self.assertEqual(self.gf.clearance().risk_calls, 0)

    def test_beta_positive_reports_its_own_timing_keys(self):
        res = self.gf.search(self.start, self.goal, alpha=0.25, clearance_beta=0.25)
        self.assertEqual(set(res) - self.KEYS,
                         {"clearance_edt_ms", "clearance_risk_ms", "clearance_source"})
        self.assertEqual(res["clearance_source"], "computed")
        self.assertGreaterEqual(res["total_ms"],
                                res["preprocess_ms"] + res["search_ms"] - 1e-6)
        again = self.gf.search(self.start, self.goal, alpha=0.25, clearance_beta=0.25)
        self.assertEqual(again["clearance_source"], "cache")
        self.assertEqual(again["clearance_edt_ms"], 0.0)


class CacheTests(unittest.TestCase):
    def setUp(self):
        occ, start, goal = wall_open_case()
        self.occ, self.start, self.goal = occ, start, goal
        self.gf = GuidanceField(occ)
        self.gf.build(straight(start, goal))

    def test_distance_transform_runs_once_for_repeated_searches(self):
        for beta in (0.10, 0.25, 0.40):
            self.gf.search(self.start, self.goal, alpha=0.25, clearance_beta=beta,
                           clearance_threshold=3.0)
        cache = self.gf.clearance()
        self.assertEqual(cache.edt_calls, 1)      # one map, one EDT
        self.assertEqual(cache.risk_calls, 1)     # beta does not change the risk map
        for beta in (0.10, 0.25, 0.40):
            self.gf.search(self.start, self.goal, alpha=0.25, clearance_beta=beta)
        self.assertEqual(cache.edt_calls, 1)
        self.assertEqual(cache.risk_calls, 1)
        for ds in (2.0, 4.0):
            self.gf.search(self.start, self.goal, alpha=0.25, clearance_beta=0.25,
                           clearance_threshold=ds)
        self.assertEqual(cache.edt_calls, 1)
        self.assertEqual(cache.risk_calls, 3)     # one risk map per distinct parameter set

    def test_goal_change_does_not_recompute_the_distance_transform(self):
        self.gf.search((20, 4), (20, 36), alpha=0.25, clearance_beta=0.25)
        self.gf.search((6, 6), (34, 34), alpha=0.25, clearance_beta=0.25)
        self.assertEqual(self.gf.clearance().edt_calls, 1)

    def test_a_different_map_is_never_served_from_the_old_cache(self):
        other = room(41)
        other[5:35, 30] = OCC
        other_gf = GuidanceField(other)
        other_gf.build(straight(self.start, self.goal))
        self.assertGreater(other_gf.clearance().distance()[20, 31], 0.0)
        # the same shape but different content must be refused, not silently reused
        stolen = GuidanceField(other)
        stolen._clearance = self.gf.clearance()
        stolen.build(straight(self.start, self.goal))
        with self.assertRaises(ValueError):
            stolen.search(self.start, self.goal, alpha=0.25, clearance_beta=0.25)

    def test_invalidate_after_an_in_place_map_change(self):
        cache = self.gf.clearance()
        before = cache.distance().copy()
        cache.invalidate()
        np.testing.assert_array_equal(cache.distance(), before)
        self.assertEqual(cache.edt_calls, 2)

    def test_external_penalty_map_matches_the_internal_one(self):
        risk = self.gf.clearance().risk("quadratic_hinge", 3.0, 2.0, 0.7)
        internal = self.gf.search(self.start, self.goal, alpha=0.25, clearance_beta=0.25)
        external = self.gf.search(self.start, self.goal, alpha=0.25, clearance_beta=0.25,
                                  clearance_penalty_map=risk)
        np.testing.assert_array_equal(internal["path"], external["path"])
        self.assertEqual(internal["expanded"], external["expanded"])
        self.assertEqual(external["clearance_source"], "provided")
        self.assertEqual(external["clearance_edt_ms"], 0.0)


# --------------------------------------------------------------------------- #
# the search is still GBFS, and still safe
# --------------------------------------------------------------------------- #
class SearchPropertyTests(unittest.TestCase):
    def test_priority_term_keeps_search_a_pure_ranking_without_accumulated_cost(self):
        """Scaling/shifting J keeps every comparison, so the trace must not move.

        ``_best_first`` ranks nodes by ``J(r, c)`` only; a loop that accumulated ``g`` would
        not be invariant under ``J -> 2J + 5``. This pins the "still GBFS, no A*" promise.
        """
        from field_guided_search.search import _best_first, normalise_distance, normalise_field
        occ, start, goal = wall_open_case()
        gf = GuidanceField(occ)
        gf.build(straight(start, goal))
        d_hat, _ = normalise_distance(gf.grid, goal, mode="manhattan")
        f_hat = normalise_field(gf.field, gf.clip, gf.weight)
        j = 0.25 * d_hat - 0.75 * f_hat + 0.25 * gf.clearance().risk("quadratic_hinge", 3.0)
        plain = _best_first(gf.grid, start, goal, lambda r, c: float(j[r, c]))
        scaled = _best_first(gf.grid, start, goal, lambda r, c: 2.0 * float(j[r, c]) + 5.0)
        np.testing.assert_array_equal(plain["trace"], scaled["trace"])
        np.testing.assert_array_equal(plain["path"], scaled["path"])
        self.assertEqual(plain["expanded"], scaled["expanded"])

    def test_paths_stay_valid_and_collision_free(self):
        for case in (wall_open_case(), wall_goal_case(), corridor_case()):
            occ, start, goal = case
            gf = GuidanceField(occ)
            gf.build(straight(start, goal))
            for beta in (0.0, 0.10, 0.25, 0.40, 0.60):
                res = gf.search(start, goal, alpha=0.25, clearance_beta=beta)
                self.assertEqual(res["status"], "success", msg=f"beta={beta}")
                self.assertTrue(gf.is_valid(res["path"], start, goal)["valid"], msg=str(beta))

    def test_wall_hugging_open_room_is_improved(self):
        """Plan §5.3 case 1: hugging a wall with open space on the other side must improve."""
        occ, start, goal = wall_open_case()
        gf = GuidanceField(occ)
        gf.build(straight(start, goal))
        d = gf.clearance().distance()
        base = gf.search(start, goal, alpha=0.25)
        base_d = d[base["path"][:, 0], base["path"][:, 1]]
        improved = False
        for beta in (0.10, 0.25, 0.40):
            res = gf.search(start, goal, alpha=0.25, clearance_beta=beta)
            d_new = d[res["path"][:, 0], res["path"][:, 1]]
            if d_new.mean() > base_d.mean() and np.percentile(d_new, 10) > np.percentile(base_d, 10):
                improved = True
                self.assertLessEqual(len(res["path"]), len(base["path"]) * 1.10)
        self.assertTrue(improved, "no beta improved the clearance of the wall-hugging path")

    def test_forced_single_cell_corridor_still_reaches_the_goal(self):
        occ, start, goal = corridor_case()
        gf = GuidanceField(occ)
        gf.build(straight(start, goal))
        for beta in (0.0, 0.25, 0.60):
            res = gf.search(start, goal, alpha=0.25, clearance_beta=beta)
            self.assertEqual(res["status"], "success")
            self.assertTrue(gf.is_valid(res["path"], start, goal)["valid"])

    def test_goal_next_to_a_wall_is_reachable(self):
        occ, start, goal = wall_goal_case()
        gf = GuidanceField(occ)
        gf.build(straight(start, goal))
        for beta in (0.0, 0.25, 0.60):
            res = gf.search(start, goal, alpha=0.25, clearance_beta=beta)
            self.assertEqual(res["status"], "success")
            self.assertEqual(tuple(res["path"][-1]), goal)

    def test_start_equal_goal(self):
        occ, start, _ = wall_open_case()
        gf = GuidanceField(occ)
        gf.build(straight(start, start))
        for beta in (0.0, 0.25):
            res = gf.search(start, start, alpha=0.25, clearance_beta=beta)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["length"], 0)

    def test_unreachable_and_cutoff_are_unchanged(self):
        occ = room(15)
        occ[:, 7] = OCC                      # splits the map in two
        gf = GuidanceField(occ)
        gf.build(straight((7, 3), (7, 11)))
        for beta in (0.0, 0.25):
            self.assertEqual(gf.search((7, 3), (7, 11), clearance_beta=beta)["status"],
                             "unreachable")
            cut = gf.search((7, 3), (7, 11), clearance_beta=beta, cutoff=3)
            self.assertEqual(cut["status"], "cutoff")
            self.assertLessEqual(cut["expanded"], 3)


# --------------------------------------------------------------------------- #
# optional scipy
# --------------------------------------------------------------------------- #
class ScipyOptionalTests(unittest.TestCase):
    def test_beta_zero_works_without_scipy_and_beta_positive_explains_itself(self):
        occ, start, goal = wall_open_case()
        gf = GuidanceField(occ)
        gf.build(straight(start, goal))
        risk = gf.clearance().risk("quadratic_hinge", 3.0, 2.0, 0.7)
        fresh = GuidanceField(occ)
        fresh.build(straight(start, goal))
        blocked = {"scipy": None, "scipy.ndimage": None}
        with mock.patch.dict(sys.modules, blocked):
            self.assertEqual(fresh.search(start, goal, alpha=0.25)["status"], "success")
            with self.assertRaises(ImportError) as ctx:
                fresh.search(start, goal, alpha=0.25, clearance_beta=0.25)
            self.assertIn("pip install -e", str(ctx.exception))
            provided = fresh.search(start, goal, alpha=0.25, clearance_beta=0.25,
                                    clearance_penalty_map=risk)
            self.assertEqual(provided["status"], "success")

    def test_functional_interface_forwards_the_new_arguments(self):
        occ, start, goal = wall_open_case()
        via_field = GuidanceField(occ)
        via_field.build(straight(start, goal))
        internal = via_field.search(start, goal, alpha=0.25, clearance_beta=0.25)
        from field_guided_search import guided_search
        one_shot = guided_search(occ, start, goal, field=via_field.field, alpha=0.25,
                                 clearance_beta=0.25)
        np.testing.assert_array_equal(internal["path"], one_shot["path"])
        self.assertEqual(internal["expanded"], one_shot["expanded"])
        with_traj = guided_search(occ, start, goal, trajectory=straight(start, goal),
                                  alpha=0.25, clearance_beta=0.25)
        np.testing.assert_array_equal(internal["path"], with_traj["path"])
        # and the grid-level entry point accepts a precomputed map
        risk = via_field.clearance().risk("quadratic_hinge", 3.0, 2.0, 0.7)
        grid_res = grid_search(via_field.grid, start, goal, via_field.field, alpha=0.25,
                               clearance_beta=0.25, clearance_penalty_map=risk)
        self.assertEqual(grid_res["expanded"], internal["expanded"])


if __name__ == "__main__":
    unittest.main()
