"""Four-step asynchronous receding-horizon planner for long CARLA routes."""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .path_profile import PathProfile
from .planner_adapter import PlanResult, PlannerAdapter, check_acceptance
from .rolling_frame import WorldSceneFrame
from .rolling_occupancy import CarlaMapRasterizer


class RoutePolyline:
    """Arc-length indexed, immutable global route."""

    def __init__(self, points_world):
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 2)
        if len(points) < 2:
            raise ValueError("global route needs at least two points")
        keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0),
                                                       axis=1) > 1e-6))
        self.xy = points[keep]
        self.s = np.concatenate(([0.0], np.cumsum(
            np.linalg.norm(np.diff(self.xy, axis=0), axis=1))))

    @property
    def total_length(self) -> float:
        return float(self.s[-1])

    def project(self, point_world, min_s: float = 0.0) -> float:
        point = np.asarray(point_world, dtype=np.float64).reshape(2)
        start = max(int(np.searchsorted(self.s, float(min_s), side="left")) - 2, 0)
        index = start + int(np.argmin(np.linalg.norm(
            self.xy[start:] - point[None], axis=1)))
        return float(self.s[index])

    def sample(self, s_query: float) -> np.ndarray:
        value = float(np.clip(s_query, 0.0, self.total_length))
        return np.array([np.interp(value, self.s, self.xy[:, 0]),
                         np.interp(value, self.s, self.xy[:, 1])])

    def section(self, start_s: float, end_s: float, spacing_m: float = 2.0):
        a = float(np.clip(start_s, 0.0, self.total_length))
        b = float(np.clip(max(end_s, a), 0.0, self.total_length))
        n = max(2, int(math.ceil((b - a) / float(spacing_m))) + 1)
        values = np.linspace(a, b, n)
        points = np.stack((np.interp(values, self.s, self.xy[:, 0]),
                           np.interp(values, self.s, self.xy[:, 1])), axis=1)
        return values, points

    def select_horizon(self, start_world, min_route_s: float,
                       lookahead_m: float, window_size_m: float,
                       window_margin_m: float):
        start_s = self.project(start_world, min_route_s)
        max_s = min(start_s + float(lookahead_m), self.total_length)
        # Back off until the complete route section fits the square, not just
        # its two endpoints.  This matters for U turns and curved junctions.
        end_s = max_s
        while end_s > start_s + 10.0:
            _, section = self.section(start_s, end_s)
            span = section.max(axis=0) - section.min(axis=0)
            if np.all(span <= float(window_size_m) - 2.0 * float(window_margin_m)):
                return start_s, end_s, section
            end_s -= 5.0
        _, section = self.section(start_s, end_s)
        return start_s, end_s, section


@dataclass
class RollingPlan:
    plan_id: int
    result: PlanResult
    profile: PathProfile
    frame: WorldSceneFrame
    occupancy: np.ndarray
    route_start_s: float
    route_goal_s: float
    handoff_world: np.ndarray
    requested_wall_time: float
    completed_wall_time: float
    route_mean_error_m: float
    route_max_error_m: float
    acceptance: Dict[str, Any]

    @property
    def planning_ms(self) -> float:
        return float(self.result.quality.get("total_planning_ms",
                                             self.result.planning_ms))


class FourStepRollingPlanner:
    """Own one model and one worker; every request executes exactly 4 forwards."""

    def __init__(self, rasterizer: CarlaMapRasterizer, route: RoutePolyline,
                 planner_cfg: Dict[str, Any], controller_cfg: Dict[str, Any]):
        self.rasterizer = rasterizer
        self.route = route
        self.cfg = dict(planner_cfg or {})
        self.controller_cfg = dict(controller_cfg or {})
        steps = int(self.cfg.get("reverse_steps", 4))
        if steps != 4:
            raise ValueError("continuous planner requires reverse_steps=4, got %d" % steps)
        self.adapter = PlannerAdapter(
            processed_root=self.cfg.get("processed_root"),
            device=self.cfg.get("device"),
            corridor_region_override=self.cfg.get("corridor_region"),
            config_path=self.cfg.get("model_config"))
        # Four-step deployment schedule.  The warm-up/guided split is
        # configurable, while total network forwards remains exactly four.
        self.adapter.engine.alm_cfg = dict(self.adapter.engine.alm_cfg or {})
        self.adapter.engine.alm_cfg["warmup_reverse_steps"] = int(
            self.cfg.get("four_step_warmup_reverse_steps", 2))
        for source, target in (
                ("alm_activation_inner_steps", "activation_inner_steps"),
                ("alm_inner_steps", "inner_steps"),
                ("alm_max_curve_step_scene", "max_curve_step_scene")):
            if source in self.cfg:
                self.adapter.engine.alm_cfg[target] = self.cfg[source]
        self.executor = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix="trajsafe-plan")
        self.future: Optional[Future] = None
        self._next_id = 1

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _route_error(self, curve, route_section):
        curve = np.asarray(curve, dtype=np.float64)
        route = np.asarray(route_section, dtype=np.float64)
        values = []
        for chunk in np.array_split(curve, max(1, int(math.ceil(len(curve) / 32)))):
            d = np.linalg.norm(chunk[:, None, :] - route[None, :, :], axis=2)
            values.extend(d.min(axis=1).tolist())
        return float(np.mean(values)), float(np.max(values))

    def _body_free_rate(self, profile: PathProfile, frame: WorldSceneFrame,
                        occupancy: np.ndarray) -> float:
        half_l = 0.5 * float(self.cfg.get("ego_length_m", 4.18))
        half_w = 0.5 * float(self.cfg.get("ego_width_m", 1.99))
        heading = np.asarray(profile.heading, dtype=np.float64)
        along = np.stack((np.cos(heading), np.sin(heading)), axis=1)
        right = np.stack((-np.sin(heading), np.cos(heading)), axis=1)
        corners = np.stack([
            profile.xy + along * half_l + right * half_w,
            profile.xy + along * half_l - right * half_w,
            profile.xy - along * half_l - right * half_w,
            profile.xy - along * half_l + right * half_w,
        ], axis=1).reshape(-1, 2)
        grid = np.rint(frame.grid_from_world(corners)).astype(np.int64)
        res = int(occupancy.shape[0])
        inside = ((grid[:, 0] >= 0) & (grid[:, 0] < res)
                  & (grid[:, 1] >= 0) & (grid[:, 1] < res))
        free = np.zeros(len(grid), dtype=bool)
        free[inside] = occupancy[grid[inside, 1], grid[inside, 0]] == 0
        return float(free.mean())

    def _plan(self, plan_id: int, start_world, min_route_s: float,
              obstacle_polygons, requested: float) -> RollingPlan:
        size = float(self.cfg.get("window_size_m", 160.0))
        margin = float(self.cfg.get("window_margin_m", 10.0))
        start_s, goal_s, section = self.route.select_horizon(
            start_world, min_route_s,
            float(self.cfg.get("lookahead_m", 90.0)), size, margin)
        section = section.copy()
        section[0] = np.asarray(start_world, dtype=np.float64)
        frame = WorldSceneFrame.around_polyline(section, size_m=size,
                                                 margin_m=margin)
        occupancy = self.rasterizer.rasterize(
            frame, obstacle_polygons_world=obstacle_polygons,
            obstacle_erosion_cells=int(self.cfg.get("obstacle_erosion_cells", 8)))
        physical_occupancy = self.rasterizer.rasterize(
            frame, obstacle_polygons_world=obstacle_polygons,
            obstacle_erosion_cells=0)
        condition = frame.scene_from_world(
            np.stack((np.asarray(start_world), self.route.sample(goal_s))))
        route_limit = float(self.cfg.get("route_max_error_m", 12.0))
        base_seed = int(self.cfg.get("diffusion_seed", 43))
        attempts = max(1, int(self.cfg.get("max_seed_attempts", 4)))
        candidates = []
        total_planning_ms = 0.0
        for attempt in range(attempts):
            seed = base_seed + attempt
            result = self.adapter.plan(
                occupancy=occupancy, condition_scene=condition, frame=frame,
                seed=seed,
                model_id=str(self.cfg.get("model_id", "A_oneshot:best_task")),
                alm_enabled=True,
                sample_key="rolling_%06d_seed%d" % (plan_id, seed),
                split="rolling", index=plan_id, steps=4)
            total_planning_ms += float(result.planning_ms)
            if int(result.quality.get("reverse_steps", -1)) != 4:
                raise RuntimeError("sampler executed %s reverse steps, expected 4"
                                   % result.quality.get("reverse_steps"))
            acceptance = check_acceptance(
                result,
                wheelbase_m=float(self.controller_cfg.get("wheelbase_m", 2.641)),
                max_steer_rad=float(self.controller_cfg.get("max_steer_rad", 1.2217)),
                require_guided=bool(self.cfg.get("require_guided", True)))
            membership_limit = float(self.cfg.get(
                "min_corridor_membership", 0.95))
            for check in acceptance.get("checks", []):
                if check.get("name") == "corridor_membership":
                    value = float(check.get("value") or 0.0)
                    check.update({"ok": value >= membership_limit,
                                  "limit": ">= %.3f" % membership_limit})
            profile = PathProfile.from_world_curve(result.curve_world,
                                                    self.controller_cfg)
            curvature_index = int(np.argmax(profile.curvature))
            result.quality["curvature_peak_s_m"] = float(
                profile.s[curvature_index])
            result.quality["curvature_p99"] = float(np.percentile(
                profile.curvature, 99.0))
            endpoint_ignore = float(self.cfg.get(
                "curvature_endpoint_ignore_m", 0.6))
            is_final_horizon = goal_s >= self.route.total_length - 1e-3
            validation_end = profile.total_length - endpoint_ignore
            if not is_final_horizon:
                validation_end = min(
                    validation_end,
                    float(self.cfg.get("curvature_validation_horizon_m", 50.0)))
            interior = ((profile.s >= endpoint_ignore)
                        & (profile.s <= validation_end))
            interior_curvature = float(profile.curvature[interior].max()
                                       if interior.any()
                                       else profile.curvature.max())
            curvature_limit = (math.tan(float(self.controller_cfg.get(
                "max_steer_rad", 1.2217)))
                / float(self.controller_cfg.get("wheelbase_m", 2.641)))
            result.quality["max_abs_curvature_interior"] = interior_curvature
            # ``np.gradient`` uses a one-sided derivative at both B-spline
            # endpoints.  Those isolated samples are not a meaningful vehicle
            # curvature (the local endpoint is a handoff/braking point), so the
            # rolling gate uses the interior while retaining the raw maximum.
            for check in acceptance.get("checks", []):
                if check.get("name") == "curvature_within_vehicle":
                    check.update({
                        "name": "curvature_within_vehicle_interior",
                        "ok": interior_curvature <= curvature_limit,
                        "value": round(interior_curvature, 5),
                        "limit": "<= %.5f 1/m over executable [%.1f, %.1f] m"
                                 % (curvature_limit, endpoint_ignore,
                                    validation_end)})
            acceptance["failed"] = [row["name"] for row in
                                     acceptance.get("checks", [])
                                     if not row.get("ok")]
            acceptance["ok"] = not acceptance["failed"]
            body_rate = self._body_free_rate(profile, frame,
                                             physical_occupancy)
            required_body_rate = float(self.cfg.get(
                "required_body_free_rate", 0.99))
            body_check = {"name": "body_in_physical_drivable_area",
                          "ok": body_rate >= required_body_rate,
                          "value": round(body_rate, 5),
                          "limit": ">= %.3f" % required_body_rate}
            acceptance["checks"].append(body_check)
            result.quality["physical_body_free_rate"] = body_rate
            if not body_check["ok"]:
                acceptance["failed"].append(body_check["name"])
                acceptance["ok"] = False
            mean_error, max_error = self._route_error(profile.xy, section)
            acceptance["route"] = {"mean_error_m": mean_error,
                                   "max_error_m": max_error,
                                   "limit_m": route_limit,
                                   "ok": max_error <= route_limit}
            acceptance["ok"] = bool(acceptance["ok"]
                                    and max_error <= route_limit)
            if max_error > route_limit:
                acceptance.setdefault("failed", []).append(
                    "global_route_deviation")
            result.quality["diffusion_seed"] = seed
            result.quality["seed_attempt"] = attempt + 1
            candidates.append((result, acceptance, profile,
                               mean_error, max_error))
            if acceptance["ok"]:
                break
        # Prefer a fully accepted candidate; otherwise return the candidate
        # with the fewest failed gates and smallest curvature/route deviation.
        result, acceptance, profile, mean_error, max_error = min(
            candidates,
            key=lambda row: (not row[1]["ok"],
                             len(row[1].get("failed", [])),
                             float(row[0].quality.get("max_abs_curvature",
                                                      float("inf"))),
                             row[4]))
        result.quality["total_planning_ms"] = total_planning_ms
        result.quality["seed_attempts_run"] = len(candidates)
        result.quality["seed_attempt_diagnostics"] = [
            {"seed": int(row[0].quality["diffusion_seed"]),
             "ok": bool(row[1]["ok"]),
             "failed": list(row[1].get("failed", [])),
             "max_abs_curvature": float(row[0].quality.get(
                 "max_abs_curvature", float("nan"))),
             "curvature_peak_s_m": float(row[0].quality.get(
                 "curvature_peak_s_m", float("nan"))),
             "curvature_p99": float(row[0].quality.get(
                 "curvature_p99", float("nan"))),
             "max_abs_curvature_interior": float(row[0].quality.get(
                 "max_abs_curvature_interior", float("nan"))),
             "physical_body_free_rate": float(row[0].quality.get(
                 "physical_body_free_rate", float("nan"))),
             "route_max_error_m": float(row[4])}
            for row in candidates]
        return RollingPlan(
            plan_id=plan_id, result=result, profile=profile, frame=frame,
            occupancy=occupancy, route_start_s=start_s, route_goal_s=goal_s,
            handoff_world=np.asarray(start_world, dtype=np.float64),
            requested_wall_time=requested, completed_wall_time=time.time(),
            route_mean_error_m=mean_error, route_max_error_m=max_error,
            acceptance=acceptance)

    def submit(self, start_world, min_route_s: float,
               obstacle_polygons=None) -> int:
        if self.future is not None and not self.future.done():
            raise RuntimeError("a rolling plan is already pending")
        plan_id = self._next_id
        self._next_id += 1
        requested = time.time()
        polygons = [np.asarray(p, dtype=np.float64).copy()
                    for p in (obstacle_polygons or [])]
        self.future = self.executor.submit(
            self._plan, plan_id, np.asarray(start_world, dtype=np.float64).copy(),
            float(min_route_s), polygons, requested)
        return plan_id

    def pending(self) -> bool:
        return self.future is not None and not self.future.done()

    def poll(self) -> Optional[RollingPlan]:
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        return future.result()

    def plan_blocking(self, start_world, min_route_s: float = 0.0,
                      obstacle_polygons=None) -> RollingPlan:
        self.submit(start_world, min_route_s, obstacle_polygons)
        future, self.future = self.future, None
        return future.result()
