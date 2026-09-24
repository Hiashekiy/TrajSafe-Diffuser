"""Planning adapter: one Engine.generate call, plus the hard acceptance gate.

The adapter owns no model logic.  It calls the SAME
diffusion-dashboard/engine_carla.py::Engine the Diffusion Lens dashboard calls,
with the same checkpoint, seed, occupancy and condition, and then converts the
scene-space result into every frame the CARLA side needs.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .frame import LocalFrame, wrap_to_pi
from . import occupancy as occ_mod

__all__ = ["PlanResult", "PlannerAdapter", "check_acceptance"]


@dataclass
class PlanResult:
    """Everything the closed-loop side consumes, in both scene and world frame."""

    curve_scene: np.ndarray          # [128,2]
    curve_world: np.ndarray          # [128,2]
    raw_curve_scene: np.ndarray
    raw_curve_world: np.ndarray
    controls_scene: np.ndarray       # [32,2]
    candidates_scene: np.ndarray     # [M,128,2]
    candidates_world: np.ndarray
    ellipse_scene: List[Dict[str, Any]] = field(default_factory=list)
    ellipse_world: List[np.ndarray] = field(default_factory=list)
    corridor_scene: List[np.ndarray] = field(default_factory=list)
    corridor_world: List[np.ndarray] = field(default_factory=list)
    selected_index: int = -1
    guided: bool = False
    alm_status: str = "unknown"
    planning_ms: float = 0.0
    validation: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    condition_scene: Optional[np.ndarray] = None
    start_world: Optional[np.ndarray] = None
    goal_world: Optional[np.ndarray] = None

    # -------------------------------------------------------------- geometry
    def max_abs_curvature(self, spacing_m: float = 0.20) -> float:
        """Max |curvature| AFTER arc-length resampling.

        The raw 128 scene samples are not uniform in arc length, so a plain
        np.gradient on them reports garbage (it produced 9.5 1/m for a
        perfectly smooth curve).  Curvature is only meaningful on a uniform
        arc-length table, which is also what the controller consumes.
        """
        from .path_profile import resample_by_arc_length

        xy = np.asarray(self.curve_world, dtype=np.float64)
        if len(xy) < 3:
            return 0.0
        xy, s = resample_by_arc_length(xy, float(spacing_m))
        d = np.gradient(xy, axis=0)
        ds = np.linalg.norm(d, axis=1)
        ds[ds < 1e-9] = 1e-9
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        kappa = np.gradient(heading) / ds
        return float(np.abs(kappa).max())

    def length_m(self) -> float:
        xy = np.asarray(self.curve_world, dtype=np.float64)
        return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _ellipse_vertices_scene(center_scene, shape4, n: int = 48) -> np.ndarray:
    """Return an ellipse outline in scene coordinates.

    The ellipse head predicts semi-axes in scene units.  Keeping the outline in
    scene coordinates avoids the old bridge's hard-coded 40 m scale and makes
    the adapter work with both the legacy 80 m frame and the 160 m world frame.
    """
    log_a, log_b, cos2t, sin2t = [float(v) for v in shape4[:4]]
    a_scene = math.exp(log_a)
    b_scene = math.exp(log_b)
    theta = 0.5 * math.atan2(sin2t, cos2t)
    t = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False)
    scene_pts = np.stack((a_scene * np.cos(t), b_scene * np.sin(t)), axis=1)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]], dtype=np.float64)
    center = np.asarray(center_scene, dtype=np.float64).reshape(1, 2)
    return scene_pts @ rot.T + center


class PlannerAdapter:
    """Thin wrapper around the dashboard inference engine."""

    def __init__(self, processed_root: Optional[str] = None, device: str = None,
                 corridor_region_override: Optional[Dict[str, Any]] = None,
                 config_path: Optional[str] = None):
        import torch

        # In an installed bridge ``REPO_ROOT`` is the model checkout.  During
        # package development this file is one level deeper, so also accept the
        # current checkout; this keeps the source package directly testable.
        model_root = occ_mod.REPO_ROOT
        if not os.path.isdir(os.path.join(model_root, "diffusion-dashboard")):
            cwd = os.path.abspath(os.getcwd())
            if os.path.isdir(os.path.join(cwd, "diffusion-dashboard")):
                model_root = cwd
        if model_root not in sys.path:
            sys.path.insert(0, model_root)
        # Running the uninstalled bridge imports ``src.carla_bridge`` from a
        # namespace package before the repository's regular ``src`` package is
        # visible.  Extend that already-loaded namespace so dashboard imports
        # such as ``src.utils`` resolve without requiring installation first.
        src_package = sys.modules.get("src")
        model_src = os.path.join(model_root, "src")
        if src_package is not None and hasattr(src_package, "__path__") \
                and model_src not in src_package.__path__:
            src_package.__path__.append(model_src)
        dashboard = os.path.join(model_root, "diffusion-dashboard")
        if not os.path.isfile(os.path.join(dashboard, "engine_carla.py")):
            raise FileNotFoundError("cannot locate diffusion-dashboard/engine_carla.py")
        if dashboard not in sys.path:
            sys.path.insert(0, dashboard)
        from engine_carla import Engine  # noqa: E402  (path injected above)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.engine = Engine(device, processed_root=processed_root,
                             config_path=config_path)
        if corridor_region_override:
            # The convex corridor is what actually keeps the curve away from the
            # kerb, and it defaults to an 0.8 m margin -- less than the 0.995 m
            # half width of the car, so a "free" centreline still put the body
            # over the edge.  The bridge raises it to half width + margin.
            region = dict((self.engine.corridor_cfg or {}).get("region") or {})
            region.update(dict(corridor_region_override))
            self.engine.corridor_cfg = dict(self.engine.corridor_cfg or {},
                                            region=region)
            print("[planner] corridor region override: %s" % region)

    # ------------------------------------------------------------------ plan
    def plan(self, occupancy: np.ndarray, condition_scene: np.ndarray,
             frame: LocalFrame, seed: int = 43, model_id: str = "best_task",
             alm_enabled: bool = True, sample_key: str = "test_0056",
             split: str = "test", index: int = 56,
             ellipse_points: int = 48, obstacles=None,
             steps: Optional[int] = None,
             times: Optional[List[int]] = None) -> PlanResult:
        condition = np.asarray(condition_scene, dtype=np.float32).reshape(2, 2)
        t0 = time.perf_counter()
        payload = self.engine.generate(
            sample_key=sample_key, split=split, index=int(index),
            occupancy=np.asarray(occupancy, dtype=np.float32),
            condition=condition, seed=int(seed), model_id=model_id,
            alm_enabled=bool(alm_enabled),
            steps=steps, times=times,
        )
        planning_ms = (time.perf_counter() - t0) * 1000.0

        curve_scene = np.asarray(payload["x0_history"][-1], dtype=np.float64)
        raw_scene = np.asarray(payload["x0_raw_history"][-1], dtype=np.float64)
        controls_scene = np.asarray(payload["control_history"][-1], dtype=np.float64)
        candidates_scene = np.asarray(payload["topology"]["candidate_paths"],
                                      dtype=np.float64)
        mask = np.asarray(payload["topology"].get("candidate_mask",
                                                  np.ones(len(candidates_scene))),
                          dtype=bool)

        ell_scene: List[Dict[str, Any]] = []
        ell_world: List[np.ndarray] = []
        last_ellipse = payload["ellipse_history"][-1]
        for center, shape4 in zip(np.asarray(last_ellipse["center"]),
                                  np.asarray(last_ellipse["shape4"])):
            ell_scene.append({"center": np.asarray(center, dtype=np.float64),
                              "shape4": np.asarray(shape4, dtype=np.float64)})
            ell_world.append(frame.world_from_scene(
                _ellipse_vertices_scene(center, shape4, ellipse_points)))

        corr_scene: List[np.ndarray] = []
        corr_world: List[np.ndarray] = []
        corridor = payload.get("corridor") or {}
        for cell in corridor.get("cells", []):
            poly = np.asarray(cell.get("polygon") or [], dtype=np.float64)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue
            corr_scene.append(poly)
            corr_world.append(frame.world_from_scene(poly))

        validation = dict(payload.get("final_validation") or {})
        validation["alm_status"] = payload["alm"]["status"]
        validation["guided"] = bool(payload["activation"].get("guided", False))
        validation["selected_idx"] = int(payload["topology"]["selected_idx"])
        validation["curve_in_free"] = occ_mod.scene_points_are_free(occupancy,
                                                                   curve_scene)
        validation["raw_in_free"] = occ_mod.scene_points_are_free(occupancy,
                                                                 raw_scene)
        validation["free_rate_curve"] = float(
            occ_mod.free_mask(occupancy, curve_scene).mean())

        result = PlanResult(
            curve_scene=curve_scene,
            curve_world=frame.world_from_scene(curve_scene),
            raw_curve_scene=raw_scene,
            raw_curve_world=frame.world_from_scene(raw_scene),
            controls_scene=controls_scene,
            candidates_scene=candidates_scene,
            candidates_world=frame.world_from_scene(
                candidates_scene.reshape(-1, 2)).reshape(candidates_scene.shape),
            ellipse_scene=ell_scene,
            ellipse_world=ell_world,
            corridor_scene=corr_scene,
            corridor_world=corr_world,
            selected_index=int(payload["topology"]["selected_idx"]),
            guided=bool(payload["activation"].get("guided", False)),
            alm_status=str(payload["alm"]["status"]),
            planning_ms=planning_ms,
            validation=validation,
            quality={"candidate_mask": mask.tolist(),
                     "num_candidates": int(candidates_scene.shape[0]),
                     "candidate_count_valid": int(mask.sum()),
                     "completion_rate": float(payload["topology"]
                                              .get("completion_rate", 0.0))
                     if isinstance(payload["topology"], dict) else 0.0,
                     "reverse_steps": int(payload.get("steps", 0))},
            condition_scene=np.asarray(condition, dtype=np.float64),
            start_world=frame.world_from_scene(
                np.asarray(condition[0], dtype=np.float64).reshape(1, 2))[0],
            goal_world=frame.world_from_scene(
                np.asarray(condition[1], dtype=np.float64).reshape(1, 2))[0],
        )
        result.quality["length_m"] = result.length_m()
        result.quality["endpoint_scene_error"] = float(
            np.linalg.norm(result.curve_scene[-1] - condition[1]))
        result.quality["endpoint_world_error_m"] = float(
            np.linalg.norm(result.curve_world[-1] - result.goal_world))
        result.quality["max_abs_curvature"] = result.max_abs_curvature()
        if obstacles:
            # the planner treats the curve as a point; this is the real body gap
            result.quality["min_obstacle_body_gap_m"] = obstacle_body_gap(
                result, frame, obstacles)
        return result


def obstacle_body_gap(result: PlanResult, frame: LocalFrame, obstacles,
                      ego_half_length_m: float = 2.09,
                      ego_half_width_m: float = 0.995,
                      samples: int = 64) -> float:
    """Smallest body-to-body gap between the planned ego and any box obstacle.

    The planner treats the trajectory as a POINT, so an obstacle mask only keeps
    the centre line away.  This measures the real thing: it walks the ego
    footprint along the planned curve and returns the minimum distance to every
    obstacle rectangle (which is why obstacles are stored in local metres).
    """
    curve_local = frame.to_local(np.asarray(result.curve_world, dtype=np.float64))
    if len(curve_local) < 2:
        return float("inf")
    headings = np.arctan2(np.gradient(curve_local[:, 1]),
                          np.gradient(curve_local[:, 0]))
    offsets = np.stack((ego_half_length_m * np.cos(np.linspace(0.0, 2.0 * math.pi,
                                                               int(samples))),
                        ego_half_width_m * np.sin(np.linspace(0.0, 2.0 * math.pi,
                                                              int(samples)))),
                       axis=1)
    best = float("inf")
    for box in obstacles:
        cx = float(box["x_local"])
        cy = float(box["y_local"])
        heading = math.radians(float(box["yaw_world_deg"]) - float(frame.yaw_deg))
        along = np.array([math.cos(heading), math.sin(heading)])
        across = np.array([-math.sin(heading), math.cos(heading)])
        half_l = 0.5 * float(box["length_m"])
        half_w = 0.5 * float(box["width_m"])
        for index in range(len(curve_local)):
            rot = np.array([[math.cos(headings[index]), -math.sin(headings[index])],
                            [math.sin(headings[index]), math.cos(headings[index])]])
            points = curve_local[index][None, :] + offsets @ rot.T
            delta = points - np.array([cx, cy])
            local_x = delta @ across      # box width direction
            local_y = delta @ along       # box length direction
            gap = np.hypot(np.maximum(np.abs(local_x) - half_w, 0.0),
                           np.maximum(np.abs(local_y) - half_l, 0.0))
            best = min(best, float(gap.min()))
    return best


def check_acceptance(result: PlanResult, wheelbase_m: float = 2.8,
                     max_steer_rad: float = math.radians(70.0),
                     require_guided: bool = True,
                     min_obstacle_body_gap_m: Optional[float] = None,
                     required_body_gap_m: float = 0.30,
                     body_free_rate: Optional[float] = None,
                     required_body_free_rate: float = 0.99) -> Dict[str, Any]:
    """Hard gate from section 7 of the integration plan."""
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, value: Any, limit: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "value": value,
                       "limit": limit})

    v = result.validation

    def num(value, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if out == out else float(default)   # NaN -> default

    add("guided", result.guided, result.guided, "True")
    add("alm_status_guided", result.alm_status == "guided", result.alm_status,
        "guided")
    add("final_collision_false", not bool(v.get("final_collision", True)),
        v.get("final_collision"), "False")
    add("endpoint_error", num(v.get("endpoint_error"), 1.0) <= 1e-3,
        v.get("endpoint_error"), "<= 1e-3 scene")
    add("max_constraint_violation",
        num(v.get("final_max_constraint_violation"), 1.0) <= 5e-3,
        v.get("final_max_constraint_violation"), "<= 5e-3 scene")
    add("corridor_membership",
        num(v.get("final_corridor_membership_rate"), 0.0) >= 0.99,
        v.get("final_corridor_membership_rate"), ">= 0.99")
    add("dense_curve_free", bool(v.get("curve_in_free", False)),
        v.get("curve_in_free"), "True")

    kappa_max = result.max_abs_curvature()
    kappa_limit = math.tan(float(max_steer_rad)) / float(wheelbase_m)
    add("curvature_within_vehicle", kappa_max <= kappa_limit,
        round(kappa_max, 5), "<= %.5f 1/m" % kappa_limit)

    if min_obstacle_body_gap_m is not None:
        add("obstacle_body_gap", float(min_obstacle_body_gap_m) >= float(required_body_gap_m),
            None if min_obstacle_body_gap_m != min_obstacle_body_gap_m
            else round(float(min_obstacle_body_gap_m), 3),
            ">= %.2f m" % float(required_body_gap_m))

    if body_free_rate is not None:
        add("body_in_drivable_area", num(body_free_rate, 0.0) >= float(required_body_free_rate),
            round(num(body_free_rate, 0.0), 4), ">= %.2f" % float(required_body_free_rate))

    ok = all(c["ok"] for c in checks) if require_guided else True
    return {"ok": bool(ok), "checks": checks,
            "failed": [c["name"] for c in checks if not c["ok"]]}
