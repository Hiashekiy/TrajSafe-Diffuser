"""Closed-loop runner: plan once, save, then drive the same plan in CARLA.

    plan   -> Engine.generate (guided ALM), one single sampling call
           -> acceptance gate, saved to plan_test_0056.npz / .json
    drive  -> load that plan, reproduce Town03 + the recorded anchor pose,
              Pure Pursuit + PID, stop at the goal
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import occupancy as occ_mod
from .controller import PurePursuitPID
from .frame import LocalFrame, wrap_to_pi
from .path_profile import PathProfile
from .planner_adapter import PlannerAdapter, PlanResult, check_acceptance
from .scenario import (CameraRecorder, CarlaSession, destroy_actors,
                       probe_steer_sign, spawn_ego, spawn_static_vehicle)

__all__ = ["SAMPLE", "load_yaml", "resolve_paths", "run_plan", "save_plan",
           "load_plan", "run_drive"]


SAMPLE = {
    "key": "test_0056",
    "split": "test",
    "index": 56,               # index inside data/carla_processed/test
    "dataset_sample_id": 1252,
    "episode_id": 70,
    "town": "Town03",
    "anchor_raw_index": 104,
    "seed": 43,
    "model_id": "best_task",
    # goal moved up the side road to the area marked on the operator's frame:
    # local (35.4, 33.9) m = 35.4 m ahead, 33.9 m to the right.
    "condition_scene": [[-0.75, 0.0], [0.1353, 0.8473]],
}


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def repo_root() -> str:
    return occ_mod.REPO_ROOT


def workspace_root() -> str:
    return occ_mod.WORKSPACE_ROOT


@dataclass
class SampleContext:
    sample: Dict[str, Any]
    frame: LocalFrame
    occupancy: np.ndarray
    occupancy_live: Optional[np.ndarray]
    agreement: float
    episode_dir: str
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    occupancy_no_obstacles: Optional[np.ndarray] = None
    model_frame: Optional[Any] = None
    latest_acceptance: Optional[Dict[str, Any]] = None


def build_sample_context(cfg: Dict[str, Any],
                         sample: Dict[str, Any] = None) -> SampleContext:
    sample = dict(sample or SAMPLE)
    dataset_root = cfg["carla_bridge"]["dataset_root"]
    processed_root = cfg["carla_bridge"]["processed_root"]
    town = sample["town"]
    episode_dir = os.path.join(dataset_root, "episodes", town,
                               "episode_%06d" % int(sample["episode_id"]))
    frame = LocalFrame.from_episode(episode_dir, int(sample["anchor_raw_index"]))
    print("[sample] %s | %s | %s" % (sample["key"], town, frame.describe()))
    stored = occ_mod.load_dataset_occupancy(processed_root, sample["split"],
                                            sample["index"])
    live = None
    agreement = float("nan")
    try:
        quads = occ_mod.load_town_quads(dataset_root, town)
        live = occ_mod.build_canonical_occupancy(quads, frame)
        agreement = occ_mod.occupancy_agreement(live, stored)
        print("[occupancy] live CARLA raster vs frozen dataset snapshot: "
              "agreement %.4f" % agreement)
    except Exception as exc:
        print("[occupancy] live rasterisation unavailable: %s" % exc)
    source = str(cfg["planner"].get("occupancy_source", "dataset"))
    occupancy = stored if source == "dataset" else (live if live is not None else stored)
    print("[occupancy] planning source = %s (free ratio %.4f)"
          % (source, float((occupancy == 0).mean())))

    scenario_cfg = cfg.get("scenario") or {}
    margin = float(scenario_cfg.get("obstacle_margin_m", 0.35))
    obstacles: List[Dict[str, Any]] = []
    for entry in scenario_cfg.get("obstacles") or []:
        yaw_world = float(frame.yaw_deg) + float(entry.get("yaw_offset_deg", 0.0))
        xy_world = frame.to_world(np.array([[float(entry["x_local_m"]),
                                             float(entry["y_local_m"])]]))[0]
        obstacles.append({
            "name": str(entry.get("name", "obstacle")),
            "blueprint": str(entry.get("blueprint", "vehicle.audi.tt")),
            "x_local": float(entry["x_local_m"]),
            "y_local": float(entry["y_local_m"]),
            "yaw_world_deg": yaw_world,
            "length_m": float(entry.get("length_m", 4.18)),
            "width_m": float(entry.get("width_m", 1.99)),
            "world_xy": [float(xy_world[0]), float(xy_world[1])],
        })
    occupancy_before = occupancy
    if obstacles:
        occupancy = occ_mod.add_box_obstacles(occupancy, frame, obstacles, margin)
        for entry in obstacles:
            print("[obstacle] %s at world (%.2f, %.2f) local (%.2f, %.2f) "
                  "yaw %.1f deg  footprint+margin %.2f x %.2f m"
                  % (entry["name"], entry["world_xy"][0], entry["world_xy"][1],
                     entry["x_local"], entry["y_local"], entry["yaw_world_deg"],
                     entry["length_m"] + 2 * margin, entry["width_m"] + 2 * margin))
        print("[occupancy] after obstacles: free ratio %.4f (was %.4f)"
              % (float((occupancy == 0).mean()), float((occupancy_before == 0).mean())))
    return SampleContext(sample=sample, frame=frame, occupancy=occupancy,
                         occupancy_live=live, agreement=agreement,
                         episode_dir=episode_dir, obstacles=obstacles,
                         occupancy_no_obstacles=occupancy_before)


# --------------------------------------------------------------------- plan


def run_plan(cfg: Dict[str, Any], ctx: SampleContext) -> PlanResult:
    planner_cfg = cfg["planner"]
    scenario_cfg = cfg.get("scenario") or {}
    adapter = PlannerAdapter(processed_root=cfg["carla_bridge"]["processed_root"],
                             device=planner_cfg.get("device"),
                             corridor_region_override=planner_cfg.get("corridor_region"))
    result = adapter.plan(
        occupancy=ctx.occupancy,
        condition_scene=np.asarray(ctx.sample["condition_scene"], dtype=np.float64),
        frame=ctx.frame,
        seed=int(planner_cfg.get("diffusion_seed", ctx.sample.get("seed", 43))),
        model_id=str(planner_cfg.get("model_id", ctx.sample.get("model_id", "best_task"))),
        alm_enabled=bool(planner_cfg.get("alm_enabled", True)),
        sample_key=ctx.sample["key"],
        split=ctx.sample["split"],
        index=int(ctx.sample["index"]),
        obstacles=ctx.obstacles,
    )
    # the planner sees the curve as a point; this is the real 4.18 x 1.99 m body
    # against the untouched drivable area
    from .path_profile import PathProfile

    profile = PathProfile.from_world_curve(result.curve_world, cfg.get("controller") or {})
    local = ctx.frame.to_local(profile.xy)
    angle = np.arctan2(np.gradient(local[:, 1]), np.gradient(local[:, 0]))
    rate, penetration = occ_mod.body_free_rate(
        ctx.occupancy, local, angle,
        float(scenario_cfg.get("ego_length_m", 4.18)) * 0.5,
        float(scenario_cfg.get("ego_width_m", 1.99)) * 0.5)
    result.quality["body_free_rate"] = rate
    result.quality["body_penetration_m"] = penetration
    return result


def run_latest_plan(cfg: Dict[str, Any], ctx: SampleContext) -> PlanResult:
    """Plan the legacy fixed scene with the current 160 m deployment model."""
    from .continuous_demo import _global_route
    from .rolling_occupancy import CarlaMapRasterizer
    from .rolling_frame import WorldSceneFrame
    from .rolling_planner import RoutePolyline

    planner_cfg = dict(cfg.get("planner") or {})
    reverse_steps = int(planner_cfg.get("reverse_steps", 16))
    if reverse_steps < 1 or reverse_steps > 16:
        raise ValueError("planner.reverse_steps must be in [1, 16]")
    carla_cfg = dict(cfg.get("carla") or {})
    session = CarlaSession(
        host=str(carla_cfg.get("host", "127.0.0.1")),
        port=int(carla_cfg.get("port", 2000)),
        town=str(carla_cfg.get("town", "Town03_Opt")),
        fixed_dt=float(carla_cfg.get("fixed_dt", 0.05)),
        vendor_dir=str(carla_cfg.get("python_vendor", "")),
        timeout_s=float(carla_cfg.get("timeout_s", 120.0)),
        reload_world=bool(carla_cfg.get("reload_world", True)))
    try:
        session.connect()
        carla_map = session.world.get_map()
        start_world = np.asarray(ctx.frame.anchor_xy, dtype=np.float64)
        legacy_condition = np.asarray(ctx.sample["condition_scene"], dtype=np.float64)
        goal_world = ctx.frame.world_from_scene(legacy_condition)[1]
        carla = session.carla
        start_tf = carla.Transform(carla.Location(
            x=float(start_world[0]), y=float(start_world[1]), z=float(ctx.frame.z)))
        goal_tf = carla.Transform(carla.Location(
            x=float(goal_world[0]), y=float(goal_world[1]), z=float(ctx.frame.z)))
        traced = _global_route(
            session, start_tf, goal_tf,
            float(planner_cfg.get("route_sampling_resolution_m", 2.0)),
            str(carla_cfg.get("python_agents", "")))
        route_points = np.vstack((start_world, traced.xy, goal_world))
        route = RoutePolyline(route_points)
        obstacle_polygons = []
        margin = float(planner_cfg.get("dynamic_obstacle_margin_m", 0.30))
        for box in ctx.obstacles:
            center = np.asarray(box["world_xy"], dtype=np.float64)
            yaw = math.radians(float(box["yaw_world_deg"]))
            along = np.array([math.cos(yaw), math.sin(yaw)])
            right = np.array([-math.sin(yaw), math.cos(yaw)])
            half_l = 0.5 * float(box["length_m"]) + margin
            half_w = 0.5 * float(box["width_m"]) + margin
            obstacle_polygons.append(np.stack((
                center + along * half_l + right * half_w,
                center + along * half_l - right * half_w,
                center - along * half_l - right * half_w,
                center - along * half_l + right * half_w)))
        rasterizer = CarlaMapRasterizer.from_carla_map(
            carla_map, float(planner_cfg.get("lane_sample_spacing_m", 1.5)))
        _, route_section = route.section(0.0, route.total_length)
        frame = WorldSceneFrame.around_polyline(
            route_section, size_m=float(planner_cfg.get("window_size_m", 160.0)),
            margin_m=float(planner_cfg.get("window_margin_m", 12.0)))
        occupancy = rasterizer.rasterize(
            frame, obstacle_polygons_world=obstacle_polygons,
            obstacle_erosion_cells=int(planner_cfg.get(
                "obstacle_erosion_cells", 0)))
        condition = frame.scene_from_world(np.stack((start_world, goal_world)))
        adapter = PlannerAdapter(
            processed_root=planner_cfg.get("processed_root"),
            device=planner_cfg.get("device"),
            corridor_region_override=planner_cfg.get("corridor_region"),
            config_path=planner_cfg.get("model_config"))
        result = adapter.plan(
            occupancy=occupancy, condition_scene=condition, frame=frame,
            seed=int(planner_cfg.get("diffusion_seed", 43)),
            model_id=str(planner_cfg.get("model_id", "A_oneshot:best_task")),
            alm_enabled=bool(planner_cfg.get("alm_enabled", True)),
            sample_key="fixed_latest_%s" % ctx.sample["key"],
            split="fixed", index=int(ctx.sample["index"]),
            steps=reverse_steps)
        profile = PathProfile.from_world_curve(result.curve_world,
                                                cfg.get("controller") or {})
        half_l = 0.5 * float(planner_cfg.get("ego_length_m", 4.18))
        half_w = 0.5 * float(planner_cfg.get("ego_width_m", 1.99))
        along = np.stack((np.cos(profile.heading), np.sin(profile.heading)), axis=1)
        right = np.stack((-np.sin(profile.heading), np.cos(profile.heading)), axis=1)
        corners = np.stack((profile.xy + along * half_l + right * half_w,
                            profile.xy + along * half_l - right * half_w,
                            profile.xy - along * half_l - right * half_w,
                            profile.xy - along * half_l + right * half_w),
                           axis=1).reshape(-1, 2)
        grid = np.rint(frame.grid_from_world(corners)).astype(np.int64)
        inside = ((grid[:, 0] >= 0) & (grid[:, 0] < occupancy.shape[1])
                  & (grid[:, 1] >= 0) & (grid[:, 1] < occupancy.shape[0]))
        free = np.zeros(len(grid), dtype=bool)
        free[inside] = occupancy[grid[inside, 1], grid[inside, 0]] == 0
        body_rate = float(free.mean())
        acceptance = check_acceptance(
            result,
            wheelbase_m=float(cfg["controller"].get("wheelbase_m", 2.641)),
            max_steer_rad=float(cfg["controller"].get("max_steer_rad", 1.2217)),
            require_guided=bool(planner_cfg.get("require_guided", True)),
            body_free_rate=body_rate)
        distances = np.linalg.norm(
            result.curve_world[:, None, :] - route_section[None, :, :], axis=2)
        route_errors = distances.min(axis=1)
        result.quality.update({
            "body_free_rate": body_rate,
            "route_length_m": route.total_length,
            "route_mean_error_m": float(route_errors.mean()),
            "route_max_error_m": float(route_errors.max()),
            "fixed_scene_latest_model": True,
        })
        ctx.occupancy = occupancy
        ctx.occupancy_no_obstacles = None
        ctx.model_frame = frame
        ctx.latest_acceptance = acceptance
        return result
    finally:
        session.close()


def save_plan(result: PlanResult, ctx: SampleContext, cfg: Dict[str, Any],
              npz_path: str, json_path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    candidates = np.asarray(result.candidates_scene, dtype=np.float64)
    ellipse_center = np.asarray([e["center"] for e in result.ellipse_scene],
                                dtype=np.float64).reshape(-1, 2)
    ellipse_shape4 = np.asarray([e["shape4"] for e in result.ellipse_scene],
                                dtype=np.float64).reshape(-1, 4)
    corridor_lengths = np.asarray([len(p) for p in result.corridor_scene], dtype=np.int64)
    corridor_flat = (np.concatenate(result.corridor_scene, axis=0)
                     if result.corridor_scene
                     else np.zeros((0, 2), dtype=np.float64))
    corridor_world_lengths = np.asarray([len(p) for p in result.corridor_world],
                                        dtype=np.int64)
    corridor_world_flat = (np.concatenate(result.corridor_world, axis=0)
                           if result.corridor_world
                           else np.zeros((0, 2), dtype=np.float64))
    ellipse_world_lengths = np.asarray([len(p) for p in result.ellipse_world],
                                       dtype=np.int64)
    ellipse_world_flat = (np.concatenate(result.ellipse_world, axis=0)
                          if result.ellipse_world
                          else np.zeros((0, 2), dtype=np.float64))
    np.savez_compressed(
        npz_path,
        curve_scene=result.curve_scene,
        curve_world=result.curve_world,
        raw_curve_scene=result.raw_curve_scene,
        raw_curve_world=result.raw_curve_world,
        controls_scene=result.controls_scene,
        candidates_scene=candidates,
        candidates_world=np.asarray(result.candidates_world, dtype=np.float64),
        condition_scene=np.asarray(result.condition_scene, dtype=np.float64),
        ellipse_center=ellipse_center,
        ellipse_shape4=ellipse_shape4,
        ellipse_world_flat=ellipse_world_flat,
        ellipse_world_lengths=ellipse_world_lengths,
        corridor_flat=corridor_flat,
        corridor_lengths=corridor_lengths,
        corridor_world_flat=corridor_world_flat,
        corridor_world_lengths=corridor_world_lengths,
        start_world=np.asarray(result.start_world, dtype=np.float64),
        goal_world=np.asarray(result.goal_world, dtype=np.float64),
        occupancy=np.asarray(ctx.occupancy, dtype=np.uint8),
    )
    payload = {
        "sample": ctx.sample,
        "frame": {"anchor_xy": ctx.frame.anchor_xy.tolist(),
                  "forward_xy": ctx.frame.forward_xy.tolist(),
                  "right_xy": ctx.frame.right_xy.tolist(),
                  "yaw_deg": ctx.frame.yaw_deg, "z": ctx.frame.z},
        "occupancy": {"source": cfg["planner"].get("occupancy_source", "dataset"),
                      "live_vs_dataset_agreement": ctx.agreement},
        "planning_ms": result.planning_ms,
        "guided": result.guided,
        "alm_status": result.alm_status,
        "selected_index": result.selected_index,
        "validation": result.validation,
        "quality": result.quality,
        "start_world": np.asarray(result.start_world).tolist(),
        "goal_world": np.asarray(result.goal_world).tolist(),
        "ellipse": [{"center": np.asarray(e["center"]).tolist(),
                     "shape4": np.asarray(e["shape4"]).tolist()}
                    for e in result.ellipse_scene],
        "corridor_cells": [np.asarray(p).tolist() for p in result.corridor_scene],
        "npz": os.path.basename(npz_path),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def load_plan(npz_path: str) -> PlanResult:
    data = np.load(npz_path)
    centers = np.asarray(data["ellipse_center"], dtype=np.float64)
    shapes = np.asarray(data["ellipse_shape4"], dtype=np.float64)
    lengths = np.asarray(data["corridor_lengths"], dtype=np.int64)
    flat = np.asarray(data["corridor_flat"], dtype=np.float64)
    def _split(flat, lengths):
        out, offset = [], 0
        for count in lengths:
            out.append(flat[offset:offset + int(count)])
            offset += int(count)
        return out

    corridor = _split(flat, lengths)
    corridor_world = _split(np.asarray(data["corridor_world_flat"], dtype=np.float64),
                            np.asarray(data["corridor_world_lengths"], dtype=np.int64))
    ellipse_world = _split(np.asarray(data["ellipse_world_flat"], dtype=np.float64),
                           np.asarray(data["ellipse_world_lengths"], dtype=np.int64))
    return PlanResult(
        curve_scene=np.asarray(data["curve_scene"], dtype=np.float64),
        curve_world=np.asarray(data["curve_world"], dtype=np.float64),
        raw_curve_scene=np.asarray(data["raw_curve_scene"], dtype=np.float64),
        raw_curve_world=np.asarray(data["raw_curve_world"], dtype=np.float64),
        controls_scene=np.asarray(data["controls_scene"], dtype=np.float64),
        candidates_scene=np.asarray(data["candidates_scene"], dtype=np.float64),
        candidates_world=np.asarray(data["candidates_world"], dtype=np.float64),
        ellipse_scene=[{"center": c, "shape4": s} for c, s in zip(centers, shapes)],
        ellipse_world=ellipse_world,
        corridor_scene=corridor,
        corridor_world=corridor_world,
        condition_scene=np.asarray(data["condition_scene"], dtype=np.float64),
        start_world=np.asarray(data["start_world"], dtype=np.float64),
        goal_world=np.asarray(data["goal_world"], dtype=np.float64),
        guided=True, alm_status="guided",
    )


# -------------------------------------------------------------------- drive


def run_drive(cfg: Dict[str, Any], ctx: SampleContext, plan: PlanResult,
              out_dir: str, record_video: bool = True) -> Dict[str, Any]:
    carla_cfg = cfg["carla"]
    ctrl_cfg = dict(cfg.get("controller") or {})
    dt = float(carla_cfg.get("fixed_dt", 0.05))

    path = PathProfile.from_world_curve(plan.curve_world, ctrl_cfg)
    print("[path] %s" % path.describe())
    planning_frame = ctx.model_frame or ctx.frame
    if ctx.model_frame is None:
        in_window = ctx.frame.in_window(ctx.frame.to_local(plan.curve_world))
    else:
        in_window = ctx.model_frame.contains_world(plan.curve_world)
    if not in_window:
        raise RuntimeError("planned curve leaves the planning window")

    session = CarlaSession(host=carla_cfg.get("host", "127.0.0.1"),
                           port=int(carla_cfg.get("port", 2000)),
                           town=str(carla_cfg.get("town", ctx.sample["town"])),
                           fixed_dt=dt,
                           vendor_dir=str(carla_cfg.get("python_vendor", "")),
                           timeout_s=float(carla_cfg.get("timeout_s", 60.0)),
                           reload_world=bool(carla_cfg.get("reload_world", True)))
    manifest: Dict[str, Any] = {"sample": ctx.sample,
                                "path": path.describe(),
                                "carla": {"host": session.host, "port": session.port,
                                          "town": session.town, "fixed_dt": dt}}
    actors: List[Any] = []
    recorder = None
    try:
        session.connect()
        blueprint = str(carla_cfg.get("ego_blueprint", "vehicle.audi.tt"))
        spawn_z = float(carla_cfg.get("spawn_z_offset_m", 0.30))

        # 1) steering-sign probe with a throwaway vehicle -------------------
        probe_sign = float(ctrl_cfg.get("steer_sign", 0.0) or 0.0)
        if not probe_sign:
            probe, _ = spawn_ego(session, ctx.frame, blueprint, spawn_z,
                                 role_name="steerProbe")
            for _ in range(4):
                session.world.tick()
            info = probe_steer_sign(session, probe,
                                    magnitude=float(ctrl_cfg.get("probe_steer_rad", 0.2)),
                                    speed=float(ctrl_cfg.get("probe_speed_mps", 1.5)),
                                    ticks=int(ctrl_cfg.get("probe_ticks", 30)))
            manifest["steer_probe"] = info
            if info["usable"]:
                probe_sign = info["steer_sign"]
            else:
                probe_sign = float(ctrl_cfg.get("default_steer_sign", 1.0))
                print("[control] probe unusable -> default steer_sign %+.1f"
                      % probe_sign)
            destroy_actors([probe])
            for _ in range(4):
                session.world.tick()
        manifest["steer_sign"] = probe_sign
        print("[control] using steer_sign %+.1f" % probe_sign)

        # 2) static obstacles (parked cars the planner was told about) -------
        obstacle_actors: List[Any] = []
        for entry in ctx.obstacles:
            actor = spawn_static_vehicle(session, entry["blueprint"],
                                         entry["world_xy"], entry["yaw_world_deg"],
                                         spawn_z, role_name="obstacle_" + entry["name"])
            obstacle_actors.append(actor)
            actors.append(actor)
            session.world.tick()

        # 3) real ego ------------------------------------------------------
        ego, spawn_transform = spawn_ego(session, ctx.frame, blueprint, spawn_z)
        actors.append(ego)
        collisions: List[Dict[str, Any]] = []
        collision_bp = session.world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = session.world.spawn_actor(collision_bp,
                                                     session.carla.Transform(),
                                                     attach_to=ego)
        actors.append(collision_sensor)
        collision_sensor.listen(lambda event: collisions.append({
            "frame": int(event.frame),
            "other": str(event.other_actor.type_id) if event.other_actor else None,
            "impulse": float(event.normal_impulse.x) ** 2 + float(event.normal_impulse.y) ** 2,
        }))

        alm_enabled = bool(cfg["planner"].get("alm_enabled", True))
        mode = "guided" if alm_enabled else "raw"
        plan_free = occ_mod.free_mask(ctx.occupancy,
                                      np.asarray(plan.curve_scene, dtype=np.float64))
        plan_free_rate = float(plan_free.mean())
        body_gap = plan.quality.get("min_obstacle_body_gap_m")
        print("[plan] mode=%s  points off-road %d/%d (%.1f%%)"
              % (mode, int((~plan_free).sum()), len(plan_free),
                 100.0 * (1.0 - plan_free_rate)))

        if record_video:
            video_cfg = cfg.get("video") or {}
            video_path = os.path.join(out_dir, "trajsafe_%s.mp4" % mode)
            from .overlay import BirdEyeProjector, OverlayGeometry

            projector = BirdEyeProjector(int(video_cfg.get("width", 1280)),
                                         int(video_cfg.get("height", 720)),
                                         float(video_cfg.get("fov", 90.0)),
                                         float(video_cfg.get("camera_height_m", 70.0)))
            overlay = None
            center_world = None
            if bool(video_cfg.get("overlay", True)):
                overlay = OverlayGeometry(projector, plan, ctx.obstacles, planning_frame,
                                          occupancy=ctx.occupancy)
            if bool(video_cfg.get("fixed_center", True)):
                center = video_cfg.get("center_local_m", [30.0, 0.0])
                center_world = ctx.frame.to_world(
                    np.array([[float(center[0]), float(center[1])]]))[0]
            print("[video] %.1f x %.1f m visible, %.2f px/m, fixed=%s, overlay=%s"
                  % (projector.visible_width_m(), projector.visible_height_m(),
                     projector.s, center_world is not None, overlay is not None))
            recorder = CameraRecorder(session, ego, video_path,
                                      width=int(video_cfg.get("width", 1280)),
                                      height=int(video_cfg.get("height", 720)),
                                      fps=int(video_cfg.get("fps", 10)),
                                      height_m=float(video_cfg.get("camera_height_m", 70.0)),
                                      fov=float(video_cfg.get("fov", 90.0)),
                                      enabled=bool(video_cfg.get("enabled", True)),
                                      world_up=bool(video_cfg.get("world_up", True)),
                                      fixed_center_world=center_world,
                                      overlay=overlay,
                                      overlay_options={
                                          "show_corridor": bool(video_cfg.get("show_corridor", True)),
                                          "show_ellipses": bool(video_cfg.get("show_ellipses", False)),
                                          "show_candidates": bool(video_cfg.get("show_candidates", False)),
                                          "show_raw": bool(video_cfg.get("show_raw", alm_enabled)),
                                          "show_occupancy": bool(video_cfg.get("show_occupancy", True)),
                                          "show_executed": bool(video_cfg.get("show_executed", True))})
            manifest["video"] = {"path": video_path, "errors": recorder.errors,
                                 "visible_width_m": projector.visible_width_m(),
                                 "pixels_per_m": projector.s}
        else:
            projector = None
        executed = []

        controller = PurePursuitPID(ctrl_cfg, steer_sign=probe_sign)
        session.world.tick()
        ego.apply_control(session.carla.VehicleControl(hand_brake=False))
        for _ in range(5):
            session.world.tick()

        max_frames = int(float(carla_cfg.get("max_sim_seconds", 60.0)) / dt)
        abort_cross = float(ctrl_cfg.get("tracking_abort_m", 1.0))
        max_cross = 0.0
        history: List[Dict[str, Any]] = []
        status = "running"
        start_wall = time.time()
        for frame in range(max_frames):
            session.world.tick()
            transform = ego.get_transform()
            velocity = ego.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
            command = controller.command(path,
                                         np.array([transform.location.x,
                                                   transform.location.y]),
                                         float(transform.rotation.yaw),
                                         speed, dt)
            longitudinal_mode = str(ctrl_cfg.get("longitudinal_mode", "speed_pid"))
            if command.brake_hard and not command.stop:
                steer_limit = max(float(ctrl_cfg.get("max_steer_rad", 0.61)), 1e-3)
                steer_norm = float(np.clip(command.steer_rad / steer_limit, -1.0, 1.0))
                ego.apply_control(session.carla.VehicleControl(
                    throttle=0.0, brake=1.0, steer=steer_norm, hand_brake=False))
            elif longitudinal_mode == "accel":
                ackermann = session.carla.VehicleAckermannControl(
                    steer=float(command.steer_rad),
                    steer_speed=float(ctrl_cfg.get("steer_speed_rad_s", 0.6)),
                    speed=0.0,
                    acceleration=float(command.accel_mps2),
                    jerk=float(ctrl_cfg.get("jerk_mps3", 3.0)))
            else:
                ackermann = session.carla.VehicleAckermannControl(
                    steer=float(command.steer_rad),
                    steer_speed=float(ctrl_cfg.get("steer_speed_rad_s", 1.0)),
                    speed=float(command.v_ref_mps),
                    acceleration=float(command.accel_mps2),
                    jerk=float(ctrl_cfg.get("jerk_mps3", 3.0)))
            ego.apply_ackermann_control(ackermann)
            history.append({
                "frame": frame, "t": round(frame * dt, 4),
                "x": transform.location.x, "y": transform.location.y,
                "z": transform.location.z, "yaw_deg": transform.rotation.yaw,
                "v_mps": speed, "v_ref_mps": command.v_ref_mps,
                "accel_cmd": command.accel_mps2,
                "steer_cmd_rad": command.steer_rad,
                "s_m": command.s_m, "cross_track_m": command.cross_track_m,
                "heading_error_rad": command.heading_error_rad,
                "remaining_m": command.remaining_m})
            max_cross = max(max_cross, abs(command.cross_track_m))
            executed.append((transform.location.x, transform.location.y))
            if recorder is not None:
                recorder.follow()
                recorder.executed_world = np.asarray(executed, dtype=np.float64)
                recorder.hud_lines = [
                    "TrajSafe-Diffuser latest   %s   %d-step DDIM" %
                    (mode.upper(), int(plan.quality.get("reverse_steps", 0))),
                    "t=%.1fs   v=%.2f m/s   s=%.1f / %.1f m"
                    % (frame * dt, speed, command.s_m, path.total_length),
                    "cross-track %+.2f m (max %.2f)   heading err %+.1f deg"
                    % (command.cross_track_m, max_cross,
                       math.degrees(command.heading_error_rad)),
                    "plan off-road %.1f%%   body gap %s m   collisions %d"
                    % (100.0 * (1.0 - plan_free_rate),
                       "n/a" if body_gap is None else "%.2f" % body_gap,
                       len(collisions)),
                ]
                recorder.write_pending()
            # keep the parked cars parked (a vehicle with no control set can
            # still creep); the ego is deliberately NOT in this list
            for actor in obstacle_actors:
                try:
                    actor.apply_control(session.carla.VehicleControl(hand_brake=True))
                except Exception:
                    pass
            if collisions:
                status = "collision"
                break
            if abs(command.cross_track_m) > abort_cross:
                status = "lateral_abort"
                break
            if command.stop:
                status = "arrived"
                break
        else:
            status = "timeout"

        try:
            ego.apply_control(session.carla.VehicleControl(hand_brake=True))
        except Exception:
            pass
        for _ in range(5):
            session.world.tick()
            if recorder is not None:
                recorder.write_pending()

        final = ego.get_transform()
        goal_xy = np.asarray(plan.curve_world[-1], dtype=np.float64)
        final_xy = np.array([final.location.x, final.location.y])
        cross = np.array([h["cross_track_m"] for h in history], dtype=np.float64)
        speeds = np.array([h["v_mps"] for h in history], dtype=np.float64)
        manifest.update({
            "status": status,
            "wall_seconds": round(time.time() - start_wall, 2),
            "frames": len(history),
            "sim_seconds": round(len(history) * dt, 2),
            "final_pose": {"x": final.location.x, "y": final.location.y,
                           "yaw_deg": final.rotation.yaw},
            "goal_world": goal_xy.tolist(),
            "final_distance_to_goal_m": float(np.linalg.norm(final_xy - goal_xy)),
            "cross_track": {"rms_m": float(np.sqrt((cross ** 2).mean())) if len(cross) else None,
                            "max_abs_m": float(np.abs(cross).max()) if len(cross) else None},
            "speed": {"mean_mps": float(speeds.mean()) if len(speeds) else None,
                      "max_mps": float(speeds.max()) if len(speeds) else None},
            "collisions": collisions,
            "mode": mode,
            "alm_status": plan.alm_status,
            "plan_off_road_points": int((~plan_free).sum()),
            "plan_free_rate": plan_free_rate,
            "plan_min_obstacle_body_gap_m": body_gap,
            "video_frames": 0 if recorder is None else recorder.frames,
            "spawn_transform": {"x": spawn_transform.location.x,
                                "y": spawn_transform.location.y,
                                "z": spawn_transform.location.z,
                                "yaw_deg": spawn_transform.rotation.yaw},
        })
        if history:
            csv_path = os.path.join(out_dir, "trajectory_%s.csv" % mode)
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
            manifest["trajectory_csv"] = csv_path
            # drawn only AFTER the csv exists, otherwise it would plot the
            # previous run's trace
            try:
                from .visualize import plot_executed_trace

                occupancy = np.load(os.path.join(
                    out_dir, "plan_%s.npz" % ctx.sample["key"]))["occupancy"]
                manifest["overlay_png"] = plot_executed_trace(
                    csv_path, plan, occupancy,
                    os.path.join(out_dir, "plan_vs_executed_%s.png" % mode),
                    frame=planning_frame)
                print("[plot] overlay -> %s" % manifest["overlay_png"])
            except Exception as exc:  # pragma: no cover - plotting is optional
                print("[plot] overlay skipped: %s" % exc)
        return manifest
    finally:
        if recorder is not None:
            recorder.close()
        destroy_actors(actors)
        session.close()
