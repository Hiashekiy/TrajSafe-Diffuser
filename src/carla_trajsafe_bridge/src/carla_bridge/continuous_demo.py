"""Long-distance CARLA drive with asynchronous four-step replanning."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from .controller import PurePursuitPID
from .frame import wrap_to_pi
from .rolling_occupancy import (CarlaMapRasterizer,
                                vehicle_obstacle_polygons)
from .rolling_planner import FourStepRollingPlanner, RoutePolyline
from .scenario import (CameraRecorder, CarlaSession, destroy_actors,
                       spawn_ego_at_transform)


def _global_route(session: CarlaSession, start_transform, goal_transform,
                  sampling_resolution: float, agents_path: str = "") -> RoutePolyline:
    candidates = [agents_path]
    vendor = str(getattr(session, "vendor_dir", "") or "")
    if vendor:
        candidates.extend([
            os.path.dirname(vendor),
            os.path.join(os.path.dirname(os.path.dirname(vendor)),
                         "PythonAPI", "carla"),
        ])
    candidates.extend([
        "E:/CARLA_0.9.16/PythonAPI/carla",
        "C:/CARLA_0.9.16/PythonAPI/carla",
    ])
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "agents")):
            absolute = os.path.abspath(candidate)
            if absolute not in sys.path:
                sys.path.insert(0, absolute)
            break
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ImportError as exc:
        raise RuntimeError(
            "CARLA agents package is unavailable; add PythonAPI/carla to "
            "python_vendor before running the continuous demo") from exc
    planner = GlobalRoutePlanner(session.world.get_map(),
                                 float(sampling_resolution))
    traced = planner.trace_route(start_transform.location,
                                  goal_transform.location)
    if len(traced) < 2:
        raise RuntimeError("CARLA global route planner returned no route")
    return RoutePolyline([[wp.transform.location.x, wp.transform.location.y]
                          for wp, _option in traced])


def _pick_transforms(carla_map, route_cfg):
    spawns = carla_map.get_spawn_points()
    if len(spawns) < 2:
        raise RuntimeError("map has fewer than two spawn points")
    start_index = int(route_cfg.get("start_spawn_index", 0)) % len(spawns)
    goal_index = int(route_cfg.get("goal_spawn_index", -1))
    if goal_index < 0:
        start = spawns[start_index].location
        distance = [(i, (p.location.x - start.x) ** 2
                        + (p.location.y - start.y) ** 2)
                    for i, p in enumerate(spawns) if i != start_index]
        goal_index = max(distance, key=lambda row: row[1])[0]
    goal_index %= len(spawns)
    if goal_index == start_index:
        raise ValueError("start and goal spawn indices must differ")
    return spawns[start_index], spawns[goal_index], start_index, goal_index


def _apply_command(session, ego, command, ctrl_cfg):
    if command.brake_hard:
        limit = max(float(ctrl_cfg.get("max_steer_rad", 1.2217)), 1e-3)
        ego.apply_control(session.carla.VehicleControl(
            throttle=0.0, brake=1.0,
            steer=float(np.clip(command.steer_rad / limit, -1.0, 1.0)),
            hand_brake=False))
        return
    ackermann = session.carla.VehicleAckermannControl(
        steer=float(command.steer_rad),
        steer_speed=float(ctrl_cfg.get("steer_speed_rad_s", 1.0)),
        speed=float(command.v_ref_mps),
        acceleration=float(command.accel_mps2),
        jerk=float(ctrl_cfg.get("jerk_mps3", 3.0)))
    ego.apply_ackermann_control(ackermann)


def run_continuous(cfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    carla_cfg = dict(cfg.get("carla") or {})
    route_cfg = dict(cfg.get("route") or {})
    planner_cfg = dict(cfg.get("planner") or {})
    ctrl_cfg = dict(cfg.get("controller") or {})
    if int(planner_cfg.get("reverse_steps", 4)) != 4:
        raise ValueError("this entry point only permits planner.reverse_steps: 4")
    dt = float(carla_cfg.get("fixed_dt", 0.05))
    session = CarlaSession(
        host=str(carla_cfg.get("host", "127.0.0.1")),
        port=int(carla_cfg.get("port", 2000)),
        town=str(carla_cfg.get("town", "Town03_Opt")), fixed_dt=dt,
        vendor_dir=str(carla_cfg.get("python_vendor", "")),
        timeout_s=float(carla_cfg.get("timeout_s", 120.0)),
        reload_world=bool(carla_cfg.get("reload_world", True)))
    actors: List[Any] = []
    rolling = None
    recorder = None
    video_overlay = None
    history: List[Dict[str, Any]] = []
    plan_log: List[Dict[str, Any]] = []
    collisions: List[Dict[str, Any]] = []
    ground_contacts: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {"status": "initializing", "reverse_steps": 4}
    try:
        session.connect()
        carla_map = session.world.get_map()
        start_tf, goal_tf, start_idx, goal_idx = _pick_transforms(carla_map,
                                                                  route_cfg)
        route = _global_route(session, start_tf, goal_tf,
                              float(route_cfg.get("sampling_resolution_m", 2.0)),
                              str(carla_cfg.get("python_agents", "")))
        manifest["route"] = {"start_spawn_index": start_idx,
                             "goal_spawn_index": goal_idx,
                             "length_m": route.total_length,
                             "points": len(route.xy)}
        ego, spawn_tf = spawn_ego_at_transform(
            session, start_tf,
            blueprint=str(carla_cfg.get("ego_blueprint", "vehicle.audi.tt")),
            spawn_z_offset=float(carla_cfg.get("spawn_z_offset_m", 0.30)))
        actors.append(ego)
        collision_bp = session.world.get_blueprint_library().find(
            "sensor.other.collision")
        collision_sensor = session.world.spawn_actor(
            collision_bp, session.carla.Transform(), attach_to=ego)
        actors.append(collision_sensor)
        def _on_collision(event):
            other = (str(event.other_actor.type_id)
                     if event.other_actor else "unknown")
            impulse = math.sqrt(event.normal_impulse.x ** 2
                                + event.normal_impulse.y ** 2
                                + event.normal_impulse.z ** 2)
            row = {"frame": int(event.frame), "other": other,
                   "impulse": float(impulse)}
            if other == "static.ground":
                ground_contacts.append(row)
            else:
                collisions.append(row)
        collision_sensor.listen(_on_collision)

        rasterizer = CarlaMapRasterizer.from_carla_map(
            carla_map, float(planner_cfg.get("lane_sample_spacing_m", 1.5)))
        rolling = FourStepRollingPlanner(rasterizer, route, planner_cfg,
                                         ctrl_cfg)
        start_xy = np.array([spawn_tf.location.x, spawn_tf.location.y])
        obstacles = vehicle_obstacle_polygons(
            session.world, exclude_actor_id=ego.id,
            margin_m=float(planner_cfg.get("dynamic_obstacle_margin_m", 0.3)))
        print("[rolling] initial four-step plan...")
        active = rolling.plan_blocking(start_xy, 0.0, obstacles)
        if not active.acceptance["ok"]:
            raise RuntimeError("initial plan rejected: %s | checks=%s | route=%s | attempts=%s"
                               % (active.acceptance.get("failed"),
                                  active.acceptance.get("checks"),
                                  active.acceptance.get("route"),
                                  active.result.quality.get(
                                      "seed_attempt_diagnostics")))
        plan_log.append(_plan_record(active, "activated"))
        video_cfg = dict(cfg.get("video") or {})
        if bool(video_cfg.get("enabled", True)):
            from .overlay import BirdEyeProjector
            from .rolling_overlay import RollingOverlay

            projector = BirdEyeProjector(
                int(video_cfg.get("width", 1024)),
                int(video_cfg.get("height", 1024)),
                float(video_cfg.get("fov", 90.0)),
                float(video_cfg.get("camera_height_m", 45.0)))
            video_overlay = RollingOverlay(projector, route.xy)
            video_overlay.set_active(active)
            video_path = os.path.join(out_dir, str(video_cfg.get(
                "filename", "trajsafe_continuous.mp4")))
            recorder = CameraRecorder(
                session, ego, video_path,
                width=int(video_cfg.get("width", 1024)),
                height=int(video_cfg.get("height", 1024)),
                fps=int(video_cfg.get("fps", 10)),
                height_m=float(video_cfg.get("camera_height_m", 45.0)),
                fov=float(video_cfg.get("fov", 90.0)), enabled=True,
                world_up=True, overlay=video_overlay)
            manifest["video"] = {"path": video_path,
                                 "visible_width_m": projector.visible_width_m()}
        controller = PurePursuitPID(
            ctrl_cfg, steer_sign=float(ctrl_cfg.get("steer_sign", 1.0) or 1.0))
        pending_handoff_s = None
        ready = None
        route_progress = active.route_start_s
        replans = 0
        rejected = 0
        status = "running"
        executed = []
        ego.apply_control(session.carla.VehicleControl(hand_brake=False))
        for _ in range(5):
            session.world.tick()

        max_frames = int(float(carla_cfg.get("max_sim_seconds", 300.0)) / dt)
        for frame_number in range(max_frames):
            frame_wall_start = time.perf_counter()
            if recorder is not None:
                recorder.follow()
            session.world.tick()
            transform = ego.get_transform()
            velocity = ego.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
            xy = np.array([transform.location.x, transform.location.y])
            command = controller.command(active.profile, xy,
                                         float(transform.rotation.yaw), speed, dt)
            route_progress = max(route_progress, route.project(xy, route_progress - 5.0))

            if rolling.future is not None and rolling.future.done():
                try:
                    candidate = rolling.poll()
                    if not candidate.acceptance["ok"]:
                        rejected += 1
                        plan_log.append(_plan_record(candidate, "rejected"))
                        pending_handoff_s = None
                        if video_overlay is not None:
                            video_overlay.clear_candidate()
                    elif pending_handoff_s is not None \
                            and command.s_m > pending_handoff_s + float(
                                planner_cfg.get("passed_handoff_tolerance_m", 2.0)):
                        rejected += 1
                        plan_log.append(_plan_record(candidate, "stale"))
                        pending_handoff_s = None
                        if video_overlay is not None:
                            video_overlay.clear_candidate()
                    else:
                        old_heading = active.profile.sample_at(
                            float(pending_handoff_s))[1]
                        heading_jump = abs(wrap_to_pi(
                            float(candidate.profile.heading[0]) - old_heading))
                        if heading_jump > math.radians(float(
                                planner_cfg.get("max_handoff_heading_deg", 12.0))):
                            rejected += 1
                            plan_log.append(_plan_record(candidate,
                                                         "heading_rejected"))
                            pending_handoff_s = None
                            if video_overlay is not None:
                                video_overlay.clear_candidate()
                        else:
                            ready = candidate
                            plan_log.append(_plan_record(candidate, "ready"))
                            if video_overlay is not None:
                                video_overlay.set_ready(candidate)
                except Exception as exc:
                    rejected += 1
                    pending_handoff_s = None
                    plan_log.append({"status": "failed", "error": str(exc)})
                    if video_overlay is not None:
                        video_overlay.clear_candidate()

            if ready is not None and pending_handoff_s is not None \
                    and command.s_m >= pending_handoff_s - float(
                        planner_cfg.get("switch_lead_m", 0.8)):
                active = ready
                ready = None
                pending_handoff_s = None
                controller.reset()
                replans += 1
                plan_log.append(_plan_record(active, "activated"))
                if video_overlay is not None:
                    video_overlay.set_active(active)
                command = controller.command(active.profile, xy,
                                             float(transform.rotation.yaw),
                                             speed, dt)

            global_remaining = route.total_length - route_progress
            trigger = (command.s_m >= float(planner_cfg.get("replan_every_m", 25.0))
                       or command.remaining_m <= float(
                           planner_cfg.get("replan_remaining_m", 45.0)))
            if trigger and not rolling.pending() and ready is None \
                    and global_remaining > float(planner_cfg.get("final_radius_m", 3.0)) \
                    and active.route_goal_s < route.total_length - 1e-3:
                handoff_s = min(command.s_m + float(
                    planner_cfg.get("handoff_ahead_m", 15.0)),
                    active.profile.total_length - 2.0)
                if handoff_s > command.s_m + 2.0:
                    handoff, _heading, _curvature, _speed = \
                        active.profile.sample_at(handoff_s)
                    obstacles = vehicle_obstacle_polygons(
                        session.world, exclude_actor_id=ego.id,
                        margin_m=float(planner_cfg.get(
                            "dynamic_obstacle_margin_m", 0.3)))
                    pending_id = rolling.submit(handoff, route_progress, obstacles)
                    pending_handoff_s = handoff_s
                    if video_overlay is not None:
                        video_overlay.set_pending(pending_id, handoff)

            _apply_command(session, ego, command, ctrl_cfg)
            history.append({
                "frame": frame_number, "t": frame_number * dt,
                "x": xy[0], "y": xy[1], "yaw_deg": transform.rotation.yaw,
                "speed_mps": speed, "plan_id": active.plan_id,
                "plan_s_m": command.s_m, "plan_remaining_m": command.remaining_m,
                "route_s_m": route_progress, "route_remaining_m": global_remaining,
                "cross_track_m": command.cross_track_m,
                "pending": rolling.pending(), "ready": ready is not None})
            executed.append(xy.copy())
            if recorder is not None:
                recorder.executed_world = np.asarray(executed, dtype=np.float64)
                state = ("READY %s" % ready.plan_id if ready is not None
                         else ("PLANNING %s" % video_overlay.pending_id
                               if rolling.pending() and video_overlay is not None
                               else "TRACKING"))
                recorder.hud_lines = [
                    "active plan #%d | %s | four-step DDIM" %
                    (active.plan_id, state),
                    "route %.1f / %.1f m | local %.1f / %.1f m" %
                    (route_progress, route.total_length, command.s_m,
                     active.profile.total_length),
                    "speed %.2f m/s | cross-track %+.2f m | replans %d" %
                    (speed, command.cross_track_m, replans),
                    "ALM %s | rejected %d | collisions %d" %
                    (active.result.alm_status, rejected, len(collisions)),
                ]
                recorder.write_pending()
            if collisions:
                status = "collision"
                break
            if abs(command.cross_track_m) > float(ctrl_cfg.get(
                    "tracking_abort_m", 2.0)):
                status = "lateral_abort"
                break
            if global_remaining <= float(planner_cfg.get("final_radius_m", 3.0)) \
                    and command.stop:
                status = "arrived"
                break
            if command.stop and not rolling.pending() and ready is None:
                status = "controlled_stop_no_plan"
                break
            if bool(carla_cfg.get("real_time_pacing", True)):
                remaining_wall = dt - (time.perf_counter() - frame_wall_start)
                if remaining_wall > 0.0:
                    time.sleep(remaining_wall)
        else:
            status = "timeout"

        ego.apply_control(session.carla.VehicleControl(hand_brake=True))
        for _ in range(5):
            session.world.tick()
            if recorder is not None:
                recorder.write_pending()
        manifest.update({"status": status, "frames": len(history),
                         "sim_seconds": len(history) * dt,
                         "replans_activated": replans,
                         "plans_rejected": rejected,
                         "collisions": collisions,
                         "ground_contacts": ground_contacts,
                         "plans": plan_log,
                         "route_progress_m": route_progress,
                         "route_remaining_m": route.total_length - route_progress})
        if recorder is not None:
            manifest["video"].update({"frames": recorder.frames,
                                      "errors": list(recorder.errors)})
        os.makedirs(out_dir, exist_ok=True)
        if history:
            csv_path = os.path.join(out_dir, "continuous_trajectory.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            manifest["trajectory_csv"] = csv_path
        return manifest
    finally:
        if recorder is not None:
            recorder.close()
        if rolling is not None:
            rolling.close()
        destroy_actors(actors)
        session.close()


def _plan_record(plan, status: str) -> Dict[str, Any]:
    return {"plan_id": int(plan.plan_id), "status": status,
            "planning_ms": float(plan.planning_ms), "reverse_steps": 4,
            "route_start_s": float(plan.route_start_s),
            "route_goal_s": float(plan.route_goal_s),
            "length_m": float(plan.profile.total_length),
            "route_mean_error_m": float(plan.route_mean_error_m),
            "route_max_error_m": float(plan.route_max_error_m),
            "diffusion_seed": int(plan.result.quality.get("diffusion_seed", -1)),
            "seed_attempts_run": int(plan.result.quality.get("seed_attempts_run", 1)),
            "guided": bool(plan.result.guided),
            "alm_status": str(plan.result.alm_status),
            "acceptance": plan.acceptance,
            "frame": plan.frame.describe()}


def save_manifest(manifest: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
