"""CARLA side of the demo: connection, scene reproduction, sensors, probing.

Scene reproduction for sample test_0056
---------------------------------------
The sample comes from episode 70 of the shipped CARLA v1 dataset:
town Town03, raw trajectory index 104 (data/carla_v1/samples.jsonl,
sample_id 1252).  The dataset was collected with NO NPCs -- the only obstacles
in the occupancy are non-driving-lane cells (sidewalk, buildings, the opposite
carriageway), which already exist in the Town03 map.  Restoring the scene is
therefore exactly:

  1. load Town03,
  2. put the ego on the recorded anchor pose (x, y, yaw) with the recorded
   2D basis as the local frame,
  3. drive.

Anything that claims to be an "obstacle" beyond step 2 would be invented, so
nothing is invented here.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .frame import LocalFrame, forward_right_from_yaw

__all__ = ["CarlaSession", "spawn_ego", "spawn_ego_at_transform",
           "probe_steer_sign", "CameraRecorder"]


def _import_carla(vendor_dir: str):
    if vendor_dir and vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    import carla

    return carla


@dataclass
class CarlaSession:
    """Connected client + world in synchronous mode."""

    host: str = "127.0.0.1"
    port: int = 2000
    town: str = "Town03"
    fixed_dt: float = 0.05
    vendor_dir: str = "E:/CarDataSample/tools/carla/_vendor"
    timeout_s: float = 60.0
    quality: str = "Low"
    reload_world: bool = True
    carla: Any = None
    client: Any = None
    world: Any = None
    original_settings: Any = None

    def connect(self) -> "CarlaSession":
        self.carla = _import_carla(self.vendor_dir)
        self.client = self.carla.Client(self.host, int(self.port))
        self.client.set_timeout(float(self.timeout_s))
        version = self.client.get_server_version()
        world = self.client.get_world()
        current = world.get_map().name.rsplit("/", 1)[-1]
        if current != self.town:
            world = self.client.load_world(self.town)
        elif self.reload_world:
            # a long-lived server drifts (leftover synchronous state makes
            # try_spawn_actor start failing); a reload guarantees the same
            # clean world every run, which the reproducibility clause needs.
            world = self.client.reload_world()
        self.world = world
        self.original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(self.fixed_dt)
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        world.set_weather(self.carla.WeatherParameters.ClearNoon)
        print("[carla] server %s | map %s | fixed_dt=%.3f"
              % (version, world.get_map().name, self.fixed_dt))
        return self

    # ------------------------------------------------------------- lifecycle
    def tick(self):
        return self.world.tick()

    def close(self, restore: bool = True) -> None:
        try:
            if restore and self.original_settings is not None and self.world is not None:
                self.world.apply_settings(self.original_settings)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print("[carla] restore settings failed: %s" % exc)

    def ground_z(self, x: float, y: float, z_hint: float = 0.0) -> float:
        carla = self.carla
        waypoint = self.world.get_map().get_waypoint(
            carla.Location(x=float(x), y=float(y), z=float(z_hint)))
        if waypoint is None:
            raise RuntimeError("no waypoint at (%.3f, %.3f)" % (x, y))
        return float(waypoint.transform.location.z), waypoint


def spawn_ego(session: CarlaSession, frame: LocalFrame, blueprint: str = "vehicle.audi.tt",
              spawn_z_offset: float = 0.30, role_name: str = "trajSafeEgo"):
    """Spawn the ego exactly on the recorded anchor pose."""
    carla = session.carla
    library = session.world.get_blueprint_library()
    bp = library.find(blueprint)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    z_ground, waypoint = session.ground_z(frame.anchor_xy[0], frame.anchor_xy[1], frame.z)
    transform = carla.Transform(
        carla.Location(x=float(frame.anchor_xy[0]), y=float(frame.anchor_xy[1]),
                       z=float(z_ground) + float(spawn_z_offset)),
        carla.Rotation(yaw=float(frame.yaw_deg)))
    vehicle = session.world.try_spawn_actor(bp, transform)
    if vehicle is None:
        raise RuntimeError("failed to spawn %s at %s" % (blueprint, transform))
    vehicle.set_autopilot(False)
    vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    bbox = vehicle.bounding_box
    print("[carla] ego spawned at (%.3f, %.3f, %.3f) yaw=%.4f (lane z=%.3f) bbox=%s"
          % (transform.location.x, transform.location.y, transform.location.z,
             transform.rotation.yaw, z_ground, bbox.extent))
    return vehicle, transform


def spawn_ego_at_transform(session: CarlaSession, source_transform,
                           blueprint: str = "vehicle.audi.tt",
                           spawn_z_offset: float = 0.30,
                           role_name: str = "trajSafeEgo"):
    """Spawn an ego on an arbitrary CARLA waypoint/spawn transform."""
    carla = session.carla
    bp = session.world.get_blueprint_library().find(blueprint)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    location = source_transform.location
    z_ground, waypoint = session.ground_z(location.x, location.y, location.z)
    yaw = float(source_transform.rotation.yaw)
    if waypoint is not None:
        yaw = float(waypoint.transform.rotation.yaw)
    transform = carla.Transform(
        carla.Location(x=float(location.x), y=float(location.y),
                       z=float(z_ground) + float(spawn_z_offset)),
        carla.Rotation(yaw=yaw))
    vehicle = session.world.try_spawn_actor(bp, transform)
    if vehicle is None:
        raise RuntimeError("failed to spawn %s at %s" % (blueprint, transform))
    vehicle.set_autopilot(False)
    vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    return vehicle, transform


def spawn_static_vehicle(session: CarlaSession, blueprint: str, xy_world,
                         yaw_deg: float, spawn_z_offset: float = 0.30,
                         role_name: str = "trajSafeObstacle"):
    """Park a vehicle (hand brake on, never moved again) at a world pose."""
    carla = session.carla
    library = session.world.get_blueprint_library()
    bp = library.find(blueprint)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    if bp.has_attribute("color"):
        bp.set_attribute("color", "40,40,40")
    z_ground, _waypoint = session.ground_z(float(xy_world[0]), float(xy_world[1]), 0.0)
    transform = carla.Transform(
        carla.Location(x=float(xy_world[0]), y=float(xy_world[1]),
                       z=float(z_ground) + float(spawn_z_offset)),
        carla.Rotation(yaw=float(yaw_deg)))
    actor = session.world.try_spawn_actor(bp, transform)
    if actor is None:
        raise RuntimeError("failed to spawn obstacle %s at %s" % (blueprint, transform))
    actor.apply_control(carla.VehicleControl(hand_brake=True))
    extent = actor.bounding_box.extent
    print("[carla] obstacle %s parked at (%.3f, %.3f) yaw=%.2f extent=(%.2f, %.2f)"
          % (blueprint, transform.location.x, transform.location.y,
             transform.rotation.yaw, extent.x, extent.y))
    return actor


def destroy_actors(actors: List[Any]) -> None:
    for actor in actors:
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except Exception:
            pass


def probe_steer_sign(session: CarlaSession, vehicle: Any, magnitude: float = 0.20,
                     speed: float = 1.5, ticks: int = 30,
                     throttle: float = 0.55) -> Dict[str, Any]:
    """Measure which sign of VehicleAckermannControl.steer turns RIGHT.

    Returns steer_sign: multiply a vehicle-frame steering angle (positive =
    turn right) by it to get the CARLA command value.
    """
    carla = session.carla
    # NOTE: measured on CARLA 0.9.16, VehicleAckermannControl from a standstill
    # keeps the car parked (it reports throttle but never moves), so the probe
    # uses plain throttle + steer, which is the same steering convention.
    vehicle.apply_control(carla.VehicleControl(hand_brake=False))
    for _ in range(5):
        session.world.tick()
    start = vehicle.get_transform()
    yaw0 = float(start.rotation.yaw)
    p0 = np.array([start.location.x, start.location.y], dtype=np.float64)
    for _ in range(int(ticks)):
        vehicle.apply_control(carla.VehicleControl(
            throttle=float(throttle), steer=float(magnitude),
            brake=0.0, hand_brake=False))
        session.world.tick()
    end = vehicle.get_transform()
    p1 = np.array([end.location.x, end.location.y], dtype=np.float64)
    _forward, right = forward_right_from_yaw(yaw0)
    lateral = float((p1 - p0) @ right)
    travelled = float(np.linalg.norm(p1 - p0))
    sign = 1.0 if lateral > 0.0 else -1.0
    info = {"steer_sign": sign, "usable": bool(travelled > 0.5),
            "probe_throttle": float(throttle),
            "lateral_m": lateral, "travelled_m": travelled,
            "probe_steer_rad": float(magnitude), "yaw_start_deg": yaw0,
            "yaw_end_deg": float(end.rotation.yaw)}
    print("[carla] steer probe: throttle %.2f steer=+%.3f -> lateral %+.3f m "
          "over %.3f m -> steer_sign=%+.0f%s" % (throttle, magnitude, lateral, travelled, sign,
                                  "" if travelled > 0.5 else "  (UNRELIABLE: car did not move)"))
    return info


class CameraRecorder:
    """Top-down RGB camera that writes an MP4 while the demo runs."""

    def __init__(self, session: CarlaSession, vehicle: Any, path: str,
                 width: int = 1280, height: int = 720, fps: int = 10,
                 height_m: float = 30.0, fov: float = 60.0, enabled: bool = True,
                 world_up: bool = True, fixed_center_world=None,
                 overlay=None, overlay_options=None):
        self.session = session
        self.path = str(path)
        self.enabled = bool(enabled)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.height_m = float(height_m)
        self.fov = float(fov)
        self.world_up = bool(world_up)
        self.vehicle = None
        self._carla = None
        self.fixed_center = (None if fixed_center_world is None
                             else np.asarray(fixed_center_world, dtype=np.float64))
        self.overlay = overlay
        self.overlay_options = dict(overlay_options or {})
        # fed by the driver every tick so the camera frame can carry the trace
        self.executed_world = None
        self.hud_lines = []
        self.camera = None
        self.writer = None
        self.frames = 0
        self.errors: List[str] = []
        self._pending: List[np.ndarray] = []
        if not self.enabled:
            return
        try:
            import cv2

            self.cv2 = cv2
        except Exception as exc:
            self.errors.append("cv2 unavailable: %s" % exc)
            self.enabled = False
            return
        carla = session.carla
        blueprint = session.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self.width))
        blueprint.set_attribute("image_size_y", str(self.height))
        blueprint.set_attribute("fov", str(self.fov))
        blueprint.set_attribute("sensor_tick", str(1.0 / max(self.fps, 1)))
        self.vehicle = vehicle
        self._carla = carla
        if self.fixed_center is not None:
            # one static bird's-eye camera for the whole run: the complete
            # junction stays in frame and the projection never moves
            transform = carla.Transform(
                carla.Location(x=float(self.fixed_center[0]),
                               y=float(self.fixed_center[1]),
                               z=self.height_m),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
            self.camera = session.world.spawn_actor(blueprint, transform)
        elif self.world_up:
            # fixed world orientation (north-up map view); follow() re-centres
            # the camera on the ego before every tick
            pose = vehicle.get_transform().location
            transform = carla.Transform(
                carla.Location(x=pose.x, y=pose.y, z=self.height_m),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
            self.camera = session.world.spawn_actor(blueprint, transform)
        else:
            transform = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=self.height_m),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
            self.camera = session.world.spawn_actor(blueprint, transform,
                                                    attach_to=vehicle)
        self.camera.listen(self._on_image)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = self.cv2.VideoWriter(self.path, fourcc, float(self.fps),
                                           (self.width, self.height))

    def _on_image(self, image) -> None:
        if not self.enabled:
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self._pending.append(array.copy())
        if len(self._pending) > 8:
            self._pending.pop(0)

    def camera_xy(self):
        """World XY the current frame was rendered from."""
        if self.fixed_center is not None:
            return self.fixed_center
        pose = self.vehicle.get_transform().location
        return np.array([pose.x, pose.y], dtype=np.float64)

    def follow(self) -> None:
        """Re-centre a world-up camera on the ego (call once per tick)."""
        if (not self.enabled or self.camera is None or not self.world_up
                or self.fixed_center is not None):
            return
        pose = self.vehicle.get_transform()
        self.camera.set_transform(self._carla.Transform(
            self._carla.Location(x=pose.location.x, y=pose.location.y,
                                 z=self.height_m),
            self._carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))

    def write_pending(self) -> None:
        if not self.enabled or self.writer is None:
            return
        while self._pending:
            frame = self._pending.pop(0)
            if self.overlay is not None:
                try:
                    frame = self.overlay.draw(frame, self.camera_xy(),
                                              self.executed_world,
                                              self.hud_lines,
                                              **self.overlay_options)
                except Exception as exc:  # pragma: no cover - drawing is optional
                    if not self.errors:
                        self.errors.append("overlay: %s" % exc)
                    self.overlay = None
            self.writer.write(np.ascontiguousarray(frame))
            self.frames += 1

    def close(self) -> None:
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.camera.destroy()
            except Exception:
                pass
        if self.writer is not None:
            self.writer.release()
