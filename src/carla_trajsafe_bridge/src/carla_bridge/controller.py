"""Pure Pursuit (lateral) + PID (longitudinal) -> CARLA Ackermann command.

Sign conventions
----------------
The controller works in the CARLA vehicle frame: +x forward, +y right.  A
target to the right gives alpha > 0 and therefore delta > 0, meaning "turn
right".  Whether CARLA AckermannControl wants a positive steer for a right turn
is measured at run time (see scenario.probe_steer_sign) and applied through
controller.steer_sign -- never hard-coded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .frame import forward_right_from_yaw, wrap_to_pi
from .path_profile import PathProfile

__all__ = ["ControlCommand", "PurePursuitPID"]


@dataclass
class ControlCommand:
    steer_rad: float          # already in CARLA sign convention
    steer_raw_rad: float      # vehicle-frame steering angle
    accel_mps2: float
    v_ref_mps: float
    speed_mps: float
    target_index: int
    target_s_m: float
    s_m: float
    cross_track_m: float
    heading_error_rad: float
    remaining_m: float
    stop: bool = False
    brake_hard: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "steer_rad": self.steer_rad,
            "steer_raw_rad": self.steer_raw_rad,
            "accel_mps2": self.accel_mps2,
            "v_ref_mps": self.v_ref_mps,
            "speed_mps": self.speed_mps,
            "target_index": self.target_index,
            "target_s_m": self.target_s_m,
            "s_m": self.s_m,
            "cross_track_m": self.cross_track_m,
            "heading_error_rad": self.heading_error_rad,
            "remaining_m": self.remaining_m,
            "stop": self.stop,
            "brake_hard": self.brake_hard,
        }


class PurePursuitPID:
    """Discrete-time tracker; call command() once per simulation tick."""

    def __init__(self, cfg: Dict, steer_sign: float = 1.0):
        self.cfg = dict(cfg or {})
        self.wheelbase = float(self.cfg.get("wheelbase_m", 2.8))
        self.steer_sign = float(steer_sign)
        self.lookahead_base = float(self.cfg.get("lookahead_base_m", 1.8))
        self.lookahead_gain = float(self.cfg.get("lookahead_speed_gain", 0.45))
        self.lookahead_min = float(self.cfg.get("lookahead_min_m", 2.0))
        self.lookahead_max = float(self.cfg.get("lookahead_max_m", 4.0))
        self.delta_max = float(self.cfg.get("max_steer_rad",
                                            math.radians(35.0)))
        self.heading_gain = float(self.cfg.get("heading_gain", 0.25))
        pid = self.cfg.get("pid") or {}
        self.kp = float(pid.get("kp", 1.2))
        self.ki = float(pid.get("ki", 0.08))
        self.kd = float(pid.get("kd", 0.05))
        self.accel_min = float(self.cfg.get("accel_min_mps2", -4.0))
        self.accel_max = float(self.cfg.get("accel_max_mps2", 2.5))
        self.stop_radius = float(self.cfg.get("stop_radius_m", 0.6))
        self.stop_speed = float(self.cfg.get("stop_speed_mps", 0.3))
        self.creep_accel = float(self.cfg.get("creep_accel_mps2", 0.9))
        self.tracking_abort_m = float(self.cfg.get("tracking_abort_m", 1.0))
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._index = 0

    # ------------------------------------------------------------- internals
    def _lookahead(self, v: float) -> float:
        return float(np.clip(self.lookahead_base + self.lookahead_gain * max(v, 0.0),
                             self.lookahead_min, self.lookahead_max))

    # --------------------------------------------------------------- command
    def command(self, path: PathProfile, xy_world, yaw_deg: float, v_mps: float,
                dt: float) -> ControlCommand:
        p = np.asarray(xy_world, dtype=np.float64).reshape(2)
        index, s_now, cross = path.project(p, self._index)
        self._index = index

        lookahead = self._lookahead(v_mps)
        target, heading_t, _curv, _v = path.sample_at(s_now + lookahead)
        forward, right = forward_right_from_yaw(yaw_deg)
        d = target - p
        alpha = math.atan2(float(d @ right), float(d @ forward))
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), lookahead)

        path_forward = np.array([math.cos(heading_t), math.sin(heading_t)])
        heading_error = wrap_to_pi(heading_t - math.radians(float(yaw_deg)))
        delta = float(np.clip(delta + self.heading_gain * heading_error,
                              -self.delta_max, self.delta_max))

        remaining = path.total_length - s_now
        _xy_now, _h, _k, v_ref = path.sample_at(s_now)
        error = v_ref - float(v_mps)
        self._integral = float(np.clip(self._integral + error * dt, -5.0, 5.0))
        deriv = 0.0 if self._prev_error is None else (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        accel = self.kp * error + self.ki * self._integral + self.kd * deriv
        accel = float(np.clip(accel, self.accel_min, self.accel_max))
        if v_ref > 0.2 and v_mps < 0.15:
            accel = max(accel, self.creep_accel)

        stop = bool(remaining <= self.stop_radius and v_mps <= self.stop_speed)
        # CARLA's Ackermann speed loop decelerates much more gently than the
        # requested feed-forward acceleration, so the last metre is stopped
        # with an explicit full-brake VehicleControl instead of trusting it.
        brake_hard = bool(remaining <= self.stop_radius)
        if stop or brake_hard:
            accel = self.accel_min
        return ControlCommand(
            steer_rad=self.steer_sign * delta,
            steer_raw_rad=delta,
            accel_mps2=accel,
            v_ref_mps=float(v_ref),
            speed_mps=float(v_mps),
            target_index=int(index),
            target_s_m=float(s_now + lookahead),
            s_m=float(s_now),
            cross_track_m=float(cross),
            heading_error_rad=float(heading_error),
            remaining_m=float(remaining),
            stop=stop,
            brake_hard=brake_hard,
        )
