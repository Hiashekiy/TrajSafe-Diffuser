"""Coordinate frames for the CARLA <-> TrajSafe-Diffuser bridge.

Only THREE frames exist in this package, and every conversion between them is
implemented here once.  No call site is allowed to hand-roll a yaw rotation or
a vertical flip.

world   CARLA world XY in metres (z is carried separately).
local   the ego-anchored metric frame of one dataset sample:

            x_local = (world_xy - anchor_xy) . forward_xy     metres ahead
            y_local = (world_xy - anchor_xy) . right_xy       metres right

        forward_xy / right_xy are the RECORDED ego basis of the anchor frame
        (data/carla_v1/episodes/<Town>/episode_XXXXXX/ego_frame_vectors.npy),
        never a yaw computed by hand.
scene   the model frame, [-1, 1]^2 over the fixed 80 m x 80 m window:

            x_scene = 2 * (x_local + 10) / 80 - 1
            y_scene = y_local / 40

grid    the canonical 256 x 256 image the network actually consumes:

            col = (x_scene + 1) * 0.5 * (res - 1)
            row = (y_scene + 1) * 0.5 * (res - 1)

Facts verified against the shipped data (see src/carla_bridge/frame.py __main__):

  * data/carla_processed/test/occupancy.npy is ALREADY canonical: it equals
    np.flipud(rasterize_local_occupancy(...)) of the raw CARLA lane
    rasterisation (99.33 percent of the 65536 cells agree; the remainder is the
    dataset cleaning margin).  A raw rasterisation therefore needs exactly one
    flipud, which is what raw_occupancy_to_canonical() does.
  * the dataset crop is ego-forward-biased: x_local in [-10, 70] m and
    y_local in [-40, 40] m, so scene x is NOT in [-1, 1] for every local point
    (a point 70 m ahead has x_scene = 1 while y_scene stays free).
  * trajectory_raw.npy column 4 (named yaw) is in DEGREES; forward_xy from
    ego_frame_vectors.npy equals (cos(yaw), sin(yaw)) to 1e-6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: dataset crop limits, metres (tools/carla/map_cache.py)
LOCAL_X_MIN = -10.0
LOCAL_X_MAX = 70.0
LOCAL_Y_MIN = -40.0
LOCAL_Y_MAX = 40.0
#: scene <-> local scale: 1 scene unit = 40 m in both axes
SCENE_SCALE_M = 40.0
DEFAULT_RES = 256

__all__ = [
    "LOCAL_X_MIN", "LOCAL_X_MAX", "LOCAL_Y_MIN", "LOCAL_Y_MAX",
    "SCENE_SCALE_M", "DEFAULT_RES", "LocalFrame", "raw_occupancy_to_canonical",
    "wrap_to_pi", "forward_right_from_yaw", "yaw_from_forward",
]


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi)."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def forward_right_from_yaw(yaw_deg: float) -> tuple:
    """CARLA yaw (degrees) -> unit forward / right vectors in world XY.

    CARLA is left-handed with +x forward and +y right at yaw = 0, so a positive
    yaw rotates forward towards right:

        forward = (cos y, sin y)      right = (-sin y, cos y)

    Checked against ego_frame_vectors.npy of episode 70 (agreement 1e-6).
    """
    yaw = math.radians(float(yaw_deg))
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return forward, right


def yaw_from_forward(forward_xy: Sequence) -> float:
    """Unit forward XY -> yaw in DEGREES (the trajectory_raw unit)."""
    fx, fy = float(forward_xy[0]), float(forward_xy[1])
    return math.degrees(math.atan2(fy, fx))


def raw_occupancy_to_canonical(raw: np.ndarray) -> np.ndarray:
    """Flip a raw CARLA lane rasterisation into the canonical model image.

    The raw rasteriser (tools/carla/map_cache.local_to_pixel) puts
    y_local = +40 m on row 0; the canonical image puts y_scene = -1 on row 0.
    Exactly one flipud is required, and this is the only place that does it.
    """
    arr = np.asarray(raw)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("occupancy must be a square 2D array, got %r" % (arr.shape,))
    return np.ascontiguousarray(np.flipud(arr))


@dataclass(frozen=True)
class LocalFrame:
    """Ego-anchored metric frame of one dataset sample (see module docstring)."""

    anchor_xy: np.ndarray
    forward_xy: np.ndarray
    right_xy: np.ndarray
    z: float = 0.0
    yaw_deg: float = 0.0

    # ------------------------------------------------------------- building
    @classmethod
    def from_anchor(
        cls,
        anchor_xy: Sequence,
        forward_xy: Sequence,
        right_xy: Sequence,
        z: float = 0.0,
        yaw_deg: float = None,
    ) -> "LocalFrame":
        anchor = np.asarray(anchor_xy, dtype=np.float64).reshape(2)
        forward = np.asarray(forward_xy, dtype=np.float64).reshape(2)
        right = np.asarray(right_xy, dtype=np.float64).reshape(2)
        for name, vec in (("forward_xy", forward), ("right_xy", right)):
            norm = float(np.linalg.norm(vec))
            if abs(norm - 1.0) > 1e-3:
                raise ValueError("%s must be a unit vector, |v| = %.6f" % (name, norm))
        if abs(float(forward @ right)) > 1e-3:
            raise ValueError("forward_xy and right_xy must be orthogonal")
        yaw = yaw_from_forward(forward) if yaw_deg is None else float(yaw_deg)
        return cls(anchor_xy=anchor, forward_xy=forward, right_xy=right,
                   z=float(z), yaw_deg=yaw)

    @classmethod
    def from_episode(cls, episode_dir, index: int) -> "LocalFrame":
        """Build the frame of one raw trajectory index of one episode.

        episode_dir holds trajectory_raw.npy [N,13] and ego_frame_vectors.npy
        [N,4] = [forward_x, forward_y, right_x, right_y] (dataset schema).
        """
        import os

        raw = np.load(os.path.join(str(episode_dir), "trajectory_raw.npy"))
        frames = np.load(os.path.join(str(episode_dir), "ego_frame_vectors.npy"))
        i = int(index)
        if not 0 <= i < len(raw):
            raise IndexError("index %d out of range for %d raw frames" % (i, len(raw)))
        row = raw[i]
        fr = frames[i]
        return cls.from_anchor(anchor_xy=(float(row[1]), float(row[2])),
                               forward_xy=(float(fr[0]), float(fr[1])),
                               right_xy=(float(fr[2]), float(fr[3])),
                               z=float(row[3]), yaw_deg=float(row[4]))

    # ------------------------------------------------------- world <-> local
    def to_local(self, xy_world) -> np.ndarray:
        arr = np.asarray(xy_world, dtype=np.float64)
        delta = arr - self.anchor_xy
        return np.stack((delta @ self.forward_xy, delta @ self.right_xy), axis=-1)

    def to_world(self, xy_local) -> np.ndarray:
        arr = np.asarray(xy_local, dtype=np.float64)
        return (self.anchor_xy
                + arr[..., 0:1] * self.forward_xy
                + arr[..., 1:2] * self.right_xy)

    # ------------------------------------------------------- local <-> scene
    @staticmethod
    def local_to_scene(xy_local) -> np.ndarray:
        arr = np.asarray(xy_local, dtype=np.float64)
        out = np.empty_like(arr)
        out[..., 0] = 2.0 * (arr[..., 0] - LOCAL_X_MIN) / (LOCAL_X_MAX - LOCAL_X_MIN) - 1.0
        out[..., 1] = arr[..., 1] / SCENE_SCALE_M
        return out

    @staticmethod
    def scene_to_local(xy_scene) -> np.ndarray:
        arr = np.asarray(xy_scene, dtype=np.float64)
        out = np.empty_like(arr)
        out[..., 0] = (arr[..., 0] + 1.0) * 0.5 * (LOCAL_X_MAX - LOCAL_X_MIN) + LOCAL_X_MIN
        out[..., 1] = arr[..., 1] * SCENE_SCALE_M
        return out

    # ------------------------------------------------------- world <-> scene
    def to_scene(self, xy_world) -> np.ndarray:
        return self.local_to_scene(self.to_local(xy_world))

    def world_from_scene(self, xy_scene) -> np.ndarray:
        return self.to_world(self.scene_to_local(xy_scene))

    # -------------------------------------------------------- scene <-> grid
    @staticmethod
    def scene_to_grid(xy_scene, res: int = DEFAULT_RES) -> np.ndarray:
        arr = np.asarray(xy_scene, dtype=np.float64)
        scale = 0.5 * (int(res) - 1)
        out = np.empty_like(arr)
        out[..., 0] = (arr[..., 0] + 1.0) * scale
        out[..., 1] = (arr[..., 1] + 1.0) * scale
        return out

    @staticmethod
    def grid_to_scene(xy_grid, res: int = DEFAULT_RES) -> np.ndarray:
        arr = np.asarray(xy_grid, dtype=np.float64)
        scale = 0.5 * (int(res) - 1)
        out = np.empty_like(arr)
        out[..., 0] = arr[..., 0] / scale - 1.0
        out[..., 1] = arr[..., 1] / scale - 1.0
        return out

    # ------------------------------------------------------------- scenario
    def in_window(self, xy_local, margin_m: float = 0.0) -> bool:
        arr = np.asarray(xy_local, dtype=np.float64)
        return bool(
            np.all(arr[..., 0] >= LOCAL_X_MIN + margin_m)
            and np.all(arr[..., 0] <= LOCAL_X_MAX - margin_m)
            and np.all(arr[..., 1] >= LOCAL_Y_MIN + margin_m)
            and np.all(arr[..., 1] <= LOCAL_Y_MAX - margin_m)
        )

    def describe(self) -> str:
        return ("anchor=(%.3f, %.3f) yaw=%.4f deg forward=(%.6f, %.6f) "
                "right=(%.6f, %.6f) z=%.3f"
                % (self.anchor_xy[0], self.anchor_xy[1], self.yaw_deg,
                   self.forward_xy[0], self.forward_xy[1],
                   self.right_xy[0], self.right_xy[1], self.z))


def _self_test() -> None:
    frame = LocalFrame.from_anchor((132.085, 62.487), (0.9999966, -0.00259818),
                                   (0.00259818, 0.9999966), z=1.818, yaw_deg=-0.1489)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-60.0, 60.0, size=(512, 2)) + np.array([132.085, 62.487])
    back = frame.to_world(frame.to_local(pts))
    err = float(np.abs(back - pts).max())
    assert err < 1e-4, err
    back2 = frame.world_from_scene(frame.to_scene(pts))
    err2 = float(np.abs(back2 - pts).max())
    assert err2 < 1e-4, err2
    corners_scene = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    corners_grid = LocalFrame.scene_to_grid(corners_scene)
    expect = np.array([[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0]])
    assert np.allclose(corners_grid, expect), corners_grid
    start_local = LocalFrame.scene_to_local(np.array([-0.75, 0.0]))
    assert np.allclose(start_local, [0.0, 0.0], atol=1e-6), start_local
    right_world = frame.to_world(np.array([[0.0, 10.0]]))[0]
    lat = float((right_world - frame.anchor_xy) @ frame.right_xy)
    assert lat > 9.99, lat
    # goal of the shipped sample: 37.64 m ahead, 21.35 m to the right
    goal_local = LocalFrame.scene_to_local(np.array([0.19110211730003357,
                                                     0.5338726043701172]))
    assert abs(goal_local[0] - 37.644) < 1e-3, goal_local
    assert abs(goal_local[1] - 21.355) < 1e-3, goal_local
    print("frame self-test OK (round-trip max err %.3e m, goal local %s)"
          % (max(err, err2), np.round(goal_local, 4)))


if __name__ == "__main__":
    _self_test()
