"""Bird's-eye camera overlay: world geometry drawn onto the recorded frames.

The demo camera is a plain downward-looking RGB camera with a FIXED world
orientation (pitch -90, yaw 0, roll 0), so the projection is a similarity
transform with no perspective distortion left to calibrate:

    u = cx + s * (P.y - C.y)
    v = cy - s * (P.x - C.x)          s = f / h,  f = (W/2) / tan(fov/2)

(camera right = world +Y and camera up = world +X for that rotation, and
CARLA's fov attribute is the horizontal field of view).

Because moving the camera is a pure TRANSLATION in pixel space, the static
geometry is projected once and then only shifted by (-s*C.y, +s*C.x) per frame,
which keeps the per-frame cost to a handful of cv2 calls.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["BirdEyeProjector", "OverlayGeometry", "obstacle_world_corners"]

# BGR colours
CORRIDOR = (60, 180, 60)
ELLIPSE = (255, 140, 0)
CANDIDATE = (120, 120, 120)
GUIDED = (255, 0, 255)
RAW = (0, 170, 255)
EXECUTED = (255, 255, 255)
START = (0, 220, 0)
GOAL = (0, 0, 230)
OBSTACLE = (0, 0, 255)


def obstacle_world_corners(box: Dict[str, Any], frame=None) -> np.ndarray:
    """World XY corners of one scenario box (used for the red rectangles)."""
    center = np.asarray(box["world_xy"], dtype=np.float64)
    yaw = math.radians(float(box["yaw_world_deg"]))
    along = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    half_l = 0.5 * float(box["length_m"])
    half_w = 0.5 * float(box["width_m"])
    return np.stack((center + along * half_l + right * half_w,
                     center + along * half_l - right * half_w,
                     center - along * half_l - right * half_w,
                     center - along * half_l + right * half_w))


class BirdEyeProjector:
    """World XY -> pixel, for a camera looking straight down from height h."""

    def __init__(self, width: int, height: int, fov_deg: float,
                 camera_height_m: float):
        self.width = int(width)
        self.height = int(height)
        self.f = 0.5 * self.width / math.tan(math.radians(float(fov_deg)) * 0.5)
        self.s = float(self.f) / float(camera_height_m)
        self.cx = 0.5 * self.width
        self.cy = 0.5 * self.height

    def base(self, world_xy) -> np.ndarray:
        """Pixel coords assuming the camera sits at world (0, 0)."""
        arr = np.asarray(world_xy, dtype=np.float64)
        u = self.cx + self.s * arr[..., 1]
        v = self.cy - self.s * arr[..., 0]
        return np.stack((u, v), axis=-1)

    def offset(self, cam_xy) -> np.ndarray:
        return np.array([-self.s * float(cam_xy[1]), self.s * float(cam_xy[0])],
                        dtype=np.float64)

    def visible_width_m(self) -> float:
        return float(self.width) / self.s

    def visible_height_m(self) -> float:
        return float(self.height) / self.s


class OverlayGeometry:
    """Static plan geometry projected once, shifted per frame."""

    def __init__(self, projector: BirdEyeProjector, plan, obstacles=None,
                 frame=None, occupancy=None):
        self.p = projector
        self.frame = frame
        self.occupancy = (None if occupancy is None
                          else np.asarray(occupancy, dtype=np.uint8))
        self._tint = None
        self.corridor = [projector.base(p) for p in getattr(plan, "corridor_world", [])]
        self.ellipses = [projector.base(e) for e in getattr(plan, "ellipse_world", [])]
        candidates = getattr(plan, "candidates_world", None)
        self.candidates = []
        if candidates is not None and len(candidates):
            for path in np.asarray(candidates, dtype=np.float64):
                self.candidates.append(projector.base(path))
        self.curve = projector.base(np.asarray(plan.curve_world, dtype=np.float64))
        raw = getattr(plan, "raw_curve_world", None)
        self.raw = (projector.base(np.asarray(raw, dtype=np.float64))
                    if raw is not None and len(raw) else None)
        if frame is not None and getattr(plan, "start_world", None) is not None:
            self.start = projector.base(
                np.asarray(plan.start_world, dtype=np.float64)[None])[0]
            self.goal = projector.base(
                np.asarray(plan.goal_world, dtype=np.float64)[None])[0]
        else:
            self.start = self.curve[0]
            self.goal = self.curve[-1]
        self.obstacles = []
        self.obstacle_labels: List[str] = []
        if obstacles and frame is not None:
            for box in obstacles:
                self.obstacles.append(projector.base(obstacle_world_corners(box, frame)))
                self.obstacle_labels.append(str(box.get("name", "obstacle")))
        self.obstacle_centres = []
        if obstacles and frame is not None:
            for box in obstacles:
                self.obstacle_centres.append(projector.base(np.asarray([box["world_xy"]]))[0])

    # ------------------------------------------------------- occupancy layer
    def _tint_image(self) -> np.ndarray:
        """BGRA tint of the canonical occupancy: green drivable, red not."""
        if self._tint is None:
            occ = self.occupancy
            rgba = np.zeros((occ.shape[0], occ.shape[1], 4), dtype=np.uint8)
            free = occ == 0
            rgba[free] = (70, 210, 70, 60)
            rgba[~free] = (60, 60, 235, 60)
            self._tint = rgba
        return self._tint

    def _grid_to_camera(self, cam_xy) -> np.ndarray:
        import cv2

        res = int(self.occupancy.shape[0])
        src = np.float32([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        scene = np.stack((src[:, 0] / (res - 1) * 2.0 - 1.0,
                          src[:, 1] / (res - 1) * 2.0 - 1.0), axis=1)
        world = self.frame.world_from_scene(scene)
        pixels = self.p.base(world) + self.p.offset(cam_xy)[None, :]
        return cv2.getAffineTransform(src, pixels.astype(np.float32))

    def _draw_occupancy(self, frame: np.ndarray, cam_xy) -> None:
        import cv2

        matrix = self._grid_to_camera(cam_xy)
        warped = cv2.warpAffine(self._tint_image(), matrix,
                                (frame.shape[1], frame.shape[0]),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha = (warped[:, :, 3:4].astype(np.float32) / 255.0)
        rgb = warped[:, :, :3].astype(np.float32)
        frame[:] = (frame.astype(np.float32) * (1.0 - alpha) + rgb * alpha
                    ).astype(np.uint8)

    # ------------------------------------------------------------- drawing
    @staticmethod
    def _clip(pts: np.ndarray, offset: np.ndarray, limit: float = 20000.0) -> np.ndarray:
        out = np.asarray(pts, dtype=np.float64) + offset[None, :]
        out = np.clip(out, -limit, limit)
        return np.rint(out).astype(np.int32)

    @staticmethod
    def _dashed(img, pts: np.ndarray, colour, thickness: int = 2,
                dash: int = 9, gap: int = 7) -> None:
        import cv2

        if len(pts) < 2:
            return
        carry = 0
        for i in range(len(pts) - 1):
            a = pts[i].astype(np.float64)
            b = pts[i + 1].astype(np.float64)
            seg = float(np.linalg.norm(b - a))
            if seg < 1e-6:
                continue
            direction = (b - a) / seg
            t = -carry
            while t < seg:
                t0 = max(t, 0.0)
                t1 = min(t + dash, seg)
                if t1 > t0:
                    cv2.line(img, tuple(np.rint(a + t0 * direction).astype(int)),
                             tuple(np.rint(a + t1 * direction).astype(int)),
                             colour, thickness, cv2.LINE_AA)
                t += dash + gap
            carry = (seg - t0) % (dash + gap)
            carry = (dash + gap) - carry if carry else 0

    def draw(self, frame: np.ndarray, cam_xy, executed_xy=None,
             hud: Optional[Sequence[str]] = None,
             show_corridor: bool = True, show_ellipses: bool = False,
             show_candidates: bool = False, show_raw: bool = True,
             show_executed: bool = True, show_occupancy: bool = True) -> np.ndarray:
        import cv2

        offset = self.p.offset(cam_xy)

        if show_occupancy and self.occupancy is not None and self.frame is not None:
            self._draw_occupancy(frame, cam_xy)

        if show_corridor and self.corridor:
            layer = frame.copy()
            for poly in self.corridor:
                pts = self._clip(poly, offset)
                if len(pts) >= 3:
                    cv2.fillPoly(layer, [pts], CORRIDOR)
            cv2.addWeighted(layer, 0.22, frame, 0.78, 0.0, frame)

        if show_candidates:
            for path in self.candidates:
                cv2.polylines(frame, [self._clip(path, offset)], False,
                              CANDIDATE, 1, cv2.LINE_AA)

        for poly in (self.ellipses if show_ellipses else []):
            cv2.polylines(frame, [self._clip(poly, offset)], True,
                          ELLIPSE, 1, cv2.LINE_AA)

        if show_raw and self.raw is not None:
            self._dashed(frame, self._clip(self.raw, offset), RAW, 2)

        cv2.polylines(frame, [self._clip(self.curve, offset)], False,
                      GUIDED, 2, cv2.LINE_AA)

        for poly in self.obstacles:
            cv2.polylines(frame, [self._clip(poly, offset)], True,
                          OBSTACLE, 2, cv2.LINE_AA)

        if show_executed and executed_xy is not None and len(executed_xy) > 1:
            trace = self.p.base(np.asarray(executed_xy, dtype=np.float64))
            cv2.polylines(frame, [self._clip(trace, offset)], False,
                          EXECUTED, 2, cv2.LINE_AA)

        sx, sy = self._clip(self.start[None, :], offset)[0]
        gx, gy = self._clip(self.goal[None, :], offset)[0]
        cv2.circle(frame, (int(sx), int(sy)), 7, START, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(gx), int(gy)), 7, GOAL, -1, cv2.LINE_AA)

        for line_index, text in enumerate(hud or []):
            origin = (14, 30 + 26 * line_index)
            cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return frame
