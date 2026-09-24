"""Dynamic bird's-eye overlay for the continuous rolling planner."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .overlay import BirdEyeProjector


class RollingOverlay:
    """Mutable overlay: global route, active/ready plans and handoff point."""

    ROUTE = (255, 210, 40)
    ACTIVE = (255, 0, 255)
    READY = (0, 220, 255)
    EXECUTED = (255, 255, 255)
    HANDOFF = (0, 165, 255)

    def __init__(self, projector: BirdEyeProjector, route_world):
        self.p = projector
        self.route = self.p.base(np.asarray(route_world, dtype=np.float64))
        self.active = None
        self.ready = None
        self.handoff = None
        self.active_id = None
        self.ready_id = None
        self.pending_id = None

    def set_active(self, plan) -> None:
        self.active = self.p.base(plan.profile.xy)
        self.active_id = int(plan.plan_id)
        self.ready = None
        self.ready_id = None
        self.pending_id = None
        self.handoff = None

    def set_pending(self, plan_id: int, handoff_world) -> None:
        self.pending_id = int(plan_id)
        self.handoff = self.p.base(
            np.asarray(handoff_world, dtype=np.float64).reshape(1, 2))[0]

    def set_ready(self, plan) -> None:
        self.ready = self.p.base(plan.profile.xy)
        self.ready_id = int(plan.plan_id)
        self.pending_id = None

    def clear_candidate(self) -> None:
        self.ready = None
        self.ready_id = None
        self.pending_id = None
        self.handoff = None

    def _pixels(self, base, cam_xy):
        points = np.asarray(base, dtype=np.float64) + self.p.offset(cam_xy)
        return np.rint(np.clip(points, -20000, 20000)).astype(np.int32)

    @staticmethod
    def _dashed(frame, points, colour, thickness=2, stride=10):
        if points is None or len(points) < 2:
            return
        for i in range(0, len(points) - 1, 2):
            j = min(i + 1, len(points) - 1)
            cv2.line(frame, tuple(points[i]), tuple(points[j]), colour,
                     thickness, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, cam_xy, executed_xy=None,
             hud: Optional[Sequence[str]] = None, **_options) -> np.ndarray:
        route = self._pixels(self.route, cam_xy)
        self._dashed(frame, route, self.ROUTE, 1)

        if self.active is not None:
            cv2.polylines(frame, [self._pixels(self.active, cam_xy)], False,
                          self.ACTIVE, 3, cv2.LINE_AA)
        if self.ready is not None:
            self._dashed(frame, self._pixels(self.ready, cam_xy),
                         self.READY, 3)
        if executed_xy is not None and len(executed_xy) > 1:
            trace = self.p.base(np.asarray(executed_xy, dtype=np.float64))
            cv2.polylines(frame, [self._pixels(trace, cam_xy)], False,
                          self.EXECUTED, 2, cv2.LINE_AA)
        if self.handoff is not None:
            point = self._pixels(self.handoff[None], cam_xy)[0]
            cv2.circle(frame, tuple(point), 8, self.HANDOFF, 2, cv2.LINE_AA)

        # Legend and HUD use outlined text so they remain readable on every map.
        legend = "route cyan | active magenta | ready yellow | executed white"
        lines = [legend] + list(hud or [])
        for index, line in enumerate(lines):
            origin = (16, 30 + 27 * index)
            cv2.putText(frame, str(line), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, str(line), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

