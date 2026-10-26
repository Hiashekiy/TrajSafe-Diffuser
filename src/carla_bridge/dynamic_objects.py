"""Snapshot-only object state, hysteresis, and conservative box prediction."""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Body:
    actor_id: int
    xy: np.ndarray  # bounding box centre, including actor-relative offset
    yaw: float
    velocity: np.ndarray
    half_length: float
    half_width: float
    frame: int = 0
    time: float = 0.0

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    def predict(self, t):
        # Constant world velocity: no access to NPC future scripts.
        return replace(self, xy=self.xy+self.velocity*t)

    def corners(self, margin=0.0):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, s], [-s, c]])
        offsets = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
        return offsets * [self.half_length+margin, self.half_width+margin] @ rotation + self.xy


def intersect(a, b, margin=0.25):
    pa, pb = a.corners(margin), b.corners(margin)
    for yaw in (a.yaw, b.yaw):
        for axis in (np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])):
            x, y = pa@axis, pb@axis
            if x.max() < y.min() or y.max() < x.min():
                return False
    return True


def snapshot_body(actor, snapshot, box=None):
    state = snapshot.find(actor.id)
    if state is None:
        raise RuntimeError('ACTOR_MISSING_FROM_SNAPSHOT')
    tf, v = state.get_transform(), state.get_velocity()
    box = actor.bounding_box if box is None else box
    # Transform.transform(Location) mutates the passed Location in CARLA.
    # Never transform the cached box.location in place: it would drift every tick.
    offset = np.array([box.location.x, box.location.y, box.location.z, 1.])
    centre = np.asarray(tf.get_matrix()) @ offset
    return Body(actor.id, centre[:2].copy(),
                np.deg2rad(tf.rotation.yaw+box.rotation.yaw), np.array([v.x, v.y]),
                float(box.extent.x), float(box.extent.y), snapshot.frame, snapshot.timestamp.elapsed_seconds)


class StaticTracker:
    def __init__(self):
        self.stationary = set()
        self.since = {}

    def update(self, objects, now):
        alive = {x.actor_id for x in objects}
        self.stationary.intersection_update(alive)
        self.since = {k: v for k, v in self.since.items() if k in alive}
        for obj in objects:
            threshold = 1.2 if obj.actor_id in self.stationary else 0.5
            trigger = obj.speed > threshold if obj.actor_id in self.stationary else obj.speed < threshold
            if not trigger:
                self.since.pop(obj.actor_id, None)
            elif now-self.since.setdefault(obj.actor_id, now) >= 1.0:
                if obj.actor_id in self.stationary:
                    self.stationary.remove(obj.actor_id)
                else:
                    self.stationary.add(obj.actor_id)
                self.since.pop(obj.actor_id, None)
        return [x for x in objects if x.actor_id in self.stationary]
