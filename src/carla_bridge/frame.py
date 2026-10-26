"""160 m canonical frame. Rows increase with scene y; no image flip here.

The old E-drive LocalFrame has an 80 m forward-biased crop. It must never be
used for the current full-scene checkpoint.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SceneFrame:
    origin: np.ndarray
    yaw: float  # radians, CARLA +y is right
    size: float = 160.0
    resolution: int = 256

    @property
    def basis(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, s], [-s, c]])

    def local(self, world):
        return (np.asarray(world) - self.origin) @ self.basis.T

    def scene(self, world):
        return self.local(world) / (self.size / 2)

    def world(self, scene):
        return np.asarray(scene) * (self.size / 2) @ self.basis + self.origin

    def grid(self, world):
        return (self.scene(world) + 1) * (self.resolution / 2)

    def free(self, occupancy, world):
        g = self.grid(world)
        valid = np.isfinite(g).all(axis=-1) & (g >= 0).all(axis=-1) & (g < self.resolution).all(axis=-1)
        ij = np.floor(np.nan_to_num(g)).astype(int).clip(0, self.resolution-1)
        return valid & (occupancy[ij[..., 1], ij[..., 0]] == 0)


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi
