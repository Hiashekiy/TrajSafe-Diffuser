"""State normalization helpers.

All trajectory states are [ax, ay, x, y, vx, vy].  The LimitsNormalizer maps
the 6D state (and optionally the 2D position) to the range [-1, 1] per
dimension.  Data preparation saves a JSON file that round-trips with no error.
"""

import json
import os

import numpy as np


class LimitsNormalizer:
    """maps [xmin, xmax] -> [-1, 1] per dimension."""

    def __init__(self, X, eps=1e-8):
        X = np.asarray(X, dtype=np.float64).reshape(-1, X.shape[-1])
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        self.eps = eps
        self.dim = self.mins.size

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = (x - self.mins) / (self.maxs - self.mins + self.eps)
        return 2.0 * x - 1.0

    def unnormalize(self, x, eps=1e-4):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -1.0, 1.0)
        x = (x + 1.0) / 2.0
        return x * (self.maxs - self.mins + self.eps) + self.mins

    def to_dict(self):
        return {"mins": self.mins.tolist(), "maxs": self.maxs.tolist(), "eps": self.eps}

    @classmethod
    def from_dict(cls, d):
        obj = cls(np.zeros((1, len(d["mins"])), dtype=np.float64), eps=d.get("eps", 1e-8))
        obj.mins = np.asarray(d["mins"], dtype=np.float64)
        obj.maxs = np.asarray(d["maxs"], dtype=np.float64)
        return obj


def state_normalizer(states):
    """states [N, ... ,6] -> LimitsNormalizer over the 6D state."""
    return LimitsNormalizer(np.asarray(states, dtype=np.float64).reshape(-1, 6))


def save_normalization(path, normalizer, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"type": "state6_limits", "state": normalizer.to_dict()}
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_normalization(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = LimitsNormalizer.from_dict(data["state"])
    return n, data


def round_trip_error(normalizer, x):
    return float(np.max(np.abs(normalizer.unnormalize(normalizer.normalize(x)) - x)))
