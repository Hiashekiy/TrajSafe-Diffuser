"""Clearance risk maps: the obstacle distance transform and the risk functions added to B3.

Nothing in this module searches.  :mod:`field_guided_search.search` adds ``beta * R(D)`` to the
priority map it already builds, so the queue loop, the neighbour order and the
first-discovery-sets-the-parent rule stay exactly as they were.

``D`` is the Euclidean distance, **in cells**, from each free cell to the nearest obstacle
cell centre; the outside of the map counts as obstacle, so a fully free map still has finite
clearance near its border.  It is a relative index, not a continuous geometric gap to the
obstacle surface, and it does not model the robot footprint -- dilate the occupancy map first
if you need that (plan §3.1).

Three risk functions are supported, all non-negative, bounded by 1 and non-increasing in ``D``:

============== ========================================== ==================
mode           formula                                    parameters
============== ========================================== ==================
quadratic_hinge ``[max(0, 1 - D/d_s)]^2``                 ``threshold``
exponential    ``exp(-D/sigma)``                          ``sigma``
logistic       ``1 / (1 + exp((D - d_s)/tau))``           ``threshold``, ``tau``
============== ========================================== ==================

The hinge is the default because it reaches exactly 0 at ``D >= d_s``: beyond the threshold
there is no incentive left to drift further away from the guidance field.
"""

from __future__ import annotations

import time

import numpy as np

RISK_MODES = ("quadratic_hinge", "exponential", "logistic")

_SCIPY_HINT = ("clearance-aware search needs scipy for the Euclidean distance transform: "
               "pip install -e '.[fast]'  (or pass a precomputed clearance_penalty_map)")

_RANGE_TOL = 1e-9


# --------------------------------------------------------------------------- #
# distance transform
# --------------------------------------------------------------------------- #
def _edt():
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_SCIPY_HINT) from exc
    return distance_transform_edt


def clearance_distance(grid) -> np.ndarray:
    """``D`` in cells: distance from each free cell to the nearest obstacle cell.

    The map is padded with one obstacle ring so cells on the border are not reported as
    infinitely clear.
    """
    padded = np.pad(grid.free, 1, mode="constant", constant_values=False)
    dist = _edt()(padded)[1:-1, 1:-1].astype(np.float64)
    dist[~grid.free] = 0.0
    return dist


# --------------------------------------------------------------------------- #
# validation + risk functions
# --------------------------------------------------------------------------- #
def validate_risk_params(beta, mode, threshold, sigma, tau) -> tuple[float, str, float, float, float]:
    """Reject bad clearance knobs early instead of silently clipping them during the search."""
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"clearance_beta must be finite and >= 0, got {beta!r}")
    if mode not in RISK_MODES:
        raise ValueError(f"clearance_mode must be one of {RISK_MODES}, got {mode!r}")
    out = []
    for name, value in (("clearance_threshold", threshold), ("clearance_sigma", sigma),
                        ("clearance_tau", tau)):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        out.append(value)
    return beta, mode, out[0], out[1], out[2]


def clearance_risk(distance, mode: str = "quadratic_hinge", threshold: float = 3.0,
                   sigma: float = 2.0, tau: float = 0.7) -> np.ndarray:
    """``R(D)`` in ``[0, 1]`` for a distance map in cells."""
    mode = str(mode)
    if mode not in RISK_MODES:
        raise ValueError(f"clearance_mode must be one of {RISK_MODES}, got {mode!r}")
    d = np.asarray(distance, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"the distance map must be 2-D, got shape {d.shape}")
    if not np.all(np.isfinite(d)):
        raise ValueError("the distance map contains NaN or Inf")
    if d.size and float(d.min()) < 0.0:
        raise ValueError(f"the distance map must be >= 0, got minimum {float(d.min())!r}")
    if mode == "quadratic_hinge":
        if not (float(threshold) > 0.0):
            raise ValueError(f"threshold must be > 0 for {mode}, got {threshold!r}")
        excess = np.maximum(0.0, 1.0 - d / float(threshold))
        return excess * excess
    if mode == "exponential":
        if not (float(sigma) > 0.0):
            raise ValueError(f"sigma must be > 0 for {mode}, got {sigma!r}")
        return np.exp(-d / float(sigma))
    if not (float(threshold) > 0.0) or not (float(tau) > 0.0):
        raise ValueError(f"threshold and tau must be > 0 for {mode}, "
                         f"got {threshold!r} / {tau!r}")
    # 0.5*(1 - tanh(z/2)) is the logistic function without the overflow of exp for large z
    z = (d - float(threshold)) / float(tau)
    return 0.5 * (1.0 - np.tanh(0.5 * z))


def validate_penalty_map(penalty_map, grid, name: str = "clearance_penalty_map") -> np.ndarray:
    """Shape / finiteness / range check for a caller-supplied risk map (plan §3.4)."""
    arr = np.asarray(penalty_map)
    if arr.dtype == object or np.iscomplexobj(arr):
        raise ValueError(f"{name} must be a real-valued array, got dtype {arr.dtype}")
    arr = arr.astype(np.float64, copy=False)
    if arr.shape != grid.free.shape:
        raise ValueError(f"{name} must have shape {grid.free.shape} "
                         f"(the map), got {arr.shape}")
    values = arr[grid.free]
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite on every free cell")
    if values.size and (float(values.min()) < -_RANGE_TOL or float(values.max()) > 1.0 + _RANGE_TOL):
        raise ValueError(f"{name} must stay in [0, 1], "
                         f"got [{float(values.min()):.6g}, {float(values.max()):.6g}]")
    return arr


# --------------------------------------------------------------------------- #
# per-map cache
# --------------------------------------------------------------------------- #
class ClearanceRiskMap:
    """Distance transform and risk maps of **one** grid, computed at most once per key.

    ``D`` does not depend on start, goal or the guidance field, so one instance can serve
    every search on the same map -- including several ``beta``/``d_s`` combinations, which is
    what makes a warm re-plan pay only for the search loop.  The cache is bound to the grid it
    was built from: if the occupancy map changes, build a new ``Grid``/``GuidanceField`` or
    call :meth:`invalidate`.
    """

    def __init__(self, grid):
        self.grid = grid
        self._distance: np.ndarray | None = None
        self._risk: dict[tuple, np.ndarray] = {}
        self.edt_ms = 0.0
        self.risk_ms = 0.0
        self.edt_calls = 0
        self.risk_calls = 0

    # ------------------------------------------------------------------ #
    def distance(self) -> np.ndarray:
        """``D`` in cells; computed on the first call only."""
        if self._distance is None:
            t0 = time.perf_counter()
            self._distance = clearance_distance(self.grid)
            self.edt_ms += (time.perf_counter() - t0) * 1000.0
            self.edt_calls += 1
        return self._distance

    def risk(self, mode: str = "quadratic_hinge", threshold: float = 3.0, sigma: float = 2.0,
             tau: float = 0.7) -> np.ndarray:
        """``R(D)`` for this map; cached per ``(mode, threshold, sigma, tau)``."""
        key = (str(mode), float(threshold), float(sigma), float(tau))
        cached = self._risk.get(key)
        if cached is None:
            t0 = time.perf_counter()
            cached = clearance_risk(self.distance(), mode, threshold, sigma, tau)
            self.risk_ms += (time.perf_counter() - t0) * 1000.0
            self.risk_calls += 1
            self._risk[key] = cached
        return cached

    def invalidate(self) -> None:
        """Drop everything after the occupancy map changed in place."""
        self._distance = None
        self._risk.clear()

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"ClearanceRiskMap({self.grid.height}x{self.grid.width}, "
                f"edt={self.edt_calls}, risk={self.risk_calls})")
