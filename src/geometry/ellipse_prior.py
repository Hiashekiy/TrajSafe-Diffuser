"""Ellipse-prior substitution for 实验 3 (实验/docs/实验3_*.md §2).

The ONLY thing this module changes is the pair ``(semi-axes, orientation)`` that
is handed to the convex-region constructor.  The network's safety feature
``H_ell``, the skeleton selection, the region/bridge algorithm, the ALM and the
DDIM updates are untouched: callers keep passing the same centres and receive a
rewritten ``shape4``.

``shape4 = [log a, log b, cos 2*theta, sin 2*theta]`` (scene units, see
``src.geometry.convex_region.ellipse_shape4_to_quadratic``).

Modes
-----
``learned``        the network prediction (identity -- the default; the existing
                   pipeline is bit-for-bit unchanged)
``fixed_circle``   ``a = b = r0``, orientation irrelevant
``fixed_ellipse``  ``a = a0, b = b0``, oriented along the local skeleton tangent
"""

from __future__ import annotations

import numpy as np

MODES = ("learned", "fixed_circle", "fixed_ellipse")


def skeleton_tangent_angles(gamma, gamma_length: int, anchors) -> np.ndarray:
    """Tangent angle of the selected dense skeleton at the corridor anchors.

    ``gamma``: ``[G,2]`` dense skeleton points (uniform parameterisation),
    ``gamma_length``: number of valid points, ``anchors``: ``[H]`` progress
    values in ``[0,1]``.  Returns ``[H]`` angles in radians; a degenerate
    (zero-length) skeleton falls back to a constant 0 direction, which is the
    single predefined fallback of 实验 3 §2.1 step 4.
    """
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1)
    pts = np.asarray(gamma, dtype=np.float64)
    n = int(min(max(gamma_length, 0), len(pts)))
    if n < 2:
        return np.zeros(anchors.shape, dtype=np.float64)
    pts = pts[:n]
    s = np.linspace(0.0, 1.0, n)
    x = np.interp(anchors, s, pts[:, 0])
    y = np.interp(anchors, s, pts[:, 1])
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return np.zeros(anchors.shape, dtype=np.float64)
    tx = np.gradient(x)
    ty = np.gradient(y)
    norm = np.hypot(tx, ty)
    if not np.isfinite(norm).all() or float(norm.max()) <= 1e-12:
        return np.zeros(anchors.shape, dtype=np.float64)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return np.arctan2(ty / norm, tx / norm)


def apply_ellipse_prior(shape4, centers, gamma, gamma_length, anchors,
                        mode: str = "learned", radius: float | None = None,
                        semi_major: float | None = None,
                        semi_minor: float | None = None):
    """Return the ``shape4`` that actually enters the convex-region constructor.

    ``shape4`` is ``[H,4]`` (single sample).  ``learned`` returns the input array
    unchanged (same object), so the default pipeline cannot be perturbed.
    """
    if mode in (None, "", "learned"):
        return shape4
    if mode not in MODES:
        raise ValueError("unknown ellipse prior mode %r (expected one of %s)"
                         % (mode, MODES))
    shp = np.asarray(shape4, dtype=np.float64).reshape(-1, 4).copy()
    if mode == "fixed_circle":
        if radius is None or not np.isfinite(radius) or radius <= 0:
            raise ValueError("fixed_circle needs a positive frozen radius")
        shp[:, 0] = np.log(float(radius))
        shp[:, 1] = np.log(float(radius))
        shp[:, 2] = 1.0
        shp[:, 3] = 0.0
        return shp
    if semi_major is None or semi_minor is None:
        raise ValueError("fixed_ellipse needs frozen semi-major/-minor axes")
    a0, b0 = float(max(semi_major, semi_minor)), float(min(semi_major,
                                                          semi_minor))
    if not np.isfinite(a0) or not np.isfinite(b0) or b0 <= 0:
        raise ValueError("fixed_ellipse axes must be positive and finite")
    theta = skeleton_tangent_angles(gamma, gamma_length, anchors)
    if theta.shape[0] != shp.shape[0]:
        raise ValueError("anchor count %d != shape4 rows %d"
                         % (theta.shape[0], shp.shape[0]))
    shp[:, 0] = np.log(a0)
    shp[:, 1] = np.log(b0)
    shp[:, 2] = np.cos(2.0 * theta)
    shp[:, 3] = np.sin(2.0 * theta)
    return shp


def prior_from_config(config: dict | None) -> dict:
    """Normalise the ``corridor.ellipse_prior`` block (absent -> learned)."""
    cfg = dict(config or {})
    block = dict(cfg.get("ellipse_prior") or {})
    return dict(mode=str(block.get("mode", "learned") or "learned"),
                radius=block.get("radius_scene"),
                semi_major=block.get("semi_major_scene"),
                semi_minor=block.get("semi_minor_scene"))


def describe(config: dict | None) -> str:
    info = prior_from_config(config)
    if info["mode"] == "learned":
        return "learned"
    if info["mode"] == "fixed_circle":
        return "fixed_circle(r=%.5f scene)" % float(info["radius"])
    return "fixed_ellipse(a=%.5f, b=%.5f scene)" % (
        float(info["semi_major"]), float(info["semi_minor"]))
