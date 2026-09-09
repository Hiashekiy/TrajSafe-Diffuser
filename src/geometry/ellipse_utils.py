"""Ellipse geometry utilities shared by label generation and losses.

The ellipse is stored as a "Q" form: (p - c)^T Q (p - c) <= 1.
Only patch_Q_to_world is used by the active pipeline (offline_iris_wrapper).
"""
import numpy as np


def patch_Q_to_world(Q_pix, local_res):
    """Convert a quadratic-form matrix from patch-pixel to world coords.

    Mapping is p_pix = local_res * (p_world - anchor) + (half - 0.5), so the
    quadratic form scales by local_res^2.
    """
    return (float(local_res) ** 2) * np.asarray(Q_pix, dtype=np.float64)


def physical_ellipse_center(p, e6, absolute=False):
    """Return the physical ellipse centre from a trajectory anchor ``p``.

    ``e6`` stores the centre either as an offset from ``p`` (default,
    ``ellipse_center_mode == 'offset'``) or as an absolute scene coordinate
    (``ellipse_center_mode == 'absolute'``).  Works with torch and numpy.
    """
    if absolute:
        return e6[..., :2]
    return p + e6[..., :2]


def center_to_offset(center, p):
    """Offset representation ``center - p`` (used when generating E6 data)."""
    return center - p
