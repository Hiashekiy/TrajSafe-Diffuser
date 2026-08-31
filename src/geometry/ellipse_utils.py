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
