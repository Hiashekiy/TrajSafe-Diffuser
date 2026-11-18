"""Compose the accepted DMA and third-party sknw stages."""

from dataclasses import dataclass
import numpy as np
from .delta_medial_axis import DeltaMedialAxis
from .skeleton_graph import build_skeleton_graph


@dataclass
class PreparedMap:
    """Arrays share the input H×W grid; EDT remains in pixel units."""

    occupancy: np.ndarray
    skeleton: np.ndarray
    thin_skeleton: np.ndarray
    distance_px: np.ndarray
    nearest_background_rc: np.ndarray
    graph: dict


def prepare_occupancy_map(occupancy, delta_px=3.0, resolution_m_per_pixel=0.625):
    """Prepare binary occupancy (0=free, 1=occupied), without modifying it."""
    occupancy = np.asarray(occupancy)
    if occupancy.ndim != 2 or not np.isin(occupancy, [0, 1]).all():
        raise ValueError("occupancy must be a binary 2D array (0=free, 1=occupied)")
    engine = DeltaMedialAxis(occupancy == 0)
    skeleton = engine.extract(delta_px)
    distance = engine.distance[1:-1, 1:-1].copy()
    graph, thin = build_skeleton_graph(skeleton, distance, resolution_m_per_pixel)
    graph["parameters"]["delta_px"] = float(delta_px)
    graph["parameters"]["outside_map"] = "occupied one-pixel padding"
    # Convert padded EDT projection coordinates back to original row/column.
    # Values -1/H/W are valid exterior background sites.
    projection = engine.projection[:, 1:-1, 1:-1].copy() - 1
    return PreparedMap(occupancy.copy(), skeleton, thin, distance, projection, graph)
