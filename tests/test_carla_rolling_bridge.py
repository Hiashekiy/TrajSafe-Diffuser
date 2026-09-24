import os
import sys

import numpy as np


BRIDGE_SRC = os.path.join(os.path.dirname(__file__), "..", "src",
                          "carla_trajsafe_bridge", "src")
if BRIDGE_SRC not in sys.path:
    sys.path.insert(0, BRIDGE_SRC)

from carla_bridge.rolling_frame import WorldSceneFrame
from carla_bridge.rolling_occupancy import CarlaMapRasterizer, LaneSegment
from carla_bridge.rolling_planner import RoutePolyline


def test_world_scene_frame_roundtrip_and_scale():
    frame = WorldSceneFrame([10.0, 20.0])
    points = np.array([[-70.0, -60.0], [10.0, 20.0], [90.0, 100.0]])
    scene = frame.scene_from_world(points)
    assert np.allclose(scene, [[-1, -1], [0, 0], [1, 1]])
    assert np.allclose(frame.world_from_scene(scene), points)
    assert frame.scene_scale_m == 80.0


def test_route_horizon_fits_complete_bend():
    route = RoutePolyline([[0, 0], [50, 0], [50, 50], [100, 50]])
    start_s, goal_s, section = route.select_horizon(
        [0, 0], 0.0, lookahead_m=90.0, window_size_m=160.0,
        window_margin_m=12.0)
    assert start_s == 0.0
    assert goal_s == 90.0
    frame = WorldSceneFrame.around_polyline(section, margin_m=12.0)
    assert frame.contains_world(section, margin_m=12.0)


def test_rasterizer_canonical_axis_and_dynamic_obstacle():
    frame = WorldSceneFrame([0.0, 0.0])
    rasterizer = CarlaMapRasterizer([
        LaneSegment(np.array([-50.0, 0.0]), np.array([50.0, 0.0]), 4.0)
    ])
    box = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], dtype=float)
    occupancy = rasterizer.rasterize(
        frame, obstacle_polygons_world=[box], obstacle_erosion_cells=0)
    center = np.rint(frame.grid_from_world([[0.0, 0.0]])[0]).astype(int)
    road = np.rint(frame.grid_from_world([[20.0, 0.0]])[0]).astype(int)
    assert occupancy[center[1], center[0]] == 1
    assert occupancy[road[1], road[0]] == 0

