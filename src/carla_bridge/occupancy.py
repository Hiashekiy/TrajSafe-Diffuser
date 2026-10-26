"""Rasterise only approved road polygons and quasi-static actor footprints."""
import cv2
import numpy as np
from scipy.ndimage import label


def road_quads(waypoints):
    quads = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        corners = []
        for w in (a, b):
            p, r = w.transform.location, w.transform.get_right_vector()
            d = w.lane_width/2
            corners.extend([[p.x+r.x*d, p.y+r.y*d], [p.x-r.x*d, p.y-r.y*d]])
        quads.append(np.array(corners)[[0, 1, 3, 2]])
    return quads


def raster(frame, quads, static_objects=(), inflation_cells=4, inflation_lateral_cells=None):
    road = np.ones((frame.resolution, frame.resolution), np.uint8)
    for q in quads:
        cv2.fillPoly(road, [np.rint(frame.grid(q)).astype(np.int32)], 0)
    obstacles = np.zeros_like(road)
    for body in static_objects:
        cv2.fillPoly(obstacles, [np.rint(frame.grid(body.corners())).astype(np.int32)], 1)
    if inflation_cells:
        # Exactly four cells on real actor obstacles. Never erode/widen real roads.
        nx = 2*inflation_cells+1
        ny = 2*(inflation_cells if inflation_lateral_cells is None else inflation_lateral_cells)+1
        obstacles = cv2.dilate(obstacles, np.ones((ny, nx), np.uint8))
    return np.maximum(road, obstacles), road


def connected(frame, occupancy, start, goal):
    if not frame.free(occupancy, np.array([start, goal])).all():
        return False
    labels, _ = label(occupancy == 0)
    a, b = np.floor(frame.grid(np.array([start, goal]))).astype(int)
    return labels[a[1], a[0]] == labels[b[1], b[0]]


def footprint_safe(path, body, frame, road, static_objects, road_area=None):
    """Whole curve: filled rectangles and conservative swept road/body envelopes."""
    from dataclasses import replace
    from .dynamic_objects import intersect
    if road_area is not None:
        from shapely.geometry import Polygon
        from shapely.prepared import prep
        from .safety_supervisor import swept_box
        allowed = prep(road_area.buffer(1e-6))
        static_polygons = [Polygon(obj.corners(.15)) for obj in static_objects]
    previous = None
    for p, yaw in zip(path.xy, path.yaw):
        test = replace(body, xy=p, yaw=float(yaw))
        corners = test.corners(0.15)
        if road_area is not None:
            envelope = Polygon(corners) if previous is None else swept_box(previous,test,.15)
            if not allowed.covers(envelope) or any(envelope.intersects(poly) for poly in static_polygons):
                return False
            previous = test
            continue
        g = frame.grid(corners)
        if (g < 0).any() or (g > frame.resolution-1).any():
            return False
        lo = np.floor(g.min(axis=0)).astype(int)
        hi = np.ceil(g.max(axis=0)).astype(int)
        mask = np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1), np.uint8)
        cv2.fillPoly(mask, [np.rint(g-lo).astype(np.int32)], 1)
        if np.any(road[lo[1]:hi[1]+1, lo[0]:hi[0]+1][mask > 0]):
            return False
        if any(intersect(test, obj, margin=0.15) for obj in static_objects):
            return False
    return True
