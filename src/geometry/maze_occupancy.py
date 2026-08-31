"""Construct global Maze2D occupancy and local 128x128 Neural-IRIS input crops.
Coordinate convention:
  - world coordinates are continuous maze units ((x,y) = (col,row))
  - global occupancy bitmap: 1=wall, 0=free, built at global_res px/unit
  - local Neural-IRIS patch is 128x128 centred on an anchor world point
"""
import numpy as np

PATCH_SIZE = 128


def crop_local_patch(global_occ, anchor_xy, global_res, local_res=20.0,
                     patch_size=PATCH_SIZE):
    """Return patch_size x patch_size uint8 occupancy crop (1=wall).
    """
    ax, ay = anchor_xy
    h, w = global_occ.shape
    half = patch_size / 2.0
    px = np.arange(patch_size)
    py = np.arange(patch_size)
    XX, YY = np.meshgrid(px, py)   # XX=x pixel, YY=y pixel
    wx = ax - half / local_res + (XX + 0.5) / local_res
    wy = ay - half / local_res + (YY + 0.5) / local_res

    rx = np.floor(wx * global_res).astype(np.int64)
    ry = np.floor(wy * global_res).astype(np.int64)
    inb = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)
    patch = np.ones((patch_size, patch_size), dtype=np.uint8)
    rx_c = np.clip(rx, 0, w - 1)
    ry_c = np.clip(ry, 0, h - 1)
    global_vals = global_occ[ry_c, rx_c]
    patch[inb] = global_vals[inb]
    patch[global_vals == 0] = 0
    patch[~inb] = 1

    def patch_to_world(px_, py_):
        x = ax - half / local_res + (px_ + 0.5) / local_res
        y = ay - half / local_res + (py_ + 0.5) / local_res
        return np.stack([x, y], axis=-1)

    def world_to_patch(x, y):
        px_ = (x - ax) * local_res + half - 0.5
        py_ = (y - ay) * local_res + half - 0.5
        return px_, py_

    return patch, patch_to_world, world_to_patch, local_res
