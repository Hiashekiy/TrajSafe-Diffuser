"""Scene frame: the single source of truth for world <-> scene normalization.

Convention (fixed, do not scatter elsewhere):
- Scene coordinates are in [-1, 1]^2.  x increases to the right, y increases upward.
- The occupancy / SDF grid is stored as [row=H(y), col=W(x)], with
  row 0 = scene_y = -1  (bottom)  and  row (H-1) = scene_y = +1 (top).
  This matches grid_sample with grid coords == scene coords (no flip) and
  imshow(origin="lower").
- Mapping from a maze's world extent to the canonical scene [-1,1]^2 uses a
  UNIFORM scale + centring, so the maze's larger dimension maps to [-1, 1] and
  geometry is NOT distorted (circles stay circles).  Areas outside the maze
  extent are treated as wall.

Only values of type float32/float64 tensors or numpy arrays are expected here.
"""

import numpy as np
import torch
import torch.nn.functional as F


def uniform_scale_center(extent):
    """Return (scale, cx, cy) mapping world -> scene: p_scene = s * (p_world - c)."""
    x0, x1, y0, y1 = extent
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    s = 2.0 / max(dx, dy)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return s, cx, cy


class SceneFrame:
    """World <-> scene coordinate converter for one maze."""

    def __init__(self, extent):
        self.extent = tuple(float(v) for v in extent)
        self.s, self.cx, self.cy = uniform_scale_center(self.extent)

    def world_to_scene(self, p):
        """p [...,2] world -> scene [-1,1]."""
        p = p - torch.Tensor([self.cx, self.cy]).to(p.device).to(p.dtype)
        return self.s * p

    def scene_to_world(self, p):
        """p [...,2] scene -> world."""
        return p / self.s + torch.Tensor([self.cx, self.cy]).to(p.device).to(p.dtype)

    def world_to_scene_np(self, p):
        p = np.asarray(p, dtype=np.float64) - np.array([self.cx, self.cy], dtype=np.float64)
        return (self.s * p).astype(np.float32)

    def scene_to_world_np(self, p):
        p = np.asarray(p, dtype=np.float64) / self.s + np.array([self.cx, self.cy], dtype=np.float64)
        return p.astype(np.float32)

    def scale(self):
        return self.s


def build_scene_occupancy(frame, occ_orig, extent_orig, gres_orig, res=256):
    """Resample a maze's occupancy grid into a res x res scene occupancy map.

    occ_orig: [ny, nx] uint8/float array of the maze (1 = wall, 0 = free),
              defined over extent_orig at gres_orig (pixels per world unit).
    Returns: [res, res] float32 (1 = wall, 0 = free) in scene [-1,1]^2.
    """
    x0, x1, y0, y1 = extent_orig
    ny, nx = occ_orig.shape
    # scene cell centres
    xs = np.linspace(-1.0, 1.0, res + 1)[:-1] + 1.0 / res
    ys = np.linspace(-1.0, 1.0, res + 1)[:-1] + 1.0 / res
    sx, sy = np.meshgrid(xs, ys, indexing="xy")  # [res,res]
    cell_scene = np.stack([sx.reshape(-1), sy.reshape(-1)], axis=-1)  # [res*res,2]
    cell_world = frame.scene_to_world_np(cell_scene)                  # [res*res,2]
    px = np.round((cell_world[:, 0] - x0) * gres_orig - 0.5).astype(int)
    py = np.round((cell_world[:, 1] - y0) * gres_orig - 0.5).astype(int)
    valid = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    out = np.ones((res * res,), dtype=np.float32)
    px_c = np.clip(px, 0, nx - 1)
    py_c = np.clip(py, 0, ny - 1)
    out[valid] = occ_orig[py_c[valid], px_c[valid]]
    return out.reshape(res, res).astype(np.float32)


def build_scene_sdf(occ):
    """Signed distance from a binary scene occupancy (1=wall).  free>0, wall<0.

    Returns values in **scene units** (occupancy grid spans scene [-1,1]^2);
    distance_transform_edt returns per-pixel distances, so we scale by the cell
    size cell = 2 / W (scene [-1,1] spans W grid cells).
    """
    import scipy.ndimage as ndimage
    wall = occ.astype(bool)
    free = ~wall
    d_free = ndimage.distance_transform_edt(free)
    d_wall = ndimage.distance_transform_edt(wall)
    cell = 2.0 / float(occ.shape[1])          # scene units per grid cell
    return ((d_free - d_wall) * cell).astype(np.float32)


def sample_sdf_scene(sdf_map, p_scene):
    """Differentiable bilinear sample of a scene SDF map at scene points.

    sdf_map: [B,1,H,W] over scene [-1,1]^2.  p_scene: [B,Hp,2] in [-1,1].
    Returns [B,Hp] **in scene units**.  The stored SDF is expected to already be
    in scene units (see build_scene_sdf); no rescaling is done here.
    """
    grid = p_scene.unsqueeze(2)  # [B,Hp,1,2]
    out = F.grid_sample(sdf_map, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.squeeze(1).squeeze(-1)  # [B,Hp]
