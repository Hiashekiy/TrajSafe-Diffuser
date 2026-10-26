"""Signed distance field of a preprocessed occupancy grid (第4章 4.1.2).

WHY THIS MODULE EXISTS
----------------------
The four external safety baselines of section 4.2 (Penalty / Projected /
Primal-Dual / SDF-ALM) all optimise the SAME physical safety requirement:
"the trajectory must not enter the obstacle region of the preprocessed
occupancy grid".  That requirement is expressed with a 2-D signed distance
field of the occupancy grid, so all four share one implementation:

    d_occ(p) > 0   free space
    d_occ(p) = 0   obstacle boundary
    d_occ(p) < 0   inside an obstacle

and the constraint handed to every processor is

    g_h^{SDF}(Q) = d_safe - d_occ(B_h Q) <= 0 ,      d_safe = 0

(``d_safe = 0`` because the k=4 cache ``--border-mode protect`` already carries
the safety margin; see scripts/data/carla_full/05_erode_occupancy.py).

TWO FIELDS, ON PURPOSE
----------------------
``query(points, with_boundary=False)``  -> ``d_occ``: the occupancy SDF alone.
    This is the field the REPORTED metrics use (guide 4.1.3 §2/§3/§5:
    ``r_n^SDF = max_h [d_safe - d_occ(p_{n,h})]``), so the numbers in the main
    table are exactly the formula of the guide.

``query(points, with_boundary=True)``   -> ``min(d_occ, d_box)``: the
    "带地图边界的 SDF" of the guide's execution order step 4.  ``d_box`` is the
    (negative) distance to the 256^2 crop, so leaving the map becomes a
    violation the optimiser can SEE.  This is the field the four processors
    minimise.  Without it the constraint set and the collision metric disagree:
    ``_free_mask`` (src/diffusion/sampler.py) counts ``|p| > 1`` as a collision
    while the occupancy SDF says nothing about the crop -- the same class of bug
    that was already measured once in the corridor code (commit bf37aa3,
    docs/历史实验记录_160k8p.md §12.8).

Both are reported per sample (``sdf_*`` from d_occ, ``sdf_full_*`` from the
combined field) so no reader has to guess which convention a column used.

CONVENTIONS (verified against the repo, see tests/test_sdf_field.py)
-------------------------------------------------------------------
* occupancy ``[B,1,H,W]`` float in {0,1}, **1 = obstacle**, stored in the
  canonical orientation produced by ``00_build_processed.py``: the ROW index
  runs with +y and the COLUMN index with +x (``row 0 <-> y = -1``).
* scene coordinates in [-1,1]^2 are fed DIRECTLY to ``F.grid_sample`` with
  ``align_corners=False`` -- byte-identical convention to
  ``sampler._free_mask``, so the ``0.5`` iso-level here and the collision test
  there cut the same contour.
* everything is in SCENE units; 1 cell = ``2/H`` scene = 0.625 m for the
  160 m / 256 px dataset (``scene_to_meter = 80.0``).  ``to_metres`` converts.
* distances use the cell-centre convention: a free cell adjacent to an
  obstacle cell has ``d_occ = +0.5 cell``, i.e. the obstacle surface sits half
  a cell away from the cell centre, which is where the bilinear 0.5 contour of
  the occupancy actually is.

GRADIENTS
---------
The gradient is NOT autograd through ``grid_sample``: the bilinear interpolant
of an EDT has a vanishing gradient deep inside large obstacle blobs, which
stalls the penalty/dual methods exactly where they need a direction.  Instead
``distance_transform_edt(..., return_indices=True)`` gives, for every cell, the
NEAREST cell of the opposite class; the unit vector towards it is the exact
gradient of the Euclidean distance function, and it is non-zero everywhere
inside obstacles too.  Those two fields (a unit vector per cell) are bilinearly
interpolated like the value field, so ``grad`` is continuous and cheap.

The interpolated direction is RE-NORMALISED to unit length, because the exact
gradient of a distance function always has unit norm and the constraint
optimisers scale their step by it.  On the medial axis two opposite unit
vectors interpolate towards zero, where re-normalisation would amplify noise,
so below ``_DIR_FLOOR`` the nearest cell's direction is used instead (one extra
nearest-neighbour ``grid_sample``, only on the affected points).
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

__all__ = ["OccupancySDF", "build_occupancy_sdf"]

#: the occupancy iso-level that separates free from obstacle (same as
#: ``sampler._free_mask`` and ``dense_validation``).
OCCUPIED_LEVEL = 0.5

#: below this interpolated direction norm we are on (or next to) the medial
#: axis, where re-normalising would amplify cancellation noise; fall back to the
#: nearest cell's direction there.
_DIR_FLOOR = 0.2


class OccupancySDF:
    """Per-sample signed distance field of a batch of occupancy grids.

    Build once per evaluation chunk (≈5 ms/sample) and query thousands of times:
    a query is one ``grid_sample`` on a ``[B,1,H,W]`` buffer.
    """

    def __init__(self, occ: torch.Tensor, *, scene_to_meter: float = 80.0,
                 map_margin: float = 0.0,
                 dtype: torch.dtype = torch.float32) -> None:
        if occ.ndim != 4 or occ.shape[1] != 1:
            raise ValueError("occ must be [B,1,H,W], got %s" % (tuple(occ.shape),))
        self.shape = tuple(int(v) for v in occ.shape)
        self.batch = self.shape[0]
        self.height = self.shape[2]
        self.width = self.shape[3]
        self.cell = 2.0 / float(self.width)          # scene units per cell
        self.scene_to_meter = float(scene_to_meter)
        self.cell_m = self.cell * self.scene_to_meter
        self.map_margin = float(map_margin)
        self.dtype = dtype
        self.device = occ.device

        occ_np = occ.detach().to("cpu").to(torch.float32).numpy()
        sdf_np = np.empty_like(occ_np, dtype=np.float32)
        dir_np = np.empty((self.batch, 2, self.height, self.width), dtype=np.float32)
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        for b in range(self.batch):
            obstacle = occ_np[b, 0] > OCCUPIED_LEVEL
            # distance from every cell to the nearest cell of the other class,
            # plus the index of that cell -> exact distance-function gradient
            d_free, (iy_f, ix_f) = ndimage.distance_transform_edt(
                ~obstacle, return_indices=True)
            d_obs, (iy_o, ix_o) = ndimage.distance_transform_edt(
                obstacle, return_indices=True)
            sdf_np[b, 0] = np.where(~obstacle,
                                    (d_free - 0.5) * self.cell,
                                    -(d_obs - 0.5) * self.cell)
            # unit vector in the direction of INCREASING d_occ:
            #   free side     -> away from the nearest obstacle cell
            #   obstacle side -> towards the nearest free cell
            # (the returned EDT index is the nearest cell of the OTHER class, so
            #  the free side needs the negated difference; getting this backwards
            #  makes every gradient descent ASCEND the violation -- caught by
            #  tests/test_sdf_field.py::test_gradient_matches_finite_differences)
            vx = np.where(~obstacle, xx - ix_f, ix_o - xx).astype(np.float32)
            vy = np.where(~obstacle, yy - iy_f, iy_o - yy).astype(np.float32)
            norm = np.hypot(vx, vy)
            norm[norm < 1e-9] = 1.0
            dir_np[b, 0] = vx / norm                 # +x = column direction
            dir_np[b, 1] = vy / norm                 # +y = row direction

        self.sdf = torch.as_tensor(sdf_np, device=self.device, dtype=dtype)
        self.direction = torch.as_tensor(dir_np, device=self.device, dtype=dtype)

    # ------------------------------------------------------------------ query
    def _sample(self, field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """``field [B,K,H,W]``, ``points [B,N,2]`` -> ``[B,K,N]`` (bilinear)."""
        grid = points.to(field.dtype).unsqueeze(1)            # [B,1,N,2] = (x,y)
        out = F.grid_sample(field, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
        return out[:, :, 0, :]

    def box_field(self, points: torch.Tensor) -> torch.Tensor:
        """``d_box(points)`` -> ``[B,N]``: signed distance to the crop, scene units.

        Positive inside the inset box, negative outside.  For an axis-aligned box
        the distance to the boundary is ``min(1-m-|x|, 1-m-|y|)``.
        """
        inset = 1.0 - self.map_margin
        free_x = inset - points[..., 0].abs()
        free_y = inset - points[..., 1].abs()
        return torch.minimum(free_x, free_y)

    def _box_grad(self, points: torch.Tensor) -> torch.Tensor:
        """Exact gradient of :meth:`box_field` -> ``[B,N,2]``."""
        inset = 1.0 - self.map_margin
        free_x = inset - points[..., 0].abs()
        free_y = inset - points[..., 1].abs()
        use_x = free_x <= free_y
        gx = -torch.sign(points[..., 0])
        gy = -torch.sign(points[..., 1])
        zero = torch.zeros_like(gx)
        return torch.stack([torch.where(use_x, gx, zero),
                            torch.where(use_x, zero, gy)], dim=-1)

    def query(self, points: torch.Tensor, *, with_boundary: bool = False,
              with_grad: bool = True):
        """SDF value (and exact gradient) at ``points [B,N,2]``, scene units.

        ``with_boundary=False`` -> ``d_occ`` alone (the reported metric field).
        ``with_boundary=True``  -> ``min(d_occ, d_box)`` (the optimised field).
        Returns ``value [B,N]`` or ``(value, grad [B,N,2])``.
        """
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError("points must be [B,N,2], got %s" % (tuple(points.shape),))
        if points.shape[0] != self.batch:
            raise ValueError("expected %d samples, got %d"
                             % (self.batch, points.shape[0]))
        value = self._sample(self.sdf, points)[:, 0]
        if not with_grad:
            if with_boundary:
                value = torch.minimum(value, self.box_field(points))
            return value
        grad = self._sample(self.direction, points).permute(0, 2, 1)   # [B,N,2]
        grad = self._unit_direction(grad, points)
        if with_boundary:
            box_v = self.box_field(points)
            box_g = self._box_grad(points)
            use_box = (box_v < value).unsqueeze(-1)
            value = torch.where(use_box[..., 0], box_v, value)
            grad = torch.where(use_box, box_g, grad)
        return value, grad

    def _unit_direction(self, grad: torch.Tensor, points: torch.Tensor
                        ) -> torch.Tensor:
        """Re-normalise the interpolated direction; nearest cell on the medial axis."""
        norm = grad.norm(dim=-1, keepdim=True)
        unit = grad / norm.clamp_min(1e-12)
        weak = (norm < _DIR_FLOOR).squeeze(-1)                  # [B,N]
        if bool(weak.any()):
            fallback = self._sample_nearest(self.direction, points)
            fallback = fallback.permute(0, 2, 1)
            fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            unit = torch.where(weak.unsqueeze(-1), fallback, unit)
        return unit

    def _sample_nearest(self, field: torch.Tensor, points: torch.Tensor
                        ) -> torch.Tensor:
        grid = points.to(field.dtype).unsqueeze(1)
        out = F.grid_sample(field, grid, mode="nearest",
                            padding_mode="border", align_corners=False)
        return out[:, :, 0, :]

    # ---------------------------------------------------------------- metrics
    def stats(self, points: torch.Tensor, *, d_safe: float = 0.0,
              with_boundary: bool = False) -> dict:
        """Per-sample guide metrics of one trajectory, in SCENE units.

        ``r_n = max_h [d_safe - d_occ(p_h)]`` (guide 4.1.3 §2), its non-negative
        part ``V_n = max(0, r_n)`` (§3), ``D_min,n = min_h d_occ`` and the safe
        point ratio (§5).  ``feasible`` uses the guide's ``eps = 1e-3``.
        """
        value = self.query(points, with_boundary=with_boundary, with_grad=False)
        residual = float(d_safe) - value                       # [B,N]
        max_residual = residual.max(dim=1).values              # [B]
        violation = torch.relu(max_residual)                   # [B]
        min_distance = value.min(dim=1).values                 # [B]
        safe_rate = (value >= float(d_safe)).to(torch.float32).mean(dim=1)
        return {
            "max_residual": max_residual,
            "violation": violation,
            "mean_violation": torch.relu(residual).mean(dim=1),
            "min_distance": min_distance,
            "safe_point_rate": safe_rate,
            "feasible": violation <= 1e-3,
        }

    # ------------------------------------------------------------- conversions
    def to_metres(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scene_to_meter

    def select(self, index: int) -> "OccupancySDF":
        """A batch-1 view of sample ``index`` WITHOUT recomputing the EDT.

        The per-sample LP/QP path (``sdf_guidance._project_one``) needs a field
        whose batch matches the single polygon it projects; copying the view is
        free, re-running ``distance_transform_edt`` is not.
        """
        clone = copy.copy(self)
        clone.sdf = self.sdf[index:index + 1]
        clone.direction = self.direction[index:index + 1]
        clone.batch = 1
        return clone

    def __repr__(self) -> str:                                  # pragma: no cover
        return ("OccupancySDF(B=%d, %dx%d, cell=%.5f scene=%.4f m, m=%g, s2m=%g)"
                % (self.batch, self.height, self.width, self.cell, self.cell_m,
                   self.map_margin, self.scene_to_meter))


def build_occupancy_sdf(occ: torch.Tensor, *, scene_to_meter: float = 80.0,
                        map_margin: float = 0.0,
                        dtype: torch.dtype = torch.float32) -> OccupancySDF:
    """Convenience wrapper around :class:`OccupancySDF`."""
    return OccupancySDF(occ, scene_to_meter=scene_to_meter,
                        map_margin=map_margin, dtype=dtype)
