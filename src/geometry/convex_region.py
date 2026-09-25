"""Convex safe regions for the predicted ellipses.

Two related utilities live here:

* :class:`EllipseRegionBuilder` — the fast, batched local convex-region builder
  used by the corridor construction.  It works on an **absolute centre** plus an
  anisotropy metric:

      ``build_from_metric(center, quadratic)``   <- the core primitive
      ``build_from_ellipse(center, shape4)``     <- ellipse  ->  metric wrapper

  Real obstacle-boundary points from a local map are combined with a dense ring
  of points along that local map's outer border, and all points go through the
  Neural-IRIS Mahalanobis ordering and greedy filtering mechanism.

  The centre is ABSOLUTE (scene coordinates).  The old ``center = p0 + delta``
  semantics is GONE: the ellipse centres are the fixed Skeleton progress points
  ``c_i = Gamma_m(i/(Q-1))`` and the head only predicts the shape (there is no
  centre head).

* :func:`halfspaces_to_vertices` — convert ``A x <= b`` into polygon vertices
  (used for the overlap check and for drawing).

Unlike Neural-IRIS, the ellipse here is ``p = P u + c`` with ``||u|| <= 1``, so
the quadratic form used for the metric ordering is ``Q = P^{-2}``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull, HalfspaceIntersection

_ELLIPSE_LOG_CLAMP = (-6.0, 0.7)


def _obstacle_boundary_points(occ: torch.Tensor, dilation: int = 0) -> torch.Tensor:
    """Return obstacle boundary cell centres in scene ``[-1,1]^2``."""
    wall = occ > 0.5
    if dilation > 0:
        k = 2 * int(dilation) + 1
        wall = F.max_pool2d(
            wall[None, None].float(), kernel_size=k, stride=1,
            padding=int(dilation),
        )[0, 0] > 0.5
    up = torch.zeros_like(wall)
    down = torch.zeros_like(wall)
    left = torch.zeros_like(wall)
    right = torch.zeros_like(wall)
    up[1:] = wall[:-1]
    down[:-1] = wall[1:]
    left[:, 1:] = wall[:, :-1]
    right[:, :-1] = wall[:, 1:]
    boundary = wall & ~(up & down & left & right)
    ij = boundary.nonzero(as_tuple=False)
    if ij.numel() == 0:
        return occ.new_empty((0, 2))
    h, w = wall.shape
    x = -1.0 + (ij[:, 1].to(occ.dtype) + 0.5) * (2.0 / float(w))
    y = -1.0 + (ij[:, 0].to(occ.dtype) + 0.5) * (2.0 / float(h))
    return torch.stack((x, y), dim=-1)


def _halfspaces_are_bounded(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return whether 2-D halfspace normals positively span every direction.

    A 2-D intersection is bounded exactly when its active normal angles are not
    contained in any closed semicircle. Equivalently, every circular gap
    between consecutive normal angles is strictly smaller than pi. Three faces
    are sufficient when they form a triangle; there is no four-face rule.
    """
    if A.shape[-2] < 2:
        return torch.zeros(mask.shape[:-1], dtype=torch.bool, device=mask.device)
    count = mask.sum(dim=-1)
    angle = torch.atan2(A[..., 1], A[..., 0]).remainder(2.0 * torch.pi)
    ordered = angle.masked_fill(~mask, torch.inf).sort(dim=-1).values
    positions = torch.arange(A.shape[-2], device=A.device)
    adjacent_valid = positions[:-1] < (count[..., None] - 1)
    adjacent_gap = (ordered[..., 1:] - ordered[..., :-1]).masked_fill(
        ~adjacent_valid, -torch.inf)
    last_index = (count - 1).clamp_min(0)[..., None]
    last = ordered.gather(-1, last_index).squeeze(-1)
    wrap_gap = ordered[..., 0] + 2.0 * torch.pi - last
    max_gap = torch.maximum(adjacent_gap.max(dim=-1).values, wrap_gap)
    return (count >= 3) & torch.isfinite(max_gap) & (max_gap < torch.pi - 1e-6)


def ellipse_shape4_to_quadratic(shape4: torch.Tensor, axis_min: float = 2e-3,
                                axis_max: float = 2.0) -> torch.Tensor:
    """``[B,H,4] = [log a, log b, cos 2t, sin 2t]`` -> metric ``[B,H,2,2]``.

    ``a = exp(log a)``, ``b = exp(log b)`` and

        Q = R(theta) diag(1/a^2, 1/b^2) R(theta)^T,  theta = atan2(sin2t, cos2t)/2
    """
    if shape4.shape[-1] < 4:
        raise ValueError("shape4 must have shape [...,4]")
    clean = torch.nan_to_num(shape4, nan=0.0, posinf=0.0, neginf=0.0)
    axes = clean[..., 0:2].clamp(*_ELLIPSE_LOG_CLAMP).exp()
    axes = axes.clamp(float(axis_min), float(axis_max))
    theta = 0.5 * torch.atan2(clean[..., 3], clean[..., 2])
    u = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    v = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=-1)
    return (u[..., :, None] * u[..., None, :] /
            axes[..., 0, None, None].square() +
            v[..., :, None] * v[..., None, :] /
            axes[..., 1, None, None].square())


class EllipseRegionBuilder:
    """Return one padded ``A x <= b`` convex region for every query point.

    The centre is ABSOLUTE.  Two entry points exist:

    * :meth:`build_from_ellipse` (absolute centre + ``shape4``)
    * :meth:`build_from_metric` (absolute centre + quadratic metric)

    Shapes are ``A[B,H,M,2]``, ``b[B,H,M]``, ``mask[B,H,M]`` and
    ``valid[B,H]``. ``M`` is the largest number of faces actually generated in
    this call; shorter regions are padded. Obstacles and faces are never capped.
    """

    def __init__(self, map_tensor: torch.Tensor, config: dict | None = None):
        cfg = config or {}
        if map_tensor.ndim != 4 or map_tensor.shape[1] != 1:
            raise ValueError("map_tensor must have shape [B,1,H,W]")
        self.maps = map_tensor.detach()
        self.margin = max(0.0, float(cfg.get("safety_margin", 0.02)))
        self.window_half = max(
            self.margin + 1e-4, float(cfg.get("obstacle_window_half", 0.35)))
        self.dilation = max(0, int(cfg.get("obstacle_dilation", 0)))
        self.chunk_size = max(1, int(cfg.get("corridor_chunk_size", 512)))
        self.axis_min = max(1e-5, float(cfg.get("ellipse_axis_min", 2e-3)))
        self.axis_max = max(self.axis_min, float(cfg.get("ellipse_axis_max", 2.0)))
        map_scale = 2.0 / float(max(map_tensor.shape[-2:]))
        # Neural-IRIS uses 1e-3 in patch pixels. The default below is the same
        # tolerance converted to scene coordinates.
        self.filter_eps = max(
            0.0, float(cfg.get("filter_eps", 1e-3 * map_scale)))
        self.guidance_dilation = max(0, int(cfg.get("guidance_dilation_cells", 1)))
        self.guidance_threshold = float(cfg.get("guidance_occupancy_threshold", 1e-3))
        # --- map-boundary faces -------------------------------------------------
        # Every cell is intersected with the scene box [-1,1]^2 (inset by
        # ``map_boundary_margin``), i.e. four extra halfspaces
        #     ±x <= 1 - margin,  ±y <= 1 - margin.
        # WHY: the local window ring (``obstacle_window_half`` = 0.35 scene) and
        # the obstacle-cut faces never mention the crop edge, so near the border
        # a cell could extend up to 0.28 scene (22 m) OUTSIDE the 256^2 window.
        # The ALM then happily certifies a curve that leaves the map, while the
        # collision metric (``_free_mask``: ``|p| <= 1``) counts leaving the map
        # as a collision -- the constraint set and the metric disagreed, and the
        # optimiser had no reason to touch those samples at all.  Measured on the
        # C=48 k=4 run: 6 of the 9 test collisions were pure crop excursions
        # (0.17-2.0 m outside), with the pack reporting ``violation <= 0`` and
        # `1` inner iteration.  See docs/CAMPAIGN_160K8P_RESULTS.md section 12.
        #
        # ``map_boundary_margin`` insets the box: a projection lands exactly ON
        # the constraint boundary, so a face at |x| = 1 would put the curve at
        # the very edge where float noise decides the collision.  Defaults to the
        # corridor's own ``safety_margin``.
        self.map_boundary = bool(cfg.get("map_boundary_faces", True))
        self.map_boundary_margin = max(
            0.0, float(cfg.get("map_boundary_margin", self.margin)))
        self.segment_collision_samples = max(
            2, int(cfg.get("segment_collision_samples", 8)))
        if self.guidance_dilation > 0:
            k = 2 * self.guidance_dilation + 1
            self.guidance_maps = F.max_pool2d(
                self.maps.float(), kernel_size=k, stride=1,
                padding=self.guidance_dilation,
            ).to(self.maps.dtype)
        else:
            self.guidance_maps = self.maps
        self._groups = self._prepare_map_groups()

    def waypoint_needs_guidance(self, p: torch.Tensor) -> torch.Tensor:
        """Physical collision/clearance gate for ALM, shape ``[B,H]``."""
        grid = p[:, :, None, :]
        occupancy = F.grid_sample(
            self.guidance_maps.to(p.dtype), grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )[:, 0, :, 0]
        outside = (p.abs() > 1.0).any(dim=-1)
        return (occupancy > self.guidance_threshold) | outside

    def segment_needs_guidance(self, p: torch.Tensor) -> torch.Tensor:
        """Collision/clearance gate for incoming segments ``[p[i-1],p[i]]``."""
        alpha = torch.linspace(
            0.0, 1.0, self.segment_collision_samples,
            dtype=p.dtype, device=p.device,
        )
        samples = (p[:, :-1, None] * (1.0 - alpha[None, None, :, None]) +
                   p[:, 1:, None] * alpha[None, None, :, None])
        occupancy = F.grid_sample(
            self.guidance_maps.to(p.dtype), samples, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )[:, 0]
        outside = (samples.abs() > 1.0).any(dim=-1)
        return (occupancy > self.guidance_threshold).any(dim=-1) | outside.any(dim=-1)

    def _prepare_map_groups(self):
        """Group identical maps and cache their boundary points once."""
        batch = self.maps.shape[0]
        if batch == 0:
            return []
        first = self.maps[0, 0]
        if batch == 1 or torch.equal(self.maps[:, 0], first.expand_as(self.maps[:, 0])):
            groups = [(torch.arange(batch, device=self.maps.device), first)]
        else:
            pending = list(range(batch))
            groups = []
            while pending:
                root = pending.pop(0)
                same, rest = [root], []
                for i in pending:
                    (same if torch.equal(self.maps[i, 0], self.maps[root, 0])
                     else rest).append(i)
                pending = rest
                groups.append((torch.tensor(same, device=self.maps.device),
                               self.maps[root, 0]))
        out = []
        for indices, occ in groups:
            points = _obstacle_boundary_points(occ, self.dilation)
            out.append((indices, points))
        return out

    def _map_boundary_faces(self, count: int, dtype, device):
        """``±x <= 1-m``, ``±y <= 1-m`` broadcast to ``count`` cells.

        The four halfspaces that keep a cell inside the 256^2 scene window, so
        the ALM can SEE "leaving the map" and the pack agrees with the collision
        metric on it.  See the comment in ``__init__``.
        """
        ins = 1.0 - self.map_boundary_margin
        normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                               dtype=dtype, device=device)
        A = normals[None].expand(int(count), -1, -1).contiguous()
        b = torch.full((int(count), 4), ins, dtype=dtype, device=device)
        mask = torch.ones((int(count), 4), dtype=torch.bool, device=device)
        return A, b, mask

    def _local_border_points(self, centers: torch.Tensor) -> torch.Tensor:
        """Dense obstacle-point ring around each ellipse-centred local map."""
        height, width = self.maps.shape[-2:]
        nx = max(2, int(round(2.0 * self.window_half / (2.0 / width))) + 1)
        ny = max(2, int(round(2.0 * self.window_half / (2.0 / height))) + 1)
        x = torch.linspace(-self.window_half, self.window_half, nx,
                           dtype=centers.dtype, device=centers.device)
        y = torch.linspace(-self.window_half, self.window_half, ny,
                           dtype=centers.dtype, device=centers.device)
        top = torch.stack((x, torch.full_like(x, self.window_half)), dim=-1)
        bottom = torch.stack((x, torch.full_like(x, -self.window_half)), dim=-1)
        if ny > 2:
            side_y = y[1:-1]
            right = torch.stack(
                (torch.full_like(side_y, self.window_half), side_y), dim=-1)
            left = torch.stack(
                (torch.full_like(side_y, -self.window_half), side_y), dim=-1)
            offsets = torch.cat((top, right, bottom.flip(0), left.flip(0)), dim=0)
        else:
            offsets = torch.cat((top, bottom.flip(0)), dim=0)
        return centers[:, None, :] + offsets[None, :, :]

    def _greedy_faces(self, centers: torch.Tensor, quadratics: torch.Tensor,
                      points: torch.Tensor, initial_active: torch.Tensor):
        """Neural-IRIS one-time metric sort followed by active-point filtering."""
        size = len(centers)
        if points.shape[1] == 0 or not bool(initial_active.any()):
            return (centers.new_zeros((size, 0, 2)),
                    centers.new_zeros((size, 0)),
                    torch.zeros((size, 0), dtype=torch.bool,
                                device=centers.device))
        diff = points - centers[:, None]
        metric = torch.einsum("snd,sde,sne->sn", diff, quadratics, diff)
        order = torch.argsort(metric, dim=1, stable=True)
        active = torch.gather(initial_active, 1, order)
        rows = torch.arange(size, device=centers.device)
        generated_A, generated_b, generated_mask = [], [], []

        while bool(active.any()):
            has_point = active.any(dim=1)
            first = active.to(torch.int8).argmax(dim=1)
            nearest_idx = order[rows, first]
            obs = points[rows, nearest_idx]
            obs_diff = obs - centers
            normal = torch.einsum("sde,se->sd", quadratics, obs_diff)
            norm = normal.norm(dim=-1)
            has_face = has_point & torch.isfinite(norm) & (norm > 1e-7)
            normal = normal / norm.clamp_min(1e-7)[:, None]
            rhs = (normal * obs).sum(dim=-1) - self.margin
            generated_A.append(torch.where(
                has_face[:, None], normal, torch.zeros_like(normal)))
            generated_b.append(torch.where(
                has_face, rhs, torch.zeros_like(rhs)))
            generated_mask.append(has_face)

            projection = torch.einsum("sd,snd->sn", normal, points)
            keep = torch.gather(
                projection <= rhs[:, None] + self.filter_eps, 1, order)
            active &= keep | ~has_face[:, None]
            active[rows[has_point], first[has_point]] = False

        return (torch.stack(generated_A, dim=1),
                torch.stack(generated_b, dim=1),
                torch.stack(generated_mask, dim=1))

    def build_from_ellipse(self, center: torch.Tensor, shape4: torch.Tensor,
                           return_diagnostics: bool = False):
        """ABSOLUTE ellipse centre + shape4 -> padded ``A x <= b`` regions.

        ``center [B,H,2]`` is already in scene coordinates; there is no
        ``p0 + delta`` offset any more.  ``shape4 [B,H,4]`` is
        ``[log a, log b, cos 2theta, sin 2theta]``.
        """
        if shape4.shape[:2] != center.shape[:2] or shape4.shape[-1] < 4:
            raise ValueError("shape4 must have shape [B,H,4]")
        quadratic = ellipse_shape4_to_quadratic(
            shape4, self.axis_min, self.axis_max)
        return self.build_from_metric(center, quadratic,
                                      return_diagnostics=return_diagnostics)

    def build_from_metric(self, center: torch.Tensor, quadratic: torch.Tensor,
                          return_diagnostics: bool = False):
        """Core primitive: absolute centre + anisotropic metric -> regions.

        ``center [B,H,2]``, ``quadratic [B,H,2,2]``.  The metric is what the
        Mahalanobis ordering uses; an isotropic metric (identity) therefore
        recovers a point-seeded Euclidean region, which is exactly what the
        gap-bridge uses.
        """
        if center.ndim != 3 or center.shape[-1] != 2:
            raise ValueError("center must have shape [B,H,2]")
        if tuple(quadratic.shape) != tuple(center.shape[:-1]) + (2, 2):
            raise ValueError("quadratic must have shape [B,H,2,2]")
        batch, horizon, _ = center.shape
        dtype, device = center.dtype, center.device
        region_complete = torch.ones(batch, horizon, dtype=torch.bool,
                                     device=device)

        center_finite = torch.isfinite(center).all(dim=-1)
        quadratic_finite = torch.isfinite(quadratic).all(dim=(-1, -2))
        clean_center = torch.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
        clean_quadratic = torch.nan_to_num(quadratic, nan=0.0, posinf=0.0,
                                           neginf=0.0)

        group_results = []
        max_faces = 0
        for batch_indices, obstacle_points in self._groups:
            centers = clean_center[batch_indices].reshape(-1, 2)
            quadratics = clean_quadratic[batch_indices].reshape(-1, 2, 2)
            chunk_results = []

            for begin in range(0, len(centers), self.chunk_size):
                end = min(begin + self.chunk_size, len(centers))
                c = centers[begin:end]
                quadratic = quadratics[begin:end]
                border_points = self._local_border_points(c)
                border_active = torch.ones(
                    len(c), border_points.shape[1], dtype=torch.bool, device=device)
                border_A, border_b, border_mask = self._greedy_faces(
                    c, quadratic, border_points, border_active)

                if obstacle_points.numel() != 0:
                    real_points = obstacle_points[None].expand(len(c), -1, -1)
                    real_diff = real_points - c[:, None]
                    in_local_map = (real_diff.abs() <= self.window_half).all(dim=-1)
                    real_A, real_b, real_mask = self._greedy_faces(
                        c, quadratic, real_points, in_local_map)
                else:
                    real_A = c.new_zeros((len(c), 0, 2))
                    real_b = c.new_zeros((len(c), 0))
                    real_mask = torch.zeros(
                        (len(c), 0), dtype=torch.bool, device=device)

                # Border faces are generated independently so real-obstacle
                # filtering cannot discard the boundary ring before it closes
                # every recession direction.
                chunk_A = torch.cat((real_A, border_A), dim=1)
                chunk_b = torch.cat((real_b, border_b), dim=1)
                chunk_mask = torch.cat((real_mask, border_mask), dim=1)
                if self.map_boundary:
                    box_A, box_b, box_mask = self._map_boundary_faces(
                        len(c), dtype, device)
                    chunk_A = torch.cat((chunk_A, box_A), dim=1)
                    chunk_b = torch.cat((chunk_b, box_b), dim=1)
                    chunk_mask = torch.cat((chunk_mask, box_mask), dim=1)
                max_faces = max(max_faces, chunk_A.shape[1])
                chunk_results.append((begin, end, chunk_A, chunk_b, chunk_mask))
            group_results.append((batch_indices, len(centers), chunk_results))

        A = torch.zeros(batch, horizon, max_faces, 2, dtype=dtype, device=device)
        b = torch.zeros(batch, horizon, max_faces, dtype=dtype, device=device)
        face_mask = torch.zeros(batch, horizon, max_faces,
                                dtype=torch.bool, device=device)
        for batch_indices, count, chunk_results in group_results:
            local_A = clean_center.new_zeros((count, max_faces, 2))
            local_b = clean_center.new_zeros((count, max_faces))
            local_mask = torch.zeros((count, max_faces), dtype=torch.bool,
                                     device=device)
            for begin, end, chunk_A, chunk_b, chunk_mask in chunk_results:
                faces = chunk_A.shape[1]
                local_A[begin:end, :faces] = chunk_A
                local_b[begin:end, :faces] = chunk_b
                local_mask[begin:end, :faces] = chunk_mask
            A[batch_indices] = local_A.reshape(
                len(batch_indices), horizon, max_faces, 2)
            b[batch_indices] = local_b.reshape(
                len(batch_indices), horizon, max_faces)
            face_mask[batch_indices] = local_mask.reshape(
                len(batch_indices), horizon, max_faces)

        finite_faces = torch.isfinite(A).all(dim=-1) & torch.isfinite(b)
        face_mask &= finite_faces
        center_violation = (A * clean_center[:, :, None]).sum(dim=-1) - b
        max_center_violation = center_violation.masked_fill(
            ~face_mask, -torch.inf).max(dim=-1).values
        # ``A x <= b`` feasibility of the centre itself.  A region that does not
        # contain its own anchor is never allowed into the corridor.
        center_inside = (max_center_violation <= 1e-5) & torch.isfinite(
            max_center_violation)
        face_count = face_mask.sum(dim=-1)
        bounded = _halfspaces_are_bounded(A, face_mask)
        # face_count only counts generated separator rows (not the visible edge
        # count) and therefore stays diagnostic-only.  ``center_inside`` IS part
        # of the validity contract (report section 4.3).
        # ``center_inside`` is NO LONGER part of the validity contract (removed
        # on request).  Measured reason: a route that hugs an obstacle puts an
        # ellipse centre closer to the wall than ``safety_margin``, the cut face
        # is then placed BEHIND that centre, and the cell was rejected -- and
        # because ONE rejected base cell voids the WHOLE corridor
        # (``invalid_base_region:113``), such samples ended with an empty pack
        # and an ALM that never ran (test_0342 of the C=48 k=4 run: 9/128 cells
        # rejected, corridor EMPTY).  The flag is still computed and reported in
        # the diagnostics; it just does not invalidate the region any more.
        valid = (center_finite & quadratic_finite & region_complete & bounded)
        if not return_diagnostics:
            return A, b, face_mask, valid
        diagnostics = {
            "center": clean_center,
            "center_finite": center_finite,
            "quadratic_finite": quadratic_finite,
            "region_complete": region_complete,
            "center_inside": center_inside,
            "face_count": face_count,
            "bounded": bounded,
            "max_center_violation": max_center_violation,
        }
        return A, b, face_mask, valid, diagnostics

    # deprecated shim: the old (p0 + delta, 6-vector) semantics is gone.
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "EllipseRegionBuilder(center, shape4) 的旧 delta-centre 语义已删除；"
            "请使用 build_from_ellipse(center, shape4) 或 "
            "build_from_metric(center, quadratic)")


def halfspaces_to_vertices(
    A: np.ndarray,
    b: np.ndarray,
    interior_point: np.ndarray | None = None,
) -> np.ndarray | None:
    """Convert A x <= b to counter-clockwise polygon vertices.

    Uses scipy.spatial.HalfspaceIntersection with interior_point as a
    strictly-feasible point (the ellipse centre usually).  Returns None if the
    polytope is empty, degenerate, or the intersection fails.
    """
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(A) < 3:
        return None

    # Drop redundant/num-degenerate rows.
    norms = np.linalg.norm(A, axis=1)
    keep = norms > 1e-8
    if not np.any(keep):
        return None
    A = A[keep]
    b = b[keep]
    if len(A) < 3:
        return None

    if interior_point is None:
        interior_point = np.zeros(2, dtype=float)
    interior_point = np.asarray(interior_point, dtype=float).reshape(2)
    # scipy REQUIRES a strictly feasible point.  The ellipse centre is not one
    # any more for cells whose faces were cut behind it (see the ``valid`` note
    # in ``build_from_metric``), so fall back to interior-point-free enumeration.
    if float((A @ interior_point - b).max()) > 1e-9:
        return _vertices_by_pair_enumeration(A, b)

    # scipy convention: [A, -b] represents A x - b <= 0  (i.e. A x <= b).
    halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        intersections = np.asarray(hs.intersections, dtype=float)
        if len(intersections) < 3:
            return _vertices_by_pair_enumeration(A, b)
        hull = ConvexHull(intersections)
        return intersections[hull.vertices]
    except Exception:
        return _vertices_by_pair_enumeration(A, b)


def _vertices_by_pair_enumeration(A: np.ndarray, b: np.ndarray,
                                  tol: float = 1e-6):
    """Vertices of a BOUNDED ``{x : A x <= b}`` with no interior point.

    Every vertex of a bounded 2-D polytope is the intersection of two faces, so
    enumerating the pairs and keeping the feasible ones is exact; the convex
    hull of the result gives the polygon in counter-clockwise order.  Returns
    None if fewer than three vertices survive (empty/degenerate polytope).
    """
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1)
    pts = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            det = A[i, 0] * A[j, 1] - A[i, 1] * A[j, 0]
            if abs(det) < 1e-12:
                continue                       # parallel faces
            p = np.array([(b[i] * A[j, 1] - A[i, 1] * b[j]) / det,
                          (A[i, 0] * b[j] - b[i] * A[j, 0]) / det])
            if (A @ p - b).max() <= tol:
                pts.append(p)
    if len(pts) < 3:
        return None
    keep = []
    for q in pts:                              # merge repeat vertices
        if not keep or np.min(np.linalg.norm(np.asarray(keep) - q, axis=1)) > 1e-7:
            keep.append(q)
    if len(keep) < 3:
        return None
    pts = np.asarray(keep)
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:
        return None
