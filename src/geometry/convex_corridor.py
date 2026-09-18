"""Fast per-ellipse convex safe-region construction.

There is exactly one predicted ellipse and one convex region per trajectory
waypoint. For ellipse k, real obstacle-boundary points from a local map are
combined with a dense ring of points along that local map's outer border. The
local map and border ring are centred at the predicted physical ellipse centre.
All points then go through the Neural-IRIS Mahalanobis ordering and greedy
filtering mechanism.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


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


class EllipseRegionBuilder:
    """Return one padded ``A x <= b`` safe region for every predicted ellipse.

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

    def __call__(self, p0: torch.Tensor, e0: torch.Tensor,
                 return_diagnostics: bool = False):
        if p0.ndim != 3 or p0.shape[-1] != 2:
            raise ValueError("p0 must have shape [B,H,2]")
        if e0.shape[:2] != p0.shape[:2] or e0.shape[-1] < 6:
            raise ValueError("e0 must have shape [B,H,6]")
        batch, horizon, _ = p0.shape
        dtype, device = p0.dtype, p0.device
        region_complete = torch.ones(batch, horizon, dtype=torch.bool, device=device)

        clean_e = torch.nan_to_num(e0, nan=0.0, posinf=0.0, neginf=0.0)
        center_all = p0 + clean_e[..., :2]
        axes = clean_e[..., 2:4].clamp(-6.0, 0.7).exp()
        axes = axes.clamp(self.axis_min, self.axis_max)
        theta = 0.5 * torch.atan2(clean_e[..., 5], clean_e[..., 4])
        u = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        v = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=-1)
        quadratic_all = (
            u[..., :, None] * u[..., None, :] /
            axes[..., 0, None, None].square() +
            v[..., :, None] * v[..., None, :] /
            axes[..., 1, None, None].square()
        )

        group_results = []
        max_faces = 0
        for batch_indices, obstacle_points in self._groups:
            centers = center_all[batch_indices].reshape(-1, 2)
            quadratics = quadratic_all[batch_indices].reshape(-1, 2, 2)
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
                max_faces = max(max_faces, chunk_A.shape[1])
                chunk_results.append((begin, end, chunk_A, chunk_b, chunk_mask))
            group_results.append((batch_indices, len(centers), chunk_results))

        A = torch.zeros(batch, horizon, max_faces, 2, dtype=dtype, device=device)
        b = torch.zeros(batch, horizon, max_faces, dtype=dtype, device=device)
        face_mask = torch.zeros(batch, horizon, max_faces,
                                dtype=torch.bool, device=device)
        for batch_indices, count, chunk_results in group_results:
            local_A = p0.new_zeros((count, max_faces, 2))
            local_b = p0.new_zeros((count, max_faces))
            local_mask = torch.zeros((count, max_faces), dtype=torch.bool, device=device)
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
        center_violation = (A * center_all[:, :, None]).sum(dim=-1) - b
        max_center_violation = center_violation.masked_fill(
            ~face_mask, -torch.inf).max(dim=-1).values
        center_inside = max_center_violation <= 1e-5
        center_finite = torch.isfinite(center_all).all(dim=-1)
        quadratic_finite = torch.isfinite(quadratic_all).all(dim=(-1, -2))
        face_count = face_mask.sum(dim=-1)
        bounded = _halfspaces_are_bounded(A, face_mask)
        # Diagnostic only: face_count counts generated separator rows, not the
        # visible edge count after redundant halfspaces are removed. Neither it
        # nor center_inside determines validity.
        valid = center_finite & quadratic_finite & region_complete & bounded
        if not return_diagnostics:
            return A, b, face_mask, valid
        diagnostics = {
            "center": center_all,
            "center_finite": center_finite,
            "quadratic_finite": quadratic_finite,
            "region_complete": region_complete,
            "center_inside": center_inside,
            "face_count": face_count,
            "bounded": bounded,
            "max_center_violation": max_center_violation,
        }
        return A, b, face_mask, valid, diagnostics


# Backward-compatible import name for callers outside the current sampler.
SegmentCorridorBuilder = EllipseRegionBuilder
