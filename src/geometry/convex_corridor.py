"""Fast per-ellipse convex safe-region construction.

There is exactly one predicted ellipse and one convex region per trajectory
waypoint. For ellipse k, obstacle boundary points are collected in a local box
centred at its physical centre and separated with the same greedy geometry as
Neural-IRIS. All ellipses in a chunk run in parallel on the sampling device.
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


class EllipseRegionBuilder:
    """Return one padded ``A x <= b`` safe region for every predicted ellipse.

    Shapes are ``A[B,H,M,2]``, ``b[B,H,M]``, ``mask[B,H,M]`` and
    ``valid[B,H]``. Four faces bound the local obstacle-query box; remaining
    faces are the Neural-IRIS greedy obstacle separators.
    """

    def __init__(self, map_tensor: torch.Tensor, config: dict | None = None):
        cfg = config or {}
        if map_tensor.ndim != 4 or map_tensor.shape[1] != 1:
            raise ValueError("map_tensor must have shape [B,1,H,W]")
        self.maps = map_tensor.detach()
        self.max_faces = max(4, int(cfg.get("max_faces", 20)))
        self.margin = max(0.0, float(cfg.get("safety_margin", 0.02)))
        self.window_half = max(
            self.margin + 1e-4, float(cfg.get("obstacle_window_half", 0.35)))
        self.dilation = max(0, int(cfg.get("obstacle_dilation", 0)))
        self.chunk_size = max(1, int(cfg.get("corridor_chunk_size", 512)))
        self.axis_min = max(1e-5, float(cfg.get("ellipse_axis_min", 2e-3)))
        self.axis_max = max(self.axis_min, float(cfg.get("ellipse_axis_max", 2.0)))
        self.dedup_eps = max(0.0, float(cfg.get("dedup_eps", 1e-6)))
        self.max_obstacle_points = max(0, int(cfg.get("max_obstacle_points", 4096)))
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
            if self.max_obstacle_points and len(points) > self.max_obstacle_points:
                pick = torch.linspace(
                    0, len(points) - 1, self.max_obstacle_points,
                    device=points.device,
                ).long()
                points = points[pick]
            out.append((indices, points))
        return out

    def __call__(self, p0: torch.Tensor, e0: torch.Tensor):
        if p0.ndim != 3 or p0.shape[-1] != 2:
            raise ValueError("p0 must have shape [B,H,2]")
        if e0.shape[:2] != p0.shape[:2] or e0.shape[-1] < 6:
            raise ValueError("e0 must have shape [B,H,6]")
        batch, horizon, _ = p0.shape
        dtype, device = p0.dtype, p0.device
        A = torch.zeros(batch, horizon, self.max_faces, 2,
                        dtype=dtype, device=device)
        b = torch.zeros(batch, horizon, self.max_faces,
                        dtype=dtype, device=device)
        face_mask = torch.zeros(batch, horizon, self.max_faces,
                                dtype=torch.bool, device=device)
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

        # Bound every region by exactly the local box whose obstacles we query.
        xmin = (center_all[..., 0] - self.window_half).clamp_min(-1.0 + self.margin)
        xmax = (center_all[..., 0] + self.window_half).clamp_max(1.0 - self.margin)
        ymin = (center_all[..., 1] - self.window_half).clamp_min(-1.0 + self.margin)
        ymax = (center_all[..., 1] + self.window_half).clamp_max(1.0 - self.margin)
        box_A = p0.new_tensor(((1.0, 0.0), (-1.0, 0.0),
                              (0.0, 1.0), (0.0, -1.0)))
        A[:, :, :4] = box_A
        b[:, :, 0] = xmax
        b[:, :, 1] = -xmin
        b[:, :, 2] = ymax
        b[:, :, 3] = -ymin
        face_mask[:, :, :4] = True

        obstacle_slots = self.max_faces - 4
        for batch_indices, obstacle_points in self._groups:
            if obstacle_points.numel() == 0:
                continue
            centers = center_all[batch_indices].reshape(-1, 2)
            quadratics = quadratic_all[batch_indices].reshape(-1, 2, 2)
            local_A = A[batch_indices].reshape(-1, self.max_faces, 2).clone()
            local_b = b[batch_indices].reshape(-1, self.max_faces).clone()
            local_mask = face_mask[batch_indices].reshape(-1, self.max_faces).clone()
            local_complete = torch.zeros(len(centers), dtype=torch.bool, device=device)

            for begin in range(0, len(centers), self.chunk_size):
                end = min(begin + self.chunk_size, len(centers))
                c = centers[begin:end]
                quadratic = quadratics[begin:end]
                diff = obstacle_points[None] - c[:, None]
                metric = torch.einsum("snd,sde,sne->sn", diff, quadratic, diff)
                # Only points inside the bounded local query box matter.
                active = (diff.abs() <= self.window_half + self.margin).all(dim=-1)

                for slot in range(obstacle_slots):
                    ranked = metric.masked_fill(~active, torch.inf)
                    nearest_value, nearest_idx = ranked.min(dim=1)
                    has_face = torch.isfinite(nearest_value)
                    if not bool(has_face.any()):
                        break
                    row = torch.arange(end - begin, device=device)
                    obs = obstacle_points[nearest_idx]
                    obs_diff = obs - c
                    normal = torch.einsum("sde,se->sd", quadratic, obs_diff)
                    norm = normal.norm(dim=-1)
                    has_face &= torch.isfinite(norm) & (norm > 1e-7)
                    normal = normal / norm.clamp_min(1e-7)[:, None]
                    rhs = (normal * obs).sum(dim=-1) - self.margin
                    face = 4 + slot
                    local_A[begin:end, face] = torch.where(
                        has_face[:, None], normal, local_A[begin:end, face])
                    local_b[begin:end, face] = torch.where(
                        has_face, rhs, local_b[begin:end, face])
                    local_mask[begin:end, face] = has_face

                    projection = torch.einsum("sd,nd->sn", normal, obstacle_points)
                    keep = projection <= rhs[:, None] + self.dedup_eps
                    active &= keep | ~has_face[:, None]
                    active[row[has_face], nearest_idx[has_face]] = False
                local_complete[begin:end] = ~active.any(dim=1)

            A[batch_indices] = local_A.reshape(
                len(batch_indices), horizon, self.max_faces, 2)
            b[batch_indices] = local_b.reshape(
                len(batch_indices), horizon, self.max_faces)
            face_mask[batch_indices] = local_mask.reshape(
                len(batch_indices), horizon, self.max_faces)
            region_complete[batch_indices] = local_complete.reshape(
                len(batch_indices), horizon)

        finite_faces = torch.isfinite(A).all(dim=-1) & torch.isfinite(b)
        face_mask &= finite_faces
        center_violation = (A * center_all[:, :, None]).sum(dim=-1) - b
        center_inside = (center_violation.masked_fill(~face_mask, -torch.inf)
                         .max(dim=-1).values <= 1e-5)
        valid = (torch.isfinite(center_all).all(dim=-1) &
                 torch.isfinite(quadratic_all).all(dim=(-1, -2)) &
                 region_complete & center_inside &
                 (face_mask.sum(dim=-1) >= 4))
        return A, b, face_mask, valid


# Backward-compatible import name for callers outside the current sampler.
SegmentCorridorBuilder = EllipseRegionBuilder
