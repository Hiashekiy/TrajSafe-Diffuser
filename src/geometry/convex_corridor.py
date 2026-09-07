"""Batched Neural-IRIS-style convex corridors for inference-time guidance.

The builder keeps the reference Neural-IRIS geometry: obstacle boundary points
are ordered in the predicted ellipse metric, the nearest active obstacle adds a
separating plane, and points already separated by that plane are discarded.

Unlike the NumPy reference, all trajectory segments in a chunk are processed
in parallel on the sampling device.  Occupancy boundaries are extracted only
once when the builder is created and reused across every reverse diffusion
step.  The only Python loop is the small, fixed number of requested faces.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _obstacle_boundary_points(occ: torch.Tensor, dilation: int = 0) -> torch.Tensor:
    """Return obstacle boundary cell centres in scene ``[-1, 1]^2``.

    ``occ`` is ``[H,W]`` with one denoting an obstacle.  The row axis is the
    scene y axis (the dataset is displayed with ``origin='lower'``).
    """
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


class SegmentCorridorBuilder:
    """Build padded ``A x <= b`` corridors for all trajectory segments.

    Output shapes are ``A[B,H-1,M,2]``, ``b[B,H-1,M]``, a face mask with the
    same first four dimensions as ``b``, and ``valid[B,H-1]``.  Four scene-box
    faces are always present; the remaining faces come from obstacles.
    """

    def __init__(self, map_tensor: torch.Tensor, config: dict | None = None):
        cfg = config or {}
        if map_tensor.ndim != 4 or map_tensor.shape[1] != 1:
            raise ValueError("map_tensor must have shape [B,1,H,W]")
        self.maps = map_tensor.detach()
        self.max_faces = max(4, int(cfg.get("max_faces", 8)))
        self.margin = max(0.0, float(cfg.get("safety_margin", 0.02)))
        self.dilation = max(0, int(cfg.get("obstacle_dilation", 0)))
        self.chunk_size = max(1, int(cfg.get("corridor_chunk_size", 512)))
        self.axis_min = max(1e-5, float(cfg.get("ellipse_axis_min", 2e-3)))
        self.axis_max = max(self.axis_min, float(cfg.get("ellipse_axis_max", 2.0)))
        self.dedup_eps = max(0.0, float(cfg.get("dedup_eps", 1e-6)))
        self.max_obstacle_points = max(0, int(cfg.get("max_obstacle_points", 4096)))
        self._groups = self._prepare_map_groups()

    def _prepare_map_groups(self):
        """Group identical maps and extract their boundary points once."""
        batch = self.maps.shape[0]
        if batch == 0:
            return []
        first = self.maps[0, 0]
        if batch == 1 or torch.equal(self.maps[:, 0], first.expand_as(self.maps[:, 0])):
            groups = [(torch.arange(batch, device=self.maps.device), first)]
        else:
            # Mixed maps are uncommon at inference.  Grouping by exact equality
            # still avoids recomputing boundaries for repeated maze maps.
            pending = list(range(batch))
            groups = []
            while pending:
                root = pending.pop(0)
                same = [root]
                rest = []
                for i in pending:
                    if torch.equal(self.maps[i, 0], self.maps[root, 0]):
                        same.append(i)
                    else:
                        rest.append(i)
                pending = rest
                groups.append((torch.tensor(same, device=self.maps.device),
                               self.maps[root, 0]))

        out = []
        for indices, occ in groups:
            points = _obstacle_boundary_points(occ, self.dilation)
            if self.max_obstacle_points and len(points) > self.max_obstacle_points:
                # Deterministic uniform thinning; current 256x256 maze maps are
                # normally below this cap, so no geometry is discarded there.
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
        segments = horizon - 1
        dtype, device = p0.dtype, p0.device

        A = torch.zeros(batch, segments, self.max_faces, 2,
                        dtype=dtype, device=device)
        b = torch.zeros(batch, segments, self.max_faces,
                        dtype=dtype, device=device)
        face_mask = torch.zeros(batch, segments, self.max_faces,
                                dtype=torch.bool, device=device)

        # Normalised scene boundary faces, inset by the requested safety margin.
        box_A = p0.new_tensor(((1.0, 0.0), (-1.0, 0.0),
                              (0.0, 1.0), (0.0, -1.0)))
        box_b = p0.new_full((4,), 1.0 - self.margin)
        A[:, :, :4] = box_A
        b[:, :, :4] = box_b
        face_mask[:, :, :4] = True
        region_complete = torch.ones(batch, segments, dtype=torch.bool, device=device)

        # Segment k uses the ellipse predicted at its left endpoint k.
        center_all = p0[:, :-1] + e0[:, :-1, :2]
        axes_all = e0[:, :-1, 2:4].clamp(-6.0, 0.7).exp()
        axes_all = axes_all.clamp(self.axis_min, self.axis_max)
        theta_all = 0.5 * torch.atan2(e0[:, :-1, 5], e0[:, :-1, 4])

        obstacle_slots = self.max_faces - 4
        if obstacle_slots == 0:
            valid = torch.isfinite(center_all).all(dim=-1)
            return A, b, face_mask, valid

        for batch_indices, obstacle_points in self._groups:
            if obstacle_points.numel() == 0:
                continue
            centers = center_all[batch_indices].reshape(-1, 2)
            axes = axes_all[batch_indices].reshape(-1, 2)
            theta = theta_all[batch_indices].reshape(-1)
            local_A = A[batch_indices].reshape(-1, self.max_faces, 2).clone()
            local_b = b[batch_indices].reshape(-1, self.max_faces).clone()
            local_mask = face_mask[batch_indices].reshape(-1, self.max_faces).clone()
            local_complete = torch.zeros(len(centers), dtype=torch.bool, device=device)

            for begin in range(0, len(centers), self.chunk_size):
                end = min(begin + self.chunk_size, len(centers))
                c = centers[begin:end]
                ax = axes[begin:end]
                th = theta[begin:end]
                cos_t, sin_t = torch.cos(th), torch.sin(th)
                u = torch.stack((cos_t, sin_t), dim=-1)
                v = torch.stack((-sin_t, cos_t), dim=-1)

                diff = obstacle_points[None] - c[:, None]
                du = (diff * u[:, None]).sum(dim=-1)
                dv = (diff * v[:, None]).sum(dim=-1)
                metric = (du / ax[:, 0, None]).square() + \
                         (dv / ax[:, 1, None]).square()
                active = torch.ones_like(metric, dtype=torch.bool)

                for slot in range(obstacle_slots):
                    ranked = metric.masked_fill(~active, torch.inf)
                    nearest_value, nearest_idx = ranked.min(dim=1)
                    has_face = torch.isfinite(nearest_value)
                    if not bool(has_face.any()):
                        break

                    row = torch.arange(end - begin, device=device)
                    obs = obstacle_points[nearest_idx]
                    obs_diff = obs - c
                    obs_du = (obs_diff * u).sum(dim=-1)
                    obs_dv = (obs_diff * v).sum(dim=-1)
                    normal = (u * (obs_du / ax[:, 0].square())[:, None] +
                              v * (obs_dv / ax[:, 1].square())[:, None])
                    norm = normal.norm(dim=-1)
                    has_face &= torch.isfinite(norm) & (norm > 1e-7)
                    normal = normal / norm.clamp_min(1e-7)[:, None]
                    rhs = (normal * obs).sum(dim=-1) - self.margin

                    face = 4 + slot
                    local_A[begin:end, face] = torch.where(
                        has_face[:, None], normal,
                        local_A[begin:end, face],
                    )
                    local_b[begin:end, face] = torch.where(
                        has_face, rhs, local_b[begin:end, face],
                    )
                    local_mask[begin:end, face] = has_face

                    # Neural-IRIS greedy pruning: retain only obstacle points
                    # still inside the new half-space, and remove the selected
                    # point explicitly so every iteration makes progress.
                    projection = torch.einsum("sd,nd->sn", normal, obstacle_points)
                    keep = projection <= (rhs[:, None] + self.dedup_eps)
                    active &= keep | ~has_face[:, None]
                    active[row[has_face], nearest_idx[has_face]] = False

                # A capped face set is safe only if every obstacle boundary
                # point has been separated. Incomplete regions are marked
                # invalid instead of being presented to ALM as verified.
                local_complete[begin:end] = ~active.any(dim=1)

            local_A = local_A.reshape(len(batch_indices), segments,
                                      self.max_faces, 2)
            local_b = local_b.reshape(len(batch_indices), segments, self.max_faces)
            local_mask = local_mask.reshape(len(batch_indices), segments,
                                            self.max_faces)
            A[batch_indices] = local_A
            b[batch_indices] = local_b
            face_mask[batch_indices] = local_mask
            region_complete[batch_indices] = local_complete.reshape(
                len(batch_indices), segments)

        finite_faces = torch.isfinite(A).all(dim=-1) & torch.isfinite(b)
        face_mask &= finite_faces
        center_violation = (A * center_all[:, :, None]).sum(dim=-1) - b
        center_inside = (center_violation.masked_fill(~face_mask, -torch.inf)
                         .max(dim=-1).values <= 1e-5)
        valid = (torch.isfinite(center_all).all(dim=-1) & region_complete &
                 center_inside & (face_mask.sum(dim=-1) >= 4))
        return A, b, face_mask, valid
