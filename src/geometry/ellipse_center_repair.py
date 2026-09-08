"""Sequential repair of predicted physical ellipse centres."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.geometry.polytope_projection import project_point_to_polytope_2d


@dataclass
class CenterRepairResult:
    centers: torch.Tensor
    ellipse: torch.Tensor
    A: torch.Tensor
    b: torch.Tensor
    face_mask: torch.Tensor
    valid: torch.Tensor
    propagation_A: torch.Tensor
    propagation_b: torch.Tensor
    propagation_mask: torch.Tensor
    propagation_valid: torch.Tensor
    stats: dict[str, torch.Tensor]


def reencode_ellipse_centers(
    ellipse: torch.Tensor,
    centers: torch.Tensor,
    trajectory: torch.Tensor,
) -> torch.Tensor:
    """Encode fixed physical centres relative to the current trajectory."""
    out = ellipse.clone()
    out[..., :2] = centers - trajectory
    return out


class EllipseCenterRepair:
    """Repair centres against the latest valid propagation region.

    A failed rebuilt region is retained as an invalid current region.  Only the
    separate propagation region is reused for repairing subsequent centres.
    """

    def __init__(self, region_builder, feasibility_tol: float = 1e-6):
        self.builder = region_builder
        self.feasibility_tol = float(feasibility_tol)

    def _start_cell_box(self, start: torch.Tensor):
        maps = self.builder.maps
        batch, _, height, width = maps.shape
        max_faces = self.builder.max_faces
        A = start.new_zeros((batch, max_faces, 2))
        b = start.new_zeros((batch, max_faces))
        mask = torch.zeros((batch, max_faces), dtype=torch.bool, device=start.device)
        A[:, :4] = start.new_tensor(((1.0, 0.0), (-1.0, 0.0),
                                     (0.0, 1.0), (0.0, -1.0)))
        col = torch.floor((start[:, 0] + 1.0) * (width / 2.0)).long()
        row = torch.floor((start[:, 1] + 1.0) * (height / 2.0)).long()
        inside = ((start.abs() <= 1.0).all(dim=-1) &
                  torch.isfinite(start).all(dim=-1))
        col = col.clamp(0, width - 1)
        row = row.clamp(0, height - 1)
        xmin = -1.0 + col.to(start.dtype) * (2.0 / width)
        xmax = xmin + 2.0 / width
        ymin = -1.0 + row.to(start.dtype) * (2.0 / height)
        ymax = ymin + 2.0 / height
        b[:, :4] = torch.stack((xmax, -xmin, ymax, -ymin), dim=-1)
        mask[:, :4] = True
        cell_free = maps[torch.arange(batch, device=start.device), 0, row, col] <= 0.5
        valid = inside & cell_free & self.builder.points_are_free(start[:, None])[:, 0]
        return A, b, mask, valid

    def __call__(
        self,
        trajectory: torch.Tensor,
        ellipse: torch.Tensor,
        start: torch.Tensor,
    ) -> CenterRepairResult:
        if trajectory.ndim != 3 or trajectory.shape[-1] != 2:
            raise ValueError("trajectory must have shape [B,H,2]")
        if ellipse.shape[:2] != trajectory.shape[:2] or ellipse.shape[-1] < 6:
            raise ValueError("ellipse must have shape [B,H,6]")
        batch, horizon, _ = trajectory.shape
        clean_e = torch.nan_to_num(ellipse, nan=0.0, posinf=0.0, neginf=0.0)
        raw_centers = trajectory + clean_e[..., :2]
        raw_A, raw_b, raw_mask, raw_valid = self.builder.build_from_centers(
            raw_centers, clean_e)
        raw_free = self.builder.points_are_free(raw_centers)

        A = torch.zeros_like(raw_A)
        b = torch.zeros_like(raw_b)
        face_mask = torch.zeros_like(raw_mask)
        valid = torch.zeros_like(raw_valid)
        centers = raw_centers.clone()

        centers[:, 0] = start
        first_A, first_b, first_mask, first_region_valid = (
            self.builder.build_from_centers(start[:, None], clean_e[:, :1]))
        start_free = self.builder.points_are_free(start[:, None])[:, 0]
        first_valid = first_region_valid[:, 0] & start_free
        A[:, 0] = torch.where(first_valid[:, None, None], first_A[:, 0], A[:, 0])
        b[:, 0] = torch.where(first_valid[:, None], first_b[:, 0], b[:, 0])
        face_mask[:, 0] = torch.where(
            first_valid[:, None], first_mask[:, 0], face_mask[:, 0])
        valid[:, 0] = first_valid

        fallback_A, fallback_b, fallback_mask, fallback_valid = (
            self._start_cell_box(start))
        use_fallback = ~first_valid & fallback_valid
        propagation_A = torch.where(
            first_valid[:, None, None], first_A[:, 0], fallback_A)
        propagation_b = torch.where(
            first_valid[:, None], first_b[:, 0], fallback_b)
        propagation_mask = torch.where(
            first_valid[:, None], first_mask[:, 0], fallback_mask)
        propagation_valid = first_valid | use_fallback

        actually_moved = torch.zeros_like(raw_valid)
        projection_distance = trajectory.new_zeros((batch, horizon))
        propagation_reuse = torch.zeros_like(raw_valid)
        overlap_eligible = torch.zeros_like(raw_valid)
        common_seed_valid = torch.zeros_like(raw_valid)

        for k in range(1, horizon):
            projected, projection_ok = project_point_to_polytope_2d(
                raw_centers[:, k], propagation_A, propagation_b,
                propagation_mask, propagation_valid,
                feasibility_tol=self.feasibility_tol,
            )
            candidate_centers = torch.where(
                projection_ok[:, None], projected, raw_centers[:, k])
            if bool(projection_ok.any()):
                rebuilt_A, rebuilt_b, rebuilt_mask, rebuilt_valid = (
                    self.builder.build_from_centers(
                        candidate_centers[:, None], clean_e[:, k:k + 1]))
            else:
                rebuilt_A = raw_A[:, k:k + 1]
                rebuilt_b = raw_b[:, k:k + 1]
                rebuilt_mask = raw_mask[:, k:k + 1]
                rebuilt_valid = torch.zeros_like(raw_valid[:, k:k + 1])
            center_free = self.builder.points_are_free(
                candidate_centers[:, None])[:, 0]
            current_valid = projection_ok & rebuilt_valid[:, 0] & center_free
            centers[:, k] = candidate_centers
            distance = (projected - raw_centers[:, k]).norm(dim=-1)
            projection_distance[:, k] = torch.where(
                projection_ok, distance,
                projection_distance[:, k],
            )
            actually_moved[:, k] = projection_ok & (distance > 1e-6)

            previous_violation = (
                torch.einsum("bfd,bd->bf", propagation_A, candidate_centers) -
                propagation_b)
            inside_previous = (
                previous_violation.masked_fill(~propagation_mask, -torch.inf)
                .max(dim=-1).values <= self.feasibility_tol)
            current_violation = (
                torch.einsum("bfd,bd->bf", rebuilt_A[:, 0], candidate_centers) -
                rebuilt_b[:, 0])
            inside_current = (
                current_violation.masked_fill(~rebuilt_mask[:, 0], -torch.inf)
                .max(dim=-1).values <= self.feasibility_tol)
            overlap_eligible[:, k] = current_valid
            common_seed_valid[:, k] = (
                current_valid & inside_previous & inside_current)

            chosen_A = rebuilt_A[:, 0]
            chosen_b = rebuilt_b[:, 0]
            chosen_mask = rebuilt_mask[:, 0]
            A[:, k] = torch.where(
                current_valid[:, None, None], chosen_A, A[:, k])
            b[:, k] = torch.where(
                current_valid[:, None], chosen_b, b[:, k])
            face_mask[:, k] = torch.where(
                current_valid[:, None], chosen_mask, face_mask[:, k])
            valid[:, k] = current_valid

            propagation_reuse[:, k] = ~current_valid & propagation_valid
            propagation_A = torch.where(
                current_valid[:, None, None], chosen_A, propagation_A)
            propagation_b = torch.where(
                current_valid[:, None], chosen_b, propagation_b)
            propagation_mask = torch.where(
                current_valid[:, None], chosen_mask, propagation_mask)
            propagation_valid = propagation_valid | current_valid

        repaired_e = reencode_ellipse_centers(clean_e, centers, trajectory)
        post_free = self.builder.points_are_free(centers)
        moved_count = actually_moved.sum().clamp_min(1)
        overlap_count = overlap_eligible[:, 1:].sum().clamp_min(1)
        stats = {
            "center_raw_unsafe_rate": (~raw_free).float().mean(),
            "center_repair_rate": actually_moved.float().mean(),
            "center_projection_mean": projection_distance.sum() / moved_count,
            "center_projection_max": projection_distance.max(),
            "center_post_unsafe_rate": (~post_free).float().mean(),
            "adjacent_region_overlap_rate": (
                common_seed_valid[:, 1:].sum() / overlap_count),
            "region_build_failure_rate": (~valid).float().mean(),
            "propagation_reuse_rate": (
                propagation_reuse[:, 1:].float().mean()
                if horizon > 1 else trajectory.new_zeros(())),
            "region_valid_rate_raw": raw_valid.float().mean(),
            "region_valid_rate_repaired": valid.float().mean(),
        }
        return CenterRepairResult(
            centers, repaired_e, A, b, face_mask, valid,
            propagation_A, propagation_b, propagation_mask,
            propagation_valid, stats,
        )
