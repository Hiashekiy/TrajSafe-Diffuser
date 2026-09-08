"""Exact Euclidean projection onto small two-dimensional polytopes."""
from __future__ import annotations

import torch


def project_point_to_polytope_2d(
    point: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    face_mask: torch.Tensor,
    valid: torch.Tensor | None = None,
    feasibility_tol: float = 1e-6,
    parallel_tol: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``point`` onto ``{x | A x <= b}`` using its faces and vertices.

    In 2-D the closest point is either the input itself, an orthogonal
    projection onto one active face, or the intersection of two active faces.
    The implementation is pure Torch and supports arbitrary leading batches.
    Failed projections leave the input unchanged and return ``success=False``.
    """
    if point.ndim < 1 or A.ndim < 2 or point.shape[-1] != 2 or A.shape[-1] != 2:
        raise ValueError("point and A must end in dimensions 2 and [M,2]")
    if A.shape[:-2] != point.shape[:-1] or b.shape != A.shape[:-1]:
        raise ValueError("incompatible point, A and b shapes")
    if face_mask.shape != b.shape:
        raise ValueError("face_mask must have the same shape as b")

    leading = point.shape[:-1]
    faces = A.shape[-2]
    flat_point = point.reshape(-1, 2)
    flat_A = A.reshape(-1, faces, 2)
    flat_b = b.reshape(-1, faces)
    flat_mask = face_mask.reshape(-1, faces)
    flat_valid = (torch.ones(len(flat_point), dtype=torch.bool, device=point.device)
                  if valid is None else valid.reshape(-1).bool())

    normals_sq = flat_A.square().sum(dim=-1)
    usable_faces = (flat_mask & torch.isfinite(flat_A).all(dim=-1) &
                    torch.isfinite(flat_b) & (normals_sq > parallel_tol))
    safe_normals_sq = normals_sq.clamp_min(parallel_tol)
    signed = torch.einsum("nfd,nd->nf", flat_A, flat_point) - flat_b
    face_candidates = (flat_point[:, None] -
                       signed[..., None] * flat_A / safe_normals_sq[..., None])

    pair_i, pair_j = torch.triu_indices(
        faces, faces, offset=1, device=point.device)
    ai, aj = flat_A[:, pair_i], flat_A[:, pair_j]
    bi, bj = flat_b[:, pair_i], flat_b[:, pair_j]
    det = ai[..., 0] * aj[..., 1] - ai[..., 1] * aj[..., 0]
    pair_usable = (usable_faces[:, pair_i] & usable_faces[:, pair_j] &
                   torch.isfinite(det) & (det.abs() > parallel_tol))
    safe_det = torch.where(pair_usable, det, torch.ones_like(det))
    pair_candidates = torch.stack((
        (bi * aj[..., 1] - ai[..., 1] * bj) / safe_det,
        (ai[..., 0] * bj - bi * aj[..., 0]) / safe_det,
    ), dim=-1)

    candidates = torch.cat((flat_point[:, None], face_candidates,
                            pair_candidates), dim=1)
    source_usable = torch.cat((
        flat_valid[:, None],
        usable_faces & flat_valid[:, None],
        pair_usable & flat_valid[:, None],
    ), dim=1)
    violation = (torch.einsum("ncd,nfd->ncf", candidates, flat_A) -
                 flat_b[:, None])
    feasible = ((violation <= feasibility_tol) |
                ~usable_faces[:, None]).all(dim=-1)
    finite = torch.isfinite(candidates).all(dim=-1)
    candidate_valid = source_usable & feasible & finite
    distance_sq = (candidates - flat_point[:, None]).square().sum(dim=-1)
    distance_sq = distance_sq.masked_fill(~candidate_valid, torch.inf)
    best_distance, best_index = distance_sq.min(dim=1)
    success = torch.isfinite(best_distance)
    best = candidates[torch.arange(len(candidates), device=point.device), best_index]
    projected = torch.where(success[:, None], best, flat_point)
    return projected.reshape(*leading, 2), success.reshape(*leading)
