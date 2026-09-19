"""Shared helpers for the report-faithful test suite."""

from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.geometry.ellipse_raster import ellipse_soft_mask  # noqa: E402
from src.geometry.ellipse_shape import shape4_to_abtheta  # noqa: E402
from src.models.trajsafe import TrajSafePlanner  # noqa: E402

SKELETON_ROOT = os.path.join(REPO_ROOT, "data", "skeleton")
SCENES_ROOT = os.path.join(REPO_ROOT, "data", "scenes")


def tiny_model(horizon=8, d_model=32, traj_blocks=2, skeleton_blocks=1,
               final_blocks=2, map_res=64, global_res=8, geo_res=16):
    cfg = dict(horizon=horizon, d_model=d_model, num_heads=4,
               traj_blocks=traj_blocks, skeleton_blocks=skeleton_blocks,
               final_blocks=final_blocks, ffn_dim=64, map_res=map_res,
               global_mem_res=global_res, geo_decode_res=geo_res,
               geo_mem_res=geo_res // 2, coord_hidden=32, head_hidden=32,
               dropout=0.0, assert_shapes=True)
    return TrajSafePlanner(cfg)


def tiny_batch(B=2, H=8, M=3, L=8, G=24, res=64, mask_res=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    cond = torch.rand(B, 2, 2, generator=g) * 1.6 - 0.8
    occ = torch.zeros(B, 1, res, res)
    occ[:, :, 0, :] = 1.0
    occ[:, :, -1, :] = 1.0

    candidate_xy = torch.rand(B, M, L, 2, generator=g) * 1.6 - 0.8
    geometry = torch.zeros(B, M, G, 2)
    for b in range(B):
        for m in range(M):
            t = torch.linspace(0.0, 1.0, G)
            base = cond[b, 0][None] * (1.0 - t[:, None]) + cond[b, 1][None] * t[:, None]
            offset = (m - (M - 1) / 2.0) * 0.05
            geometry[b, m] = base + torch.tensor([0.0, offset])
    glen = torch.full((B, M), G, dtype=torch.long)

    mask = torch.zeros(B, M, dtype=torch.bool)
    mask[:, 0] = True
    if M > 1:
        mask[0, 1] = True
    if M > 2:
        glen[:, -1] = 0
    glen = torch.where(mask, glen, torch.zeros_like(glen))

    idx = torch.linspace(0, G - 1, H).long()
    pos = geometry[:, 0][:, idx]

    shape4_gt = torch.zeros(B, H, 4)
    shape4_gt[..., 0] = torch.log(torch.tensor(0.10))
    shape4_gt[..., 1] = torch.log(torch.tensor(0.05))
    shape4_gt[..., 2] = 1.0
    shape_valid = torch.ones(B, H, dtype=torch.bool)
    a, b, theta = shape4_to_abtheta(shape4_gt)
    # GT mask is centred on the GT trajectory waypoint (no centre label).
    ellipse_mask = ellipse_soft_mask(pos, a, b, theta, mask_res, 10.0)

    return dict(
        pos=pos.clone(),
        cond=cond, occ=occ,
        features=candidate_xy, candidate_xy=candidate_xy,
        geom=geometry, geom_len=glen, mask=mask,
        candidate_mask=mask, candidate_geometry=geometry,
        candidate_geometry_lengths=glen,
        topology_best=torch.zeros(B, dtype=torch.long),
        has_candidate=mask.any(dim=-1),
        ellipse_shape4_gt=shape4_gt,
        shape_valid=shape_valid,
        ellipse_mask=ellipse_mask,
        t=torch.full((B,), 5, dtype=torch.long),
        ab=torch.full((B,), 0.5),
    )
