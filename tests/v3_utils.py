"""Shared helpers for the V3 test suite."""

from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.skeleton_v3 import SkeletonPlannerV3  # noqa: E402

V3_ROOT = os.path.join(REPO_ROOT, "data", "processed_scene_v3")
SOURCE_ROOT = os.path.join(REPO_ROOT, "data", "processed_scene_v1")


def tiny_model(horizon=8, d_model=32, traj_blocks=2, fusion_blocks=2,
               map_res=64, global_res=8, geo_res=16, b_min=0.02, b_max=0.30,
               a_max=0.80):
    cfg = dict(horizon=horizon, d_model=d_model, num_heads=4,
               traj_blocks=traj_blocks, fusion_blocks=fusion_blocks, ffn_dim=64,
               map_res=map_res, global_mem_res=global_res, geo_decode_res=geo_res,
               geo_mem_res=geo_res // 2, head_hidden=32, path_hidden=32,
               path_transformer_layers=1, dropout=0.0)
    return SkeletonPlannerV3(cfg, dict(b_min=b_min, b_max=b_max, a_max=a_max))


def tiny_batch(B=2, H=8, M=3, L=8, G=24, res=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    cond = torch.rand(B, 2, 2, generator=g) * 1.6 - 0.8
    occ = torch.zeros(B, 1, res, res)
    occ[:, :, 0, :] = 1.0
    occ[:, :, -1, :] = 1.0
    features = torch.rand(B, M, L, 5, generator=g) * 1.6 - 0.8
    features[..., 4] = torch.linspace(0, 1, L)[None, None, :]
    geometry = torch.rand(B, M, G, 2, generator=g) * 1.6 - 0.8
    glen = torch.full((B, M), G, dtype=torch.long)
    glen[:, -1] = 0                      # last candidate invalid
    mask = torch.zeros(B, M, dtype=torch.bool)
    mask[:, 0] = True
    mask[0, 1] = True
    lengths = torch.rand(B, M, generator=g) + 0.5
    p0 = torch.rand(B, H, 2, generator=g) * 1.6 - 0.8
    p0[:, 0] = cond[:, 0]
    p0[:, -1] = cond[:, 1]
    t = torch.full((B,), 5, dtype=torch.long)
    ab = torch.full((B,), 0.5)
    return dict(pos=p0, cond=cond, occ=occ, features=features, geom=geometry,
                geom_len=glen, mask=mask, lengths=lengths, t=t, ab=ab)
