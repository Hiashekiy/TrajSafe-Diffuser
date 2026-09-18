"""Shared helpers for the V2 test suite (not a test module itself)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.skeleton import SkeletonPlanner  # noqa: E402

MAPS_DIR = os.path.join(REPO_ROOT, "data", "processed_scene_v1", "maps")
SKELETON_DIR = os.path.join(REPO_ROOT, "data", "processed_scene_v2", "skeletons")


def tiny_model(horizon=8, d_model=32, detach=True, traj_blocks=2,
               refine_blocks=1, map_res=64, global_res=8, geo_res=16):
    cfg = dict(horizon=horizon, d_model=d_model, num_heads=4,
               traj_blocks=traj_blocks, refine_blocks=refine_blocks, ffn_dim=64,
               map_res=map_res, global_mem_res=global_res, geo_decode_res=geo_res,
               geo_mem_res=geo_res // 2, head_hidden=32, path_hidden=32,
               dropout=0.0)
    return SkeletonPlanner(cfg, dict(detach_trajectory_feature=detach))


def tiny_batch(B=2, H=8, M=3, L=8, res=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    cond = torch.rand(B, 2, 2, generator=g) * 1.6 - 0.8
    occ = torch.zeros(B, 1, res, res)
    occ[:, :, 0, :] = 1.0
    occ[:, :, -1, :] = 1.0
    cand = torch.rand(B, M, L, 5, generator=g) * 1.6 - 0.8
    cand[..., 4] = torch.linspace(0, 1, L)[None, None, :]
    mask = torch.zeros(B, M, dtype=torch.bool)
    mask[:, 0] = True
    mask[0, 1] = True
    lengths = torch.rand(B, M, generator=g)
    return dict(cond=cond, occ=occ, cand=cand, mask=mask, lengths=lengths)


def straight_path(B, M, L, start=-0.5, goal=0.5):
    t = torch.linspace(0.0, 1.0, L)[None, None, :, None]
    x = start + (goal - start) * t
    p = torch.cat([x, torch.zeros_like(x)], dim=-1)
    return p.expand(B, M, L, 2).contiguous()
