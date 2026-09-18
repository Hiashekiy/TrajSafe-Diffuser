"""Skeleton-Topology-Grounded Trajectory Diffusion (V2).

One diffusion state (the trajectory), one forward pipeline:

    x_T -> ... -> x_tc        trajectory-only reverse diffusion
    P_hat^c = x0_p_base        coarse clean estimate at the commit step
    m ~ Cat(pi)                TopologySelector ranks the safe candidates
    s_i                        ProgressHead places the K ellipses on P_m
    c_i = gamma_m(s_i)         ellipse centre (geometric, never predicted)
    E_i                        EllipseShapeHead at the FIXED centre
    x_tc -> ... -> x_0         the SAME trajectory diffusion, refined with the
                               ellipse tokens as conditioning

The forward pass is split into three explicit entry points (docs/V2.md section
28) instead of one giant forward with twenty optional flags:

    base = model.encode_trajectory(p_t, occ, cond, t, ab)
    topo = model.score_candidates(base, candidate_paths, candidate_mask, ...)
    out  = model.refine_with_path(base, selected_path, selected_path_feat)

There is no ellipse diffusion state anywhere in this model: the V1 state
(p_t, e_t) no longer exists.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..joint.joint_planner import scene_grid_centres
from ..joint.scene_cnn import SceneCNN
from ..position_encoding import (Sinusoidal1DPositionEmbedding,
                                 Sinusoidal2DPositionEmbedding,
                                 SinusoidalTimestepEmbedding)
from .ellipse_shape_head import EllipseShapeHead
from .path_ops import gather_path_points
from .progress_head import ProgressHead
from .topology_selector import TopologySelector
from .traj_blocks import RefineBlock, TrajBlock

__all__ = ["SkeletonPlanner"]


class SkeletonPlanner(nn.Module):
    def __init__(self, model_cfg, topo_cfg=None):
        super().__init__()
        topo_cfg = dict(topo_cfg or {})
        self.horizon = int(model_cfg["horizon"])
        self.d_model = int(model_cfg.get("d_model", 128))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.traj_blocks = int(model_cfg.get("traj_blocks",
                                             model_cfg.get("joint_blocks", 8)))
        self.refine_blocks = int(model_cfg.get("refine_blocks", 2))
        self.ffn_dim = int(model_cfg.get("ffn_dim", 512))
        self.map_res = int(model_cfg.get("map_res", 256))
        self.global_res = int(model_cfg.get("global_mem_res",
                                            model_cfg.get("mem_res", 16)))
        self.geo_mem_res = int(model_cfg.get("geo_mem_res", 32))
        self.geo_decode_res = model_cfg.get("geo_decode_res")
        self.geo_sigma = float(model_cfg.get("geo_sigma", 0.25))
        self.geo_bias_clip = float(model_cfg.get("geo_bias_clip", 8.0))
        dropout = float(model_cfg.get("dropout", 0.0))
        head_hidden = int(model_cfg.get("head_hidden", 256))
        path_hidden = int(model_cfg.get("path_hidden", 128))
        H = self.horizon

        self.scene_cnn = SceneCNN(
            d_model=self.d_model, res=self.map_res, global_res=self.global_res,
            geo_decode_res=(int(self.geo_decode_res) if self.geo_decode_res else None),
            geo_mem_res=self.geo_mem_res)

        self.spatial_pe = Sinusoidal2DPositionEmbedding(self.d_model)
        self.plan_pe = Sinusoidal1DPositionEmbedding(self.d_model)
        self.time_pe = SinusoidalTimestepEmbedding(self.d_model)
        self.plan_idx = torch.arange(H)
        self.traj_type = nn.Embedding(2, self.d_model)
        self.role_type = nn.Embedding(2, self.d_model)
        self.mlp_p = nn.Linear(2, self.d_model)

        self.blocks = nn.ModuleList([
            TrajBlock(self.d_model, self.num_heads, self.ffn_dim, H, dropout)
            for _ in range(self.traj_blocks)
        ])
        self.head_p = nn.Linear(self.d_model, 2)

        # ---- topology / progress / ellipse shape (sections 13, 19, 26) ----
        self.selector = TopologySelector(
            self.d_model, self.num_heads, hidden=head_hidden, dropout=dropout,
            path_hidden=path_hidden,
            detach_trajectory_feature=bool(
                topo_cfg.get("detach_trajectory_feature", True)))
        self.progress = ProgressHead(self.d_model, self.num_heads,
                                     hidden=head_hidden, dropout=dropout)
        self.shape = EllipseShapeHead(
            self.d_model, self.num_heads, self.spatial_pe, hidden=head_hidden,
            dropout=dropout, geo_mem_res=self.geo_mem_res,
            geo_sigma=self.geo_sigma, geo_bias_clip=self.geo_bias_clip)

        # ---- ellipse tokens -> trajectory refinement (section 27) --------
        self.mlp_e = nn.Linear(6, self.d_model)
        self.refine = nn.ModuleList([
            RefineBlock(self.d_model, self.num_heads, self.ffn_dim, H, dropout)
            for _ in range(self.refine_blocks)
        ])
        self.head_p_refine = nn.Linear(self.d_model, 2)

    # ------------------------------------------------------------------
    def _scene_tokens(self, occ):
        enc = self.scene_cnn(occ)
        gf = enc["global"]
        gf = gf.flatten(2).transpose(1, 2)                  # [B,G*G,d]
        grid_g = scene_grid_centres(self.global_res, gf.device)
        global_map = gf + self.spatial_pe(grid_g)[None]

        geo_mem = None
        if enc["geometry"] is not None:
            zf = enc["geometry"].flatten(2).transpose(1, 2)
            grid_z = scene_grid_centres(self.geo_mem_res, zf.device)
            geo_mem = zf + self.spatial_pe(grid_z)[None]
        return global_map, geo_mem

    def _global_mem(self, occ, cond, B):
        global_map, geo_mem = self._scene_tokens(occ)
        dev = global_map.device
        start, goal = cond[:, 0], cond[:, 1]
        start_role = self.role_type(torch.zeros(B, device=dev, dtype=torch.long))
        goal_role = self.role_type(torch.ones(B, device=dev, dtype=torch.long))
        h_s = (self.spatial_pe(start) + start_role)[:, None, :]
        h_g = (self.spatial_pe(goal) + goal_role)[:, None, :]
        return torch.cat([h_s, h_g, global_map], dim=1), geo_mem

    # ------------------------------------------------------------------
    def encode_trajectory(self, p_t, occ, cond, t, ab):
        """First half: trajectory-only noisy-state encoder.

        Returns traj_feat [B,H,D], x0_p_base [B,H,2], global_mem, geo_mem.
        """
        B, H, _ = p_t.shape
        dev = p_t.device
        global_mem, geo_mem = self._global_mem(occ, cond, B)

        psi = self.plan_pe(self.plan_idx.to(dev))[None]
        h_t = self.time_pe(t)

        w = ab[:, None, None].to(p_t.dtype)
        p_pe = self.spatial_pe(p_t) * w
        traj_type = self.traj_type.weight[0][None, None, :]
        x = self.mlp_p(p_t) + psi + traj_type + p_pe

        for blk in self.blocks:
            x = blk(x, global_mem, h_t)
        return {
            "traj_feat": x,
            "x0_p_base": self.head_p(x),
            "global_mem": global_mem,
            "geo_mem": geo_mem,
            "h_t": h_t,
            "ab": ab,
        }

    # ------------------------------------------------------------------
    def score_candidates(self, base, candidate_paths, candidate_mask,
                         candidate_lengths=None):
        """Rank the already-complete candidate topologies."""
        if candidate_lengths is None:
            candidate_lengths = torch.zeros(
                candidate_paths.shape[:2], device=candidate_paths.device,
                dtype=candidate_paths.dtype)
        return self.selector(base["traj_feat"], base["x0_p_base"],
                             candidate_paths, candidate_mask, candidate_lengths)

    # ------------------------------------------------------------------
    def refine_with_path(self, base, selected_path, selected_path_feat):
        """Second half: progress -> ellipse centre -> shape -> refine the SAME
        trajectory diffusion with the ellipse tokens as conditioning.

        selected_path      [B,L,2] scene coordinates of the committed candidate
        selected_path_feat [B,L,D] its PathEncoder tokens
        """
        s, fused = self.progress(base["traj_feat"], selected_path_feat)
        center = gather_path_points(selected_path, s)
        shape4 = self.shape(center, fused, base["geo_mem"], ab=base.get("ab"))

        tok = torch.cat([center, shape4], dim=-1)           # [B,K,6]
        h_e = (self.mlp_e(tok)
               + self.plan_pe(self.plan_idx.to(tok.device))[None])

        x = base["traj_feat"]
        for blk in self.refine:
            x = blk(x, base["global_mem"], base["h_t"], h_e)
        return {
            "x0_p": self.head_p_refine(x),
            "traj_feat": x,
            "progress": s,
            "ellipse_center": center,
            "ellipse_shape4": shape4,
        }
