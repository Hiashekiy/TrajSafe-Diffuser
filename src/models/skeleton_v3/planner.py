"""V3 planner: Skeleton-Topology-Grounded Trajectory Diffusion (dynamic).

ONE diffusion state, and the WHOLE network runs at EVERY reverse timestep:

    P_t -> P~_0 -> S_m -> s -> c = gamma_m(s) -> E -> P^_0 -> P_{t-1}

There is no commit timestep, no cached selection and no "before/after tc" stage
switch: the trajectory is decoded twice by the SAME head_p (once from the
backbone for skeleton matching, once after the joint fusion) and only the second
one drives DDIM.

Interface (used by both train_v3 and sampler_v3):

    base  = planner.encode_trajectory(p_t, occ, cond, t, ab)
    coarse = planner.coarse_trajectory(base)
    topo  = planner.score_candidates(base, coarse, cand_feat, cand_mask, cand_len)
    ell   = planner.build_ellipses(base, path_feat_sel, geom_sel, geom_len_sel, ab)
    hfin  = planner.fuse(base, ell["tokens"])
    final = planner.final_trajectory(hfin)
    out   = planner.forward_all(...)          # training convenience
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..joint.joint_planner import scene_grid_centres
from ..joint.scene_cnn import SceneCNN
from ..position_encoding import (Sinusoidal1DPositionEmbedding,
                                 Sinusoidal2DPositionEmbedding,
                                 SinusoidalTimestepEmbedding)
from .blocks import JointFusionBlock, TrajBlock
from .ellipse_head import EllipseHead
from .geometry import gather_dense_path_points
from .progress_head import ProgressHead
from .topology_selector import TopologySelector

__all__ = ["SkeletonPlannerV3"]


class SkeletonPlannerV3(nn.Module):
    def __init__(self, model_cfg, ellipse_cfg=None):
        super().__init__()
        ellipse_cfg = dict(ellipse_cfg or {})
        self.horizon = int(model_cfg["horizon"])
        self.d_model = int(model_cfg.get("d_model", 128))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.traj_blocks = int(model_cfg.get("traj_blocks", 8))
        self.fusion_blocks = int(model_cfg.get("fusion_blocks", 3))
        self.ffn_dim = int(model_cfg.get("ffn_dim", 512))
        self.map_res = int(model_cfg.get("map_res", 256))
        self.global_res = int(model_cfg.get("global_mem_res",
                                            model_cfg.get("mem_res", 16)))
        self.geo_mem_res = int(model_cfg.get("geo_mem_res", 32))
        self.geo_decode_res = model_cfg.get("geo_decode_res")
        dropout = float(model_cfg.get("dropout", 0.0))
        head_hidden = int(model_cfg.get("head_hidden", 256))
        path_hidden = int(model_cfg.get("path_hidden", 128))
        path_layers = int(model_cfg.get("path_transformer_layers", 2))
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
        # ONE trajectory head, shared by the coarse and the final decode
        self.head_p = nn.Linear(self.d_model, 2)

        self.selector = TopologySelector(
            self.d_model, self.num_heads, hidden=head_hidden, dropout=dropout,
            path_hidden=path_hidden, path_layers=path_layers)
        self.progress = ProgressHead(self.d_model, self.num_heads,
                                     hidden=head_hidden, dropout=dropout)
        self.ellipse = EllipseHead(
            self.d_model, self.num_heads, self.spatial_pe, hidden=head_hidden,
            dropout=dropout,
            b_min=float(ellipse_cfg.get("b_min", 0.02)),
            b_max=float(ellipse_cfg.get("b_max", 0.30)),
            a_max=float(ellipse_cfg.get("a_max", 0.80)),
            geo_mem_res=self.geo_mem_res,
            geo_sigma=float(model_cfg.get("geo_sigma", 0.25)),
            geo_bias_clip=float(model_cfg.get("geo_bias_clip", 8.0)))

        self.mlp_e = nn.Linear(6, self.d_model)
        self.fusion = nn.ModuleList([
            JointFusionBlock(self.d_model, self.num_heads, self.ffn_dim, H, dropout)
            for _ in range(self.fusion_blocks)
        ])

    # ------------------------------------------------------------------ scene
    def _scene_tokens(self, occ):
        enc = self.scene_cnn(occ)
        gf = enc["global"].flatten(2).transpose(1, 2)
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
        start_role = self.role_type(torch.zeros(B, device=dev, dtype=torch.long))
        goal_role = self.role_type(torch.ones(B, device=dev, dtype=torch.long))
        h_s = (self.spatial_pe(cond[:, 0]) + start_role)[:, None, :]
        h_g = (self.spatial_pe(cond[:, 1]) + goal_role)[:, None, :]
        return torch.cat([h_s, h_g, global_map], dim=1), geo_mem

    @staticmethod
    def hard_endpoints(p, cond):
        p = p.clone()
        p[:, 0] = cond[:, 0]
        p[:, -1] = cond[:, 1]
        return p

    # -------------------------------------------------------- trajectory side
    def encode_trajectory(self, p_t, occ, cond, t, ab):
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
        return {"traj_feat": x, "global_mem": global_mem, "geo_mem": geo_mem,
                "h_t": h_t, "ab": ab}

    def coarse_trajectory(self, base, cond):
        return self.hard_endpoints(self.head_p(base["traj_feat"]), cond)

    def final_trajectory(self, h_final, cond):
        return self.hard_endpoints(self.head_p(h_final), cond)

    # ------------------------------------------------------------------ topo
    def score_candidates(self, base, coarse, candidate_features, candidate_mask,
                         candidate_lengths):
        return self.selector(base["traj_feat"], coarse, base["h_t"],
                             candidate_features, candidate_mask,
                             candidate_lengths)

    # -------------------------------------------------------------- ellipses
    def build_ellipses(self, base, coarse, path_feat, geometry,
                       geometry_lengths, ab):
        """path_feat [B,L,D] of the SELECTED candidate; geometry [B,G,2] dense."""
        s, fused = self.progress(base["traj_feat"], path_feat, base["h_t"])
        degenerate = (geometry_lengths < 2)
        center = gather_dense_path_points(geometry, geometry_lengths, s)
        if bool(degenerate.any()):
            # no candidate (or a degenerate chain): fall back to the coarse
            # trajectory points so the forward stays finite; the losses mask
            # these rows out entirely.
            center = torch.where(degenerate[:, None, None], coarse, center)
        ell = self.ellipse(fused, center, base["geo_mem"], base["h_t"], ab)
        token_in = torch.cat([center, ell["shape4"]], dim=-1)
        h_e = (self.mlp_e(token_in)
               + self.plan_pe(self.plan_idx.to(token_in.device))[None])
        if ell.get("geo_attn") is not None:
            h_e = h_e + ell["geo_attn"]
        ell.update({"progress": s, "center": center, "tokens": h_e,
                    "fused": fused})
        return ell

    # ----------------------------------------------------------- joint fusion
    def fuse(self, base, ellipse_tokens):
        z = torch.stack([base["traj_feat"], ellipse_tokens], dim=2)
        B, H = base["traj_feat"].shape[0], self.horizon
        z = z.reshape(B, 2 * H, self.d_model)
        for blk in self.fusion:
            z = blk(z, base["global_mem"], base["h_t"])
        z = z.view(B, H, 2, self.d_model)
        return z[:, :, 0]

    # -------------------------------------------------------------- full pass
    def forward_all(self, p_t, occ, cond, t, ab, candidate_features,
                    candidate_mask, candidate_lengths, geometry,
                    geometry_lengths, select_index=None):
        """One full V3 forward.  select_index [B] overrides the choice (used by
        the training schedule: GT teacher forcing -> predicted)."""
        base = self.encode_trajectory(p_t, occ, cond, t, ab)
        coarse = self.coarse_trajectory(base, cond)
        topo = self.score_candidates(base, coarse, candidate_features,
                                     candidate_mask, candidate_lengths)
        return self.finish(base, coarse, topo, select_index, geometry,
                           geometry_lengths, candidate_mask, cond, ab)

    def finish(self, base, coarse, topo, select_index, geometry,
               geometry_lengths, candidate_mask, cond, ab):
        """Second half of the forward, given an explicit selection.

        Split out so the training schedule (GT teacher forcing -> predicted) can
        pick the index AFTER the topology scores are known; encode/coarse/score
        do not depend on the choice, so this is still a single forward.
        """
        B = base["traj_feat"].shape[0]
        ar = torch.arange(B, device=base["traj_feat"].device)
        if select_index is None:
            idx = topo["pi"].argmax(dim=-1)
        else:
            idx = select_index.long()
        idx = torch.where(candidate_mask.any(dim=1), idx,
                          torch.zeros_like(idx))
        path_feat = topo["path_feat"][ar, idx]
        geom = geometry[ar, idx]
        geom_len = geometry_lengths[ar, idx]
        ell = self.build_ellipses(base, coarse, path_feat, geom, geom_len, ab)
        h_final = self.fuse(base, ell["tokens"])
        final = self.final_trajectory(h_final, cond)
        return {"base": base, "coarse": coarse, "topo": topo, "ellipse": ell,
                "h_final": h_final, "final": final, "selected_idx": idx}
