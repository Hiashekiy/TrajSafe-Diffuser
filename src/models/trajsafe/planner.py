"""TrajSafe-Diffuser planner, control-space version.

The diffusion state is the 32-control cubic B-spline polygon:

    Q_t in R^{B x 32 x 2}          <- noisy state, hard-conditioned endpoints
        |  BSplineCodec.decode_controls (fixed, differentiable)
    P_t in R^{B x 128 x 2}         <- the SAME 128-token curve the network sees

The network internals are unchanged (128 trajectory tokens, Skeleton match,
128 ellipse geometries) and the only progress definition is the fixed buffer

    s_i = i / 127           (there is NO learned progress head any more)

Forward chain::

    C_G, C_E = SceneCNN(occ)
    h_t      = TimeEmbedding(t)

    Q_t  = hard_control_endpoints(Q_t, cond)
    P_t  = B_128 Q_t
    H_traj  = TrajectoryBackbone(TrajectoryEncoder(P_t), C_G, h_t)
    coarse_raw = Head_P(H_traj);  Q_coarse = Fit(coarse_raw, start, goal)
    coarse     = B_128 Q_coarse

    H_S   = SkeletonEncoder(candidate_xy)
    R     = MatchBlock(H_traj, H_S, h_t)
    pi    = TopologyHead(R, candidate_mask)
    m     = m* (training) or argmax(pi) (inference)
    H_path = PathFeatureHead(R[m])
    center = Gamma_m(fixed s)            # fixed Skeleton centre, no head
    H_ell  = EllipseGeometry(H_path, center, C_E, h_t, ab)
    shape4 = EllipseShapeHead(H_ell, h_t)

    F       = MLP_fuse([H_traj, H_path, H_ell])
    H_clean = FinalDenoiser(F, C_G, h_t)
    raw_final = Head_P(H_clean)
    Q_final   = Fit(raw_final, start, goal)
    final     = B_128 Q_final
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...geometry.bspline import BSplineCodec, TrajectoryToControlHead
from ...geometry.ellipse_raster import scene_grid_centres
from ..common.scene_cnn import SceneCNN
from ..position_encoding import (Sinusoidal1DPositionEmbedding,
                                 Sinusoidal2DPositionEmbedding,
                                 SinusoidalTimestepEmbedding)
from .blocks import MatchBlock, TrajBlock
from .ellipse import EllipseGeometry
from .encoders import SkeletonEncoder, TrajectoryEncoder
from .fusion import FinalDenoiser, FusionMLP
from .geometry import CurveDecoder
from .heads import EllipseShapeHead, PathFeatureHead, TopologyHead

__all__ = ["TrajSafePlanner"]


class TrajSafePlanner(nn.Module):
    def __init__(self, model_cfg, ellipse_cfg=None, bspline_cfg=None):
        super().__init__()
        ellipse_cfg = dict(ellipse_cfg or {})
        bspline_cfg = dict(bspline_cfg or {})
        self.horizon = int(model_cfg["horizon"])
        self.d_model = int(model_cfg.get("d_model", 128))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.traj_blocks = int(model_cfg.get("traj_blocks", 8))
        self.final_blocks = int(model_cfg.get("final_blocks", 3))
        self.skeleton_blocks = int(model_cfg.get("skeleton_blocks", 2))
        self.ffn_dim = int(model_cfg.get("ffn_dim", 512))
        self.map_res = int(model_cfg.get("map_res", 256))
        self.global_res = int(model_cfg.get("global_mem_res",
                                            model_cfg.get("mem_res", 16)))
        self.geo_mem_res = int(model_cfg.get("geo_mem_res", 32))
        self.geo_decode_res = model_cfg.get("geo_decode_res")
        dropout = float(model_cfg.get("dropout", 0.0))
        head_hidden = int(model_cfg.get("head_hidden", 256))
        coord_hidden = int(model_cfg.get("coord_hidden",
                                         model_cfg.get("path_hidden", 256)))
        self.assert_shapes = bool(model_cfg.get("assert_shapes", True))

        geo_sigma = float(model_cfg.get(
            "geo_sigma", ellipse_cfg.get("geo_sigma", 0.25)))
        geo_bias_clip = float(model_cfg.get(
            "geo_bias_clip", ellipse_cfg.get("geo_bias_clip", 8.0)))
        H = self.horizon

        # ---- fixed B-spline codec (control <-> curve) ---------------------
        degree = int(bspline_cfg.get("degree", model_cfg.get("bspline_degree", 3)))
        self.num_controls = int(bspline_cfg.get(
            "num_controls", model_cfg.get("num_controls", 32)))
        curve_points = int(bspline_cfg.get(
            "curve_points", model_cfg.get("curve_points", H)))
        knots_path = bspline_cfg.get("knots")
        knots = bspline_cfg.get("knots_vector")
        self.bspline = BSplineCodec(
            degree=degree, num_controls=self.num_controls,
            curve_points=curve_points, knots=knots, knots_path=knots_path,
            endpoint_constrained=bool(bspline_cfg.get(
                "endpoint_constrained", True)))
        self.traj_to_control = TrajectoryToControlHead(self.bspline)

        self.scene_cnn = SceneCNN(
            d_model=self.d_model, res=self.map_res, global_res=self.global_res,
            geo_decode_res=(int(self.geo_decode_res) if self.geo_decode_res
                            else None),
            geo_mem_res=self.geo_mem_res)

        # ---- the three encodings -----------------------------------------
        self.spatial_pe = Sinusoidal2DPositionEmbedding(self.d_model)
        self.index_pe = Sinusoidal1DPositionEmbedding(self.d_model)
        self.time_pe = SinusoidalTimestepEmbedding(self.d_model)

        # ---- trajectory branch -------------------------------------------
        self.traj_encoder = TrajectoryEncoder(
            self.d_model, self.spatial_pe, self.index_pe, H, coord_hidden)
        self.traj_backbone = nn.ModuleList([
            TrajBlock(self.d_model, self.num_heads, self.ffn_dim, H, dropout)
            for _ in range(self.traj_blocks)
        ])
        # ONE trajectory head, shared by the coarse and final decode
        self.head_p = nn.Linear(self.d_model, 2)

        # ---- skeleton branch -------------------------------------------
        self.skeleton_encoder = SkeletonEncoder(
            self.d_model, self.num_heads, self.spatial_pe, self.index_pe,
            blocks=self.skeleton_blocks, ffn_dim=self.ffn_dim,
            dropout=dropout, hidden=coord_hidden)
        self.match_block = MatchBlock(self.d_model, self.num_heads,
                                      self.ffn_dim, dropout)
        self.topology_head = TopologyHead(self.d_model, head_hidden)
        self.path_feature_head = PathFeatureHead(self.d_model, head_hidden)

        # ---- ellipse branch (centre from the FIXED progress on Gamma_m) ---
        self.curve_decoder = CurveDecoder()
        self.ellipse_geometry = EllipseGeometry(
            self.d_model, self.num_heads, self.spatial_pe,
            geo_mem_res=self.geo_mem_res, geo_sigma=geo_sigma,
            geo_bias_clip=geo_bias_clip, hidden=coord_hidden, dropout=dropout)
        self.ellipse_shape_head = EllipseShapeHead(
            self.d_model, head_hidden, dropout)

        # ---- fusion + final denoiser -----------------------------------
        self.fusion_mlp = FusionMLP(self.d_model)
        self.final_denoiser = FinalDenoiser(
            self.d_model, self.num_heads, self.ffn_dim, H,
            blocks=self.final_blocks, dropout=dropout)

        # ---- the ONLY progress definition: a fixed buffer -----------------
        self.register_buffer("fixed_progress", torch.linspace(0.0, 1.0, H))

    # ------------------------------------------------------------------ scene
    def scene_tokens(self, occ: torch.Tensor):
        """occ [B,1,R,R] -> C_G [B,N_G,D], C_E [B,N_E,D] or None."""
        enc = self.scene_cnn(occ)
        gf = enc["global"].flatten(2).transpose(1, 2)
        grid_g = scene_grid_centres(self.global_res, gf.device, dtype=gf.dtype)
        c_g = gf + self.spatial_pe(grid_g)[None]
        c_e = None
        if enc["geometry"] is not None:
            zf = enc["geometry"].flatten(2).transpose(1, 2)
            grid_e = scene_grid_centres(self.geo_mem_res, zf.device,
                                        dtype=zf.dtype)
            c_e = zf + self.spatial_pe(grid_e)[None]
        return c_g, c_e

    # ------------------------------------------------------------ endpoints
    @staticmethod
    def hard_control_endpoints(q: torch.Tensor,
                               cond: torch.Tensor) -> torch.Tensor:
        """q [B,C,2]; cond [B,2,2] -> q with q[:,0]=start, q[:,-1]=goal.

        Because the knot vector is clamped, this is EXACTLY the curve endpoint
        hard condition: decode(q)[:, 0] == start and decode(q)[:, -1] == goal.
        """
        return BSplineCodec.hard_control_endpoints(q, cond)

    # backward-compatible alias (control space only; never apply it to a curve)
    hard_endpoints = hard_control_endpoints

    def decode_controls(self, q: torch.Tensor) -> torch.Tensor:
        return self.bspline.decode_controls(q)

    # ------------------------------------------------------------- trajectory
    def encode_trajectory(self, p_t: torch.Tensor, c_g: torch.Tensor,
                          h_t: torch.Tensor) -> torch.Tensor:
        x = self.traj_encoder(p_t)
        for blk in self.traj_backbone:
            x = blk(x, c_g, h_t)
        return x

    # ---------------------------------------------------------------- full
    def forward_all(self, q_t: torch.Tensor, occ: torch.Tensor,
                    cond: torch.Tensor, t: torch.Tensor, ab: torch.Tensor,
                    candidate_xy: torch.Tensor, candidate_mask: torch.Tensor,
                    geometry: torch.Tensor, geometry_lengths: torch.Tensor,
                    select_index: torch.Tensor | None = None):
        """One full forward pass of the control-space diffusion model.

        ``q_t`` is the 32-control diffusion state.  ``select_index`` is the
        training-time m* (argmin nDTW); when it is ``None`` inference routing
        ``argmax(pi)`` is used.  Invalid rows are routed to slot 0, masked out
        of every loss, and their output degenerates to the coarse curve.
        """
        B, C, _ = q_t.shape
        if C != self.num_controls:
            raise ValueError("q_t must be [B,%d,2], got %s"
                             % (self.num_controls, tuple(q_t.shape)))
        dev = q_t.device
        H = self.horizon

        # control-space hard conditioning, then the fixed decode
        q_t = self.hard_control_endpoints(q_t, cond)
        p_t = self.bspline.decode_controls(q_t)

        c_g, c_e = self.scene_tokens(occ)
        h_t = self.time_pe(t.to(dev))

        h_traj = self.encode_trajectory(p_t, c_g, h_t)
        coarse_raw = self.head_p(h_traj)
        q_coarse = self.traj_to_control(coarse_raw, cond)
        coarse = self.bspline.decode_controls(q_coarse)

        h_s = self.skeleton_encoder(candidate_xy)              # [B,M,L,D]
        r = self.match_block(h_traj, h_s, h_t)                 # [B,M,H,D]
        topo = self.topology_head(r, candidate_mask)

        if select_index is None:
            idx = topo["pi"].argmax(dim=-1)
        else:
            idx = select_index.to(dev).long()
        has_cand = candidate_mask.any(dim=-1)
        idx = torch.where(has_cand, idx, torch.zeros_like(idx))
        ar = torch.arange(B, device=dev)
        r_use = r[ar, idx]                                     # [B,H,D]
        h_path = self.path_feature_head(r_use)                 # [B,H,D]

        # FIXED progress: s_i = i / (H-1); NO learned progress anywhere.
        s = self.fixed_progress.to(dtype=p_t.dtype)[None].expand(B, H)
        gamma = geometry[ar, idx]                              # [B,G,2]
        gamma_len = geometry_lengths[ar, idx]                  # [B]
        center = self.curve_decoder(gamma, gamma_len, s)       # [B,H,2]
        center = torch.where(has_cand[:, None, None], center, coarse)

        h_ell, a_e = self.ellipse_geometry(h_path, center, c_e, h_t, ab)
        shape = self.ellipse_shape_head(h_ell, h_t)

        f = self.fusion_mlp(h_traj, h_path, h_ell)
        h_clean = self.final_denoiser(f, c_g, h_t)
        raw_final = self.head_p(h_clean)
        q_final = self.traj_to_control(raw_final, cond)
        final = self.bspline.decode_controls(q_final)
        final = torch.where(has_cand[:, None, None], final, coarse)

        out = {
            "input_control": q_t,
            "input_curve": p_t,
            "q_coarse": q_coarse,
            "coarse_raw": coarse_raw,
            "coarse": coarse,
            "control": q_final,
            "raw_curve": raw_final,
            "final": final,
            "H_traj": h_traj,
            "H_S": h_s,
            "R": r,
            "topo": topo,
            "selected_idx": idx,
            "has_candidate": has_cand,
            "R_use": r_use,
            "H_path": h_path,
            "gamma": gamma,
            "gamma_lengths": gamma_len,
            "ellipse": {
                "progress": s,
                "center": center,
                "H_ell": h_ell,
                "shape_raw": shape["raw"],
                "shape4": shape["shape4"],
                "a": shape["a"],
                "b": shape["b"],
                "theta": shape["theta"],
                "geo_attn": a_e,
            },
            "F": f,
            "H_clean": h_clean,
        }
        if self.assert_shapes:
            self._assert_forward_shapes(out, B, H)
        return out

    # ------------------------------------------------------------- assertions
    def _assert_forward_shapes(self, out: dict, B: int, H: int) -> None:
        D = self.d_model
        C = self.num_controls
        M = out["H_S"].shape[1]
        L = out["H_S"].shape[2]
        assert out["input_control"].shape == (B, C, 2)
        assert out["input_curve"].shape == (B, H, 2)
        assert out["H_traj"].shape == (B, H, D), out["H_traj"].shape
        assert out["H_S"].shape == (B, M, L, D), out["H_S"].shape
        assert out["R"].shape == (B, M, H, D), out["R"].shape
        assert out["topo"]["pi"].shape == (B, M), out["topo"]["pi"].shape
        assert out["H_path"].shape == (B, H, D), out["H_path"].shape
        assert out["control"].shape == (B, C, 2), out["control"].shape
        assert out["q_coarse"].shape == (B, C, 2), out["q_coarse"].shape
        assert out["ellipse"]["progress"].shape == (B, H)
        assert out["ellipse"]["center"].shape == (B, H, 2)
        assert out["ellipse"]["H_ell"].shape == (B, H, D)
        assert out["ellipse"]["shape4"].shape == (B, H, 4)
        assert out["F"].shape == (B, H, D)
        assert out["H_clean"].shape == (B, H, D)
        assert out["final"].shape == (B, H, 2)
        assert out["coarse"].shape == (B, H, 2)

        s = out["ellipse"]["progress"]
        assert torch.allclose(s[:, 0], torch.zeros_like(s[:, 0]), atol=1e-6)
        assert torch.allclose(s[:, -1], torch.ones_like(s[:, -1]), atol=1e-6)
        assert bool((s[:, 1:] > s[:, :-1]).all())
        assert bool((out["ellipse"]["a"] >= out["ellipse"]["b"] - 1e-7).all())
        assert bool(torch.isfinite(out["ellipse"]["shape4"]).all())
        direction = (out["ellipse"]["shape4"][..., 2] ** 2
                     + out["ellipse"]["shape4"][..., 3] ** 2)
        assert torch.allclose(direction, torch.ones_like(direction), atol=1e-5)
        # clamped B-spline: the hard control endpoints ARE the curve endpoints
        for key in ("input_curve", "coarse", "final"):
            p = out[key]
            err0 = (p[:, 0] - out["input_curve"][:, 0]).abs().max()
            err1 = (p[:, -1] - out["input_curve"][:, -1]).abs().max()
            assert float(err0.detach()) < 1e-5 and float(err1.detach()) < 1e-5
