"""TrajSafe-Diffuser planner, control-token version.

The diffusion state is the C-control cubic B-spline polygon

    Q_t in R^{B x C x 2}          C = model.num_controls (config, default 32)

and the network NEVER sees decoded trajectory points any more.  The control
polygon itself is the token sequence; ``128`` survives only as the number of
Skeleton / ellipse geometry queries (``model.num_safety_queries``).

Forward chain (``model.control_space: true``)::

    C_G, C_E = SceneCNN(occ)
    h_t      = TimeEmbedding(t)

    Q_t   = hard_control_endpoints(Q_t, cond)          # exact curve endpoints
    H_ctrl = ControlEncoder(Q_t)                       # [B,C,D]
    H_ctrl = ControlBackbone(H_ctrl, C_G, h_t)         # [B,C,D]
    H_fb   = FeedbackEncoder([Q0_safe_prev, Delta_prev, valid])   # optional
    H_ctrl = FeedbackFusion(H_ctrl, H_fb, valid)       # gated residual
    Q~_coarse = Head_Q(H_ctrl)                         # [B,C,2]
    Q_coarse  = BoundaryDecoder(Q~_coarse)             # exact endpoints

    H_S   = SkeletonEncoder(candidate_xy)              # [B,M,L,D]
    R     = MatchBlock(H_ctrl, H_S, h_t)               # [B,M,C,D]
    pi    = TopologyHead(R, candidate_mask)
    m     = m* (training) or argmax(pi) (inference)
    H_path = PathFeatureHead(R[m])                     # [B,C,D]

    H_Gamma = SafetyQueryHead(H_S[m])                  # [B,Q,D]  Q = L
    c_i     = Gamma_m(s_i),  s_i = i/(Q-1)             # fixed, no head
    H_ell   = EllipseGeometry(H_Gamma, c, C_E, h_t, ab)   # [B,Q,D]
    shape4  = EllipseShapeHead(H_ell, h_t)                # [B,Q,4]

    A_safe = SafetyControlFusion(H_path, H_ell, h_t)   # [B,C,D] cross attention
    F      = FusionMLP([H_ctrl, H_path, A_safe])       # [B,C,D]
    H_clean = FinalDenoiser(F, C_G, h_t)               # [B,C,D]
    Q~_final = Head_Q(H_clean)                         # [B,C,2]
    Q_final  = BoundaryDecoder(Q~_final)               # exact endpoints

The decoded curve ``B_128 Q`` is produced ONLY at the very end for plotting,
collision evaluation and the controller.  ``L_traj`` (MSE on decoded points) no
longer exists.

Legacy path (``model.control_space: false``): the pre-refactor chain that runs
the network on the decoded 128-point curve is kept byte-for-byte so a checkpoint
trained before this refactor can still be replayed for demos; see
:meth:`TrajSafePlanner.forward_curve_tokens`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...geometry.bspline import BSplineCodec, TrajectoryToControlHead
from ...geometry.ellipse_raster import scene_grid_centres
from ..common.scene_cnn import SceneCNN
from ..position_encoding import (Sinusoidal1DPositionEmbedding,
                                 Sinusoidal2DPositionEmbedding,
                                 SinusoidalTimestepEmbedding)
from .blocks import MatchBlock, TrajBlock
from .boundary import BoundaryDecoder
from .ellipse import EllipseGeometry
from .encoders import SkeletonEncoder, TrajectoryEncoder
from .feedback import FeedbackEncoder, FeedbackFusion
from .fusion import FinalDenoiser, FusionMLP, SafetyControlFusion
from .geometry import CurveDecoder
from .heads import EllipseShapeHead, PathFeatureHead, TopologyHead

__all__ = ["TrajSafePlanner"]


class TrajSafePlanner(nn.Module):
    def __init__(self, model_cfg, ellipse_cfg=None, bspline_cfg=None):
        super().__init__()
        ellipse_cfg = dict(ellipse_cfg or {})
        bspline_cfg = dict(bspline_cfg or {})
        model_cfg = dict(model_cfg or {})

        # ---- the ONE trajectory representation: C B-spline controls --------
        # ``bspline.num_controls`` (codec) and ``model.num_controls`` (network /
        # diffusion state) are two views of the same number and must agree.
        c_bs = bspline_cfg.get("num_controls")
        c_md = model_cfg.get("num_controls")
        if c_bs is not None and c_md is not None and int(c_bs) != int(c_md):
            raise ValueError(
                "config mismatch: bspline.num_controls=%s != "
                "model.num_controls=%s" % (c_bs, c_md))
        self.num_controls = int(c_bs if c_bs is not None else (c_md or 32))
        if self.num_controls < 2:
            raise ValueError("num_controls must be >= 2")

        # density of the (final, plot/metric only) B-spline decode
        self.curve_points = int(bspline_cfg.get(
            "curve_points", model_cfg.get("horizon", 128)))
        # backwards compatible alias: ``horizon`` used to be the curve density
        self.horizon = self.curve_points
        # number of Skeleton / ellipse geometry queries
        self.num_safety_queries = int(model_cfg.get(
            "num_safety_queries", self.curve_points))
        # relative-bias table length of the self-attention stacks; decoupled
        # from the token count so both the C control tokens and the legacy 128
        # curve tokens can use the same table (checkpoint compatibility).
        self.rel_bias_len = int(model_cfg.get(
            "rel_bias_len",
            max(self.num_controls, self.num_safety_queries, self.curve_points)))

        self.control_space = bool(model_cfg.get("control_space", True))

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

        # ---- fixed B-spline codec (control <-> curve) ---------------------
        degree = int(bspline_cfg.get("degree", model_cfg.get("bspline_degree", 3)))
        knots_path = bspline_cfg.get("knots")
        knots = bspline_cfg.get("knots_vector")
        self.bspline = BSplineCodec(
            degree=degree, num_controls=self.num_controls,
            curve_points=self.curve_points, knots=knots, knots_path=knots_path,
            endpoint_constrained=bool(bspline_cfg.get(
                "endpoint_constrained", True)))
        # legacy-only helper (endpoint-constrained LS fit of a decoded curve)
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

        # ---- control branch ----------------------------------------------
        # (attribute names kept from the curve-token version: the module is
        #  token-count agnostic, which is what lets an old checkpoint load)
        self.traj_encoder = TrajectoryEncoder(
            self.d_model, self.spatial_pe, self.index_pe, self.curve_points,
            coord_hidden)
        self.traj_backbone = nn.ModuleList([
            TrajBlock(self.d_model, self.num_heads, self.ffn_dim,
                      self.curve_points, dropout,
                      rel_bias_len=self.rel_bias_len)
            for _ in range(self.traj_blocks)
        ])
        # ONE control head, shared by the coarse and the final decode
        self.head_p = nn.Linear(self.d_model, 2)

        # ---- historical safety feedback (optional, model.feedback) ---------
        # 上一轮 ALM 的 Q0_safe 与修正量 Delta = Q0_safe - Q0_raw 作为这一轮
        # 的额外条件，经 FeedbackEncoder -> 门控残差注入 h_ctrl。  zero_init
        # 让新模块在微调开始时严格等价于旧网络；valid = 0 时是精确恒等。
        fb_cfg = dict(model_cfg.get("feedback") or {})
        self.feedback_enabled = bool(fb_cfg.get("enabled", False))
        self.feedback_hidden = int(fb_cfg.get("hidden", 64))
        self.feedback_encoder = None
        self.feedback_fusion = None
        if self.feedback_enabled:
            self.feedback_encoder = FeedbackEncoder(
                self.d_model, hidden=self.feedback_hidden,
                zero_init=bool(fb_cfg.get("zero_init", True)),
                dropout=float(fb_cfg.get("dropout", 0.0)))
            self.feedback_fusion = FeedbackFusion(
                self.d_model, gate_bias=float(fb_cfg.get("gate_bias", 0.0)))

        # ---- skeleton branch -------------------------------------------
        self.skeleton_encoder = SkeletonEncoder(
            self.d_model, self.num_heads, self.spatial_pe, self.index_pe,
            blocks=self.skeleton_blocks, ffn_dim=self.ffn_dim,
            dropout=dropout, hidden=coord_hidden)
        self.match_block = MatchBlock(self.d_model, self.num_heads,
                                      self.ffn_dim, dropout)
        self.topology_head = TopologyHead(self.d_model, head_hidden)
        self.path_feature_head = PathFeatureHead(self.d_model, head_hidden)
        # selected-Skeleton -> safety/ellipse geometry query (own parameters)
        self.safety_query_head = PathFeatureHead(self.d_model, head_hidden)
        # safety (Q tokens) -> control (C tokens) cross attention
        self.safety_cross_attention = SafetyControlFusion(
            self.d_model, self.num_heads, dropout)

        # ---- fixed boundary decoder (no parameters, never trained) --------
        boundary_cfg = dict(model_cfg.get("boundary_decoder") or {})
        self.boundary_decoder = BoundaryDecoder(
            profile=boundary_cfg.get("profile"),
            span=boundary_cfg.get("span"))

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
            self.d_model, self.num_heads, self.ffn_dim, self.curve_points,
            blocks=self.final_blocks, dropout=dropout,
            rel_bias_len=self.rel_bias_len)

        # ---- the ONLY progress definition: a fixed buffer -----------------
        # s_i = i / (Q-1) on the selected dense Skeleton (Q safety queries).
        self.register_buffer("fixed_progress",
                             torch.linspace(0.0, 1.0, self.num_safety_queries))

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

    # --------------------------------------------------------------- helpers
    def progress_for(self, count: int, device=None, dtype=None) -> torch.Tensor:
        """Fixed progress buffer of length ``count`` (``i / (count - 1)``)."""
        count = int(count)
        if count == int(self.fixed_progress.numel()):
            dev = device if device is not None else self.fixed_progress.device
            dt = dtype if dtype is not None else self.fixed_progress.dtype
            return self.fixed_progress.to(device=dev, dtype=dt)
        return torch.linspace(0.0, 1.0, max(count, 2),
                              device=device, dtype=dtype)

    # -------------------------------------------------------------- feedback
    def feedback_features(self, h_ctrl: torch.Tensor,
                          feedback_control: torch.Tensor | None = None,
                          feedback_delta: torch.Tensor | None = None,
                          feedback_valid: torch.Tensor | None = None):
        """Build ``(h_fb [B,C,D] | None, valid [B])`` from the previous step.

        ``feedback_control`` / ``feedback_delta`` are ``[B,C,2]`` and
        ``feedback_valid`` is ``[B]`` (``[B,1]`` / ``[B,C,1]`` are accepted).
        Any missing argument means "no reliable history": the valid flag is 0
        and the fusion becomes the identity (plain diffusion behaviour), which
        is what the sampler's warm-up phase needs.  Invalid rows are zeroed
        BEFORE the encoder, so stale tensors can never leak into the network.
        """
        if not self.feedback_enabled or self.feedback_encoder is None:
            return None, None
        B, C, D = h_ctrl.shape
        dev, dt = h_ctrl.device, h_ctrl.dtype

        def _coords(x, name):
            if x is None:
                return torch.zeros(B, C, 2, device=dev, dtype=dt)
            x = x.to(device=dev, dtype=dt)
            if x.shape != (B, C, 2):
                raise ValueError("%s must be [%d,%d,2], got %s"
                                 % (name, B, C, tuple(x.shape)))
            return x

        control = _coords(feedback_control, "feedback_control")
        delta = _coords(feedback_delta, "feedback_delta")
        if feedback_valid is None:
            valid = torch.zeros(B, device=dev, dtype=dt)
        else:
            v = feedback_valid.to(device=dev).reshape(B, -1)
            valid = (v.amax(dim=1) if v.shape[1] > 0
                     else torch.zeros(B, device=dev)).to(dt)
        keep = valid[:, None, None]
        control = control * keep
        delta = delta * keep
        features = torch.cat([control, delta, keep.expand(B, C, 1)], dim=-1)
        return self.feedback_encoder(features), valid

    @staticmethod
    def _resample_curve(p: torch.Tensor, count: int) -> torch.Tensor:
        """p [B,H,2] -> [B,count,2] (linear; degenerate fallback only)."""
        if p.shape[1] == int(count):
            return p
        y = F.interpolate(p.transpose(1, 2), size=int(count), mode="linear",
                          align_corners=True)
        return y.transpose(1, 2)

    def _route_topology(self, r: torch.Tensor, topo: dict,
                        candidate_mask: torch.Tensor,
                        select_index: torch.Tensor | None):
        B = r.shape[0]
        dev = r.device
        if select_index is None:
            idx = topo["pi"].argmax(dim=-1)
        else:
            idx = select_index.to(dev).long()
        has_cand = candidate_mask.any(dim=-1)
        idx = torch.where(has_cand, idx, torch.zeros_like(idx))
        ar = torch.arange(B, device=dev)
        return idx, has_cand, ar

    # ---------------------------------------------------------------- full
    def forward_all(self, q_t: torch.Tensor, occ: torch.Tensor,
                    cond: torch.Tensor, t: torch.Tensor, ab: torch.Tensor,
                    candidate_xy: torch.Tensor, candidate_mask: torch.Tensor,
                    geometry: torch.Tensor, geometry_lengths: torch.Tensor,
                    select_index: torch.Tensor | None = None,
                    feedback_control: torch.Tensor | None = None,
                    feedback_delta: torch.Tensor | None = None,
                    feedback_valid: torch.Tensor | None = None):
        """One full forward pass of the diffusion model (mode aware).

        ``q_t`` is the C-control diffusion state.  ``select_index`` is the
        training-time m* (argmin nDTW); when it is ``None`` inference routing
        ``argmax(pi)`` is used.  Invalid rows are routed to slot 0, masked out
        of every loss, and their output degenerates to the coarse polygon.

        ``feedback_control`` [B,C,2] / ``feedback_delta`` [B,C,2] /
        ``feedback_valid`` [B] are the PREVIOUS reverse step's ALM result
        (``Q0_safe_prev`` / ``Q0_safe_prev - Q0_raw_prev`` / verification flag).
        All three default to ``None``, which means "no history": the feedback
        branch is gated off and the network runs exactly as the plain diffusion
        model (warm-up, evaluation of an old checkpoint, ablation A).
        """
        if self.control_space:
            return self.forward_controls(
                q_t, occ, cond, t, ab, candidate_xy, candidate_mask,
                geometry, geometry_lengths, select_index=select_index,
                feedback_control=feedback_control,
                feedback_delta=feedback_delta, feedback_valid=feedback_valid)
        return self.forward_curve_tokens(
            q_t, occ, cond, t, ab, candidate_xy, candidate_mask,
            geometry, geometry_lengths, select_index=select_index,
            feedback_control=feedback_control, feedback_delta=feedback_delta,
            feedback_valid=feedback_valid)

    # --------------------------------------------------- control-token chain
    def forward_controls(self, q_t: torch.Tensor, occ: torch.Tensor,
                         cond: torch.Tensor, t: torch.Tensor, ab: torch.Tensor,
                         candidate_xy: torch.Tensor,
                         candidate_mask: torch.Tensor,
                         geometry: torch.Tensor,
                         geometry_lengths: torch.Tensor,
                         select_index: torch.Tensor | None = None,
                         feedback_control: torch.Tensor | None = None,
                         feedback_delta: torch.Tensor | None = None,
                         feedback_valid: torch.Tensor | None = None):
        """Control-space chain: C control tokens, Q safety geometry queries.

        The TRAINING chain is controls-only: nothing in the loss reads a decoded
        curve, and the two purely diagnostic decodes (``input_curve``,
        ``raw_curve``) are computed under ``no_grad`` so no 128-point trajectory
        tensor ever enters the autograd graph.  Only two decodes stay connected:

        * ``coarse`` - the fallback for samples without any candidate (it feeds
          ``torch.where`` for ``center`` / ``final``);
        * ``final``  - needed by the validation metrics (curve RMSE, collision).

        Both are produced at the very END of the chain, after every learned
        module, so they cannot influence the representation.
        """
        B, C, _ = q_t.shape
        if C != self.num_controls:
            raise ValueError("q_t must be [B,%d,2], got %s"
                             % (self.num_controls, tuple(q_t.shape)))
        dev = q_t.device

        # control-space hard conditioning (clamped knots => exact curve ends)
        q_t = self.hard_control_endpoints(q_t, cond)
        # DIAGNOSTIC ONLY (never read by a loss): the noisy state's curve
        with torch.no_grad():
            p_t = self.bspline.decode_controls(q_t)

        c_g, c_e = self.scene_tokens(occ)
        h_t = self.time_pe(t.to(dev))

        # ---- control backbone -------------------------------------------
        h_ctrl = self.encode_trajectory(q_t, c_g, h_t)          # [B,C,D]

        # ---- historical safety feedback (previous ALM result) -------------
        # h_ctrl is REPLACED by the gated fusion, so the feedback conditions the
        # whole downstream chain (coarse polygon, Skeleton matching, topology,
        # path feature, safety cross attention, fusion MLP, final denoiser) -
        # this is a full fine-tune, not an inference-time bolt-on.
        h_fb, fb_valid = self.feedback_features(
            h_ctrl, feedback_control, feedback_delta, feedback_valid)
        if h_fb is not None:
            h_ctrl = self.feedback_fusion(h_ctrl, h_fb, fb_valid)

        q_coarse_raw = self.head_p(h_ctrl)                      # [B,C,2]
        q_coarse = self.boundary_decoder(q_coarse_raw, cond)    # exact ends
        coarse = self.bspline.decode_controls(q_coarse)         # [B,H,2]

        # ---- topology ----------------------------------------------------
        h_s = self.skeleton_encoder(candidate_xy)               # [B,M,L,D]
        r = self.match_block(h_ctrl, h_s, h_t)                  # [B,M,C,D]
        topo = self.topology_head(r, candidate_mask)
        idx, has_cand, ar = self._route_topology(r, topo, candidate_mask,
                                                 select_index)
        r_use = r[ar, idx]                                      # [B,C,D]
        h_path = self.path_feature_head(r_use)                  # [B,C,D]

        # ---- Q safety geometry queries on the SELECTED Skeleton -----------
        h_gamma = self.safety_query_head(h_s[ar, idx])          # [B,Q,D]
        Q = h_gamma.shape[1]
        s = self.progress_for(Q, device=dev, dtype=h_gamma.dtype)
        s = s[None].expand(B, Q)
        gamma = geometry[ar, idx]                               # [B,G,2]
        gamma_len = geometry_lengths[ar, idx]                   # [B]
        center = self.curve_decoder(gamma, gamma_len, s)        # [B,Q,2]
        fallback = self._resample_curve(coarse, Q)
        center = torch.where(has_cand[:, None, None], center, fallback)

        h_ell, a_e = self.ellipse_geometry(h_gamma, center, c_e, h_t, ab)
        shape = self.ellipse_shape_head(h_ell, h_t)

        # ---- safety -> control cross attention + fusion --------------------
        h_safe_ctrl = self.safety_cross_attention(h_path, h_ell, h_t)
        f = self.fusion_mlp(h_ctrl, h_path, h_safe_ctrl)
        h_clean = self.final_denoiser(f, c_g, h_t)
        q_raw_final = self.head_p(h_clean)                      # [B,C,2]
        q_final = self.boundary_decoder(q_raw_final, cond)      # [B,C,2]
        # DIAGNOSTIC ONLY: the free network prediction decoded, i.e. "x0 before
        # the fixed boundary decoder" (dashboard / plots; never a loss input)
        with torch.no_grad():
            raw_curve = self.bspline.decode_controls(q_raw_final)
        final = self.bspline.decode_controls(q_final)
        final = torch.where(has_cand[:, None, None], final, coarse)

        out = {
            "input_control": q_t,
            "input_curve": p_t,
            "q_coarse_raw": q_coarse_raw,
            "coarse_raw": q_coarse_raw,
            "q_coarse": q_coarse,
            "coarse": coarse,
            "q_raw_final": q_raw_final,
            "control_raw": q_raw_final,
            "control": q_final,
            "raw_curve": raw_curve,
            "final": final,
            "H_traj": h_ctrl,
            "H_ctrl": h_ctrl,
            "H_fb": h_fb,
            "feedback_valid": fb_valid,
            "H_S": h_s,
            "R": r,
            "topo": topo,
            "selected_idx": idx,
            "has_candidate": has_cand,
            "R_use": r_use,
            "H_path": h_path,
            "H_safety": h_gamma,
            "H_safety_ell": h_ell,
            "A_safety": h_safe_ctrl,
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
            self._assert_forward_shapes(out, B, C, Q)
        return out

    # --------------------------------------------------------- legacy chain
    def forward_curve_tokens(self, q_t: torch.Tensor, occ: torch.Tensor,
                             cond: torch.Tensor, t: torch.Tensor,
                             ab: torch.Tensor, candidate_xy: torch.Tensor,
                             candidate_mask: torch.Tensor,
                             geometry: torch.Tensor,
                             geometry_lengths: torch.Tensor,
                             select_index: torch.Tensor | None = None,
                             feedback_control: torch.Tensor | None = None,
                             feedback_delta: torch.Tensor | None = None,
                             feedback_valid: torch.Tensor | None = None):
        """Pre-refactor chain: the network runs on the decoded 128-point curve.

        Kept ONLY so a checkpoint trained before the control-space refactor can
        still be replayed (demo / regression).  Training uses
        :meth:`forward_controls`; see ``checkpoint.detect_architecture``.
        The historical-feedback arguments are accepted for interface parity and
        ignored: this chain has no control-token feature stream to condition.
        """
        B, C, _ = q_t.shape
        if C != self.num_controls:
            raise ValueError("q_t must be [B,%d,2], got %s"
                             % (self.num_controls, tuple(q_t.shape)))
        dev = q_t.device
        H = self.curve_points

        q_t = self.hard_control_endpoints(q_t, cond)
        p_t = self.bspline.decode_controls(q_t)

        c_g, c_e = self.scene_tokens(occ)
        h_t = self.time_pe(t.to(dev))

        h_traj = self.encode_trajectory(p_t, c_g, h_t)          # [B,H,D]
        coarse_raw = self.head_p(h_traj)                        # curve-space
        q_coarse = self.traj_to_control(coarse_raw, cond)       # LS fit
        coarse = self.bspline.decode_controls(q_coarse)

        h_s = self.skeleton_encoder(candidate_xy)               # [B,M,L,D]
        r = self.match_block(h_traj, h_s, h_t)                  # [B,M,H,D]
        topo = self.topology_head(r, candidate_mask)
        idx, has_cand, ar = self._route_topology(r, topo, candidate_mask,
                                                 select_index)
        r_use = r[ar, idx]
        h_path = self.path_feature_head(r_use)

        s = torch.linspace(0.0, 1.0, H, device=dev, dtype=p_t.dtype)
        s = s[None].expand(B, H)
        gamma = geometry[ar, idx]
        gamma_len = geometry_lengths[ar, idx]
        center = self.curve_decoder(gamma, gamma_len, s)
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
            "q_coarse_raw": q_coarse,
            "coarse_raw": coarse_raw,
            "q_coarse": q_coarse,
            "coarse": coarse,
            "q_raw_final": q_final,
            "control_raw": q_final,
            "control": q_final,
            "raw_curve": raw_final,
            "final": final,
            "H_traj": h_traj,
            "H_ctrl": None,
            "H_fb": None,
            "feedback_valid": None,
            "H_S": h_s,
            "R": r,
            "topo": topo,
            "selected_idx": idx,
            "has_candidate": has_cand,
            "R_use": r_use,
            "H_path": h_path,
            "H_safety": None,
            "H_safety_ell": h_ell,
            "A_safety": None,
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
            self._assert_legacy_shapes(out, B, H)
        return out

    # ------------------------------------------------------------ assertions
    def _assert_common_shapes(self, out: dict, B: int) -> None:
        D = self.d_model
        C = self.num_controls
        M = out["H_S"].shape[1]
        L = out["H_S"].shape[2]
        assert out["input_control"].shape == (B, C, 2)
        assert out["H_S"].shape == (B, M, L, D), out["H_S"].shape
        assert out["topo"]["pi"].shape == (B, M), out["topo"]["pi"].shape
        assert out["coarse"].shape[-1] == 2
        assert out["final"].shape[-1] == 2
        assert out["ellipse"]["center"].shape[-1] == 2
        assert out["ellipse"]["shape4"].shape[-1] == 4
        a, b = out["ellipse"]["a"], out["ellipse"]["b"]
        assert bool((a >= b - 1e-7).all())
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

    def _assert_forward_shapes(self, out: dict, B: int, C: int, Q: int) -> None:
        D = self.d_model
        H = self.curve_points
        self._assert_common_shapes(out, B)
        assert C == self.num_controls
        assert out["H_traj"].shape == (B, C, D), out["H_traj"].shape
        assert out["R"].shape == (B, out["H_S"].shape[1], C, D)
        assert out["H_path"].shape == (B, C, D), out["H_path"].shape
        assert out["F"].shape == (B, C, D), out["F"].shape
        assert out["H_clean"].shape == (B, C, D), out["H_clean"].shape
        assert out["control"].shape == (B, C, 2), out["control"].shape
        assert out["q_coarse"].shape == (B, C, 2), out["q_coarse"].shape
        assert out["q_raw_final"].shape == (B, C, 2)
        if self.feedback_enabled:
            assert out["H_fb"].shape == (B, C, D), out["H_fb"].shape
            assert out["feedback_valid"].shape == (B,)
            assert bool(torch.isfinite(out["H_fb"]).all())
        assert out["raw_curve"].shape == (B, H, 2)
        assert out["H_safety"].shape == (B, Q, D), out["H_safety"].shape
        assert out["A_safety"].shape == (B, C, D), out["A_safety"].shape
        assert out["ellipse"]["progress"].shape == (B, Q)
        assert out["ellipse"]["center"].shape == (B, Q, 2)
        assert out["ellipse"]["H_ell"].shape == (B, Q, D)
        assert out["ellipse"]["shape4"].shape == (B, Q, 4)
        s = out["ellipse"]["progress"]
        assert torch.allclose(s[:, 0], torch.zeros_like(s[:, 0]), atol=1e-6)
        assert torch.allclose(s[:, -1], torch.ones_like(s[:, -1]), atol=1e-6)
        assert bool((s[:, 1:] > s[:, :-1]).all())
        assert out["final"].shape == (B, H, 2)
        assert out["coarse"].shape == (B, H, 2)

    def _assert_legacy_shapes(self, out: dict, B: int, H: int) -> None:
        D = self.d_model
        self._assert_common_shapes(out, B)
        assert out["H_traj"].shape == (B, H, D), out["H_traj"].shape
        assert out["R"].shape == (B, out["H_S"].shape[1], H, D)
        assert out["H_path"].shape == (B, H, D)
        assert out["F"].shape == (B, H, D)
        assert out["H_clean"].shape == (B, H, D)
        assert out["ellipse"]["center"].shape == (B, H, 2)
        assert out["ellipse"]["H_ell"].shape == (B, H, D)
        assert out["ellipse"]["shape4"].shape == (B, H, 4)
        assert out["final"].shape == (B, H, 2)
        assert out["coarse"].shape == (B, H, 2)
