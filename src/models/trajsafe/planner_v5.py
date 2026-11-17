"""New single-path framework (S2): ONE clean x0, no coarse branch, no topology.

    q_t [B,C,2]              diffusion state (C = 32 controls, endpoints + heading)
      -> Control Encoder
      -> Control Backbone           (N_T = 1 TrajBlock)
      -> [optional] ALM feedback    (gated residual, valid=0 -> exact identity)
      -> [optional] single path branch:  SkeletonEncoder(1 path) -> MatchBlock
                                        -> PathFeatureHead -> h_path [B,C,D]
      -> [optional] safety branch:   h_path -> resample Q=64 -> safety_query_head
                                        -> EllipseGeometry -> EllipseShapeHead
                                        -> SafetyControlFusion -> h_safety [B,C,D]
      -> FusionMLP(3 streams) + EXPLICIT absent-stream embeddings
      -> Final Denoiser            (N_F = 4 TrajBlock)
      -> head_p -> BoundaryDecoder (position + heading) -> x0 controls
      -> decode -> 128-point curve (plot / metric only)

Deliberate differences from ``planner.py`` (see the framework plan §2, §13):

* there is **no** ``q_coarse`` / ``q_raw_coarse`` / ``coarse`` output at all;
* there is **no** ``TopologyHead``: the path input is ONE ``active_path [B,128,2]``;
* a missing path / safety stream is represented by a **learned null token plus a
  learned absent-stream embedding** (never by "all zeros == absent"), so step 1
  of the four-step sampler is a trained mode, not an accident;
* the endpoint condition optionally includes **heading** (Q1 / Q[C-2] pushed to
  ``S + ell * t0`` and ``G - ell * t1`` with the same profile as the
  BoundaryDecoder, so the fix is smooth).

The legacy ``planner.py`` is kept untouched only so old checkpoints can still be
loaded by ``--arch auto`` for reproduction; nothing in the new chain imports it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...geometry.bspline import BSplineCodec
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
from .heads import EllipseShapeHead, PathFeatureHead

__all__ = ["TrajSafePlannerV5", "resample_tokens", "resample_curve"]


def resample_curve(path: torch.Tensor, count: int) -> torch.Tensor:
    """path [B,L,2] -> [B,count,2] uniform in arc length (differentiable)."""
    B, L, _ = path.shape
    if L == count:
        return path
    seg = torch.linalg.norm(path[:, 1:] - path[:, :-1], dim=-1)      # [B,L-1]
    cum = torch.cumsum(seg, dim=-1)
    total = cum[:, -1:].clamp_min(1e-9)
    s = torch.linspace(0.0, 1.0, count, device=path.device, dtype=path.dtype)
    target = s[None] * total                                          # [B,count]
    idx = torch.searchsorted(cum, target.contiguous(), right=False)
    idx = idx.clamp(0, L - 2)
    seg_len = seg.gather(1, idx).clamp_min(1e-9)
    prev = cum.gather(1, idx) - seg_len
    w = ((target - prev) / seg_len).clamp(0.0, 1.0)[..., None]
    p0 = path.gather(1, idx[..., None].expand(-1, -1, 2))
    p1 = path.gather(1, (idx + 1)[..., None].expand(-1, -1, 2))
    out = p0 + w * (p1 - p0)
    out[:, 0] = path[:, 0]
    out[:, -1] = path[:, -1]
    return out


def resample_tokens(h: torch.Tensor, count: int) -> torch.Tensor:
    """h [B,L,D] -> [B,count,D] linear interpolation over the token index."""
    B, L, D = h.shape
    if L == count:
        return h
    x = torch.linspace(0, L - 1, count, device=h.device, dtype=h.dtype)
    i0 = x.floor().long().clamp(0, L - 1)
    i1 = (i0 + 1).clamp(0, L - 1)
    w = (x - i0.to(x.dtype))[None, :, None]
    return h.gather(1, i0[None, :, None].expand(B, -1, D)) * (1 - w) \
        + h.gather(1, i1[None, :, None].expand(B, -1, D)) * w


class TrajSafePlannerV5(nn.Module):
    """Single clean x0, single path input, optional heading endpoint condition."""

    def __init__(self, model_cfg, ellipse_cfg=None, bspline_cfg=None):
        super().__init__()
        model_cfg = dict(model_cfg or {})
        ellipse_cfg = dict(ellipse_cfg or {})
        bspline_cfg = dict(bspline_cfg or {})

        c_bs = bspline_cfg.get("num_controls")
        c_md = model_cfg.get("num_controls")
        if c_bs is not None and c_md is not None and int(c_bs) != int(c_md):
            raise ValueError("config mismatch: bspline.num_controls=%s != "
                             "model.num_controls=%s" % (c_bs, c_md))
        self.num_controls = int(c_bs if c_bs is not None else (c_md or 32))
        self.curve_points = int(bspline_cfg.get(
            "curve_points", model_cfg.get("horizon", 128)))
        self.horizon = self.curve_points
        self.num_safety_queries = int(model_cfg.get("num_safety_queries", 64))
        self.path_points = int(model_cfg.get("path_points", self.curve_points))
        self.rel_bias_len = int(model_cfg.get(
            "rel_bias_len", max(self.num_controls, self.num_safety_queries,
                                self.curve_points, self.path_points)))

        self.d_model = int(model_cfg.get("d_model", 128))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.traj_blocks = int(model_cfg.get("traj_blocks", 1))
        self.final_blocks = int(model_cfg.get("final_blocks", 3))
        self.skeleton_blocks = int(model_cfg.get("skeleton_blocks", 2))
        self.ffn_dim = int(model_cfg.get("ffn_dim", 512))
        self.map_res = int(model_cfg.get("map_res", 256))
        self.global_res = int(model_cfg.get("global_mem_res", 16))
        dropout = float(model_cfg.get("dropout", 0.0))
        head_hidden = int(model_cfg.get("head_hidden", 256))
        coord_hidden = int(model_cfg.get("coord_hidden", 256))
        geo_sigma = float(model_cfg.get("geo_sigma", 0.25))
        geo_bias_clip = float(model_cfg.get("geo_bias_clip", 8.0))

        # heading endpoint condition (§8): ell is a GEOMETRIC tangent length
        hcfg = dict(model_cfg.get("heading") or {})
        self.heading_enabled = bool(hcfg.get("enabled", True))
        self.heading_goal = bool(hcfg.get("goal", True))
        self.heading_ell = float(hcfg.get("ell_scene", 0.04))
        self.heading_ell_min = float(hcfg.get("ell_min_scene", 0.01))
        self.heading_ell_max_frac = float(hcfg.get("ell_max_frac", 0.20))

        fb_cfg = dict(model_cfg.get("feedback") or {})
        self.feedback_enabled = bool(fb_cfg.get("enabled", True))
        self.feedback_hidden = int(fb_cfg.get("hidden", 64))

        degree = int(bspline_cfg.get("degree", 3))
        self.bspline = BSplineCodec(
            degree=degree, num_controls=self.num_controls,
            curve_points=self.curve_points,
            knots=bspline_cfg.get("knots_vector"),
            knots_path=bspline_cfg.get("knots"),
            endpoint_constrained=bool(bspline_cfg.get(
                "endpoint_constrained", True)))

        self.scene_cnn = SceneCNN(d_model=self.d_model, res=self.map_res,
                                  global_res=self.global_res)
        self.spatial_pe = Sinusoidal2DPositionEmbedding(self.d_model)
        self.index_pe = Sinusoidal1DPositionEmbedding(self.d_model)
        self.time_pe = SinusoidalTimestepEmbedding(self.d_model)

        self.control_encoder = TrajectoryEncoder(
            self.d_model, self.spatial_pe, self.index_pe, self.curve_points,
            coord_hidden)
        self.control_backbone = nn.ModuleList([
            TrajBlock(self.d_model, self.num_heads, self.ffn_dim,
                      self.curve_points, dropout, rel_bias_len=self.rel_bias_len)
            for _ in range(self.traj_blocks)])

        self.path_encoder = SkeletonEncoder(
            self.d_model, self.num_heads, self.spatial_pe, self.index_pe,
            blocks=self.skeleton_blocks, ffn_dim=self.ffn_dim, dropout=dropout,
            hidden=coord_hidden)
        self.match_block = MatchBlock(self.d_model, self.num_heads,
                                      self.ffn_dim, dropout)
        self.path_head = PathFeatureHead(self.d_model, head_hidden)
        self.safety_query_head = PathFeatureHead(self.d_model, head_hidden)
        self.ellipse_geometry = EllipseGeometry(
            self.d_model, self.num_heads, self.spatial_pe,
            map_res=self.global_res, geo_sigma=geo_sigma,
            geo_bias_clip=geo_bias_clip, hidden=coord_hidden, dropout=dropout)
        self.ellipse_shape_head = EllipseShapeHead(
            self.d_model, head_hidden, dropout)
        self.safety_fusion = SafetyControlFusion(
            self.d_model, self.num_heads, dropout)

        # explicit absent-stream representation (trained, never "zeros")
        self.path_null = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.safety_null = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.absent_emb = nn.Parameter(torch.zeros(2, self.d_model))
        nn.init.normal_(self.path_null, std=0.02)
        nn.init.normal_(self.safety_null, std=0.02)
        nn.init.normal_(self.absent_emb, std=0.02)

        self.fusion = FusionMLP(self.d_model, num_streams=3)
        self.final_denoiser = FinalDenoiser(
            self.d_model, self.num_heads, self.ffn_dim, self.curve_points,
            blocks=self.final_blocks, dropout=dropout,
            rel_bias_len=self.rel_bias_len)
        self.head_p = nn.Linear(self.d_model, 2)

        boundary_cfg = dict(model_cfg.get("boundary_decoder") or {})
        self.boundary_decoder = BoundaryDecoder(
            profile=boundary_cfg.get("profile"),
            span=boundary_cfg.get("span"))

        if self.feedback_enabled:
            self.feedback_encoder = FeedbackEncoder(
                self.d_model, hidden=self.feedback_hidden,
                zero_init=bool(fb_cfg.get("zero_init", True)),
                dropout=float(fb_cfg.get("dropout", 0.0)))
            self.feedback_fusion = FeedbackFusion(
                self.d_model, gate_bias=float(fb_cfg.get("gate_bias", 0.0)))
        else:
            self.feedback_encoder = None
            self.feedback_fusion = None

        self.register_buffer("fixed_progress",
                             torch.linspace(0.0, 1.0, self.num_safety_queries))

    # ------------------------------------------------------------------ utils
    @property
    def control_space(self) -> bool:
        return True

    def scene_tokens(self, occ: torch.Tensor):
        """occ [B,1,R,R] -> C_G [B,16*16,D] (the ONLY map feature)."""
        from ...geometry.ellipse_raster import scene_grid_centres
        feat = self.scene_cnn(occ)
        flat = feat.flatten(2).transpose(1, 2)
        grid = scene_grid_centres(self.global_res, flat.device, dtype=flat.dtype)
        return flat + self.spatial_pe(grid)[None]

    def decode_controls(self, q: torch.Tensor) -> torch.Tensor:
        return self.bspline.decode_controls(q)

    def _ell_of(self, cond: torch.Tensor) -> torch.Tensor:
        """Geometric tangent length (scene units), clipped per sample."""
        d = torch.linalg.norm(cond[:, 1] - cond[:, 0], dim=-1)        # [B]
        base = torch.full_like(d, self.heading_ell)
        hi = self.heading_ell_max_frac * d + 1e-6
        return torch.maximum(torch.minimum(base, hi),
                             torch.full_like(d, self.heading_ell_min))

    def apply_heading(self, q: torch.Tensor, cond: torch.Tensor,
                      heading: torch.Tensor | None = None,
                      heading_valid: torch.Tensor | None = None,
                      valid_goal: bool | None = None) -> torch.Tensor:
        """Push Q1 / Q[C-2] toward the endpoint tangents, keeping Q0/Q[-1].

        Uses the BoundaryDecoder profile so the neighbours move coherently (the
        same "smooth fix" idea as the position condition, not a hard teleport).
        ``heading`` [B,2,2] = (t0, t1) unit vectors in SCENE coordinates.
        """
        if not self.heading_enabled or heading is None:
            return q
        B, C, _ = q.shape
        dev, dt = q.device, q.dtype
        ell = self._ell_of(cond)                                      # [B]
        valid = (heading_valid if heading_valid is not None
                 else torch.ones(B, 2, device=dev, dtype=torch.bool))
        valid = valid.bool()
        ws, wg = self.boundary_decoder.weights(C, device=dev, dtype=dt)
        # profile starting at the SECOND control (Q0 must stay exact)
        prof = ws.clone()
        prof[0] = 0.0
        q_out = q
        # The DIRECTION is the constraint, the segment length is the network's:
        # keeping the current |Q1-Q0| (clipped) means the soft profile blend
        # rotates the tangent instead of fighting the learned control spacing.
        seg = torch.linalg.norm(cond[:, 1] - cond[:, 0], dim=-1, keepdim=True)
        hi = 0.30 * seg
        def _len(cur):
            return torch.maximum(torch.minimum(cur, hi),
                                 torch.full_like(cur, float(self.heading_ell_min)))
        # start tangent: target for Q1
        t0 = heading[:, 0] / heading[:, 0].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        l0 = _len((q_out[:, 1] - q_out[:, 0]).norm(dim=-1, keepdim=True))
        tgt1 = cond[:, 0] + l0 * t0
        d1 = (tgt1 - q_out[:, 1]) * valid[:, 0][:, None].to(dt)
        q_out = q_out + prof[None, :, None] * d1[:, None, :]
        if self.heading_goal and (valid_goal is None or valid_goal):
            t1 = heading[:, 1] / heading[:, 1].norm(dim=-1, keepdim=True).clamp_min(1e-9)
            l1 = _len((q_out[:, C - 1] - q_out[:, C - 2]).norm(dim=-1, keepdim=True))
            tgt2 = cond[:, 1] - l1 * t1
            d2 = (tgt2 - q_out[:, C - 2]) * valid[:, 1][:, None].to(dt)
            prof_g = torch.flip(prof, dims=[0])
            q_out = q_out + prof_g[None, :, None] * d2[:, None, :]
        # the two endpoints stay exact
        q_out = q_out.clone()
        q_out[:, 0] = cond[:, 0]
        q_out[:, C - 1] = cond[:, 1]
        return q_out

    def hard_control_endpoints(self, q: torch.Tensor, cond: torch.Tensor,
                               heading: torch.Tensor | None = None,
                               heading_valid: torch.Tensor | None = None
                               ) -> torch.Tensor:
        q = q.clone()
        q[:, 0] = cond[:, 0]
        q[:, -1] = cond[:, 1]
        return self.apply_heading(q, cond, heading, heading_valid)

    # ---------------------------------------------------------------- forward
    def forward(self, q_t: torch.Tensor, occ: torch.Tensor, cond: torch.Tensor,
                t: torch.Tensor, ab: torch.Tensor,
                active_path: torch.Tensor | None = None,
                path_valid: torch.Tensor | None = None,
                feedback_delta: torch.Tensor | None = None,
                feedback_valid: torch.Tensor | None = None,
                heading: torch.Tensor | None = None,
                heading_valid: torch.Tensor | None = None,
                use_safety: bool = True):
        """One forward pass of the new framework.

        ``active_path`` [B,128,2] is the ONLY path input (None on step 1).
        ``use_safety=False`` disables the ellipse/safety stream as well.
        """
        B, C, _ = q_t.shape
        if C != self.num_controls:
            raise ValueError("q_t must be [B,%d,2], got %s"
                             % (self.num_controls, tuple(q_t.shape)))
        dev = q_t.device

        q_t = self.hard_control_endpoints(q_t, cond, heading, heading_valid)
        c_g = self.scene_tokens(occ)                                    # [B,N,D]
        h_t = self.time_pe(t.to(dev))

        h_ctrl = self.control_encoder(q_t)                              # [B,C,D]
        for blk in self.control_backbone:
            h_ctrl = blk(h_ctrl, c_g, h_t)

        has_fb = (self.feedback_enabled and feedback_delta is not None)
        if has_fb:
            # FeedbackEncoder consumes the per-control FEATURE [B,C,3] =
            # (delta_xy, valid) — the mask is applied by the gated fusion, and
            # valid == 0 makes the whole branch the exact identity.
            valid_f = (feedback_valid if feedback_valid is not None
                       else torch.ones(B, device=dev, dtype=torch.bool))
            feats = torch.cat([feedback_delta,
                               valid_f.to(feedback_delta.dtype)[:, None, None]
                               .expand(-1, C, 1)], dim=-1)
            h_fb = self.feedback_encoder(feats)
            h_ctrl = self.feedback_fusion(h_ctrl, h_fb, valid_f)

        has_path = active_path is not None
        h_path_tokens = None
        if has_path:
            ap = active_path
            if ap.shape[1] != self.path_points:
                ap = resample_curve(ap, self.path_points)
            h_s = self.path_encoder(ap[:, None])                        # [B,1,L,D]
            r = self.match_block(h_ctrl, h_s, h_t)[:, 0]                # [B,C,D]
            h_path = self.path_head(r)                                  # [B,C,D]
            h_path_tokens = h_s[:, 0]                                   # [B,L,D]
        else:
            h_path = self.path_null.expand(B, C, self.d_model)

        has_safety = bool(has_path and use_safety)
        ell = {}
        if has_safety:
            centers = resample_curve(active_path, self.num_safety_queries)
            q_tok = resample_tokens(h_path_tokens, self.num_safety_queries)
            h_q = self.safety_query_head(q_tok)                         # [B,Q,D]
            h_ell, _ = self.ellipse_geometry(h_q, centers, c_g, h_t, ab)
            shape = self.ellipse_shape_head(h_ell, h_t)
            h_safety = self.safety_fusion(h_ctrl, h_ell, h_t)           # [B,C,D]
            ell = {"center": centers, "progress": self.fixed_progress[None].expand(B, -1),
                   "a": shape["a"], "b": shape["b"], "theta": shape["theta"],
                   "shape4": shape["shape4"], "raw": shape["raw"]}
        else:
            h_safety = self.safety_null.expand(B, C, self.d_model)

        f = self.fusion(h_ctrl, h_path, h_safety)
        if not has_path:
            f = f + self.absent_emb[0][None, None, :]
        if not has_safety:
            f = f + self.absent_emb[1][None, None, :]
        f = self.final_denoiser(f, c_g, h_t)
        q_raw = self.head_p(f)                                          # [B,C,2]
        q_out = self.boundary_decoder(q_raw, cond)
        q_out = self.apply_heading(q_out, cond, heading, heading_valid)
        final = self.bspline.decode_controls(q_out)                     # [B,128,2]

        return {"control": q_out,
                "q_raw_final": q_raw,
                "final": final,
                "has_path": torch.full((B,), bool(has_path), device=dev),
                "has_safety": torch.full((B,), bool(has_safety), device=dev),
                "has_feedback": torch.full((B,), bool(has_fb), device=dev),
                "ellipse": ell,
                "c_g": c_g, "h_t": h_t}

    # --------------------------------------------------------- batch helpers
    def ellipse_params(self, out: dict):
        """Convenience view used by losses / metrics (None when absent)."""
        return out["ellipse"] or None
