"""Native-trajectory-point diffusion model WITH the S-C48 safety-geometry branch.

State = 128 trajectory points (the delivered object itself), not B-spline controls.
Architecture = the current 1-layer scene encoder + 8-layer final denoiser (the "l1_final8"
point model) plus the safety branch of the S-C48 planner:

    C_G      = SceneCNN(occ)                              # [B,256,D]  1-layer encoder
    H_traj   = TrajectoryEncoder(p_t)                      # [B,H,D]    H = 128 points
    H_S      = SkeletonEncoder(candidate_xy)               # [B,M,L,D]  M candidates
    R        = MatchBlock(H_traj, H_S)                     # [B,M,H,D]
    pi       = TopologyHead(R, candidate_mask)             # [B,M]
    m        = select_index (train) / argmax(pi) (infer)
    H_path   = PathFeatureHead(R[m])                       # [B,H,D]
    H_Gamma  = safety_query_head(H_S[m])                   # [B,Q,D]   Q = num_safety_queries
    c_i      = Gamma_m(s_i)                                # anchor points along candidate m
    H_ell    = EllipseGeometry(H_Gamma, c_i, C_G, h_t, ab) # [B,Q,D]
    shape4   = EllipseShapeHead(H_ell, h_t)                # [B,Q,4]
    A_safe   = SafetyControlFusion(H_path, H_ell, h_t)     # [B,H,D]
    F        = FusionMLP(H_traj, H_path, A_safe)           # [B,H,D]
    coarse   = head(H_traj)
    clean    = pin_endpoints(head(FinalDenoiser(F, C_G, h_t)))

Differences from `TrajSafePlanner.forward_curve_tokens` (declared, all forced by the state
change):
  * the diffusion state IS the 128-point curve, so there is no `bspline.decode_controls`
    and no control-token stream at all (`num_controls` disappears);
  * `H_traj` replaces `H_ctrl` as the query set of every C-token module, so C = 128
    instead of 48 in `MatchBlock` / `PathFeatureHead` / `SafetyControlFusion` / `FusionMLP`;
  * the output is the clean prediction itself (x0 target), with the endpoints pinned by
    `pin_endpoints` (the control model pinned 2 of 48 controls, this pins 2 of 128 points);
  * no feedback modules: the S-C48 ablation this mirrors is `train.feedback.rollout=false`.
"""
import torch
import torch.nn as nn

from src.models.common.scene_cnn import SceneCNN
from src.models.position_encoding import (Sinusoidal1DPositionEmbedding,
                                          Sinusoidal2DPositionEmbedding,
                                          SinusoidalTimestepEmbedding)
from src.models.trajsafe.blocks import MatchBlock
from src.models.trajsafe.ellipse import EllipseGeometry
from src.models.trajsafe.encoders import SkeletonEncoder, TrajectoryEncoder
from src.models.trajsafe.fusion import FinalDenoiser, FusionMLP, SafetyControlFusion
from src.models.trajsafe.geometry import gather_dense_path_points
from src.models.trajsafe.heads import EllipseShapeHead, PathFeatureHead, TopologyHead
from src.geometry.ellipse_raster import scene_grid_centres


def pin_endpoints(points: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Hard-pin first/last waypoint to the start/goal (same rule as the point baseline)."""
    p = points.clone()
    p[:, 0, :] = cond[:, 0, :]
    p[:, -1, :] = cond[:, 1, :]
    return p


class SafetyPointDiffuser(nn.Module):
    def __init__(self, model_cfg: dict, ellipse_cfg: dict | None = None,
                 num_points: int = 128, num_safety_queries: int | None = None):
        super().__init__()
        e = dict(ellipse_cfg or {})
        d = int(model_cfg.get("d_model", 128))
        self.d_model = d
        self.num_points = int(num_points)
        self.horizon = self.num_points
        self.num_safety_queries = int(num_safety_queries or model_cfg.get("num_safety_queries", self.num_points))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.ffn_dim = int(model_cfg.get("ffn_dim", 512))
        self.dropout = float(model_cfg.get("dropout", 0.0))
        head_hidden = int(model_cfg.get("head_hidden", 256))
        self.safety_features_enabled = bool((model_cfg.get("safety_features") or {}).get("enabled", True))

        self.spatial_pe = Sinusoidal2DPositionEmbedding(d)
        self.index_pe = Sinusoidal1DPositionEmbedding(d)
        self.time_pe = SinusoidalTimestepEmbedding(d)

        self.scene_cnn = SceneCNN(d, int(model_cfg.get("map_res", 256)),
                                 int(model_cfg.get("global_mem_res", 16)))
        self.traj_encoder = TrajectoryEncoder(d, self.spatial_pe, self.index_pe,
                                              self.num_points,
                                              hidden=int(model_cfg.get("coord_hidden", 256)))
        self.coarse_head = nn.Linear(d, 2)
        self.head = nn.Linear(d, 2)
        self.fusion_mlp = FusionMLP(d, 3 if self.safety_features_enabled else 2)
        self.final_denoiser = FinalDenoiser(d, self.num_heads, self.ffn_dim, self.num_points,
                                            blocks=int(model_cfg.get("final_blocks", 8)),
                                            dropout=self.dropout,
                                            rel_bias_len=model_cfg.get("rel_bias_len"))

        # --- candidate-topology branch (the safety queries are anchored on it) ---
        self.skeleton_encoder = SkeletonEncoder(d, self.num_heads, self.spatial_pe, self.index_pe,
                                                blocks=int(model_cfg.get("skeleton_blocks", 2)),
                                                ffn_dim=self.ffn_dim, dropout=self.dropout)
        self.match_block = MatchBlock(d, self.num_heads, self.ffn_dim, dropout=self.dropout)
        self.topology_head = TopologyHead(d, head_hidden)
        self.path_head = PathFeatureHead(d, head_hidden)

        if self.safety_features_enabled:
            self.safety_query_head = PathFeatureHead(d, head_hidden)
            self.ellipse_geometry = EllipseGeometry(
                d, self.num_heads, self.spatial_pe,
                # the v4 EllipseGeometry biases over the MAP grid (16x16 tokens), not a
                # separate 32x32 memory: `map_res` must match encode_scene's token grid.
                map_res=int(model_cfg.get("global_mem_res", 16)),
                geo_sigma=float(e.get("geo_sigma", model_cfg.get("geo_sigma", 0.25))),
                geo_bias_clip=float(e.get("geo_bias_clip", model_cfg.get("geo_bias_clip", 8.0))),
                hidden=head_hidden, dropout=self.dropout)
            self.ellipse_shape_head = EllipseShapeHead(d, head_hidden, dropout=self.dropout)
            self.safety_cross_attention = SafetyControlFusion(d, self.num_heads, dropout=self.dropout)
        else:
            self.safety_query_head = None
            self.ellipse_geometry = None
            self.ellipse_shape_head = None
            self.safety_cross_attention = None

    # ------------------------------------------------------------------ encoders
    def encode_scene(self, occ: torch.Tensor) -> torch.Tensor:
        """occ [B,1,R,R] -> C_G [B,N_G,D] (16x16 tokens, the only map feature)."""
        feat = self.scene_cnn(occ)
        flat = feat.flatten(2).transpose(1, 2)
        res = int(round(float(flat.shape[1]) ** 0.5))
        grid = scene_grid_centres(res, flat.device, dtype=flat.dtype)
        return flat + self.spatial_pe(grid)[None]

    def forward_from_memory(self, points, memory, cond, t, ab, candidate_xy, candidate_mask,
                            geometry, geometry_lengths, select_index=None):
        """One denoising forward. All tensors are batched; returns the clean prediction."""
        B = points.shape[0]
        dev = points.device
        p_t = pin_endpoints(points, cond)
        h_t = self.time_pe(t.to(dev).long())
        h_traj = self.traj_encoder(p_t)                        # [B,H,D]

        h_s = self.skeleton_encoder(candidate_xy)              # [B,M,L,D]
        r = self.match_block(h_traj, h_s, h_t)                 # [B,M,H,D]
        topo = self.topology_head(r, candidate_mask)           # {'logits','pi'}
        pi = topo["pi"]
        if select_index is None:
            idx = pi.argmax(dim=-1)
        else:
            sel = select_index.to(dev).long().reshape(B)
            idx = torch.where(sel < 0, pi.argmax(dim=-1), sel)
        ar = torch.arange(B, device=dev)
        h_path = self.path_head(r[ar, idx])                    # [B,H,D]

        out = {"pi": pi, "topology_logits": topo["logits"], "selected": idx}
        if self.safety_features_enabled:
            q = self.num_safety_queries
            h_gamma = self.safety_query_head(h_s[ar, idx])     # [B,Q,D]
            s_i = torch.linspace(0., 1., q, device=dev, dtype=p_t.dtype)[None].expand(B, q)
            center = gather_dense_path_points(geometry[ar, idx], geometry_lengths[ar, idx], s_i)  # [B,Q,2]
            h_ell, a_e = self.ellipse_geometry(h_gamma, center, memory, h_t, ab)
            out["shape4"] = self.ellipse_shape_head(h_ell, h_t)   # [B,Q,4]
            out["ellipse_attention"] = a_e
            a_safe = self.safety_cross_attention(h_path, h_ell, h_t)
            fused = self.fusion_mlp(h_traj, h_path, a_safe)
        else:
            fused = self.fusion_mlp(h_traj, h_path)
        coarse = self.coarse_head(h_traj)                      # [B,H,2]
        delta = self.head(self.final_denoiser(fused, memory, h_t))
        out["coarse"] = pin_endpoints(coarse, cond)
        out["points"] = pin_endpoints(delta, cond)
        return out

    def forward(self, points, occ, cond, t, ab, candidate_xy, candidate_mask,
                geometry, geometry_lengths, select_index=None):
        memory = self.encode_scene(occ)
        return self.forward_from_memory(points, memory, cond, t, ab, candidate_xy,
                                        candidate_mask, geometry, geometry_lengths,
                                        select_index=select_index)
