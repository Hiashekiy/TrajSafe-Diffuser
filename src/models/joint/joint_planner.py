"""Occupancy-Conditioned Joint Trajectory–Geometry Diffusion Transformer (V1).

docs/联合扩散.md, sections #5-#25.  One forward, one interleaved sequence:

    Z_t = [T_1,E_1,...,T_128,E_128]          (256 tokens, d_model=128)
    C_scene = [S, G, M_1..M_256]             (258 tokens, cross-attn memory)
    f(P_t, E_t, M, s, g, t) -> (x0_P_hat [B,H,2], x0_E_hat [B,H,6])

Encodings (all additive):
    T_k = MLP_P(p_k^t) + psi(k) + e_traj + w_t * phi(p_k^t)
    E_k = MLP_E(e_k^t)  + psi(k) + e_ell  + w_t * phi(p_k^t)
    S   = phi(s) + e_start ,   G = phi(g) + e_goal
    M_j = f_j^map + phi(q_j)
    w_t = sqrt(alpha_bar_t)   (noise-aware spatial gate)
"""
import torch
import torch.nn as nn

from ..position_encoding import (Sinusoidal2DPositionEmbedding,
                                 Sinusoidal1DPositionEmbedding,
                                 SinusoidalTimestepEmbedding)
from .scene_cnn import SceneCNN
from .joint_blocks import JointBlock


def scene_grid_centres(res, device):
    """Cell centres of a res x res grid over scene [-1,1]^2 -> [res*res, 2]."""
    xs = torch.linspace(-1.0, 1.0, res + 1, device=device)[:-1] + 1.0 / res
    ys = torch.linspace(-1.0, 1.0, res + 1, device=device)[:-1] + 1.0 / res
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


class JointPlanner(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.horizon = int(model_cfg["horizon"])          # H = 128 waypoints
        self.d_model = int(model_cfg.get("d_model", 128))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.joint_blocks = int(model_cfg.get("joint_blocks", 8))
        self.ffn_dim = int(model_cfg.get("ffn_dim", 512))
        self.map_res = int(model_cfg.get("map_res", 256))
        # global memory res: 'global_mem_res' preferred, legacy 'mem_res' accepted
        self.global_res = int(model_cfg.get("global_mem_res",
                                            model_cfg.get("mem_res", 16)))
        self.geo_mem_res = int(model_cfg.get("geo_mem_res", 32))
        self.geo_decode_res = model_cfg.get("geo_decode_res")
        self.geo_sigma = float(model_cfg.get("geo_sigma", 0.25))
        self.geo_bias_clip = float(model_cfg.get("geo_bias_clip", 8.0))
        self.geo_attn_every = int(model_cfg.get("geo_attn_every", 2))
        # V1.1 fine-geometry memory: enabled only when a decoder res is given
        self._geo_enabled = (self.geo_decode_res is not None
                             and self.geo_attn_every > 0)
        # When True the ellipse token localises/attends with its OWN centre
        # (e_t[..., :2], absolute-centre representation) instead of the
        # trajectory waypoint, reducing the ellipse <-> trajectory coupling.
        self.ellipse_center_anchor = bool(model_cfg.get("ellipse_center_anchor", False))
        dropout = float(model_cfg.get("dropout", 0.0))
        H = self.horizon

        self.scene_cnn = SceneCNN(d_model=self.d_model, res=self.map_res,
                                  global_res=self.global_res,
                                  geo_decode_res=(int(self.geo_decode_res)
                                                  if self.geo_decode_res else None),
                                  geo_mem_res=self.geo_mem_res)

        # ---- shared spatial / planning / time embeddings ----
        self.spatial_pe = Sinusoidal2DPositionEmbedding(self.d_model)   # phi
        self.plan_pe = Sinusoidal1DPositionEmbedding(self.d_model)      # psi
        self.time_pe = SinusoidalTimestepEmbedding(self.d_model)        # t -> h_t
        self.plan_idx = torch.arange(H)

        # ---- token types / roles ----
        self.traj_type = nn.Embedding(2, self.d_model)     # 0=trajectory,1=ellipse
        self.role_type = nn.Embedding(2, self.d_model)     # 0=start, 1=goal

        # ---- content MLPs ----
        self.mlp_p = nn.Linear(2, self.d_model)            # MLP_P: 2 -> 128
        self.mlp_e = nn.Linear(6, self.d_model)            # MLP_E: 6 -> 128

        # ---- joint transformer ----
        # GeometryCrossAttention is only added at blocks 2/4/6/8 (docs #18)
        self.blocks = nn.ModuleList([
            JointBlock(self.d_model, self.num_heads, self.ffn_dim, H, dropout,
                       use_geo=(self._geo_enabled
                                and (i + 1) % self.geo_attn_every == 0))
            for i in range(self.joint_blocks)
        ])

        # ---- output heads (x0 prediction; docs/联合扩散.md #25 allows switching
        # from eps-prediction to x0-prediction without changing the architecture;
        # x0 heads keep the T=16 cosine sampler numerically stable) ----
        self.head_p = nn.Linear(self.d_model, 2)           # [B,H,2]  x0 of P
        self.head_e = nn.Linear(self.d_model, 6)           # [B,H,6]  x0 of E

    # ------------------------------------------------------------------
    def _scene_tokens(self, occ):
        """occ -> (global_mem, geo_mem or None).

        global tokens: C_G = [S?G added later] ... here returns plain map tokens
        [B,256,d]; geo tokens C_E [B,1024,d] (fine geometry, no S/G).
        """
        enc = self.scene_cnn(occ)
        gf = enc["global"]                                 # [B,d,G,G]
        B = gf.shape[0]
        gf = gf.flatten(2).transpose(1, 2)                 # [B,G*G,d]
        grid_g = scene_grid_centres(self.global_res, gf.device)
        global_map = gf + self.spatial_pe(grid_g)[None]    # [B,256,d]

        geo_mem = None
        if enc["geometry"] is not None:
            zf = enc["geometry"].flatten(2).transpose(1, 2)   # [B,1024,d]
            grid_z = scene_grid_centres(self.geo_mem_res, zf.device)
            geo_mem = zf + self.spatial_pe(grid_z)[None]      # C_E (no S/G)
        return global_map, geo_mem

    def _geometry_bias(self, p_t, ab, dev):
        """Noise-aware spatial bias [B,1,H,G] (docs #11-#14):
        B = -alpha_bar_t * ||p_k^t - q_j||^2 / (2 sigma^2), clamped."""
        grid = scene_grid_centres(self.geo_mem_res, dev)   # [G,2]
        dist2 = ((p_t[:, :, None, :] - grid[None, None, :, :]) ** 2).sum(dim=-1)
        strength = (ab ** 2).to(p_t.dtype)[:, None, None]  # alpha_bar_t
        bias = -strength * dist2 / (2.0 * self.geo_sigma ** 2)
        return bias.clamp(-self.geo_bias_clip, 0.0)[:, None, :, :]   # [B,1,H,G]

    def forward(self, p_t, e_t, occ, cond, t, ab):
        """p_t [B,H,2] noisy waypoints; e_t [B,H,6] noisy ellipse repr;
        occ [B,1,256,256]; cond [B,2,2] scene (start, goal); t [B] long timestep;
        ab [B] = sqrt(alpha_bar_t) (w_t gate)."""
        B, H, _ = p_t.shape
        dev = p_t.device
        start, goal = cond[:, 0], cond[:, 1]               # [B,2]

        # ---- scene memories ----
        global_map, geo_mem = self._scene_tokens(occ)      # [B,256,d], [B,1024,d]|None
        # role embeddings are per-sample [B,d]; add to the 2D PE then expand
        # to token slots so h_s/h_g stay [B,1,d] (no [B,B,d] broadcast leak)
        start_role = self.role_type(torch.zeros(B, device=dev, dtype=torch.long))
        goal_role = self.role_type(torch.ones(B, device=dev, dtype=torch.long))
        h_s = (self.spatial_pe(start) + start_role)[:, None, :]   # [B,1,d]
        h_g = (self.spatial_pe(goal) + goal_role)[:, None, :]     # [B,1,d]
        global_mem = torch.cat([h_s, h_g, global_map], dim=1)     # C_G [B,258,d]

        # ---- planning + time embeddings ----
        psi = self.plan_pe(self.plan_idx.to(dev))          # [H,d]
        psi = psi[None, :, :]                              # broadcast over B
        h_t = self.time_pe(t)                              # [B,d]

        # noise-aware spatial gate w_t * phi(p_k^t)
        w = ab[:, None, None].to(p_t.dtype)
        p_pe = self.spatial_pe(p_t) * w                    # [B,H,d]
        # Ellipse token spatial anchor: by default the trajectory waypoint (keeps
        # the original P-anchored behaviour).  With ellipse_center_anchor the
        # ellipse token uses its OWN centre (e_t[..., :2]) so it localises and
        # attends by itself, decoupling it from the trajectory point.
        e_pe = (self.spatial_pe(e_t[..., :2]) * w
                if self.ellipse_center_anchor else p_pe)

        # ---- tokens (type embeddings as [1,1,d] so no extra leading dims) ----
        traj_type = self.traj_type.weight[0][None, None, :]   # [1,1,d]
        ell_type = self.traj_type.weight[1][None, None, :]    # [1,1,d]
        T = self.mlp_p(p_t) + psi + traj_type + p_pe          # [B,H,d]
        E = self.mlp_e(e_t) + psi + ell_type + e_pe           # [B,H,d]

        # interleave [T_1,E_1,...,T_H,E_H] -> [B,2H,d]
        z = torch.stack([T, E], dim=2).reshape(B, 2 * H, self.d_model)

        # noise-aware geometry bias (once, docs #23): anchor on the ellipse's
        # own centre when decoupling, otherwise on the trajectory waypoint.
        geo_bias = None
        if self._geo_enabled:
            geo_anchor = (e_t[..., :2] if self.ellipse_center_anchor else p_t)
            geo_bias = self._geometry_bias(geo_anchor, ab, dev)

        for blk in self.blocks:
            z = blk(z, global_mem, h_t, geo_mem, geo_bias)

        # split odd/even
        z = z.view(B, H, 2, self.d_model)                  # [B,H,{T,E},d]
        h_p, h_e = z[:, :, 0], z[:, :, 1]
        return {
            "x0_p": self.head_p(h_p),                      # [B,H,2]
            "x0_e": self.head_e(h_e),                      # [B,H,6]
        }
