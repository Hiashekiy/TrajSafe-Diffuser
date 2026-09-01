"""Planner: scene encoder + diffusion-token encoder + point-scene attention +
safety fusion + trajectory decoder (global trajectory) and, sequentially,
the local ellipse branch.

Forward is driven by the zero-sum residual diffusion state Z_t [B,N,2].
Absolute position is recovered internally by integrating from start using the
hard-conditioned increments  delta_t = g/N + z_t.

Sequential design:
    z_t -> S_t -> z0 -> p_hat -> Local_2 + PE_rel -> EllipseAggregator ->
    EllipseHead -> E_k.
The ellipse branch only sees Local_2 + relative PE; it never consumes
F_traj / S_t / scene_tokens / start / goal / absolute PE.
"""

import torch
import torch.nn as nn

from .scene_encoder import SceneEncoder
from .trajectory_encoder import TrajectoryEncoder
from .local_scene_sampler import LocalSceneSampler
from .point_scene_attention import PointSceneAttention
from .safety_fusion import SafetyFusion
from .ellipse_head import EllipseHead
from .ellipse_aggregator import EllipseAggregator
from .trajectory_decoder import TrajectoryDecoder
from .trajectory_head import TrajectoryHead
from .position_encoding import Sinusoidal2DPositionEmbedding, Sinusoidal2DRelativePositionEmbedding
from src.diffusion.zerosum import integrate_positions, compute_base, zero_sum


class Planner(nn.Module):
    def __init__(self, model_cfg, geom_cfg, state_norm=None):
        super().__init__()
        self.horizon = int(model_cfg["horizon"])      # H (positions)
        self.d_model = int(model_cfg["d_model"])
        self.num_tokens = self.horizon - 1            # N = H-1 diffusion tokens
        self.state_norm = state_norm                  # unused (positions are scene coords)
        self.local_res = int(model_cfg.get("local_res", 64))

        self.scene_encoder = SceneEncoder(d_model=self.d_model, res=256, local_res=self.local_res)
        self.map_pos_embed = Sinusoidal2DPositionEmbedding(self.d_model)   # shared 2D sinusoidal (map + start/goal)

        self.trajectory_encoder = TrajectoryEncoder(
            horizon=self.horizon, d_model=self.d_model,
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg.get("trajectory_encoder_layers", 2)),
            ffn_dim=int(model_cfg.get("ffn_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)))

        self.local_sampler = LocalSceneSampler(
            local_window=int(model_cfg.get("local_window", 9)),
            d_model=self.d_model, res=self.local_res)

        self.point_scene_attention = PointSceneAttention(
            d_model=self.d_model, num_heads=int(model_cfg["num_heads"]),
            dropout=float(model_cfg.get("dropout", 0.1)))
        self.safety_fusion = SafetyFusion(
            d_model=self.d_model, ffn_dim=int(model_cfg.get("ffn_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)))

        self.trajectory_decoder = TrajectoryDecoder(
            d_model=self.d_model, num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg.get("trajectory_decoder_layers", 4)),
            ffn_dim=int(model_cfg.get("ffn_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)))
        self.residual_head = TrajectoryHead(d_model=self.d_model)

        # ---- ellipse branch (sequential, local-only) ----
        self.ellipse_pe_embed = Sinusoidal2DRelativePositionEmbedding(
            self.d_model, scale=float(model_cfg.get("ellipse_relative_pe_scale", 128.0)))
        self.ellipse_aggregator = EllipseAggregator(
            d_model=self.d_model, num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg.get("ellipse_aggregator_layers", 2)),
            ffn_dim=int(model_cfg.get("ffn_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)))
        self.ellipse_head = EllipseHead(d_model=self.d_model)

        self.type_embed = nn.Embedding(2, self.d_model)   # 0=start, 1=goal
        self.ellipse_enabled = True        # Phase1 turns this off to skip the ellipse branch
        self._scene_cache = {}

    def _scene_grid(self, Hs, Ws, device):
        """Scene cell centres [Hs*Ws,2] on the [-1,1]^2 grid (x along W, y along H)."""
        xs = torch.linspace(-1.0, 1.0, Ws + 1, device=device)[:-1] + 1.0 / Ws
        ys = torch.linspace(-1.0, 1.0, Hs + 1, device=device)[:-1] + 1.0 / Hs
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    def forward(self, z_t, t, map_tensor, cond):
        """z_t [B,N,2], t [B], map_tensor [B,1,256,256], cond [B,2,2] (start, goal)."""
        B, N, _ = z_t.shape
        start = cond[:, 0]                       # [B,2] scene
        goal = cond[:, 1]                        # [B,2] scene
        g = goal - start
        base = compute_base(g, N)
        delta_t = base + z_t
        pos_t = integrate_positions(start, delta_t)          # [B,H,2] scene

        # ---- scene encoding ----
        sc = self.scene_encoder(map_tensor)
        mem = sc["memory"]        # [B,C,16,16]
        loc = sc["local"]         # [B,C,local_res,local_res]
        if mem.shape[0] != B:
            mem = mem.expand(B, -1, -1, -1).contiguous()
            loc = loc.expand(B, -1, -1, -1).contiguous()
        # global memory + absolute map position (serves global trajectory planning)
        Bm, Cm, Hm, Wm = mem.shape
        mem_grid = self._scene_grid(Hm, Wm, mem.device)
        mem_pos = self.map_pos_embed(mem_grid).view(Cm, Hm, Wm)[None]
        mem_pe = mem + mem_pos
        scene_tokens = mem_pe.flatten(2).transpose(1, 2)      # [B,Hm*Wm,C]
        # local feature + absolute map position (used ONLY by the trajectory branch sample #1)
        Bl, Cl2, Hl, Wl = loc.shape
        loc_grid = self._scene_grid(Hl, Wl, loc.device)
        loc_pos = self.map_pos_embed(loc_grid).view(Cl2, Hl, Wl)[None]
        loc_pe = loc + loc_pos

        # ---- trajectory tokens + point-scene ----
        F_traj = self.trajectory_encoder(z_t, t)              # [B,N,C]
        local_scene = self.local_sampler(loc_pe, pos_t[:, 1:, :])  # [B,N,w*w,C]
        A_t = self.point_scene_attention(F_traj, local_scene) # [B,N,C]
        S_t = self.safety_fusion(F_traj, A_t)                 # [B,N,C]

        # ---- condition memory C = [h_start; h_goal; scene_tokens] ----
        h_s = self.map_pos_embed(start)[:, None, :] + self.type_embed(torch.tensor(0, device=z_t.device))[None, None, :]
        h_g = self.map_pos_embed(goal)[:, None, :] + self.type_embed(torch.tensor(1, device=z_t.device))[None, None, :]
        cond_mem = torch.cat([h_s, h_g, scene_tokens], dim=1)  # [B,2+Hm*Wm,C]

        # ---- trajectory decoding ----
        T_t = self.trajectory_decoder(S_t, cond_mem)
        z0_raw = self.residual_head(T_t)                       # [B,N,2]
        z0_proj = zero_sum(z0_raw)
        delta_pred = base + z0_proj
        pos_pred = integrate_positions(start, delta_pred)      # [B,H,2]
        p_hat = pos_pred[:, 1:, :]                            # [B,N,2]

        # ---- sequential ellipse branch (Local_2 + PE_rel only) ----
        if self.ellipse_enabled:
            local_2 = self.local_sampler(loc, p_hat)            # [B,N,w*w,C] (no abs PE)
            pe_rel = self.ellipse_pe_embed(self.local_sampler.offs_xy)  # [w*w,C]
            local_2_pe = local_2 + pe_rel[None, None, :, :]     # [B,N,w*w,C]
            e_k = self.ellipse_aggregator(local_2_pe)           # [B,N,C]
            ellipse_out = self.ellipse_head(e_k, p_hat)
        else:
            ellipse_out = {
                "center": None, "r1": None, "r2": None,
                "dir": None, "theta": None, "raw": None,
            }

        return {
            "z0_pred": z0_raw,
            "pos_pred": pos_pred,
            "pos_t": pos_t,
            "shared_feature": S_t,
            "ellipse_center": ellipse_out["center"],
            "ellipse_radii": (None if ellipse_out["r1"] is None else torch.stack([ellipse_out["r1"], ellipse_out["r2"]], dim=-1)),
            "ellipse_dir": ellipse_out["dir"],
            "ellipse_theta": ellipse_out["theta"],
        }