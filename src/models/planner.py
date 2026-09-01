"""Planner: scene encoder + diffusion-token encoder + point-scene attention +
safety fusion + ellipse head + global decoder with goal/start condition memory.

Forward is driven by the zero-sum residual diffusion state Z_t [B,N,2].
Absolute position is recovered internally by integrating from start using the
hard-conditioned increments  delta_t = g/N + z_t.
"""

import torch
import torch.nn as nn

from .scene_encoder import SceneEncoder
from .trajectory_encoder import TrajectoryEncoder
from .local_scene_sampler import LocalSceneSampler
from .point_scene_attention import PointSceneAttention
from .safety_fusion import SafetyFusion
from .ellipse_head import EllipseHead
from .trajectory_decoder import TrajectoryDecoder
from .trajectory_head import TrajectoryHead
from .position_encoding import Sinusoidal2DPositionEmbedding
from src.diffusion.zerosum import integrate_positions, compute_base


class Planner(nn.Module):
    def __init__(self, model_cfg, geom_cfg, state_norm=None):
        super().__init__()
        self.horizon = int(model_cfg["horizon"])      # H (positions)
        self.d_model = int(model_cfg["d_model"])
        self.num_tokens = self.horizon - 1            # N = H-1 diffusion tokens
        self.state_norm = state_norm                  # unused (positions are scene coords)

        self.scene_encoder = SceneEncoder(d_model=self.d_model, res=256)
        self.map_pos_embed = Sinusoidal2DPositionEmbedding(self.d_model)

        self.trajectory_encoder = TrajectoryEncoder(
            horizon=self.horizon, d_model=self.d_model,
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg.get("trajectory_encoder_layers", 2)),
            ffn_dim=int(model_cfg.get("ffn_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)))

        self.local_sampler = LocalSceneSampler(
            local_window=int(model_cfg.get("local_window", 7)),
            d_model=self.d_model, res=256)

        self.point_scene_attention = PointSceneAttention(
            d_model=self.d_model, num_heads=int(model_cfg["num_heads"]),
            dropout=float(model_cfg.get("dropout", 0.1)))
        self.safety_fusion = SafetyFusion(
            d_model=self.d_model, ffn_dim=int(model_cfg.get("ffn_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)))
        self.ellipse_head = EllipseHead(d_model=self.d_model)
        self.trajectory_decoder = TrajectoryDecoder(
            d_model=self.d_model, num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg.get("trajectory_decoder_layers", 4)),
            ffn_dim=int(model_cfg.get("ffn_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)))
        self.residual_head = TrajectoryHead(d_model=self.d_model)

        # start / goal condition encoders
        self.start_encoder = nn.Sequential(nn.Linear(2, self.d_model), nn.SiLU(),
                                           nn.Linear(self.d_model, self.d_model))
        self.goal_encoder = nn.Sequential(nn.Linear(2, self.d_model), nn.SiLU(),
                                          nn.Linear(self.d_model, self.d_model))
        self.type_embed = nn.Embedding(2, self.d_model)   # 0=start, 1=goal
        self.waypoint_pos_embed = Sinusoidal2DPositionEmbedding(self.d_model)  # for p_k -> ellipse head
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
        loc = sc["local"]         # [B,C,256,256]
        if mem.shape[0] != B:     # sample() passes a single map but B trajectories
            mem = mem.expand(B, -1, -1, -1).contiguous()
            loc = loc.expand(B, -1, -1, -1).contiguous()
        # global memory + map position
        Bm, Cm, Hm, Wm = mem.shape
        mem_grid = self._scene_grid(Hm, Wm, mem.device)
        mem_pos = self.map_pos_embed(mem_grid).view(Cm, Hm, Wm)[None]  # [1,C,H,W]
        mem_pe = mem + mem_pos
        scene_tokens = mem_pe.flatten(2).transpose(1, 2)      # [B,Hm*Wm,C]
        # local feature + map position (full res)
        Bl, Cl2, Hl, Wl = loc.shape
        loc_grid = self._scene_grid(Hl, Wl, loc.device)
        loc_pos = self.map_pos_embed(loc_grid).view(Cl2, Hl, Wl)[None]
        loc_pe = loc + loc_pos

        # ---- trajectory tokens + point-scene ----
        F_traj = self.trajectory_encoder(z_t, t)              # [B,N,C]
        # sample at the N arriving waypoints p_1..p_{H-1} (one per diffusion token)
        local_scene = self.local_sampler(loc_pe, pos_t[:, 1:, :])  # [B,N,w*w,C]
        A_t = self.point_scene_attention(F_traj, local_scene) # [B,N,C]
        S_t = self.safety_fusion(F_traj, A_t)                 # [B,N,C]

        # ---- condition memory C = [h_s; h_g; scene_tokens] ----
        h_s = self.start_encoder(start)[:, None, :] + self.type_embed(torch.tensor(0, device=z_t.device))[None, None, :]
        h_g = self.goal_encoder(g)[:, None, :] + self.type_embed(torch.tensor(1, device=z_t.device))[None, None, :]
        cond_mem = torch.cat([h_s, h_g, scene_tokens], dim=1)  # [B,2+Hm*Wm,C]

        # waypoint position embedding for the ellipse head (per arriving waypoint)
        pos_k = pos_t[:, 1:, :]                            # [B,N,2] scene
        pos_emb_k = self.waypoint_pos_embed(pos_k)         # [B,N,C]
        ellipse_out = self.ellipse_head(S_t, pos_emb_k)
        T_t = self.trajectory_decoder(S_t, cond_mem)
        z0_raw = self.residual_head(T_t)                       # [B,N,2]

        return {
            "z0_pred": z0_raw,
            "ellipse_center": ellipse_out["center"],
            "ellipse_radii": torch.stack([ellipse_out["r1"], ellipse_out["r2"]], dim=-1),
            "ellipse_dir": ellipse_out["dir"],
            "ellipse_theta": ellipse_out["theta"],
            "pos_t": pos_t,
            "shared_feature": S_t,
        }
