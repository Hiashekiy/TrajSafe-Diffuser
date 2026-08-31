"""Planner: end-to-end model combining scene encoder, trajectory encoder,
point-scene attention, safety fusion, ellipse head, and trajectory decoder."""

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


class Planner(nn.Module):
    def __init__(self, model_cfg, geom_cfg, state_norm):
        super().__init__()
        self.horizon = int(model_cfg["horizon"])
        self.state_dim = int(model_cfg.get("state_dim", 6))
        self.d_model = int(model_cfg["d_model"])
        self.extent = tuple(geom_cfg["extent"])
        self.state_norm = state_norm

        self.scene_encoder = SceneEncoder(d_model=self.d_model)
        self.map_pos_embed = Sinusoidal2DPositionEmbedding(self.d_model)
        self.trajectory_encoder = TrajectoryEncoder(
            horizon=self.horizon, d_model=self.d_model,
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg.get("trajectory_encoder_layers", 2)),
            ffn_dim=int(model_cfg.get("ffn_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)))
        self.local_sampler = LocalSceneSampler(local_window=int(model_cfg.get("local_window", 5)))
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
        self.trajectory_head = TrajectoryHead(d_model=self.d_model)
        self._scene_cache = {}

    def _scene_grid(self, Hs, Ws, device, extent=None):
        x0, x1, y0, y1 = extent if extent is not None else self.extent
        # Map tensors are packed as [B,1,H,W] with H = y / rows, W = x / cols.
        # The position grid must therefore vary x along the width axis (W) and
        # y along the height axis (H), matching grid_sample / local_sampler.
        xs = torch.linspace(x0, x1, Ws + 1, device=device)[:-1] + (x1 - x0) / (2 * Ws)
        ys = torch.linspace(y0, y1, Hs + 1, device=device)[:-1] + (y1 - y0) / (2 * Hs)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    def _unnormalize_pos(self, p_norm, state_norm=None):
        sn = state_norm if state_norm is not None else self.state_norm
        dtype = p_norm.dtype
        mins = torch.as_tensor(sn.mins[2:4], device=p_norm.device, dtype=dtype)
        maxs = torch.as_tensor(sn.maxs[2:4], device=p_norm.device, dtype=dtype)
        eps = sn.eps
        p = (p_norm + 1.0) / 2.0
        return p * (maxs - mins + eps) + mins

    def forward(self, x_t, t, map_tensor, cond=None, extent=None, state_norm=None):
        B, H, _ = x_t.shape
        extent = extent if extent is not None else self.extent
        state_norm = state_norm if state_norm is not None else self.state_norm
        # Scene encoding may be cached during inference (no grad) but must be
        # recomputed per call when gradients are enabled (training), otherwise
        # the cached tensor carries a stale autograd graph that triggers
        # "backward through the graph a second time".
        if not torch.is_grad_enabled():
            key = (id(map_tensor), tuple(map_tensor.shape))
            if key not in self._scene_cache:
                raw = self.scene_encoder(map_tensor).detach()
                if raw.shape[0] != B:
                    raw = raw.expand(B, -1, -1, -1).contiguous()
                Hs, Ws = raw.shape[-2], raw.shape[-1]
                scene_xy = self._scene_grid(Hs, Ws, x_t.device, extent=extent)
                map_pos = self.map_pos_embed(scene_xy)
                self._scene_cache[key] = (raw, scene_xy, map_pos, Hs, Ws)
            scene_map, scene_xy, map_pos, Hs, Ws = self._scene_cache[key]
        else:
            scene_map = self.scene_encoder(map_tensor)
            if scene_map.shape[0] != B:
                scene_map = scene_map.expand(B, -1, -1, -1).contiguous()
            Hs, Ws = scene_map.shape[-2], scene_map.shape[-1]
            scene_xy = self._scene_grid(Hs, Ws, x_t.device, extent=extent)
            map_pos = self.map_pos_embed(scene_xy)
        if scene_map.shape[0] != B:
            scene_map = scene_map.expand(B, -1, -1, -1).contiguous()
        map_pos_grid = map_pos.view(scene_map.shape[1], Hs, Ws)
        scene_map_pe = scene_map + map_pos_grid
        scene_memory = scene_map_pe.flatten(2).transpose(1, 2)   # [B,Ns,C]

        F_traj = self.trajectory_encoder(x_t, t, cond=cond)   # [B,H,C]
        p_world = self._unnormalize_pos(x_t[..., 2:4], state_norm=state_norm)     # [B,H,2]
        local_scene = self.local_sampler(scene_map_pe, p_world, extent)
        A_t = self.point_scene_attention(F_traj, local_scene)
        S_t = self.safety_fusion(F_traj, A_t)

        ellipse_out = self.ellipse_head(S_t)
        T_t = self.trajectory_decoder(S_t, scene_memory)
        x0_pred = self.trajectory_head(T_t)

        return {
            "x0_pred": x0_pred,
            "ellipse_center": ellipse_out["center"],
            "ellipse_radii": torch.stack([ellipse_out["r1"], ellipse_out["r2"]], dim=-1),
            "ellipse_dir": ellipse_out["dir"],
            "ellipse_theta": ellipse_out["theta"],
            "shared_feature": S_t,
        }
