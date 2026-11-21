"""Map and control state conditional diffusion, with V5 endpoint modulation."""

import torch
from torch import nn

from ..geometry.bspline import BSplineCodec
from ..geometry.ellipse_raster import scene_grid_centres
from .common.blocks import AdaLN
from .position_encoding import (Sinusoidal1DPositionEmbedding,
                                Sinusoidal2DPositionEmbedding,
                                SinusoidalTimestepEmbedding)
from .trajsafe.blocks import TrajBlock, TrajSelfAttention
from .trajsafe.boundary import BoundaryDecoder, modulate_heading
from .trajsafe.encoders import TrajectoryEncoder


class MapStateDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = config['model']
        width = model['width']
        heads = model['heads']
        self.num_controls = model['num_controls']
        self.bspline = BSplineCodec(degree=3, num_controls=self.num_controls,
                                    curve_points=128)
        self.boundary = BoundaryDecoder(profile=model['boundary_profile'])
        self.heading_min_length = model['heading_min_length']
        channels = [1, 16, 32, 64, width]
        layers = []
        for cin, cout in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                           nn.GroupNorm(4, cout), nn.SiLU()])
        self.map_encoder = nn.Sequential(*layers)
        self.register_buffer('map_centres', scene_grid_centres(16, 'cpu'), persistent=False)
        self.spatial = Sinusoidal2DPositionEmbedding(width)
        self.state_encoder = TrajectoryEncoder(width, self.spatial,
                                               Sinusoidal1DPositionEmbedding(width),
                                               self.num_controls, hidden=256)
        self.time = SinusoidalTimestepEmbedding(width)
        self.state_norm = AdaLN(width, width)
        self.state_attention = TrajSelfAttention(width, heads, self.num_controls)
        self.denoiser = nn.ModuleList([
            TrajBlock(width, heads, model['ffn_dim'], self.num_controls)
            for _ in range(model['blocks'])
        ])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))

    def encode_map(self, occupancy):
        features = self.map_encoder(occupancy).flatten(2).transpose(1, 2)
        return features + self.spatial(self.map_centres)[None]

    def denoise(self, noisy, map_tokens, endpoints, t, heading, heading_valid):
        state = modulate_heading(self.boundary, noisy, endpoints, heading,
                                 heading_valid, min_length=self.heading_min_length)
        tokens = self.state_encoder(state)
        time = self.time(t)
        tokens = tokens + self.state_attention(self.state_norm(tokens, time))
        for block in self.denoiser:
            tokens = block(tokens, map_tokens, time)
        clean = self.boundary(self.head(tokens).float(), endpoints)
        return modulate_heading(self.boundary, clean, endpoints, heading,
                                heading_valid, min_length=self.heading_min_length)

    def forward(self, noisy, occupancy, endpoints, t, heading, heading_valid):
        return self.denoise(noisy, self.encode_map(occupancy), endpoints, t,
                            heading, heading_valid)
