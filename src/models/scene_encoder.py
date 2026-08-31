"""Scene Encoder: lightweight 2D CNN that converts the occupancy map into a
spatial feature map."""

import torch.nn as nn


class SceneEncoder(nn.Module):
    def __init__(self, d_model=128, hidden=64, down=2):
        super().__init__()
        self.d_model = d_model
        self.down = down
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=down, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, d_model, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
        )

    def forward(self, map_tensor):
        # map_tensor [B,1,Hm,Wm]
        return self.conv(map_tensor)
