"""Scene Encoder: a lightweight U-Net that converts an occupancy map into
(1) a low-resolution global scene feature (bottleneck 16x16) and
(2) a full-resolution local scene feature (256x256).

Input:   [B,1,256,256] occupancy over the scene frame [-1,1]^2.
Output:  dict with
         "memory": [B,C,16,16]    -- bottleneck, used as the global Scene Memory
         "local": [B,C,256,256]   -- full-res, used for local window sampling
"""

import torch
import torch.nn as nn


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.GroupNorm(1, cout),
        nn.SiLU(),
    )


def _down(cin, cout):
    return nn.Sequential(
        _conv_block(cin, cout),
        nn.Conv2d(cout, cout, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(1, cout),
        nn.SiLU(),
    )


class SceneEncoder(nn.Module):
    """U-Net encoder with bottleneck 16x16 and full-res local output."""

    def __init__(self, d_model=128, enc=(32, 64, 96, 128, 128), res=256):
        super().__init__()
        self.res = res
        self.d_model = d_model
        c = list(enc)                     # encoder channels
        self.stem = _conv_block(1, c[0])
        self.downs = nn.ModuleList([_down(c[i], c[i + 1]) for i in range(len(c) - 1)])
        self.bottleneck = _conv_block(c[-1], c[-1])

        # decoder: res/16 -> res/8 -> res/4 -> res/2 -> res (4 up stages)
        self.skip_ch = [c[3], c[2], c[1], c[0]]   # matched encoder scales
        self.up_out = [128, 96, 64, 32]           # target channels per up stage
        prev_in = c[-1]
        ups = []
        for i in range(4):
            cin = prev_in + self.skip_ch[i]
            ups.append(_conv_block(cin, self.up_out[i]))
            prev_in = self.up_out[i]
        self.ups = nn.ModuleList(ups)
        self.out_conv = nn.Conv2d(self.up_out[-1], d_model, kernel_size=3, padding=1)
        self.memory_conv = nn.Conv2d(c[-1], d_model, kernel_size=1)

    def forward(self, map_tensor):
        # encoder
        x = self.stem(map_tensor)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        mem = self.bottleneck(x)               # [B,c_last,res/16,res/16]
        # decoder
        u = mem
        for i, up in enumerate(self.ups):
            u = torch.nn.functional.interpolate(u, scale_factor=2, mode="bilinear",
                                                align_corners=False)
            skip = skips[-(i + 2)]
            if u.shape[-2:] != skip.shape[-2:]:
                u = torch.nn.functional.interpolate(u, size=skip.shape[-2:],
                                                    mode="bilinear", align_corners=False)
            u = torch.cat([u, skip], dim=1)
            u = up(u)
        local = self.out_conv(u)               # [B,d_model,res,res]
        memory = self.memory_conv(mem)         # [B,d_model,res/16,res/16]
        return {"memory": memory, "local": local}
