"""Scene Encoder: a lightweight U-Net that converts an occupancy map into
(1) a low-resolution global scene feature (bottleneck 16x16) and
(2) a reduced-resolution local scene feature (local_res x local_res, default 64).

Input:   [B,1,256,256] occupancy over the scene frame [-1,1]^2.
Output:  dict with
         "memory": [B,C,16,16]        -- bottleneck, used as the global Scene Memory
         "local": [B,C,local_res,local_res] -- reduced-res, used for local window sampling
"""

import math

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
    """U-Net encoder with bottleneck 16x16 and a reduced-res local output.

    The decoder only up-samples from the res/16 bottleneck to ``local_res``
    (16 -> 32 -> 64 for local_res=64), rather than restoring full 256.
    """

    def __init__(self, d_model=128, enc=(32, 64, 96, 128, 128), res=256, local_res=64):
        super().__init__()
        self.res = res
        self.local_res = int(local_res)
        self.d_model = d_model
        c = list(enc)                     # encoder channels
        self.stem = _conv_block(1, c[0])
        self.downs = nn.ModuleList([_down(c[i], c[i + 1]) for i in range(len(c) - 1)])
        self.bottleneck = _conv_block(c[-1], c[-1])

        # number of decoder up-stages: res/16 -> local_res
        bottleneck_res = res // 16
        n_up = int(round(math.log2(self.local_res / bottleneck_res)))
        n_up = max(1, n_up)
        self.n_up = n_up
        # skip channels for each up-stage (encoder outputs at the matched scale)
        # encoder skips: [stem@res, d1@res/2, d2@res/4, d3@res/8, d4@res/16]
        # for up-stage i target res, skip = skips[-(i+2)] = c[-2-i]
        self.skip_ch = [c[-2 - i] for i in range(n_up)]
        up_out_all = [128, 96, 64]
        self.up_out = up_out_all[:n_up]
        prev_in = c[-1]
        ups = []
        for i in range(n_up):
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
        # decoder (stop at local_res)
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
        local = self.out_conv(u)               # [B,d_model,local_res,local_res]
        memory = self.memory_conv(mem)         # [B,d_model,res/16,res/16]
        return {"memory": memory, "local": local}