"""Scene CNN (V2): ONE 16x16 map memory.

Encoder:  256 -> ... -> global_res (16x16), i.e. the only map feature the
network builds::

    occ [B,1,R,R] -> C_G [B,d,16,16]

There is no second, finer map memory any more.  The earlier V1.1 variant ran a
fine-geometry decoder (``F16 -(up+fuse F32)-> D32 -(up+fuse F64)-> D64
-(stride-2 conv)-> G32``) and gave the ellipse branch a separate ``C_E`` at
32x32, while the trajectory backbone used ``C_G`` at 16x16.  Both consumers -
the control backbone and the ellipse-geometry cross attention - now read the
SAME 16x16 feature map, so there is exactly one map representation in the whole
model and no ``geo_mem_res`` / ``geo_decode_res`` knob to keep consistent.
"""
import math

import torch
import torch.nn as nn


def _conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(1, cout),
        nn.SiLU(),
    )


def _default_channels(n_down):
    """channels sized n_down+1, matching the historical 16x16 stack when n_down=4."""
    base = [32, 64, 96, 128]
    ch = list(base)
    while len(ch) - 1 < n_down:
        ch.append(128)
    return ch[:n_down + 1]


class SceneCNN(nn.Module):
    def __init__(self, d_model=128, res=256, global_res=16, channels=None):
        super().__init__()
        self.res = int(res)
        self.global_res = int(global_res)
        self.d_model = d_model
        n_down = round(math.log2(res / global_res))
        assert res == global_res * (2 ** n_down), \
            f"res {res} not a power-of-two multiple of global_res {global_res}"
        if channels is None:
            channels = _default_channels(n_down)
        assert len(channels) - 1 == n_down
        c = list(channels)

        self.stem = _conv_block(1, c[0])
        stages = []
        cin = c[0]
        for cout in c[1:]:
            stage = nn.Sequential(
                _conv_block(cin, cout),
                nn.Conv2d(cout, cout, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, cout),
                nn.SiLU(),
            )
            stages.append(stage)
            cin = cout
        self.downs = nn.ModuleList(stages)
        self.bottleneck = _conv_block(c[-1], c[-1])
        self.out = nn.Conv2d(c[-1], d_model, kernel_size=1)

    def forward(self, occ):
        """occ [B,1,R,R] -> C_G [B,d,global_res,global_res]."""
        x = self.stem(occ)
        for i, stage in enumerate(self.downs):
            x = stage(x)
        x = self.bottleneck(x)
        return self.out(x)
