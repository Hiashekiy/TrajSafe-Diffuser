"""Scene CNN: occupancy map [B,1,256,256] -> memory features [B,d,16,16].

Simple stride-2 downsampling stack, no decoder / local branch (V1 only needs the
16x16 global features; docs/联合扩散.md #5).
"""
import torch.nn as nn


def _conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(1, cout),
        nn.SiLU(),
    )


class SceneCNN(nn.Module):
    def __init__(self, d_model=128, res=256, mem_res=16, channels=(32, 64, 96, 128, 128)):
        super().__init__()
        self.res = int(res)
        self.mem_res = int(mem_res)
        c = list(channels)
        self.stem = _conv_block(1, c[0])
        downs = []
        cin = c[0]
        for cout in c[1:]:
            downs.append(_conv_block(cin, cout))
            downs.append(nn.Conv2d(cout, cout, kernel_size=3, stride=2, padding=1))
            downs.append(nn.GroupNorm(1, cout))
            downs.append(nn.SiLU())
            cin = cout
        self.downs = nn.Sequential(*downs)
        self.bottleneck = _conv_block(c[-1], c[-1])
        self.out = nn.Conv2d(c[-1], d_model, kernel_size=1)
        # resolution check: 256 -> 128 -> 64 -> 32 -> 16
        n_down = len(c) - 1
        assert res // (2 ** n_down) == mem_res, \
            f"res {res} with {n_down} downs reaches {res // (2 ** n_down)}, want {mem_res}"

    def forward(self, occ):
        """occ [B,1,H,W] -> [B,d,mem_res,mem_res]."""
        x = self.stem(occ)
        x = self.downs(x)
        x = self.bottleneck(x)
        return self.out(x)
