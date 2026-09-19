"""Scene CNN (V1.1): global encoder + fine-geometry decoder.

Encoder (kept from V1):  256 -> ... -> global_res (16x16), retaining the
intermediate maps F64 (96ch @64x64) and F32 (128ch @32x32) for skips.

Decoder (new, docs/联合扩散 V1.1):
    F16 -(up+fuse F32)-> D32 -(up+fuse F64)-> D64 -(stride-2 conv)-> G32
so the 32x32 geometry memory absorbs F16+F32+F64.

Forward returns a dict:
    {"global":   [B,d,global_res,global_res],       # V1 global memory
     "geometry": [B,d,geo_mem_res,geo_mem_res]}     # fine geometry or None
When geo_decode_res is None the decoder is not built (legacy single-memory mode),
and the encoder is identical to the original V1 SceneCNN.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _UpFuse(nn.Module):
    """bilinear x2 up + concat encoder skip + two conv blocks (docs #3/#4)."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        if skip_ch:
            cin = in_ch + skip_ch
        else:
            cin = in_ch
        self.cat_skip = skip_ch > 0
        self.net = nn.Sequential(
            _conv_block(cin, out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.cat_skip:
            x = torch.cat([x, skip], dim=1)
        return self.net(x)


class SceneCNN(nn.Module):
    def __init__(self, d_model=128, res=256, global_res=16,
                 geo_decode_res=None, geo_mem_res=32, channels=None):
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

        # encoder intermediate maps: resolution -> channels
        self.enc_ch = {}
        for i in range(n_down):
            self.enc_ch[res // (2 ** (i + 1))] = c[i + 1]

        # ---- fine-geometry decoder (optional) ----
        self.geo_decode_res = int(geo_decode_res) if geo_decode_res else None
        self.geo_mem_res = int(geo_mem_res)
        self._geo_enabled = self.geo_decode_res is not None
        if self._geo_enabled:
            assert self.geo_decode_res >= global_res * 2
            assert self.geo_decode_res == 2 * self.geo_mem_res, \
                "default fine path expects geo_decode_res = 2 * geo_mem_res"
            cur = global_res
            self.up_fuses = nn.ModuleList()
            while cur < self.geo_decode_res:
                cur *= 2
                skip_ch = self.enc_ch.get(cur, 0)
                self.up_fuses.append(_UpFuse(d_model, skip_ch, d_model))
            self.merge = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, d_model),
                nn.SiLU(),
            )
        else:
            self.up_fuses = nn.ModuleList()
            self.merge = nn.Identity()

    def forward(self, occ):
        """occ [B,1,H,W] -> {"global": [B,d,G,G], "geometry": [B,d,GM,GM] or None}."""
        x = self.stem(occ)
        maps = {}
        for i, stage in enumerate(self.downs):
            x = stage(x)
            maps[self.res // (2 ** (i + 1))] = x
        fb = self.bottleneck(maps[self.global_res])
        global_f = self.out(fb)                                  # [B,d,G,G]

        if not self._geo_enabled:
            return {"global": global_f, "geometry": None}

        u = global_f
        cur = self.global_res
        for fuse in self.up_fuses:
            cur *= 2
            u = fuse(u, maps.get(cur))
        g = self.merge(u)                                        # [B,d,GM,GM]
        return {"global": global_f, "geometry": g}
