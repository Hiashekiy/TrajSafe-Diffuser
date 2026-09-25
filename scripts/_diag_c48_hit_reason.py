#!/usr/bin/env python
"""_diag_c48_hit_reason.py - WHAT exactly is the collision on the C=48 failures?

The eval only reports ``final_collision`` (a bool).  ``_free_mask`` in
``src/diffusion/sampler.py`` declares a dense curve point a collision when

    bilinear(occupancy, p) > 0.5          -> it sits on (or within half a cell
                                             of) an OCCUPIED cell
    OR  |p_x| > 1  or  |p_y| > 1          -> it left the 256^2 CROP

Those are two very different failures and the eval merges them.  This script
takes the FINAL control polygon from the dashboard backend
(``control_history[-1]``, the exact 48 controls the sampler finished with),
decodes it at the SAME 512 dense parameters ``dense_validation`` uses, and
splits the non-free points by cause.

    python scripts/_diag_c48_hit_reason.py [idx ...]
"""
from __future__ import annotations

import json
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, "D:/ProjectDirectory/Neural-IRISDiffuser")
from src.geometry.bspline import BSplineCodec  # noqa: E402

ROOT = "D:/ProjectDirectory/Neural-IRISDiffuser"
API = "http://127.0.0.1:8765"
DATASET = "160k4p_c48"
MODEL = "K4P_c48_oneshot:best_task"
RES, H = 256, 512
FAILURES = [167, 291, 325, 333, 342, 343, 348, 366, 368]

_CODEC = None


def codec():
    global _CODEC
    if _CODEC is None:
        from src.geometry.bspline import default_knots
        _CODEC = BSplineCodec(degree=3, num_controls=48, curve_points=H,
                              knots=default_knots(48, 3))
    return _CODEC


def generate(idx, steps=16, seed=0, alm=True, model=MODEL, dataset=DATASET):
    s = json.load(urllib.request.urlopen(
        "%s/sample?dataset=%s&split=test&index=%d" % (API, dataset, idx), timeout=600))
    body = dict(sample_key=s["key"], split="test", dataset=dataset, model_id=model,
                seed=seed, condition=s["condition"], obstacles=[],
                alm_enabled=alm, steps=steps)
    req = urllib.request.Request("%s/generate" % API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def bilinear(occ: np.ndarray, p: np.ndarray) -> np.ndarray:
    gx = (p[:, 0] + 1.0) * (RES / 2.0) - 0.5
    gy = (p[:, 1] + 1.0) * (RES / 2.0) - 0.5
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - x0, gy - y0
    cx0, cx1 = np.clip(x0, 0, RES - 1), np.clip(x0 + 1, 0, RES - 1)
    cy0, cy1 = np.clip(y0, 0, RES - 1), np.clip(y0 + 1, 0, RES - 1)
    return (occ[cy0, cx0] * (1 - fx) * (1 - fy) + occ[cy0, cx1] * fx * (1 - fy)
            + occ[cy1, cx0] * (1 - fx) * fy + occ[cy1, cx1] * fx * fy)


def nearest_obstacle_m(occ: np.ndarray, p: np.ndarray) -> np.ndarray:
    wy, wx = np.where(occ > 0.5)
    if len(wx) == 0:
        return np.full(len(p), np.inf)
    cx = (wx + 0.5) * (2.0 / RES) - 1.0
    cy = (wy + 0.5) * (2.0 / RES) - 1.0
    d = np.sqrt((p[:, 0:1] - cx[None, :]) ** 2 + (p[:, 1:2] - cy[None, :]) ** 2)
    return d.min(axis=1) * 80.0


def main(argv):
    occ_all = np.load("%s/data/carla_processed_160k4p_c48/test/occupancy.npy" % ROOT,
                      mmap_mode="r")
    idxs = [int(a) for a in argv[1:]] or FAILURES
    cc = codec()
    params = torch.linspace(0.0, 1.0, H)
    basis = cc.basis_at(params).numpy().astype(np.float64)          # [H,48]

    print("idx  | reported free | reproduced |  on-obstacle  out-of-crop | "
          "worst bilinear | nearest obstacle | curve bbox")
    print("-" * 122)
    for i in idxs:
        out = generate(i)
        q = np.asarray(out["control_history"][-1], dtype=np.float64)     # [48,2]
        p = basis @ q                                                    # [512,2]
        occ = np.asarray(occ_all[i], dtype=np.float64)
        val = bilinear(occ, p)
        oob = (np.abs(p) > 1.0).any(axis=1)
        on_obs = (val > 0.5) & ~oob
        free = (~oob) & (val <= 0.5)
        fv = out["final_validation"]
        d = nearest_obstacle_m(occ, p[on_obs]) if on_obs.any() else np.array([])
        print("%-4d | %10.6f | %10.6f | %11d %12d | %14.3f | %15s | x[%.2f,%.2f] y[%.2f,%.2f]"
              % (i, fv["final_free_rate"], free.mean(), int(on_obs.sum()), int(oob.sum()),
                 float(val[on_obs].max()) if on_obs.any() else float("nan"),
                 ("%.2f m" % d.min()) if d.size else "-",
                 p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()), flush=True)
        if on_obs.any():
            j = np.where(on_obs)[0]
            print("       on-obstacle params: %s ... (%d pts, param range %.3f-%.3f)"
                  % (j[:8].tolist(), len(j), j.min() / (H - 1), j.max() / (H - 1)))
        if oob.any():
            j = np.where(oob)[0]
            print("       out-of-crop params: %s ... (%d pts, param range %.3f-%.3f)"
                  % (j[:8].tolist(), len(j), j.min() / (H - 1), j.max() / (H - 1)))


if __name__ == "__main__":
    main(sys.argv)
