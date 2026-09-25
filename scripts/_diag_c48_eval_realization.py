#!/usr/bin/env python
"""_diag_c48_eval_realization.py - the SAME realization as the official eval.

``evaluate_chunk`` calls ``sample()`` once per chunk of 32, and ``sample()`` does
``torch.manual_seed(seed)`` once, so the noise a sample receives depends on its
POSITION INSIDE ITS CHUNK.  A single-sample dashboard run therefore uses a
different draw -- the free rates differ (e.g. sample 325: 0.832031 in the eval
vs 0.826172 standalone).

This script replays the eval EXACTLY (same dataset order, same chunk boundaries,
same seed, same steps, same ALM config) for the chunks that contain the failing
indexes, and only then splits each sample's non-free dense points into

    ON-OBSTACLE   bilinear(occupancy, p) > 0.5     -> a real obstacle hit
    OUT-OF-CROP   |p_x| > 1 or |p_y| > 1           -> left the 256^2 window

using the project's own ``_dense_decode`` / criterion, and checks its own
free-rate against the value the eval stored in outputs/eval_k4p_c48_own_alm.json.

    python scripts/_diag_c48_eval_realization.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = "D:/ProjectDirectory/Neural-IRISDiffuser"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate  # noqa: E402
from src.diffusion.sampler import _dense_decode, _free_mask, sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402

CFG = "configs/config_160k4p_c48_oneshot.yaml"
CKPT = "outputs/oneshot_k4p_c48/ckpt/best_task.pt"
EVAL_JSON = "outputs/eval_k4p_c48_own_alm.json"
TARGETS = [167, 291, 325, 333, 342, 343, 348, 366, 368]
CHUNK, STEPS, SEED, RES, DENSE = 32, 16, 0, 256, 512


def bilinear(occ: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Value of ``F.grid_sample(occ, p, bilinear, border, align_corners=False)``."""
    gx = (p[:, 0] + 1.0) * (RES / 2.0) - 0.5
    gy = (p[:, 1] + 1.0) * (RES / 2.0) - 0.5
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - x0, gy - y0
    cx0, cx1 = np.clip(x0, 0, RES - 1), np.clip(x0 + 1, 0, RES - 1)
    cy0, cy1 = np.clip(y0, 0, RES - 1), np.clip(y0 + 1, 0, RES - 1)
    return (occ[cy0, cx0] * (1 - fx) * (1 - fy) + occ[cy0, cx1] * fx * (1 - fy)
            + occ[cy1, cx0] * (1 - fx) * fy + occ[cy1, cx1] * fx * fy)


def main():
    cfg = load_config(CFG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CarlaSplineDataset("test", cfg["data"]["processed_root"],
                            geometry_points=1280, num_controls=num_controls(cfg),
                            num_safety_queries=num_safety_queries(cfg))
    collate = make_collate(ds)
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2"),
                             beta_start=cfg["diffusion"].get("beta_start", 1e-4),
                             beta_end=cfg["diffusion"].get("beta_end", 0.02)).to(device)
    model, _, _ = load_model(cfg, CKPT, device=device, verbose=False)
    alm_cfg, corridor_cfg = cfg.get("alm"), cfg.get("corridor")
    codec = model.bspline
    n = len(ds)
    ref = json.load(open(os.path.join(ROOT, EVAL_JSON)))["results"][0]

    chunks = sorted({i // CHUNK for i in TARGETS})
    print("replaying chunks %s (targets %s)" % (chunks, TARGETS))
    print()
    hdr = ("idx  | eval free | replay free | match | non-free | ON-OBSTACLE | "
           "OUT-OF-CROP | worst bilinear | excursion(m)")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    with torch.no_grad():
        for c in chunks:
            lo, hi = c * CHUNK, min((c + 1) * CHUNK, n)
            sub = collate([ds[i] for i in range(lo, hi)])
            out = sample(model, schedule, sub["cond"], sub["occupancy"],
                         sub["candidate_xy"], sub["candidate_mask"],
                         sub["candidate_geometry"],
                         sub["candidate_geometry_lengths"], device=device,
                         steps=STEPS, seed=SEED, return_trace=False,
                         alm_config=alm_cfg, corridor_config=corridor_cfg)
            q = out["control"]
            for j, i in enumerate(range(lo, hi)):
                if i not in TARGETS:
                    continue
                qj = q[j:j + 1]
                p = _dense_decode(codec, qj, DENSE)[0].float().cpu().numpy()
                occ_row = sub["occupancy"][j]                  # [1,R,R]
                free = _free_mask(occ_row.cpu(), torch.as_tensor(p)).numpy()
                occ = occ_row[0].float().cpu().numpy()
                val = bilinear(occ, p)
                oob = (np.abs(p) > 1.0).any(axis=1)
                on_obs = (val > 0.5) & ~oob
                exc = max(0.0, float(np.abs(p).max() - 1.0)) * 80.0
                ev_free = float(ref["per_sample_free_rate"][i])
                mine = float(free.mean())
                rows.append(dict(idx=i, eval_free=ev_free, replay_free=mine,
                                 non_free=int((~free).sum()),
                                 on_obstacle=int(on_obs.sum()),
                                 out_of_crop=int(oob.sum()),
                                 worst_bilinear=(float(val[on_obs].max())
                                                 if on_obs.any() else None),
                                 excursion_m=exc))
                print("%-4d | %9.6f | %11.6f | %-5s | %8d | %11d | %11d | %14s | %8.2f"
                      % (i, ev_free, mine, "YES" if abs(ev_free - mine) < 1e-6 else "NO",
                         int((~free).sum()), int(on_obs.sum()), int(oob.sum()),
                         ("%.3f" % val[on_obs].max()) if on_obs.any() else "-", exc),
                      flush=True)
    print()
    tot_obs = sum(r["on_obstacle"] for r in rows)
    tot_oob = sum(r["out_of_crop"] for r in rows)
    print("TOTAL over the %d failures: %d non-free points  =  %d on-obstacle  +  %d out-of-crop"
          % (len(rows), sum(r["non_free"] for r in rows), tot_obs, tot_oob))
    print("samples with >=1 on-obstacle point: %s"
          % [r["idx"] for r in rows if r["on_obstacle"]])
    print("samples with >=1 out-of-crop point: %s"
          % [r["idx"] for r in rows if r["out_of_crop"]])
    with open(os.path.join(ROOT, "outputs/diag_c48_eval_realization.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print("written outputs/diag_c48_eval_realization.json")


if __name__ == "__main__":
    main()
