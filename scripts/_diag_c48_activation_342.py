#!/usr/bin/env python
"""_diag_c48_activation_342.py - WHY sample 342 never got a corridor.

The eval only stores ``alm_status = activation_failed``.  There are three very
different ways to end up there:

  1. the convex regions themselves are rejected -> ``failure_reason`` is
     ``invalid_base_region:k`` (region k is unbounded / its own centre is outside
     it / its polygon has ~zero area);
  2. the regions are fine but the corridor does not close -> overlap below
     ``corridor.min_overlap_ratio`` / no bridge -> ``corridor_invalid``;
  3. every topology candidate was tried -> ``no_more_candidates``.

This replays the OFFICIAL test protocol (test split, chunk 32, 16 steps, seed 0,
``K4P_c48_oneshot:best_task``) so the states are the eval's own, then, at EVERY
reverse step where activation was attempted, rebuilds the 128 convex regions with
the same ``EllipseRegionBuilder`` + config and reports the per-cell diagnostics
(``valid`` / ``bounded`` / ``center_inside`` / face counts).

    python scripts/_diag_c48_activation_342.py --index 342
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate  # noqa: E402
from src.diffusion.sampler import sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.convex_region import EllipseRegionBuilder  # noqa: E402
from src.geometry.safety_corridor import build_safety_corridor  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config, num_controls, num_safety_queries  # noqa: E402

CFG = "configs/config_160k4p_c48_oneshot.yaml"
CKPT = "outputs/oneshot_k4p_c48/ckpt/best_task.pt"
CHUNK, STEPS, SEED = 32, 16, 0


def region_report(builder, center, shape4):
    A, b, mask, valid, diag = builder.build_from_ellipse(
        center, shape4, return_diagnostics=True)
    n = int(valid.shape[1])
    faces = mask.sum(dim=-1)[0]
    return dict(
        n_cells=n,
        n_valid=int(valid[0].sum()),
        n_bounded=int(diag["bounded"][0].sum()),
        n_center_inside=int(diag["center_inside"][0].sum()),
        n_finite=int((diag["center_finite"] & diag["quadratic_finite"])[0].sum()),
        face_min=int(faces.min()), face_med=int(faces.median()),
        face_max=int(faces.max()),
        bad_cells=[int(i) for i in torch.where(~valid[0])[0][:8]],
        bad_center_violation=[round(float(v), 5) for v
                              in diag["max_center_violation"][0][~valid[0]][:8]],
        n_center_violating_margin=int(
            (diag["max_center_violation"][0] > 0).sum()),
        margin=float(builder.margin),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=342)
    args = ap.parse_args()

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
    lo = (args.index // CHUNK) * CHUNK
    hi = min(lo + CHUNK, len(ds))
    sub = collate([ds[i] for i in range(lo, hi)])
    with torch.no_grad():
        out = sample(model, schedule, sub["cond"], sub["occupancy"],
                     sub["candidate_xy"], sub["candidate_mask"],
                     sub["candidate_geometry"], sub["candidate_geometry_lengths"],
                     device=device, steps=STEPS, seed=SEED, return_trace=True,
                     alm_config=cfg.get("alm"), corridor_config=cfg.get("corridor"))
    j = args.index - lo
    info = out["activation_info"]
    print("=== index %d (%s) ===" % (args.index, "test_%04d" % args.index))
    print("  alm_status        : %s" % out["alm_status"][j])
    print("  guided            : %s" % bool(out["guided"][j]))
    print("  activation_step   : %s" % info.get("activation_step", ["n/a"])[j]
          if isinstance(info.get("activation_step"), list) else "")
    print("  attempts          : %d" % info["attempts"][j])
    print("  failure_reason    : %r" % info["failure_reason"][j])
    print("  topology_fallback : %s" % info["topology_fallback"][j])
    print("  candidate_trials  : %s" % info["candidate_trials"])
    print("  region_face_counts: %s" % info["region_face_counts"][:12])
    print("  overlap min/mean  : %s / %s" % (info["overlap_min"], info["overlap_mean"]))
    print("  selected_idx      : %s" % out["selected_idx"][j].item())

    # ---- rebuild the 128 regions at every step that was NOT yet guided -----
    occ_row = sub["occupancy"][j:j + 1]
    builder_cfg = dict(cfg.get("corridor"))
    anchors = torch.linspace(0.0, 1.0, model.num_safety_queries, device=device)
    print()
    print("  step | t    | 128 凸区域: valid/bounded/center_inside | faces min/med/max | bad cells")
    print("  " + "-" * 100)
    for k, tr in enumerate(out["trace"]):
        if bool(tr["guided"][j]):
            continue
        builder = EllipseRegionBuilder(occ_row, builder_cfg)
        rep = region_report(builder,
                            tr["ellipse_center"][j][None],
                            tr["ellipse_shape4"][j][None])
        print("  %4d | %-4s | %3d / %3d / %3d %s | %3d / %3d / %3d | %s"
              % (k, int(tr["t"]), rep["n_valid"], rep["n_bounded"],
                 rep["n_center_inside"], rep["n_finite"], rep["face_min"],
                 rep["face_med"], rep["face_max"], rep["bad_cells"]))
        if rep["bad_cells"]:
            print("        这 9 个单元的中心违反自己的面，量 = %s  (safety_margin = %.4f scene = %.2f m)"
                  % (rep["bad_center_violation"], rep["margin"],
                     rep["margin"] * 80.0))
        # and the corridor itself, on the topology candidate that was selected
        try:
            corr = build_safety_corridor(
                builder, tr["ellipse_center"][j], tr["ellipse_shape4"][j],
                anchors.cpu().numpy(),
                gamma=sub["candidate_geometry"][j, 0].cpu().numpy(),
                gamma_lengths=int(sub["candidate_geometry_lengths"][j, 0].item()),
                config=cfg.get("corridor"))
            ov = corr.overlap_ratio
            print("        corridor: valid=%s reason=%r cells=%d overlap_min=%s"
                  % (corr.valid, corr.failure_reason, corr.num_cells,
                     (round(min(ov), 4) if ov else None)))
        except Exception as exc:                       # noqa: BLE001
            print("        corridor: raised %s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
