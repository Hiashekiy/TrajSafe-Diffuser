"""Extract the intermediate clean states (x0) of the trained diffusion model.

For every test sample we run the FULL reverse schedule (plain DDIM, no corridor,
no ALM -- ``guidance="none"``) and record the network's clean prediction
``x0 = model(...)["control"]`` decoded to the 128-point curve after K reverse
forwards, for K in {1,2,3,4,5} plus the final step.  These curves are the soft
reference the grid search is guided by.

    python scripts/grid_search/extract_diffusion_x0.py \
        --config configs/config_160k4p_c48_oneshot.yaml \
        --ckpt outputs/models/oneshot_k4p_c48/ckpt/best_task.pt \
        --split test --limit 420 --batch 16 \
        --out outputs/grid_search/diffusion_x0/test

Output (single .npz + .json):
    indices   [S]        dataset indices
    cond      [S,2,2]    start / goal in scene coords
    curves    [S,K,128,2] x0 curves for the recorded reverse steps
    steps     [K]        number of executed reverse forwards (K value)
    times     [S,K]      timestep t of that forward
    batch_ms  [S]        amortised wall clock of the whole 16-step reverse pass
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
    __file__))))
sys.path.insert(0, ROOT)

from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate
from src.diffusion.sampler import sample
from src.diffusion.schedule import NoiseSchedule
from src.utils.checkpoint import load_model
from src.utils.config import load_config, num_controls, num_safety_queries

K_STEPS = (1, 2, 3, 4, 5)          # K = number of executed reverse forwards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_160k4p_c48_oneshot.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=420)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--guidance", default="none",
                    help="'none' = plain DDIM (default); 'alm' = paper's corridor")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = CarlaSplineDataset(args.split, cfg["data"]["processed_root"],
                            geometry_points=int((cfg.get("topology") or {}).get(
                                "candidate_geometry_points", 1280)),
                            require_labels=False, num_controls=num_controls(cfg),
                            num_safety_queries=num_safety_queries(cfg))
    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model, ckpt, _ = load_model(cfg, args.ckpt, arch="auto", device=device)
    model.eval()
    print("[x0] %s controls=%d T=%d load ok"
          % (os.path.basename(args.ckpt), model.num_controls,
             schedule.num_timesteps), flush=True)

    idx_all = list(range(args.offset, min(args.offset + args.limit, len(ds))))
    chunks, conds, times, batch_ms = [], [], [], []
    t_start = time.perf_counter()
    for start in range(0, len(idx_all), args.batch):
        idxs = idx_all[start:start + args.batch]
        batch = make_collate(ds)([ds[i] for i in idxs])
        t0 = time.perf_counter()
        out = sample(model, schedule, batch["cond"].to(device),
                     batch["occupancy"].to(device),
                     batch["candidate_xy"].to(device),
                     batch["candidate_mask"].to(device),
                     batch["candidate_geometry"].to(device),
                     batch["candidate_geometry_lengths"].to(device),
                     device=device, steps=None, seed=args.seed,
                     return_trace=True, guidance=args.guidance,
                     alm_config=cfg.get("alm") or {},
                     corridor_config=cfg.get("corridor") or {})
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / len(idxs)
        trace = out["trace"]
        curves = np.stack([trace[k - 1]["final"].cpu().numpy()
                           for k in K_STEPS], axis=1)      # [B,K,128,2]
        chunks.append(curves)
        conds.append(batch["cond"].numpy())
        times.append(np.array([int(trace[k - 1]["t"]) for k in K_STEPS]))
        batch_ms.extend([ms] * len(idxs))
        print("[x0] %d/%d  %.0f ms/sample (16-step reverse pass)"
              % (min(start + args.batch, len(idx_all)), len(idx_all), ms),
              flush=True)

    curves = np.concatenate(chunks, axis=0)
    cond = np.concatenate(conds, axis=0)
    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, "diffusion_x0_%s_%d_%d.npz"
                       % (args.split, args.offset, args.offset + len(idx_all)))
    np.savez_compressed(npz, indices=np.asarray(idx_all), cond=cond,
                        curves=curves.astype(np.float32),
                        steps=np.asarray(K_STEPS),
                        times=np.stack(times)[0],
                        batch_ms=np.asarray(batch_ms, dtype=np.float32),
                        guided=np.zeros(len(idx_all), bool))
    meta = {
        "config": args.config, "ckpt": args.ckpt, "split": args.split,
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "guidance": args.guidance, "num_controls": int(model.num_controls),
        "timesteps": int(schedule.num_timesteps), "seed": args.seed,
        "n_samples": len(idx_all), "k_steps": list(K_STEPS),
        "timestep_of_k": dict(zip(map(str, K_STEPS),
                                  map(int, np.stack(times)[0]))),
        "wall_seconds": time.perf_counter() - t_start,
        "mean_reverse_pass_ms": float(np.mean(batch_ms)),
        "prediction_type": cfg["diffusion"].get("prediction_type", "x0"),
        "note": "x0 = out['control'] (boundary-decoded controls) decoded to the "
                "128-point curve; guidance='none' means noise -> plain DDIM, "
                "no corridor and no ALM, so the reference may cross obstacles.",
    }
    with open(os.path.join(args.out, "diffusion_x0_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[x0] saved", npz, "curves", curves.shape, flush=True)


if __name__ == "__main__":
    main()
