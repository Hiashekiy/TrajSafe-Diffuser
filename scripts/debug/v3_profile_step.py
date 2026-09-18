"""Single-step profiler for V3 (NOT training).

Answers one question: how long does ONE V3 forward+backward take?  The first
3-batch smoke test showed batch 1 at ~409 s while batches 2/3 were fast, and that
has not been explained yet; this isolates whether a single step is inherently
slow or whether it was a one-off (concurrent process, CUDA warm-up, ...).

    python scripts/debug/v3_profile_step.py --device cuda --repeat 5

It creates the model, builds synthetic tensors of the real shapes, and times
forward + backward only - no optimizer, no dataset, no epochs.
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.models.skeleton_v3 import SkeletonPlannerV3
from src.losses.v3_losses import (ellipse_area_loss, ellipse_safety_loss,
                                  trajectory_x0_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v3_skeleton.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    topo = cfg.get("topology") or {}
    B, H = args.batches, int(cfg["model"]["horizon"])
    M = int(topo.get("num_candidates", 4))
    L = int(topo.get("candidate_points", 128))
    G = int(topo.get("candidate_geometry_points", 1280))

    model = SkeletonPlannerV3(cfg["model"], cfg.get("ellipse")).to(device)
    model.train()
    p0 = torch.randn(B, H, 2, device=device)
    cond = torch.randn(B, 2, 2, device=device)
    occ = torch.zeros(B, 1, 256, 256, device=device)
    occ[:, :, 0, :] = 1.0
    feats = torch.randn(B, M, L, 5, device=device)
    mask = torch.ones(B, M, dtype=torch.bool, device=device)
    lengths = torch.rand(B, M, device=device) + 0.5
    geom = torch.randn(B, M, G, 2, device=device) * 0.5
    glen = torch.full((B, M), G, dtype=torch.long, device=device)
    t = torch.full((B,), 5, dtype=torch.long, device=device)
    ab = torch.full((B,), 0.5, device=device)

    print("device=%s B=%d H=%d M=%d L=%d G=%d" % (device, B, H, M, L, G),
          flush=True)

    cuda = device.type == "cuda"

    def sync():
        # CUDA is asynchronous: without this every segment timing below is a
        # kernel-LAUNCH time, not a kernel time.
        if cuda:
            torch.cuda.synchronize()

    def timed(call):
        sync()
        t0 = time.time()
        out = call()
        sync()
        return out, time.time() - t0

    # warm-up pass (cudnn autotune, lazy init) - never reported
    warm = model.forward_all(p0, occ, cond, t, ab, feats, mask, lengths,
                             geom, glen, select_index=None)
    warm_loss = trajectory_x0_loss(warm["final"], p0)
    sync()
    warm_loss.backward()
    model.zero_grad(set_to_none=True)
    sync()
    print("warm-up done", flush=True)

    times = []
    for i in range(args.repeat):
        out, t_fwd = timed(lambda: model.forward_all(
            p0, occ, cond, t, ab, feats, mask, lengths, geom, glen,
            select_index=None))
        loss = (trajectory_x0_loss(out["final"], p0)
                + trajectory_x0_loss(out["coarse"], p0))

        def _losses():
            safe, _, _ = ellipse_safety_loss(
                out["ellipse"]["center"], out["ellipse"]["a"],
                out["ellipse"]["b"], out["ellipse"]["theta"], occ)
            return loss + safe + ellipse_area_loss(
                out["ellipse"]["a"], out["ellipse"]["b"],
                model.ellipse.a_max, model.ellipse.b_max)

        total_loss, t_loss = timed(_losses)
        _, t_bwd = timed(lambda: (total_loss.backward(),
                                  model.zero_grad(set_to_none=True)))
        times.append((t_fwd, t_loss, t_bwd))
        print("  pass %d: forward %.3fs  losses %.3fs  backward %.3fs  total %.3fs"
              % (i, t_fwd, t_loss, t_bwd, t_fwd + t_loss + t_bwd), flush=True)
    if len(times) > 1:
        later = times[1:]
        n = len(later)
        print("steady-state forward %.3fs losses %.3fs backward %.3fs total %.3fs"
              % (sum(t[0] for t in later) / n, sum(t[1] for t in later) / n,
                 sum(t[2] for t in later) / n,
                 sum(sum(t) for t in later) / n))
    print("first pass total %.3fs" % sum(times[0]))


if __name__ == "__main__":
    main()
