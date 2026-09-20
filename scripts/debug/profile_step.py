"""Single-step profiler for the report-faithful TrajSafe-Diffuser.

    python scripts/debug/profile_step.py --device cuda --repeat 5

Creates the model and synthetic tensors of the real shapes and times
forward + losses + backward only (no optimizer, no dataset, no epochs).
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config, num_controls, num_safety_queries
from src.models.trajsafe import TrajSafePlanner
from src.losses.losses import (boundary_control_loss, control_smoothness_loss,
                               control_x0_loss, ellipse_safety_loss,
                               ellipse_shape_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    topo = cfg.get("topology") or {}
    B = args.batches
    C = num_controls(cfg)                      # control tokens (trajectory)
    Q = num_safety_queries(cfg)                # safety / ellipse queries
    H = int(cfg.get("bspline", {}).get("curve_points",
                                       cfg["model"]["horizon"]))
    M = int(topo.get("num_candidates", 4))
    L = int(topo.get("candidate_points", 128))
    G = int(topo.get("candidate_geometry_points", 1280))

    model = TrajSafePlanner(cfg["model"], cfg.get("ellipse_label"),
                            cfg.get("bspline")).to(device)
    model.train()
    p0 = torch.randn(B, C, 2, device=device)          # control polygon Q_t
    cond = torch.randn(B, 2, 2, device=device)
    occ = torch.zeros(B, 1, 256, 256, device=device)
    occ[:, :, 0, :] = 1.0
    cand_xy = torch.randn(B, M, L, 2, device=device)
    mask = torch.ones(B, M, dtype=torch.bool, device=device)
    geom = torch.randn(B, M, G, 2, device=device) * 0.5
    glen = torch.full((B, M), G, dtype=torch.long, device=device)
    shape_gt = torch.zeros(B, Q, 4, device=device)
    shape_gt[..., 0] = torch.log(torch.tensor(0.1))
    shape_gt[..., 1] = torch.log(torch.tensor(0.05))
    shape_gt[..., 2] = 1.0
    valid = torch.ones(B, Q, dtype=torch.bool, device=device)
    has_cand = mask.any(dim=-1)
    t = torch.full((B,), 5, dtype=torch.long, device=device)
    ab = torch.full((B,), 0.5, device=device)

    print("device=%s B=%d C=%d Q=%d curve=%d M=%d L=%d G=%d"
          % (device, B, C, Q, H, M, L, G),
          flush=True)
    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def timed(call):
        sync()
        t0 = time.time()
        out = call()
        sync()
        return out, time.time() - t0

    warm = model.forward_all(p0, occ, cond, t, ab, cand_xy, mask, geom, glen,
                             select_index=None)
    warm_loss = control_x0_loss(warm["q_raw_final"], p0)
    sync()
    warm_loss.backward()
    model.zero_grad(set_to_none=True)
    sync()
    print("warm-up done", flush=True)

    times = []
    for i in range(args.repeat):
        out, t_fwd = timed(lambda: model.forward_all(
            p0, occ, cond, t, ab, cand_xy, mask, geom, glen,
            select_index=None))
        ell = out["ellipse"]

        ws, wg = model.boundary_decoder.weights(C, device=device)

        def _losses():
            loss = (control_x0_loss(out["q_raw_final"], p0)
                    + control_x0_loss(out["q_coarse_raw"], p0)
                    + control_smoothness_loss(out["q_raw_final"], p0)
                    + boundary_control_loss(out["q_raw_final"], p0, cond, ws, wg))
            loss = loss + ellipse_shape_loss(ell["shape4"], shape_gt, valid,
                                             has_cand)
            safe, _, _ = ellipse_safety_loss(
                ell["center"], ell["a"], ell["b"], ell["theta"], occ,
                sample_mask=has_cand)
            return loss + safe

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
