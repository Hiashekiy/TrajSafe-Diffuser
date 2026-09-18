"""Does the topology selector actually earn its keep?

Only ~3% of the OD pairs in this dataset have more than one legal candidate, so
the end-to-end metrics cannot show it.  This script evaluates the selector
directly on exactly those samples:

    p_t = sqrt(ab_t) p_GT + sqrt(1-ab_t) eps   (t = commit_t, the sampler state
                                                at which the topology commits)
    pi  = model.score_candidates(encode_trajectory(p_t, ...))
    correct  <=>  argmax pi == topology_best   (the nDTW-best candidate)

    python scripts/debug/v2_selector_eval.py --ckpt outputs/v2_best_snapshot.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from src.models.skeleton import SkeletonPlanner
from src.datasets.skeleton_dataset import SkeletonDataset, make_collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v2_skeleton.yaml")
    ap.add_argument("--ckpt", default="outputs/v2_best_snapshot.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    topo = cfg.get("topology", {})
    commit_t = int(topo.get("commit_t", 7))
    source = cfg["data"].get("source", "data/processed_scene_v1")
    base_dir = cfg["data"].get("base", "data/processed_scene_v2")
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    ds = SkeletonDataset(args.split, source, base_dir,
                         mask_res=int(cfg["loss"].get("ellipse_safe_res", 64)),
                         mask_tau=float(cfg["loss"].get("ellipse_mask_tau", 10.0)))
    n_valid = ds.cand_mask.sum(axis=1)
    sel = np.nonzero(n_valid >= 2)[0]
    print("[selector] %s: %d / %d OD pairs have >= 2 candidates (%.2f%%)"
          % (args.split, len(sel), len(ds), 100.0 * len(sel) / len(ds)), flush=True)
    if len(sel) == 0:
        return

    schedule = NoiseSchedule(cfg["diffusion"]["timesteps"],
                             beta_schedule=cfg["diffusion"].get(
                                 "beta_schedule", "squaredcos_cap_v2")).to(device)
    model = SkeletonPlanner(cfg["model"], topo).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()

    torch.manual_seed(0)
    n_correct = 0
    margins, pi_best, pi_second, hit_at_1 = [], [], [], []
    per_class = {}
    acc_by_n = {}
    with torch.no_grad():
        for start in range(0, len(sel), 64):
            idxs = sel[start:start + 64]
            batch = make_collate(ds)([ds[int(i)] for i in idxs])
            cond = batch["cond"].to(device)
            occ = batch["map_tensor"].to(device)
            cand = batch["candidate_paths"].to(device)
            cmask = batch["candidate_mask"].to(device)
            clen = batch["candidate_lengths"].to(device)
            p0 = batch["pos"].to(device)
            best = batch["topology_best"].to(device)
            t = torch.full((len(idxs),), commit_t, device=device, dtype=torch.long)
            ab = schedule.sqrt_alphas_cumprod[t].to(device).float()
            s1 = schedule.sqrt_one_minus_alphas_cumprod[t].to(device).float()
            p_t = ab[:, None, None] * p0 + s1[:, None, None] * torch.randn_like(p0)
            p_t[:, 0] = cond[:, 0]
            p_t[:, -1] = cond[:, 1]
            base = model.encode_trajectory(p_t, occ, cond, t, ab)
            topo_out = model.score_candidates(base, cand, cmask, clen)
            pi = topo_out["pi"].cpu().numpy()
            pred = pi.argmax(axis=1)
            b = best.cpu().numpy()
            n_correct += int((pred == b).sum())
            for r in range(len(idxs)):
                k = int(n_valid[idxs[r]])
                acc_by_n.setdefault(k, []).append(float(pred[r] == b[r]))
                order = np.sort(pi[r])[::-1]
                pi_best.append(float(pi[r, b[r]]))
                margins.append(float(order[0] - order[1]))
                hit_at_1.append(float(pi[r, b[r]] >= order[0] - 1e-9))

    acc = n_correct / len(sel)
    res = {
        "split": args.split, "ckpt": args.ckpt, "commit_t": commit_t,
        "n_multi_candidate": int(len(sel)),
        "n_total": int(len(ds)),
        "selector_accuracy": float(acc),
        "chance_level_mean": float(np.mean([1.0 / n_valid[i] for i in sel])),
        "pi_of_best_mean": float(np.mean(pi_best)),
        "top1_top2_margin_mean": float(np.mean(margins)),
        "argmax_is_best_rate": float(np.mean(hit_at_1)),
        "accuracy_by_num_candidates": {
            str(k): float(np.mean(v)) for k, v in sorted(acc_by_n.items())},
        "count_by_num_candidates": {
            str(k): int(len(v)) for k, v in sorted(acc_by_n.items())},
    }
    out = args.out or os.path.join(os.path.dirname(args.ckpt),
                                   "selector_eval_%s.json" % args.split)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
