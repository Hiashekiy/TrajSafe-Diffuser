"""Verify the k=4 border-protected TEST cache against raw/k8p and the loader.

Checks
------
1. every artefact of ``data/carla_processed_160k4p/test`` loads through
   ``CarlaSplineDataset`` (the exact loader training/eval use);
2. deployed GT geometry sanity: GT free rate (``sampler._free_mask``) and the
   straight start->goal line on raw / k4p / k8p occupancy;
3. GT vs the OFFLINE corridor of its own cache (must be feasible: the pack is
   built from the GT route);
4. per-sample free-space delta vs k8p, plus the two known hard cases.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate  # noqa: E402
from src.diffusion.sampler import _free_mask  # noqa: E402

ROOTS = {
    "raw160": "data/carla_processed_160",
    "k4p": "data/carla_processed_160k4p",
    "k8p": "data/carla_processed_160k8p",
}
HARD = {"test_0167": 167, "test_0291": 291}


def offline_violation(p: torch.Tensor, batch: dict):
    """(max_violation, membership_rate) of a curve vs the offline GT corridor."""
    a = batch["alm_cell_a"].detach().float()
    b = batch["alm_cell_b"].detach().float()
    cv = batch["alm_cell_valid"].detach().bool()
    nrm = a.norm(dim=-1).clamp_min(1e-9)
    signed = (torch.einsum("bcfk,bhk->bhcf", a, p) - b[:, None]) / nrm[:, None]
    worst = signed.amax(dim=-1).masked_fill(~cv[:, None, :], float("inf"))
    viol = torch.nan_to_num(worst.amin(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    return viol.max(dim=1).values, (viol <= 0).float().mean(dim=1)


def main():
    dev = "cpu"
    sets = {}
    for tag, root in ROOTS.items():
        ds = CarlaSplineDataset(split="test", processed_root=root,
                                num_controls=32, num_safety_queries=128)
        batch = make_collate(ds)([ds[i] for i in range(len(ds))])
        sets[tag] = batch
        print("[load] %-6s n=%d ok keys=%d" % (tag, len(ds), len(batch)))

    base = sets["k8p"]
    n = int(base["pos"].shape[0])
    cond = base["cond"]
    print("\n=== 1. GT geometry (deployed _free_mask over 128 GT points) ===")
    gt_free, line_free, gt_viol, gt_member = {}, {}, {}, {}
    for tag, batch in sets.items():
        pos, occ = batch["pos"], batch["occupancy"]
        fr = []
        for i in range(n):
            fr.append(float(_free_mask(occ[i], pos[i]).float().mean()))
        gt_free[tag] = np.asarray(fr)
        # straight start->goal line, 128 points
        s, g = cond[:, 0], cond[:, 1]
        alpha = torch.linspace(0.0, 1.0, 128).view(1, 128, 1)
        line = s[:, None, :] * (1 - alpha) + g[:, None, :] * alpha
        lf = [float(_free_mask(occ[i], line[i]).float().mean()) for i in range(n)]
        line_free[tag] = np.asarray(lf)
        v, m = offline_violation(pos, batch)
        gt_viol[tag] = v.numpy()
        gt_member[tag] = m.numpy()
        print("%-6s GT free mean=%.4f min=%.4f  collide(any pt)=%d/%d | "
              "straight line free mean=%.4f collide=%d/%d | "
              "GT vs own corridor: max=%.4f feasible=%d/%d member=%.4f"
              % (tag, gt_free[tag].mean(), gt_free[tag].min(),
                 int((gt_free[tag] < 1.0).sum()), n,
                 line_free[tag].mean(), int((line_free[tag] < 1.0).sum()), n,
                 float(gt_viol[tag].max()), int((gt_viol[tag] <= 1e-3).sum()), n,
                 float(gt_member[tag].mean())))

    print("\n=== 2. asset statistics ===")
    for tag, batch in sets.items():
        cv = batch["alm_cell_valid"].detach().bool()
        ok = batch["alm_valid"].detach().bool()
        cm = batch["candidate_mask"].detach().bool()
        sv = batch["shape_valid"].detach().bool()
        print("%-6s alm_valid=%.4f cells/sample=%.1f  candidates valid=%.2f "
              "empty=%.3f  shape_valid=%.4f  topo best hist=%s"
              % (tag, ok.float().mean(), cv.sum(1).float().mean(),
                 cm.sum(1).float().mean(), float((cm.sum(1) == 0).float().mean()),
                 sv.float().mean(),
                 np.bincount(batch["topology_best"].numpy(), minlength=4).tolist()))

    print("\n=== 3. k4p vs k8p difficulty (GT is free in every map, so use AREA "
          "and the straight start->goal line) ===")
    for tag, batch in sets.items():
        occ = batch["occupancy"]
        area = np.asarray([float((occ[i] == 0).float().mean()) for i in range(n)])
        print("%-6s free AREA mean=%.4f min=%.4f  straight-line free mean=%.4f "
              "collide=%d/%d" % (tag, area.mean(), area.min(), line_free[tag].mean(),
                                 int((line_free[tag] < 1.0).sum()), n))
    d = line_free["k4p"] - line_free["k8p"]
    print("k4p-vs-k8p straight-line free delta: mean=%+.4f min=%+.4f max=%+.4f  "
          "harder(k4p lower)=%d/%d  easier=%d/%d  equal=%d"
          % (d.mean(), d.min(), d.max(), int((d < -1e-9).sum()), n,
             int((d > 1e-9).sum()), n, int((np.abs(d) <= 1e-9).sum())))
    hard = np.argsort(line_free["k4p"])[:10]
    print("lowest k4p straight-line free: " + ", ".join(
        "%d(%.3f|k8 %.3f)" % (i, line_free["k4p"][i], line_free["k8p"][i]) for i in hard))
    for name, i in HARD.items():
        print("%-10s k4p free=%.4f line=%.4f corr(max=%.4f member=%.3f) | "
              "k8p free=%.4f line=%.4f corr(max=%.4f member=%.3f)"
              % (name, gt_free["k4p"][i], line_free["k4p"][i], gt_viol["k4p"][i],
                 gt_member["k4p"][i], gt_free["k8p"][i], line_free["k8p"][i],
                 gt_viol["k8p"][i], gt_member["k8p"][i]))
    print("\n[ok] k4p test cache is loadable and self-consistent")


if __name__ == "__main__":
    main()
