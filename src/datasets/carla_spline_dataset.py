"""CARLA dataset for the control-space (32 B-spline controls) TrajSafe-Diffuser.

Consumes the FIXED processed snapshot written by ``scripts/data/carla/*``
(see ``docs/CARLA_BSPLINE_PIPELINE_SPEC.md``).  It never scans the live
``data/carla_v1`` directory, so a dataset that is still being collected cannot
change the DataLoader mid-training.

Returned per sample:

    pos                    [128,2]  curve GT (scene) = trajectory_128
    control_gt             [32,2]   endpoint-constrained control GT
    cond                   [2,2]    [start, goal]
    occupancy              [1,256,256] float32 (canonical, already y-flipped)
    candidate_xy           [M,128,2]
    candidate_mask         [M]      bool
    candidate_geometry     [M,G,2]  dense safe curve in scene coords (padded)
    candidate_geometry_lengths [M]  long
    topology_best          int      argmin_m nDTW(curve_gt, S_m)
    has_candidate          bool
    ellipse_shape4_gt      [128,4]
    shape_valid            [128]    bool

There is deliberately no progress target, no ellipse-centre target and no
stored ``ellipse_mask``: the ellipse centres are the fixed Skeleton centres
``Gamma_m(i/127)`` and the GT soft mask is built at training time from the
DETACHED predicted centres plus the offline ``shape4`` labels.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["CarlaSplineDataset", "make_collate", "make_loader", "RES"]

RES = 256
LOG_INTERVAL = 500


class CarlaSplineDataset(Dataset):
    def __init__(self, split, processed_root, geometry_points=1280,
                 limit=None, indices=None, require_labels=True):
        self.split = str(split)
        self.processed_root = str(processed_root)
        self.dir = os.path.join(self.processed_root, self.split)
        self.geometry_points = int(geometry_points)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(
                "processed split not found: %s\n"
                "run scripts/data/carla/00_clean_dataset.py, then "
                "01_build_candidates.py and 02_build_ellipse_labels.py" % self.dir)

        def _load(name, mmap=False):
            path = os.path.join(self.dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError("missing %s" % path)
            return np.load(path, mmap_mode="r" if mmap else None)

        self.conditions = _load("conditions.npy")
        self.control_gt = _load("control_gt.npy")
        self.curve_gt = _load("curve_gt.npy")
        self.occupancy = _load("occupancy.npy", mmap=True)
        self.episode_id = _load("episode_id.npy")
        self.sample_id = _load("sample_id.npy")
        self.candidate_xy = _load("candidate_xy.npy", mmap=True)
        self.candidate_mask = _load("candidate_mask.npy")
        self.candidate_lengths = _load("candidate_lengths.npy")
        self.offsets = _load("candidate_geometry_offsets.npy")
        self.geom_lengths = _load("candidate_geometry_lengths.npy")
        self.geometry = _load("candidate_geometry.npy", mmap=True)
        self.best = _load("topology_best.npy")
        # sampling / evaluation do not need the offline ellipse labels
        try:
            self.shape4_gt = _load("ellipse_shape4_gt.npy")
            self.shape_valid = _load("shape_valid.npy").astype(bool)
        except FileNotFoundError:
            if require_labels:
                raise
            n0 = len(self.conditions)
            self.shape4_gt = np.zeros((n0, 128, 4), np.float32)
            self.shape_valid = np.zeros((n0, 128), bool)

        n = len(self.conditions)
        if limit is not None:
            n = min(int(limit), n)
        self.indices = np.arange(n) if indices is None else np.asarray(indices)
        self.num_candidates = int(self.candidate_xy.shape[1])
        self.candidate_points = int(self.candidate_xy.shape[2])
        self.horizon = int(self.curve_gt.shape[1])
        self.num_controls = int(self.control_gt.shape[1])
        self.cell = 2.0 / float(RES)
        if int(self.geom_lengths.max(initial=0)) > self.geometry_points:
            raise ValueError(
                "dense geometry longer than the padding budget: %d > %d "
                "(raise topology.candidate_geometry_points)"
                % (int(self.geom_lengths.max()), self.geometry_points))

    def __len__(self):
        return len(self.indices)

    # ------------------------------------------------------------- geometry
    def _dense_geometry(self, i: int, cond: np.ndarray):
        M, G = self.num_candidates, self.geometry_points
        geom = np.zeros((M, G, 2), dtype=np.float32)
        glen = np.asarray(self.geom_lengths[i], dtype=np.int64).copy()
        for m in range(M):
            n = int(glen[m])
            if n <= 0:
                glen[m] = 0
                continue
            lo, hi = int(self.offsets[i, m]), int(self.offsets[i, m + 1])
            px = np.asarray(self.geometry[lo:hi], dtype=np.float32)
            n = min(n, len(px))
            glen[m] = n
            geom[m, :n] = (px[:n] + 0.5) * self.cell - 1.0
            if n >= 2:
                # endpoints are the continuous start/goal, not cell centres
                geom[m, 0] = cond[0]
                geom[m, n - 1] = cond[1]
        return geom, glen

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        cond = np.asarray(self.conditions[i], dtype=np.float32).reshape(2, 2)
        geom, glen = self._dense_geometry(i, cond)
        mask = np.asarray(self.candidate_mask[i]).astype(bool)
        glen = np.where(mask, glen, 0)
        shape_valid = np.asarray(self.shape_valid[i]).copy()
        shape_valid = np.logical_and(shape_valid, mask.any())
        occ = np.asarray(self.occupancy[i], dtype=np.float32)
        if occ.shape != (RES, RES):
            raise ValueError("occupancy[%d] has shape %s" % (i, occ.shape))
        return {
            "pos": torch.as_tensor(np.asarray(self.curve_gt[i], dtype=np.float32)),
            "control_gt": torch.as_tensor(
                np.asarray(self.control_gt[i], dtype=np.float32)),
            "cond": torch.from_numpy(cond),
            "occupancy": torch.from_numpy(occ)[None],
            "candidate_xy": torch.as_tensor(
                np.asarray(self.candidate_xy[i], dtype=np.float32)),
            "candidate_mask": torch.from_numpy(mask),
            "candidate_geometry": torch.from_numpy(geom),
            "candidate_geometry_lengths": torch.as_tensor(glen, dtype=torch.long),
            "candidate_lengths": torch.as_tensor(
                np.asarray(self.candidate_lengths[i], dtype=np.float32)),
            "topology_best": int(self.best[i]),
            "has_candidate": bool(mask.any()),
            "ellipse_shape4_gt": torch.as_tensor(
                np.asarray(self.shape4_gt[i], dtype=np.float32)),
            "shape_valid": torch.from_numpy(shape_valid),
            "episode_id": int(self.episode_id[i]),
            "sample_id": int(self.sample_id[i]),
        }


def make_collate(ds: CarlaSplineDataset):
    def collate(batch):
        out = {
            "pos": torch.stack([b["pos"] for b in batch]),
            "control_gt": torch.stack([b["control_gt"] for b in batch]),
            "cond": torch.stack([b["cond"] for b in batch]),
            "occupancy": torch.stack([b["occupancy"] for b in batch]),
            "candidate_xy": torch.stack([b["candidate_xy"] for b in batch]),
            "candidate_mask": torch.stack([b["candidate_mask"] for b in batch]),
            "candidate_geometry": torch.stack(
                [b["candidate_geometry"] for b in batch]),
            "candidate_geometry_lengths": torch.stack(
                [b["candidate_geometry_lengths"] for b in batch]),
            "candidate_lengths": torch.stack(
                [b["candidate_lengths"] for b in batch]),
            "topology_best": torch.tensor([b["topology_best"] for b in batch],
                                          dtype=torch.long),
            "has_candidate": torch.tensor([b["has_candidate"] for b in batch],
                                          dtype=torch.bool),
            "ellipse_shape4_gt": torch.stack(
                [b["ellipse_shape4_gt"] for b in batch]),
            "shape_valid": torch.stack([b["shape_valid"] for b in batch]),
            "episode_id": torch.tensor([b["episode_id"] for b in batch],
                                       dtype=torch.long),
            "sample_id": torch.tensor([b["sample_id"] for b in batch],
                                      dtype=torch.long),
        }
        # backward-compatible alias used by the sampler / dashboard
        out["map_tensor"] = out["occupancy"]
        return out
    return collate


def make_loader(split, processed_root, batch_size=16, shuffle=True,
                num_workers=0, geometry_points=1280, limit=None, indices=None,
                require_labels=True):
    ds = CarlaSplineDataset(split, processed_root,
                            geometry_points=geometry_points, limit=limit,
                            indices=indices, require_labels=require_labels)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=False, collate_fn=make_collate(ds))
    return loader, ds
