"""V3 dataset (docs section 22).

Returns ONLY what the V3 network needs:

    pos, cond, maze_id, map_tensor
    candidate_features      [M, L, 5]   network feature path F_m^S
    candidate_mask          [M]
    candidate_lengths       [M]
    candidate_geometry      [M, G, 2]   DENSE safe cell chain (padded)
    candidate_geometry_lengths [M]      valid points per candidate
    topology_best           scalar      argmin_m nDTW(P0, S_m)
    has_candidate           scalar

There is no progress_gt, no ellipse_shape4_gt, no shape_valid, no
ellipse_center_gt and no ellipse_mask: V3 supervises topology / alignment /
safety / area directly, and the ellipse centre is a function of the network
output, not a precomputed label.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

MAZE_NAMES = ["umaze", "medium", "large"]
RES = 256


class SkeletonDatasetV3(Dataset):
    def __init__(self, split, source_root, v3_root, geometry_points=1280):
        self.split = str(split)
        self.geometry_points = int(geometry_points)
        split_dir = os.path.join(v3_root, self.split)
        source_dir = os.path.join(source_root, self.split)

        self.pos = np.load(os.path.join(source_dir, "positions.npy"))
        self.cond = np.load(os.path.join(source_dir, "conditions.npy"))
        self.mid = np.load(os.path.join(source_dir, "maze_id.npy"))

        self.features = np.load(os.path.join(split_dir, "candidate_features.npy"),
                                mmap_mode="r")
        self.mask = np.load(os.path.join(split_dir, "candidate_mask.npy"))
        self.lengths = np.load(os.path.join(split_dir, "candidate_lengths.npy"))
        self.offsets = np.load(os.path.join(split_dir,
                                            "candidate_geometry_offsets.npy"))
        self.geom_lengths = np.load(os.path.join(split_dir,
                                                 "candidate_geometry_lengths.npy"))
        self.geometry = np.load(os.path.join(split_dir, "candidate_geometry.npy"),
                                mmap_mode="r")
        self.best = np.load(os.path.join(split_dir, "topology_best.npy"))

        if self.geom_lengths.size and int(self.geom_lengths.max()) > self.geometry_points:
            raise ValueError(
                "dense geometry longer than the padding budget: %d > %d "
                "(raise topology.candidate_geometry_points)"
                % (int(self.geom_lengths.max()), self.geometry_points))

        maps_dir = os.path.join(source_root, "maps")
        self.maps = [torch.as_tensor(np.load(os.path.join(maps_dir, "%s.npy" % m)),
                                     dtype=torch.float32)[None, None]
                     for m in MAZE_NAMES]
        self.horizon = int(self.pos.shape[1])
        self.num_candidates = int(self.features.shape[1])
        self.cell = 2.0 / float(RES)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        maze_id = int(self.mid[idx])
        M, G = self.num_candidates, self.geometry_points
        geom = np.zeros((M, G, 2), dtype=np.float32)
        glen = self.geom_lengths[idx].astype(np.int64).copy()
        for m in range(M):
            n = int(glen[m])
            if n <= 0:
                glen[m] = 0
                continue
            lo, hi = int(self.offsets[idx, m]), int(self.offsets[idx, m + 1])
            px = np.asarray(self.geometry[lo:hi], dtype=np.float32)
            geom[m, :n] = (px + 0.5) * self.cell - 1.0
        mask = self.mask[idx].astype(bool)
        glen = np.where(mask, glen, 0)
        return {
            "pos": torch.as_tensor(self.pos[idx], dtype=torch.float32),
            "cond": torch.as_tensor(self.cond[idx], dtype=torch.float32),
            "maze_id": maze_id,
            "candidate_features": torch.as_tensor(
                np.array(self.features[idx]), dtype=torch.float32),
            "candidate_mask": torch.from_numpy(mask),
            "candidate_lengths": torch.as_tensor(self.lengths[idx],
                                                 dtype=torch.float32),
            "candidate_geometry": torch.from_numpy(geom),
            "candidate_geometry_lengths": torch.as_tensor(glen, dtype=torch.long),
            "topology_best": int(self.best[idx]),
            "has_candidate": bool(mask.any()),
        }


def make_collate(ds):
    def collate(batch):
        mid = torch.tensor([b["maze_id"] for b in batch], dtype=torch.long)
        return {
            "pos": torch.stack([b["pos"] for b in batch]),
            "cond": torch.stack([b["cond"] for b in batch]),
            "maze_id": mid,
            "map_tensor": torch.stack([ds.maps[i] for i in mid]).squeeze(1),
            "candidate_features": torch.stack([b["candidate_features"] for b in batch]),
            "candidate_mask": torch.stack([b["candidate_mask"] for b in batch]),
            "candidate_lengths": torch.stack([b["candidate_lengths"] for b in batch]),
            "candidate_geometry": torch.stack([b["candidate_geometry"] for b in batch]),
            "candidate_geometry_lengths": torch.stack(
                [b["candidate_geometry_lengths"] for b in batch]),
            "topology_best": torch.tensor([b["topology_best"] for b in batch],
                                          dtype=torch.long),
            "has_candidate": torch.tensor([b["has_candidate"] for b in batch],
                                          dtype=torch.bool),
        }
    return collate


def make_loader(split, source_root, v3_root, batch_size, shuffle,
                num_workers=0, geometry_points=1280):
    ds = SkeletonDatasetV3(split, source_root, v3_root,
                           geometry_points=geometry_points)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=False, collate_fn=make_collate(ds))
    return loader, ds
