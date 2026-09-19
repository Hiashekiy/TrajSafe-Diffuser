"""Dataset for the TrajSafe-Diffuser.

Vocabulary:

    candidate_xy          [M, L, 2]   fixed-length network Skeleton S_m
    candidate_geometry    [M, G, 2]   dense safe Skeleton Curve Gamma_m
    ellipse_shape4_gt     [H, 4]      [log a*, log b*, cos 2t*, sin 2t*]
    shape_valid           [H]         whether a safe shape label exists
    ellipse_mask          [H, R_e, R_e]  soft GT mask from the SAME rasteriser
                                      used by the loss, centred on the GT
                                      trajectory waypoint p_i^GT

There is intentionally NO ``ellipse_center_gt`` and NO ``progress_gt`` in the
training dependency chain: the Progress Head is supervised by comparing the
decoded centre c_i = Gamma(s_i) directly with p_i^GT (SmoothL1), not with a
projected skeleton label.

The full soft masks are generated lazily in ``__getitem__`` from the compact
``shape4_gt`` labels and the GT trajectory.  They are numerically identical to a
stored pre-computation because :func:`ellipse_soft_mask` is deterministic, and
lazy generation avoids the ~18 GB that a stored [17295, 128, 64, 64] float
tensor would require on this machine.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.geometry.ellipse_raster import ellipse_soft_mask
from src.geometry.ellipse_shape import shape4_to_abtheta

MAZE_NAMES = ["umaze", "medium", "large"]
RES = 256

_LABEL_FILES = {
    "ellipse_shape4_gt": "ellipse_shape4_gt.npy",
    "shape_valid": "shape_valid.npy",
}


class SkeletonDataset(Dataset):
    def __init__(self, split, scenes_root, skeleton_root, geometry_points=1280,
                 ellipse_mask_res=64, ellipse_mask_tau=10.0, mazes=None):
        self.split = str(split)
        self.geometry_points = int(geometry_points)
        self.ellipse_mask_res = int(ellipse_mask_res)
        self.ellipse_mask_tau = float(ellipse_mask_tau)
        split_dir = os.path.join(skeleton_root, self.split)
        source_dir = os.path.join(scenes_root, self.split)

        self.pos = np.load(os.path.join(source_dir, "positions.npy"))
        self.cond = np.load(os.path.join(source_dir, "conditions.npy"))
        self.mid = np.load(os.path.join(source_dir, "maze_id.npy"))
        self.mazes = None if mazes is None else list(mazes)
        if self.mazes:
            ids = [MAZE_NAMES.index(m) if isinstance(m, str) else int(m)
                   for m in self.mazes]
            self.indices = np.nonzero(np.isin(self.mid, ids))[0]
        else:
            self.indices = np.arange(len(self.mid))

        xy_path = os.path.join(split_dir, "candidate_xy.npy")
        self.xy = (np.load(xy_path, mmap_mode="r") if os.path.exists(xy_path)
                   else None)
        if self.xy is None:
            # Legacy candidate_features.npy stores [x, y, tx, ty, u]; only the
            # first two columns are the report's network input S_m.
            self.features = np.load(
                os.path.join(split_dir, "candidate_features.npy"),
                mmap_mode="r")
        else:
            self.features = None
        self.mask = np.load(os.path.join(split_dir, "candidate_mask.npy"))
        self.offsets = np.load(os.path.join(split_dir,
                                            "candidate_geometry_offsets.npy"))
        self.geom_lengths = np.load(os.path.join(split_dir,
                                                 "candidate_geometry_lengths.npy"))
        self.geometry = np.load(os.path.join(split_dir, "candidate_geometry.npy"),
                                mmap_mode="r")
        self.best = np.load(os.path.join(split_dir, "topology_best.npy"))

        labels = {}
        for key, name in _LABEL_FILES.items():
            path = os.path.join(split_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "missing %s; run\n"
                    "  python scripts/data/03_build_ellipse_labels.py "
                    "--config <config>" % path)
            labels[key] = np.load(path)
        self.shape4_gt = labels["ellipse_shape4_gt"]
        self.shape_valid = labels["shape_valid"].astype(bool)

        if self.geom_lengths.size and int(self.geom_lengths.max()) > self.geometry_points:
            raise ValueError(
                "dense geometry longer than the padding budget: %d > %d "
                "(raise topology.candidate_geometry_points)"
                % (int(self.geom_lengths.max()), self.geometry_points))

        maps_dir = os.path.join(scenes_root, "maps")
        self.maps = [torch.as_tensor(np.load(os.path.join(maps_dir, "%s.npy" % m)),
                                     dtype=torch.float32)[None, None]
                     for m in MAZE_NAMES]
        self.horizon = int(self.pos.shape[1])
        src = self.xy if self.xy is not None else self.features
        self.num_candidates = int(src.shape[1])
        self.candidate_points = int(src.shape[2])
        self.cell = 2.0 / float(RES)

    def __len__(self):
        return len(self.indices)

    # ------------------------------------------------------------- geometry
    def _dense_geometry(self, idx: int, cond: np.ndarray) -> tuple:
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
            # Endpoints are continuous start/goal, not cell centres; restore
            # them exactly so the offline labels and the online Gamma_m agree.
            if n >= 2:
                geom[m, 0] = cond[0]
                geom[m, n - 1] = cond[1]
        return geom, glen

    def _candidate_xy(self, idx: int) -> np.ndarray:
        if self.xy is not None:
            return np.array(self.xy[idx], dtype=np.float32)
        # [M, L, 5] -> [M, L, 2]
        return np.array(self.features[idx, :, :, :2], dtype=np.float32)

    def _ellipse_mask(self, center: np.ndarray, shape4: np.ndarray,
                      valid: np.ndarray, maze_id: int) -> torch.Tensor:
        res = self.ellipse_mask_res
        H = int(center.shape[0])
        if res <= 0:
            return torch.zeros(H, 1, 1, dtype=torch.float32)
        c = torch.as_tensor(center, dtype=torch.float32)
        s4 = torch.as_tensor(shape4, dtype=torch.float32)
        a, b, theta = shape4_to_abtheta(s4)
        occ = self.maps[maze_id]
        with torch.no_grad():
            mask = ellipse_soft_mask(c[None], a[None], b[None], theta[None],
                                     int(res), self.ellipse_mask_tau)[0]
        valid_t = torch.as_tensor(valid.astype(np.float32))
        return mask * valid_t[:, None, None]

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        maze_id = int(self.mid[i])
        cond = np.asarray(self.cond[i], dtype=np.float32).reshape(2, 2)
        geom, glen = self._dense_geometry(i, cond)

        mask = self.mask[i].astype(bool)
        glen = np.where(mask, glen, 0)
        shape_valid = self.shape_valid[i].copy()
        shape_valid = np.logical_and(shape_valid, mask.any())
        pos = np.asarray(self.pos[i], dtype=np.float32)
        shape4_gt = np.asarray(self.shape4_gt[i], dtype=np.float32)
        # GT mask is centred on the GT trajectory waypoint; no centre label.
        ellipse_mask = self._ellipse_mask(pos, shape4_gt, shape_valid, maze_id)
        return {
            "pos": torch.as_tensor(pos, dtype=torch.float32),
            "cond": torch.from_numpy(cond),
            "maze_id": maze_id,
            "candidate_xy": torch.from_numpy(self._candidate_xy(i)),
            "candidate_mask": torch.from_numpy(mask),
            "candidate_geometry": torch.from_numpy(geom),
            "candidate_geometry_lengths": torch.as_tensor(glen, dtype=torch.long),
            "topology_best": int(self.best[i]),
            "has_candidate": bool(mask.any()),
            "ellipse_shape4_gt": torch.from_numpy(shape4_gt),
            "shape_valid": torch.from_numpy(shape_valid),
            "ellipse_mask": ellipse_mask,
        }


def make_collate(ds):
    def collate(batch):
        mid = torch.tensor([b["maze_id"] for b in batch], dtype=torch.long)
        return {
            "pos": torch.stack([b["pos"] for b in batch]),
            "cond": torch.stack([b["cond"] for b in batch]),
            "maze_id": mid,
            "map_tensor": torch.stack([ds.maps[i] for i in mid]).squeeze(1),
            "candidate_xy": torch.stack([b["candidate_xy"] for b in batch]),
            "candidate_mask": torch.stack([b["candidate_mask"] for b in batch]),
            "candidate_geometry": torch.stack(
                [b["candidate_geometry"] for b in batch]),
            "candidate_geometry_lengths": torch.stack(
                [b["candidate_geometry_lengths"] for b in batch]),
            "topology_best": torch.tensor([b["topology_best"] for b in batch],
                                          dtype=torch.long),
            "has_candidate": torch.tensor([b["has_candidate"] for b in batch],
                                          dtype=torch.bool),
            "ellipse_shape4_gt": torch.stack(
                [b["ellipse_shape4_gt"] for b in batch]),
            "shape_valid": torch.stack([b["shape_valid"] for b in batch]),
            "ellipse_mask": torch.stack([b["ellipse_mask"] for b in batch]),
        }
    return collate


def make_loader(split, scenes_root, skeleton_root, batch_size, shuffle,
                num_workers=0, geometry_points=1280,
                ellipse_mask_res=64, ellipse_mask_tau=10.0, mazes=None):
    ds = SkeletonDataset(split, scenes_root, skeleton_root,
                           geometry_points=geometry_points,
                           ellipse_mask_res=ellipse_mask_res,
                           ellipse_mask_tau=ellipse_mask_tau, mazes=mazes)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=False, collate_fn=make_collate(ds))
    return loader, ds
