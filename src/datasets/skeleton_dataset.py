"""Dataset for V2 (Skeleton-Topology-Grounded Trajectory Diffusion).

Returns exactly the V2 contract (docs/V2.md section 36):

    pos                 [H, 2]      GT trajectory (scene coords)
    cond                [2, 2]      scene (start, goal)
    maze_id             scalar
    map_tensor          [1, 256, 256]   occupancy, 1 = obstacle
    candidate_paths     [M, L, 5]   [x, y, tx, ty, u]
    candidate_mask      [M]
    candidate_lengths   [M]
    topology_target     [M]         soft nDTW target
    topology_best       scalar
    progress_gt         [K]
    ellipse_shape4_gt   [K, 4]      [log a, log b, cos 2t, sin 2t]
    shape_valid         [K]         fixed-centre IRIS label available
    ellipse_center_gt   [K, 2]      gamma_m*(s_gt) (for the GT mask / metrics)
    ellipse_mask        [K, R, R]   uint8 soft GT mask
    has_candidate       scalar

V2 does NOT return e6 and does not need the SDF: the ellipse centre is
c_i = gamma_m(s_i), and the GT mask is rasterised here with the very same soft
mask function the loss uses.

Trajectories come from the frozen V1 dataset (source_root); only the candidate
topologies, the progress target and the fixed-centre shape labels are new
(v2_root), so nothing is duplicated.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.geometry.ellipse_raster import ellipse_soft_mask
from src.geometry.skeleton_paths import interpolate_path

MAZE_NAMES = ["umaze", "medium", "large"]


class SkeletonDataset(Dataset):
    def __init__(self, split, source_root, v2_root, shape_dir=None,
                 mask_res=64, mask_tau=10.0):
        self.split = str(split)
        self.mask_res = int(mask_res)
        self.mask_tau = float(mask_tau)
        split_dir = os.path.join(v2_root, self.split)
        source_dir = os.path.join(source_root, self.split)

        self.pos = np.load(os.path.join(source_dir, "positions.npy"))
        self.cond = np.load(os.path.join(source_dir, "conditions.npy"))
        self.mid = np.load(os.path.join(source_dir, "maze_id.npy"))

        self.cand_paths = np.load(os.path.join(split_dir, "candidate_paths.npy"),
                                  mmap_mode="r")
        self.cand_mask = np.load(os.path.join(split_dir, "candidate_mask.npy"))
        self.cand_lengths = np.load(os.path.join(split_dir, "candidate_lengths.npy"))
        self.topo_target = np.load(os.path.join(split_dir, "topology_target.npy"))
        self.topo_best = np.load(os.path.join(split_dir, "topology_best.npy"))
        self.progress_gt = np.load(os.path.join(split_dir, "progress_gt.npy"))

        self.shape_dir = shape_dir or os.path.join(v2_root, "skeletons")
        shape_dir = self.shape_dir
        self.shape4 = [np.load(os.path.join(shape_dir, "%s_shape4.npy" % m))
                       for m in MAZE_NAMES]
        self.shape_valid = [np.load(os.path.join(shape_dir,
                                                 "%s_shape_valid.npy" % m))
                            for m in MAZE_NAMES]
        maps_dir = os.path.join(source_root, "maps")
        self.maps = [torch.as_tensor(np.load(os.path.join(maps_dir, "%s.npy" % m)),
                                     dtype=torch.float32)[None, None]
                     for m in MAZE_NAMES]
        self.horizon = int(self.pos.shape[1])
        self.shape_res = int(self.shape4[0].shape[0])
        self._near_skeleton = {}

    def __len__(self):
        return len(self.pos)

    # ------------------------------------------------------------------
    def _nearest_skeleton_indices(self, maze_id):
        """Precomputed nearest-skeleton-pixel index map for one maze."""
        cached = self._near_skeleton.get(maze_id)
        if cached is None:
            from scipy import ndimage

            from src.geometry.skeleton_graph import load_graph_npz

            path = os.path.join(self.shape_dir, "%s.npz" % MAZE_NAMES[maze_id])
            skeleton = load_graph_npz(path).skeleton
            if not skeleton.any():
                raise ValueError("empty skeleton cache for maze %d" % maze_id)
            _, indices = ndimage.distance_transform_edt(~skeleton,
                                                        return_indices=True)
            cached = (indices[0].astype(np.int32), indices[1].astype(np.int32))
            self._near_skeleton[maze_id] = cached
        return cached

    def shape_at(self, maze_id, center_scene):
        """Fixed-centre IRIS label of the NEAREST skeleton pixel + validity.

        The continuous centre stays gamma_m(s_i); the connector cells of a
        candidate are not skeleton pixels, so the nearest skeleton pixel is used
        for supervision (docs/V2.md section 25).
        """
        iy, ix = self._nearest_skeleton_indices(int(maze_id))
        res = self.shape_res
        center_scene = np.asarray(center_scene, dtype=np.float64).reshape(-1, 2)
        px = np.clip(np.rint((center_scene[:, 0] + 1.0) / 2.0 * res - 0.5),
                     0, res - 1).astype(int)
        py = np.clip(np.rint((center_scene[:, 1] + 1.0) / 2.0 * res - 0.5),
                     0, res - 1).astype(int)
        sy = iy[py, px]
        sx = ix[py, px]
        s4 = self.shape4[maze_id][sy, sx]
        ok = self.shape_valid[maze_id][sy, sx]
        return s4.astype(np.float32), ok.astype(bool)

    def _shape_lookup(self, maze_id, center_scene):
        return self.shape_at(maze_id, center_scene)

    def __getitem__(self, idx):
        maze_id = int(self.mid[idx])
        cand = np.array(self.cand_paths[idx], dtype=np.float32)    # [M,L,5]
        mask = self.cand_mask[idx].astype(bool)
        best = int(self.topo_best[idx])
        progress = self.progress_gt[idx].astype(np.float32)

        has_candidate = bool(mask.any())
        if has_candidate:
            path = cand[best, :, :2].astype(np.float64)
            center = interpolate_path(path, progress.astype(np.float64))
            shape4_gt, ok = self._shape_lookup(maze_id, center)
        else:
            center = np.zeros((self.horizon, 2), dtype=np.float64)
            shape4_gt = np.zeros((self.horizon, 4), dtype=np.float32)
            ok = np.zeros(self.horizon, dtype=bool)
            progress = np.zeros(self.horizon, dtype=np.float32)

        # GT soft mask, rasterised with the SAME function the loss uses
        c_t = torch.from_numpy(center).float()[None]
        s4_t = torch.from_numpy(shape4_gt).float()[None]
        a = torch.exp(s4_t[..., 0].clamp(-8.0, 8.0))
        b = torch.exp(s4_t[..., 1].clamp(-8.0, 8.0))
        theta = 0.5 * torch.atan2(s4_t[..., 3], s4_t[..., 2])
        soft = ellipse_soft_mask(c_t, a, b, theta, self.mask_res, self.mask_tau)
        gt_mask = (soft[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

        return {
            "pos": torch.as_tensor(self.pos[idx], dtype=torch.float32),
            "cond": torch.as_tensor(self.cond[idx], dtype=torch.float32),
            "maze_id": maze_id,
            "candidate_paths": torch.from_numpy(cand),
            "candidate_mask": torch.from_numpy(mask),
            "candidate_lengths": torch.as_tensor(self.cand_lengths[idx],
                                                 dtype=torch.float32),
            "topology_target": torch.as_tensor(self.topo_target[idx],
                                               dtype=torch.float32),
            "topology_best": best,
            "progress_gt": torch.from_numpy(progress),
            "ellipse_shape4_gt": torch.from_numpy(shape4_gt),
            "shape_valid": torch.from_numpy(ok),
            "ellipse_center_gt": torch.from_numpy(center.astype(np.float32)),
            "ellipse_mask": gt_mask,
            "has_candidate": has_candidate,
        }


def make_collate(ds):
    def collate(batch):
        mid = torch.tensor([b["maze_id"] for b in batch], dtype=torch.long)
        return {
            "pos": torch.stack([b["pos"] for b in batch]),
            "cond": torch.stack([b["cond"] for b in batch]),
            "maze_id": mid,
            "map_tensor": torch.stack([ds.maps[i] for i in mid]).squeeze(1),
            "candidate_paths": torch.stack([b["candidate_paths"] for b in batch]),
            "candidate_mask": torch.stack([b["candidate_mask"] for b in batch]),
            "candidate_lengths": torch.stack([b["candidate_lengths"] for b in batch]),
            "topology_target": torch.stack([b["topology_target"] for b in batch]),
            "topology_best": torch.tensor([b["topology_best"] for b in batch],
                                          dtype=torch.long),
            "progress_gt": torch.stack([b["progress_gt"] for b in batch]),
            "ellipse_shape4_gt": torch.stack([b["ellipse_shape4_gt"] for b in batch]),
            "shape_valid": torch.stack([b["shape_valid"] for b in batch]),
            "ellipse_center_gt": torch.stack([b["ellipse_center_gt"] for b in batch]),
            "ellipse_mask": torch.stack([b["ellipse_mask"] for b in batch]),
            "has_candidate": torch.tensor([bool(b["has_candidate"]) for b in batch],
                                          dtype=torch.bool),
        }
    return collate


def make_loader(split, source_root, v2_root, batch_size, shuffle,
                num_workers=0, mask_res=64, mask_tau=10.0, shape_dir=None):
    ds = SkeletonDataset(split, source_root, v2_root, shape_dir=shape_dir,
                         mask_res=mask_res, mask_tau=mask_tau)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=False, collate_fn=make_collate(ds))
    return loader, ds
