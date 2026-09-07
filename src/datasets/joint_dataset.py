"""Dataset + loader for the V1 joint diffusion data (data/processed_scene_v1).

Layout:
  <split>/positions.npy  [N,H,2]   scene waypoints (p0=start ... p_{H-1}=goal)
  <split>/conditions.npy [N,2,2]   scene (start, goal)
  <split>/ellipses6.npy  [N,H,6]   e = [dx,dy,log a,log b,cos2t,sin2t], dx=c-p
  <split>/ellipse_masks64_u8.h5 [N,H,64,64] quantized GT soft masks
  <split>/maze_id.npy    [N]       0/1/2 -> umaze/medium/large
  maps/{maze}.npy        [256,256] occupancy (1=wall) over scene [-1,1]^2
  maps/{maze}_sdf.npy    [256,256] scene-unit signed distance (free>0)
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

RES = 256
MAZE_NAMES = ["umaze", "medium", "large"]


class JointDataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = split_dir
        self.pos = np.load(os.path.join(split_dir, "positions.npy"))      # [N,H,2]
        self.cond = np.load(os.path.join(split_dir, "conditions.npy"))    # [N,2,2]
        self.e6 = np.load(os.path.join(split_dir, "ellipses6.npy"))       # [N,H,6]
        self.mid = np.load(os.path.join(split_dir, "maze_id.npy"))        # [N]
        self.mask_path = os.path.join(split_dir, "ellipse_masks64_u8.h5")
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(
                f"Precomputed GT ellipse masks not found: {self.mask_path}. "
                "Run scripts/data/09_precompute_gt_ellipse_masks.py first."
            )
        with h5py.File(self.mask_path, "r") as f:
            masks = f["masks"]
            if masks.shape[:2] != self.e6.shape[:2]:
                raise ValueError(
                    f"GT mask shape {masks.shape} does not match e6 {self.e6.shape}"
                )
            self.mask_res = int(masks.shape[-1])
            self.mask_tau = float(masks.attrs["tau"])
        self._mask_file = None
        maps_dir = os.path.join(os.path.dirname(split_dir), "maps")
        self.maps = [torch.as_tensor(np.load(os.path.join(maps_dir, f"{m}.npy")),
                                     dtype=torch.float32)[None, None]
                     for m in MAZE_NAMES]
        self.sdfs = [torch.as_tensor(np.load(os.path.join(maps_dir, f"{m}_sdf.npy")),
                                     dtype=torch.float32)[None, None]
                     for m in MAZE_NAMES]

    def __len__(self):
        return len(self.pos)

    def _masks(self):
        # Open lazily so Dataset construction remains safe if DataLoader worker
        # settings change. The active Windows configuration uses num_workers=0.
        if self._mask_file is None:
            self._mask_file = h5py.File(self.mask_path, "r")
        return self._mask_file["masks"]

    def __getitem__(self, idx):
        return {
            "pos": torch.as_tensor(self.pos[idx], dtype=torch.float32),
            "cond": torch.as_tensor(self.cond[idx], dtype=torch.float32),
            "e6": torch.as_tensor(self.e6[idx], dtype=torch.float32),
            # Quantized soft mask [H,R,R]. Keep uint8 through collation and
            # transfer; the loss converts only one trajectory chunk at a time.
            "ellipse_mask": torch.from_numpy(self._masks()[idx]),
            "maze_id": int(self.mid[idx]),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mask_file"] = None
        return state

    def __del__(self):
        mask_file = getattr(self, "_mask_file", None)
        if mask_file is not None:
            try:
                mask_file.close()
            except Exception:
                pass


def make_collate(ds):
    def collate(batch):
        mid = torch.tensor([b["maze_id"] for b in batch], dtype=torch.long)
        return {
            "pos": torch.stack([b["pos"] for b in batch]),
            "cond": torch.stack([b["cond"] for b in batch]),
            "e6": torch.stack([b["e6"] for b in batch]),
            "ellipse_mask": torch.stack([b["ellipse_mask"] for b in batch]),
            "maze_id": mid,
            "map_tensor": torch.stack([ds.maps[i] for i in mid]).squeeze(1),
            "sdf_tensor": torch.stack([ds.sdfs[i] for i in mid]).squeeze(1),
        }
    return collate


def make_loader(split_dir, batch_size, shuffle, num_workers=0):
    ds = JointDataset(split_dir)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=False, collate_fn=make_collate(ds)), ds
