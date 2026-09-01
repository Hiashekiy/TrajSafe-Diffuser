"""Dataset + loader for the scene-normalized zero-sum bridge data.

Data layout (data/processed_scene):
  <split>/positions.npy       [N,H,2]  scene coords
  <split>/conditions.npy      [N,2,2]  scene (start, goal)
  <split>/ellipse_params.npy  [N,H-1,4] scene (center_xy, r1, r2)
  <split>/ellipse_Q.npy       [N,H-1,2,2] scene
  <split>/ellipse_valid.npy   [N,H-1]  bool
  <split>/maze_id.npy         [N]      0/1/2
  maps/{maze}.npy             [256,256] occupancy (1=wall)
  maps/{maze}_sdf.npy         [256,256] signed distance (free>0, wall<0)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

RES = 256
MAZE_NAMES = ["umaze", "medium", "large"]


class SceneDataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = split_dir
        self.pos = np.load(os.path.join(split_dir, "positions.npy"))
        self.cond = np.load(os.path.join(split_dir, "conditions.npy"))
        self.ep = np.load(os.path.join(split_dir, "ellipse_params.npy"))
        self.eq = np.load(os.path.join(split_dir, "ellipse_Q.npy"))
        self.ev = np.load(os.path.join(split_dir, "ellipse_valid.npy"))
        self.mid = np.load(os.path.join(split_dir, "maze_id.npy"))
        maps_dir = os.path.join(os.path.dirname(split_dir), "maps")
        self.maps = [torch.as_tensor(np.load(os.path.join(maps_dir, f"{m}.npy")),
                                     dtype=torch.float32)[None, None]
                     for m in MAZE_NAMES]
        self.sdfs = [torch.as_tensor(np.load(os.path.join(maps_dir, f"{m}_sdf.npy")),
                                     dtype=torch.float32)[None, None]
                     for m in MAZE_NAMES]

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        return {
            "maze_id": int(self.mid[idx]),
            "pos": torch.as_tensor(self.pos[idx], dtype=torch.float32),
            "cond": torch.as_tensor(self.cond[idx], dtype=torch.float32),
            "ellipse_params": torch.as_tensor(self.ep[idx], dtype=torch.float32),
            "ellipse_Q": torch.as_tensor(self.eq[idx], dtype=torch.float32),
            "ellipse_valid": torch.as_tensor(self.ev[idx], dtype=torch.bool),
        }


def make_collate(dataset):
    maps = dataset.maps
    sdfs = dataset.sdfs

    def collate(batch):
        mid = torch.tensor([b["maze_id"] for b in batch], dtype=torch.long)
        map_t = torch.stack([maps[i] for i in mid]).squeeze(1)   # [B,1,256,256]
        sdf_t = torch.stack([sdfs[i] for i in mid]).squeeze(1)
        return {
            "pos": torch.stack([b["pos"] for b in batch]),
            "cond": torch.stack([b["cond"] for b in batch]),
            "ellipse_params": torch.stack([b["ellipse_params"] for b in batch]),
            "ellipse_Q": torch.stack([b["ellipse_Q"] for b in batch]),
            "ellipse_valid": torch.stack([b["ellipse_valid"] for b in batch]),
            "map_tensor": map_t, "sdf_tensor": sdf_t, "maze_id": mid,
        }
    return collate


def make_loader(split_dir, batch_size, shuffle, num_workers=0):
    ds = SceneDataset(split_dir)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, drop_last=False,
                                       collate_fn=make_collate(ds))


def load_maze_scene(maze, split="test", n=None, base="data/processed_scene",
                    rng=None, extent=None):
    """Return (frame, occ, sdf, conds_scene) for one maze.

    extent: the maze's world extent (from config).  frame converts scene<->world.
    """
    from src.geometry.scene_frame import SceneFrame
    if maze not in MAZE_NAMES:
        raise ValueError(f"unknown maze: {maze}")
    frame = SceneFrame(extent)
    mi = MAZE_NAMES.index(maze)
    mid_all = np.load(os.path.join(base, split, "maze_id.npy"))
    cond_all = np.load(os.path.join(base, split, "conditions.npy"))
    sel = np.where(mid_all == mi)[0]
    if n is not None:
        if rng is not None:
            n_sel = min(n, len(sel))
            sel = np.sort(rng.choice(sel, size=n_sel, replace=False))
        else:
            sel = sel[:n]
    occ = np.load(os.path.join(base, "maps", f"{maze}.npy"))
    sdf = np.load(os.path.join(base, "maps", f"{maze}_sdf.npy"))
    return frame, occ, sdf, cond_all[sel]
