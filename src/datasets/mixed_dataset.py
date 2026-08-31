"""Sample-level mixed dataset + collate (per-sample map/SDF)."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

EXTENT = (0.0, 8.0, 0.0, 8.0)

class MixedDataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = split_dir
        self.traj = np.load(os.path.join(split_dir, "trajectories.npy"))
        self.cond = np.load(os.path.join(split_dir, "conditions.npy"))
        self.ep = np.load(os.path.join(split_dir, "ellipse_params.npy"))
        self.eq = np.load(os.path.join(split_dir, "ellipse_Q.npy"))
        self.ev = np.load(os.path.join(split_dir, "ellipse_valid.npy"))
        self.mid = np.load(os.path.join(split_dir, "maze_id.npy"))
        maps_dir = os.path.join(os.path.dirname(split_dir), "maps")
        self.maze_names = ["umaze", "medium", "large"]
        self.maps = [torch.as_tensor(np.load(os.path.join(maps_dir, f"{m}.npy")), dtype=torch.float32)[None, None] for m in self.maze_names]
        self.sdfs = [torch.as_tensor(np.load(os.path.join(maps_dir, f"{m}_sdf.npy")), dtype=torch.float32)[None, None] for m in self.maze_names]

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        return {
            "maze_id": int(self.mid[idx]),
            "traj": torch.as_tensor(self.traj[idx], dtype=torch.float32),
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
        map_t = torch.stack([maps[i] for i in mid]).squeeze(1)   # [B,1,80,80]
        sdf_t = torch.stack([sdfs[i] for i in mid]).squeeze(1)   # [B,1,80,80]
        return {
            "traj": torch.stack([b["traj"] for b in batch]),
            "cond": torch.stack([b["cond"] for b in batch]),
            "ellipse_params": torch.stack([b["ellipse_params"] for b in batch]),
            "ellipse_Q": torch.stack([b["ellipse_Q"] for b in batch]),
            "ellipse_valid": torch.stack([b["ellipse_valid"] for b in batch]),
            "map_tensor": map_t, "sdf_tensor": sdf_t, "maze_id": mid,
        }
    return collate

def make_loader(split_dir, batch_size, shuffle, num_workers=0):
    ds = MixedDataset(split_dir)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, drop_last=False,
                                       collate_fn=make_collate(ds))
