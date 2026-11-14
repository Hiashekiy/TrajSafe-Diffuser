"""Load only point targets, maps and endpoints; no control/skeleton files."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class PointDataset(Dataset):
    def __init__(self,split,root,num_points=128,limit=None):
        directory=Path(root)/split
        self.points=np.load(directory/'curve_gt.npy',mmap_mode='r')
        self.occupancy=np.load(directory/'occupancy.npy',mmap_mode='r')
        self.conditions=np.load(directory/'conditions.npy',mmap_mode='r')
        if self.points.shape[1:]!=(num_points,2):raise ValueError('trajectory target shape mismatch')
        if len({len(self.points),len(self.occupancy),len(self.conditions)})!=1:raise ValueError('dataset length mismatch')
        self.size=min(len(self.points),limit) if limit is not None else len(self.points)
    def __len__(self):return self.size
    def __getitem__(self,index):
        points=torch.tensor(np.array(self.points[index]),dtype=torch.float32)
        cond=torch.tensor(np.array(self.conditions[index]),dtype=torch.float32)
        occ=torch.tensor(np.array(self.occupancy[index]),dtype=torch.float32)
        if occ.ndim==2:occ=occ[None]
        if (points[[0,-1]]-cond).abs().max()>1e-4:raise ValueError('target endpoints differ from conditioning')
        return {'pos':points,'occupancy':occ,'cond':cond}

def make_point_loader(split,root,batch_size,shuffle,num_workers=0,limit=None,num_points=128):
    dataset=PointDataset(split,root,num_points,limit)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True),dataset
