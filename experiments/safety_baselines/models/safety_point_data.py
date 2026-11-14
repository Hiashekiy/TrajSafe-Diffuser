"""Dataset for the native-point safety model: 128-point curves + candidate topology + labels.

Everything below already exists in `data/processed/160m-k4p-c48/` (the same cache S-C48
was trained on), so no data regeneration is required:

    curve_gt.npy           (N,128,2)      x0 target (arc-length parameterised polyline)
    occupancy.npy          (N,256,256)    map
    conditions.npy         (N,2,2)        start / goal
    candidate_xy.npy       (N,M,128,2)    M candidate paths (the safety queries anchor here)
    candidate_mask.npy     (N,M)          valid candidates
    candidate_geometry*.npy               dense candidate polylines (variable length, packed)
    topology_best.npy      (N,)           offline m* (argmin nDTW); used as the CE target
    ellipse_shape4_gt.npy  (N,128,4)      [log a, log b, cos2t, sin2t] per anchor
    shape_valid.npy        (N,128)        label validity
    alm_cell_a/b/valid.npy (N,128,12,2)/(N,128,12)/(N,128)   corridor half-space pack
    alm_anchor_s.npy       (N,128)        corridor progress parameter

Design notes
------------
* `candidate_geometry` is packed (`offsets`/`lengths`). Each sample is resampled to a FIXED
  `geometry_points` grid by uniform arc length, because the network multiplies it by a
  batch tensor; the resampling uses the same helper the S-C48 dataset used, so the anchors
  c_i = Gamma_m(s_i) keep their meaning (s = normalised arc length).
* No B-spline anything: `control_gt.npy` is deliberately NOT loaded.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.trajsafe.geometry import gather_dense_path_points


class SafetyPointDataset(Dataset):
    def __init__(self, split, root, geometry_points=512, limit=None, indices=None):
        self.dir = Path(root) / str(split)
        if not self.dir.is_dir():
            raise FileNotFoundError('processed split not found: %s' % self.dir)
        self.split = str(split)
        self.geometry_points = int(geometry_points)
        load = lambda n, mm=False: np.load(self.dir / n, mmap_mode='r' if mm else None)
        self.points = load('curve_gt.npy', mm=True)
        self.occupancy = load('occupancy.npy', mm=True)
        self.conditions = load('conditions.npy')
        self.candidate_xy = load('candidate_xy.npy', mm=True)
        self.candidate_mask = load('candidate_mask.npy')
        self.candidate_len = load('candidate_lengths.npy')
        self.offsets = np.asarray(load('candidate_geometry_offsets.npy'))
        self.geom_len = np.asarray(load('candidate_geometry_lengths.npy'))
        self.geometry = load('candidate_geometry.npy', mm=True)
        self.best = np.asarray(load('topology_best.npy'))
        self.shape4 = load('ellipse_shape4_gt.npy')
        self.shape_valid = load('shape_valid.npy')
        self.cell_a = load('alm_cell_a.npy', mm=True)
        self.cell_b = load('alm_cell_b.npy', mm=True)
        self.cell_valid = load('alm_cell_valid.npy', mm=True)
        self.anchor_s = load('alm_anchor_s.npy', mm=True)
        self.sample_id = load('sample_id.npy')
        n = len(self.points)
        self.size = n if limit is None else min(int(limit), n)
        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
            self.size = len(self.indices)
        else:
            self.indices = np.arange(self.size)

    def __len__(self):
        return self.size

    def _dense_geometry(self, i, cond):
        """[M, G, 2] candidate polylines in scene coords, plus their lengths.

        The candidate path at the codec resolution (`candidate_xy`, 128 points) is used as
        the geometry instead of the packed `candidate_geometry` polyline: it is the SAME
        polyline the skeleton encoder consumes, `candidate_xy` is already in scene
        coordinates, and it removes a per-sample Python interpolation loop (measured
        ~3 min/epoch -> ~40 s/epoch). Declared difference from the S-C48 dataset, which
        resampled the packed polyline (1280 cells) instead.
        """
        M = self.candidate_xy.shape[1]
        xy = np.asarray(self.candidate_xy[i], dtype=np.float32).copy()   # [M,128,2]
        xy[:, 0] = cond[0]
        xy[:, -1] = cond[1]
        glen = np.full((M,), xy.shape[1], dtype=np.int64)
        mask = np.asarray(self.candidate_mask[i]).astype(bool)
        glen = np.where(mask, glen, 0)
        return torch.from_numpy(xy), torch.as_tensor(glen, dtype=torch.long)

    def __getitem__(self, k):
        i = int(self.indices[k])
        cond = np.asarray(self.conditions[i], dtype=np.float32).reshape(2, 2)
        geom, glen = self._dense_geometry(i, cond)
        mask = np.asarray(self.candidate_mask[i]).astype(bool)
        glen = torch.where(torch.as_tensor(mask), glen, torch.zeros_like(glen))
        c = lambda a, dt=np.float32: torch.from_numpy(np.array(a, dtype=dt))
        return {
            'curve_gt': c(self.points[i]),
            'occupancy': c(self.occupancy[i])[None],
            'cond': torch.from_numpy(cond),
            'candidate_xy': c(self.candidate_xy[i]),
            'candidate_mask': torch.from_numpy(mask),
            'geometry': geom,
            'geometry_lengths': glen,
            'topology_best': torch.tensor(int(self.best[i]), dtype=torch.long),
            'shape4_gt': c(self.shape4[i]),
            'shape_valid': torch.from_numpy(np.asarray(self.shape_valid[i]).astype(bool)),
            'cell_a': c(self.cell_a[i]),
            'cell_b': c(self.cell_b[i]),
            'cell_valid': torch.from_numpy(np.asarray(self.cell_valid[i]).astype(bool)),
            'anchor_s': c(self.anchor_s[i]),
            'sample_id': torch.tensor(int(self.sample_id[i]), dtype=torch.long),
            'index': torch.tensor(i, dtype=torch.long),
        }


def make_safety_point_loader(split, root, batch_size, shuffle, num_workers=0, limit=None,
                             indices=None, geometry_points=512):
    ds = SafetyPointDataset(split, root, geometry_points=geometry_points, limit=limit, indices=indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        drop_last=False, pin_memory=True)
    return loader, ds
