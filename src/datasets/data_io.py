"""Helpers to load the mixed [0,8]^2 dataset for sampling / eval / plotting.

These replace the per-maze single-scene loaders so scripts can run against the
model (outputs/ckpt) on the mixed data.
"""
import os
import numpy as np
import torch

from .normalization import load_normalization
from .mixed_dataset import EXTENT
from src.utils.metrics import interp_trajectory
from src.geometry.sdf_utils import sample_sdf_torch

MAZE_INDEX = {"umaze": 0, "medium": 1, "large": 2}
DATA_BASE = "data/processed/mixed"


def load_maze(maze, split="test", n=None, base=DATA_BASE, rng=None):
    """Return (norm, occupancy[ny,nx], sdf[ny,nx], conds[N,2]) for one maze.

    conds are normalized start/goal positions from the split, filtered to
    the requested maze via maze_id.

    If ``rng`` is given (an ``numpy.random.Generator``), the ``n`` test cases
    are sampled randomly from the maze test split instead of always taking the
    first ``n``.  This lets callers run a different case set each time, while
    still being reproducible when ``rng`` is seeded.
    """
    if maze not in MAZE_INDEX:
        raise ValueError(f"unknown maze: {maze}")
    norm, _ = load_normalization(os.path.join(base, "normalization.json"))
    occ = np.load(os.path.join(base, "maps", f"{maze}.npy"))
    sdf = np.load(os.path.join(base, "maps", f"{maze}_sdf.npy"))
    mid = np.load(os.path.join(base, split, "maze_id.npy"))
    cond_all = np.load(os.path.join(base, split, "conditions.npy"))
    sel = np.where(mid == MAZE_INDEX[maze])[0]
    if n is not None:
        if rng is not None:
            n_sel = min(n, len(sel))
            sel = np.sort(rng.choice(sel, size=n_sel, replace=False))
        else:
            sel = sel[: n]
    return norm, occ, sdf, cond_all[sel]


def sdf_metrics(pos_world, sdf, device="cuda", extent=EXTENT, interp_steps=8):
    """Collision / clearance on the densified path, via the shared SDF field.

    pos_world: [H,2] world positions in [0,8]^2.  Returns a metrics dict.
    """
    dense = interp_trajectory(pos_world, interp_steps=interp_steps)
    sdf_t = torch.as_tensor(sdf, dtype=torch.float32).to(device)[None, None]
    pts = torch.as_tensor(dense, dtype=torch.float32).to(device)[None]
    d = sample_sdf_torch(sdf_t.expand(1, -1, -1, -1), pts, extent).cpu().numpy()[0]
    collision = d <= 0.0
    return {
        "collision_rate": float(np.mean(collision)),
        "mean_clearance": float(np.mean(d)),
        "clearance_p05": float(np.percentile(d, 5)),
        "n_collisions": int(np.sum(collision)),
        "n_dense_points": int(len(dense)),
    }


def unnorm_positions(p_norm, norm):
    mins = np.asarray(norm.mins[2:4], dtype=np.float64)
    maxs = np.asarray(norm.maxs[2:4], dtype=np.float64)
    eps = norm.eps
    p = np.asarray(p_norm, dtype=np.float64)
    return (p + 1.0) / 2.0 * (maxs - mins + eps) + mins
