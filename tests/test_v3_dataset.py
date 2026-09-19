"""V3 dataset / visualisation regression tests.

The dataset must expose exactly the report's ellipse-label fields and the lazy
GT mask must be numerically identical to the shared soft rasteriser.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.skeleton_dataset_v3 import (MAZE_NAMES,  # noqa: E402
                                              SkeletonDatasetV3)
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.ellipse_raster import ellipse_soft_mask  # noqa: E402
from src.geometry.ellipse_shape import shape4_to_abtheta  # noqa: E402

from v3_utils import tiny_batch, tiny_model  # noqa: E402

V3_ROOT = os.path.join(REPO_ROOT, "data", "processed_scene_v3")
SOURCE_ROOT = os.path.join(REPO_ROOT, "data", "processed_scene_v1")
HAS_DATA = os.path.exists(os.path.join(V3_ROOT, "test", "ellipse_shape4_gt.npy"))

needs_data = pytest.mark.skipif(not HAS_DATA, reason="V3 ellipse shape labels not built")


@needs_data
def test_dataset_provides_the_training_fields():
    ds = SkeletonDatasetV3("test", SOURCE_ROOT, V3_ROOT, geometry_points=1280,
                           ellipse_mask_res=32, mazes=["large"])
    item = ds[0]
    H, M, L = ds.horizon, ds.num_candidates, ds.candidate_points
    G = ds.geometry_points
    assert item["candidate_xy"].shape == (M, L, 2)
    assert item["candidate_geometry"].shape == (M, G, 2)
    assert item["ellipse_shape4_gt"].shape == (H, 4)
    assert item["shape_valid"].shape == (H,)
    assert item["shape_valid"].dtype == torch.bool
    assert item["ellipse_mask"].shape == (H, 32, 32)
    assert torch.isfinite(item["ellipse_shape4_gt"]).all()
    # The projection chain must not be a training dependency any more.
    assert "ellipse_center_gt" not in item
    assert "progress_gt" not in item


@needs_data
def test_maze_filter_keeps_only_large():
    ds = SkeletonDatasetV3("test", SOURCE_ROOT, V3_ROOT, geometry_points=1280,
                           ellipse_mask_res=16, mazes=["large"])
    assert len(ds) > 0
    for i in range(min(len(ds), 10)):
        assert int(ds[i]["maze_id"]) == MAZE_NAMES.index("large")


@needs_data
def test_align_target_is_the_gt_trajectory():
    ds = SkeletonDatasetV3("test", SOURCE_ROOT, V3_ROOT, geometry_points=1280,
                           ellipse_mask_res=16, mazes=["large"])
    item = ds[0]
    pos = item["pos"]
    assert pos.shape == (ds.horizon, 2)
    assert torch.isfinite(pos).all()
    # endpoints of the GT trajectory are the conditioning endpoints
    assert torch.allclose(pos[0], item["cond"][0], atol=1e-6)
    assert torch.allclose(pos[-1], item["cond"][1], atol=1e-6)


@needs_data
def test_lazy_mask_matches_the_shared_rasteriser_and_masks_invalid():
    ds = SkeletonDatasetV3("test", SOURCE_ROOT, V3_ROOT, geometry_points=1280,
                           ellipse_mask_res=32, mazes=["large"])
    item = ds[0]
    center = item["pos"]              # GT trajectory waypoint = mask centre
    shape4 = item["ellipse_shape4_gt"]
    valid = item["shape_valid"]
    a, b, theta = shape4_to_abtheta(shape4)
    expected = ellipse_soft_mask(center[None], a[None], b[None], theta[None],
                                 32, 10.0)[0]
    expected = expected * valid.float()[:, None, None]
    assert torch.allclose(item["ellipse_mask"], expected, atol=1e-6)
    assert float(item["ellipse_mask"][~valid].abs().sum()) == 0.0


@needs_data
def test_dense_geometry_endpoints_are_exact():
    ds = SkeletonDatasetV3("test", SOURCE_ROOT, V3_ROOT, geometry_points=1280,
                           ellipse_mask_res=32, mazes=["large"])
    checked = 0
    for idx in range(0, min(len(ds), 40)):
        item = ds[idx]
        cond = item["cond"].numpy()
        geom = item["candidate_geometry"].numpy()
        glen = item["candidate_geometry_lengths"].numpy()
        mask = item["candidate_mask"].numpy()
        for m in np.nonzero(mask)[0]:
            n = int(glen[m])
            assert n >= 2, (idx, m)
            assert np.allclose(geom[m, 0], cond[0], atol=1e-6), (idx, m)
            assert np.allclose(geom[m, n - 1], cond[1], atol=1e-6), (idx, m)
            checked += 1
    assert checked > 0


def test_trace_plot_draws_the_whole_trajectory(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import sample_v3

    H, res = 12, 64
    occ = np.zeros((res, res), dtype=np.float32)
    geom = np.zeros((2, 20, 2), dtype=np.float32)
    geom[0, :, 0] = np.linspace(-0.8, 0.8, 20)
    glen = np.array([20, 0])
    t = np.linspace(0.0, 1.0, H)[:, None]
    trace = [{
        "t": 5, "s": 0,
        "coarse": np.concatenate([t * 1.6 - 0.8, np.zeros((H, 1))], axis=1),
        "final": np.concatenate([t * 1.6 - 0.8, np.full((H, 1), 0.2)], axis=1),
        "p": np.zeros((H, 2), dtype=np.float32),
        "selected_idx": np.array(0),
        "pi": np.array([0.7, 0.3]),
        "ellipse_center": np.zeros((H, 2), dtype=np.float32),
    }]
    out_png = str(tmp_path / "trace.png")
    axes = sample_v3.plot_trace(occ, trace, geom, glen, out_png, "unit")
    assert os.path.exists(out_png) and os.path.getsize(out_png) > 0
    ax = axes[0][0]
    assert len(ax.lines[1].get_xdata()) == H
    assert len(ax.lines[2].get_xdata()) == H
    assert float(np.ptp(ax.lines[1].get_xdata())) > 10.0


def test_sampler_trace_is_per_step_and_sliceable():
    import src.diffusion.sampler_v3 as sampler_v3

    model = tiny_model(horizon=8)
    b = tiny_batch(B=3, M=2, H=8)
    out = sampler_v3.sample_v3(model, NoiseSchedule(16), b["cond"], b["occ"],
                               b["candidate_xy"], b["candidate_mask"],
                               b["candidate_geometry"],
                               b["candidate_geometry_lengths"],
                               device="cpu", steps=4, seed=0, return_trace=True)
    assert len(out["trace"]) == 4
    for k in range(3):
        for step in out["trace"]:
            assert step["coarse"][k].shape == (model.horizon, 2)
            assert step["final"][k].shape == (model.horizon, 2)
            assert step["selected_idx"][k].shape == ()
            assert step["pi"][k].shape == (2,)
            assert step["ellipse_center"][k].shape == (model.horizon, 2)
    assert torch.isfinite(out["p"]).all()
