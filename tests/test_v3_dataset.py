"""V3 dataset / visualisation regression tests.

Covers two review findings:

    * the dense cell-chain geometry is stored as int16 CELL indices, so the two
      continuous endpoints (start / goal) come back snapped to cell centres.
      The dataset restores them exactly, otherwise L_align carries a constant
      error of up to half a cell at both ends.
    * the per-step trace plot used to index the batch dimension twice and drew a
      single waypoint instead of the whole trajectory.
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

from src.datasets.skeleton_dataset_v3 import SkeletonDatasetV3  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402

from v3_utils import tiny_batch, tiny_model  # noqa: E402

V3_ROOT = os.path.join(REPO_ROOT, "data", "processed_scene_v3")
SOURCE_ROOT = os.path.join(REPO_ROOT, "data", "processed_scene_v1")
HAS_DATA = os.path.exists(os.path.join(V3_ROOT, "test", "candidate_geometry.npy"))

needs_data = pytest.mark.skipif(not HAS_DATA, reason="V3 preprocessing not built")


@needs_data
def test_dense_geometry_endpoints_are_exact():
    ds = SkeletonDatasetV3("test", SOURCE_ROOT, V3_ROOT, geometry_points=1280)
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
            # the inner points are cell centres and stay on the half-integer grid
            inner = geom[m, 1:n - 1]
            if len(inner):
                cell = 2.0 / 256.0
                px = (inner + 1.0) / cell - 0.5
                assert np.abs(px - np.rint(px)).max() < 1e-4, (idx, m)
            checked += 1
    assert checked > 0


@needs_data
def test_dense_geometry_padding_is_zero_and_lengths_agree():
    ds = SkeletonDatasetV3("test", SOURCE_ROOT, V3_ROOT, geometry_points=1280)
    item = ds[0]
    glen = item["candidate_geometry_lengths"]
    mask = item["candidate_mask"]
    geom = item["candidate_geometry"]
    assert bool((glen[~mask] == 0).all())
    for m in np.nonzero(mask.numpy())[0]:
        n = int(glen[m])
        assert float(geom[m, n:].abs().sum()) == 0.0
    assert bool((glen <= ds.geometry_points).all())


def test_trace_plot_draws_the_whole_trajectory(tmp_path):
    """plot_trace must plot all H waypoints of the per-sample [H, 2] arrays."""
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
    # line 0 = skeleton chain, 1 = coarse, 2 = final
    assert len(ax.lines[1].get_xdata()) == H
    assert len(ax.lines[2].get_xdata()) == H
    assert len(ax.lines[1].get_ydata()) == H
    assert float(np.ptp(ax.lines[1].get_xdata())) > 10.0


def test_sampler_trace_is_per_step_and_sliceable():
    """The trace entry point used by sample_v3: [B, H, 2] -> slice -> [H, 2]."""
    import src.diffusion.sampler_v3 as sampler_v3

    model = tiny_model()
    b = tiny_batch(B=3, M=2)
    out = sampler_v3.sample_v3(model, NoiseSchedule(16), b["cond"], b["occ"],
                               b["features"], b["mask"], b["lengths"], b["geom"],
                               b["geom_len"], device="cpu", steps=4, seed=0,
                               return_trace=True)
    assert len(out["trace"]) == 4
    for k in range(3):
        frame = {name: step[name][k].numpy() for name, step in
                 ((n, s) for s in out["trace"] for n in
                  ("coarse", "final", "selected_idx", "pi", "ellipse_center"))}
        assert frame["coarse"].shape == (model.horizon, 2)
        assert frame["final"].shape == (model.horizon, 2)
        assert frame["selected_idx"].shape == ()
        assert frame["pi"].shape == (2,)
        assert frame["ellipse_center"].shape == (model.horizon, 2)
    assert torch.isfinite(out["p"]).all()
