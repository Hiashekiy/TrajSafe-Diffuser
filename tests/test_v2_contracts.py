"""V2 contract tests: the specification's hard prohibitions.

These are grep-level guards (docs/V2.md section 21 and 39) so that the V1 joint
P/E formulation cannot silently come back into the V2 code path.
"""

from __future__ import annotations

import inspect
import os
import re
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.skeleton_dataset import SkeletonDataset  # noqa: E402
from src.geometry.skeleton_paths import CandidateConfig  # noqa: E402

V2_FILES = [
    "src/models/skeleton/skeleton_planner.py",
    "src/models/skeleton/topology_selector.py",
    "src/models/skeleton/progress_head.py",
    "src/models/skeleton/ellipse_shape_head.py",
    "src/models/skeleton/path_encoder.py",
    "src/models/skeleton/traj_blocks.py",
    "src/models/skeleton/path_ops.py",
    "src/diffusion/sampler_v2.py",
    "src/losses/v2_losses.py",
    "src/datasets/skeleton_dataset.py",
    "src/geometry/fixed_center_iris.py",
    "src/geometry/skeleton_paths.py",
    "train_v2.py",
    "sample_v2.py",
    "evaluate_v2.py",
]

FORBIDDEN = [
    r"p_pred\s*\+\s*e_pred",          # V1 centre = trajectory + offset
    r"p\s*\+\s*delta_c",              # V1 delta-centre parameterisation
    r"e_pred\[\.\.\.,\s*:2\]",
    r"\bdelta_c\b",
    r"x0_e_center_safe",
    r"ellipse_center_safety_loss",
    r"lambda_center_safe",
]


def _source(path):
    with open(os.path.join(REPO_ROOT, path), "r", encoding="utf-8") as f:
        return f.read()


def _code_only(src):
    """Drop docstrings and comments so prose cannot trip the pattern guards."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    return re.sub(r"#[^\n]*", "", src)


def test_no_legacy_center_formula_in_v2():
    """Only c_i = gamma_m(s_i) may define an ellipse centre in V2."""
    for path in V2_FILES:
        src = _code_only(_source(path))
        for pattern in FORBIDDEN:
            assert re.search(pattern, src) is None, (
                "%s still contains %r" % (path, pattern))


def test_v2_config_has_no_center_safe_loss():
    with open(os.path.join(REPO_ROOT, "configs", "config_v2_skeleton.yaml"),
              "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "lambda_center_safe" not in cfg["loss"]
    assert cfg["loss"]["lambda_topology"] == 1.0
    assert cfg["loss"]["lambda_progress"] == 0.5
    assert cfg["loss"]["lambda_shape"] == 1.0
    assert cfg["topology"]["commit_t"] == 7
    assert cfg["topology"]["num_candidates"] == 4
    assert cfg["topology"]["detach_trajectory_feature"] is True
    assert cfg["skeleton"]["safety_dilation_cells"] == 1


def test_candidate_config_covers_the_topology_section():
    with open(os.path.join(REPO_ROOT, "configs", "config_v2_skeleton.yaml"),
              "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cc = CandidateConfig.from_dict(cfg["topology"], strict=False)
    assert cc.num_candidates == 4 and cc.candidate_points == 128
    assert cc.max_length_ratio == 1.5 and cc.dedup_jaccard == 0.75
    assert cc.raw_k == 16 and cc.anchor_candidates == 16


def test_sampler_v2_does_not_import_the_v1_state():
    src = _source("src/diffusion/sampler_v2.py")
    assert "sampler_v1" not in src
    assert "import torch" in src


def test_planner_forward_is_split_into_three_entry_points():
    from src.models.skeleton import SkeletonPlanner
    for name in ("encode_trajectory", "score_candidates", "refine_with_path"):
        assert callable(getattr(SkeletonPlanner, name, None))
    src = inspect.getsource(SkeletonPlanner)
    # no giant forward with optional topology / ellipse flags
    assert "def forward(" not in src


def test_trajectory_backbone_has_no_ellipse_input():
    from src.models.skeleton import SkeletonPlanner
    sig = inspect.signature(SkeletonPlanner.encode_trajectory)
    assert list(sig.parameters) == ["self", "p_t", "occ", "cond", "t", "ab"]


def test_dataset_returns_the_v2_contract(tmp_path):
    from v2_utils import MAPS_DIR, SKELETON_DIR
    base = os.path.join(REPO_ROOT, "data", "processed_scene_v2")
    source = os.path.join(REPO_ROOT, "data", "processed_scene_v1")
    if not os.path.exists(os.path.join(base, "val", "candidate_paths.npy")):
        pytest.skip("run scripts/data/11_build_skeleton_candidates.py first")
    ds = SkeletonDataset("val", source, base, shape_dir=SKELETON_DIR)
    item = ds[0]
    # map_tensor is added by the collate function (it comes from maze_id)
    expected = {
        "pos", "cond", "maze_id", "candidate_paths",
        "candidate_mask", "candidate_lengths", "topology_target",
        "topology_best", "progress_gt", "ellipse_shape4_gt", "shape_valid",
        "ellipse_center_gt", "ellipse_mask", "has_candidate",
    }
    assert expected.issubset(set(item))
    assert "e6" not in item and "sdf_tensor" not in item
    K = item["pos"].shape[0]
    assert tuple(item["candidate_paths"].shape) == (4, 128, 5)
    assert tuple(item["candidate_mask"].shape) == (4,)
    assert item["progress_gt"].shape == (K,)
    assert item["ellipse_shape4_gt"].shape == (K, 4)
    assert item["ellipse_mask"].shape == (K, 64, 64)
    assert item["ellipse_mask"].dtype == np.uint8 or item["ellipse_mask"].dtype == __import__("torch").uint8
    assert (item["progress_gt"][0] == 0.0) and abs(float(item["progress_gt"][-1]) - 1.0) < 1e-5
    assert bool((item["progress_gt"][1:] >= item["progress_gt"][:-1] - 1e-6).all())


def test_progress_gt_lies_on_the_gt_best_candidate():
    """The dataset centre must be gamma_m*(s_gt) on the teacher-forced path."""
    from v2_utils import SKELETON_DIR
    from src.geometry.skeleton_paths import nearest_arclength
    base = os.path.join(REPO_ROOT, "data", "processed_scene_v2")
    source = os.path.join(REPO_ROOT, "data", "processed_scene_v1")
    if not os.path.exists(os.path.join(base, "val", "candidate_paths.npy")):
        pytest.skip("run scripts/data/11_build_skeleton_candidates.py first")
    ds = SkeletonDataset("val", source, base, shape_dir=SKELETON_DIR)
    for idx in range(5):
        item = ds[idx]
        if not bool(item["has_candidate"]):
            continue
        best = int(item["topology_best"])
        poly = item["candidate_paths"][best, :, :2].numpy()
        center = item["ellipse_center_gt"].numpy()
        _, dist = nearest_arclength(center, poly)
        assert np.all(dist < 1e-5)
        assert np.all(item["shape_valid"].numpy())


def test_skeleton_graph_has_no_sdf_dependency():
    for path in ("src/geometry/skeleton_graph.py", "src/geometry/skeleton_paths.py"):
        assert "sdf" not in _source(path).lower()
