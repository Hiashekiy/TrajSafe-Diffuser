import numpy as np
import pytest
import torch

from src.diffusion.sampler_v5 import _build_safe_grids, _search_once
from src.grid_search.field_search import guidance_class


def scene(spec, rc):
    return spec.cell_to_world(np.asarray(rc, dtype=float)[..., ::-1])


def setup_map():
    occ = torch.zeros(1, 1, 48, 48)
    occ[0, 0, 8:36, 24] = 1
    cfg = {"grid_search": {"scene_to_meter": 24, "robot_radius_m": 1.,
                           "safety_margin_m": 0.5}}
    grids, spec, _ = _build_safe_grids(occ, cfg)
    return occ, cfg, grids, spec


def test_inflated_fields_and_original_network_map():
    occ, cfg, grids, spec = setup_map()
    original = occ.clone()
    safe = grids[0][0]
    assert not safe.free[20, 23]  # free in original, blocked by footprint
    assert not safe.free[0, 10]  # outside map also blocks the footprint
    assert occ[0, 0, 20, 23] == 0
    gf = guidance_class()((~safe.free).astype(np.uint8), weight="quadratic", lam=.01)
    gf.build(np.array([[4., 5.], [40., 42.]]))
    assert np.isnan(gf.field[20, 23])
    assert gf.clearance().distance()[20, 23] == 0
    np.testing.assert_array_equal(occ.numpy(), original.numpy())


def test_search_routes_around_wall_with_scene_coordinate_conversion():
    occ, cfg, grids, spec = setup_map()
    endpoints = scene(spec, [[20, 6], [20, 41]])
    cond = torch.tensor(endpoints[None], dtype=torch.float32)
    guide = torch.tensor(np.linspace(*endpoints, 128)[None], dtype=torch.float32)
    path, stats = _search_once(grids, spec, cond, guide, cfg, 0)
    assert stats["status"] == "SUCCESS"
    assert stats["backend"] == "field_guided"
    assert stats["guide_valid"]
    assert path.shape == (128, 2)
    np.testing.assert_allclose(path[[0, -1]], endpoints, atol=1e-6)
    rc = np.rint(spec.world_to_cell(path)[:, ::-1]).astype(int)
    assert grids[0][0].free[rc[:, 0], rc[:, 1]].all()
    assert np.abs(rc[:, 0] - 20).max() > 12
    assert occ[0, 0, 20, 23] == 0


def test_blocked_endpoint_is_not_silently_replaced_by_straight_line():
    _, cfg, grids, spec = setup_map()
    endpoints = scene(spec, [[20, 23], [20, 41]])
    with pytest.raises(RuntimeError, match="start blocked"):
        _search_once(grids, spec, torch.tensor(endpoints[None]),
                     torch.tensor(np.linspace(*endpoints, 128)[None]), cfg, 0)


def test_unreachable_map_reports_failure():
    _, cfg, grids, spec = setup_map()
    grids[0][0].free[:, 24] = False
    endpoints = scene(spec, [[20, 6], [20, 41]])
    with pytest.raises(RuntimeError, match="unreachable"):
        _search_once(grids, spec, torch.tensor(endpoints[None]),
                     torch.tensor(np.linspace(*endpoints, 128)[None]), cfg, 0)
