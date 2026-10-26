import torch
from helpers import tiny_batch, tiny_model


def test_removed_features_keep_geometry_and_gradients():
    model = tiny_model(path_features={'enabled': False}, safety_query={'center_encoding': False}, feedback={'enabled': True})
    b = tiny_batch()
    out = model.forward_all(b['pos'], b['occ'], b['cond'], b['t'], b['ab'], b['candidate_xy'], b['candidate_mask'], b['candidate_geometry'], b['candidate_geometry_lengths'])
    assert model.path_feature_head is None and out['H_path'] is None
    assert model.ellipse_geometry.mlp_c is None
    assert model.fusion_mlp.num_streams == 1
    assert out['H_safety'].shape == (2, 8, 32)
    assert torch.allclose(out['control'][:, 0], b['cond'][:, 0], atol=1e-6)
    assert torch.allclose(out['control'][:, -1], b['cond'][:, 1], atol=1e-6)
    (out['control'].square().mean() + out['ellipse']['shape4'].square().mean()).backward()
    assert torch.isfinite(model.safety_query_head.net[1].weight.grad).all()
    assert torch.isfinite(model.fusion_mlp.net[1].weight.grad).all()
    assert not any(k.startswith(('path_feature_head.', 'ellipse_geometry.mlp_c.')) for k in model.state_dict())


def test_centers_only_affect_distance_bias():
    geom = tiny_model(safety_query={'center_encoding': False}).ellipse_geometry
    query, centers = torch.randn(2, 8, 32), torch.randn(2, 8, 2)
    out, _ = geom(query, centers, None, torch.randn(2, 32), None)
    assert torch.equal(out, query)
    assert not torch.equal(geom._geo_bias(centers, torch.ones(2)), geom._geo_bias(centers + 0.1, torch.ones(2)))
