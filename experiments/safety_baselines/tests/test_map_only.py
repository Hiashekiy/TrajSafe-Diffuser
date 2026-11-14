import torch
from src.utils.config import load_config
from src.models.trajsafe import TrajSafePlanner
from experiments.safety_baselines.models.map_only import MapOnlyDiffuser,load_baseline

def test_architecture_and_common_shapes():
    cfg=load_config('experiments/safety_baselines/configs/map_only_l1_final8.yaml')
    m=MapOnlyDiffuser(cfg)
    assert len(m.traj_backbone)==1 and len(m.final_denoiser.blocks)==8
    forbidden=('skeleton','topology','ellipse','safety','feedback','match_block','path_feature')
    assert not any(any(s in name for s in forbidden) for name in m.state_dict())
    ours=TrajSafePlanner(load_config('experiments/safety_baselines/configs/v4_map16_l1_final8_e100.yaml')['model'],cfg.get('ellipse_label'),cfg['bspline'])
    for name,v in m.state_dict().items():
        if name.startswith('fusion_mlp.'):continue
        assert name in ours.state_dict() and ours.state_dict()[name].shape==v.shape

def test_forward_gradient_endpoints_and_strict_reload(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(42)
    cfg=load_config('experiments/safety_baselines/configs/map_only_l1_final8.yaml');m=MapOnlyDiffuser(cfg).eval()
    q=torch.randn(1,48,2);occ=torch.zeros(1,1,256,256);cond=torch.tensor([[[-.7,-.6],[.7,.6]]]);t=torch.tensor([8])
    out=m(q,occ,cond,t)
    torch.testing.assert_close(out['control'][:,0],cond[:,0],atol=1e-6,rtol=0)
    torch.testing.assert_close(out['control'][:,-1],cond[:,1],atol=1e-6,rtol=0)
    out['q_raw_final'].square().mean().backward()
    grads=[p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    path=tmp_path/'baseline.pt';torch.save({'architecture':m.architecture,'model_state':m.state_dict(),'config':cfg},path)
    restored,_=load_baseline(path)
    torch.testing.assert_close(out['control'],restored(q,occ,cond,t)['control'],atol=0,rtol=0)


def test_scene_feature_cache_contract():
 torch.set_num_threads(1);torch.manual_seed(7)
 cfg=load_config('experiments/safety_baselines/configs/map_only_l1_final8.yaml');m=MapOnlyDiffuser(cfg).eval()
 q=torch.randn(1,48,2);occ=torch.zeros(1,1,256,256);cond=torch.tensor([[[-.8,0],[.8,0]]]);t=torch.tensor([7])
 with torch.no_grad():
  a=m(q,occ,cond,t);b=m.forward_from_memory(q,m.encode_scene(occ),cond,t)
 torch.testing.assert_close(a['control'],b['control'],atol=0,rtol=0)
