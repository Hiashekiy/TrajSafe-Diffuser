"""Reuse the existing engineering SDF correctors on the map-only checkpoint.

These modes are labelled legacy SDF guidance, never the seven paper algorithms.
No candidate path, corridor, synthetic trajectory state or refitting is used.
"""
import torch
from src.geometry.sdf_field import build_occupancy_sdf
from src.diffusion.sdf_guidance import apply_sdf_guidance
from experiments.safety_baselines.adapters.baseline_sampler import sample_baseline

@torch.no_grad()
def sample_legacy_sdf(model,schedule,occ,cond,mode,seed=0,steps=16,config=None):
    if mode not in ('none','penalty','projection','primal_dual','alm_sdf'):
        raise ValueError('unknown legacy SDF mode')
    if mode=='none':
        return sample_baseline(model,schedule,occ,cond,seed,0,steps),[]
    cfg=dict(config or {})
    sdf=build_occupancy_sdf(occ,scene_to_meter=80.,map_margin=0.)
    active=torch.ones(len(cond),dtype=torch.bool,device=occ.device)
    lam=None;trace=[];count=0
    def correct(q,t):
        nonlocal lam,count
        count+=1
        if count<=int(cfg.get('warmup_reverse_steps',0)):return q
        out,lam,stats=apply_sdf_guidance(mode,q,sdf,model.bspline,cond,lam,cfg,active)
        trace.append({'t':int(t),'status':list(stats['status']),
            'violation_before':stats['violation_before'].cpu().tolist(),
            'violation_after':stats['violation_after'].cpu().tolist()})
        return model.bspline.hard_control_endpoints(out,cond)
    q=sample_baseline(model,schedule,occ,cond,seed,0,steps,corrector=correct)
    return q,trace
