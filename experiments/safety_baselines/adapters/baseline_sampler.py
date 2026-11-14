"""Plain DDIM for map-only x0 model; no skeleton/safety/GT inputs."""
import torch
from src.diffusion.sampler import pick_times
from src.geometry.bspline import BSplineCodec

@torch.no_grad()
def sample_baseline(model,schedule,occ,cond,seed=0,seed_offset=0,steps=16,corrector=None):
    model.eval();device=occ.device
    # Match existing Ours protocol exactly; fixed chunk32 is part of protocol.
    # This legacy seed+chunk-offset rule is NOT invariant to changing chunk size.
    torch.manual_seed(int(seed)+int(seed_offset))
    q=torch.randn(len(cond),model.num_controls,2,device=device,dtype=torch.float32)
    q=BSplineCodec.hard_control_endpoints(q,cond)
    times=pick_times(schedule.num_timesteps,steps)
    if times is None:pairs=[(t,t-1) for t in range(schedule.num_timesteps-1,-1,-1)]
    else:pairs=[(times[i],times[i-1]) for i in range(len(times)-1,0,-1)]+[(times[0],-1)]
    a=schedule.sqrt_alphas_cumprod.to(device).float();s=schedule.sqrt_one_minus_alphas_cumprod.to(device).float()
    memory=model.encode_scene(occ) if hasattr(model,'encode_scene') else None
    for t,nxt in pairs:
        tvec=torch.full((len(q),),t,device=device,dtype=torch.long)
        x0=(model.forward_from_memory(q,memory,cond,tvec) if memory is not None else model(q,occ,cond,tvec))['control']
        if corrector is not None:x0=corrector(x0,t)
        if not torch.isfinite(x0).all():raise RuntimeError('nonfinite_model_output')
        q=x0 if nxt<0 else a[nxt]*x0+s[nxt]*(q-a[t]*x0)/s[t]
        q=BSplineCodec.hard_control_endpoints(q,cond)
    if not torch.isfinite(q).all():raise RuntimeError('nonfinite_sample')
    return q
