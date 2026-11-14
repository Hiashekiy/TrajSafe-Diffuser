"""Direct point-space DDIM. The returned tensor is the trajectory itself."""
import numpy as np
import torch
from experiments.safety_baselines.models.map_only_points import pin_endpoints
from src.diffusion.sampler import pick_times

@torch.no_grad()
def sample_points(model,schedule,occupancy,cond,seed=0,steps=16,init=None):
    torch.manual_seed(seed)
    start=torch.randn(len(cond),model.num_points,2,device=cond.device) if init is None else torch.as_tensor(np.asarray(init,dtype=np.float32),device=cond.device).reshape(len(cond),model.num_points,2)
    points=pin_endpoints(start,cond)
    memory=model.encode_scene(occupancy)
    times=pick_times(schedule.num_timesteps,steps)
    pairs=[(t,t-1) for t in range(schedule.num_timesteps-1,-1,-1)] if times is None else [(times[i],times[i-1]) for i in range(len(times)-1,0,-1)]+[(times[0],-1)]
    for t,nxt in pairs:
        clean=model.forward_from_memory(points,memory,cond,torch.full((len(cond),),t,device=cond.device,dtype=torch.long))['trajectory']
        if not torch.isfinite(clean).all():raise RuntimeError('nonfinite point prediction')
        if nxt<0:points=clean
        else:
            a=schedule.sqrt_alphas_cumprod.to(points).view(-1);s=schedule.sqrt_one_minus_alphas_cumprod.to(points).view(-1)
            points=a[nxt]*clean+s[nxt]*(points-a[t]*clean)/s[t]
        points=pin_endpoints(points,cond)
    return points
