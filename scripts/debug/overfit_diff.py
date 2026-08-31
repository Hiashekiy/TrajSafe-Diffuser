"""Overfit a small set with ONLY the trajectory reconstruction loss to isolate
whether the diffusion backbone can generate non-collapsed trajectories."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.geometry.d4rl_geometry import build_occupancy_grid
from src.datasets.normalization import load_normalization
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.conditioning import apply_endpoint_condition
from src.models.planner import Planner
from src.losses.trajectory_loss import l_diff

cfg=load_config('configs/config.yaml'); set_seed(int(cfg['env']['seed']))
device='cuda'
ext=tuple(cfg['geometry']['extent']); gres=float(cfg['geometry']['global_res'])
occ,sdf,_=build_occupancy_grid(cfg['data']['maze'],extent=ext,global_res=gres,inflate_particle=True)
map_t=torch.as_tensor(occ,dtype=torch.float32).cuda()[None,None]
proc=cfg['data']['processed_dir']; norm,_=load_normalization(os.path.join(proc,'normalization.json'))
# take 32 training windows
tr=np.load(os.path.join(proc,'train','trajectories.npy'))[:32]
co=np.load(os.path.join(proc,'train','conditions.npy'))[:32]
traj_t=torch.as_tensor(tr,dtype=torch.float32).cuda()
cond_t=torch.as_tensor(co,dtype=torch.float32).cuda()
schedule=NoiseSchedule(cfg['diffusion']['timesteps'],beta_schedule=cfg['diffusion']['beta_schedule'],beta_start=cfg['diffusion']['beta_start'],beta_end=cfg['diffusion']['beta_end']).to(device)
model=Planner(cfg['model'],cfg['geometry'],norm).cuda()
opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
mins=np.asarray(norm.mins[2:4]); maxs=np.asarray(norm.maxs[2:4]); eps=norm.eps
def unnorm(p): return (p+1.0)/2.0*(maxs-mins+eps)+mins
model.train()
for step in range(120):
    t=torch.randint(0,schedule.num_timesteps,(32,),device=device)
    x_t=schedule.q_sample(traj_t.double(),t).float()
    x_t=apply_endpoint_condition(x_t, cond_t, 128, obs_start=2, obs_end=4)
    out=model(x_t,t,map_t.expand(32,-1,-1,-1),cond=cond_t)
    x0p=apply_endpoint_condition(out['x0_pred'], cond_t, 128, obs_start=2, obs_end=4)
    loss=l_diff(x0p, traj_t, lambda_var=1.0, lambda_var_vel=2.0)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if (step+1)%20==0:
        with torch.no_grad():
            pred=model(x_t,t,map_t.expand(32,-1,-1,-1),cond=cond_t)
            pw=unnorm(pred['x0_pred'].cpu().numpy()[:, :, 2:4])
            gt=unnorm(traj_t.cpu().numpy()[:, :, 2:4])
            print(f'step {step+1:3d} L_diff={loss.item():.4f}  pred_x_std={pw[:,:,0].std():.3f} pred_y_std={pw[:,:,1].std():.3f}  gt_x_std={gt[:,:,0].std():.3f} gt_y_std={gt[:,:,1].std():.3f}')
print('OVERFIT DONE')
