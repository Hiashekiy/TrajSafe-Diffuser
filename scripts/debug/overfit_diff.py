"""Overfit a small set with ONLY the residual reconstruction loss to isolate
whether the zero-sum bridge backbone can fit data without collapsing."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.zerosum import compute_z0
from src.models.planner import Planner
from src.losses.total_loss import total_loss
from src.datasets.scene_dataset import make_loader

cfg = load_config('configs/config.yaml'); set_seed(int(cfg['env']['seed']))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ld = make_loader('data/processed_scene/train', 8, False, 0)
batch = next(iter(ld))
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
model = Planner(cfg['model'], cfg['geometry'], None).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = NoiseSchedule(cfg['diffusion']['timesteps']).to(device)
T = sched.num_timesteps
model.train()
for step in range(30):
    t = torch.randint(0, T, (batch['pos'].shape[0],), device=device)
    z0, _, _, _ = compute_z0(batch['pos'], batch['cond'])
    zt = sched.q_sample_zero_sum(z0, t)
    out = model(zt, t, batch['map_tensor'], batch['cond'])
    loss = total_loss(out, batch, cfg['loss'], device=device, phase='traj', epoch=0,
                      ellipse_cfg=cfg.get('ellipse_loss', {}))
    opt.zero_grad(); loss['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if (step + 1) % 5 == 0:
        print(f"step {step+1:3d} L_Z={loss['L_Z'].item():.4f} total={loss['total'].item():.4f}")
print('OVERFIT DONE')
