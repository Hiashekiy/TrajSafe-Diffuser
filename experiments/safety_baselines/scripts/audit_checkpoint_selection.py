"""Phase A3: is the training-time checkpoint selector aligned with real sampling?

The training script ranks epochs by `task_score = val_rmse + 80 * vertex_collision_rate`
computed on only `val_batches=12` batches (192 of 1550 val samples) and by the val
loss. This script recomputes both quantities for the saved checkpoints on

  * the 192-sample prefix the trainer actually used,
  * the full validation split,
  * several disjoint 192-sample slices,

and prints them next to the end-to-end DDIM sampling metrics, so the alignment
between "what the selector optimises" and "what the sampler delivers" is visible.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only_points import load_point_baseline
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.models.point_losses import point_x0_loss

@torch.no_grad()
def metrics(model,schedule,ds,index_list,batch=32):
    """Denoiser-side metrics at uniform random t (same definition the trainer uses)."""
    total=0.;rmse=0.;hitrate=0.;n=0
    g=torch.Generator(device='cuda').manual_seed(7)
    for start in range(0,len(index_list),batch):
        idx=index_list[start:start+batch]
        q=torch.stack([ds[i]['pos'] for i in idx]).cuda()
        occ=torch.stack([ds[i]['occupancy'] for i in idx]).cuda()
        cond=torch.stack([ds[i]['cond'] for i in idx]).cuda()
        t=torch.randint(0,schedule.num_timesteps,(len(q),),device='cuda',generator=g)
        a=schedule.sqrt_alphas_cumprod[t].float()[:,None,None];s=schedule.sqrt_one_minus_alphas_cumprod[t].float()[:,None,None]
        qt=a*q+s*torch.randn(q.shape,device='cuda',generator=g)
        p=model(qt,occ,cond,t)['trajectory']
        total+=float(point_x0_loss(p,q))*len(q)
        r=torch.sqrt(((p-q)**2).sum(-1).mean(-1))*80
        rmse+=float(r.sum());R=occ.shape[-1]
        ix=torch.floor((p[...,0]+1)*R/2).long();iy=torch.floor((p[...,1]+1)*R/2).long()
        inside=(ix>=0)&(iy>=0)&(ix<R)&(iy<R)
        hit=(occ[torch.arange(len(p),device='cuda')[:,None],0,iy.clamp(0,R-1),ix.clamp(0,R-1)]>.5)|~inside
        hitrate+=float(hit.float().mean())*len(q);n+=len(q)
    return {'x0_loss':total/n,'x0_rmse_m':rmse/n,'vertex_collision_rate':hitrate/n,
            'task_score':rmse/n+80*hitrate/n,'n':n}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',default='experiments/safety_baselines/results/map_only_points_t256_e100')
    ap.add_argument('--out',default='experiments/safety_baselines/results/phase_a_base_model/checkpoint_selection.json')
    a=ap.parse_args();run=ROOT/a.run
    ds=PointDataset('val',json.loads((run/'config_snapshot.json').read_text(encoding='utf8'))['data']['processed_root'])
    N=len(ds);prefix=list(range(min(192,N)));full=list(range(N))
    slices=[list(range(k*192,min((k+1)*192,N))) for k in range(min(6,N//192)) if len(range(k*192,min((k+1)*192,N)))==192]
    out={'val_size':N,'trainer_val_samples':len(prefix),'slices':len(slices),'checkpoints':{}}
    for tag,f in [('best_task','ckpt/best_task.pt'),('best','ckpt/best.pt'),('latest','ckpt/latest.pt')]:
        model,ck=load_point_baseline(run/f,'cuda')
        d=ck['config']['diffusion'];sched=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
        rec={'epoch':ck.get('epoch'),'prefix192':metrics(model,sched,ds,prefix),'full':metrics(model,sched,ds,full),
             'slices':[metrics(model,sched,ds,s)['task_score'] for s in slices]}
        out['checkpoints'][tag]=rec
        print(tag,'epoch',rec['epoch'],'prefix192 task %.2f'%rec['prefix192']['task_score'],
            'full task %.2f'%rec['full']['task_score'],'full rmse %.2f'%rec['full']['x0_rmse_m'],
            'slice task',[round(v,2) for v in rec['slices']],flush=True)
    Path(ROOT/a.out).write_text(json.dumps(out,indent=2),encoding='utf8');print('wrote',a.out)

if __name__=='__main__':main()
