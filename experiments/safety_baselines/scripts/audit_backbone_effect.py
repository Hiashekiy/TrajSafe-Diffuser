"""Diagnostic: why does "correct the clean prediction, return to DDIM" outperform
the paper Langevin/projection loops on this frozen base?

This is NOT one of the seven paper methods and NOT an Ours run. It isolates two
things the paper loops cannot separate:

  backbone   DDIM (deterministic, ends on the clean prediction, no residual noise)
             vs SGLD (annealed Langevin in the noisy state, ends on p_0)
  constraint the same two constraint objects the paper loops use:
             per-vertex free-cell rows, and the PDM App.C.3 whole-segment set

Only the map, the endpoints, the checkpoint and the shared evaluator are used.
The correction is applied to the denoiser's clean prediction, which is the
representation logic of the old engineering corrector (the user allowed reusing
that logic, not the paper-side algorithms).
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import pick_times
from experiments.safety_baselines.models.map_only_points import load_point_baseline,pin_endpoints
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_evaluator import evaluate_points
from experiments.safety_baselines.adapters.exact_map import ExactMapConstraint
from experiments.safety_baselines.adapters.paper_segment_constraints import CircleSegmentConstraints,obstacle_circles
from experiments.safety_baselines.algorithms.solvers import Budget,SolverFailure

def project_vertices(p,occ,cond,eps=1e-7):
    """Old-corrector logic in point space: push interior vertices out of occupied cells."""
    mp=ExactMapConstraint(occ,2.,0.);h,_=mp.query(p)
    if h[1:-1].min()>0:return p,0.
    m=h[1:-1]<=eps;q=p.copy();q[1:-1][m]+=(-h[1:-1][m]+eps)[:,None]*mp.query(p)[1][1:-1][m]
    return q,float(max(0.,-h[1:-1].min()))

@torch.no_grad()
def ddim_with_correction(model,schedule,occ_t,cond_t,occ,cond,seed,steps=16,corrector=None):
    torch.manual_seed(seed)
    points=pin_endpoints(torch.randn(1,model.num_points,2,device=cond_t.device),cond_t)
    memory=model.encode_scene(occ_t)
    times=pick_times(schedule.num_timesteps,steps)
    pairs=[(t,t-1) for t in range(schedule.num_timesteps-1,-1,-1)] if times is None else \
        [(times[i],times[i-1]) for i in range(len(times)-1,0,-1)]+[(times[0],-1)]
    p=points[0].cpu().numpy().astype(float);moved=[]
    for t,nxt in pairs:
        clean=model.forward_from_memory(points,memory,cond_t,torch.full((1,),t,device=cond_t.device,dtype=torch.long))['trajectory']
        if not torch.isfinite(clean).all():raise RuntimeError('nonfinite point prediction')
        c=pin_endpoints(clean,cond_t)
        if corrector is not None:
            cc,shift=corrector(c[0].cpu().numpy().astype(float),occ,cond)
            moved.append(shift)
            c=pin_endpoints(torch.as_tensor(cc,dtype=c.dtype,device=c.device)[None],cond_t)
        if nxt<0:points=c
        else:
            a=schedule.sqrt_alphas_cumprod.to(points).view(-1);s=schedule.sqrt_one_minus_alphas_cumprod.to(points).view(-1)
            points=a[nxt]*c+s[nxt]*(points-a[t]*c)/s[t]
        points=pin_endpoints(points,cond_t)
    return points[0].cpu().numpy().astype(float),moved

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments/safety_baselines/configs/seven_native_points_aligned_t16.json')
    ap.add_argument('--indices',default='0,81,163,244,326');ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--out',default='experiments/safety_baselines/docs/backbone_vs_constraint_audit.json')
    a=ap.parse_args();cfg=json.loads((ROOT/a.config).read_text(encoding='utf8'))
    model,ck=load_point_baseline(ROOT/cfg['checkpoint'],'cuda')
    d=ck['config']['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=PointDataset('val',ck['config']['data']['processed_root'])
    rows=[]
    for index in [int(v) for v in a.indices.split(',')]:
        row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();gt=row['pos'].numpy()
        occ_t=row['occupancy'][None].cuda();cond_t=row['cond'][None].cuda()
        centers,radii,audit=obstacle_circles(occ,2.)
        seg=CircleSegmentConstraints(cond,128,centers,radii)
        def seg_corrector(p,occ,cond):
            before=seg.residual(p[1:-1].reshape(-1))
            try:
                x,info=seg.project(p[1:-1].reshape(-1),Budget(120),maxiter=40)
                q=p.copy();q[1:-1]=x.reshape(-1,2);return q,max(0.,before)
            except SolverFailure:
                return p,max(0.,before)
        variants={'ddim_none':None,'ddim_vertex_correction':project_vertices,'ddim_segment_correction':seg_corrector}
        for label,corrector in variants.items():
            t0=time.perf_counter()
            try:
                p,moved=ddim_with_correction(model,schedule,occ_t,cond_t,occ,cond,a.seed,16,corrector)
                m=evaluate_points(p,occ,cond,gt)
                rows.append({'index':index,'variant':label,'vertex_collision':m['vertex_collision'],
                    'segment_only_collision':m['segment_only_collision'],'path_collision':m['collision'],
                    'planning_success':m['planning_success'],'rmse_m':m['rmse_m'],
                    'violation_max_m':m['violation_max_m'],'segment_residual_m':seg.residual(p[1:-1].reshape(-1))*80,
                    'seconds':time.perf_counter()-t0,'correction_steps':len(moved),
                    'mean_correction_m':float(np.mean(moved)*80) if moved else 0.,
                    'max_correction_m':float(np.max(moved)*80) if moved else 0.})
            except Exception as exc:
                rows.append({'index':index,'variant':label,'error':repr(exc)[:200],'seconds':time.perf_counter()-t0})
            print(index,label,rows[-1].get('vertex_collision'),rows[-1].get('segment_only_collision'),
                rows[-1].get('path_collision'),rows[-1].get('rmse_m'),flush=True)
    agg={}
    for lab in sorted({r['variant'] for r in rows}):
        rr=[r for r in rows if r['variant']==lab];ok=[r for r in rr if 'rmse_m' in r]
        agg[lab]={'n':len(rr),'returned':len(ok),
            'vertex_collisions':int(sum(r['vertex_collision'] for r in ok)),
            'segment_only_collisions':int(sum(r['segment_only_collision'] for r in ok)),
            'path_collisions':int(sum(r['path_collision'] for r in ok)),
            'planning_success':int(sum(r['planning_success'] for r in ok)),
            'rmse_m_mean':float(np.mean([r['rmse_m'] for r in ok])) if ok else None,
            'violation_m_max':float(np.max([r['violation_max_m'] for r in ok])) if ok else None,
            'segment_residual_m_max':float(np.max([r['segment_residual_m'] for r in ok])) if ok else None,
            'seconds_mean':float(np.mean([r['seconds'] for r in rr])),
            'mean_correction_m':float(np.mean([r['mean_correction_m'] for r in ok])) if ok else None}
    Path(ROOT/a.out).write_text(json.dumps({'purpose':'backbone vs constraint-object diagnostic; not a paper method, not Ours',
        'config':a.config,'indices':a.indices,'seed':a.seed,'rows':rows,'aggregate':agg},indent=2),encoding='utf8')
    print(json.dumps(agg,indent=2));print('wrote',a.out)

if __name__=='__main__':main()
