"""Sampler/base compatibility gate: does each loop recover the learned prior?

Unconstrained diagnostics only: the same val case, the same NumPy seed, no
constraints and no projection, so any deviation is attributable to the sampler
formula (noise schedule, score units, VP/VE mapping, step size, budget) and not
to the safety machinery. DDIM is the reference the checkpoint was trained with.

Each row also reports the *structural* residual noise floor of its last level:
a Langevin loop targets p_t = N(sqrt(alphabar_t) x0, 1-alphabar_t), so its
delivered trajectory carries at least sigma_min of per-coordinate noise. That
number is a property of the frozen noise schedule, not of an implementation bug,
and separates "the loop does not converge" from "the base has no fine level".
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only_points import load_point_baseline
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_sampler import sample_points
from experiments.safety_baselines.adapters.point_evaluator import evaluate_points

class Unconstrained:
    """Task view with no constraints: the projection operator is the identity."""
    def __init__(self,model,schedule,occ,cond):
        from experiments.safety_baselines.adapters.native_points import NativePointTask
        self.base=NativePointTask(model,schedule,occ,cond)
        self.H=self.base.H;self.D=self.base.D;self.ab=self.base.ab;self.betas=self.base.betas
        self.sigma=self.base.sigma;self.endpoints=self.base.endpoints
    def unpack(self,x):return self.base.unpack(x)
    def points(self,x):return self.base.points(x)
    def initial(self,rng,particles=1,ve=False):return rng.normal(size=(particles,self.D))*(self.sigma[-1] if ve else 1.)
    def clean(self,x,t,ve=False):return self.base.clean(x,t,ve)
    def score(self,x,t,ve=False):return self.base.score(x,t,ve)
    def constraints(self,x):
        import scipy.sparse as sp
        return np.zeros(1),sp.csr_matrix((1,self.D))
    def project(self,target,budget):
        return np.asarray(target,dtype=float).copy(),{'status':'unconstrained_identity'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='experiments/safety_baselines/configs/seven_native_points_aligned_t16.json')
    ap.add_argument('--indices',default='0,81,163,244,326');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--out',default='experiments/safety_baselines/docs/sampler_compatibility_gate.json')
    a=ap.parse_args();cfg=json.loads((ROOT/a.config).read_text(encoding='utf8'))
    ckpt=ROOT/cfg['checkpoint'];model,ck=load_point_baseline(ckpt,'cuda')
    d=ck['config']['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=PointDataset('val',ck['config']['data']['processed_root'])
    ab=schedule.alphas_cumprod.cpu().numpy();sigma=np.sqrt((1-ab)/ab)
    floor_m=float(sigma[0]*80*np.sqrt(2))
    from experiments.safety_baselines.algorithms.pdm.runner import run as pdm_run
    from experiments.safety_baselines.algorithms.constrained_diffusers.runner import run as cd_run
    from experiments.safety_baselines.algorithms.solvers import Budget
    rows=[]
    for index in [int(v) for v in a.indices.split(',')]:
        row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();gt=row['pos'].numpy()
        torch.cuda.synchronize();t0=time.perf_counter()
        q=sample_points(model,schedule,row['occupancy'][None].cuda(),row['cond'][None].cuda(),seed=a.seed)
        torch.cuda.synchronize();ddim=q[0].cpu().numpy();secs=time.perf_counter()-t0
        m=evaluate_points(ddim,occ,cond,gt)
        rows.append({'index':index,'diagnostic':'ddim_nominal','rmse_m':m['rmse_m'],'vertex_collision':m['vertex_collision'],
            'path_collision':m['collision'],'seconds':secs,'path_length_ratio':m['path_length_ratio'],
            'mean_turning_deg':m['mean_turning_deg']})
        task=Unconstrained(model,schedule,occ,cond)
        for label,cfg_over in [('cd_langevin_vp',{'substeps':1}),('cd_langevin_vp_sub4',{'substeps':4})]:
            c=dict(cfg,cd_substeps=cfg_over['substeps'],cd_levels=64,cd_extra_steps=64,particles=1)
            trace=[];t0=time.perf_counter()
            try:
                x=cd_run(task,'Projected',np.random.default_rng(a.seed),c,Budget(300),trace)
                p=task.unpack(x);m=evaluate_points(p,occ,cond,gt)
                rows.append({'index':index,'diagnostic':label,'rmse_m':m['rmse_m'],'vertex_collision':m['vertex_collision'],
                    'path_collision':m['collision'],'iterations':len(trace),'seconds':time.perf_counter()-t0,
                    'path_length_ratio':m['path_length_ratio'],'mean_turning_deg':m['mean_turning_deg']})
            except Exception as exc:
                rows.append({'index':index,'diagnostic':label,'error':repr(exc)[:200],'seconds':time.perf_counter()-t0})
        for label,over in [('pdm_paper_ve',{'pdm_step_convention':'paper','pdm_parameterization':'ve'}),
                           ('pdm_official_ald_ve',{'pdm_step_convention':'official_ald','pdm_parameterization':'ve'}),
                           ('pdm_official_ald_vp',{'pdm_step_convention':'official_ald','pdm_parameterization':'vp'})]:
            c=dict(cfg,pdm_inner=20,**over);trace=[];t0=time.perf_counter()
            try:
                x=pdm_run(task,'PDM',np.random.default_rng(a.seed),c,Budget(300),trace)
                p=task.unpack(x);m=evaluate_points(p,occ,cond,gt)
                rows.append({'index':index,'diagnostic':label,'rmse_m':m['rmse_m'],'vertex_collision':m['vertex_collision'],
                    'path_collision':m['collision'],'iterations':len(trace),'seconds':time.perf_counter()-t0,
                    'first_step_mixing_ratio':trace[0]['mixing_ratio'],'last_step_mixing_ratio':trace[-1]['mixing_ratio'],
                    'path_length_ratio':m['path_length_ratio'],'mean_turning_deg':m['mean_turning_deg']})
            except Exception as exc:
                rows.append({'index':index,'diagnostic':label,'error':repr(exc)[:200],'seconds':time.perf_counter()-t0})
        print(index,'done',flush=True)
    labels=sorted({r['diagnostic'] for r in rows})
    agg={lab:{'n':0,'rmse_mean_m':None,'vertex_collisions':0,'path_collisions':0,'errors':0,'seconds_mean':None}
        for lab in labels}
    for lab in labels:
        rr=[r for r in rows if r['diagnostic']==lab];ok=[r for r in rr if 'rmse_m' in r]
        agg[lab]={'n':len(rr),'rmse_mean_m':float(np.mean([r['rmse_m'] for r in ok])) if ok else None,
            'rmse_max_m':float(np.max([r['rmse_m'] for r in ok])) if ok else None,
            'vertex_collisions':int(sum(r['vertex_collision'] for r in ok)),'path_collisions':int(sum(r['path_collision'] for r in ok)),
            'errors':len(rr)-len(ok),'seconds_mean':float(np.mean([r['seconds'] for r in rr]))}
    out={'purpose':'unconstrained sampler/base compatibility gate; not a method comparison',
        'config':a.config,'seed':a.seed,'schedule':{'timesteps':int(d['timesteps']),'beta_schedule':d['beta_schedule'],
        'alphabar_min':float(ab.min()),'sigma_min_scene':float(sigma[0]),'sigma_max_scene':float(sigma[-1]),
        'sigma_min_m':float(sigma[0]*80),'langevin_lowest_level_floor_rmse_m':floor_m},
        'rows':rows,'aggregate':agg}
    Path(ROOT/a.out).write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(agg,indent=2))
    print('floor_rmse_m_from_lowest_trained_level',round(floor_m,3));print('wrote',a.out)

if __name__=='__main__':main()
