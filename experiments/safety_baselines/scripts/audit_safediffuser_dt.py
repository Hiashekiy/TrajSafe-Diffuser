"""Audit the SafeDiffuser time-scale (delta tau) treatment.

Paper: QP (13) is solved ONCE per diffusion step j and the resulting control
nu^j* is applied over the whole controllable-form interval (eq. 5-6), where
"the diffusion time step length Delta tau is chosen to be small enough", and the
barrier condition is  db/dtau . nu + eps (b - gamma) >= 0  -- i.e. it is a
condition on the interval [j, j+1].

This audit compares the current implementation (cbf_substeps sub-intervals per
diffusion step, QP re-solved on each sub-interval, barrier condition still
written for the *whole* interval) against substeps=1 (one QP per diffusion step,
as Algorithm 1 prescribes) on the same cases and the same base.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.safety_baselines.models.map_only_points import load_point_baseline
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_evaluator import evaluate_points
from experiments.safety_baselines.algorithms.runner import run_method

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments/safety_baselines/configs/seven_native_points_aligned_t256.json')
    ap.add_argument('--indices',default='0,81,163,244,326')
    ap.add_argument('--methods',default='RoS-diffuser,ReS-diffuser,TVS-diffuser')
    ap.add_argument('--substeps',default='1,16')
    ap.add_argument('--out',default='experiments/safety_baselines/docs/safediffuser_dt_audit.json')
    a=ap.parse_args();cfg=json.loads((ROOT/a.config).read_text(encoding='utf8'))
    import torch
    from src.diffusion.schedule import NoiseSchedule
    model,ck=load_point_baseline(ROOT/cfg['checkpoint'],'cuda')
    d=ck['config']['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=PointDataset('val',ck['config']['data']['processed_root'])
    rows=[]
    for index in [int(v) for v in a.indices.split(',')]:
        row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();gt=row['pos'].numpy()
        for method in a.methods.split(','):
            for sub in [int(v) for v in a.substeps.split(',')]:
                c=dict(cfg,cbf_substeps=sub);t0=time.perf_counter()
                r=run_method(method,model,schedule,occ,cond,index,0,c)
                m=evaluate_points(r['trajectory'],occ,cond,gt) if r['trajectory'] is not None else None
                rows.append({'index':index,'method':method,'cbf_substeps':sub,'status':r['status'],
                    'failure_reason':r['failure_reason'],'seconds':time.perf_counter()-t0,
                    'len_ratio':None if m is None else m['path_length_ratio'],
                    'turning_deg':None if m is None else m['mean_turning_deg'],
                    'vertex_collision':None if m is None else m['vertex_collision'],
                    'segment_only_collision':None if m is None else m['segment_only_collision'],
                    'path_collision':None if m is None else m['collision'],
                    'planning_success':None if m is None else m['planning_success'],
                    'rmse_m':None if m is None else m['rmse_m'],
                    'violation_max_m':None if m is None else m['violation_max_m'],
                    'integration_steps_mean':float(np.mean([t['integration_steps'] for t in r['per_step_trace']])) if r['per_step_trace'] else None})
                print(index,method,'substeps',sub,r['status'],rows[-1]['len_ratio'],rows[-1]['violation_max_m'],flush=True)
    agg={}
    for method in a.methods.split(','):
        for sub in [int(v) for v in a.substeps.split(',')]:
            rr=[r for r in rows if r['method']==method and r['cbf_substeps']==sub]
            ok=[r for r in rr if r['len_ratio'] is not None]
            agg['%s|substeps=%d'%(method,sub)]={'n':len(rr),'returned':len(ok),
                'len_ratio_mean':float(np.mean([r['len_ratio'] for r in ok])) if ok else None,
                'turning_deg_mean':float(np.mean([r['turning_deg'] for r in ok])) if ok else None,
                'vertex_collisions':int(sum(r['vertex_collision'] for r in ok)),
                'segment_only_collisions':int(sum(r['segment_only_collision'] for r in ok)),
                'path_collisions':int(sum(r['path_collision'] for r in ok)),
                'planning_success':int(sum(r['planning_success'] for r in ok)),
                'rmse_m_mean':float(np.mean([r['rmse_m'] for r in ok])) if ok else None,
                'violation_max_m':float(np.max([r['violation_max_m'] for r in ok])) if ok else None,
                'integration_steps_mean':float(np.mean([r['integration_steps_mean'] for r in ok])) if ok else None,
                'seconds_mean':float(np.mean([r['seconds'] for r in rr]))}
    Path(ROOT/a.out).write_text(json.dumps({'purpose':'SafeDiffuser delta-tau / substep treatment audit',
        'config':a.config,'indices':a.indices,'rows':rows,'aggregate':agg},indent=2),encoding='utf8')
    print(json.dumps(agg,indent=2));print('wrote',a.out)

if __name__=='__main__':main()
