"""Phase E: does the discrete barrier certificate actually stop the penetration?

Same cases, same seed, same QP: only `cbf_discrete_check` changes. Reports the
delivered clearance, halvings and rejects per level.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only_points import load_point_baseline
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_evaluator import evaluate_points
from experiments.safety_baselines.algorithms.runner import run_method,initial_noise

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments/safety_baselines/configs/seven_native_points_vertexall_t256.json')
    ap.add_argument('--indices',default='0,81,163')
    ap.add_argument('--methods',default='RoS-diffuser,TVS-diffuser,ReS-diffuser')
    ap.add_argument('--out',default='experiments/safety_baselines/results/phase_e_safediffuser/discrete_barrier.jsonl')
    a=ap.parse_args();cfg=json.loads((ROOT/a.config).read_text(encoding='utf8'))
    model,ck=load_point_baseline(ROOT/cfg['checkpoint'],'cuda')
    d=ck['config']['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=PointDataset('val',ck['config']['data']['processed_root'])
    out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True);rows=[]
    with out.open('w',encoding='utf8') as f:
        for index in [int(v) for v in a.indices.split(',')]:
            row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();gt=row['pos'].numpy()
            init=initial_noise(index,0,128);init[[0,-1]]=cond
            for method in a.methods.split(','):
                for check in (False,True):
                    c=dict(cfg,cbf_discrete_check=check);t0=time.perf_counter()
                    r=run_method(method,model,schedule,occ,cond,index,0,c,init_noise=init)
                    m=evaluate_points(r['trajectory'],occ,cond,gt) if r['trajectory'] is not None else None
                    tr=r['per_step_trace'] or []
                    rec={'index':index,'method':method,'discrete':check,'status':r['status'],
                        'failure_reason':r['failure_reason'],'seconds':time.perf_counter()-t0,
                        'level_retries':int(sum(max(0,t.get('level_attempts',1)-1) for t in tr)),
                        'levels_discrete_ok':int(sum(bool(t.get('discrete_ok')) for t in tr)),
                        'levels_discrete_bad':int(sum(t.get('discrete_ok') is False for t in tr)),
                        'metrics':m}
                    rows.append(rec);f.write(json.dumps(rec)+'\n');f.flush()
                    print(index,method,'discrete',check,r['status'],
                        None if m is None else ('vtx %s path %s minclear %.2f m viol %.2f m'%(
                        m['vertex_collision'],m['collision'],m['min_clearance_m'],m['violation_max_m'])),
                        'retries',rec['level_retries'],'bad',rec['levels_discrete_bad'],flush=True)
    agg={}
    for key in {(r['method'],r['discrete']) for r in rows}:
        rr=[r for r in rows if (r['method'],r['discrete'])==key];ok=[r for r in rr if r['metrics']]
        agg['%s|discrete=%s'%key]={'n':len(rr),'returned':len(ok),
            'vertex_collisions':int(sum(r['metrics']['vertex_collision'] for r in ok)),
            'path_collisions':int(sum(r['metrics']['collision'] for r in ok)),
            'violation_max_m_max':float(np.max([r['metrics']['violation_max_m'] for r in ok])) if ok else None,
            'min_clearance_m_mean':float(np.mean([r['metrics']['min_clearance_m'] for r in ok])) if ok else None,
            'level_retries':int(sum(r['level_retries'] for r in rr)),'levels_discrete_bad':int(sum(r['levels_discrete_bad'] for r in rr)),
            'seconds_mean':float(np.mean([r['seconds'] for r in rr]))}
    (out.parent/'discrete_barrier_summary.json').write_text(json.dumps(agg,indent=2),encoding='utf8')
    print(json.dumps(agg,indent=2))

if __name__=='__main__':main()
