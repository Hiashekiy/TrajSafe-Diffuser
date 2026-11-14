"""Small validation of the seven paper loops on the native-point base.

Writes one JSONL record per (index, method, seed) with the delivered polyline,
the internal state, per-level traces, timings and independent evaluation
metrics. Fails loudly if the delivered array is not the native 128-point
trajectory, if fixed endpoints moved, or if the state and the output differ.
"""
import argparse,collections,datetime,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only_points import load_point_baseline
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_sampler import sample_points
from experiments.safety_baselines.adapters.point_evaluator import evaluate_points
from experiments.safety_baselines.algorithms.runner import run_method,METHODS,derived_seed,initial_noise

def aggregate(records,methods):
    out={}
    for method in methods:
        rr=[r for r in records if r['method']==method];ret=[r for r in rr if r.get('metrics')]
        ok=[r for r in rr if r['status']=='ok']
        seg=[r['metrics']['segment_residual_scene'] for r in ret if r['metrics'].get('segment_residual_scene') is not None]
        out[method]={'attempts':len(rr),'returned':len(ret),'solver_failures':len(rr)-len(ret),'status_ok':len(ok),
            'vertex_collisions':int(sum(r['metrics']['vertex_collision'] for r in ret)),
            'segment_only_collisions':int(sum(r['metrics']['segment_only_collision'] for r in ret)),
            'path_collisions':int(sum(r['metrics']['collision'] for r in ret)),
            'planning_success':int(sum(r['metrics']['planning_success'] for r in ret)),
            'rmse_m_mean':float(np.mean([r['metrics']['rmse_m'] for r in ret])) if ret else None,
            'violation_m_max':float(np.max([r['metrics']['violation_max_m'] for r in ret])) if ret else None,
            'segment_residual_m_max':float(np.max(seg)*80) if seg else None,
            'failure_reasons':dict(collections.Counter(r['failure_reason'] for r in rr if r['failure_reason'])),
            'time_mean_s':float(np.mean([r['wall_time'] for r in rr])) if rr else None}
    return out

def resolve_indices(cfg,split,spec,dataset_size):
    """val/test indices are strictly separated; nothing is reused across splits."""
    if spec in ('frozen20','default'):
        key='validation_indices' if split=='val' else 'test_indices'
        if key not in cfg:raise ValueError('config has no %s for split=%s'%(key,split))
        idx=list(cfg[key])
    elif spec=='all':idx=list(range(dataset_size))
    else:idx=[int(v) for v in spec.split(',')]
    bad=[v for v in idx if v<0 or v>=dataset_size]
    if bad:raise IndexError('indices %s out of range for split=%s (size %d)'%(bad,split,dataset_size))
    if len(set(idx))!=len(idx):raise ValueError('duplicate indices in %s'%spec)
    return idx

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments/safety_baselines/configs/seven_native_points_paper_t256.json')
    ap.add_argument('--indices',default='frozen20',help="'frozen20' or comma separated validation indices")
    ap.add_argument('--methods',default='all')
    ap.add_argument('--seeds',default='0')
    ap.add_argument('--out-dir',default=None)
    ap.add_argument('--split',default='val',choices=['val','test'],help='val is used for checkpoint selection; test is the final protocol')
    ap.add_argument('--ddim-steps',type=int,default=16,help='DDIM steps for the unconstrained baseline (match the budget of the loops under review)')
    a=ap.parse_args()
    base=ROOT/'experiments/safety_baselines';cfg=json.loads((ROOT/a.config).read_text(encoding='utf8'))

    methods=METHODS+['Diffuser'] if a.methods=='all' else a.methods.split(',')
    seeds=[int(v) for v in a.seeds.split(',')]
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/a.out_dir if a.out_dir else base/'results'/('native_points_aligned_'+stamp)
    out.mkdir(parents=True,exist_ok=True)
    ckpt=ROOT/cfg['checkpoint'];model,ck=load_point_baseline(ckpt,'cuda');d=ck['config']['diffusion']
    schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=PointDataset(a.split,ck['config']['data']['processed_root'])
    indices=resolve_indices(cfg,a.split,a.indices,len(ds))
    ids=np.load(Path(ck['config']['data']['processed_root'])/a.split/'sample_id.npy')
    manifest={'status':'running','checkpoint':cfg['checkpoint'],'checkpoint_hash':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        'config_path':a.config,'config_hash':hashlib.sha256((ROOT/a.config).read_bytes()).hexdigest(),
        'split':a.split,'indices':indices,'scene_ids':[int(v) for v in ids[indices]],'seeds':seeds,'methods':methods,'expected_records':len(indices)*len(seeds)*len(methods),
        'completed_records':0,'representation':'direct128points; no spline, lifting or refit','config':cfg,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['algorithms','adapters','models'] for p in (base/folder).rglob('*.py')}}
    def write(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    write();records=[]
    try:
        with threadpool_limits(limits=1),(out/'results.jsonl').open('w',encoding='utf8') as f:
            for index in indices:
                row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();scene=int(ids[index])
                for seed in seeds:
                    init=initial_noise(scene,seed,128);init[[0,-1]]=cond  # paired protocol: one shared start
                    for method in methods:
                        if method=='Diffuser':
                            torch.cuda.synchronize();start=time.perf_counter()
                            q=sample_points(model,schedule,row['occupancy'][None].cuda(),row['cond'][None].cuda(),
                                seed=derived_seed(scene,seed,method)%(2**31-1),init=init,steps=a.ddim_steps)
                            torch.cuda.synchronize()
                            result={'trajectory':q[0].cpu().tolist(),'status':'ok','failure_reason':None,
                                'wall_time':time.perf_counter()-start,'method':method,'per_step_trace':[]}
                        else:result=run_method(method,model,schedule,occ,cond,scene,seed,cfg,init_noise=init)
                        p=result['trajectory'];metrics=None
                        if p is not None:
                            if np.asarray(p).shape!=(128,2):raise ValueError('output is not native trajectory points')
                            if result.get('internal_state_trajectory') is not None and not np.array_equal(
                                np.asarray(p),np.asarray(result['internal_state_trajectory'])):raise ValueError('internal/output representation mismatch')
                            if not np.allclose(np.asarray(p)[[0,-1]],cond,rtol=0,atol=1e-7):raise ValueError('fixed endpoints changed')
                            metrics=evaluate_points(p,occ,cond,row['pos'].numpy())
                            if result.get('internal_max_violation_scene') is not None:
                                metrics['segment_residual_scene']=float(result['internal_max_violation_scene'])
                        record={**result,'scene_id':scene,'index':index,'seed':seed,'metrics':metrics}
                        f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();records.append(record)
                        manifest['completed_records']=len(records);write()
                        print(len(records),'/',manifest['expected_records'],index,seed,method,result['status'],
                            result['failure_reason'],'vtx',None if metrics is None else metrics['vertex_collision'],
                            'seg',None if metrics is None else metrics['segment_only_collision'],
                            'path',None if metrics is None else metrics['collision'],
                            's',round(result['wall_time'],2),flush=True)
        summary=aggregate(records,methods)
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        (out/'summary.md').write_text(summary_markdown(summary,manifest),encoding='utf8')
        manifest.update(status='complete',representation_gate='passed');write();print('COMPLETE',out,flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));write();raise

def summary_markdown(summary,manifest):
    lines=['# Native point validation','',f"config `{manifest['config_path']}`  hash `{manifest['config_hash'][:12]}`",
        f"checkpoint `{manifest['checkpoint']}`",f"indices {len(manifest['indices'])}  seeds {manifest['seeds']}  records {manifest['completed_records']}",
        '','|method|returned|vertex coll.|segment-only coll.|path coll.|planning success|RMSE m|max violation m|segment residual m|time s|failures|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for m,v in summary.items():
        lines.append('|{}|{}/{}|{}|{}|{}|{}|{:.2f}|{:.3f}|{}|{:.2f}|{}|'.format(m,v['returned'],v['attempts'],
            v['vertex_collisions'],v['segment_only_collisions'],v['path_collisions'],v['planning_success'],
            v['rmse_m_mean'] if v['rmse_m_mean'] is not None else float('nan'),
            v['violation_m_max'] if v['violation_m_max'] is not None else float('nan'),
            'n/a' if v['segment_residual_m_max'] is None else '{:.3f}'.format(v['segment_residual_m_max']),
            v['time_mean_s'],v['failure_reasons']))
    lines+=['','Not a paper comparison: 20 fixed validation cases, one seed. Collision counts use the',
        'independent occupancy evaluator on the delivered 128-point polyline.']
    return '\n'.join(lines)

if __name__=='__main__':main()
