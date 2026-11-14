"""Small paired validation of unmodified old correctors on the new base."""
import collections,datetime,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from threadpoolctl import threadpool_limits
from src.datasets.carla_spline_dataset import CarlaSplineDataset
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only import load_baseline
from experiments.safety_baselines.adapters.legacy_sdf import sample_legacy_sdf
from experiments.safety_baselines.adapters.evaluator import evaluate_curve

def main():
    base=ROOT/'experiments/safety_baselines'
    cfg=json.loads((base/'configs/seven_curve_control_t16.json').read_text())
    model,ck=load_baseline(ROOT/cfg['checkpoint'],'cuda');d=ck['config']['diffusion']
    schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=CarlaSplineDataset('val',ck['config']['data']['processed_root'],num_controls=48)
    out=base/'results'/('legacy_reuse_val_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
    records=[]
    # Original function defaults, no parameter selection on test outcomes.
    with threadpool_limits(limits=1),(out/'results.jsonl').open('w',encoding='utf8') as f:
        for index in cfg['validation_indices']:
            row=ds[index];occ=row['occupancy'][None].cuda();cond=row['cond'][None].cuda()
            for mode in ['none','penalty','projection','primal_dual','alm_sdf']:
                torch.cuda.synchronize();start=time.perf_counter()
                q,trace=sample_legacy_sdf(model,schedule,occ,cond,mode,seed=0,config={'inner_steps':5})
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                metrics=evaluate_curve(model.bspline,q[0].cpu().numpy(),occ[0,0].cpu().numpy(),cond[0].cpu().numpy(),row['pos'].numpy())
                metrics.pop('_parameters',None)
                if not torch.equal(q[:,[0,-1]],cond):raise ValueError('endpoint changed')
                if any(a>b+1e-6 for step in trace for a,b in zip(step['violation_after'],step['violation_before'])):raise ValueError('legacy guard violated')
                r={'split':'val','index':index,'scene_id':int(row['sample_id']),'method':'legacy_'+mode,'seed':0,'wall_time':elapsed,'metrics':metrics,'trace':trace,'trajectory':q[0].cpu().tolist()}
                f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();records.append(r)
                print(index,mode,'collision',metrics['collision'],'max_violation_m',round(metrics['common_sdf_violation_max_m'],3),'seconds',round(elapsed,3),flush=True)
    summary={mode:{'n':sum(r['method']==mode for r in records),'collisions':sum(r['metrics']['collision'] for r in records if r['method']==mode),'correction_statuses':dict(collections.Counter(status for r in records if r['method']==mode for step in r['trace'] for status in step['status']))} for mode in sorted({r['method'] for r in records})}
    (out/'summary.json').write_text(json.dumps({'status':'complete','purpose':'engineering reuse validation, NOT seven-paper reproduction','checkpoint':cfg['checkpoint'],'indices':cfg['validation_indices'],'same_initial_seed':True,'methods':summary},indent=2),encoding='utf8')
    print('COMPLETE',out,flush=True);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
