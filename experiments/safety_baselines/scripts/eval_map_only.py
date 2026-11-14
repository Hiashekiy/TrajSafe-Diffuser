"""Evaluate map-only Diffuser using the existing 512-point common evaluator.
No baseline optimizer, skeleton conditions, corridor or GT enters sampling.
"""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from src.datasets.carla_spline_dataset import CarlaSplineDataset,make_collate
from src.diffusion.schedule import NoiseSchedule
from src.geometry.sdf_field import build_occupancy_sdf
from scripts.eval_chapter4 import evaluate_controls
from experiments.safety_baselines.models.map_only import load_baseline
from experiments.safety_baselines.adapters.baseline_sampler import sample_baseline
CKPT='experiments/safety_baselines/results/map_only_l1_final8_e100_20261112_212929/ckpt/best_task.pt'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--samples',type=int,default=0);args=ap.parse_args()
    out=Path(args.out).resolve();base=ROOT/'experiments/safety_baselines/results'
    if not out.is_relative_to(base.resolve()):raise ValueError('output outside experiment results')
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() or out.with_suffix('.jsonl').exists():raise FileExistsError('refuse overwrite')
    model,ck=load_baseline(ROOT/CKPT,'cuda');cfg=ck['config'];dc=cfg['diffusion'];scale=cfg['data']['scene_to_meter']
    schedule=NoiseSchedule(dc['timesteps'],dc['beta_schedule'],dc['beta_start'],dc['beta_end']).to('cuda')
    ds=CarlaSplineDataset('test',cfg['data']['processed_root'],geometry_points=1280,limit=args.samples or None,num_controls=48);collate=make_collate(ds)
    records=[];times=[];started=time.perf_counter();chunk=32
    for start in range(0,len(ds),chunk):
        batch=collate([ds[i] for i in range(start,min(start+chunk,len(ds)))]);occ=batch['occupancy'].cuda();cond=batch['cond'].cuda()
        torch.cuda.synchronize();t=time.perf_counter()
        q=sample_baseline(model,schedule,occ,cond,args.seed,start,16)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        if start//chunk>=3:times.append(elapsed*1000/len(q))
        sdf=build_occupancy_sdf(occ,scene_to_meter=scale)
        # Existing evaluator reads offline corridors for diagnostics only; never sampler input.
        metrics=evaluate_controls(model.bspline,q,occ,cond,batch['pos'].cuda(),sdf,None,batch,scale)
        keys=['collision','rmse_m','smoothness_m','sdf_feasible','sdf_violation_m','sdf_min_distance_m','endpoint_error','map_excursion','free_rate']
        for i in range(len(q)):
            rec={'method':'Diffuser','sample_id':int(batch['sample_id'][i]),'index':start+i,'seed':args.seed,'status':'returned','control':q[i].cpu().tolist()}
            rec.update({k:metrics[k][i].item() for k in keys});records.append(rec)
        print(f'{start+len(q)}/{len(ds)} collisions={sum(r["collision"] for r in records)}',flush=True)
    means={k:statistics.mean(r[k] for r in records) for k in keys}
    aggregate={'n':len(records),'rmse_m':means['rmse_m'],'collision_rate':means['collision'],'sdf_feasible_rate':means['sdf_feasible'],'sdf_mean_violation_m':means['sdf_violation_m'],'sdf_min_distance_m':means['sdf_min_distance_m'],'smoothness_m':means['smoothness_m'],'endpoint_error':means['endpoint_error'],'map_excursion_rate':means['map_excursion'],'runtime_ms_mean':statistics.mean(times) if times else None}
    payload={'method':'Diffuser','checkpoint':CKPT,'checkpoint_hash':hashlib.sha256((ROOT/CKPT).read_bytes()).hexdigest(),'seed':args.seed,'split':'test','n_samples':len(ds),'chunk':32,'steps':16,'dense_points':512,'warmup_chunks':3,'status':'complete','wall_seconds':time.perf_counter()-started,'aggregate':aggregate,'limitations':['legacy fixed chunk32 noise protocol; changing chunk changes noise','512-point common evaluator; no continuous guarantee or adaptive refinement','timing covers sampling with map encoder; excludes model/data load and evaluation-only SDF','seven paper safety runners not yet validated']}
    out.with_suffix('.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf8');out.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf8');print(aggregate,flush=True)
if __name__=='__main__':main()
