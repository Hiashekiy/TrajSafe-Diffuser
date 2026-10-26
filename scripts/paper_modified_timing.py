"""Idle timing for the new checkpoints; four acceleration cases interleaved."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from ch4_perf_timing import load,run,CarlaSplineDataset,make_collate
from src.diffusion.sampler import sample
from src.diffusion.schedule import NoiseSchedule
from src.geometry.sdf_field import build_occupancy_sdf
from eval_chapter4 import evaluate_controls

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    target=Path(args.run);m=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
    ds=CarlaSplineDataset('test',m['processed_dataset'],geometry_points=1280);coll=make_collate(ds)
    cfg,model=load(m['configs']['D'],str(target/'D/ckpt/best_task.pt'))
    result={'protocol':{'n':len(ds),'seed':0,'chunk':32,'final_project':False,'idle_after_training':True},'methods':{},'acceleration':{}}
    for guidance in ('none','penalty','primal_dual','alm_sdf'):
        extra={} if guidance=='none' else json.loads((target/'sweeps'/f'{guidance}.json').read_text(encoding='utf-8'))['winner']
        result['methods'][guidance]=run(cfg,model,ds,coll,16,guidance,extra,measure_metrics=False)
    del model;torch.cuda.empty_cache()
    cfg,model=load(m['configs']['A'],str(target/'A/ckpt/best_task.pt'))
    result['methods']['TrajSafe']=run(cfg,model,ds,coll,16,'alm',measure_metrics=False)
    # Import the original accumulator without interpreting this script's CLI.
    oldargv=sys.argv;sys.argv=[oldargv[0]]
    from ch4_perf_stride import Acc
    sys.argv=oldargv
    cases=((16,1),(4,1),(16,8),(4,8));accs={case:Acc() for case in cases}
    sch=NoiseSchedule(16,beta_schedule=cfg['diffusion'].get('beta_schedule','squaredcos_cap_v2'))
    s2m=float(cfg['data'].get('scene_to_meter',80))
    for start in range(0,len(ds),32):
        batch=coll([ds[i] for i in range(start,min(start+32,len(ds)))])
        inputs=(model,sch,batch['cond'].cuda(),batch['occupancy'].cuda(),batch['candidate_xy'].cuda(),batch['candidate_mask'].cuda(),batch['candidate_geometry'].cuda(),batch['candidate_geometry_lengths'].cuda())
        for steps,stride in cases:
            alm=dict(cfg['alm'],final_project=False,guidance='alm')
            corridor=dict(cfg.get('corridor') or {},region_stride=stride)
            kw=dict(device='cuda',steps=steps,seed=0,seed_offset=start,return_trace=False,alm_config=alm,corridor_config=corridor,guidance='alm')
            if start==0:sample(*inputs,**kw)
            torch.cuda.synchronize(); w0,c0=time.perf_counter(),time.process_time()
            out=sample(*inputs,**kw);torch.cuda.synchronize()
            wall,cpu=time.perf_counter()-w0,time.process_time()-c0
            sdf=build_occupancy_sdf(batch['occupancy'].cuda(),scene_to_meter=s2m,map_margin=float(cfg['alm'].get('map_margin',0)))
            metrics=evaluate_controls(model.bspline,out['control'],batch['occupancy'].cuda(),batch['cond'].cuda(),batch['pos'].cuda(),sdf,None,batch,s2m)
            accs[(steps,stride)].add(wall,cpu,out,metrics)
        print('timing',start+len(batch['cond']),'/',len(ds),flush=True)
    for (steps,stride),a in accs.items():
        result['acceleration'][f'steps{steps}_stride{stride}']=dict(n=a.n,wall_ms=1000*a.wall/a.n,cpu_ms=1000*a.cpu/a.n,
            network_ms=1000*a.net/a.n,corridor_ms=1000*a.corr/a.n,final_projection_ms=1000*a.proj/a.n,
            collisions=a.coll,feasible_rate=a.feas/a.feas_n,rmse_m=a.rmse/a.feas_n,smoothness_m=a.smooth/a.feas_n,
            base_cells=a.cells/max(a.seen,1),bridges=a.bridges/max(a.seen,1),corridor_failures=a.fails)
    (target/'timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
