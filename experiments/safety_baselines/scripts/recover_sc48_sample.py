"""Phase 1-2: sample the recovered S-C48 checkpoint with its OWN historical code.

The checkpoint `outputs/models/oneshot_k4p_c48_s/ckpt/best_task.pt` (epoch 65) was
trained 2026-09-25, before commit c1d97bf (2026-10-26) changed `SceneCNN`,
`FeedbackEncoder` and `planner`. Loading it with the current code drops 20 scene-CNN
decoder tensors and leaves `feedback_encoder.net.0.weight` random, so this script
imports the code tree of `c1d97bf^` (extracted read-only to
`experiments/safety_baselines/legacy_sc48_code`) and samples with it:

    guidance="none"  -> plain DDIM: no corridor, no feedback, no ALM, no projection
    steps=16         -> the native 16-level training schedule

Output: one npz per run with, per scene, the 48 control points, the 512-point decoded
curve (u grid saved), the per-level trace and the historical dense_validation block.
"""
import argparse,json,os,sys,time
LEGACY=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','legacy_sc48_code'))
sys.path.insert(0,LEGACY)
import numpy as np
import torch
from src.utils.config import load_config,num_controls,curve_points
from src.utils.checkpoint import load_model
from src.diffusion.schedule import NoiseSchedule
from src.datasets.carla_spline_dataset import CarlaSplineDataset,make_collate
from src.diffusion.sampler import sample as legacy_sample

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='outputs/models/oneshot_k4p_c48_s/ckpt/best_task.pt')
    ap.add_argument('--config',default=os.path.join(LEGACY,'configs/config_160k4p_c48_s_nofb.yaml'))
    ap.add_argument('--split',default='val')
    ap.add_argument('--indices',default='0,81,163,244,326,407,489,570,652,733,815,896,978,1059,1141,1222,1304,1385,1467,1549')
    ap.add_argument('--steps',type=int,default=16)
    ap.add_argument('--guidance',default='none')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--curve-points',type=int,default=512)
    ap.add_argument('--out',default='experiments/safety_baselines/results/sc48_recovery/sc48_samples.npz')
    a=ap.parse_args()
    root=os.path.dirname(LEGACY);proj=os.path.dirname(os.path.dirname(root))
    ck=os.path.join(proj,a.checkpoint)
    cfg=load_config(a.config)
    model,state,report=load_model(cfg,ck,device='cuda',verbose=False)
    if report['missing'] or report['unexpected'] or report['adapted']:
        raise SystemExit('checkpoint/config mismatch with the historical code: %s'%{
            k:report[k] for k in ('missing','unexpected','adapted')})
    model.eval()
    d=cfg['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d.get('beta_start',1e-4),d.get('beta_end',.02)).to('cuda')
    ds=CarlaSplineDataset(a.split,cfg['data']['processed_root'])
    collate=make_collate(ds)
    print('num_controls=%s curve_points=%s T=%s step_ckpt=%s'%(num_controls(cfg),curve_points(cfg),d['timesteps'],ck))
    ckdata=torch.load(ck,map_location='cpu',weights_only=False)
    meta={'checkpoint':a.checkpoint,'epoch':int(ckdata.get('epoch')),'params_M':sum(p.numel() for p in model.parameters())/1e6,
          'num_controls':int(model.num_controls),'curve_points':int(model.bspline.curve_points),
          'timesteps':int(d['timesteps']),'steps':a.steps,'guidance':a.guidance,'seed':a.seed,
          'split':a.split,'indices':[int(v) for v in a.indices.split(',')],
          'legacy_commit':'c1d97bf^','compat_report':{k:report[k] for k in ('missing','unexpected','adapted','arch')}}
    u=torch.linspace(0.,1.,a.curve_points,device='cuda')
    store={'u':u.cpu().numpy(),'meta':np.array(json.dumps(meta)),'sample_id':[],
           'Q':[],'curve':[],'curve128':[],'control_hit':[],'trace':[],'hist_val':[],'seconds':[]}
    u128=torch.linspace(0.,1.,128,device='cuda')
    with torch.no_grad():
        for index in meta['indices']:
            item=ds[index];batch=collate([item])
            b={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
            torch.manual_seed(a.seed)
            torch.cuda.synchronize();t0=time.perf_counter()
            out=legacy_sample(model,schedule,b['cond'],b['occupancy'],b['candidate_xy'],b['candidate_mask'],
                b['candidate_geometry'],b['candidate_geometry_lengths'],device='cuda',steps=a.steps,seed=a.seed,
                return_trace=True,guidance=a.guidance)
            torch.cuda.synchronize();secs=time.perf_counter()-t0
            store['seconds'].append(float(secs))
            q=out['q'] if 'q' in out else out['control']
            Q=q[0].detach().cpu().numpy()
            curve=(model.bspline.basis_at(u)@q[0]).detach().cpu().numpy()
            curve128=(model.bspline.basis_at(u128)@q[0]).detach().cpu().numpy()
            occ=np.asarray(item['occupancy'][0].numpy(),dtype=bool)
            occR=occ.shape[-1]
            def hit(pts):
                inside=((pts>=-1)&(pts<1)).all(1);cell=np.floor((pts+1)*occR/2).astype(int);cell=np.clip(cell,[0,0],[occR-1,occR-1])
                return bool((~inside|occ[cell[:,1],cell[:,0]]).any())
            store['sample_id'].append(int(ds.sample_id[index]))
            store['Q'].append(Q);store['curve'].append(curve);store['curve128'].append(curve128)
            store['control_hit'].append(hit(Q))
            store['trace'].append(json.dumps([{k:v for k,v in rec.items() if isinstance(v,(int,float,str,bool)) or v is None}
                                              for rec in (out.get('trace') or [])])[:200000])
            fv=out.get('final_validation') or {}
            if isinstance(fv,list):fv=fv[0] if fv else {}
            store['hist_val'].append(json.dumps({k:(v if isinstance(v,(int,float,bool)) or v is None else None) for k,v in (fv or {}).items()})[:20000])
            print(index,'sample_id',store['sample_id'][-1],'control_hit',store['control_hit'][-1],
                  'hist_val keys=',sorted((fv or {}).keys())[:6],'collision=',(fv or {}).get('collision'),
                  flush=True)
    outp=os.path.join(proj,a.out);os.makedirs(os.path.dirname(outp),exist_ok=True)
    np.savez_compressed(outp,u=store['u'],meta=store['meta'],sample_id=np.asarray(store['sample_id']),
        Q=np.stack(store['Q']),curve=np.stack(store['curve']),curve128=np.stack(store['curve128']),
        control_hit=np.asarray(store['control_hit']),trace=np.asarray(store['trace']),hist_val=np.asarray(store['hist_val']),
        seconds=np.asarray(store['seconds']))
    print('wrote',outp)

if __name__=='__main__':main()
