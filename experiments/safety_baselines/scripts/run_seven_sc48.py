"""Plan B: run the seven safety loops on S-C48 (48 B-spline controls, native 16 levels).

Runs inside the historical code tree (`legacy_sc48_code`, commit c1d97bf^) because the
current HEAD cannot load the S-C48 checkpoint faithfully, and imports the current
repo's algorithm package for the loops themselves.

State = the 46 free control points. Constraints are evaluated on the decoded curve
p(u)=B(u)Q (128 points) and mapped back exactly by J_Q = -(B[:,1:-1]*grad_p), which is
what `adapters/paper_task.py:PaperTask.constraints` already does. The delivered object
is the decoded curve, scored by `adapters/unified_curve_eval.py` at 512 points.

The unconstrained row ("Diffuser") is a plain DDIM over the same 16 levels starting
from the SAME initial control noise as the seven loops, so the whole table is paired.
"""
import argparse,csv,json,os,sys,time
LEGACY=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','legacy_sc48_code'))
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))
sys.path.insert(0,LEGACY);sys.path.insert(1,ROOT)
import numpy as np
import torch
from src.utils.config import load_config
from src.utils.checkpoint import load_model
from src.diffusion.schedule import NoiseSchedule
from src.datasets.carla_spline_dataset import CarlaSplineDataset
from experiments.safety_baselines.algorithms.runner import run_method,METHODS,initial_seed
from experiments.safety_baselines.adapters.unified_curve_eval import evaluate_curve

def diffuser_ddim(task,cfg):
    """Unconstrained DDIM on the same 16 levels, from the task's own initial state."""
    x=np.asarray(task.initial_override,dtype=float).reshape(-1).copy()
    n=len(task.ab)
    for t in range(n-1,-1,-1):
        clean=task.clean(x,t)[0]
        if t==0:x=clean
        else:
            noise=(x-np.sqrt(task.ab[t])*clean)/np.sqrt(1-task.ab[t])
            x=np.sqrt(task.ab[t-1])*clean+np.sqrt(1-task.ab[t-1])*noise
        q=task.unpack(x);q[0]=task.endpoints[0];q[-1]=task.endpoints[1];x=q[1:-1].reshape(-1)
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments/safety_baselines/configs/sc48_native16_vertexall.json')
    ap.add_argument('--base-config',default=os.path.join(LEGACY,'configs/config_160k4p_c48_s_nofb.yaml'))
    ap.add_argument('--indices',default='0,81,163,244,326,407,489,570,652,733,815,896,978,1059,1141,1222,1304,1385,1467,1549')
    ap.add_argument('--methods',default=','.join(['Diffuser']+METHODS))
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--eval-points',type=int,default=512)
    ap.add_argument('--out-dir',default='experiments/safety_baselines/results/sc48_seven')
    a=ap.parse_args()
    cfg=json.loads(open(os.path.join(ROOT,a.config),encoding='utf8').read())
    bcfg=load_config(a.base_config)
    bcfg['data']['processed_root']=cfg.get('data_root','D:/ProjectDirectory/Neural-IRISDiffuser/data/processed/160m-k4p-c48')
    ck=os.path.join(ROOT,cfg['checkpoint'])
    model,state,report=load_model(bcfg,ck,device='cuda',verbose=False)
    if report['missing'] or report['unexpected'] or report['adapted']:
        raise SystemExit('historical code no longer matches the checkpoint: %s'%{k:report[k] for k in ('missing','unexpected','adapted')})
    model.eval()
    d=bcfg['diffusion'];sched=NoiseSchedule(d['timesteps'],d['beta_schedule'],d.get('beta_start',1e-4),d.get('beta_end',.02)).to('cuda')
    ds=CarlaSplineDataset('val',bcfg['data']['processed_root'])
    out=os.path.join(ROOT,a.out_dir);os.makedirs(out,exist_ok=True)
    u=torch.linspace(0.,1.,a.eval_points,device='cuda')
    B512=model.bspline.basis_at(u).detach().cpu().numpy().astype(float)
    methods=[m for m in a.methods.split(',') if m]
    rows=[]
    with torch.inference_mode():
        for index in [int(v) for v in a.indices.split(',')]:
            item=ds[index];scene=int(ds.sample_id[index])
            occ=np.asarray(item['occupancy'][0].numpy(),dtype=bool);cond=np.asarray(item['cond'],dtype=float)
            gt128=np.asarray(item['pos'].numpy(),dtype=float);gt_curve=np.stack([np.interp(u.cpu().numpy(),np.linspace(0,1,len(gt128)),gt128[:,k]) for k in (0,1)],axis=1)
            init=np.random.default_rng(initial_seed(scene,a.seed)).normal(size=(model.num_controls,2));init[[0,-1]]=cond
            curves={}
            for method in methods:
                t0=time.perf_counter()
                cand=dict(candidate_xy=np.asarray(item['candidate_xy']),candidate_mask=np.asarray(item['candidate_mask']),
                          candidate_geometry=np.asarray(item['candidate_geometry']),geometry_lengths=np.asarray(item['candidate_geometry_lengths']))
                if method=='Diffuser':
                    from experiments.safety_baselines.adapters.sc48_control import SC48ControlTask
                    task=SC48ControlTask(model,sched,occ,cond,cfg['constraint_points'],cfg.get('radius_scene',0.),cfg.get('field_kind','bilinear'),candidates=cand)
                    task.initial_override=np.asarray(init,dtype=float)[1:-1].reshape(-1)
                    task.particle_seed=initial_seed(scene,a.seed)
                    value=diffuser_ddim(task,cfg);status='ok';reason=None;trace=[]
                else:
                    r=run_method(method,model,sched,occ,cond,scene,a.seed,cfg,init_noise=init,task_extra={'candidates':cand})
                    value=None if r['trajectory'] is None else r['trajectory']
                    status=r['status'];reason=r['failure_reason'];trace=r['per_step_trace']
                    if value is not None:value=np.asarray(value,dtype=float)[1:-1].reshape(-1)
                if value is None:
                    rows.append({'scene':scene,'index':index,'method':method,'status':status,'failure_reason':reason,
                                 'seconds':time.perf_counter()-t0,'curve_collision':None})
                    print(scene,method,'FAILED',reason,flush=True);continue
                q=np.zeros((model.num_controls,2));q[1:-1]=np.asarray(value).reshape(model.num_controls-2,2);q[0]=cond[0];q[-1]=cond[1]
                curve=B512@q
                m=evaluate_curve(curve,occ,cond,gt_curve,u.cpu().numpy())
                m.update(scene=scene,index=index,method=method,status=status,failure_reason=reason,
                         seconds=time.perf_counter()-t0,nfe=int(cfg.get('cd_levels',16)) if method!='Diffuser' else len(sched.alphas_cumprod),
                         trace_len=len(trace or []))
                rows.append(m);curves[method]=curve
                print(scene,method,'path',m['curve_collision'],'succ',m['planning_success'],
                      'clear %.1f'%m['min_clearance_m'],'rmse %.1f'%m['rmse_m'],'%.1fs'%m['seconds'],flush=True)
            np.savez_compressed(os.path.join(out,'scene_%d_curves.npz'%scene),u=u.cpu().numpy(),gt=gt_curve,
                                occ=occ,cond=cond,**{k:v for k,v in curves.items()})
            with open(os.path.join(out,'results.jsonl'),'a',encoding='utf8') as f:
                for r in rows:
                    if r['scene']==scene:f.write(json.dumps(r)+'\n')
    if rows:
        keys=sorted({k for r in rows for k in r})
        with open(os.path.join(out,'seven_sc48.csv'),'w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        agg={}
        for method in methods:
            rr=[r for r in rows if r['method']==method];ok=[r for r in rr if r.get('curve_collision') is not None]
            agg[method]={'n':len(rr),'returned':len(ok),
                'path_collisions':int(sum(r['curve_collision'] for r in ok)),
                'vertex_collisions':int(sum(r['curve_vertex_collision'] for r in ok)),
                'successes':int(sum(r['planning_success'] for r in ok)),
                'min_clearance_m_mean':float(np.mean([r['min_clearance_m'] for r in ok])) if ok else None,
                'violation_max_m':float(np.max([r['violation_max_m'] for r in ok])) if ok else None,
                'rmse_m_mean':float(np.mean([r['rmse_m'] for r in ok])) if ok else None,
                'path_length_ratio_mean':float(np.mean([r['path_length_ratio'] for r in ok])) if ok else None,
                'max_curvature_mean':float(np.mean([r['max_curvature_1pm'] for r in ok])) if ok else None,
                'mean_second_diff_m_mean':float(np.mean([r['mean_second_difference_m'] for r in ok])) if ok else None,
                'seconds_mean':float(np.mean([r['seconds'] for r in rr]))}
        with open(os.path.join(out,'seven_sc48_summary.json'),'w',encoding='utf8') as f:
            json.dump({'config':a.config,'checkpoint':cfg['checkpoint'],'seed':a.seed,
                       'indices':[int(v) for v in a.indices.split(',')],'aggregate':agg},f,indent=2)
        for k,v in agg.items():print('%-14s'%k,json.dumps(v))
        print('wrote',out)

if __name__=='__main__':main()
