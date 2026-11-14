"""Phase 3: unified evaluation of the recovered S-C48 against the current map-only base.

Both models are scored by ONE evaluator on ONE set of scenes (same sample_id, same
occupancy, same endpoints, same seed). The two models produce different objects --
S-C48 delivers 48 B-spline control points whose *curve* is the plan, the map-only
base delivers 128 waypoints whose *polyline* is the plan -- so the comparable
quantity is the curve itself, sampled on the same grid:

    S-C48      : p(u) = basis_at(u) @ Q                    (cubic B-spline)
    map-only   : p(u) = piecewise-linear(128 waypoints at u)

Metrics (identical code for both): dense collision via per-cell crossing between
consecutive dense samples, min clearance from the analytic occupancy distance,
endpoint error, RMSE against the reference curve, path length ratio, curvature,
smoothness, and the cost (NFE / wall time).
"""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only_points import load_point_baseline,pin_endpoints
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_sampler import sample_points
from experiments.safety_baselines.adapters.constraints import MapConstraint

SCENE_TO_METER=80.

def dense_from_curve(points_on_grid,occ,refine=3):
    """Densify a polyline until every grid crossing is represented (same rule as the
    repo's curve evaluator): split intervals that are long or near an obstacle."""
    H,W=occ.shape;field=MapConstraint(occ,extent=2.,radius=0.)
    p=np.asarray(points_on_grid,dtype=float)
    for _ in range(refine):
        h,_=field.query(p)
        near=(np.minimum(h[:-1],h[1:])<4/max(H,W))|(np.linalg.norm(np.diff(p,axis=0),axis=1)>1/max(H,W))
        if not near.any():break
        mid=(p[:-1][near]+p[1:][near])/2
        p=np.sort(np.r_[p,mid],axis=0) if False else np.vstack([p,mid])
        # keep order: rebuild by sorting on the parameter is not needed for the cell test,
        # but the length/curvature metrics are computed on the ORIGINAL grid below.
    return p

def evaluate_curve(points,occ,cond,gt_curve,u):
    p=np.asarray(points,dtype=float);occ=np.asarray(occ,dtype=bool);H,W=occ.shape
    field=MapConstraint(occ,extent=2.,radius=0.)
    # dense collision: per-cell crossing between consecutive samples of the SAME grid
    samples=[]
    for a,b in zip(p[:-1],p[1:]):
        delta=b-a;cuts=[0.,1.]
        for axis,size in [(0,W),(1,H)]:
            if abs(delta[axis])>1e-15:
                t=(np.linspace(-1,1,size+1)-a[axis])/delta[axis];cuts.extend(t[(t>0)&(t<1)])
        cuts=np.unique(cuts);t=np.unique(np.r_[cuts,(cuts[:-1]+cuts[1:])/2]);samples.append(a+t[:,None]*delta)
    dense=np.concatenate(samples)
    def hit(v):
        inside=((v>=-1)&(v<1)).all(1);cell=np.floor((v+1)*np.array([W,H])/2).astype(int);cell=np.clip(cell,[0,0],[W-1,H-1])
        return ~inside|occ[cell[:,1],cell[:,0]]
    h,_=field.query(p)
    step=np.diff(p,axis=0);seg=np.linalg.norm(step,axis=1);length=float(seg.sum())
    ref=float(np.linalg.norm(np.diff(gt_curve,axis=0),axis=1).sum())
    pu=p*SCENE_TO_METER;d1=np.gradient(pu,u,axis=0);d2=np.gradient(d1,u,axis=0)
    curv=np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/np.maximum(np.linalg.norm(d1,axis=1)**3,1e-12)
    acc=p[2:]-2*p[1:-1]+p[:-2]
    endpoint=float(max(np.linalg.norm(p[0]-cond[0]),np.linalg.norm(p[-1]-cond[1]))*SCENE_TO_METER)
    collision=bool(hit(dense).any())
    return {'curve_collision':collision,'curve_vertex_collision':bool(hit(p).any()),
        'planning_success':(not collision) and endpoint<=.01,'endpoint_error_m':endpoint,
        'min_clearance_m':float(h.min()*SCENE_TO_METER),'violation_max_m':float(np.maximum(-h,0).max()*SCENE_TO_METER),
        'rmse_m':float(np.sqrt(np.mean(np.sum((p-np.asarray(gt_curve))**2,axis=1)))*SCENE_TO_METER),
        'path_length_m':length*SCENE_TO_METER,'path_length_ratio':length/max(ref,1e-12),
        'max_curvature_1pm':float(curv.max()),'mean_curvature_1pm':float(curv.mean()),
        'mean_second_difference_m':float(np.linalg.norm(acc,axis=1).mean()*SCENE_TO_METER),
        'dense_points':int(len(dense)),'grid_points':int(len(p))}

def polyline_at_u(points,u):
    """Piecewise-linear curve through `points` sampled at parameters u in [0,1]."""
    s=np.linspace(0.,1.,len(points));out=np.empty((len(u),2))
    for d in range(2):out[:,d]=np.interp(u,s,points[:,d])
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sc48',default='experiments/safety_baselines/results/sc48_recovery/sc48_native16_s0.npz')
    ap.add_argument('--config',default='experiments/safety_baselines/configs/seven_native_points_vertexall_t256.json')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--ddim-steps',type=int,default=256)
    ap.add_argument('--out-dir',default='experiments/safety_baselines/results/sc48_recovery')
    a=ap.parse_args()
    out=ROOT/a.out_dir;out.mkdir(parents=True,exist_ok=True)
    z=np.load(ROOT/a.sc48,allow_pickle=True)
    meta=json.loads(str(z['meta']));u=z['u'];curves=z['curve'];curves128=z['curve128'];Q=z['Q'];ids=list(z['sample_id'])
    sc48_seconds=list(z['seconds']) if 'seconds' in z.files else [None]*len(ids)
    cfg=json.loads((ROOT/a.config).read_text(encoding='utf8'))
    model,ck=load_point_baseline(ROOT/cfg['checkpoint'],'cuda')
    d=ck['config']['diffusion'];sched=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=PointDataset('val',ck['config']['data']['processed_root'])
    sid=np.load(Path(ck['config']['data']['processed_root'])/'val'/'sample_id.npy')
    rows=[]
    for k,scene in enumerate(ids):
        index=int(np.where(sid==scene)[0][0])
        row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();gt=row['pos'].numpy()
        # reference curve: the GT polyline at the same u grid
        gt_curve=polyline_at_u(gt,u)
        # --- B: S-C48 (already sampled with the historical code) ---
        t0=time.perf_counter();m_b=evaluate_curve(curves[k],occ,cond,gt_curve,u);t_eval=time.perf_counter()-t0
        m_b.update(method='S-C48',scene=int(scene),index=index,steps=meta['steps'],nfe=meta['steps'],seconds=sc48_seconds[k],eval_seconds=t_eval,
            control_polygon_collision=bool(z['control_hit'][k]))
        m_128=evaluate_curve(curves128[k],occ,cond,polyline_at_u(gt,np.linspace(0,1,128)),np.linspace(0,1,128))
        m_b['curve128_vertex_collision']=m_128['curve_vertex_collision']
        rows.append(m_b)
        # --- A: map-only best.pt, DDIM on the full 256-level ladder, same torch seed ---
        torch.manual_seed(a.seed)
        init=torch.randn(1,128,2).numpy();init[0,[0,-1]]=cond
        torch.cuda.synchronize();t0=time.perf_counter()
        q=sample_points(model,sched,row['occupancy'][None].cuda(),row['cond'][None].cuda(),seed=a.seed,steps=a.ddim_steps,init=init)
        torch.cuda.synchronize();t_a=time.perf_counter()-t0
        poly=q[0].cpu().numpy();curve_a=polyline_at_u(poly,u)
        m_a=evaluate_curve(curve_a,occ,cond,gt_curve,u);m_a.update(method='map-only',scene=int(scene),index=index,
            steps=a.ddim_steps,nfe=a.ddim_steps,seconds=t_a,vertex_collision_polyline=bool(evaluate_curve(poly,occ,cond,polyline_at_u(gt,np.linspace(0,1,128)),np.linspace(0,1,128))['curve_vertex_collision']))
        rows.append(m_a)
        # figure
        fig,ax=plt.subplots(figsize=(6,6),dpi=120)
        ax.imshow(occ,cmap='gray_r',origin='lower',extent=[-1,1,-1,1],alpha=.85)
        ax.plot(gt[:,0],gt[:,1],'--',color='tab:gray',lw=1,label='GT')
        ax.plot(cond[:,0],cond[:,1],'o',color='tab:green',ms=6,label='start/goal')
        ax.plot(curves[k][:,0],curves[k][:,1],'-',color='tab:red',lw=1.8,label='S-C48 (16-level DDIM)')
        ax.plot(curve_a[:,0],curve_a[:,1],'-',color='tab:blue',lw=1.4,label=f'map-only (DDIM-{a.ddim_steps})')
        ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'scene {int(scene)}',fontsize=9);ax.legend(fontsize=6,loc='lower right')
        fig.tight_layout();fig.savefig(out/f'scene_{int(scene)}_trajectories.png');plt.close(fig)
        print(int(scene),'S-C48 path',m_b['curve_collision'],'clear %.1f'%m_b['min_clearance_m'],
              '| map-only path',m_a['curve_collision'],'clear %.1f'%m_a['min_clearance_m'],flush=True)
    with (out/'unified_eval.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    agg={}
    for method in ('S-C48','map-only'):
        rr=[r for r in rows if r['method']==method];n=len(rr)
        agg[method]={'n':n,'path_collisions':int(sum(r['curve_collision'] for r in rr)),
            'vertex_collisions_128':int(sum(r.get('curve128_vertex_collision',r.get('vertex_collision_polyline',False)) for r in rr)),
            'successes':int(sum(r['planning_success'] for r in rr)),
            'min_clearance_m_mean':float(np.mean([r['min_clearance_m'] for r in rr])),
            'violation_max_m':float(np.max([r['violation_max_m'] for r in rr])),
            'rmse_m_mean':float(np.mean([r['rmse_m'] for r in rr])),
            'path_length_ratio_mean':float(np.mean([r['path_length_ratio'] for r in rr])),
            'max_curvature_mean':float(np.mean([r['max_curvature_1pm'] for r in rr])),
            'mean_second_diff_m_mean':float(np.mean([r['mean_second_difference_m'] for r in rr])),
            'nfe':int(rr[0]['nfe']),'seconds_mean':float(np.mean([r['seconds'] for r in rr]))}
    (out/'unified_eval_summary.json').write_text(json.dumps({'sc48_meta':meta,'aggregate':agg,'rows':rows},indent=2),encoding='utf8')
    print(json.dumps(agg,indent=2))

if __name__=='__main__':main()
