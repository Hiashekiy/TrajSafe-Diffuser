"""Audit of the PDM App.C.3 segment constraint set and its projection on real maps.

For a fixed validation index this records: the audited occupancy -> circle
composition, the GT path residual, and the cost/feasibility of projecting
perturbed GT paths at several displacement scales. No model, no paper loop: this
isolates the geometry and the numerical projection.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.safety_baselines.adapters.paper_segment_constraints import CircleSegmentConstraints,obstacle_circles
from experiments.safety_baselines.algorithms.solvers import Budget,SolverFailure

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='data/processed/160m-k4p-c48')
    ap.add_argument('--split',default='val');ap.add_argument('--indices',default='0,81,163,244,326')
    ap.add_argument('--scales',default='0.02,0.05,0.1,0.2');ap.add_argument('--maxiter',type=int,default=20)
    ap.add_argument('--out',default='experiments/safety_baselines/docs/segment_projection_audit.json')
    a=ap.parse_args();root=Path(a.root)
    occ=np.load(root/a.split/'occupancy.npy',mmap_mode='r');cond=np.load(root/a.split/'conditions.npy',mmap_mode='r')
    gt=np.load(root/a.split/'curve_gt.npy',mmap_mode='r')
    rows=[]
    for index in [int(v) for v in a.indices.split(',')]:
        m=np.asarray(occ[index]).astype(bool);m=m[0] if m.ndim==3 else m
        centers,radii,audit=obstacle_circles(m,2.)
        seg=CircleSegmentConstraints(cond[index],len(gt[index]),centers,radii)
        x0=np.asarray(gt[index])[1:-1].reshape(-1)
        base={'index':index,'circles':len(centers),'audit':audit,'gt_residual_scene':seg.residual(x0),
              'gt_residual_m':seg.residual(x0)*80,'vertices':int(len(gt[index]))}
        rows.append(base)
        rng=np.random.default_rng(0)
        for scale in [float(v) for v in a.scales.split(',')]:
            target=x0+rng.normal(size=x0.shape)*scale
            t0=time.perf_counter()
            try:
                xx,info=seg.project(target,Budget(120),maxiter=a.maxiter)
                rows.append({**base,'scale_scene':scale,'scale_m':scale*80,'status':info['status'],
                    'residual_scene':info['residual'],'seconds':time.perf_counter()-t0,
                    'iterations':info['iterations'],'displacement_scene':float(np.abs(xx-target).max()),
                    'objective':float(.5*np.sum((xx-target)**2))})
            except SolverFailure as exc:
                rows.append({**base,'scale_scene':scale,'scale_m':scale*80,'status':'solver_failure','reason':str(exc)[:200],'seconds':time.perf_counter()-t0})
        print(index,'circles',len(centers),'gt_residual_m',round(base['gt_residual_m'],3),flush=True)
    Path(a.out).write_text(json.dumps({'rows':rows},indent=2),encoding='utf8')
    ok=[r for r in rows if 'scale_scene' in r]
    print('projections',len(ok),'feasible',sum(r['status']!='solver_failure' for r in ok),
        'mean_s',round(np.mean([r['seconds'] for r in ok]),3) if ok else None,flush=True)
    print('wrote',a.out,flush=True)

if __name__=='__main__':main()
