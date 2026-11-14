"""Evaluate the delivered polyline itself, including all grid-crossed cells.

All methods are scored by this one evaluator (no method-specific metric). The
polyline is the object that would be executed, so it is what is measured:
dense per-cell sampling for collisions, analytic signed distance to the free
boundary for clearance, and second-difference/turning terms for smoothness.
"""
import numpy as np
from experiments.safety_baselines.adapters.constraints import MapConstraint

SCENE_TO_METER=80.

def _dense_samples(p,occ):
    H,W=occ.shape;samples=[]
    for a,b in zip(p[:-1],p[1:]):
        delta=b-a;cuts=[0.,1.]
        for axis,size in [(0,W),(1,H)]:
            if abs(delta[axis])>1e-15:
                t=(np.linspace(-1,1,size+1)-a[axis])/delta[axis];cuts.extend(t[(t>0)&(t<1)])
        cuts=np.unique(cuts);t=np.unique(np.r_[cuts,(cuts[:-1]+cuts[1:])/2]);samples.append(a+t[:,None]*delta)
    return np.concatenate(samples)

def evaluate_points(points,occ,cond,gt):
    p=np.asarray(points,dtype=float);occ=np.asarray(occ,dtype=bool);H,W=occ.shape
    if p.shape!=np.asarray(gt).shape or not np.isfinite(p).all():raise ValueError('invalid native point output')
    def hit(v):
        inside=((v>=-1)&(v<1)).all(1);cell=np.floor((v+1)*np.array([W,H])/2).astype(int);cell=np.clip(cell,[0,0],[W-1,H-1])
        return ~inside|occ[cell[:,1],cell[:,0]]
    dense=_dense_samples(p,occ)
    vertex_hit=hit(p);collision=bool(hit(dense).any())
    sdf=MapConstraint(occ,extent=2.,radius=0.);h,_=sdf.query(dense)
    endpoint=float(np.linalg.norm(p[[0,-1]]-np.asarray(cond),axis=1).max()*SCENE_TO_METER)
    step=np.diff(p,axis=0);seg=np.linalg.norm(step,axis=1);length=float(seg.sum())
    ref_length=float(np.linalg.norm(np.diff(np.asarray(gt,dtype=float),axis=0),axis=1).sum())
    unit=step/np.maximum(seg[:,None],1e-12)
    turn=np.degrees(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1.,1.))) if len(unit)>1 else np.zeros(1)
    acc=p[2:]-2*p[1:-1]+p[:-2]
    jerk=p[3:]-3*p[2:-1]+3*p[1:-2]-p[:-3]
    return {
        'collision':collision,'vertex_collision':bool(vertex_hit.any()),
        'segment_only_collision':collision and not vertex_hit.any(),
        'planning_success':not collision and endpoint<=.01,
        'endpoint_error_m':endpoint,
        'rmse_m':float(np.sqrt(np.mean(np.sum((p-np.asarray(gt))**2,axis=1)))*SCENE_TO_METER),
        'violation_max_m':float(np.maximum(-h,0).max()*SCENE_TO_METER),
        'min_clearance_m':float(h.min()*SCENE_TO_METER),
        'evaluation_points':int(len(dense)),
        'path_length_m':length*SCENE_TO_METER,
        'path_length_ratio':length/max(ref_length,1e-12),
        'mean_turning_deg':float(turn.mean()) if len(unit)>1 else 0.,
        'max_turning_deg':float(turn.max()) if len(unit)>1 else 0.,
        'mean_second_difference_m':float(np.linalg.norm(acc,axis=1).mean()*SCENE_TO_METER),
        'max_second_difference_m':float(np.linalg.norm(acc,axis=1).max()*SCENE_TO_METER) if len(acc) else 0.,
        'mean_jerk_m':float(np.linalg.norm(jerk,axis=1).mean()*SCENE_TO_METER) if len(jerk) else 0.,
        'representation':'native_polyline_no_fit','continuous_curve_certificate':False}
