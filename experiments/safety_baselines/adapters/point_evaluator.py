"""Evaluate the delivered polyline itself, including all grid-crossed cells."""
import numpy as np
from experiments.safety_baselines.adapters.constraints import MapConstraint

def evaluate_points(points,occ,cond,gt):
    p=np.asarray(points,dtype=float);occ=np.asarray(occ,dtype=bool);H,W=occ.shape
    if p.shape!=np.asarray(gt).shape or not np.isfinite(p).all():raise ValueError('invalid native point output')
    def hit(v):
        inside=((v>=-1)&(v<1)).all(1);cell=np.floor((v+1)*np.array([W,H])/2).astype(int);cell=np.clip(cell,[0,0],[W-1,H-1])
        return ~inside|occ[cell[:,1],cell[:,0]]
    # Intersections partition each segment into intervals lying in single cells.
    samples=[]
    for a,b in zip(p[:-1],p[1:]):
        delta=b-a;cuts=[0.,1.]
        for axis,size in [(0,W),(1,H)]:
            if abs(delta[axis])>1e-15:
                t=(np.linspace(-1,1,size+1)-a[axis])/delta[axis];cuts.extend(t[(t>0)&(t<1)])
        cuts=np.unique(cuts);t=np.unique(np.r_[cuts,(cuts[:-1]+cuts[1:])/2]);samples.append(a+t[:,None]*delta)
    dense=np.concatenate(samples);vertex_hit=hit(p);collision=bool(hit(dense).any())
    sdf=MapConstraint(occ,extent=2.,radius=0.);h,_=sdf.query(dense)
    endpoint=float(np.linalg.norm(p[[0,-1]]-np.asarray(cond),axis=1).max()*80)
    # Geometry of the delivered polyline itself: a sampler that ends on a noisy
    # marginal instead of the clean prediction returns a sawtooth (length ratio
    # >> 1, mean turning angle >> 0) whose vertices can still be collision free.
    step=np.diff(p,axis=0);length=float(np.linalg.norm(step,axis=1).sum());ref_length=float(np.linalg.norm(np.diff(np.asarray(gt,dtype=float),axis=0),axis=1).sum())
    unit=step/np.maximum(np.linalg.norm(step,axis=1,keepdims=True),1e-12)
    turning=float(np.degrees(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1.,1.))).mean()) if len(unit)>1 else 0.
    return {'collision':collision,'vertex_collision':bool(vertex_hit.any()),'segment_only_collision':collision and not vertex_hit.any(),'planning_success':not collision and endpoint<=.01,'endpoint_error_m':endpoint,'rmse_m':float(np.sqrt(np.mean(np.sum((p-np.asarray(gt))**2,axis=1)))*80),'violation_max_m':float(np.maximum(-h,0).max()*80),'evaluation_points':len(dense),'path_length_m':length*80,'path_length_ratio':length/max(ref_length,1e-12),'mean_turning_deg':turning,'representation':'native_polyline_no_fit','continuous_curve_certificate':False}
