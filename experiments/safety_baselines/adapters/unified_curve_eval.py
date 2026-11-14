"""One curve evaluator for every representation (Phase F / S-C48 comparison).

The two base models deliver different objects -- 48 B-spline control points (the plan
is the decoded curve) and 128 waypoints (the plan is the polyline) -- so the only
comparable quantity is the curve itself. This module evaluates a curve given as a
sequence of points on a shared parameter grid `u`, and every caller passes its curve
sampled on that same grid:

    S-C48     : p(u) = basis_at(u) @ Q
    map-only  : p(u) = piecewise-linear interpolation of the 128 waypoints

It is the rule set used by `adapters/evaluator.py:evaluate_curve` (512 base grid plus
near-obstacle refinement) and additionally reports the smoothness terms.
"""
import numpy as np
from experiments.safety_baselines.adapters.constraints import MapConstraint

SCENE_TO_METER=80.

def polyline_at_u(points,u):
    """Piecewise-linear curve through `points`, sampled at parameters u in [0,1]."""
    points=np.asarray(points,dtype=float);s=np.linspace(0.,1.,len(points))
    return np.stack([np.interp(u,s,points[:,k]) for k in (0,1)],axis=1)

def _cell_hit(values,occ):
    H,W=occ.shape
    inside=((values>=-1)&(values<1)).all(1)
    cell=np.floor((values+1)*np.array([W,H])/2).astype(int)
    cell=np.clip(cell,[0,0],[W-1,H-1])
    return ~inside|occ[cell[:,1],cell[:,0]]

def dense_crossings(p,occ):
    """Every grid crossing of the polyline p, plus interval midpoints."""
    H,W=occ.shape;samples=[]
    for a,b in zip(p[:-1],p[1:]):
        delta=b-a;cuts=[0.,1.]
        for axis,size in [(0,W),(1,H)]:
            if abs(delta[axis])>1e-15:
                t=(np.linspace(-1,1,size+1)-a[axis])/delta[axis];cuts.extend(t[(t>0)&(t<1)])
        cuts=np.unique(cuts)
        t=np.unique(np.r_[cuts,(cuts[:-1]+cuts[1:])/2])
        samples.append(a+t[:,None]*delta)
    return np.concatenate(samples)

def evaluate_curve(points,occ,cond,gt_curve,u):
    """Collision / clearance / shape metrics of one curve on the shared grid `u`."""
    p=np.asarray(points,dtype=float);occ=np.asarray(occ,dtype=bool)
    if not np.isfinite(p).all():raise ValueError('nonfinite_curve')
    field=MapConstraint(occ,extent=2.,radius=0.)
    dense=dense_crossings(p,occ)
    h,_=field.query(p)
    step=np.diff(p,axis=0);seg=np.linalg.norm(step,axis=1);length=float(seg.sum())
    ref=float(np.linalg.norm(np.diff(gt_curve,axis=0),axis=1).sum())
    pu=p*SCENE_TO_METER;d1=np.gradient(pu,u,axis=0);d2=np.gradient(d1,u,axis=0)
    curv=np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/np.maximum(np.linalg.norm(d1,axis=1)**3,1e-12)
    ustep=step/np.maximum(seg[:,None],1e-12)
    turn=np.degrees(np.arccos(np.clip(np.sum(ustep[:-1]*ustep[1:],axis=1),-1.,1.))) if len(ustep)>1 else np.zeros(1)
    acc=p[2:]-2*p[1:-1]+p[:-2]
    collision=bool(_cell_hit(dense,occ).any())
    endpoint=float(max(np.linalg.norm(p[0]-cond[0]),np.linalg.norm(p[-1]-cond[1]))*SCENE_TO_METER)
    return {'curve_collision':collision,'curve_vertex_collision':bool(_cell_hit(p,occ).any()),
        'planning_success':(not collision) and endpoint<=.01,'endpoint_error_m':endpoint,
        'min_clearance_m':float(h.min()*SCENE_TO_METER),
        'violation_max_m':float(np.maximum(-h,0).max()*SCENE_TO_METER),
        'rmse_m':float(np.sqrt(np.mean(np.sum((p-np.asarray(gt_curve))**2,axis=1)))*SCENE_TO_METER),
        'path_length_m':length*SCENE_TO_METER,'path_length_ratio':length/max(ref,1e-12),
        'max_curvature_1pm':float(curv.max()),'mean_curvature_1pm':float(curv.mean()),
        'max_turning_deg':float(turn.max()),'mean_turning_deg':float(turn.mean()),
        'mean_second_difference_m':float(np.linalg.norm(acc,axis=1).mean()*SCENE_TO_METER) if len(acc) else 0.,
        'dense_points':int(len(dense)),'grid_points':int(len(p)),'grid_u':int(len(u))}
