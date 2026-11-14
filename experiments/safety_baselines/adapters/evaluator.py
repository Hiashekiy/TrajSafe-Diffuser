"""Common occupancy evaluator independent of every optimizer (all nine methods).
512 base parameters, then near-obstacle and long-segment midpoint refinements.
These are discrete checks and do not certify continuous safety.
"""
import numpy as np
from experiments.safety_baselines.adapters.constraints import MapConstraint

def evaluate_curve(codec,Q,occ,cond,gt,endpoint_tol_m=.01,radius_scene=0.,initial_points=512,parameters=None):
    Q=np.asarray(Q,dtype=float)
    if not np.isfinite(Q).all():raise ValueError('nonfinite_trajectory')
    field=MapConstraint(occ,extent=2.,radius=radius_scene);H,W=occ.shape
    u=np.linspace(0,1,initial_points) if parameters is None else np.asarray(parameters);base_u=np.linspace(0,1,initial_points)
    def decode(v):return codec.basis_at(v).detach().cpu().numpy()@Q
    for _ in range(3 if parameters is None else 0):
        p=decode(u);h,_=field.query(p)
        near=(np.minimum(h[:-1],h[1:])<4/max(H,W))|(np.linalg.norm(np.diff(p,axis=0),axis=1)>1/max(H,W))
        if not near.any():break
        u=np.sort(np.r_[u,(u[:-1][near]+u[1:][near])/2])
    p=decode(u);h,_=field.query(p)
    inside=((p>=-1)&(p<1)).all(1);cell=np.zeros((len(p),2),dtype=int)
    cell[inside]=np.floor((p[inside]+1)*np.array([W,H])/2).astype(int)
    cell=np.clip(cell,[0,0],[W-1,H-1])
    hit=np.ones(len(p),dtype=bool);hit[inside]=np.asarray(occ,dtype=bool)[cell[inside,1],cell[inside,0]]
    collision=bool(hit.any())
    endpoint=float(max(np.linalg.norm(p[0]-cond[0]),np.linalg.norm(p[-1]-cond[-1]))*80)
    arrived=endpoint<=endpoint_tol_m;success=arrived and not collision
    violation=np.maximum(-h,0)*80
    length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*80);direct=np.linalg.norm(cond[1]-cond[0])*80
    # Geometry on uniform 512 points, derivatives in physical metres.
    pu=decode(base_u)*80;d1=np.gradient(pu,base_u,axis=0);d2=np.gradient(d1,base_u,axis=0)
    curvature=np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/np.maximum(np.linalg.norm(d1,axis=1)**3,1e-12)
    pred=codec.basis.detach().cpu().numpy().astype(float)@Q
    rmse=float(np.sqrt(np.mean(np.sum((pred-np.asarray(gt))**2,axis=1)))*80)
    if not np.isfinite(rmse) or not np.isfinite(violation).all() or not np.isfinite(curvature).all():raise ValueError('nonfinite_evaluation')
    return {'collision':collision,'common_sdf_feasible':bool((h>=-1e-8).all()),'arrived':arrived,'planning_success':success,'endpoint_error_m':endpoint,'common_sdf_violation_max_m':float(violation.max()),'common_sdf_violation_positive_mean_m':float(violation[violation>0].mean()) if (violation>0).any() else 0.,'common_sdf_min_m':float(h.min()*80),'clearance_success_m':float(h.min()*80) if success else None,'detour_success':float(length/direct) if success and direct>1e-9 else None,'max_curvature_success':float(curvature.max()) if success else None,'mean_square_curvature_success':float(np.mean(curvature**2)) if success else None,'rmse_m':rmse,'nrmse':rmse/160,'evaluation_points':len(u),'continuous_certified':False,'_parameters':u.tolist()}
