"""Native point state constrained on whole polyline segments (PDM App. C.3).

`NativePointTask` projects only the vertices onto free cells. PDM's task
constraint set is "no segment intersects an obstacle circle", so this subclass
replaces the constraint object (and the projection) while keeping the identical
learned point prior, endpoints, score and output representation.
"""
import numpy as np
from experiments.safety_baselines.adapters.native_points import NativePointTask
from experiments.safety_baselines.adapters.paper_segment_constraints import CircleSegmentConstraints,obstacle_circles

class SegmentPointTask(NativePointTask):
    constraint_object='whole_polyline_segments_vs_circle_union'
    partial_projection=True
    projection_maxiter=8
    projection_maxiter_final=40
    projection_tol=1e-6
    def __init__(self,model,schedule,occupancy,cond,points=128,radius=0.,field_kind='exact',margin=0.,circles=None):
        super().__init__(model,schedule,occupancy,cond,points,radius,field_kind)
        if circles is None:
            centers,radii,audit=obstacle_circles(occupancy,extent=2.,margin=margin)
        else:
            centers,radii=circles;audit={'geometry':'supplied','circles':int(len(centers))}
        self.circles=(np.asarray(centers,dtype=float),np.asarray(radii,dtype=float))
        self.segment_map=CircleSegmentConstraints(self.endpoints,self.H,centers,radii)
        self.segment_audit=dict(audit,margin_scene=float(margin))
        # The pointwise separable-QP shortcut is only valid for per-vertex rows.
        self.cbf_solve=None
    def constraints(self,x):return self.segment_map.constraints(x)
    def project(self,target,budget,allow_partial=False,**kw):
        return self.segment_map.project(target,budget,
            maxiter=self.projection_maxiter if allow_partial else self.projection_maxiter_final,
            tol=self.projection_tol,allow_partial=allow_partial,**kw)
    def point_residual(self,x):return float(self.map.query(self.unpack(x))[0].min())
