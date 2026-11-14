"""Reuse the old occupancy SDF, not the old correction algorithms.

Differentiate its scalar interpolant exactly for the new QP/projection methods.
The old interpolated unit direction is a guidance direction, not always the
derivative of that interpolant, so it must not be used for KKT certificates.
"""
import numpy as np
import torch
from src.geometry.sdf_field import build_occupancy_sdf

class LegacyConstraintField:
    def __init__(self, occupancy, radius=0., device='cuda',gradient_mode='scalar'):
        self.occ=np.asarray(occupancy,dtype=bool);self.radius=float(radius)
        self.gradient_mode=gradient_mode
        occ=torch.as_tensor(self.occ,device=device,dtype=torch.float64)[None,None]
        self.field=build_occupancy_sdf(occ,scene_to_meter=80.,map_margin=0.,dtype=torch.float64)

    def query(self, points):
        if self.gradient_mode=='legacy_direction':
            p=np.asarray(points,dtype=float);x=torch.as_tensor(p.reshape(1,-1,2),device=self.field.device,dtype=torch.float64)
            with torch.no_grad():h,direction=self.field.query(x,with_boundary=True,with_grad=True)
            return h.cpu().numpy().reshape(p.shape[:-1])-self.radius,direction.cpu().numpy().reshape(p.shape)
        return self.scalar_query(points)

    def scalar_query(self, points):
        p=np.asarray(points,dtype=float);shape=p.shape
        with torch.enable_grad():
            x=torch.tensor(p.reshape(1,-1,2),device=self.field.device,dtype=torch.float64,requires_grad=True)
            h=self.field.query(x,with_boundary=True,with_grad=False)-self.radius
            grad=torch.autograd.grad(h.sum(),x)[0]
        return h.detach().cpu().numpy().reshape(shape[:-1]),grad.detach().cpu().numpy().reshape(shape)
