"""Differentiable cell-centre SDF in metres, row grows with +y.
Map covers [-extent/2,extent/2]^2. Already-inflated grids use radius=0.
Discrete SDF constraints provide no continuous-safety certificate.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

class MapConstraint:
    def __init__(self, occupancy, extent=160., radius=0.):
        self.occ=np.asarray(occupancy,dtype=bool)
        if self.occ.ndim!=2 or min(self.occ.shape)<2 or extent<=0 or radius<0:
            raise ValueError('invalid map')
        self.extent=float(extent);self.radius=float(radius)
        H,W=self.occ.shape;self.spacing=np.array([extent/W,extent/H])
        # Pad by obstacles: map exterior is unsafe even for empty maps.
        padded=np.pad(~self.occ,1,constant_values=False)
        inside=distance_transform_edt(padded,sampling=(extent/H,extent/W))[1:-1,1:-1]
        outside=distance_transform_edt(self.occ,sampling=(extent/H,extent/W))
        self.field=np.where(self.occ,-outside,inside)
        self.field-=np.sign(self.field)*min(self.spacing)/2
        if self.occ.all():self.field[:]=-extent

    def query(self, points):
        p=np.asarray(points,dtype=float)
        if p.shape[-1]!=2 or not np.isfinite(p).all():raise ValueError('invalid points')
        H,W=self.occ.shape
        uv=(p+self.extent/2)/self.spacing-.5
        clipped=np.clip(uv,[0,0],[W-1,H-1])
        ij=np.minimum(np.floor(clipped).astype(int),[W-2,H-2])
        f=clipped-ij;x,y=ij[...,0],ij[...,1];a,b=f[...,0],f[...,1]
        z00=self.field[y,x];z10=self.field[y,x+1];z01=self.field[y+1,x];z11=self.field[y+1,x+1]
        val=(1-a)*(1-b)*z00+a*(1-b)*z10+(1-a)*b*z01+a*b*z11
        grad=np.stack([(1-b)*(z10-z00)+b*(z11-z01),(1-a)*(z01-z00)+a*(z11-z10)],axis=-1)/self.spacing
        grad*=((uv>=0)&(uv<=np.array([W-1,H-1])))
        margins=self.extent/2-np.abs(p);axis=np.argmin(margins,axis=-1)
        box=np.min(margins,axis=-1);boxgrad=np.zeros_like(p)
        np.put_along_axis(boxgrad,axis[...,None],-np.take_along_axis(np.sign(p),axis[...,None],axis=-1),axis=-1)
        use=box<val
        return np.minimum(val,box)-self.radius,np.where(use[...,None],boxgrad,grad)

    def control_constraints(self, controls, basis):
        Q=np.asarray(controls);B=np.asarray(basis)
        h,grad=self.query(B@Q)
        jac=-B[:,:,None]*grad[:,None,:]
        return -h,jac.reshape(len(B),-1)
