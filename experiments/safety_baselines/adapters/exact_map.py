"""Analytic signed distance to the union of free occupancy cells.
Same occupied-cell set, no learned geometry/GT or extra margin. Unlike a bilinear
EDT, the gradient does not vanish on artificial medial-axis pixel plateaus.
Distance is nonsmooth only at true nearest-feature ties/corners; no C1 guarantee.
"""
import numpy as np
import shapely
class ExactMapConstraint:
 def __init__(self,occupancy,extent=2.,radius=0.):
  self.occ=np.asarray(occupancy,dtype=bool);self.extent=float(extent);self.radius=radius
  H,W=self.occ.shape;free=~self.occ;edges=[];normals=[];dx=extent/W;dy=extent/H;a=-extent/2
  def edge(x0,y0,x1,y1,n):edges.append([[x0,y0],[x1,y1]]);normals.append(n)
  for y,x in np.argwhere(free[:,:-1]!=free[:,1:]):edge(a+(x+1)*dx,a+y*dy,a+(x+1)*dx,a+(y+1)*dy,[-1,0] if free[y,x] else [1,0])
  for y,x in np.argwhere(free[:-1]!=free[1:]):edge(a+x*dx,a+(y+1)*dy,a+(x+1)*dx,a+(y+1)*dy,[0,-1] if free[y,x] else [0,1])
  for y in np.where(free[:,0])[0]:edge(a,a+y*dy,a,a+(y+1)*dy,[1,0])
  for y in np.where(free[:,-1])[0]:edge(-a,a+y*dy,-a,a+(y+1)*dy,[-1,0])
  for x in np.where(free[0])[0]:edge(a+x*dx,a,a+(x+1)*dx,a,[0,1])
  for x in np.where(free[-1])[0]:edge(a+x*dx,-a,a+(x+1)*dx,-a,[0,-1])
  self.edges=np.asarray(edges,dtype=float);self.normals=np.asarray(normals,dtype=float)
  self.tree=shapely.STRtree(shapely.linestrings(self.edges)) if len(edges) else None
 def query(self,points):
  p=np.asarray(points,dtype=float);shape=p.shape[:-1];v=p.reshape(-1,2)
  if not np.isfinite(v).all():raise ValueError('nonfinite_points')
  if self.tree is None:return np.full(shape,-self.extent),np.zeros_like(p)
  idx=self.tree.nearest(shapely.points(v));e=self.edges[idx];d=e[:,1]-e[:,0]
  t=np.clip(np.sum((v-e[:,0])*d,axis=1)/np.sum(d*d,axis=1),0,1);closest=e[:,0]+t[:,None]*d
  delta=v-closest;dist=np.linalg.norm(delta,axis=1);H,W=self.occ.shape
  inside=((v>=-self.extent/2)&(v<self.extent/2)).all(1);cell=np.zeros((len(v),2),dtype=int)
  cell[inside]=np.floor((v[inside]+self.extent/2)/self.extent*np.array([W,H])).astype(int)
  cell=np.clip(cell,[0,0],[W-1,H-1])
  isfree=inside.copy();isfree[inside]=~self.occ[cell[inside,1],cell[inside,0]]
  sign=np.where(isfree,1.,-1.);grad=sign[:,None]*delta/np.maximum(dist[:,None],1e-15)
  on=dist<1e-12;grad[on]=self.normals[idx[on]]
  return (sign*dist-self.radius).reshape(shape),grad.reshape(p.shape)
