"""Independent convex-polygon versus occupied grid-cell safety audit."""
import numpy as np
from src.geometry.safety_corridor import convex_polygon_intersection, polygon_area


def obstacle_boxes(occupancy):
    y, x = np.nonzero(np.asarray(occupancy) > .5)
    size = 2.0 / occupancy.shape[-1]
    return np.stack((-1+(x+.5)*size, -1+(y+.5)*size), axis=1), size/2


def audit_polygon(poly, boxes, half, tol=1e-9):
    if poly is None or len(poly)<3:
        return {'invalid':True, 'colliding_cells':0, 'overlap_area_scene2':0., 'outside':False}
    poly=np.asarray(poly, dtype=np.float64)
    outside=bool(np.any(np.abs(poly)>1+tol))
    low,high=poly.min(0),poly.max(0)
    nearby=boxes[np.all(boxes+half>low+tol,axis=1)&np.all(boxes-half<high-tol,axis=1)]
    if not len(nearby):
        return {'invalid':False,'colliding_cells':0,'overlap_area_scene2':0.,'outside':outside}
    edge=np.roll(poly,-1,axis=0)-poly
    normals=np.stack((-edge[:,1],edge[:,0]),axis=1)
    normals=normals[np.linalg.norm(normals,axis=1)>1e-12]
    normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    axes=np.concatenate((np.eye(2),normals))
    proj=poly@axes.T
    center=nearby@axes.T
    rad=half*np.abs(axes).sum(1)
    overlap=np.minimum(proj.max(0),center+rad)-np.maximum(proj.min(0),center-rad)
    hit=nearby[np.all(overlap>tol,axis=1)]
    area=0.
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
    for c in hit:
        area+=polygon_area(convex_polygon_intersection(poly,c+corners))
    return {'invalid':False,'colliding_cells':len(hit),'overlap_area_scene2':area,'outside':outside}
