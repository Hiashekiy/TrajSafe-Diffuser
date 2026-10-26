import numpy as np
from scripts.geometry_audit_utils import audit_polygon

def test_polygon_interior_and_boundary_crossing_are_checked():
    square=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    r=audit_polygon(square,np.array([[0.,0.]]),.1)
    assert r['colliding_cells']==1 and abs(r['overlap_area_scene2']-.04)<1e-10
    assert audit_polygon(square,np.array([[.6,0.]]),.1)['colliding_cells']==0
    assert audit_polygon(square,np.array([[.55,0.]]),.1)['colliding_cells']==1
    assert audit_polygon(square+2,np.zeros((0,2)),.1)['outside']

def test_sat_matches_independent_geos_for_randomized_convex_regions():
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon, box
    rng=np.random.default_rng(20261008)
    grid=np.array([(x,y) for x in np.arange(-.8,.9,.2) for y in np.arange(-.8,.9,.2)])
    for _ in range(25):
        points=rng.uniform(-.8,.8,size=(10,2))
        poly=points[ConvexHull(points).vertices]
        obstacles=grid[rng.choice(len(grid),12,replace=False)]
        result=audit_polygon(poly,obstacles,.1)
        geometry=Polygon(poly)
        areas=[geometry.intersection(box(c[0]-.1,c[1]-.1,c[0]+.1,c[1]+.1)).area for c in obstacles]
        assert result['colliding_cells']==sum(a>1e-12 for a in areas)
        assert abs(result['overlap_area_scene2']-sum(areas))<1e-10
