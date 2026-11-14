"""Whole-polyline (PDM App. C.3) constraint geometry, projection and PDM loop.

The PDM loop is additionally checked against a distribution with a known
analytic answer (Gaussian prior, half-space constraint) so that "the equations
are transcribed" is separated from "the annealing + projection actually targets
the constrained distribution".
"""
import numpy as np
import pytest
from experiments.safety_baselines.adapters.paper_segment_constraints import CircleSegmentConstraints,obstacle_circles
from experiments.safety_baselines.algorithms.solvers import Budget,SolverFailure
from experiments.safety_baselines.algorithms.pdm.runner import level_gamma,run as pdm_run

def blocked_map(H=32,rows=slice(12,20),cols=slice(12,20)):
    occ=np.zeros((H,H),bool);occ[rows,cols]=True;return occ

def test_circle_cover_contains_every_occupied_cell_and_keeps_free_space():
    occ=blocked_map()
    c,r,audit=obstacle_circles(occ,2.)
    assert audit['covered_fraction']==1.
    ys,xs=np.nonzero(occ)
    cy=-1+(ys+.5)*2./occ.shape[0];cx=-1+(xs+.5)*2./occ.shape[1]
    d=np.linalg.norm(np.stack([cx,cy],1)[:,None]-c[None],axis=2)
    assert (d<=r[None]+1e-12).any(1).all()
    assert audit['free_cells_forbidden_fraction']<0.10

def test_single_occupied_cell_is_one_circle():
    occ=np.zeros((16,16),bool);occ[8,8]=True
    c,r,audit=obstacle_circles(occ,2.)
    assert len(c)==1 and audit['radius_min_scene']>=2./16-1e-12

def test_paper_segment_set_is_stricter_than_vertex_free_cells():
    occ=blocked_map();ends=np.array([[-.8,0.],[.8,0.]])
    centers,radii,_=obstacle_circles(occ,2.)
    seg=CircleSegmentConstraints(ends,2,centers,radii)
    # both endpoints are in free cells, the straight segment crosses the block
    assert not occ[8,0] and not occ[8,31]
    assert seg.residual(np.empty(0))>0

def dense_min_distance(p,centers):
    d=np.inf
    for a,b in zip(p[:-1],p[1:]):
        t=np.linspace(0,1,2001)[:,None];dense=a+t*(b-a)
        d=min(d,float(np.linalg.norm(dense[:,None]-centers[None],axis=2).min()))
    return d

def test_projection_is_identity_when_feasible():
    occ=np.zeros((32,32),bool);occ[12:20,12:20]=True
    centers,radii,_=obstacle_circles(occ,2.)
    detour=np.array([[-.8,0.],[-.2,-.45],[.2,-.45],[.8,0.]])  # clears the block
    seg=CircleSegmentConstraints(detour[[0,-1]],4,centers,radii)
    x=detour[1:-1].reshape(-1)
    assert seg.residual(x)<=0
    out,info=seg.project(x,Budget(10))
    assert info['status']=='identity_feasible' and np.array_equal(out,x)

def test_projection_repairs_a_clipped_segment_not_only_its_vertices():
    occ=np.zeros((32,32),bool);occ[12:20,12:20]=True
    centers,radii,_=obstacle_circles(occ,2.)
    ends=np.array([[-.8,.2],[.8,.2]])          # straight path clips the block corner
    seg=CircleSegmentConstraints(ends,4,centers,radii)
    target=np.array([[-.25,.2],[.25,.2]]).reshape(-1)
    assert seg.residual(target)>0
    out,info=seg.project(target,Budget(30),maxiter=80)
    assert seg.residual(out)<=1e-6
    p=seg.unpack(out)
    np.testing.assert_array_equal(p[[0,-1]],ends)
    assert dense_min_distance(p,centers)>=radii.min()-1e-6

def test_projection_reports_failure_instead_of_clipping_to_vertices():
    occ=blocked_map();centers,radii,_=obstacle_circles(occ,2.)
    seg=CircleSegmentConstraints(np.array([[-.9,0.],[.9,0.]]),128,centers,radii)
    with pytest.raises(SolverFailure):
        seg.project(np.zeros(seg.D),Budget(5),maxiter=1)

def test_pdm_step_conventions_and_mixing():
    variance=np.array([0.012,0.04,0.3,2.,1490.])
    levels=np.array([4,3,2,1,0])
    paper=level_gamma(variance,0,levels,'paper',2e-3)
    ald=level_gamma(variance,0,levels,'official_ald',2e-3)
    np.testing.assert_allclose(paper,0.012/(2*1490.))
    np.testing.assert_allclose(ald,2e-3*0.012/0.012)
    # the paper form mixes as 1/(2 sigma_T^2) per step; the official form as eps/sigma_min^2
    assert paper/variance[0]<1e-3 and 0.1<ald/variance[0]<0.25
    assert ald/paper>100
    with pytest.raises(ValueError):level_gamma(variance,0,levels,'nope',2e-3)

class GaussToy:
    """x0 ~ N(0,1); VP marginal is N(0,1) at every level, score(x) = -x."""
    D=1
    ab=np.array([.99,.95,.85,.7,.4,.1])
    betas=np.array([.01,.05,.1,.15,.3,.3])
    sigma=np.sqrt((1-ab)/ab)
    def initial(self,rng,particles=1,ve=False):return rng.normal(size=(particles,1))*(self.sigma[-1] if ve else 1.)
    def score(self,x,t,ve=False):return -np.atleast_2d(x)
    def point(self):return self
    def constraints(self,x):return np.array([1.-np.asarray(x).reshape(-1)[0]]),np.array([[-1.]])
    def project(self,target,budget,allow_partial=False):
        return np.maximum(np.asarray(target,dtype=float),1.),{'status':'clamp_half_space','residual':float(max(0.,1.-np.asarray(target).reshape(-1)[0]))}

def truncated_normal_moments(a=1.):
    from scipy.stats import norm
    alpha=a;lam=norm.pdf(alpha)/(1.-norm.cdf(alpha))
    return float(lam),float(np.sqrt(1.+alpha*lam-lam**2))

def test_pdm_loop_matches_analytic_constrained_gaussian():
    """Annealing + projection on a known Gaussian score must recover N(0,1)|x>=1."""
    cfg={'pdm_inner':100,'pdm_levels':6,'pdm_parameterization':'vp','pdm_step_convention':'official_ald',
        'pdm_ald_eps':0.005,'projection_tol_scene':1e-9}
    samples=[]
    for seed in range(300):
        trace=[];x=pdm_run(GaussToy(),'PDM',np.random.default_rng(seed),cfg,Budget(20),trace)
        samples.append(float(np.asarray(x).reshape(-1)[0]))
    s=np.asarray(samples);mean,std=truncated_normal_moments(1.)
    assert s.min()>=1.-1e-9
    assert abs(s.mean()-mean)<0.12,(s.mean(),mean)
    assert abs(s.std()-std)<0.12,(s.std(),std)

def test_pdm_paper_convention_is_unmixable_on_a_large_sigma_expansion():
    class Wide(GaussToy):
        sigma=np.array([0.11,0.2,0.3,0.4,0.5,10.])
    cfg={'pdm_inner':5,'pdm_levels':6,'pdm_parameterization':'ve','pdm_step_convention':'paper','projection_tol_scene':1e-9}
    trace=[];pdm_run(Wide(),'PDM',np.random.default_rng(0),cfg,Budget(20),trace)
    official=[];c2=dict(cfg,pdm_step_convention='official_ald',pdm_ald_eps=2e-3)
    pdm_run(Wide(),'PDM',np.random.default_rng(0),c2,Budget(20),official)
    # gamma_t is proportional to sigma_t^2, so the *lowest* level gets the least
    # mixing exactly where the sample still has to be denoised.
    assert trace[-1]['mixing_ratio']<1e-2
    assert official[-1]['mixing_ratio']>0.1
    assert official[-1]['mixing_ratio']>10*trace[-1]['mixing_ratio']
