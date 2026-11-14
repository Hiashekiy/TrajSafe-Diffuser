import numpy as np
import torch
from experiments.safety_baselines.adapters.legacy_constraint_field import LegacyConstraintField
from src.diffusion.sdf_guidance import _constraint_rows
from src.geometry.bspline import BSplineCodec

def test_reused_scalar_constraint_matches_old_baseline():
    occ=np.zeros((32,32),bool);occ[10:22,10:22]=True
    f=LegacyConstraintField(occ,device='cpu')
    codec=BSplineCodec(num_controls=48,knots_path='auto').to(dtype=torch.float64)
    q=torch.tensor(np.random.default_rng(7).normal(size=(1,48,2))*.4,dtype=torch.float64)
    p,_,_,g=_constraint_rows(codec,q,f.field,0.)
    h,J=f.query(p[0].numpy())
    np.testing.assert_allclose(-h,g[0].numpy(),atol=1e-14)

def test_actual_sdf_derivative_and_control_chain_rule():
    occ=np.zeros((32,32),bool);occ[10:22,10:22]=True
    f=LegacyConstraintField(occ,device='cpu')
    codec=BSplineCodec(num_controls=48,knots_path='auto')
    B=codec.basis.numpy().astype(float);q=np.random.default_rng(4).normal(size=(48,2))*.3
    p=B@q;h,grad=f.query(p);direction=np.random.default_rng(8).normal(size=(48,2));eps=1e-7
    fd=(f.query(B@(q+eps*direction))[0]-f.query(B@(q-eps*direction))[0])/(2*eps)
    np.testing.assert_allclose(fd,np.sum(grad*(B@direction),axis=1),atol=2e-7)
